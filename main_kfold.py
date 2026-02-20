"""
WBCBench 2026: K-Fold Training Pipeline with OOF Predictions
=============================================================
Trains all model/loss combinations using Stratified K-Fold.
Generates out-of-fold (OOF) predictions for meta-stacking.
Generates averaged test predictions per model.

Usage:
    python main_kfold.py
"""
import argparse
import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings
import json
from utils import setup_huggingface_auth, plot_oof_confusion_matrix, get_loss_fn, compute_custom_class_weights, train_one_epoch, validate_one_epoch, extract_oof_probabilities, extract_test_probabilities
from models import MODEL_REGISTRY, HOPTIMUS_MODELS
warnings.filterwarnings('ignore')
from config import Config
from dataloader import load_data, BloodDataset, get_train_transform, get_val_transform, get_train_transform_hoptimus, get_val_transform_hoptimus, val_mini_transform, train_mini_transform

cfg = Config()


# Call this before training H-Optimus models
if any(c['model'] == 'HOptimus1' for c in cfg.CONFIGS):
    if not setup_huggingface_auth():
        print("\n❌ Removing HOptimus1 from training configs (authentication required)")
        cfg.CONFIGS = [c for c in cfg.CONFIGS if c['model'] != 'HOptimus1']
        

# Create directories
for d in [cfg.SAVE_DIR, cfg.OOF_DIR, cfg.TEST_PRED_DIR, cfg.PLOT_DIR, cfg.SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# K-FOLD TRAINING FOR A SINGLE CONFIG
# ============================================================================

def train_kfold(config, train_df, test_df, num_classes, class_weights,
                label2name, device, skf_splits):
    """
    Train a single model config across all K folds.

    Returns:
        oof_probs: np.array of shape (N_train, num_classes) — out-of-fold probabilities
        test_probs_avg: np.array of shape (N_test, num_classes) — averaged test probabilities
        test_filenames: list of test filenames
        fold_metrics: list of per-fold best val F1 scores
    """
    model_name = config['model']
    loss_name = config['loss']
    config_key = f"{model_name}_{loss_name}"

    print(f"\n{'#'*70}")
    print(f"# K-FOLD TRAINING: {config_key}")
    print(f"{'#'*70}")

    # Select appropriate transforms for this model
    use_hoptimus_transforms = model_name in HOPTIMUS_MODELS
    if use_hoptimus_transforms:
        print(f"  ℹ️  Using H-Optimus-1 normalization constants")
        train_transform = get_train_transform_hoptimus()
        val_transform = get_val_transform_hoptimus()
    else:
        train_transform = get_train_transform() 
        val_transform = get_val_transform()
    n_train = len(train_df)
    n_test = len(test_df)

    # Preallocate OOF array (filled fold by fold)
    oof_probs = np.zeros((n_train, num_classes), dtype=np.float32)

    # Accumulate test predictions across folds
    test_probs_sum = None
    test_filenames = None

    fold_metrics = []

    for fold_idx, (train_indices, val_indices) in enumerate(skf_splits):
        fold_num = fold_idx + 1
        print(f"\n{'─'*60}")
        print(f"  Fold {fold_num}/{cfg.N_FOLDS} | {config_key}")
        print(f"  Train: {len(train_indices):,} | Val: {len(val_indices):,}")
        print(f"{'─'*60}")

        # Split data
        fold_train_df = train_df.iloc[train_indices].reset_index(drop=True)
        fold_val_df = train_df.iloc[val_indices].reset_index(drop=True)

        # Datasets & loaders (using model-specific transforms)
        fold_train_ds = BloodDataset(fold_train_df, transform=train_transform)
        fold_val_ds = BloodDataset(fold_val_df, transform=val_transform)
        test_ds = BloodDataset(test_df, transform=val_transform, is_test=True)

        fold_train_loader = DataLoader(fold_train_ds, batch_size=cfg.BATCH_SIZE, shuffle=True,
                                       num_workers=cfg.NUM_WORKERS, pin_memory=True, drop_last=True)
        fold_val_loader = DataLoader(fold_val_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                     num_workers=cfg.NUM_WORKERS, pin_memory=True)
        test_loader = DataLoader(test_ds, batch_size=cfg.BATCH_SIZE, shuffle=False,
                                 num_workers=cfg.NUM_WORKERS, pin_memory=True)

        # Model, loss, optimizer, scheduler
        ModelClass = MODEL_REGISTRY[model_name]
        model = ModelClass(num_classes=num_classes, pretrained=True).to(device)
        criterion = get_loss_fn(loss_name, class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)

        best_val_f1 = 0.0
        model_path = os.path.join(cfg.SAVE_DIR, f"{config_key}_fold{fold_num}.pth")

        # Training loop
        for epoch in range(1, config['epochs'] + 1):
            train_loss, train_acc, train_f1 = train_one_epoch(
                model, fold_train_loader, criterion, optimizer, device, epoch)
            val_loss, val_acc, val_f1, _, _ = validate_one_epoch(
                model, fold_val_loader, criterion, device, epoch)
            scheduler.step()

            if epoch % 5 == 0 or epoch == config['epochs']:
                print(f"    Ep {epoch:2d}/{config['epochs']} | "
                      f"Train F1: {train_f1:.4f} | Val F1: {val_f1:.4f} | Val Acc: {val_acc:.4f}")

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch,
                             'fold': fold_num, 'val_f1': val_f1}, model_path)

        print(f"  ✓ Fold {fold_num} best Val F1: {best_val_f1:.4f}")
        fold_metrics.append(best_val_f1)

        # Load best model for this fold
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        # Extract OOF predictions for validation fold
        oof_fold_probs = extract_oof_probabilities(model, fold_val_loader, device, n_tta=5)
        oof_probs[val_indices] = oof_fold_probs

        # Extract test predictions for this fold
        test_fold_probs, test_fnames = extract_test_probabilities(model, test_loader, device, n_tta=5)
        if test_probs_sum is None:
            test_probs_sum = test_fold_probs
            test_filenames = test_fnames
        else:
            test_probs_sum += test_fold_probs

        # Free memory
        del model, optimizer, scheduler, criterion
        torch.cuda.empty_cache()

    # Average test predictions across folds
    test_probs_avg = test_probs_sum / cfg.N_FOLDS

    # Summary
    mean_f1 = np.mean(fold_metrics)
    std_f1 = np.std(fold_metrics)
    print(f"\n  {'='*50}")
    print(f"  {config_key} SUMMARY")
    print(f"  Mean Val F1: {mean_f1:.4f} ± {std_f1:.4f}")
    print(f"  Per-fold F1: {[f'{f:.4f}' for f in fold_metrics]}")
    print(f"  {'='*50}")

    return oof_probs, test_probs_avg, test_filenames, fold_metrics

# ============================================================================
# MAIN
# ============================================================================

def main(config_idx: int | None = None):
    print(f"\n{'='*70}")
    print(f"WBCBench 2026 — K-FOLD TRAINING PIPELINE")
    print(f"Folds: {cfg.N_FOLDS} | Seed: {cfg.SEED} | Device: {cfg.DEVICE}")
    print(f"{'='*70}\n")
    if config_idx is not None:
        configs = [cfg.CONFIGS[config_idx]]
    else:
        configs = cfg.CONFIGS
    # Load data
    train_df, test_df, class_names, num_classes, label2name, name2label = load_data(cfg.DATA_PATH)
    class_weights = compute_custom_class_weights(num_classes, name2label, cfg.DEVICE)

    # Create SHARED stratified K-fold splits (same for all models!)
    skf = StratifiedKFold(n_splits=cfg.N_FOLDS, shuffle=True, random_state=cfg.SEED)
    skf_splits = list(skf.split(np.arange(len(train_df)), train_df['label_id'].values))

    # Save fold indices for reproducibility
    fold_indices = {f'fold_{i+1}': {'train': train_idx.tolist(), 'val': val_idx.tolist()}
                    for i, (train_idx, val_idx) in enumerate(skf_splits)}
    with open(os.path.join(cfg.OOF_DIR, 'fold_indices.json'), 'w') as f:
        json.dump(fold_indices, f)
    print(f"✓ Fold indices saved to {cfg.OOF_DIR}/fold_indices.json\n")

    # Train all configs
    all_results = {}

    for config in configs:
        config_key = f"{config['model']}_{config['loss']}"
        try:
            oof_probs, test_probs_avg, test_filenames, fold_metrics = train_kfold(
                config, train_df, test_df, num_classes, class_weights,
                label2name, cfg.DEVICE, skf_splits
            )

            # Save OOF predictions
            oof_df = pd.DataFrame(oof_probs, columns=[f'{config_key}_class{i}' for i in range(num_classes)])
            oof_df['label'] = train_df['label_id'].values
            oof_path = os.path.join(cfg.OOF_DIR, f'oof_{config_key}.csv')
            oof_df.to_csv(oof_path, index=False)

            # Save test predictions
            test_df_pred = pd.DataFrame(test_probs_avg, columns=[f'{config_key}_class{i}' for i in range(num_classes)])
            test_df_pred['filename'] = test_filenames
            test_path = os.path.join(cfg.TEST_PRED_DIR, f'test_{config_key}.csv')
            test_df_pred.to_csv(test_path, index=False)

            # OOF metrics & confusion matrix
            oof_preds = oof_probs.argmax(axis=1)
            oof_f1 = f1_score(train_df['label_id'].values, oof_preds, average='macro')
            oof_acc = accuracy_score(train_df['label_id'].values, oof_preds)

            print(f"\n  OOF Metrics for {config_key}:")
            print(f"    F1 Macro: {oof_f1:.4f} | Accuracy: {oof_acc:.4f}")
            print(classification_report(train_df['label_id'].values, oof_preds, target_names=class_names))

            plot_oof_confusion_matrix(train_df['label_id'].values, oof_preds, class_names,
                                       config_key, cfg.PLOT_DIR)

            # Generate single-model submission (simple average of folds)
            test_preds = test_probs_avg.argmax(axis=1)
            sub_df = pd.DataFrame({
                'ID': test_filenames,
                'Target': [label2name[p] for p in test_preds]
            })
            sub_path = os.path.join(cfg.SUBMISSION_DIR, f'submission_{config_key}.csv')
            sub_df.to_csv(sub_path, index=False)

            all_results[config_key] = {
                'oof_f1': oof_f1,
                'oof_acc': oof_acc,
                'fold_f1s': fold_metrics,
                'mean_fold_f1': np.mean(fold_metrics),
                'std_fold_f1': np.std(fold_metrics),
            }

        except Exception as e:
            print(f"\n❌ Error training {config_key}: {str(e)}")
            import traceback
            traceback.print_exc()

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print(f"\n\n{'='*80}")
    print("K-FOLD TRAINING SUMMARY")
    print(f"{'='*80}")
    print(f"{'Config':<35s} | {'OOF F1':>8s} | {'Mean Fold F1':>13s} | {'Std':>6s}")
    print(f"{'─'*80}")
    for key, res in sorted(all_results.items(), key=lambda x: -x[1]['oof_f1']):
        print(f"{key:<35s} | {res['oof_f1']:>8.4f} | {res['mean_fold_f1']:>13.4f} | {res['std_fold_f1']:>6.4f}")

    if all_results:
        best_key = max(all_results, key=lambda k: all_results[k]['oof_f1'])
        print(f"\n🏆 Best single model (OOF): {best_key} — F1: {all_results[best_key]['oof_f1']:.4f}")

    print(f"\n{'='*80}")
    print("FILES GENERATED:")
    print(f"  OOF predictions:   {cfg.OOF_DIR}/oof_<config>.csv")
    print(f"  Test predictions:  {cfg.TEST_PRED_DIR}/test_<config>.csv")
    print(f"  Fold indices:      {cfg.OOF_DIR}/fold_indices.json")
    print(f"  Submissions:       {cfg.SUBMISSION_DIR}/submission_<config>.csv")
    print(f"  Model checkpoints: {cfg.SAVE_DIR}/<config>_fold<N>.pth")
    print(f"  Plots:             {cfg.PLOT_DIR}/")
    print(f"\n💡 Next step: Run autogluon_kfold_ensemble.py to build the meta-learner!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config_idx", type=int, default=None)
    args = ap.parse_args()
    main(config_idx=args.config_idx)