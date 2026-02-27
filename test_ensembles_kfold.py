"""
For test submissions, averages predictions across all K folds per model.
"""

import os
import pandas as pd
import torch
from torch.utils.data import DataLoader
from itertools import combinations
from models import MODEL_REGISTRY
from dataloader import get_val_transform, BloodDataset
from utils import evaluate_ensemble_oof, generate_submission_from_test_probs, generate_submission_live
# ============================================================================
# MAIN
# ============================================================================
from config import Config
cfg = Config()
def main():
    # Configuration

    device = cfg.DEVICE
    print(f"Using device: {device}\n")
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    # ========================================================================
    # STEP 1: Load class info
    # ========================================================================
    print("="*80)
    print("STEP 1: Loading Data")
    print("="*80)

    PHASE1_IMG_DIR = os.path.join(cfg.DATA_PATH, "phase1")
    PHASE2_TRAIN_IMG_DIR = os.path.join(cfg.DATA_PATH, "phase2/train")
    PHASE2_EVAL_IMG_DIR = os.path.join(cfg.DATA_PATH, "phase2/eval")
    PHASE2_TEST_IMG_DIR = os.path.join(cfg.DATA_PATH, "phase2/test")

    phase1_df = pd.read_csv(os.path.join(cfg.DATA_PATH, "phase1_label.csv"))
    phase2_train_df = pd.read_csv(os.path.join(cfg.DATA_PATH, "phase2_train.csv"))
    phase2_eval_df = pd.read_csv(os.path.join(cfg.DATA_PATH, "phase2_eval.csv"))
    phase2_test_df = pd.read_csv(os.path.join(cfg.DATA_PATH, "phase2_test.csv"))

    for df in [phase1_df, phase2_train_df, phase2_eval_df, phase2_test_df]:
        df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore", inplace=True)
        df.rename(columns={"ID": "filename", "labels": "label"}, inplace=True)

    phase1_df["img_dir"] = PHASE1_IMG_DIR
    phase2_train_df["img_dir"] = PHASE2_TRAIN_IMG_DIR
    phase2_eval_df["img_dir"] = PHASE2_EVAL_IMG_DIR
    phase2_test_df["img_dir"] = PHASE2_TEST_IMG_DIR

    train_df_expanded = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
    test_df = phase2_test_df.copy()

    class_names = sorted(train_df_expanded["label"].unique())
    num_classes = len(class_names)
    label2name = dict(zip(range(num_classes), class_names))
    name2label = {v: k for k, v in label2name.items()}

    train_df_expanded["label_id"] = train_df_expanded["label"].map(name2label)
    test_df["label_id"] = -1

    print(f"Classes ({num_classes}): {class_names}")
    print(f"Training samples: {len(train_df_expanded):,}")
    print(f"Test samples: {len(test_df):,}\n")

    # ========================================================================
    # STEP 2: Load precomputed OOF and test predictions
    # ========================================================================
    print("="*80)
    print("STEP 2: Loading Precomputed OOF & Test Predictions")
    print("="*80)

    oof_probs_dict = {}   # {config_key: np.array (N_train, num_classes)}
    test_probs_dict = {}  # {config_key: np.array (N_test, num_classes)}
    oof_labels = None
    test_filenames = None

    model_configs = [
        'SwinTransformer_ce', 'HybridSwin_ce', 'EfficientNet_ce',
        'SwinTransformer_focal', 'HybridSwin_focal', 'EfficientNet_focal',
        'SwinTransformer_focal_weighted', 'HybridSwin_focal_weighted', 'EfficientNet_focal_weighted',
        'HOptimus1_ce', 'HOptimus1_focal', 'HOptimus1_focal_weighted',
        'ViT_ce', 'ViT_focal', 'ViT_focal_weighted',
    ]
    new_model_configs = [
        'EfficientNet_ce',
        'EfficientNet_focal',
        # ... new models
    ]
    model_configs = new_model_configs 
    for config_key in model_configs:
        oof_path = os.path.join(cfg.OOF_DIR, f'oof_{config_key}.csv')
        test_path = os.path.join(cfg.TEST_PRED_DIR, f'test_{config_key}.csv')

        if not os.path.exists(oof_path):
            print(f"  ⚠️  Skipping {config_key} (OOF file not found)")
            continue
        if not os.path.exists(test_path):
            print(f"  ⚠️  Skipping {config_key} (test file not found)")
            continue

        # Load OOF
        oof_df = pd.read_csv(oof_path)
        prob_cols = [c for c in oof_df.columns if c != 'label']
        oof_probs_dict[config_key] = oof_df[prob_cols].values

        if oof_labels is None:
            oof_labels = oof_df['label'].values

        # Load test
        test_df_pred = pd.read_csv(test_path)
        test_prob_cols = [c for c in test_df_pred.columns if c != 'filename']
        test_probs_dict[config_key] = test_df_pred[test_prob_cols].values

        if test_filenames is None:
            test_filenames = test_df_pred['filename'].values

        print(f"  ✓ Loaded {config_key}")

    available_configs = list(oof_probs_dict.keys())

    if not available_configs:
        print("\n❌ No OOF/test prediction files found! Run main_kfold.py first.")
        return

    print(f"\n✓ Loaded {len(available_configs)} model configs\n")

    # ========================================================================
    # STEP 3: Evaluate individual models (OOF — no leakage)
    # ========================================================================
    print("="*80)
    print("STEP 3: Individual Model OOF Evaluation")
    print("="*80)

    model_scores = {}
    for config_key in available_configs:
        f1, acc, _ = evaluate_ensemble_oof(
            {config_key: oof_probs_dict[config_key]}, oof_labels, num_classes)
        model_scores[config_key] = f1
        print(f"  {config_key:40s} | OOF F1: {f1:.4f} | Acc: {acc:.4f}")

    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)

    # ========================================================================
    # STEP 4: Test ensemble strategies (instant — no inference)
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Testing Ensemble Strategies (OOF-based, instant)")
    print("="*80)

    strategies = []

    # --- Fixed-size strategies ---
    top3 = [name for name, _ in sorted_models[:3]]
    top5 = [name for name, _ in sorted_models[:5]] if len(sorted_models) >= 5 else None
    all_models = [name for name, _ in sorted_models]

    # Top-3
    strategies.append({'name': 'Top3_Uniform', 'models': top3, 'weighting': 'uniform'})
    strategies.append({'name': 'Top3_F1Weighted', 'models': top3, 'weighting': 'f1'})

    # Top-5
    if top5:
        strategies.append({'name': 'Top5_Uniform', 'models': top5, 'weighting': 'uniform'})
        strategies.append({'name': 'Top5_F1Weighted', 'models': top5, 'weighting': 'f1'})

    # All
    strategies.append({'name': 'All_Uniform', 'models': all_models, 'weighting': 'uniform'})
    strategies.append({'name': 'All_F1Weighted', 'models': all_models, 'weighting': 'f1'})

    # --- Architecture-grouped strategies ---
    arch_groups = {}
    for config_key in available_configs:
        # e.g. "SwinTransformer_ce" -> arch = "SwinTransformer"
        arch = config_key.rsplit('_', 1)[0]
        # Handle focal_weighted which has two underscores
        for arch_name in ['SwinTransformer', 'HybridSwin', 'EfficientNet']:
            if config_key.startswith(arch_name):
                arch = arch_name
                break
        arch_groups.setdefault(arch, []).append(config_key)

    for arch, members in arch_groups.items():
        if len(members) >= 2:
            strategies.append({
                'name': f'{arch}_AllLosses_Uniform',
                'models': members,
                'weighting': 'uniform'
            })
            strategies.append({
                'name': f'{arch}_AllLosses_F1Weighted',
                'models': members,
                'weighting': 'f1'
            })

    # --- Loss-grouped strategies ---
    loss_groups = {}
    for config_key in available_configs:
        for loss_name in ['focal_weighted', 'focal', 'ce']:
            if config_key.endswith(f'_{loss_name}'):
                loss_groups.setdefault(loss_name, []).append(config_key)
                break

    for loss_name, members in loss_groups.items():
        if len(members) >= 2:
            strategies.append({
                'name': f'Loss_{loss_name}_AllArchs_Uniform',
                'models': members,
                'weighting': 'uniform'
            })

    # --- Best pair / triple search (exhaustive for small N) ---
    if len(available_configs) <= 12:
        # Test all pairs
        best_pair_f1 = 0
        best_pair = None
        for combo in combinations(available_configs, 2):
            subset = {k: oof_probs_dict[k] for k in combo}
            f1, _, _ = evaluate_ensemble_oof(subset, oof_labels, num_classes)
            if f1 > best_pair_f1:
                best_pair_f1 = f1
                best_pair = list(combo)

        if best_pair:
            strategies.append({
                'name': 'BestPair_Uniform',
                'models': best_pair,
                'weighting': 'uniform'
            })
            strategies.append({
                'name': 'BestPair_F1Weighted',
                'models': best_pair,
                'weighting': 'f1'
            })

        # Test all triples
        best_triple_f1 = 0
        best_triple = None
        for combo in combinations(available_configs, 3):
            subset = {k: oof_probs_dict[k] for k in combo}
            f1, _, _ = evaluate_ensemble_oof(subset, oof_labels, num_classes)
            if f1 > best_triple_f1:
                best_triple_f1 = f1
                best_triple = list(combo)

        if best_triple:
            strategies.append({
                'name': 'BestTriple_Uniform',
                'models': best_triple,
                'weighting': 'uniform'
            })
            strategies.append({
                'name': 'BestTriple_F1Weighted',
                'models': best_triple,
                'weighting': 'f1'
            })

    # --- Run all strategies ---
    results = []

    for i, strategy in enumerate(strategies, 1):
        subset_oof = {k: oof_probs_dict[k] for k in strategy['models']}
        subset_test = {k: test_probs_dict[k] for k in strategy['models']}

        if strategy['weighting'] == 'f1':
            weights = {name: model_scores[name] for name in strategy['models']}
        else:
            weights = None

        # Evaluate on OOF
        f1, acc, preds = evaluate_ensemble_oof(subset_oof, oof_labels, num_classes, weights)

        # Generate submission from precomputed test probs
        submission_file = f"submission_{strategy['name']}.csv"
        submission_path = os.path.join(cfg.OUTPUT_DIR, submission_file)
        generate_submission_from_test_probs(
            subset_test, test_filenames, label2name, submission_path, weights)

        results.append({
            'strategy': strategy['name'],
            'n_models': len(strategy['models']),
            'weighting': strategy['weighting'],
            'oof_f1': f1,
            'oof_acc': acc,
            'models': ', '.join(strategy['models']),
            'submission_file': submission_file
        })

        if i <= 20 or i == len(strategies):  # Don't spam console for exhaustive search
            print(f"  [{i:2d}/{len(strategies)}] {strategy['name']:40s} | OOF F1: {f1:.4f} | Acc: {acc:.4f}")

    # ========================================================================
    # STEP 5 (Optional): Live TTA inference for best strategies
    # ========================================================================
    # Uncomment below to also generate TTA submissions by loading fold checkpoints.
    # This is slow but may squeeze out a bit more performance on the test set.

    print("\n" + "="*80)
    print("STEP 5: Live TTA Inference for Top Strategies")
    print("="*80)
    
    # Load fold checkpoints for top strategies
    results_df_temp = pd.DataFrame(results).sort_values('oof_f1', ascending=False)
    best_strategy = results_df_temp.iloc[0]
    best_model_names = best_strategy['models'].split(', ')
    
    # Prepare test loader
    test_dataset = BloodDataset(test_df, transform=get_val_transform(), is_test=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    models_by_fold = {}
    for config_key in best_model_names:
        arch_name = None
        for a in MODEL_REGISTRY:
            if config_key.startswith(a):
                arch_name = a
                break
        if arch_name is None:
            print(f"  ⚠️  Could not find architecture for {config_key}, skipping")
            continue
        fold_models = []
        for fold in range(1, cfg.N_FOLDS + 1):
            path = os.path.join(cfg.SAVE_DIR, f"{config_key}_fold{fold}.pth")
            if os.path.exists(path):
                model = MODEL_REGISTRY[arch_name](num_classes=num_classes).to(device)
                ckpt = torch.load(path, map_location=device)
                model.load_state_dict(ckpt['model_state_dict'])
                model.eval()
                fold_models.append(model)
        if fold_models:
            models_by_fold[config_key] = fold_models
    
    if models_by_fold:
        weights = {n: model_scores[n] for n in models_by_fold}
        tta_path = os.path.join(cfg.OUTPUT_DIR, f"submission_{best_strategy['strategy']}_TTA.csv")
        generate_submission_live(models_by_fold, test_loader, device, label2name,
                                 tta_path, weights=weights, use_tta=True, n_tta=5)
        print(f"  ✓ TTA submission saved: {tta_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    results_df = pd.DataFrame(results).sort_values('oof_f1', ascending=False)
    results_df.to_csv(os.path.join(cfg.OUTPUT_DIR, 'ensemble_summary_kfold.csv'), index=False)

    print("\n" + "="*80)
    print("ENSEMBLE RESULTS SUMMARY (K-Fold OOF — No Leakage)")
    print("="*80)
    print(f"{'Rank':<5} {'Strategy':<42} {'OOF F1':<10} {'OOF Acc':<10} {'#Models':<8}")
    print("─" * 80)

    for i, row in enumerate(results_df.itertuples(), 1):
        marker = " 🏆" if i == 1 else ""
        print(f"{i:<5} {row.strategy:<42} {row.oof_f1:<10.4f} {row.oof_acc:<10.4f} {row.n_models:<8}{marker}")
        if i >= 20:
            remaining = len(results_df) - 20
            if remaining > 0:
                print(f"  ... and {remaining} more (see ensemble_summary_kfold.csv)")
            break

    print("="*80)

    best = results_df.iloc[0]
    print(f"\n🏆 BEST ENSEMBLE:")
    print(f"   Strategy:  {best['strategy']}")
    print(f"   OOF F1:    {best['oof_f1']:.4f}")
    print(f"   OOF Acc:   {best['oof_acc']:.4f}")
    print(f"   Models:    {best['n_models']}")
    print(f"   Members:   {best['models']}")
    print(f"   File:      {best['submission_file']}")

    # Individual model ranking for reference
    print(f"\n{'─'*80}")
    print("Individual Model Ranking (OOF):")
    for i, (name, score) in enumerate(sorted_models, 1):
        print(f"  {i}. {name:40s} F1: {score:.4f}")

    print(f"\n✅ All results saved to: {cfg.OUTPUT_DIR}/")
    print(f"💡 Submit: {cfg.OUTPUT_DIR}/{best['submission_file']}")
    print("="*80)


if __name__ == "__main__":
    main()