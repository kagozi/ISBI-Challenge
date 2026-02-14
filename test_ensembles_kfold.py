"""
Standalone Ensemble Strategy Tester (K-Fold Version)
=====================================================
Run this script after main_kfold.py to test all ensemble combinations
using out-of-fold predictions (no leakage).

For test submissions, averages predictions across all K folds per model.

Usage:
    python test_ensembles_kfold.py
"""

import os
import glob
import json
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from itertools import combinations

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class ClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.head(x)


class SwinTransformer(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("swin_base_patch4_window7_224",
                                         pretrained=False, num_classes=0, in_chans=3)
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.backbone(x))


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        scale = self.global_avg_pool(x).view(batch, channels)
        scale = self.fc(scale).view(batch, channels, 1, 1)
        return x * scale


class HybridSwin(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128), nn.ReLU(),
            SEBlock(128),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3), nn.ReLU()
        )
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=512)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.swin(x)
        return self.fc(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m", pretrained=False, num_classes=0, in_chans=3)
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.backbone(x))


MODEL_REGISTRY = {
    'SwinTransformer': SwinTransformer,
    'HybridSwin': HybridSwin,
    'EfficientNet': EfficientNet,
}


# ============================================================================
# DATA PREPARATION
# ============================================================================

def advanced_clahe_preprocessing(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def get_val_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


class BloodDataset(Dataset):
    def __init__(self, df, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row["img_dir"], row["filename"])
        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform:
            image = self.transform(image=image)["image"]

        if self.is_test:
            return image, row["filename"]

        label = torch.tensor(row["label_id"], dtype=torch.long)
        return image, label


# ============================================================================
# TTA
# ============================================================================

def predict_proba_with_tta(model, images, device, n_tta=5):
    model.eval()
    preds = []
    with torch.no_grad():
        preds.append(torch.softmax(model(images), dim=1))
        if n_tta > 1:
            preds.append(torch.softmax(model(torch.flip(images, dims=[3])), dim=1))
        if n_tta > 2:
            preds.append(torch.softmax(model(torch.flip(images, dims=[2])), dim=1))
        if n_tta > 3:
            preds.append(torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1))
        if n_tta > 4:
            preds.append(torch.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1))
    return torch.stack(preds, dim=0).mean(dim=0)


# ============================================================================
# OOF-BASED ENSEMBLE EVALUATION (NO LEAKAGE)
# ============================================================================

def evaluate_ensemble_oof(oof_probs_dict, labels, num_classes, weights=None):
    """
    Evaluate an ensemble using precomputed OOF probabilities.
    No model inference needed — pure numpy, instant.

    Args:
        oof_probs_dict: {config_key: np.array of shape (N, num_classes)}
        labels: np.array of true labels (N,)
        weights: optional {config_key: weight} dict
    Returns:
        f1, acc, preds
    """
    model_names = list(oof_probs_dict.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    s = sum(weights[n] for n in model_names)
    weights = {n: weights[n] / s for n in model_names}

    ensemble_probs = np.zeros_like(list(oof_probs_dict.values())[0])
    for name in model_names:
        ensemble_probs += weights[name] * oof_probs_dict[name]

    preds = ensemble_probs.argmax(axis=1)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return f1, acc, preds


def generate_submission_from_test_probs(test_probs_dict, test_filenames, label2name,
                                        output_path, weights=None):
    """
    Generate submission using precomputed averaged test probabilities.
    No model inference needed.
    """
    model_names = list(test_probs_dict.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    s = sum(weights[n] for n in model_names)
    weights = {n: weights[n] / s for n in model_names}

    ensemble_probs = np.zeros_like(list(test_probs_dict.values())[0])
    for name in model_names:
        ensemble_probs += weights[name] * test_probs_dict[name]

    final_preds = ensemble_probs.argmax(axis=1)
    pred_labels = [label2name[int(p)] for p in final_preds]

    submission_df = pd.DataFrame({"ID": test_filenames, "Target": pred_labels})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    return submission_df


# ============================================================================
# LIVE INFERENCE ENSEMBLE (for TTA comparison on test set)
# ============================================================================

def generate_submission_live(models_by_fold, test_loader, device, label2name,
                             output_path, weights=None, use_tta=True, n_tta=5):
    """
    Generate test submission by running inference with all fold checkpoints.
    For each config, averages predictions across all folds.
    Then ensembles across configs with given weights.
    """
    config_names = list(models_by_fold.keys())

    if weights is None:
        weights = {name: 1.0 / len(config_names) for name in config_names}
    s = sum(weights[n] for n in config_names)
    weights = {n: weights[n] / s for n in config_names}

    all_ids = []
    all_probs = None

    for images, ids in tqdm(test_loader, desc="Live inference", leave=False):
        images = images.to(device)
        all_ids.extend(list(ids))

        batch_ensemble = None

        for config_name in config_names:
            fold_models = models_by_fold[config_name]
            n_folds = len(fold_models)

            # Average across folds for this config
            fold_avg = None
            for model in fold_models:
                if use_tta:
                    probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
                else:
                    model.eval()
                    with torch.no_grad():
                        probs = torch.softmax(model(images), dim=1)

                fold_avg = probs if fold_avg is None else fold_avg + probs

            fold_avg = fold_avg / n_folds

            w = weights[config_name]
            batch_ensemble = w * fold_avg if batch_ensemble is None else batch_ensemble + w * fold_avg

        batch_ensemble = batch_ensemble.detach().cpu()
        all_probs = batch_ensemble if all_probs is None else torch.cat([all_probs, batch_ensemble], dim=0)

    final_preds = all_probs.argmax(dim=1).numpy()
    pred_labels = [label2name[int(p)] for p in final_preds]

    submission_df = pd.DataFrame({"ID": all_ids, "Target": pred_labels})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    return submission_df


# ============================================================================
# MAIN
# ============================================================================

def main():
    # Configuration
    DATA_PATH = '../data'
    MODEL_DIR = 'models_kfold'
    OOF_DIR = 'oof_predictions'
    TEST_PRED_DIR = 'test_predictions'
    OUTPUT_DIR = 'ensemble_results_kfold'
    N_FOLDS = 5

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # STEP 1: Load class info
    # ========================================================================
    print("="*80)
    print("STEP 1: Loading Data")
    print("="*80)

    PHASE1_IMG_DIR = os.path.join(DATA_PATH, "phase1")
    PHASE2_TRAIN_IMG_DIR = os.path.join(DATA_PATH, "phase2/train")
    PHASE2_EVAL_IMG_DIR = os.path.join(DATA_PATH, "phase2/eval")
    PHASE2_TEST_IMG_DIR = os.path.join(DATA_PATH, "phase2/test")

    phase1_df = pd.read_csv(os.path.join(DATA_PATH, "phase1_label.csv"))
    phase2_train_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_train.csv"))
    phase2_eval_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_eval.csv"))
    phase2_test_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_test.csv"))

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
    ]

    for config_key in model_configs:
        oof_path = os.path.join(OOF_DIR, f'oof_{config_key}.csv')
        test_path = os.path.join(TEST_PRED_DIR, f'test_{config_key}.csv')

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
        submission_path = os.path.join(OUTPUT_DIR, submission_file)
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
        for fold in range(1, N_FOLDS + 1):
            path = os.path.join(MODEL_DIR, f"{config_key}_fold{fold}.pth")
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
        tta_path = os.path.join(OUTPUT_DIR, f"submission_{best_strategy['strategy']}_TTA.csv")
        generate_submission_live(models_by_fold, test_loader, device, label2name,
                                 tta_path, weights=weights, use_tta=True, n_tta=5)
        print(f"  ✓ TTA submission saved: {tta_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    results_df = pd.DataFrame(results).sort_values('oof_f1', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_summary_kfold.csv'), index=False)

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

    print(f"\n✅ All results saved to: {OUTPUT_DIR}/")
    print(f"💡 Submit: {OUTPUT_DIR}/{best['submission_file']}")
    print("="*80)


if __name__ == "__main__":
    main()