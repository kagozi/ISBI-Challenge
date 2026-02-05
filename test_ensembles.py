"""
Standalone Ensemble Strategy Tester
====================================
Run this script after training to test all ensemble combinations
without retraining models.

Usage:
    python test_ensembles.py
"""

import os
import sys
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from sklearn.model_selection import train_test_split
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

# ============================================================================
# MODEL DEFINITIONS (Copy from your main.py)
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
        super(SEBlock, self).__init__()
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
        super(HybridSwin, self).__init__()

        self.conv_stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=False, num_classes=512)

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.swin(x)
        x = self.fc(x)
        return x


class EfficientNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=False,
            num_classes=0,
            in_chans=3
        )
        num_features = self.backbone.num_features
        self.classifier = ClassificationHead(num_features, num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)


# ============================================================================
# DATA PREPARATION
# ============================================================================

def advanced_clahe_preprocessing(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    rgb_clahe = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)
    return rgb_clahe


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
# ENSEMBLE FUNCTIONS
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


def evaluate_ensemble(models_dict, val_loader, device, weights=None, use_tta=False, n_tta=5):
    model_names = list(models_dict.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}
    
    s = sum(weights.values())
    weights = {n: weights[n] / s for n in model_names}

    all_labels = []
    all_probs = []

    for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        all_labels.extend(labels.numpy().tolist())

        batch_ensemble = None
        for name in model_names:
            model = models_dict[name]
            w = weights[name]

            if use_tta:
                probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
            else:
                model.eval()
                with torch.no_grad():
                    probs = torch.softmax(model(images), dim=1)

            batch_ensemble = w * probs if batch_ensemble is None else batch_ensemble + w * probs

        all_probs.append(batch_ensemble.detach().cpu())

    all_probs = torch.cat(all_probs, dim=0)
    preds = all_probs.argmax(dim=1).numpy()

    acc = accuracy_score(all_labels, preds)
    f1 = f1_score(all_labels, preds, average="macro")

    return f1, acc, preds, all_labels


def generate_submission(models_dict, test_loader, device, label2name, 
                       output_path, weights=None, use_tta=True, n_tta=5):
    model_names = list(models_dict.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}
    
    s = sum(weights.values())
    weights = {n: weights[n] / s for n in model_names}

    all_ids = []
    all_probs = None

    for images, ids in tqdm(test_loader, desc="Generating submission", leave=False):
        images = images.to(device)
        all_ids.extend(list(ids))

        batch_ensemble = None
        for name in model_names:
            model = models_dict[name]
            w = weights[name]

            if use_tta:
                probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
            else:
                model.eval()
                with torch.no_grad():
                    probs = torch.softmax(model(images), dim=1)

            batch_ensemble = w * probs if batch_ensemble is None else batch_ensemble + w * probs

        batch_ensemble = batch_ensemble.detach().cpu()

        if all_probs is None:
            all_probs = batch_ensemble
        else:
            all_probs = torch.cat([all_probs, batch_ensemble], dim=0)

    final_preds = all_probs.argmax(dim=1).numpy()
    pred_labels = [label2name[int(p)] for p in final_preds]

    submission_df = pd.DataFrame({"ID": all_ids, "Target": pred_labels})
    submission_df.to_csv(output_path, index=False)

    return submission_df


# ============================================================================
# MAIN ENSEMBLE TESTING
# ============================================================================

def main():
    # Configuration
    DATA_PATH = '../data'
    MODEL_DIR = 'models_run_2'
    OUTPUT_DIR = 'ensemble_results'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}\n")
    
    # Load data
    print("Loading data...")
    PHASE1_IMG_DIR = os.path.join(DATA_PATH, "phase1")
    PHASE2_TRAIN_IMG_DIR = os.path.join(DATA_PATH, "phase2/train")
    PHASE2_EVAL_IMG_DIR = os.path.join(DATA_PATH, "phase2/eval")
    PHASE2_TEST_IMG_DIR = os.path.join(DATA_PATH, "phase2/test")
    
    phase1_df = pd.read_csv(os.path.join(DATA_PATH, "phase1_label.csv"))
    phase2_train_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_train.csv"))
    phase2_eval_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_eval.csv"))
    phase2_test_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_test.csv"))
    
    # Clean dataframes
    for df in [phase1_df, phase2_train_df, phase2_eval_df, phase2_test_df]:
        df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore", inplace=True)
        df.rename(columns={"ID": "filename", "labels": "label"}, inplace=True)
    
    phase1_df["img_dir"] = PHASE1_IMG_DIR
    phase2_train_df["img_dir"] = PHASE2_TRAIN_IMG_DIR
    phase2_eval_df["img_dir"] = PHASE2_EVAL_IMG_DIR
    phase2_test_df["img_dir"] = PHASE2_TEST_IMG_DIR
    
    train_df_expanded = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
    test_df = phase2_test_df.copy()
    
    # Get class information
    class_names = sorted(train_df_expanded["label"].unique())
    num_classes = len(class_names)
    label2name = dict(zip(range(num_classes), class_names))
    name2label = {v: k for k, v in label2name.items()}
    
    train_df_expanded["label_id"] = train_df_expanded["label"].map(name2label)
    test_df["label_id"] = -1
    
    # Create validation split
    train_indices, val_indices = train_test_split(
        range(len(train_df_expanded)),
        test_size=0.1,
        stratify=train_df_expanded['label_id'],
        random_state=42
    )
    
    val_df_split = train_df_expanded.iloc[val_indices].reset_index(drop=True)
    
    # Create datasets
    val_dataset = BloodDataset(val_df_split, transform=get_val_transform())
    test_dataset = BloodDataset(test_df, transform=get_val_transform(), is_test=True)
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    print(f"Classes: {class_names}")
    print(f"Validation size: {len(val_dataset):,}")
    print(f"Test size: {len(test_dataset):,}\n")
    
    # Define models to load
    model_configs = [
        {'name': 'SwinTransformer_ce', 'class': SwinTransformer, 'loss': 'ce'},
        {'name': 'HybridSwin_ce', 'class': HybridSwin, 'loss': 'ce'},
        {'name': 'EfficientNet_ce', 'class': EfficientNet, 'loss': 'ce'},
        {'name': 'SwinTransformer_focal', 'class': SwinTransformer, 'loss': 'focal'},
        {'name': 'HybridSwin_focal', 'class': HybridSwin, 'loss': 'focal'},
        {'name': 'EfficientNet_focal', 'class': EfficientNet, 'loss': 'focal'},
        {'name': 'SwinTransformer_focal_weighted', 'class': SwinTransformer, 'loss': 'focal_weighted'},
        {'name': 'HybridSwin_focal_weighted', 'class': HybridSwin, 'loss': 'focal_weighted'},
        {'name': 'EfficientNet_focal_weighted', 'class': EfficientNet, 'loss': 'focal_weighted'},
    ]
    
    # Load models
    print("Loading models...")
    loaded_models = {}
    model_scores = {}
    
    for config in model_configs:
        model_path = os.path.join(MODEL_DIR, f"{config['name']}_best.pth")
        
        if not os.path.exists(model_path):
            print(f"  ⚠️ Skipping {config['name']} (file not found)")
            continue
        
        try:
            model = config['class'](num_classes=num_classes).to(device)
            checkpoint = torch.load(model_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            loaded_models[config['name']] = model
            print(f"  ✓ Loaded {config['name']}")
        except Exception as e:
            print(f"  ❌ Error loading {config['name']}: {str(e)}")
    
    if not loaded_models:
        print("\n❌ No models loaded! Please check MODEL_DIR path.")
        return
    
    print(f"\n✓ Successfully loaded {len(loaded_models)} models\n")
    
    # Quick validation to get scores
    print("Evaluating individual models...")
    for name, model in loaded_models.items():
        f1, acc, _, _ = evaluate_ensemble(
            {name: model}, val_loader, device, use_tta=False
        )
        model_scores[name] = f1
        print(f"  {name:35s} | Val F1: {f1:.4f}")
    
    # Sort models by performance
    sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
    
    print("\n" + "="*80)
    print("TESTING ENSEMBLE STRATEGIES")
    print("="*80)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Test different strategies
    strategies = []
    
    # Top-3
    top3 = [name for name, _ in sorted_models[:3]]
    strategies.extend([
        {'name': 'Top3_Uniform_TTA', 'models': top3, 'weighting': 'uniform', 'tta': True},
        {'name': 'Top3_Weighted_TTA', 'models': top3, 'weighting': 'weighted', 'tta': True},
        {'name': 'Top3_Uniform_NoTTA', 'models': top3, 'weighting': 'uniform', 'tta': False},
        {'name': 'Top3_Weighted_NoTTA', 'models': top3, 'weighting': 'weighted', 'tta': False},
    ])
    
    # Top-5
    if len(sorted_models) >= 5:
        top5 = [name for name, _ in sorted_models[:5]]
        strategies.extend([
            {'name': 'Top5_Uniform_TTA', 'models': top5, 'weighting': 'uniform', 'tta': True},
            {'name': 'Top5_Weighted_TTA', 'models': top5, 'weighting': 'weighted', 'tta': True},
            {'name': 'Top5_Uniform_NoTTA', 'models': top5, 'weighting': 'uniform', 'tta': False},
            {'name': 'Top5_Weighted_NoTTA', 'models': top5, 'weighting': 'weighted', 'tta': False},
        ])
    
    # All models
    all_models = [name for name, _ in sorted_models]
    strategies.extend([
        {'name': 'All_Uniform_TTA', 'models': all_models, 'weighting': 'uniform', 'tta': True},
        {'name': 'All_Weighted_TTA', 'models': all_models, 'weighting': 'weighted', 'tta': True},
        {'name': 'All_Uniform_NoTTA', 'models': all_models, 'weighting': 'uniform', 'tta': False},
        {'name': 'All_Weighted_NoTTA', 'models': all_models, 'weighting': 'weighted', 'tta': False},
    ])
    
    # Run strategies
    results = []
    
    for i, strategy in enumerate(strategies, 1):
        print(f"\n[{i}/{len(strategies)}] Testing: {strategy['name']}")
        
        ensemble_models = {name: loaded_models[name] for name in strategy['models']}
        
        weights = None
        if strategy['weighting'] == 'weighted':
            weights = {name: model_scores[name] for name in strategy['models']}
        
        # Evaluate
        f1, acc, preds, labels = evaluate_ensemble(
            ensemble_models, val_loader, device, 
            weights=weights, use_tta=strategy['tta'], n_tta=5
        )
        
        print(f"  Val F1: {f1:.4f}, Val Acc: {acc:.4f}")
        
        # Generate submission
        submission_file = f"submission_{strategy['name']}.csv"
        submission_path = os.path.join(OUTPUT_DIR, submission_file)
        
        generate_submission(
            ensemble_models, test_loader, device, label2name,
            submission_path, weights=weights, use_tta=strategy['tta'], n_tta=5
        )
        
        results.append({
            'strategy': strategy['name'],
            'n_models': len(strategy['models']),
            'weighting': strategy['weighting'],
            'tta': strategy['tta'],
            'val_f1': f1,
            'val_acc': acc,
            'models': ', '.join(strategy['models']),
            'submission_file': submission_file
        })
    
    # Save results
    results_df = pd.DataFrame(results).sort_values('val_f1', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'ensemble_summary.csv'), index=False)
    
    # Print summary
    print("\n" + "="*80)
    print("ENSEMBLE RESULTS SUMMARY")
    print("="*80)
    print(f"{'Rank':<6} {'Strategy':<25} {'Val F1':<10} {'Val Acc':<10} {'Models':<8}")
    print("-" * 80)
    
    for i, row in enumerate(results_df.itertuples(), 1):
        print(f"{i:<6} {row.strategy:<25} {row.val_f1:<10.4f} {row.val_acc:<10.4f} {row.n_models:<8}")
    
    print("="*80)
    
    best = results_df.iloc[0]
    print(f"\n🏆 BEST ENSEMBLE:")
    print(f"   Strategy:  {best['strategy']}")
    print(f"   Val F1:    {best['val_f1']:.4f}")
    print(f"   Val Acc:   {best['val_acc']:.4f}")
    print(f"   Models:    {best['n_models']}")
    print(f"   File:      {best['submission_file']}")
    
    print(f"\n✅ All results saved to: {OUTPUT_DIR}/")
    print("="*80)


if __name__ == "__main__":
    main()