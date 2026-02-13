"""
WBCBench 2026: K-Fold Training Pipeline with OOF Predictions
=============================================================
Trains all model/loss combinations using Stratified K-Fold.
Generates out-of-fold (OOF) predictions for meta-stacking.
Generates averaged test predictions per model.

Usage:
    python main_kfold.py
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import timm
import warnings
import json
from datetime import datetime

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    DATA_PATH = '../data'
    N_FOLDS = 5
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directories
    SAVE_DIR = 'models_kfold'
    OOF_DIR = 'oof_predictions'
    TEST_PRED_DIR = 'test_predictions'
    PLOT_DIR = 'plots_kfold'
    SUBMISSION_DIR = 'submissions_kfold'

    # Model configs to train
    CONFIGS = [
        # {'model': 'SwinTransformer', 'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'HybridSwin',      'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'EfficientNet',     'loss': 'ce',             'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'SwinTransformer', 'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'HybridSwin',      'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'EfficientNet',     'loss': 'focal',          'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'SwinTransformer', 'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'HybridSwin',      'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'EfficientNet',     'loss': 'focal_weighted', 'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
    ]


cfg = Config()

# Create directories
for d in [cfg.SAVE_DIR, cfg.OOF_DIR, cfg.TEST_PRED_DIR, cfg.PLOT_DIR, cfg.SUBMISSION_DIR]:
    os.makedirs(d, exist_ok=True)


# ============================================================================
# DATA LOADING
# ============================================================================

def load_data(data_path):
    PHASE1_IMG_DIR = os.path.join(data_path, "phase1")
    PHASE2_TRAIN_IMG_DIR = os.path.join(data_path, "phase2/train")
    PHASE2_EVAL_IMG_DIR = os.path.join(data_path, "phase2/eval")
    PHASE2_TEST_IMG_DIR = os.path.join(data_path, "phase2/test")

    def clean_df(df):
        df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
        df = df.rename(columns={"ID": "filename", "labels": "label"})
        return df

    phase1_df = clean_df(pd.read_csv(os.path.join(data_path, "phase1_label.csv")))
    phase2_train_df = clean_df(pd.read_csv(os.path.join(data_path, "phase2_train.csv")))
    phase2_eval_df = clean_df(pd.read_csv(os.path.join(data_path, "phase2_eval.csv")))
    phase2_test_df = clean_df(pd.read_csv(os.path.join(data_path, "phase2_test.csv")))

    phase1_df["img_dir"] = PHASE1_IMG_DIR
    phase2_train_df["img_dir"] = PHASE2_TRAIN_IMG_DIR
    phase2_eval_df["img_dir"] = PHASE2_EVAL_IMG_DIR
    phase2_test_df["img_dir"] = PHASE2_TEST_IMG_DIR

    # Combine all labeled data
    train_df = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
    test_df = phase2_test_df.copy()

    # Class mapping
    class_names = sorted(train_df["label"].unique())
    num_classes = len(class_names)
    label2name = dict(zip(range(num_classes), class_names))
    name2label = {v: k for k, v in label2name.items()}

    train_df["label_id"] = train_df["label"].map(name2label)
    test_df["label_id"] = -1

    print(f"\n{'='*70}")
    print(f"DATA SUMMARY")
    print(f"{'='*70}")
    print(f"Total training samples: {len(train_df):,}")
    print(f"Test samples:           {len(test_df):,}")
    print(f"Classes ({num_classes}): {class_names}")
    print(f"{'='*70}\n")

    return train_df, test_df, class_names, num_classes, label2name, name2label


# ============================================================================
# TRANSFORMS
# ============================================================================

def advanced_clahe_preprocessing(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)


def morphology_on_lab_l(image):
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    l2 = cv2.morphologyEx(l, cv2.MORPH_CLOSE, kernel, iterations=1)
    lab2 = cv2.merge([l2, a, b])
    return cv2.cvtColor(lab2, cv2.COLOR_LAB2RGB)


def get_train_transform():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Lambda(image=lambda x, **k: cv2.bilateralFilter(x, 7, 50, 50), p=0.5),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=0.7),
        A.Lambda(image=lambda x, **k: morphology_on_lab_l(x), p=0.15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.RandomResizedCrop(size=(cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.3),
        A.OneOf([
            A.ElasticTransform(alpha=10, sigma=4, alpha_affine=3, border_mode=cv2.BORDER_REFLECT, p=1.0),
            A.GridDistortion(num_steps=5, distort_limit=0.07, border_mode=cv2.BORDER_REFLECT, p=1.0),
            A.OpticalDistortion(distort_limit=0.07, shift_limit=0.07, border_mode=cv2.BORDER_REFLECT, p=1.0),
        ], p=0.15),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.08, p=0.4),
        A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.25),
        A.OneOf([
            A.GaussNoise(var_limit=(3.0, 12.0), p=1.0),
            A.GaussianBlur(blur_limit=(3, 3), p=1.0),
            A.MedianBlur(blur_limit=3, p=1.0),
        ], p=0.25),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================================
# DATASET
# ============================================================================

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
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FocalLossWithClassWeights(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


def compute_custom_class_weights(num_classes, name2label, device):
    weights = torch.ones(num_classes, dtype=torch.float32) * 0.25
    custom = {'BNE': 0.75, 'MMY': 0.6, 'PC': 0.6, 'PMY': 0.75, 'VLY': 0.75}
    for cls_name, w in custom.items():
        if cls_name in name2label:
            weights[name2label[cls_name]] = w
    return weights.to(device)


def get_loss_fn(loss_name, class_weights):
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'focal_weighted':
        return FocalLossWithClassWeights(alpha=class_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")


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
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224", pretrained=pretrained, num_classes=0, in_chans=3)
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
        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=pretrained, num_classes=512)
        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, num_classes))

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.swin(x)
        return self.fc(x)


class EfficientNet(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m", pretrained=pretrained, num_classes=0, in_chans=3)
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.backbone(x))


MODEL_REGISTRY = {
    'SwinTransformer': SwinTransformer,
    'HybridSwin': HybridSwin,
    'EfficientNet': EfficientNet,
}


# ============================================================================
# MIXUP
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


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
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []

    pbar = tqdm(loader, desc=f"  Epoch {epoch} [TRAIN]", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        if use_mixup and np.random.rand() > 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
            optimizer.zero_grad()
            outputs = model(images)
            loss = mixup_criterion(criterion, outputs, labels_a, labels_b, lam)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels_a.cpu().numpy())
        else:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return running_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')


def validate_one_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  Epoch {epoch} [VAL]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (running_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro'),
            all_preds, all_labels)


def extract_oof_probabilities(model, loader, device, n_tta=5):
    """Extract probability predictions for OOF samples using TTA."""
    model.eval()
    all_probs = []

    for images, _ in tqdm(loader, desc="  Extracting OOF probs", leave=False):
        images = images.to(device)
        probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
        all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def extract_test_probabilities(model, loader, device, n_tta=5):
    """Extract probability predictions for test samples using TTA."""
    model.eval()
    all_probs = []
    all_filenames = []

    for images, filenames in tqdm(loader, desc="  Extracting test probs", leave=False):
        images = images.to(device)
        probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
        all_probs.append(probs.cpu().numpy())
        all_filenames.extend(filenames)

    return np.vstack(all_probs), all_filenames


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

        # Datasets & loaders
        fold_train_ds = BloodDataset(fold_train_df, transform=get_train_transform())
        fold_val_ds = BloodDataset(fold_val_df, transform=get_val_transform())
        test_ds = BloodDataset(test_df, transform=get_val_transform(), is_test=True)

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
# VISUALIZATION
# ============================================================================

def plot_oof_confusion_matrix(oof_labels, oof_preds, class_names, config_key, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(oof_labels, oof_preds)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{config_key} OOF — Counts')
    ax1.set_ylabel('True'); ax1.set_xlabel('Predicted')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, vmin=0, vmax=100)
    ax2.set_title(f'{config_key} OOF — %')
    ax2.set_ylabel('True'); ax2.set_xlabel('Predicted')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_key}_oof_cm.png'), dpi=200, bbox_inches='tight')
    plt.close()


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*70}")
    print(f"WBCBench 2026 — K-FOLD TRAINING PIPELINE")
    print(f"Folds: {cfg.N_FOLDS} | Seed: {cfg.SEED} | Device: {cfg.DEVICE}")
    print(f"{'='*70}\n")

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

    for config in cfg.CONFIGS:
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
    main()