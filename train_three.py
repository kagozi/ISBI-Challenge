"""
WBCBench 2026: Complete Improved Training Pipeline
==================================================
Includes: Denoising, Focal Loss with Label Smoothing, Mixup, TTA, 
          Stochastic Depth, Class Weights, Visualizations, Ensemble
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import timm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. DATA LOADING & PREPARATION
# ============================================================================

DATA_PATH = '../data'
PHASE1_IMG_DIR = os.path.join(DATA_PATH, "phase1")
PHASE2_TRAIN_IMG_DIR = os.path.join(DATA_PATH, "phase2/train")
PHASE2_EVAL_IMG_DIR = os.path.join(DATA_PATH, "phase2/eval")
PHASE2_TEST_IMG_DIR = os.path.join(DATA_PATH, "phase2/test")

PHASE1_CSV = os.path.join(DATA_PATH, "phase1_label.csv")
PHASE2_TRAIN_CSV = os.path.join(DATA_PATH, "phase2_train.csv")
PHASE2_EVAL_CSV = os.path.join(DATA_PATH, "phase2_eval.csv")
PHASE2_TEST_CSV = os.path.join(DATA_PATH, "phase2_test.csv")

phase1_df = pd.read_csv(PHASE1_CSV)
phase2_train_df = pd.read_csv(PHASE2_TRAIN_CSV)
phase2_eval_df = pd.read_csv(PHASE2_EVAL_CSV)
phase2_test_df = pd.read_csv(PHASE2_TEST_CSV)

def clean_df(df):
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
    df = df.rename(columns={"ID": "filename", "labels": "label"})
    return df

phase1_df = clean_df(phase1_df)
phase2_train_df = clean_df(phase2_train_df)
phase2_eval_df = clean_df(phase2_eval_df)
phase2_test_df = clean_df(phase2_test_df)

phase1_df["img_dir"] = PHASE1_IMG_DIR
phase2_train_df["img_dir"] = PHASE2_TRAIN_IMG_DIR
phase2_eval_df["img_dir"] = PHASE2_EVAL_IMG_DIR
phase2_test_df["img_dir"] = PHASE2_TEST_IMG_DIR

print("\n" + "="*70)
print("EXPANDING TRAINING DATA")
print("="*70)
train_df_expanded = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
print(f"Phase 1:          {len(phase1_df):6,} images")
print(f"Phase 2 Train:    {len(phase2_train_df):6,} images")
print(f"Phase 2 Eval:     {len(phase2_eval_df):6,} images (ADDED!)")
print(f"{'─'*70}")
print(f"TOTAL TRAINING:   {len(train_df_expanded):6,} images")
print("="*70 + "\n")

test_df = phase2_test_df.copy()

class_names = sorted(train_df_expanded["label"].unique())
num_classes = len(class_names)
label2name = dict(zip(range(num_classes), class_names))
name2label = {v: k for k, v in label2name.items()}

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Classes: {class_names}")
print(f"Num classes: {num_classes}")
print(f"Device: {device}\n")

train_df_expanded["label_id"] = train_df_expanded["label"].map(name2label)
test_df["label_id"] = -1

train_indices, val_indices = train_test_split(
    range(len(train_df_expanded)),
    test_size=0.1,
    stratify=train_df_expanded['label_id'],
    random_state=42
)

train_df_split = train_df_expanded.iloc[train_indices].reset_index(drop=True)
val_df_split = train_df_expanded.iloc[val_indices].reset_index(drop=True)

print(f"Training split:   {len(train_df_split):6,} images (90%)")
print(f"Validation split: {len(val_df_split):6,} images (10%)")
print(f"Test set:         {len(test_df):6,} images\n")


# ============================================================================
# 2. IMPROVED FOCAL LOSS WITH LABEL SMOOTHING
# ============================================================================

class FocalLoss(nn.Module):
    """
    Unified Focal Loss with optional class weights and label smoothing.
    Based on BirdCLEF top solutions.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha  # Per-class weights tensor or None
        self.gamma = gamma
        self.label_smoothing = label_smoothing
    
    def forward(self, inputs, targets):
        num_classes = inputs.size(-1)
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_targets = torch.zeros_like(inputs).scatter_(
                1, targets.unsqueeze(1), 1.0
            )
            smooth_targets = smooth_targets * (1 - self.label_smoothing) + \
                           self.label_smoothing / num_classes
            
            log_probs = F.log_softmax(inputs, dim=1)
            ce_loss = -(smooth_targets * log_probs).sum(dim=1)
            
            # Apply class weights if provided
            if self.alpha is not None:
                ce_loss = ce_loss * self.alpha[targets]
        else:
            ce_loss = F.cross_entropy(
                inputs, targets, reduction='none', weight=self.alpha
            )
        
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def compute_custom_class_weights(num_classes, device):
    """Custom class weights based on domain knowledge."""
    weights = torch.ones(num_classes, dtype=torch.float32) * 0.25
    
    custom_weights = {
        'BNE': 0.75,
        'MMY': 0.6,
        'PC': 0.6,
        'PMY': 0.75,
        'VLY': 0.75,
    }
    
    for class_name, weight in custom_weights.items():
        class_id = name2label[class_name]
        weights[class_id] = weight
    
    weights = weights.to(device)
    
    print("="*70)
    print("CUSTOM CLASS WEIGHTS")
    print("="*70)
    for i in range(num_classes):
        print(f"{label2name[i]:20s} | Weight: {weights[i]:.4f}")
    print("="*70 + "\n")
    
    return weights

class_weights = compute_custom_class_weights(num_classes, device)


# ============================================================================
# 3. AUGMENTATION WITH DENOISING
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
        A.Resize(224, 224),
        A.Lambda(image=lambda x, **k: cv2.bilateralFilter(x, 7, 50, 50), p=0.5),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=0.7),
        A.Lambda(image=lambda x, **k: morphology_on_lab_l(x), p=0.15),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.3),
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
        A.Resize(224, 224),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


# ============================================================================
# 4. DATASET
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

train_dataset = BloodDataset(train_df_split, transform=get_train_transform())
val_dataset = BloodDataset(val_df_split, transform=get_val_transform())
test_dataset = BloodDataset(test_df, transform=get_val_transform(), is_test=True)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, 
                          num_workers=4, pin_memory=True, drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                        num_workers=4, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                         num_workers=4, pin_memory=True)

print(f"Dataset sizes:")
print(f"  Train: {len(train_dataset):,}")
print(f"  Val:   {len(val_dataset):,}")
print(f"  Test:  {len(test_dataset):,}\n")


# ============================================================================
# 5. MODELS WITH STOCHASTIC DEPTH
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
    def __init__(self, num_classes, dropout=0.4, pretrained=True, drop_path_rate=0.15):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            drop_path_rate=drop_path_rate  # ← Stochastic Depth
        )
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)
        print(f"  SwinTransformer: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
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
    def __init__(self, num_classes, dropout=0.4, pretrained=True, drop_path_rate=0.15):
        super().__init__()
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
        self.swin = timm.create_model(
            'swin_base_patch4_window7_224',
            pretrained=True,
            num_classes=512,
            drop_path_rate=drop_path_rate  # ← Stochastic Depth
        )
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
    def __init__(self, num_classes, dropout=0.4, pretrained=True, drop_path_rate=0.15):
        super().__init__()
        self.backbone = timm.create_model(
            "tf_efficientnetv2_m",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3,
            drop_path_rate=drop_path_rate  # ← Stochastic Depth
        )
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)
        print(f"  EfficientNet: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")

    def forward(self, x):
        return self.classifier(self.backbone(x))


def get_model(model_name, num_classes, device, drop_path_rate=0.15):
    print(f"\n{'='*60}\nInitializing {model_name}\n{'='*60}")
    if model_name == 'SwinTransformer':
        model = SwinTransformer(num_classes=num_classes, drop_path_rate=drop_path_rate)
    elif model_name == 'HybridSwin':
        model = HybridSwin(num_classes=num_classes, drop_path_rate=drop_path_rate)
    elif model_name == 'EfficientNet':
        model = EfficientNet(num_classes=num_classes, drop_path_rate=drop_path_rate)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


# ============================================================================
# 6. MIXUP
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# 7. TRAINING
# ============================================================================

def train_epoch_with_mixup(model, loader, criterion, optimizer, device, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]")
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
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1


def validate(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc=f"Epoch {epoch} [VAL]")
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    epoch_f1 = f1_score(all_labels, all_preds, average='macro')
    return epoch_loss, epoch_acc, epoch_f1, all_preds, all_labels


# ============================================================================
# 8. TTA
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


def generate_submission_with_tta(model, test_loader, device, label2name, output_path, use_tta=True):
    model.eval()
    predictions, filenames = [], []
    
    for images, ids in tqdm(test_loader, desc="Generating predictions (TTA)" if use_tta else "Generating predictions"):
        images = images.to(device)
        if use_tta:
            probs = predict_proba_with_tta(model, images, device, n_tta=5)
            preds = probs.argmax(dim=1)
        else:
            with torch.no_grad():
                preds = model(images).argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
        filenames.extend(ids)
    
    pred_labels = [label2name[pred] for pred in predictions]
    submission_df = pd.DataFrame({'ID': filenames, 'Target': pred_labels})
    
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved: {output_path}")
    print(submission_df.head(10))
    return submission_df


# ============================================================================
# 9. VISUALIZATION
# ============================================================================

def plot_training_curves(history, model_name, loss_function, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, (train_key, val_key, ylabel) in zip(axes, [
        ('train_loss', 'val_loss', 'Loss'),
        ('train_acc', 'val_acc', 'Accuracy'),
        ('train_f1', 'val_f1', 'F1 Score')
    ]):
        ax.plot(epochs, history[train_key], 'b-', label='Train', linewidth=2)
        ax.plot(epochs, history[val_key], 'r-', label='Val', linewidth=2)
        ax.set_title(f'{model_name} - {ylabel}', fontsize=14, fontweight='bold')
        ax.set_xlabel('Epoch')
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True, alpha=0.3)
        best_epoch = np.argmax(history['val_f1']) + 1
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_{loss_function}_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, loss_function, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{model_name} - Counts', fontsize=14, fontweight='bold')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, vmin=0, vmax=100)
    ax2.set_title(f'{model_name} - Percentages', fontsize=14, fontweight='bold')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_{loss_function}_cm.png'), dpi=300, bbox_inches='tight')
    plt.close()


# ============================================================================
# 10. MAIN TRAINING
# ============================================================================

def train_model(config, train_loader, val_loader, test_loader, num_classes, 
                label2name, device, class_weights, save_dir='models'):
    model_name = config['model']
    loss_function = config['loss']
    
    print(f"\n{'#'*70}\n# Training: {model_name} with {loss_function}\n{'#'*70}\n")
    
    model = get_model(model_name, num_classes, device, drop_path_rate=config.get('drop_path_rate', 0.15))
    
    # Create loss function with label smoothing
    if loss_function == 'focal_weighted':
        criterion = FocalLoss(
            alpha=class_weights,
            gamma=2.0,
            label_smoothing=config.get('label_smoothing', 0.1)
        )
    elif loss_function == 'focal':
        criterion = FocalLoss(
            alpha=None,
            gamma=2.0,
            label_smoothing=config.get('label_smoothing', 0.1)
        )
    else:  # ce
        criterion = nn.CrossEntropyLoss(label_smoothing=config.get('label_smoothing', 0.1))
    
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_{loss_function}_best.pth")
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc, train_f1 = train_epoch_with_mixup(
            model, train_loader, criterion, optimizer, device, epoch, use_mixup=True)
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)