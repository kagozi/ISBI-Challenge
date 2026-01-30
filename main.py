"""
WBCBench 2026: Complete Improved Training Pipeline
==================================================
Includes: Denoising, Focal Loss, Mixup, TTA, Class Weights, Visualizations
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
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

DATA_PATH = './data'
PHASE1_IMG_DIR = os.path.join(DATA_PATH, "phase1")
PHASE2_TRAIN_IMG_DIR = os.path.join(DATA_PATH, "phase2/train")
PHASE2_EVAL_IMG_DIR = os.path.join(DATA_PATH, "phase2/eval")
PHASE2_TEST_IMG_DIR = os.path.join(DATA_PATH, "phase2/test")

PHASE1_CSV = os.path.join(DATA_PATH, "phase1_label.csv")
PHASE2_TRAIN_CSV = os.path.join(DATA_PATH, "phase2_train.csv")
PHASE2_EVAL_CSV = os.path.join(DATA_PATH, "phase2_eval.csv")
PHASE2_TEST_CSV = os.path.join(DATA_PATH, "phase2_test.csv")

# Load data
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

# IMPROVED: Combine ALL training data (including phase2_eval)
print("\n" + "="*70)
print("EXPANDING TRAINING DATA")
print("="*70)
train_df_expanded = pd.concat([
    phase1_df,
    phase2_train_df,
    phase2_eval_df  # ← ADDED!
], ignore_index=True)

print(f"Phase 1:          {len(phase1_df):6,} images")
print(f"Phase 2 Train:    {len(phase2_train_df):6,} images")
print(f"Phase 2 Eval:     {len(phase2_eval_df):6,} images (ADDED!)")
print(f"{'─'*70}")
print(f"TOTAL TRAINING:   {len(train_df_expanded):6,} images")
print("="*70 + "\n")

test_df = phase2_test_df.copy()

# Extract class names
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

# Create validation split from expanded training data
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
# 2. FOCAL LOSS FOR CLASS IMBALANCE
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, reduction='none', weight=self.alpha
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


def compute_class_weights(train_df, num_classes, device):
    """Compute inverse frequency weights"""
    class_counts = train_df['label_id'].value_counts().sort_index()
    total = len(train_df)
    weights = torch.tensor([
        total / (num_classes * class_counts.get(i, 1))
        for i in range(num_classes)
    ], dtype=torch.float32).to(device)
    weights = weights / weights.sum() * num_classes
    
    print("="*70)
    print("CLASS WEIGHTS (Inverse Frequency)")
    print("="*70)
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        print(f"{label2name[i]:20s} | Count: {count:6d} | Weight: {weights[i]:.4f}")
    print("="*70 + "\n")
    
    return weights

class_weights = compute_class_weights(train_df_expanded, num_classes, device)


def get_loss_fn(loss_name, num_classes):
    """Get loss function"""
    if loss_name == 'bce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return FocalLoss(alpha=class_weights, gamma=2)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

# ============================================================================
# 3. IMPROVED AUGMENTATION WITH DENOISING
# ============================================================================

def get_train_transform_with_denoising():
    """Training transform WITH denoising for Phase 2 noisy data"""
    return A.Compose([
        A.Resize(224, 224),
        
        # Denoising - critical for Phase 2
        A.Lambda(
            image=lambda x, **kwargs: cv2.bilateralFilter(x, 9, 75, 75),
            name="bilateral_denoise"
        ),
        
        # Geometric augmentations
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.3),
        
        # Staining & noise variations
        A.CLAHE(clip_limit=2.0, p=0.5),
        A.OneOf([
            A.MotionBlur(blur_limit=3, p=0.2),
            A.MedianBlur(blur_limit=3, p=0.1),
            A.GaussNoise(var_limit=(5.0, 15.0), p=0.2),
        ], p=0.3),
        
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])


def get_val_transform():
    """Validation transform with light denoising"""
    return A.Compose([
        A.Resize(224, 224),
        A.Lambda(
            image=lambda x, **kwargs: cv2.bilateralFilter(x, 5, 50, 50),
            name="bilateral_denoise"
        ),
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
            augmented = self.transform(image=image)
            image = augmented["image"]
        
        if self.is_test:
            return image, row["filename"]
        
        label = torch.tensor(row["label_id"], dtype=torch.long)
        return image, label


# Create datasets with improved transforms
train_dataset = BloodDataset(train_df_split, transform=get_train_transform_with_denoising())
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
# 5. MODEL DEFINITIONS
# ============================================================================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


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


class SwinTransformerImage(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model("swin_base_patch4_window7_224", 
                                         pretrained=pretrained, num_classes=0, in_chans=3)
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)
        print(f"  SwinTransformerImage: {sum(p.numel() for p in self.parameters())/1e6:.1f}M params")
    
    def forward(self, x):
        return self.classifier(self.backbone(x))


# SE Attention Block
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
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),  # No downsampling
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),  # No downsampling
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SEBlock(128),
            nn.Conv2d(128, 3, kernel_size=1, stride=1, padding=0, bias=False),  # Keep 3 channels for Swin input
            nn.BatchNorm2d(3),
            nn.ReLU()
        )

        self.swin = timm.create_model('swin_base_patch4_window7_224', pretrained=True, num_classes=512)

        self.fc = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.conv_stem(x)  # Ensure it maintains 224×224 size
        x = self.swin(x)  # Pass to Swin Transformer
        x = self.fc(x)  # Final classification layer
        return x


def get_model(model_name, num_classes, device):
    print(f"\n{'='*60}\nInitializing {model_name}\n{'='*60}")
    if model_name == 'SwinTransformerImage':
        model = SwinTransformerImage(num_classes=num_classes)
    elif model_name == 'HybridSwin':
        model = HybridSwin(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)


# ============================================================================
# 6. MIXUP AUGMENTATION
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
# 7. TRAINING FUNCTIONS
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
# 8. TEST-TIME AUGMENTATION
# ============================================================================

def predict_with_tta(model, images, device, n_tta=5):
    model.eval()
    predictions = []
    with torch.no_grad():
        predictions.append(torch.softmax(model(images), dim=1))
        if n_tta > 1:
            predictions.append(torch.softmax(model(torch.flip(images, dims=[3])), dim=1))
        if n_tta > 2:
            predictions.append(torch.softmax(model(torch.flip(images, dims=[2])), dim=1))
        if n_tta > 3:
            predictions.append(torch.softmax(model(torch.rot90(images, k=1, dims=[2, 3])), dim=1))
        if n_tta > 4:
            predictions.append(torch.softmax(model(torch.rot90(images, k=3, dims=[2, 3])), dim=1))
    return torch.stack(predictions).mean(dim=0).argmax(dim=1)


def generate_submission_with_tta(model, test_loader, device, label2name, output_path, use_tta=True):
    model.eval()
    predictions, filenames = [], []
    desc = "Generating predictions (with TTA)" if use_tta else "Generating predictions"
    
    for images, ids in tqdm(test_loader, desc=desc):
        images = images.to(device)
        if use_tta:
            preds = predict_with_tta(model, images, device, n_tta=5)
        else:
            with torch.no_grad():
                preds = model(images).argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
        filenames.extend(ids)
    
    pred_labels = [label2name[pred] for pred in predictions]
    submission_df = pd.DataFrame({'ID': filenames, 'Target': pred_labels})
    submission_df.to_csv(output_path, index=False)
    print(f"\n✓ Submission saved: {output_path}")
    print(submission_df.head(10))
    return submission_df


# ============================================================================
# 9. VISUALIZATION FUNCTIONS
# ============================================================================

def plot_training_curves(history, model_name, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    axes[0].plot(epochs, history['train_loss'], 'b-', label='Train', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='Val', linewidth=2)
    axes[0].set_title(f'{model_name} - Loss', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], 'b-', label='Train', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='Val', linewidth=2)
    axes[1].set_title(f'{model_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    axes[2].plot(epochs, history['train_f1'], 'b-', label='Train', linewidth=2)
    axes[2].plot(epochs, history['val_f1'], 'r-', label='Val', linewidth=2)
    axes[2].set_title(f'{model_name} - F1 Score', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('F1 Score')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    best_epoch = np.argmax(history['val_f1']) + 1
    for ax in axes:
        ax.axvline(x=best_epoch, color='g', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_training_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Training curves saved")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir='plots'):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax1, cbar_kws={'label': 'Count'})
    ax1.set_title(f'{model_name} - Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('True Label')
    ax1.set_xlabel('Predicted Label')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')
    
    sns.heatmap(cm_percent, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, cbar_kws={'label': '%'}, vmin=0, vmax=100)
    ax2.set_title(f'{model_name} - Confusion Matrix (%)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('True Label')
    ax2.set_xlabel('Predicted Label')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{model_name}_confusion_matrix.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Confusion matrix saved")


# ============================================================================
# 10. MAIN TRAINING PIPELINE
# ============================================================================

def train_model(config, train_loader, val_loader, test_loader, num_classes, 
                label2name, device, class_weights, save_dir='models'):
    model_name = config['model']
    print(f"\n{'#'*70}\n# Training: {model_name}\n{'#'*70}\n")
    
    model = get_model(model_name, num_classes, device)
    # criterion = FocalLoss(alpha=class_weights, gamma=2)
    criterion = get_loss_fn(config['loss'], num_classes)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    
    if config['scheduler'] == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    else:
        scheduler = None
    
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_best.pth")
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc, train_f1 = train_epoch_with_mixup(
            model, train_loader, criterion, optimizer, device, epoch, use_mixup=True)
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch)
        
        if scheduler:
            scheduler.step()
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_f1'].append(train_f1)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_f1'].append(val_f1)
        
        print(f"\nEpoch {epoch}/{config['epochs']}:")
        print(f"  Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, F1: {train_f1:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict(), 'epoch': epoch}, model_path)
            print(f"  ✓ New best model saved! (F1: {val_f1:.4f})")
    
    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Generate visualizations
    plot_training_curves(history, model_name)
    _, _, _, val_preds, val_labels = validate(model, val_loader, criterion, device, "FINAL")
    plot_confusion_matrix(val_labels, val_preds, class_names, model_name)
    
    print("\n" + "="*60)
    print(classification_report(val_labels, val_preds, target_names=class_names))
    
    # Generate submission with TTA
    submission_path = f"submission_{model_name}.csv"
    submission_df = generate_submission_with_tta(model, test_loader, device, label2name, 
                                                 submission_path, use_tta=True)
    
    return model, history, submission_df


# ============================================================================
# 11. RUN TRAINING
# ============================================================================

configs = [
    {'model': 'SwinTransformerImage', 'loss': 'bce', 'lr': 5e-5, 'epochs': 45, 'weight_decay': 1e-4, 'scheduler': 'cosine'},
    {'model': 'HybridSwin', 'loss': 'bce', 'lr': 5e-5, 'epochs': 45, 'weight_decay': 1e-4, 'scheduler': 'cosine'},
]


results = {}
for config in configs:
    try:
        model, history, submission = train_model(
            config, train_loader, val_loader, test_loader, 
            num_classes, label2name, device, class_weights)
        results[config['model']] = {
            'model': model, 'history': history, 
            'submission': submission, 'best_val_f1': max(history['val_f1'])
        }
    except Exception as e:
        print(f"\n❌ Error training {config['model']}: {str(e)}")
        import traceback
        traceback.print_exc()

print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)
if results:
    for name, result in results.items():
        print(f"{name:30s} | Best Val F1: {result['best_val_f1']:.4f}")
    best = max(results, key=lambda k: results[k]['best_val_f1'])
    print(f"\n🏆 Best Single Model: {best} (F1: {results[best]['best_val_f1']:.4f})")
print("="*70)


# ============================================================================
# 12. ENSEMBLE PREDICTIONS
# ============================================================================

def ensemble_predictions(models_dict, test_loader, device, label2name, 
                         output_path='submission_ensemble.csv', weights=None):
    """
    Ensemble multiple models with optional weighting.
    
    Args:
        models_dict: Dict of {model_name: model}
        test_loader: Test DataLoader
        device: torch device
        label2name: Dict mapping label_id to class name
        output_path: Path to save ensemble submission
        weights: Optional dict of {model_name: weight}. If None, uses validation F1 scores.
    
    Returns:
        submission_df: DataFrame with ensemble predictions
    """
    print("\n" + "="*70)
    print("CREATING ENSEMBLE PREDICTIONS")
    print("="*70)
    
    # Collect predictions from all models
    all_predictions = {name: [] for name in models_dict.keys()}
    filenames = []
    
    for images, ids in tqdm(test_loader, desc="Ensemble inference"):
        images = images.to(device)
        
        if len(filenames) == 0:
            filenames.extend(ids)
        
        for model_name, model in models_dict.items():
            model.eval()
            with torch.no_grad():
                # Use TTA for ensemble
                outputs = predict_with_tta(model, images, device, n_tta=5)
                # Get probabilities instead of hard predictions
                model.eval()
                with torch.no_grad():
                    raw_outputs = model(images)
                    probs = torch.softmax(raw_outputs, dim=1)
                all_predictions[model_name].append(probs.cpu())
    
    # Concatenate all predictions
    for model_name in models_dict.keys():
        all_predictions[model_name] = torch.cat(all_predictions[model_name], dim=0)
    
    # Determine weights
    if weights is None:
        # Use validation F1 scores as weights
        total_f1 = sum(results[name]['best_val_f1'] for name in models_dict.keys())
        weights = {name: results[name]['best_val_f1'] / total_f1 
                   for name in models_dict.keys()}
    
    print("\nEnsemble Weights:")
    for name, weight in weights.items():
        print(f"  {name:30s} : {weight:.4f}")
    
    # Weighted ensemble
    ensemble_probs = torch.zeros_like(all_predictions[list(models_dict.keys())[0]])
    for model_name, probs in all_predictions.items():
        weight = weights.get(model_name, 1.0 / len(models_dict))
        ensemble_probs += weight * probs
    
    # Get final predictions
    final_predictions = ensemble_probs.argmax(dim=1).numpy()
    pred_labels = [label2name[pred] for pred in final_predictions]
    
    # Create submission
    submission_df = pd.DataFrame({
        'ID': filenames,
        'Target': pred_labels
    })
    
    submission_df.to_csv(output_path, index=False)
    
    print(f"\n✓ Ensemble submission saved: {output_path}")
    print(f"  Total predictions: {len(submission_df):,}")
    print(f"  Ensemble of {len(models_dict)} models")
    print("\nSample predictions:")
    print(submission_df.head(10))
    print("="*70)
    
    return submission_df


def evaluate_ensemble_on_validation(models_dict, val_loader, device, weights=None):
    """
    Evaluate ensemble performance on validation set.
    """
    print("\n" + "="*70)
    print("EVALUATING ENSEMBLE ON VALIDATION SET")
    print("="*70)
    
    all_predictions = {name: [] for name in models_dict.keys()}
    all_labels = []
    
    for images, labels in tqdm(val_loader, desc="Validation ensemble"):
        images = images.to(device)
        
        if len(all_labels) == 0:
            all_labels.extend(labels.numpy())
        
        for model_name, model in models_dict.items():
            model.eval()
            with torch.no_grad():
                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                all_predictions[model_name].append(probs.cpu())
    
    # Concatenate
    for model_name in models_dict.keys():
        all_predictions[model_name] = torch.cat(all_predictions[model_name], dim=0)
    
    # Determine weights
    if weights is None:
        total_f1 = sum(results[name]['best_val_f1'] for name in models_dict.keys())
        weights = {name: results[name]['best_val_f1'] / total_f1 
                   for name in models_dict.keys()}
    
    # Weighted ensemble
    ensemble_probs = torch.zeros_like(all_predictions[list(models_dict.keys())[0]])
    for model_name, probs in all_predictions.items():
        weight = weights.get(model_name, 1.0 / len(models_dict))
        ensemble_probs += weight * probs
    
    # Get predictions
    ensemble_preds = ensemble_probs.argmax(dim=1).numpy()
    
    # Calculate metrics
    ensemble_acc = accuracy_score(all_labels, ensemble_preds)
    ensemble_f1 = f1_score(all_labels, ensemble_preds, average='macro')
    
    print(f"\n📊 Ensemble Validation Results:")
    print(f"  Accuracy: {ensemble_acc:.4f}")
    print(f"  Macro F1: {ensemble_f1:.4f}")
    
    print(f"\n📈 Comparison with Individual Models:")
    for name in models_dict.keys():
        individual_f1 = results[name]['best_val_f1']
        improvement = ensemble_f1 - individual_f1
        print(f"  {name:30s} : F1 = {individual_f1:.4f} | "
              f"Ensemble gain = {improvement:+.4f}")
    
    print(f"\n🎯 Best Ensemble Improvement: {ensemble_f1 - max(results[name]['best_val_f1'] for name in models_dict.keys()):+.4f}")
    
    # Classification report
    print("\nEnsemble Classification Report:")
    print("="*70)
    print(classification_report(all_labels, ensemble_preds, target_names=class_names))
    
    # Confusion matrix
    plot_confusion_matrix(all_labels, ensemble_preds, class_names, "Ensemble", save_dir='plots')
    
    print("="*70)
    
    return ensemble_f1


# ============================================================================
# 13. CREATE AND EVALUATE ENSEMBLE
# ============================================================================

if results and len(results) >= 2:
    print("\n" + "="*70)
    print("BUILDING ENSEMBLE FROM TRAINED MODELS")
    print("="*70)
    
    # Prepare models dictionary
    ensemble_models = {name: result['model'] for name, result in results.items()}
    
    # Evaluate ensemble on validation set first
    ensemble_val_f1 = evaluate_ensemble_on_validation(
        ensemble_models, val_loader, device, weights=None
    )
    
    # Generate ensemble predictions on test set
    ensemble_submission = ensemble_predictions(
        ensemble_models, 
        test_loader, 
        device, 
        label2name,
        output_path='submission_ensemble.csv',
        weights=None  # Uses validation F1 scores as weights
    )
    
    # Add ensemble to results
    results['Ensemble'] = {
        'submission': ensemble_submission,
        'best_val_f1': ensemble_val_f1
    }
    
    print("\n" + "="*70)
    print("FINAL RESULTS SUMMARY")
    print("="*70)
    for name, result in results.items():
        if 'best_val_f1' in result:
            print(f"{name:30s} | Validation F1: {result['best_val_f1']:.4f}")
    
    best_final = max(results, key=lambda k: results[k]['best_val_f1'])
    print(f"\n🏆 BEST OVERALL: {best_final} (F1: {results[best_final]['best_val_f1']:.4f})")
    print("="*70)
    
    print("\n✅ ALL SUBMISSIONS GENERATED:")
    for name in results.keys():
        if name == 'Ensemble':
            print(f"  • submission_ensemble.csv (Ensemble of all models)")
        else:
            print(f"  • submission_{name}.csv")
    
    print("\n💡 RECOMMENDATION: Submit 'submission_ensemble.csv' for best results!")
    print("="*70)

elif results:
    print("\n⚠️ Only one model trained. Need at least 2 models for ensemble.")
    print("   Using single model submission.")
else:
    print("\n❌ No models trained successfully. Cannot create ensemble.")