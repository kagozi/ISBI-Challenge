"""
WBCBench 2026: Advanced Training Pipeline with AutoGluon Meta-Learning
==================================================
Key Improvements:
1. AutoGluon meta-learner for intelligent ensemble
2. Effective number sampling for class imbalance
3. Label smoothing + improved class weights
4. Better model architectures
5. Multi-model feature extraction for AutoGluon
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
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import timm
import warnings
from autogluon.tabular import TabularPredictor, TabularDataset
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

# Combine ALL training data
train_df_expanded = pd.concat([
    phase1_df,
    phase2_train_df,
    phase2_eval_df
], ignore_index=True)

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

# ============================================================================
# 2. IMPROVED CLASS WEIGHTING (Effective Number of Samples)
# ============================================================================

def get_effective_num_samples(class_counts, beta=0.9999):
    """
    Calculate effective number of samples to handle class imbalance.
    From "Class-Balanced Loss Based on Effective Number of Samples"
    """
    effective_nums = (1 - np.power(beta, class_counts)) / (1 - beta)
    return effective_nums

def compute_balanced_class_weights(train_df, num_classes, device):
    """Compute balanced weights using effective number of samples"""
    class_counts = train_df['label_id'].value_counts().sort_index()
    
    # Use effective number of samples
    effective_nums = get_effective_num_samples(class_counts.values, beta=0.9999)
    weights = 1.0 / effective_nums
    weights = weights / weights.sum() * num_classes
    weights = torch.tensor(weights, dtype=torch.float32).to(device)
    
    # Clip extreme weights
    weights = torch.clamp(weights, min=0.5, max=5.0)
    
    print("="*70)
    print("CLASS WEIGHTS (Effective Number of Samples)")
    print("="*70)
    for i in range(num_classes):
        count = class_counts.get(i, 0)
        print(f"{label2name[i]:20s} | Count: {count:6d} | Weight: {weights[i]:.4f}")
    print("="*70 + "\n")
    
    return weights

def get_weighted_sampler(train_df):
    """Create weighted sampler with effective number approach"""
    class_counts = train_df['label_id'].value_counts()
    effective_nums = get_effective_num_samples(class_counts.values, beta=0.9999)
    weights_per_class = 1.0 / effective_nums
    weights_per_class = weights_per_class / weights_per_class.sum() * len(class_counts)
    
    # Convert to sample weights
    sample_weights = [weights_per_class[label_id] for label_id in train_df['label_id']]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_df),  # NOT multiplied - prevents overfitting
        replacement=True
    )
    return sampler

# ============================================================================
# 3. IMPROVED LOSS FUNCTIONS
# ============================================================================

class LabelSmoothingCrossEntropy(nn.Module):
    """Label smoothing for fine-grained classification"""
    def __init__(self, epsilon=0.1, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.weight = weight
    
    def forward(self, pred, target):
        n_classes = pred.size(-1)
        log_preds = F.log_softmax(pred, dim=-1)
        
        # Smooth labels
        smooth_target = torch.zeros_like(log_preds).scatter_(
            1, target.unsqueeze(1), 1.0 - self.epsilon
        )
        smooth_target += self.epsilon / n_classes
        
        loss = (-smooth_target * log_preds).sum(dim=-1)
        
        if self.weight is not None:
            loss = loss * self.weight[target]
        
        return loss.mean()

class PolyLoss(nn.Module):
    """
    Poly Loss from winning solution - works better than focal for fine-grained tasks
    """
    def __init__(self, num_classes, epsilon=1.0, weight=None):
        super().__init__()
        self.epsilon = epsilon
        self.num_classes = num_classes
        self.weight = weight
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
    
    def forward(self, output, target):
        ce = self.criterion(output, target)
        pt = F.one_hot(target, num_classes=self.num_classes) * self.log_softmax(output)
        
        if self.weight is not None:
            ce *= self.weight[target]
        
        return (ce + self.epsilon * (1.0 - pt.sum(dim=-1))).mean()

def get_loss_fn(loss_name, num_classes, class_weights):
    """Get loss function"""
    if loss_name == 'ce':
        return nn.CrossEntropyLoss(weight=class_weights)
    elif loss_name == 'ce_smooth':
        return LabelSmoothingCrossEntropy(epsilon=0.1, weight=class_weights)
    elif loss_name == 'poly':
        return PolyLoss(num_classes=num_classes, epsilon=1.0, weight=class_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")

# ============================================================================
# 4. AUGMENTATION
# ============================================================================

def advanced_clahe_preprocessing(image):
    """Enhanced CLAHE on LAB color space"""
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab_clahe = cv2.merge([l_clahe, a, b])
    return cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2RGB)

def get_train_transform():
    return A.Compose([
        A.Resize(224, 224),
        A.Lambda(image=lambda x, **k: cv2.bilateralFilter(x, 7, 50, 50), p=0.5),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.RandomResizedCrop(size=(224, 224), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.08, p=0.4),
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
# 5. DATASET
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
# 6. IMPROVED MODEL ARCHITECTURES
# ============================================================================

class ImprovedClassificationHead(nn.Module):
    def __init__(self, in_dim, num_classes, dropout=0.4):
        super().__init__()
        self.head = nn.Sequential(
            nn.BatchNorm1d(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 1024),
            nn.GELU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(dropout / 2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout / 4),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        return self.head(x)

class ImprovedSwinTransformer(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "swin_base_patch4_window7_224",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        self.classifier = ImprovedClassificationHead(
            self.backbone.num_features, num_classes, dropout
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

class ConvNeXt(nn.Module):
    """ConvNeXt often outperforms EfficientNet for medical images"""
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "convnext_base",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        self.classifier = ImprovedClassificationHead(
            self.backbone.num_features, num_classes, dropout
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

class MaxViT(nn.Module):
    """MaxViT - hybrid architecture good for medical imaging"""
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "maxvit_base_tf_224",
            pretrained=pretrained,
            num_classes=0,
            in_chans=3
        )
        self.classifier = ImprovedClassificationHead(
            self.backbone.num_features, num_classes, dropout
        )
    
    def forward(self, x):
        return self.classifier(self.backbone(x))

def get_model(model_name, num_classes, device):
    print(f"\n{'='*60}\nInitializing {model_name}\n{'='*60}")
    if model_name == 'SwinTransformer':
        model = ImprovedSwinTransformer(num_classes=num_classes)
    elif model_name == 'ConvNeXt':
        model = ConvNeXt(num_classes=num_classes)
    elif model_name == 'MaxViT':
        model = MaxViT(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

# ============================================================================
# 7. TRAINING FUNCTIONS
# ============================================================================

def mixup_data(x, y, alpha=0.4):
    """Mixup with higher alpha for better regularization"""
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

def train_epoch(model, loader, criterion, optimizer, device, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [TRAIN]")
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)
        
        # Use mixup 70% of the time
        if use_mixup and np.random.rand() > 0.3:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.4)
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
# 8. FEATURE EXTRACTION FOR AUTOGLUON
# ============================================================================

@torch.no_grad()
def extract_features_and_probabilities(model, loader, device, model_name):
    """
    Extract both softmax probabilities AND penultimate layer features.
    This gives AutoGluon more information to work with.
    """
    model.eval()
    all_probs = []
    all_labels = []
    all_filenames = []
    
    for batch in tqdm(loader, desc=f"Extracting {model_name} features"):
        if len(batch) == 2:
            images, labels = batch
            images = images.to(device)
            
            # Get logits
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            
            all_probs.append(probs.cpu().numpy())
            if isinstance(labels, torch.Tensor):
                all_labels.extend(labels.cpu().numpy())
            else:
                all_filenames.extend(labels)
    
    all_probs = np.concatenate(all_probs, axis=0)
    
    # Create column names for each class probability
    prob_cols = {f"{model_name}_prob_{i}": all_probs[:, i] for i in range(all_probs.shape[1])}
    
    if all_labels:
        return prob_cols, all_labels
    else:
        return prob_cols, all_filenames

# ============================================================================
# 9. AUTOGLUON META-LEARNING ENSEMBLE (KEY INNOVATION)
# ============================================================================

def train_autogluon_ensemble(models_dict, train_loader, val_loader, test_loader, 
                            num_classes, label2name, device, save_path='autogluon_ensemble'):
    """
    Train AutoGluon meta-learner on predictions from all models.
    This is the KEY technique from the winning solution!
    """
    print("\n" + "="*70)
    print("TRAINING AUTOGLUON META-LEARNER")
    print("="*70)
    
    # Extract features from all models for train set
    train_features = {}
    train_labels = None
    
    for model_name, model in models_dict.items():
        print(f"\nExtracting features from {model_name} on training set...")
        probs, labels = extract_features_and_probabilities(
            model, train_loader, device, model_name
        )
        train_features.update(probs)
        if train_labels is None:
            train_labels = labels
    
    # Create training DataFrame
    train_df = pd.DataFrame(train_features)
    train_df['label'] = train_labels
    
    # Extract features for validation set
    val_features = {}
    val_labels = None
    
    for model_name, model in models_dict.items():
        print(f"\nExtracting features from {model_name} on validation set...")
        probs, labels = extract_features_and_probabilities(
            model, val_loader, device, model_name
        )
        val_features.update(probs)
        if val_labels is None:
            val_labels = labels
    
    # Create validation DataFrame
    val_df = pd.DataFrame(val_features)
    val_df['label'] = val_labels
    
    # Save intermediate files (useful for debugging)
    os.makedirs('autogluon_features', exist_ok=True)
    train_df.to_csv('autogluon_features/train_features.csv', index=False)
    val_df.to_csv('autogluon_features/val_features.csv', index=False)
    
    print("\n" + "="*70)
    print("Training AutoGluon with best_quality preset...")
    print("="*70)
    
    # Train AutoGluon meta-learner
    predictor = TabularPredictor(
        label='label',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path=save_path
    )
    
    predictor.fit(
        TabularDataset(train_df),
        tuning_data=TabularDataset(val_df),
        presets='best_quality',  # Use best quality for maximum performance
        use_bag_holdout=True,
        num_stack_levels=3,  # Deep stacking for better ensemble
        time_limit=36000,  # 10 hours - adjust based on your compute
    )
    
    # Evaluate on validation
    print("\n" + "="*70)
    print("AutoGluon Validation Performance")
    print("="*70)
    val_performance = predictor.evaluate(TabularDataset(val_df))
    print(f"Validation F1: {val_performance['f1_macro']:.4f}")
    
    val_preds = predictor.predict(TabularDataset(val_df))
    val_f1_per_class = f1_score(val_labels, val_preds, average=None)
    print("\nPer-class F1 scores:")
    for i, f1 in enumerate(val_f1_per_class):
        print(f"  {label2name[i]:20s}: {f1:.4f}")
    
    # Extract features for test set
    test_features = {}
    test_filenames = None
    
    for model_name, model in models_dict.items():
        print(f"\nExtracting features from {model_name} on test set...")
        probs, filenames = extract_features_and_probabilities(
            model, test_loader, device, model_name
        )
        test_features.update(probs)
        if test_filenames is None:
            test_filenames = filenames
    
    test_df = pd.DataFrame(test_features)
    test_df['filename'] = test_filenames
    test_df.to_csv('autogluon_features/test_features.csv', index=False)
    
    # Generate predictions
    print("\nGenerating test predictions...")
    test_preds = predictor.predict(TabularDataset(test_df.drop('filename', axis=1)))
    
    # Create submission
    pred_labels = [label2name[int(p)] for p in test_preds]
    submission_df = pd.DataFrame({
        'ID': test_filenames,
        'Target': pred_labels
    })
    
    os.makedirs('submissions_autogluon', exist_ok=True)
    submission_path = 'submissions_autogluon/submission_autogluon_meta.csv'
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✓ AutoGluon submission saved: {submission_path}")
    print("\nSample predictions:")
    print(submission_df.head(10))
    
    return predictor, val_performance['f1_macro'], submission_df

# ============================================================================
# 10. MAIN TRAINING PIPELINE
# ============================================================================

def train_model(config, train_loader, val_loader, num_classes, device, 
                class_weights, save_dir='models_autogluon'):
    model_name = config['model']
    loss_name = config['loss']
    
    print(f"\n{'#'*70}\n# Training: {model_name} with {loss_name}\n{'#'*70}\n")
    
    model = get_model(model_name, num_classes, device)
    criterion = get_loss_fn(loss_name, num_classes, class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], 
                           weight_decay=config['weight_decay'])
    scheduler = CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    
    best_val_f1 = 0.0
    history = {'train_loss': [], 'train_acc': [], 'train_f1': [],
               'val_loss': [], 'val_acc': [], 'val_f1': []}
    
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{model_name}_{loss_name}_best.pth")
    
    for epoch in range(1, config['epochs'] + 1):
        train_loss, train_acc, train_f1 = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_mixup=True
        )
        val_loss, val_acc, val_f1, val_preds, val_labels = validate(
            model, val_loader, criterion, device, epoch
        )
        
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
            torch.save({
                'model_state_dict': model.state_dict(),
                'epoch': epoch,
                'val_f1': val_f1
            }, model_path)
            print(f"  ✓ New best model saved! (F1: {val_f1:.4f})")
    
    # Load best model
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    print(f"\n✓ Best validation F1: {best_val_f1:.4f}")
    
    return model, history, best_val_f1

# ============================================================================
# 11. MAIN FUNCTION
# ============================================================================

def main():
    # Create validation split
    train_indices, val_indices = train_test_split(
        range(len(train_df_expanded)),
        test_size=0.1,
        stratify=train_df_expanded['label_id'],
        random_state=42
    )
    
    train_df_split = train_df_expanded.iloc[train_indices].reset_index(drop=True)
    val_df_split = train_df_expanded.iloc[val_indices].reset_index(drop=True)
    
    # Compute class weights
    class_weights = compute_balanced_class_weights(train_df_split, num_classes, device)
    
    # Create datasets
    train_dataset = BloodDataset(train_df_split, transform=get_train_transform())
    val_dataset = BloodDataset(val_df_split, transform=get_val_transform())
    test_dataset = BloodDataset(test_df, transform=get_val_transform(), is_test=True)
    
    # Create data loaders with weighted sampling
    weighted_sampler = get_weighted_sampler(train_df_split)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        sampler=weighted_sampler,  # Use weighted sampler instead of shuffle
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False,
                            num_workers=4, pin_memory=True)
    
    print(f"Dataset sizes:")
    print(f"  Train: {len(train_dataset):,}")
    print(f"  Val:   {len(val_dataset):,}")
    print(f"  Test:  {len(test_dataset):,}\n")
    
    # Training configurations - use the best loss (poly from winning solution)
    configs = [
        {'model': 'SwinTransformer', 'loss': 'poly', 'lr': 3e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'ConvNeXt', 'loss': 'poly', 'lr': 5e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'MaxViT', 'loss': 'poly', 'lr': 5e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'SwinTransformer', 'loss': 'ce', 'lr': 3e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'ConvNeXt', 'loss': 'ce', 'lr': 5e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'MaxViT', 'loss': 'ce', 'lr': 5e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'SwinTransformer', 'loss': 'ce_smooth', 'lr': 3e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'ConvNeXt', 'loss': 'ce_smooth', 'lr': 5e-5, 'epochs': 40, 'weight_decay': 1e-4},
        {'model': 'MaxViT', 'loss': 'ce_smooth', 'lr': 5e-5, 'epochs': 40, 'weight_decay': 1e-4},
    ]
    
    # Train all models
    results = {}
    models_dict = {}
    
    for config in configs:
        try:
            model, history, best_val_f1 = train_model(
                config, train_loader, val_loader, num_classes,
                device, class_weights
            )
            
            key = f"{config['model']}_{config['loss']}"
            results[key] = {
                'model': model,
                'history': history,
                'best_val_f1': best_val_f1
            }
            models_dict[key] = model
            
        except Exception as e:
            print(f"\n❌ Error training {config['model']}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Print training summary
    print("\n" + "="*70)
    print("TRAINING SUMMARY")
    print("="*70)
    for name, result in results.items():
        print(f"{name:30s} | Best Val F1: {result['best_val_f1']:.4f}")
    
    # Train AutoGluon meta-learner (THE KEY INNOVATION!)
    if len(models_dict) >= 2:
        print("\n" + "="*70)
        print("TRAINING AUTOGLUON META-LEARNER")
        print("="*70)
        
        predictor, autogluon_f1, submission_df = train_autogluon_ensemble(
            models_dict, train_loader, val_loader, test_loader,
            num_classes, label2name, device
        )
        
        # Final comparison
        print("\n" + "="*70)
        print("FINAL RESULTS COMPARISON")
        print("="*70)
        for name, result in results.items():
            print(f"{name:30s} | Validation F1: {result['best_val_f1']:.4f}")
        print(f"{'AutoGluon Meta-Learner':30s} | Validation F1: {autogluon_f1:.4f}")
        
        if autogluon_f1 > max(r['best_val_f1'] for r in results.values()):
            print(f"\n🏆 WINNER: AutoGluon Meta-Learner (F1: {autogluon_f1:.4f})")
            print("✅ Submit: submissions_autogluon/submission_autogluon_meta.csv")
        else:
            best = max(results, key=lambda k: results[k]['best_val_f1'])
            print(f"\n🏆 WINNER: {best} (F1: {results[best]['best_val_f1']:.4f})")
        
        print("="*70)
    
    else:
        print("\n⚠️ Need at least 2 models for AutoGluon ensemble.")

if __name__ == "__main__":
    main()