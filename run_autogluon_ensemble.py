"""
Standalone AutoGluon Ensemble Script
=====================================
Run AutoGluon meta-ensemble on existing trained models WITHOUT retraining.

Usage:
    python run_autogluon_only.py

Prerequisites:
    - Trained models in models_improved/ directory
    - Model files named: <model_name>_best.pth
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, classification_report
import cv2
import timm
import warnings
import math
from typing import Optional, List
from autogluon.tabular import TabularPredictor, TabularDataset
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
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

SAVE_DIR = 'models_improved'
SUBMISSION_DIR = 'submissions_improved'

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device: {device}")

# ============================================================================
# COPY MODEL CLASSES (needed for loading checkpoints)
# ============================================================================

class LoRAQKV(nn.Module):
    def __init__(self, qkv: nn.Module, r: int, alpha: int):
        super().__init__()
        self.qkv = qkv
        self.dim = qkv.in_features
        self.r = r
        self.alpha = alpha
        
        self.lora_q_A = nn.Linear(self.dim, r, bias=False)
        self.lora_q_B = nn.Linear(r, self.dim, bias=False)
        self.lora_v_A = nn.Linear(self.dim, r, bias=False)
        self.lora_v_B = nn.Linear(r, self.dim, bias=False)
        
        nn.init.kaiming_uniform_(self.lora_q_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_q_B.weight)
        nn.init.kaiming_uniform_(self.lora_v_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_v_B.weight)
    
    def forward(self, x):
        qkv = self.qkv(x)
        new_q = self.lora_q_B(self.lora_q_A(x))
        new_v = self.lora_v_B(self.lora_v_A(x))
        scale = self.alpha / self.r
        qkv[:, :, :self.dim] += scale * new_q
        qkv[:, :, -self.dim:] += scale * new_v
        return qkv

def inject_lora_into_vit(model: nn.Module, r: int = 4, alpha: int = 4, 
                         target_layers: Optional[List[int]] = None):
    if not hasattr(model, 'blocks'):
        return model, []
    
    if target_layers is None:
        target_layers = list(range(len(model.blocks)))
    
    lora_params = []
    for i, block in enumerate(model.blocks):
        if i not in target_layers:
            continue
        old_qkv = block.attn.qkv
        block.attn.qkv = LoRAQKV(old_qkv, r, alpha)
        lora_params.extend([
            block.attn.qkv.lora_q_A.weight,
            block.attn.qkv.lora_q_B.weight,
            block.attn.qkv.lora_v_A.weight,
            block.attn.qkv.lora_v_B.weight,
        ])
    
    return model, lora_params

class FoundationModelClassifier(nn.Module):
    def __init__(self, backbone_name: str, num_classes: int, 
                 use_lora: bool = True, lora_r: int = 4, lora_alpha: int = 4,
                 pretrained: bool = False, dropout: float = 0.3):
        super().__init__()
        
        # Load backbone (pretrained=False since we'll load trained weights)
        if 'vit' in backbone_name.lower():
            self.backbone = timm.create_model(
                backbone_name, 
                pretrained=pretrained,
                num_classes=0,
                dynamic_img_size=True
            )
        elif 'convnext' in backbone_name.lower():
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0
            )
        else:
            self.backbone = timm.create_model(
                backbone_name,
                pretrained=pretrained,
                num_classes=0
            )
        
        # Apply LoRA
        self.use_lora = use_lora
        if use_lora and hasattr(self.backbone, 'blocks'):
            self.backbone, self.lora_params = inject_lora_into_vit(
                self.backbone, r=lora_r, alpha=lora_alpha
            )
            for name, param in self.backbone.named_parameters():
                if 'lora' not in name:
                    param.requires_grad = False
        else:
            self.lora_params = []
        
        # Get feature dimension
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dim = features.shape[-1]
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.LayerNorm(self.feature_dim),
            nn.Dropout(dropout),
            nn.Linear(self.feature_dim, 512),
            nn.GELU(),
            nn.LayerNorm(512),
            nn.Dropout(dropout / 2),
            nn.Linear(512, num_classes)
        )
    
    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)

# ============================================================================
# DATA LOADING
# ============================================================================

def val_transform():
    return A.Compose([
        A.LongestMaxSize(max_size=224, interpolation=cv2.INTER_LINEAR),
        A.PadIfNeeded(
            min_height=224, min_width=224,
            border_mode=cv2.BORDER_CONSTANT,
            value=(255, 255, 255)
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        A.ToTensorV2()
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

def load_data():
    print("\n" + "="*70)
    print("LOADING DATA")
    print("="*70)
    
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
    
    train_df_expanded = pd.concat([
        phase1_df, phase2_train_df, phase2_eval_df
    ], ignore_index=True)
    
    test_df = phase2_test_df.copy()
    
    class_names = sorted(train_df_expanded["label"].unique())
    num_classes = len(class_names)
    label2name = dict(zip(range(num_classes), class_names))
    name2label = {v: k for k, v in label2name.items()}
    
    train_df_expanded["label_id"] = train_df_expanded["label"].map(name2label)
    test_df["label_id"] = -1
    
    train_indices, val_indices = train_test_split(
        range(len(train_df_expanded)),
        test_size=0.1,
        stratify=train_df_expanded['label_id'],
        random_state=42
    )
    
    train_df = train_df_expanded.iloc[train_indices].reset_index(drop=True)
    val_df = train_df_expanded.iloc[val_indices].reset_index(drop=True)
    
    print(f"Training split:   {len(train_df):6,} images")
    print(f"Validation split: {len(val_df):6,} images")
    print(f"Test set:         {len(test_df):6,} images")
    print("="*70 + "\n")
    
    return train_df, val_df, test_df, num_classes, label2name, class_names

# ============================================================================
# LOAD EXISTING MODELS
# ============================================================================

def load_trained_model(model_path, config, num_classes, device):
    """Load a trained model from checkpoint"""
    print(f"\nLoading model: {config['name']}")
    
    # Create model architecture
    model = FoundationModelClassifier(
        backbone_name=config['backbone'],
        num_classes=num_classes,
        use_lora=config.get('use_lora', True),
        lora_r=config.get('lora_r', 4),
        lora_alpha=config.get('lora_alpha', 4),
        pretrained=False,  # Don't load pretrained weights
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    # Load checkpoint
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        val_f1 = checkpoint.get('val_f1', 0.0)
        print(f"  ✓ Loaded checkpoint (Val F1: {val_f1:.4f})")
        return model, val_f1
    else:
        raise FileNotFoundError(f"Model not found: {model_path}")

def discover_and_load_models(num_classes, device):
    """Automatically discover and load all trained models"""
    
    # Model configurations (must match training configs)
    configs = [
        {
            'name': 'vit_base_lora',
            'backbone': 'vit_base_patch16_224',
            'use_lora': True,
            'lora_r': 4,
            'lora_alpha': 4,
            'dropout': 0.3
        },
        {
            'name': 'vit_large_lora',
            'backbone': 'vit_large_patch16_224',
            'use_lora': True,
            'lora_r': 4,
            'lora_alpha': 4,
            'dropout': 0.3
        },
        {
            'name': 'convnext_large_lora',
            'backbone': 'convnext_large',
            'use_lora': True,
            'lora_r': 4,
            'lora_alpha': 4,
            'dropout': 0.3
        }
    ]
    
    models_dict = {}
    
    for config in configs:
        model_path = os.path.join(SAVE_DIR, f"{config['name']}_best.pth")
        
        try:
            model, val_f1 = load_trained_model(model_path, config, num_classes, device)
            models_dict[config['name']] = {
                'model': model,
                'best_val_f1': val_f1,
                'config': config
            }
        except FileNotFoundError as e:
            print(f"  ⚠️ Skipping {config['name']}: {e}")
    
    if not models_dict:
        raise RuntimeError("No trained models found! Please train models first.")
    
    print(f"\n✓ Loaded {len(models_dict)} models successfully\n")
    return models_dict

# ============================================================================
# AUTOGLUON ENSEMBLE
# ============================================================================

def create_autogluon_ensemble(models_dict, train_loader, val_loader, test_loader,
                             num_classes, label2name, device):
    """Create AutoGluon meta-ensemble from existing models"""
    
    print("\n" + "="*70)
    print("CREATING AUTOGLUON META-ENSEMBLE")
    print("="*70)
    
    # Extract probabilities
    def extract_probs(models_dict, loader, phase_name):
        all_probs_dict = {name: [] for name in models_dict.keys()}
        all_labels = []
        all_ids = []
        
        for images, labels_or_ids in tqdm(loader, desc=f"Extracting {phase_name} probs"):
            images = images.to(device)
            
            for name, result_dict in models_dict.items():
                model = result_dict['model']
                model.eval()
                with torch.no_grad():
                    probs = F.softmax(model(images), dim=1)
                all_probs_dict[name].append(probs.cpu())
            
            if phase_name != 'test':
                all_labels.extend(labels_or_ids.numpy())
            else:
                all_ids.extend(labels_or_ids)
        
        # Concatenate
        for name in all_probs_dict:
            all_probs_dict[name] = torch.cat(all_probs_dict[name], dim=0)
        
        return all_probs_dict, all_labels if phase_name != 'test' else all_ids
    
    # Extract probabilities from all splits
    train_probs, train_labels = extract_probs(models_dict, train_loader, 'train')
    val_probs, val_labels = extract_probs(models_dict, val_loader, 'val')
    test_probs, test_ids = extract_probs(models_dict, test_loader, 'test')
    
    # Create DataFrames
    def create_df(probs_dict, labels, num_classes):
        data = {}
        for model_name, probs in probs_dict.items():
            for class_idx in range(num_classes):
                col_name = f"{model_name}_class{class_idx}"
                data[col_name] = probs[:, class_idx].numpy()
        data['label'] = labels
        return pd.DataFrame(data)
    
    train_df = create_df(train_probs, train_labels, num_classes)
    val_df = create_df(val_probs, val_labels, num_classes)
    test_df = create_df(test_probs, list(range(len(test_ids))), num_classes)
    
    # Save for reference
    os.makedirs(SAVE_DIR, exist_ok=True)
    train_df.to_csv(os.path.join(SAVE_DIR, 'autogluon_train.csv'), index=False)
    val_df.to_csv(os.path.join(SAVE_DIR, 'autogluon_val.csv'), index=False)
    test_df.to_csv(os.path.join(SAVE_DIR, 'autogluon_test.csv'), index=False)
    print("✓ Saved probability features")
    
    # Train or load AutoGluon
    autogluon_path = os.path.join(SAVE_DIR, 'AutogluonModels')
    
    if os.path.exists(autogluon_path):
        print(f"\n✓ Found existing AutoGluon model")
        print("→ Loading existing model (delete models_improved/AutogluonModels/ to retrain)")
        predictor = TabularPredictor.load(autogluon_path)
    else:
        print("\nTraining AutoGluon meta-model...")
        predictor = TabularPredictor(
            label='label',
            problem_type='multiclass',
            eval_metric='f1_macro',
            path=autogluon_path
        )
        
        # Combine train and val data (AutoGluon will use bagging internally)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        predictor.fit(
            TabularDataset(combined_df),
            presets='best_quality',
            num_stack_levels=3,
            dynamic_stacking=False,  # Skip DyStack since we know 3 levels works
            time_limit=36000  # 10 hours max
        )
    
    # Evaluate (using validation data)
    val_preds = predictor.predict(TabularDataset(val_df))
    val_f1 = f1_score(val_labels, val_preds, average='macro')
    
    print(f"\n✓ AutoGluon Validation F1: {val_f1:.4f}")
    print("\nPer-class F1 scores:")
    val_f1_per_class = f1_score(val_labels, val_preds, average=None)
    for i, f1 in enumerate(val_f1_per_class):
        class_name = label2name[i] if i in label2name else f"Class_{i}"
        print(f"  {class_name:15s}: {f1:.4f}")
    
    print("\nClassification Report:")
    print(classification_report(val_labels, val_preds))
    
    # Test predictions
    test_preds = predictor.predict(TabularDataset(test_df))
    pred_labels = [label2name[int(p)] for p in test_preds]
    
    submission_df = pd.DataFrame({'ID': test_ids, 'Target': pred_labels})
    os.makedirs(SUBMISSION_DIR, exist_ok=True)
    submission_path = os.path.join(SUBMISSION_DIR, 'submission_autogluon_ensemble.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✓ AutoGluon submission saved: {submission_path}")
    
    return predictor, val_f1

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*70)
    print(" AUTOGLUON ENSEMBLE - USING EXISTING MODELS")
    print("="*70)
    print("\nThis script will:")
    print("  1. Load existing trained models from models_improved/")
    print("  2. Extract probability predictions")
    print("  3. Train AutoGluon meta-ensemble")
    print("  4. Generate submission file")
    print("\n⚠️  NO RETRAINING - Uses existing model checkpoints")
    print("="*70 + "\n")
    
    # Load data
    train_df, val_df, test_df, num_classes, label2name, class_names = load_data()
    
    # Create datasets
    train_dataset = BloodDataset(train_df, transform=val_transform())
    val_dataset = BloodDataset(val_df, transform=val_transform())
    test_dataset = BloodDataset(test_df, transform=val_transform(), is_test=True)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, 
                             num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, 
                           num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, 
                            num_workers=4, pin_memory=True)
    
    # Load existing models
    models_dict = discover_and_load_models(num_classes, device)
    
    print("="*70)
    print("LOADED MODELS:")
    print("="*70)
    for name, result in models_dict.items():
        print(f"{name:30s} | Val F1: {result['best_val_f1']:.4f}")
    print("="*70)
    
    # Create AutoGluon ensemble
    predictor, autogluon_f1 = create_autogluon_ensemble(
        models_dict, train_loader, val_loader, test_loader,
        num_classes, label2name, device
    )
    
    # Compare results
    print("\n" + "="*70)
    print("FINAL COMPARISON")
    print("="*70)
    for name, result in models_dict.items():
        print(f"{name:30s} | Val F1: {result['best_val_f1']:.4f}")
    print(f"{'AutoGluon Ensemble':30s} | Val F1: {autogluon_f1:.4f}")
    
    best_single_f1 = max(r['best_val_f1'] for r in models_dict.values())
    improvement = autogluon_f1 - best_single_f1
    
    print("\n" + "="*70)
    if improvement > 0:
        print(f"🎉 AutoGluon improved by +{improvement:.4f} F1!")
        print(f"✅ SUBMIT: submissions_improved/submission_autogluon_ensemble.csv")
    else:
        print(f"⚠️  AutoGluon did not improve ({improvement:.4f} F1)")
        best_model = max(models_dict, key=lambda k: models_dict[k]['best_val_f1'])
        print(f"✅ SUBMIT: submissions_improved/submission_{best_model}.csv instead")
    print("="*70 + "\n")

if __name__ == "__main__":
    main()