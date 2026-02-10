"""
AutoGluon Meta-Stacking Ensemble for WBCBench 2026
==================================================
Creates a meta-learner that combines predictions from all trained models.

Usage:
    python autogluon_ensemble.py

Requirements:
    pip install autogluon --break-system-packages
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report
import timm
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from autogluon.tabular import TabularPredictor, TabularDataset

# ============================================================================
# MODEL DEFINITIONS (same as test_ensembles.py)
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
# PREDICTION EXTRACTION
# ============================================================================

def predict_proba_with_tta(model, images, device, n_tta=5):
    """Get probability predictions with TTA"""
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


def extract_predictions(models_dict, data_loader, device, use_tta=True, n_tta=5):
    """
    Extract probability predictions from all models.
    Returns: DataFrame with columns [filename, label, model1_class0, model1_class1, ..., modelN_classK]
    """
    all_filenames = []
    all_labels = []
    model_predictions = {name: [] for name in models_dict.keys()}
    
    is_test = isinstance(data_loader.dataset, BloodDataset) and data_loader.dataset.is_test
    
    for batch_data in tqdm(data_loader, desc="Extracting predictions"):
        if is_test:
            images, filenames = batch_data
            labels = None
        else:
            images, labels = batch_data
            filenames = None
        
        images = images.to(device)
        
        # Get predictions from each model
        for model_name, model in models_dict.items():
            if use_tta:
                probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
            else:
                model.eval()
                with torch.no_grad():
                    probs = torch.softmax(model(images), dim=1)
            
            model_predictions[model_name].append(probs.cpu().numpy())
        
        if is_test:
            all_filenames.extend(filenames)
        else:
            all_labels.extend(labels.numpy().tolist())
    
    # Concatenate predictions
    for model_name in model_predictions.keys():
        model_predictions[model_name] = np.vstack(model_predictions[model_name])
    
    # Create DataFrame
    data = {}
    
    if is_test:
        data['filename'] = all_filenames
    else:
        data['label'] = all_labels
    
    # Add model predictions as features
    num_classes = list(model_predictions.values())[0].shape[1]
    for model_name, preds in model_predictions.items():
        for class_idx in range(num_classes):
            col_name = f"{model_name}_class{class_idx}"
            data[col_name] = preds[:, class_idx]
    
    return pd.DataFrame(data)


# ============================================================================
# MAIN AUTOGLUON STACKING
# ============================================================================

def main():
    # Configuration
    DATA_PATH = '../data'
    MODEL_DIR = 'models_run_2'
    OUTPUT_DIR = 'autogluon_ensemble'
    AUTOGLUON_DIR = os.path.join(OUTPUT_DIR, 'AutogluonModels')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(AUTOGLUON_DIR, exist_ok=True)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n{'='*80}")
    print("AutoGluon Meta-Stacking Ensemble for WBCBench 2026")
    print(f"{'='*80}")
    print(f"Device: {device}\n")
    
    # ========================================================================
    # STEP 1: Load Data
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
    
    # Create train/val split (80/20 for meta-learner)
    train_indices, val_indices = train_test_split(
        range(len(train_df_expanded)),
        test_size=0.2,  # 20% for validation
        stratify=train_df_expanded['label_id'],
        random_state=42
    )
    
    train_df_meta = train_df_expanded.iloc[train_indices].reset_index(drop=True)
    val_df_meta = train_df_expanded.iloc[val_indices].reset_index(drop=True)
    
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Meta-train size: {len(train_df_meta):,}")
    print(f"Meta-val size: {len(val_df_meta):,}")
    print(f"Test size: {len(test_df):,}\n")
    
    # Create datasets
    train_dataset = BloodDataset(train_df_meta, transform=get_val_transform())
    val_dataset = BloodDataset(val_df_meta, transform=get_val_transform())
    test_dataset = BloodDataset(test_df, transform=get_val_transform(), is_test=True)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # ========================================================================
    # STEP 2: Load Trained Models
    # ========================================================================
    print("="*80)
    print("STEP 2: Loading Trained Models")
    print("="*80)
    
    model_configs = [
        {'name': 'SwinTransformer_ce', 'class': SwinTransformer},
        {'name': 'HybridSwin_ce', 'class': HybridSwin},
        {'name': 'EfficientNet_ce', 'class': EfficientNet},
        {'name': 'SwinTransformer_focal', 'class': SwinTransformer},
        {'name': 'HybridSwin_focal', 'class': HybridSwin},
        {'name': 'EfficientNet_focal', 'class': EfficientNet},
        {'name': 'SwinTransformer_focal_weighted', 'class': SwinTransformer},
        {'name': 'HybridSwin_focal_weighted', 'class': HybridSwin},
        {'name': 'EfficientNet_focal_weighted', 'class': EfficientNet},
    ]
    
    loaded_models = {}
    
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
    
    # ========================================================================
    # STEP 3: Extract Meta-Features (Model Predictions)
    # ========================================================================
    print("="*80)
    print("STEP 3: Extracting Meta-Features from Models")
    print("="*80)
    
    use_tta = True
    n_tta = 5
    
    print(f"\nExtracting predictions with TTA={use_tta} (n_tta={n_tta})...")
    
    print("\n[1/3] Extracting train predictions...")
    train_meta_df = extract_predictions(loaded_models, train_loader, device, use_tta, n_tta)
    
    print("[2/3] Extracting validation predictions...")
    val_meta_df = extract_predictions(loaded_models, val_loader, device, use_tta, n_tta)
    
    print("[3/3] Extracting test predictions...")
    test_meta_df = extract_predictions(loaded_models, test_loader, device, use_tta, n_tta)
    
    # Save meta-features
    train_meta_path = os.path.join(OUTPUT_DIR, 'meta_features_train.csv')
    val_meta_path = os.path.join(OUTPUT_DIR, 'meta_features_val.csv')
    test_meta_path = os.path.join(OUTPUT_DIR, 'meta_features_test.csv')
    
    train_meta_df.to_csv(train_meta_path, index=False)
    val_meta_df.to_csv(val_meta_path, index=False)
    test_meta_df.to_csv(test_meta_path, index=False)
    
    print(f"\n✓ Meta-features saved:")
    print(f"  Train: {train_meta_path}")
    print(f"  Val:   {val_meta_path}")
    print(f"  Test:  {test_meta_path}")
    print(f"\nMeta-feature shape:")
    print(f"  Train: {train_meta_df.shape}")
    print(f"  Val:   {val_meta_df.shape}")
    print(f"  Test:  {test_meta_df.shape}")
    
    # ========================================================================
    # STEP 4: Train AutoGluon Meta-Learner
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 4: Training AutoGluon Meta-Learner")
    print("="*80)
    
    # Prepare data for AutoGluon
    train_data = TabularDataset(train_meta_path)
    val_data = TabularDataset(val_meta_path)
    
    # Train AutoGluon predictor
    predictor = TabularPredictor(
        label='label',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path=os.path.join(AUTOGLUON_DIR, 'wbc_metastack')
    )
    
    print("\nTraining AutoGluon with:")
    print(f"  - Preset: best_quality")
    print(f"  - Stack levels: 3")
    print(f"  - Bag holdout: True")
    print(f"  - Time limit: 3600 seconds (1 hour)")
    print("\nThis may take a while...")
    
    fit_model = predictor.fit(
        train_data,
        tuning_data=val_data,
        presets='best_quality',
        use_bag_holdout=True,
        num_stack_levels=3,
        time_limit=3600,  # 1 hour
        verbosity=2
    )
    
    # Save predictor
    predictor_path = os.path.join(OUTPUT_DIR, 'autogluon_predictor.ag')
    predictor.save(predictor_path)
    print(f"\n✓ Predictor saved: {predictor_path}")
    
    # ========================================================================
    # STEP 5: Evaluate Meta-Learner
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 5: Evaluating Meta-Learner")
    print("="*80)
    
    # Get best model info
    best_model = predictor.model_best
    print(f"\nBest model: {best_model}")
    
    # Evaluate on train set
    print("\n[1/2] Training set evaluation:")
    train_performance = predictor.evaluate(train_data)
    print(f"  F1 Macro: {train_performance['f1_macro']:.4f}")
    
    train_predictions = predictor.predict(train_data)
    train_f1_per_class = f1_score(
        np.array(train_data['label']), 
        np.array(train_predictions), 
        average=None
    )
    print(f"  Per-class F1: {train_f1_per_class}")
    
    # Evaluate on validation set
    print("\n[2/2] Validation set evaluation:")
    val_performance = predictor.evaluate(val_data)
    print(f"  F1 Macro: {val_performance['f1_macro']:.4f}")
    
    val_predictions = predictor.predict(val_data)
    val_f1_per_class = f1_score(
        np.array(val_data['label']), 
        np.array(val_predictions), 
        average=None
    )
    print(f"  Per-class F1: {val_f1_per_class}")
    
    # Detailed classification report
    print("\nDetailed Classification Report (Validation):")
    print("="*80)
    print(classification_report(
        np.array(val_data['label']), 
        np.array(val_predictions),
        target_names=[label2name[i] for i in range(num_classes)]
    ))
    
    # Model leaderboard
    print("\nModel Leaderboard:")
    print("="*80)
    leaderboard = predictor.leaderboard(val_data, silent=True)
    print(leaderboard.to_string())
    
    # ========================================================================
    # STEP 6: Generate Final Predictions
    # ========================================================================
    print("\n" + "="*80)
    print("STEP 6: Generating Final Test Predictions")
    print("="*80)
    
    test_data = TabularDataset(test_meta_path)
    test_predictions = predictor.predict(test_data)
    
    # Map predictions to class names
    test_pred_labels = [label2name[int(pred)] for pred in test_predictions]
    
    # Create submission
    submission_df = pd.DataFrame({
        'ID': test_meta_df['filename'],
        'Target': test_pred_labels
    })
    
    submission_path = os.path.join(OUTPUT_DIR, 'submission_autogluon.csv')
    submission_df.to_csv(submission_path, index=False)
    
    print(f"\n✓ Submission saved: {submission_path}")
    print(f"  Total predictions: {len(submission_df):,}")
    print("\nSample predictions:")
    print(submission_df.head(10))
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    print("\n" + "="*80)
    print("AUTOGLUON META-STACKING SUMMARY")
    print("="*80)
    print(f"Number of base models: {len(loaded_models)}")
    print(f"Meta-features per sample: {train_meta_df.shape[1] - 1}")  # -1 for label
    print(f"Best meta-model: {best_model}")
    print(f"\nValidation Performance:")
    print(f"  F1 Macro: {val_performance['f1_macro']:.4f}")
    print(f"  Accuracy: {val_performance.get('accuracy', 'N/A')}")
    print(f"\nFiles Generated:")
    print(f"  • {submission_path}")
    print(f"  • {predictor_path}")
    print(f"  • {train_meta_path}")
    print(f"  • {val_meta_path}")
    print(f"  • {test_meta_path}")
    print("="*80)
    
    # Compare with simple averaging
    print("\n" + "="*80)
    print("COMPARISON: AutoGluon vs Simple Averaging")
    print("="*80)
    
    # Simple averaging on validation set
    val_meta_no_label = val_meta_df.drop('label', axis=1)
    num_models = len(loaded_models)
    
    # Average probabilities per class
    avg_probs = np.zeros((len(val_meta_no_label), num_classes))
    for class_idx in range(num_classes):
        class_cols = [col for col in val_meta_no_label.columns if f'_class{class_idx}' in col]
        avg_probs[:, class_idx] = val_meta_no_label[class_cols].mean(axis=1)
    
    avg_predictions = avg_probs.argmax(axis=1)
    avg_f1 = f1_score(np.array(val_data['label']), avg_predictions, average='macro')
    
    print(f"Simple Averaging F1: {avg_f1:.4f}")
    print(f"AutoGluon F1:        {val_performance['f1_macro']:.4f}")
    print(f"Improvement:         {(val_performance['f1_macro'] - avg_f1)*100:+.2f}%")
    print("="*80)
    
    print("\n✅ AutoGluon meta-stacking complete!")
    print(f"💡 Submit: {submission_path}")


if __name__ == "__main__":
    main()