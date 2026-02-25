
# dataloader.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image, ImageFile
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import cv2
from torchvision import transforms
import random
import torchstain
from config import Config
# ImageFile.LOAD_TRUNCATED_IMAGES = True

cfg = Config()
# ============================================================================
# DATA LOADING
# ============================================================================

# def load_data(data_path):
#     PHASE1_IMG_DIR = os.path.join(data_path, "phase1")
#     PHASE2_TRAIN_IMG_DIR = os.path.join(data_path, "phase2/train")
#     PHASE2_EVAL_IMG_DIR = os.path.join(data_path, "phase2/eval")
#     PHASE2_TEST_IMG_DIR = os.path.join(data_path, "phase2/test")

#     def clean_df(df):
#         df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")], errors="ignore")
#         df = df.rename(columns={"ID": "filename", "labels": "label"})
#         return df

#     phase1_df = clean_df(pd.read_csv(os.path.join(data_path, "phase1_label.csv")))
#     phase2_train_df = clean_df(pd.read_csv(os.path.join(data_path, "phase2_train.csv")))
#     phase2_eval_df = clean_df(pd.read_csv(os.path.join(data_path, "phase2_eval.csv")))
#     phase2_test_df = clean_df(pd.read_csv(os.path.join(data_path, "phase2_test.csv")))

#     phase1_df["img_dir"] = PHASE1_IMG_DIR
#     phase2_train_df["img_dir"] = PHASE2_TRAIN_IMG_DIR
#     phase2_eval_df["img_dir"] = PHASE2_EVAL_IMG_DIR
#     phase2_test_df["img_dir"] = PHASE2_TEST_IMG_DIR

#     # Combine all labeled data
#     train_df = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
#     test_df = phase2_test_df.copy()

#     # Class mapping
#     class_names = sorted(train_df["label"].unique())
#     num_classes = len(class_names)
#     label2name = dict(zip(range(num_classes), class_names))
#     name2label = {v: k for k, v in label2name.items()}

#     train_df["label_id"] = train_df["label"].map(name2label)
#     test_df["label_id"] = -1

#     print(f"\n{'='*70}")
#     print(f"DATA SUMMARY")
#     print(f"{'='*70}")
#     print(f"Total training samples: {len(train_df):,}")
#     print(f"Test samples:           {len(test_df):,}")
#     print(f"Classes ({num_classes}): {class_names}")
#     print(f"{'='*70}\n")

#     return train_df, test_df, class_names, num_classes, label2name, name2label
def load_data(data_path, extra_data_path=None):
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

    train_df = pd.concat([phase1_df, phase2_train_df, phase2_eval_df], ignore_index=True)
    test_df = phase2_test_df.copy()

    # ── Dataset 2 integration ─────────────────────────────────────────────
    # Maps Dataset 2 folder names → Dataset 1 label codes
    DATASET2_LABEL_MAP = {
        'basophil':   'BA',
        'eosinophil': 'EO',
        'lymphocyte': 'LY',
        'monocyte':   'MO',
    }
    # Folders we explicitly skip (no compatible class in Dataset 1)
    DATASET2_SKIP = {'erythroblast', 'ig', 'neutrophil', 'platelet'}

    if extra_data_path is not None and os.path.isdir(extra_data_path):
        extra_rows = []
        for folder_name in sorted(os.listdir(extra_data_path)):
            folder_path = os.path.join(extra_data_path, folder_name)
            if not os.path.isdir(folder_path):
                continue
            if folder_name in DATASET2_SKIP:
                print(f"  ⏭  Skipping Dataset 2 class '{folder_name}' (no compatible mapping)")
                continue
            if folder_name not in DATASET2_LABEL_MAP:
                print(f"  ⚠️  Unknown Dataset 2 class '{folder_name}', skipping")
                continue

            target_label = DATASET2_LABEL_MAP[folder_name]
            images = [f for f in os.listdir(folder_path)
                      if f.lower().endswith((".png", ".jpg", ".jpeg"))]

            for fname in images:
                extra_rows.append({
                    "filename": fname,
                    "label":    target_label,
                    "img_dir":  folder_path,
                })

            print(f"  ✓  '{folder_name}' → '{target_label}': {len(images):,} images added")

        if extra_rows:
            extra_df = pd.DataFrame(extra_rows)
            train_df = pd.concat([train_df, extra_df], ignore_index=True)
            print(f"\n  Dataset 2 total added: {len(extra_rows):,} images\n")

    # ── Class mapping (built after all data is merged) ────────────────────
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
    # Show per-class counts so you can see the balance effect
    counts = train_df["label"].value_counts().sort_index()
    for cls, cnt in counts.items():
        print(f"  {cls:<6} {cnt:>6,}")
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
        # A.Lambda(image=lambda x, **k: cv2.bilateralFilter(x, 7, 50, 50), p=0.5),
        # A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=0.7),
        # A.Lambda(image=lambda x, **k: morphology_on_lab_l(x), p=0.15),
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
        # A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=1.0),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])





# H-Optimus-1 uses its own normalization constants
def get_train_transform_hoptimus():
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
        A.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ToTensorV2(),
    ])


def get_val_transform_hoptimus():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
        A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=1.0),
        A.Normalize(mean=(0.707223, 0.578729, 0.703617), std=(0.211883, 0.230117, 0.177517)),
        ToTensorV2(),
    ])




# T = transforms.Compose([
#     transforms.ToTensor(),
#     transforms.Lambda(lambda x: x * 255)
# ])

# normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")

# phase1_imgs = sorted(os.listdir("../data/phase1"))
# sample_imgs = random.sample(phase1_imgs, 30)

# for img_name in sample_imgs:
#     img = cv2.cvtColor(
#         cv2.imread(f"../data/phase1/{img_name}"),
#         cv2.COLOR_BGR2RGB
#     )
#     normalizer.fit(T(img))
    

# def macenko_normalize_tensor(img_tensor):
#     if not torch.is_tensor(img_tensor):
#         img_tensor = torch.as_tensor(img_tensor)

#     # After ToTensorV2 it's already (C,H,W)
#     if img_tensor.ndim == 3 and img_tensor.shape[0] != 3:
#         img_tensor = img_tensor.permute(2, 0, 1)

#     device = img_tensor.device
#     img_tensor = img_tensor.float()

#     mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3, 1, 1)
#     std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3, 1, 1)

#     # Undo ImageNet norm → [0,255]
#     img = (img_tensor * std + mean) * 255.0

#     try:
#         img_norm, _, _ = normalizer.normalize(I=img)
#     except Exception:
#         # Macenko can fail on edge-case images (very dark, almost uniform)
#         return img_tensor  # fall back to unnormalized

#     # ← Fix: normalizer returns (H,W,C), permute to (C,H,W)
#     if img_norm.ndim == 3 and img_norm.shape[0] != 3:
#         img_norm = img_norm.permute(2, 0, 1)

#     img_norm = (img_norm.float() / 255.0 - mean) / std
#     return img_norm

# def get_train_transform():
#     return A.Compose([
#         A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),

#         # Geometry (safe + effective)
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.2),
#         A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.4),
#         A.RandomResizedCrop(
#             size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
#             scale=(0.85, 1.0),
#             ratio=(0.95, 1.05),
#             p=0.3
#         ),

#         # Mild blur / noise only
#         A.OneOf([
#             A.GaussNoise(var_limit=(3.0, 12.0)),
#             A.GaussianBlur(blur_limit=3),
#         ], p=0.2),

#         # Standard normalization
#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)
#         ),
#         ToTensorV2(),

#         # 🔥 Stain normalization / augmentation
#         A.Lambda(
#             image=lambda x, **k: macenko_normalize_tensor(x),
#             p=0.7
#         ),
#     ])
    

# def get_val_transform():
#     return A.Compose([
#         A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),

#         A.Normalize(
#             mean=(0.485, 0.456, 0.406),
#             std=(0.229, 0.224, 0.225)
#         ),
#         ToTensorV2(),

#         # Always normalize stains in validation
#         A.Lambda(
#             image=lambda x, **k: macenko_normalize_tensor(x),
#             p=1.0
#         ),
#     ])

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
        image = np.array(Image.open(img_path).convert("RGB")) if isinstance(self.transform, A.Compose) else  Image.open(img_path).convert("RGB")

        if self.transform:
            # image = self.transform(image=image)["image"]
            image = self.transform(image=image)["image"] if isinstance(self.transform, A.Compose) else self.transform(image)

        if self.is_test:
            return image, row["filename"]

        label = torch.tensor(row["label_id"], dtype=torch.long)
        return image, label
    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     img_path = os.path.join(row["img_dir"], row["filename"])

    #     image = np.array(Image.open(img_path).convert("RGB"))

    #     if self.transform is not None:
    #         if isinstance(self.transform, A.Compose):
    #             image = self.transform(image=image)["image"]
    #         else:
    #             image = self.transform(image)

    #     if self.is_test:
    #         return image, row["filename"]

    #     label = torch.tensor(row["label_id"], dtype=torch.long)
    #     return image, label
