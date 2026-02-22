
# dataloader.py
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
import cv2
from torchvision import transforms
import random
import torchstain

from config import Config


cfg = Config()
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


# def get_train_transform():
#     return A.Compose([
#         A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
#         A.Lambda(image=lambda x, **k: cv2.bilateralFilter(x, 7, 50, 50), p=0.5),
#         A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=0.7),
#         A.Lambda(image=lambda x, **k: morphology_on_lab_l(x), p=0.15),
#         A.HorizontalFlip(p=0.5),
#         A.VerticalFlip(p=0.2),
#         A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.4),
#         A.RandomResizedCrop(size=(cfg.IMG_SIZE, cfg.IMG_SIZE), scale=(0.85, 1.0), ratio=(0.95, 1.05), p=0.3),
#         A.OneOf([
#             A.ElasticTransform(alpha=10, sigma=4, alpha_affine=3, border_mode=cv2.BORDER_REFLECT, p=1.0),
#             A.GridDistortion(num_steps=5, distort_limit=0.07, border_mode=cv2.BORDER_REFLECT, p=1.0),
#             A.OpticalDistortion(distort_limit=0.07, shift_limit=0.07, border_mode=cv2.BORDER_REFLECT, p=1.0),
#         ], p=0.15),
#         A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.15, hue=0.08, p=0.4),
#         A.HueSaturationValue(hue_shift_limit=8, sat_shift_limit=15, val_shift_limit=10, p=0.25),
#         A.OneOf([
#             A.GaussNoise(var_limit=(3.0, 12.0), p=1.0),
#             A.GaussianBlur(blur_limit=(3, 3), p=1.0),
#             A.MedianBlur(blur_limit=3, p=1.0),
#         ], p=0.25),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])


# def get_val_transform():
#     return A.Compose([
#         A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),
#         A.Lambda(image=lambda x, **k: advanced_clahe_preprocessing(x), p=1.0),
#         A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
#         ToTensorV2(),
#     ])


def train_mini_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(30),
        transforms.RandomAffine(degrees=0, shear=15, scale=(0.8, 1.2)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
def val_mini_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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




T = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

normalizer = torchstain.normalizers.MacenkoNormalizer(backend="torch")

phase1_imgs = sorted(os.listdir("../data/phase1"))
sample_imgs = random.sample(phase1_imgs, 30)

for img_name in sample_imgs:
    img = cv2.cvtColor(
        cv2.imread(f"../data/phase1/{img_name}"),
        cv2.COLOR_BGR2RGB
    )
    normalizer.fit(T(img))
    

def macenko_normalize_tensor(img_tensor):
    """
    img_tensor: torch.Tensor (C,H,W), normalized (ImageNet)
    """
    device = img_tensor.device

    # Undo ImageNet normalization
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
    std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
    img = img_tensor * std + mean

    # Scale to [0,255]
    img = img * 255.0

    # Stain normalization
    img_norm, _, _ = normalizer.normalize(I=img)

    # Back to [0,1]
    img_norm = img_norm / 255.0

    # Re-apply ImageNet normalization
    img_norm = (img_norm - mean) / std

    return img_norm

def get_train_transform():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),

        # Geometry (safe + effective)
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.Rotate(limit=30, border_mode=cv2.BORDER_REFLECT, p=0.4),
        A.RandomResizedCrop(
            size=(cfg.IMG_SIZE, cfg.IMG_SIZE),
            scale=(0.85, 1.0),
            ratio=(0.95, 1.05),
            p=0.3
        ),

        # Mild blur / noise only
        A.OneOf([
            A.GaussNoise(var_limit=(3.0, 12.0)),
            A.GaussianBlur(blur_limit=3),
        ], p=0.2),

        # Standard normalization
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),

        # 🔥 Stain normalization / augmentation
        A.Lambda(
            image=lambda x, **k: macenko_normalize_tensor(x),
            p=0.7
        ),
    ])
    

def get_val_transform():
    return A.Compose([
        A.Resize(cfg.IMG_SIZE, cfg.IMG_SIZE),

        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225)
        ),
        ToTensorV2(),

        # Always normalize stains in validation
        A.Lambda(
            image=lambda x, **k: macenko_normalize_tensor(x),
            p=1.0
        ),
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

    # def __getitem__(self, idx):
    #     row = self.df.iloc[idx]
    #     img_path = os.path.join(row["img_dir"], row["filename"])
    #     image = np.array(Image.open(img_path).convert("RGB")) if isinstance(self.transform, A.Compose) else  Image.open(img_path).convert("RGB")

    #     if self.transform:
    #         # image = self.transform(image=image)["image"]
    #         image = self.transform(image)["image"] if isinstance(self.transform, A.Compose) else self.transform(image)

    #     if self.is_test:
    #         return image, row["filename"]

    #     label = torch.tensor(row["label_id"], dtype=torch.long)
    #     return image, label
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(row["img_dir"], row["filename"])

        image = np.array(Image.open(img_path).convert("RGB"))

        if self.transform is not None:
            if isinstance(self.transform, A.Compose):
                image = self.transform(image=image)["image"]
            else:
                image = self.transform(image)

        if self.is_test:
            return image, row["filename"]

        label = torch.tensor(row["label_id"], dtype=torch.long)
        return image, label
