#models.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm

from config import Config
cfg = Config()
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


## Add a Vit large model
class ViT(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_large_patch16_224", pretrained=pretrained, num_classes=0, in_chans=3)
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.backbone(x))
    
class HOptimus1(nn.Module):
    """
    H-Optimus-1: State-of-the-art pathology foundation model (ViT-Large/16).
    Pretrained on large-scale histopathology data by Bioptimus.
    Loaded via timm from HuggingFace Hub.
    """
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "hf-hub:bioptimus/H-optimus-0",
            pretrained=pretrained,
            num_classes=0,       # remove default head, use as feature extractor
            init_values=1e-5,
            dynamic_img_size=False,
        )
        # ViT-L/16 outputs 1024-dim features
        embed_dim = self.backbone.num_features  # 1024
        self.classifier = ClassificationHead(embed_dim, num_classes, dropout)

    def forward(self, x):
        features = self.backbone(x)
        return self.classifier(features)
    

class VitGiantDino(nn.Module):
    def __init__(self, num_classes, dropout=0.4, pretrained=True):
        super().__init__()
        self.backbone = timm.create_model(
            "vit_giant_patch14_dinov2.lvd142m", pretrained=pretrained, num_classes=0, in_chans=3,
              dynamic_img_size=True,
            )
        self.classifier = ClassificationHead(self.backbone.num_features, num_classes, dropout)

    def forward(self, x):
        return self.classifier(self.backbone(x))


MODEL_REGISTRY = {
    'SwinTransformer': SwinTransformer,
    'HybridSwin': HybridSwin,
    'EfficientNet': EfficientNet,
    'HOptimus1': HOptimus1,
    'ViT': ViT,
    'VitGiantDino': VitGiantDino,
}

# Models that require their own normalization / transforms
HOPTIMUS_MODELS = {'HOptimus1'}