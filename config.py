
# ============================================================================
# CONFIGURATION
# ============================================================================
# config.py
import torch
import os
import argparse
class Config:
    DATA_PATH = '../data'
    N_FOLDS = 5
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE =32
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # Output directories
    SAVE_DIR = 'models_kfold'
    OOF_DIR = 'oof_predictions'
    TEST_PRED_DIR = 'test_predictions'
    PLOT_DIR = 'plots_kfold'
    SUBMISSION_DIR = 'submissions_kfold'
    OUTPUT_DIR = 'ensemble_results_kfold'
# class Config:
#     PVC_ROOT = os.environ.get("PVC_ROOT", "/pvc")

#     DATA_PATH = os.path.join(PVC_ROOT, "data")

#     # Output directories (all on the PVC)
#     OUT_ROOT = os.path.join(PVC_ROOT, "outputs")
#     SAVE_DIR = os.path.join(OUT_ROOT, "models_kfold")
#     OOF_DIR = os.path.join(OUT_ROOT, "oof_predictions")
#     TEST_PRED_DIR = os.path.join(OUT_ROOT, "test_predictions")
#     PLOT_DIR = os.path.join(OUT_ROOT, "plots_kfold")
#     SUBMISSION_DIR = os.path.join(OUT_ROOT, "submissions_kfold")
#     OUTPUT_DIR = os.path.join(OUT_ROOT, "ensemble_results_kfold")
    # Model configs to train
    
    # ── New: SWA settings ──────────────────────────────────────────────────
    USE_SWA = True
    # SWA kicks in after this fraction of epochs (e.g. 0.75 = last 25%)
    SWA_START_RATIO = 0.75
    SWA_LR = 1e-5          # low lr for SWA averaging phase

    # ── New: Label smoothing for CE loss ───────────────────────────────────
    LABEL_SMOOTHING = 0.1  # 0.0 = off, 0.1 is standard

    # ── New: Mixup only in first half of training ──────────────────────────
    MIXUP_END_RATIO = 0.5  # disable mixup after this fraction of epochs
    
    CONFIGS = [
        # H-Optimus-1: pathology foundation model (pretrained on large-scale histopathology data by Bioptimus)
        # {'model': 'HOptimus1',       'loss': 'ce',             'lr': 2e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'HOptimus1',       'loss': 'focal',          'lr': 2e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'HOptimus1',       'loss': 'focal_weighted', 'lr': 2e-5, 'epochs': 30, 'weight_decay': 1e-4},
        
        {'model': 'SwinTransformer', 'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'SwinTransformer', 'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        # {'model': 'SwinTransformer', 'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},

                

        # {'model': 'EfficientNet',     'loss': 'ce',             'lr': 1e-4, 'epochs': 5, 'weight_decay': 1e-4},
    #     {'model': 'EfficientNet',     'loss': 'focal',          'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'EfficientNet',     'loss': 'focal_weighted', 'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
        
        
    #    {'model': 'HybridSwin',      'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'HybridSwin',      'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'HybridSwin',      'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        
        # {'model': 'ViT',             'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'ViT',             'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'ViT',             'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        
        
        # {'model': 'VitGiantDino',     'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'VitGiantDino',     'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    #     {'model': 'VitGiantDino',     'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    ]