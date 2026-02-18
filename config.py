
# ============================================================================
# CONFIGURATION
# ============================================================================
# config.py
import torch
import os
import argparse
# class Config:
#     DATA_PATH = '../data'
#     N_FOLDS = 5
#     SEED = 42
#     IMG_SIZE = 224
#     BATCH_SIZE = 8
#     NUM_WORKERS = 4
#     DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

#     # Output directories
#     SAVE_DIR = 'models_kfold'
#     OOF_DIR = 'oof_predictions'
#     TEST_PRED_DIR = 'test_predictions'
#     PLOT_DIR = 'plots_kfold'
#     SUBMISSION_DIR = 'submissions_kfold'
#     OUTPUT_DIR = 'ensemble_results_kfold'
class Config:
    PVC_ROOT = os.environ.get("PVC_ROOT", "/pvc")

    DATA_PATH = os.path.join(PVC_ROOT, "data")

    # Output directories (all on the PVC)
    OUT_ROOT = os.path.join(PVC_ROOT, "outputs")
    SAVE_DIR = os.path.join(OUT_ROOT, "models_kfold")
    OOF_DIR = os.path.join(OUT_ROOT, "oof_predictions")
    TEST_PRED_DIR = os.path.join(OUT_ROOT, "test_predictions")
    PLOT_DIR = os.path.join(OUT_ROOT, "plots_kfold")
    SUBMISSION_DIR = os.path.join(OUT_ROOT, "submissions_kfold")
    OUTPUT_DIR = os.path.join(OUT_ROOT, "ensemble_results_kfold")

    N_FOLDS = 5
    SEED = 42
    IMG_SIZE = 224
    BATCH_SIZE = 2
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    # Model configs to train
    CONFIGS = [
        # H-Optimus-1: pathology foundation model (pretrained on large-scale histopathology data by Bioptimus)
        {'model': 'HOptimus1',       'loss': 'ce',             'lr': 2e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'HOptimus1',       'loss': 'focal',          'lr': 2e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'HOptimus1',       'loss': 'focal_weighted', 'lr': 2e-5, 'epochs': 30, 'weight_decay': 1e-4},
        
        {'model': 'SwinTransformer', 'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'HybridSwin',      'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'EfficientNet',     'loss': 'ce',             'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'SwinTransformer', 'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'HybridSwin',      'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'EfficientNet',     'loss': 'focal',          'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'SwinTransformer', 'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'HybridSwin',      'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'EfficientNet',     'loss': 'focal_weighted', 'lr': 1e-4, 'epochs': 30, 'weight_decay': 1e-4},
        
        {'model': 'ViT',             'loss': 'ce',             'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'ViT',             'loss': 'focal',          'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
        {'model': 'ViT',             'loss': 'focal_weighted', 'lr': 5e-5, 'epochs': 30, 'weight_decay': 1e-4},
    ]