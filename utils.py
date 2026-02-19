import os
from typing import Optional
import pandas as pd
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import cv2


# ============================================================================
# HUGGING FACE AUTHENTICATION (for gated models like H-Optimus-0)
# ============================================================================

# use environment variable in .env file for huggingface to authenticate
# def setup_huggingface_auth():
#     from dotenv import load_dotenv
#     load_dotenv()
#     token = os.getenv("HUGGINGFACE_TOKEN")
#     if token is None:
#         print("\n⚠️  H-Optimus-0 requires Hugging Face authentication")
#         print("Please set HUGGINGFACE_TOKEN in your .env file or environment variables")
#         try:
#             from huggingface_hub import login, HfFolder
#             # Try to get existing token
#             token = HfFolder.get_token()
            
#             if token is None:
#                 print("\n⚠️  H-Optimus-0 requires Hugging Face authentication")
#                 print("Please run: huggingface-cli login")
#                 print("Or set HUGGINGFACE_TOKEN environment variable")
#                 return False
            
#             print("✓ Hugging Face authentication found")
#             return True
            
#         except ImportError:
#             print("⚠️  huggingface-hub not installed")
#             print("Install with: pip install huggingface-hub --break-system-packages")
#             return False
        

def setup_huggingface_auth() -> bool:
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    if not token:
        return False
    try:
        from huggingface_hub import HfFolder
        # token = login(token=token, add_to_git_credential=False)
        token = HfFolder.get_token()
        return True
    except Exception as e:
        print(f"HF login failed: {e}")
        return False
    
# ============================================================================
# LOSS FUNCTIONS
# ============================================================================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class FocalLossWithClassWeights(nn.Module):
    def __init__(self, alpha=None, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        return focal_loss


class PolyLoss(nn.Module):
    """
    PolyLoss - better than Focal Loss for medical imaging
    From: PolyLoss (ICLR 2022)
    """
    def __init__(self, num_classes: int, epsilon: float = 1.0, 
                 class_weights: Optional[torch.Tensor] = None):
        super().__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.class_weights = class_weights
        self.log_softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction='none', 
                            weight=self.class_weights)
        one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        pt = (one_hot * self.log_softmax(logits)).sum(dim=-1)
        poly_loss = ce + self.epsilon * (1.0 - pt)
        return poly_loss.mean()
    

def compute_custom_class_weights(num_classes, name2label, device):
    weights = torch.ones(num_classes, dtype=torch.float32) * 0.25
    custom = {'BNE': 0.75, 'MMY': 0.6, 'PC': 0.6, 'PMY': 0.75, 'VLY': 0.75}
    for cls_name, w in custom.items():
        if cls_name in name2label:
            weights[name2label[cls_name]] = w
    return weights.to(device)


def get_loss_fn(loss_name, class_weights):
    if loss_name == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_name == 'focal':
        return FocalLoss()
    elif loss_name == 'focal_weighted':
        return FocalLossWithClassWeights(alpha=class_weights)
    elif loss_name == 'poly':
        return PolyLoss(num_classes=len(class_weights), epsilon=1.0, class_weights=class_weights)
    else:
        raise ValueError(f"Unknown loss: {loss_name}")



# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_oof_confusion_matrix(oof_labels, oof_preds, class_names, config_key, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    cm = confusion_matrix(oof_labels, oof_preds)
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names,
                yticklabels=class_names, ax=ax1)
    ax1.set_title(f'{config_key} OOF — Counts')
    ax1.set_ylabel('True'); ax1.set_xlabel('Predicted')
    plt.setp(ax1.get_xticklabels(), rotation=45, ha='right')

    sns.heatmap(cm_pct, annot=True, fmt='.1f', cmap='RdYlGn', xticklabels=class_names,
                yticklabels=class_names, ax=ax2, vmin=0, vmax=100)
    ax2.set_title(f'{config_key} OOF — %')
    ax2.set_ylabel('True'); ax2.set_xlabel('Predicted')
    plt.setp(ax2.get_xticklabels(), rotation=45, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{config_key}_oof_cm.png'), dpi=200, bbox_inches='tight')
    plt.close()
    

# ============================================================================
# MIXUP
# ============================================================================

def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


# ============================================================================
# TTA
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


# ============================================================================
# TRAINING FUNCTIONS
# ============================================================================

def train_one_epoch(model, loader, criterion, optimizer, device, epoch, use_mixup=True):
    model.train()
    running_loss = 0.0
    all_preds, all_labels = [], []
    scaler = torch.cuda.amp.GradScaler(enabled=(device == "cuda"))
    pbar = tqdm(loader, desc=f"  Epoch {epoch} [TRAIN]", leave=False)
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

    return running_loss / len(loader), accuracy_score(all_labels, all_preds), f1_score(all_labels, all_preds, average='macro')




def validate_one_epoch(model, loader, criterion, device, epoch):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc=f"  Epoch {epoch} [VAL]", leave=False)
        for images, labels in pbar:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (running_loss / len(loader),
            accuracy_score(all_labels, all_preds),
            f1_score(all_labels, all_preds, average='macro'),
            all_preds, all_labels)


def extract_oof_probabilities(model, loader, device, n_tta=5):
    """Extract probability predictions for OOF samples using TTA."""
    model.eval()
    all_probs = []

    for images, _ in tqdm(loader, desc="  Extracting OOF probs", leave=False):
        images = images.to(device)
        probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
        all_probs.append(probs.cpu().numpy())

    return np.vstack(all_probs)


def extract_test_probabilities(model, loader, device, n_tta=5):
    """Extract probability predictions for test samples using TTA."""
    model.eval()
    all_probs = []
    all_filenames = []

    for images, filenames in tqdm(loader, desc="  Extracting test probs", leave=False):
        images = images.to(device)
        probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
        all_probs.append(probs.cpu().numpy())
        all_filenames.extend(filenames)

    return np.vstack(all_probs), all_filenames



# ============================================================================
# OOF-BASED ENSEMBLE EVALUATION (NO LEAKAGE)
# ============================================================================

def evaluate_ensemble_oof(oof_probs_dict, labels, num_classes, weights=None):
    """
    Evaluate an ensemble using precomputed OOF probabilities.
    No model inference needed — pure numpy, instant.

    Args:
        oof_probs_dict: {config_key: np.array of shape (N, num_classes)}
        labels: np.array of true labels (N,)
        weights: optional {config_key: weight} dict
    Returns:
        f1, acc, preds
    """
    model_names = list(oof_probs_dict.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    s = sum(weights[n] for n in model_names)
    weights = {n: weights[n] / s for n in model_names}

    ensemble_probs = np.zeros_like(list(oof_probs_dict.values())[0])
    for name in model_names:
        ensemble_probs += weights[name] * oof_probs_dict[name]

    preds = ensemble_probs.argmax(axis=1)
    f1 = f1_score(labels, preds, average='macro')
    acc = accuracy_score(labels, preds)

    return f1, acc, preds


def generate_submission_from_test_probs(test_probs_dict, test_filenames, label2name,
                                        output_path, weights=None):
    """
    Generate submission using precomputed averaged test probabilities.
    No model inference needed.
    """
    model_names = list(test_probs_dict.keys())

    if weights is None:
        weights = {name: 1.0 / len(model_names) for name in model_names}

    s = sum(weights[n] for n in model_names)
    weights = {n: weights[n] / s for n in model_names}

    ensemble_probs = np.zeros_like(list(test_probs_dict.values())[0])
    for name in model_names:
        ensemble_probs += weights[name] * test_probs_dict[name]

    final_preds = ensemble_probs.argmax(axis=1)
    pred_labels = [label2name[int(p)] for p in final_preds]

    submission_df = pd.DataFrame({"ID": test_filenames, "Target": pred_labels})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    return submission_df


# ============================================================================
# LIVE INFERENCE ENSEMBLE (for TTA comparison on test set)
# ============================================================================

def generate_submission_live(models_by_fold, test_loader, device, label2name,
                             output_path, weights=None, use_tta=True, n_tta=5):
    """
    Generate test submission by running inference with all fold checkpoints.
    For each config, averages predictions across all folds.
    Then ensembles across configs with given weights.
    """
    config_names = list(models_by_fold.keys())

    if weights is None:
        weights = {name: 1.0 / len(config_names) for name in config_names}
    s = sum(weights[n] for n in config_names)
    weights = {n: weights[n] / s for n in config_names}

    all_ids = []
    all_probs = None

    for images, ids in tqdm(test_loader, desc="Live inference", leave=False):
        images = images.to(device)
        all_ids.extend(list(ids))

        batch_ensemble = None

        for config_name in config_names:
            fold_models = models_by_fold[config_name]
            n_folds = len(fold_models)

            # Average across folds for this config
            fold_avg = None
            for model in fold_models:
                if use_tta:
                    probs = predict_proba_with_tta(model, images, device, n_tta=n_tta)
                else:
                    model.eval()
                    with torch.no_grad():
                        probs = torch.softmax(model(images), dim=1)

                fold_avg = probs if fold_avg is None else fold_avg + probs

            fold_avg = fold_avg / n_folds # type: ignore

            w = weights[config_name]
            batch_ensemble = w * fold_avg if batch_ensemble is None else batch_ensemble + w * fold_avg

        batch_ensemble = batch_ensemble.detach().cpu() # type: ignore
        all_probs = batch_ensemble if all_probs is None else torch.cat([all_probs, batch_ensemble], dim=0)

    final_preds = all_probs.argmax(dim=1).numpy() # type: ignore
    pred_labels = [label2name[int(p)] for p in final_preds]

    submission_df = pd.DataFrame({"ID": all_ids, "Target": pred_labels})
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    submission_df.to_csv(output_path, index=False)

    return submission_df



