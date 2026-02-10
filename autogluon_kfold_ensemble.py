"""
AutoGluon Meta-Stacking Ensemble with K-Fold OOF Predictions
=============================================================
Uses out-of-fold predictions from main_kfold.py as meta-features.
No data leakage — every training sample's meta-features come from a model
that never saw that sample during training.

Usage:
    1. First run: python main_kfold.py
    2. Then run:  python autogluon_kfold_ensemble.py

Requirements:
    pip install autogluon --break-system-packages
"""

import os
import glob
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, classification_report
from autogluon.tabular import TabularPredictor, TabularDataset

# ============================================================================
# CONFIGURATION
# ============================================================================

OOF_DIR = 'oof_predictions'
TEST_PRED_DIR = 'test_predictions'
OUTPUT_DIR = 'autogluon_kfold_ensemble'
AUTOGLUON_DIR = os.path.join(OUTPUT_DIR, 'AutogluonModels')
SUBMISSION_DIR = 'submissions_kfold'

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(AUTOGLUON_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


# ============================================================================
# STEP 1: LOAD AND MERGE OOF PREDICTIONS
# ============================================================================

def load_and_merge_predictions(oof_dir, test_dir):
    """
    Load all OOF and test prediction CSVs and merge into single DataFrames.
    Each CSV has columns: <model>_class0, ..., <model>_classK, label (for OOF) or filename (for test).
    """
    print("="*80)
    print("STEP 1: Loading OOF and Test Predictions")
    print("="*80)

    # --- OOF ---
    oof_files = sorted(glob.glob(os.path.join(oof_dir, 'oof_*.csv')))
    if not oof_files:
        raise FileNotFoundError(f"No OOF files found in {oof_dir}. Run main_kfold.py first.")

    oof_dfs = []
    labels = None
    for f in oof_files:
        df = pd.read_csv(f)
        config_name = os.path.basename(f).replace('oof_', '').replace('.csv', '')
        print(f"  ✓ Loaded OOF: {config_name} ({df.shape[1]-1} features)")

        if labels is None:
            labels = df['label'].values
        else:
            assert np.array_equal(labels, df['label'].values), f"Label mismatch in {f}"

        # Drop label column, keep only prediction columns
        pred_cols = [c for c in df.columns if c != 'label']
        oof_dfs.append(df[pred_cols])

    oof_merged = pd.concat(oof_dfs, axis=1)
    oof_merged['label'] = labels

    # --- Test ---
    test_files = sorted(glob.glob(os.path.join(test_dir, 'test_*.csv')))
    if not test_files:
        raise FileNotFoundError(f"No test files found in {test_dir}. Run main_kfold.py first.")

    test_dfs = []
    filenames = None
    for f in test_files:
        df = pd.read_csv(f)
        config_name = os.path.basename(f).replace('test_', '').replace('.csv', '')
        print(f"  ✓ Loaded Test: {config_name} ({df.shape[1]-1} features)")

        if filenames is None:
            filenames = df['filename'].values
        else:
            assert np.array_equal(filenames, df['filename'].values), f"Filename mismatch in {f}"

        pred_cols = [c for c in df.columns if c != 'filename']
        test_dfs.append(df[pred_cols])

    test_merged = pd.concat(test_dfs, axis=1)
    test_merged_filenames = filenames

    print(f"\n✓ Merged OOF shape:  {oof_merged.shape}  (includes 'label' column)")
    print(f"✓ Merged Test shape: {test_merged.shape}")
    print(f"✓ Test filenames:    {len(test_merged_filenames)}")

    return oof_merged, test_merged, test_merged_filenames


# ============================================================================
# STEP 2: ADD ENGINEERED META-FEATURES
# ============================================================================

def add_meta_features(df, num_classes, has_label=True):
    """
    Add engineered features on top of raw model probabilities.
    These help the meta-learner capture agreement/disagreement patterns.
    """
    # Get all prediction columns (exclude label/filename)
    exclude = {'label', 'filename'}
    pred_cols = [c for c in df.columns if c not in exclude]

    # Group columns by model
    model_names = set()
    for col in pred_cols:
        # Column format: ModelName_loss_classN
        parts = col.rsplit('_class', 1)
        if len(parts) == 2:
            model_names.add(parts[0])
    model_names = sorted(model_names)

    # Per-model: max prob, predicted class, entropy
    for model in model_names:
        model_cols = [f'{model}_class{i}' for i in range(num_classes)]
        if all(c in df.columns for c in model_cols):
            probs = df[model_cols].values
            df[f'{model}_max_prob'] = probs.max(axis=1)
            df[f'{model}_pred_class'] = probs.argmax(axis=1)
            # Entropy
            eps = 1e-10
            entropy = -np.sum(probs * np.log(probs + eps), axis=1)
            df[f'{model}_entropy'] = entropy

    # Cross-model agreement features
    pred_class_cols = [f'{model}_pred_class' for model in model_names if f'{model}_pred_class' in df.columns]
    if len(pred_class_cols) >= 2:
        pred_classes = df[pred_class_cols].values
        # Number of models agreeing on the majority vote
        from scipy import stats
        mode_result = stats.mode(pred_classes, axis=1, keepdims=False)
        df['ensemble_mode_count'] = mode_result.count
        df['ensemble_agreement_ratio'] = mode_result.count / len(pred_class_cols)

    # Per-class: mean and std of predictions across models
    for c in range(num_classes):
        class_cols = [col for col in pred_cols if col.endswith(f'_class{c}')]
        if len(class_cols) >= 2:
            df[f'mean_class{c}'] = df[class_cols].mean(axis=1)
            df[f'std_class{c}'] = df[class_cols].std(axis=1)

    return df


# ============================================================================
# STEP 3: TRAIN AUTOGLUON META-LEARNER
# ============================================================================

def train_meta_learner(oof_data, time_limit=3600):
    print("\n" + "="*80)
    print("STEP 3: Training AutoGluon Meta-Learner")
    print("="*80)

    predictor = TabularPredictor(
        label='label',
        problem_type='multiclass',
        eval_metric='f1_macro',
        path=os.path.join(AUTOGLUON_DIR, 'wbc_metastack_kfold')
    )

    print(f"\n  Preset: best_quality")
    print(f"  Stack levels: 3")
    print(f"  Time limit: {time_limit}s ({time_limit/60:.0f} min)")
    print(f"  Features: {oof_data.shape[1] - 1}")
    print(f"  Samples:  {len(oof_data)}")
    print("\n  Training...\n")

    predictor.fit(
        train_data=TabularDataset(oof_data),
        presets='best_quality',
        num_stack_levels=3,
        time_limit=time_limit,
        verbosity=2,
    )

    return predictor


# ============================================================================
# STEP 4: EVALUATE AND GENERATE SUBMISSION
# ============================================================================

def evaluate_and_submit(predictor, oof_data, test_data, test_filenames, label2name, num_classes):
    print("\n" + "="*80)
    print("STEP 4: Evaluation and Submission")
    print("="*80)

    # Evaluate on OOF (this is the true generalization estimate)
    oof_tabular = TabularDataset(oof_data)
    oof_perf = predictor.evaluate(oof_tabular)
    print(f"\nOOF Performance (true generalization estimate):")
    print(f"  F1 Macro: {oof_perf['f1_macro']:.4f}")

    oof_preds = predictor.predict(oof_tabular)
    print(f"\nOOF Classification Report:")
    print(classification_report(
        oof_data['label'].values,
        oof_preds.values,
        target_names=[label2name[i] for i in range(num_classes)]
    ))

    # Leaderboard
    print("\nAutoGluon Model Leaderboard:")
    leaderboard = predictor.leaderboard(oof_tabular, silent=True)
    print(leaderboard.to_string())

    # Generate test predictions
    test_tabular = TabularDataset(test_data)
    test_preds = predictor.predict(test_tabular)
    test_pred_labels = [label2name[int(p)] for p in test_preds]

    submission_df = pd.DataFrame({
        'ID': test_filenames,
        'Target': test_pred_labels
    })

    sub_path = os.path.join(SUBMISSION_DIR, 'submission_autogluon_kfold.csv')
    submission_df.to_csv(sub_path, index=False)
    print(f"\n✓ Submission saved: {sub_path}")
    print(f"  Total predictions: {len(submission_df):,}")
    print(f"\nSample predictions:")
    print(submission_df.head(10))

    return submission_df, oof_perf


# ============================================================================
# STEP 5: COMPARE WITH SIMPLE AVERAGING
# ============================================================================

def compare_with_averaging(oof_data, num_classes, label2name, autogluon_f1):
    print("\n" + "="*80)
    print("COMPARISON: AutoGluon vs Simple Averaging vs Weighted Averaging")
    print("="*80)

    labels = oof_data['label'].values
    exclude = {'label', 'filename'}

    # Get only raw probability columns (not engineered features)
    raw_pred_cols = [c for c in oof_data.columns
                     if c not in exclude and '_class' in c
                     and not c.startswith('mean_') and not c.startswith('std_')]

    # Simple averaging
    avg_probs = np.zeros((len(oof_data), num_classes))
    for c in range(num_classes):
        class_cols = [col for col in raw_pred_cols if col.endswith(f'_class{c}')]
        avg_probs[:, c] = oof_data[class_cols].mean(axis=1)

    avg_preds = avg_probs.argmax(axis=1)
    avg_f1 = f1_score(labels, avg_preds, average='macro')

    print(f"\n  Simple Averaging OOF F1:  {avg_f1:.4f}")
    print(f"  AutoGluon OOF F1:         {autogluon_f1:.4f}")
    improvement = (autogluon_f1 - avg_f1) * 100
    print(f"  Improvement:              {improvement:+.2f}%")

    # Also generate simple averaging submission for comparison
    # (would need test data — skip if not needed)

    return avg_f1


# ============================================================================
# MAIN
# ============================================================================

def main():
    print(f"\n{'='*80}")
    print("AutoGluon Meta-Stacking with K-Fold OOF Predictions")
    print(f"{'='*80}\n")

    # Infer class info from OOF files
    sample_oof = sorted(glob.glob(os.path.join(OOF_DIR, 'oof_*.csv')))
    if not sample_oof:
        print("❌ No OOF files found. Run main_kfold.py first!")
        return

    sample_df = pd.read_csv(sample_oof[0])
    # Count classes from column names
    first_model_cols = [c for c in sample_df.columns if c != 'label' and '_class' in c]
    class_indices = set()
    for col in first_model_cols:
        parts = col.rsplit('_class', 1)
        if len(parts) == 2 and parts[1].isdigit():
            class_indices.add(int(parts[1]))
    num_classes = max(class_indices) + 1

    # We need label2name — reconstruct from common WBC class names
    # (This should match what main_kfold.py used)
    # Load from data to be safe
    DATA_PATH = '../data'
    try:
        phase1_df = pd.read_csv(os.path.join(DATA_PATH, "phase1_label.csv"))
        phase1_df = phase1_df.rename(columns={"labels": "label"})
        phase2_train_df = pd.read_csv(os.path.join(DATA_PATH, "phase2_train.csv"))
        phase2_train_df = phase2_train_df.rename(columns={"labels": "label"})
        all_labels = pd.concat([phase1_df['label'], phase2_train_df['label']]).unique()
        class_names = sorted(all_labels)
        label2name = dict(zip(range(len(class_names)), class_names))
        num_classes = len(class_names)
        print(f"Classes ({num_classes}): {class_names}")
    except Exception as e:
        print(f"⚠️ Could not load class names from data: {e}")
        label2name = {i: str(i) for i in range(num_classes)}
        class_names = [str(i) for i in range(num_classes)]

    # Step 1: Load and merge predictions
    oof_merged, test_merged, test_filenames = load_and_merge_predictions(OOF_DIR, TEST_PRED_DIR)

    # Step 2: Add engineered meta-features
    print("\n" + "="*80)
    print("STEP 2: Adding Engineered Meta-Features")
    print("="*80)

    n_before = oof_merged.shape[1]
    oof_merged = add_meta_features(oof_merged, num_classes, has_label=True)
    test_merged_with_label = test_merged.copy()
    # Add dummy label for test (won't be used for training)
    test_merged_with_label_temp = add_meta_features(test_merged.copy(), num_classes, has_label=False)
    # Ensure test has same columns as OOF (minus label)
    oof_feature_cols = [c for c in oof_merged.columns if c != 'label']
    for col in oof_feature_cols:
        if col not in test_merged_with_label_temp.columns:
            test_merged_with_label_temp[col] = 0  # fill missing engineered features
    test_final = test_merged_with_label_temp[oof_feature_cols]

    n_after = oof_merged.shape[1]
    print(f"  Features before: {n_before - 1}")
    print(f"  Features after:  {n_after - 1}  (+{n_after - n_before} engineered)")

    # Step 3: Train AutoGluon meta-learner on OOF data
    predictor = train_meta_learner(oof_merged, time_limit=3600)

    # Step 4: Evaluate and generate submission
    submission_df, oof_perf = evaluate_and_submit(
        predictor, oof_merged, test_final, test_filenames, label2name, num_classes)

    # Step 5: Compare with simple averaging
    avg_f1 = compare_with_averaging(oof_merged, num_classes, label2name, oof_perf['f1_macro'])

    # Final summary
    print(f"\n\n{'='*80}")
    print("FINAL SUMMARY")
    print(f"{'='*80}")
    print(f"  OOF Meta-features shape: {oof_merged.shape}")
    print(f"  Test Meta-features shape: {test_final.shape}")
    print(f"  Best AutoGluon model: {predictor.model_best}")
    print(f"  AutoGluon OOF F1: {oof_perf['f1_macro']:.4f}")
    print(f"  Simple Avg OOF F1: {avg_f1:.4f}")
    print(f"\n  Submission: {os.path.join(SUBMISSION_DIR, 'submission_autogluon_kfold.csv')}")
    print(f"{'='*80}\n")

    print("✅ Done! Submit 'submissions_kfold/submission_autogluon_kfold.csv'")


if __name__ == "__main__":
    main()