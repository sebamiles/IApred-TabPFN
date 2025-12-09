import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import pandas as pd
import numpy as np
from joblib import load
import logging
from functions_for_training import sequences_to_vectors
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix, matthews_corrcoef, brier_score_loss
)
from sklearn.calibration import calibration_curve
from sklearn.feature_selection import VarianceThreshold
import torch
import argparse

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_model_suffix(k):
    """Get the suffix for model files based on k value"""
    if k is None or k == 'all':
        return 'all_features'
    else:
        return f'{k}_features'

def calculate_ece(y_true, y_scores, n_bins=10):
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_scores, bin_boundaries) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    ece = 0
    valid_bins = 0
    for bin_idx in range(n_bins):
        mask = bin_indices == bin_idx
        if np.any(mask):
            bin_conf = np.mean(y_scores[mask])
            bin_acc = np.mean(y_true[mask])
            ece += np.abs(bin_acc - bin_conf) * np.sum(mask)
            valid_bins += np.sum(mask)
    return ece / valid_bins if valid_bins > 0 else 0

def update_csv_with_predictions(csv_path, model, variance_selector, feature_selector, all_feature_names, sequence_column):
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    csv_path_full = os.path.join(project_root, csv_path)

    try:
        df = pd.read_csv(csv_path_full)
    except:
        try:
            df = pd.read_excel(csv_path_full, engine='odf')
        except:
            logging.error(f"Could not read {csv_path_full}")
            return None

    sequences = df[sequence_column].tolist()
    X_new, feature_names, failed_indices = sequences_to_vectors(sequences)

    # Remove failed sequences from dataframe and sequences list
    if len(failed_indices) > 0:
        failed_indices = failed_indices.astype(int)
        df = df.drop(df.index[failed_indices]).reset_index(drop=True)
        sequences = [seq for i, seq in enumerate(sequences) if i not in failed_indices]
        logging.info(f"Removed {len(failed_indices)} sequences that failed feature extraction")

    feature_map = {name: i for i, name in enumerate(feature_names)}
    X_new_aligned = np.zeros((X_new.shape[0], len(all_feature_names)))

    for i, feature in enumerate(all_feature_names):
        if feature in feature_map:
            X_new_aligned[:, i] = X_new[:, feature_map[feature]]

    X_new_aligned = np.nan_to_num(X_new_aligned, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    X_new_aligned = np.clip(X_new_aligned, -1e6, 1e6)

    # Apply variance selector
    if variance_selector is not None:
        X_new_filtered = variance_selector.transform(X_new_aligned)
    else:
        X_new_filtered = X_new_aligned

    # Apply feature selector if it exists
    if feature_selector is not None:
        X_new_final = feature_selector.transform(X_new_filtered)
    else:
        X_new_final = X_new_filtered

    # TabPFN: class 0 = 'antigen', class 1 = 'non-antigen'
    # We want antigens to have higher scores, so use class 0 probability
    probabilities = model.predict_proba(X_new_final)[:, 0]  # Probability of antigen class
    probabilities = 1 - probabilities  # Invert scores so higher = more antigenic
    df['IApred'] = probabilities

    return df

def create_calibration_plot(antigens_df, non_antigens_df, output_path, model_name):
    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    y_scores = np.concatenate([antigens_df['IApred'].values, non_antigens_df['IApred'].values])

    plt.figure(figsize=(7, 6))
    prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
    plt.plot(prob_pred, prob_true, marker='o', color='blue', label=f'IApred-{model_name}')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Mean Predicted Probability', fontsize=18, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=18, fontweight='bold')
    plt.title(f'Calibration Curve (TabPFN - {model_name})', fontsize=24, fontweight='bold')
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_plot(antigens_df, non_antigens_df, output_path, model_name):
    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    y_scores = np.concatenate([antigens_df['IApred'].values, non_antigens_df['IApred'].values])

    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(9, 8))
    plt.plot(fpr, tpr, color='blue', label=f'IApred-{model_name} (AUC = {roc_auc:.3f})', lw=2)
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.title(f'ROC Curve (TabPFN - {model_name})', fontsize=24, fontweight='bold')
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def calculate_metrics(antigens_df, non_antigens_df):
    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    y_scores = np.concatenate([antigens_df['IApred'].values, non_antigens_df['IApred'].values])
    y_pred = (y_scores >= 0.5).astype(int)  # TabPFN threshold is 0.5

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall) if (0.5**2 * precision + recall) > 0 else 0
    j_index = sensitivity + specificity - 1
    mcc = matthews_corrcoef(y_true, y_pred)
    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr_curve, tpr_curve)
    brier = brier_score_loss(y_true, y_scores)
    ece = calculate_ece(y_true, y_scores)

    return {
        'True Positive': tp, 'True Negative': tn, 'False Positive': fp, 'False Negative': fn,
        'Accuracy': accuracy_score(y_true, y_pred), 'Precision': precision, 'Sensitivity': sensitivity,
        'Specificity': specificity, 'MCC': mcc, "Youden's J": j_index, 'F1-score': f1,
        'F0.5-score': f05, 'Brier Score': brier, 'ECE': ece, 'AUC-ROC': roc_auc
    }

def main():
    parser = argparse.ArgumentParser(description='External evaluation of TabPFN model')
    parser.add_argument('--k', type=str, default='all',
                       help='Number of features used in model (default: all, options: all, 529, 100)')
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    model_suffix = get_model_suffix(args.k)
    models_dir = f'../models/{model_suffix}'
    results_dir = f'../results/{model_suffix}'
    ensure_dir_exists(results_dir)

    try:
        model = load(os.path.join(models_dir, 'IApred_TabPFN.joblib'))
        variance_selector = load(os.path.join(models_dir, 'IApred_variance_selector.joblib'))
        all_feature_names = load(os.path.join(models_dir, 'IApred_all_feature_names.joblib'))

        # Try to load feature selector (may not exist for all_features model)
        feature_selector = None
        selector_path = os.path.join(models_dir, 'IApred_feature_selector.joblib')
        if os.path.exists(selector_path):
            feature_selector = load(selector_path)
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return

    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    antigens_csv = os.path.join(project_root, 'data/External_evaluation_antigens.csv')
    non_antigens_csv = os.path.join(project_root, 'data/External_evaluation_non-antigens.csv')

    antigens_df = update_csv_with_predictions(antigens_csv, model, variance_selector,
                                               feature_selector, all_feature_names, 'Antigen')
    non_antigens_df = update_csv_with_predictions(non_antigens_csv, model, variance_selector,
                                                   feature_selector, all_feature_names, 'Non-Antigen')

    if antigens_df is None or non_antigens_df is None:
        logging.error("Could not load external evaluation data")
        return

    # Save predictions
    antigens_df.to_csv(os.path.join(results_dir, 'External_evaluation_antigens.csv'), index=False)
    non_antigens_df.to_csv(os.path.join(results_dir, 'External_evaluation_non-antigens.csv'), index=False)

    # Calculate metrics
    metrics = calculate_metrics(antigens_df, non_antigens_df)

    # Create plots
    model_name = args.k if args.k != 'all' else 'all'
    create_roc_plot(antigens_df, non_antigens_df, os.path.join(results_dir, 'roc_curve.png'), model_name)
    create_calibration_plot(antigens_df, non_antigens_df, os.path.join(results_dir, 'calibration_curve.png'), model_name)

    # Save metrics
    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(os.path.join(results_dir, 'performance_metrics.csv'), index=False)

    print(f"\nExternal Evaluation Results (TabPFN - {model_suffix}):")
    print("=" * 50)
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    print(f"\nResults saved in {results_dir}/")

if __name__ == "__main__":
    main()
