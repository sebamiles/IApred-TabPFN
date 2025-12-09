import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from tabpfn import TabPFNClassifier
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import logging
import argparse
import torch
from data_loader import load_and_extract_features

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def get_model_suffix(k):
    """Get the suffix for model files based on k value"""
    if k is None or k == 'all':
        return 'all_features'
    else:
        return f'{k}_features'

def perform_cross_validation(X, y, k=None, cv=10):
    model_suffix = get_model_suffix(k)

    # Setup logging
    results_dir = f'../results/{model_suffix}'
    ensure_dir_exists(results_dir)
    logging.basicConfig(filename=os.path.join(results_dir, 'cv.log'), level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    stratified_kfold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)

    plt.figure(figsize=(10, 8))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    for fold, (train_idx, test_idx) in enumerate(stratified_kfold.split(X, y_encoded), 1):
        print(f"\nProcessing fold {fold}/{cv}")

        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y_encoded[train_idx], y_encoded[test_idx]

        # Remove constant features
        variance_selector = VarianceThreshold(threshold=0)
        X_train_var = variance_selector.fit_transform(X_train)
        X_test_var = variance_selector.transform(X_test)

        X_train_final = X_train_var
        X_test_final = X_test_var

        # Apply feature selection if k is specified
        if k is not None and k != 'all':
            if isinstance(k, str):
                k = int(k)
            selector = SelectKBest(f_classif, k=k)
            X_train_final = selector.fit_transform(X_train_var, y_train)
            X_test_final = selector.transform(X_test_var)

        # Train TabPFN
        model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
        model.fit(X_train_final, y_train)

        # Predict probabilities for test set
        y_pred_proba = model.predict_proba(X_test_final)[:, 0]  # Probability of antigen class
        y_pred_proba = 1 - y_pred_proba  # Invert scores so higher = more antigenic

        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)

        # Interpolate TPR to common FPR values
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_interp[0] = 0.0
        tprs.append(tpr_interp)

        # Plot individual fold ROC
        plt.plot(fpr, tpr, lw=1, alpha=0.3, label=f'Fold {fold} (AUC = {roc_auc:.2f})')

    # Calculate mean ROC
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)

    # Plot mean ROC
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (AUC = {mean_auc:.2f} ± {std_auc:.2f})', lw=2, alpha=.8)

    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2, label='±1 std. dev.')

    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    plt.xlim([-0.02, 1.02])
    plt.ylim([-0.02, 1.02])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    model_name = k if k != 'all' else 'all'
    plt.title(f'10-fold CV ROC Curves (TabPFN - {model_name} features)', fontsize=24, fontweight='bold', pad=20)
    plt.legend(loc='lower right', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.3)

    # Save plot
    plt.savefig(os.path.join(results_dir, '10fold_cv_roc.png'), dpi=300, bbox_inches='tight')
    plt.close()

    # Save results
    results_file = os.path.join(results_dir, '10fold_cv_results.txt')
    with open(results_file, 'w') as f:
        f.write(f"10-fold Cross-Validation Results (TabPFN - {model_suffix})\n")
        f.write("=" * 60 + "\n")
        f.write(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}\n")
        f.write(f"Individual fold AUCs: {aucs}\n")
        f.write(f"Mean AUC: {mean_auc:.4f}\n")
        f.write(f"AUC Standard Deviation: {std_auc:.4f}\n")

    print("\nCross-validation completed!")
    print(f"Mean AUC: {mean_auc:.4f} ± {std_auc:.4f}")
    print(f"Results saved in {results_dir}/")

    return mean_auc, std_auc

def main():
    parser = argparse.ArgumentParser(description='10-fold cross-validation for TabPFN model')
    parser.add_argument('--k', type=str, default='all',
                       help='Number of features to use (default: all, options: all, 529, 100)')
    parser.add_argument('--cv', type=int, default=10, help='Number of cross-validation folds')
    args = parser.parse_args()

    X, labels, feature_names, _, _ = load_and_extract_features()

    perform_cross_validation(X, labels, k=args.k, cv=args.cv)

if __name__ == "__main__":
    main()
