import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))

import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from tabpfn import TabPFNClassifier
import joblib
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

def train_and_save_model(X, y, feature_names, k=None):
    model_suffix = get_model_suffix(k)
    print(f"\nTraining TabPFN model ({model_suffix})...")

    # Create model and results directories for this k
    models_dir = f'../models/{model_suffix}'
    results_dir = f'../results/{model_suffix}'
    ensure_dir_exists(models_dir)
    ensure_dir_exists(results_dir)

    # Setup logging for this specific model
    log_file = os.path.join(results_dir, 'training.log')
    logging.basicConfig(filename=log_file, level=logging.INFO,
                       format='%(asctime)s - %(levelname)s - %(message)s')

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Apply VarianceThreshold to remove constant features
    variance_selector = VarianceThreshold(threshold=0.0)
    X_filtered = variance_selector.fit_transform(X)
    print(f"Features after VarianceThreshold: {X_filtered.shape[1]} (removed {X.shape[1] - X_filtered.shape[1]} constant features)")

    X_final = X_filtered
    feature_selector = None

    # Apply feature selection if k is specified and not 'all'
    if k is not None and k != 'all':
        if isinstance(k, str):
            k = int(k)
        print(f"Applying SelectKBest with k={k}...")
        feature_selector = SelectKBest(f_classif, k=k)
        X_final = feature_selector.fit_transform(X_filtered, y_encoded)
        print(f"Features after SelectKBest: {X_final.shape[1]}")

    # Train TabPFN (no hyperparameters, no SMOTE, no scaling)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    model = TabPFNClassifier(device=device, ignore_pretraining_limits=True)
    model.fit(X_final, y_encoded)

    print("\nSaving model and components...")
    model_file = os.path.join(models_dir, 'IApred_TabPFN.joblib')
    variance_file = os.path.join(models_dir, 'IApred_variance_selector.joblib')
    feature_names_file = os.path.join(models_dir, 'IApred_all_feature_names.joblib')

    joblib.dump(model, model_file)
    joblib.dump(variance_selector, variance_file)
    joblib.dump(feature_names, feature_names_file)

    if feature_selector is not None:
        selector_file = os.path.join(models_dir, 'IApred_feature_selector.joblib')
        joblib.dump(feature_selector, selector_file)
        print(f"- IApred_feature_selector.joblib")

    print(f"Files saved in '{models_dir}' directory:")
    print("- IApred_TabPFN.joblib")
    print("- IApred_variance_selector.joblib")
    print("- IApred_all_feature_names.joblib")
    print(f"\nFinal number of features used: {X_final.shape[1]}")

def main():
    parser = argparse.ArgumentParser(description='Train TabPFN model with specified number of features')
    parser.add_argument('--k', type=str, default='all',
                       help='Number of features to use (default: all, options: all, 529, 100)')
    args = parser.parse_args()

    X, labels, feature_names, _, _ = load_and_extract_features()

    train_and_save_model(X, labels, feature_names, k=args.k)

    print("\nProcess completed successfully!")

if __name__ == "__main__":
    main()
