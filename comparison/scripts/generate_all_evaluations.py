#!/usr/bin/env python3
"""
Script to run external evaluation for all 4 models and generate comparison CSV
Models:
1. SVM
2. TabPFN (all features)
3. TabPFN (529 features)
4. TabPFN (100 features)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
from joblib import load
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix, matthews_corrcoef, brier_score_loss
)
from sklearn.calibration import calibration_curve
from functions_for_training import sequences_to_vectors
import logging

logging.basicConfig(level=logging.INFO)

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

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

def load_svm_predictions(project_root):
    """Load SVM model and generate predictions"""
    model_path = os.path.join(project_root, 'svm_training/models')
    model = load(os.path.join(model_path, 'IApred_SVM.joblib'))
    scaler = load(os.path.join(model_path, 'IApred_scaler.joblib'))
    variance_selector = load(os.path.join(model_path, 'IApred_variance_selector.joblib'))
    feature_selector = load(os.path.join(model_path, 'IApred_feature_selector.joblib'))
    all_feature_names = load(os.path.join(model_path, 'IApred_all_feature_names.joblib'))
    
    # Load data
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))
    
    sequences_ant = [str(s) for s in antigens_df['Antigen'].tolist() if pd.notna(s)]
    sequences_non = [str(s) for s in non_antigens_df['Non-Antigen'].tolist() if pd.notna(s)]
    
    X_ant, feature_names_ant, failed_ant = sequences_to_vectors(sequences_ant)
    X_non, feature_names_non, failed_non = sequences_to_vectors(sequences_non)
    
    # Remove failed sequences
    if len(failed_ant) > 0:
        failed_ant = failed_ant.astype(int)
        antigens_df = antigens_df.drop(antigens_df.index[failed_ant]).reset_index(drop=True)
        valid_indices_ant = np.array([i for i in range(len(sequences_ant)) if i not in failed_ant])
        X_ant = X_ant[valid_indices_ant]
    if len(failed_non) > 0:
        failed_non = failed_non.astype(int)
        non_antigens_df = non_antigens_df.drop(non_antigens_df.index[failed_non]).reset_index(drop=True)
        valid_indices_non = np.array([i for i in range(len(sequences_non)) if i not in failed_non])
        X_non = X_non[valid_indices_non]
    
    # Align features
    feature_map_ant = {name: i for i, name in enumerate(feature_names_ant)}
    feature_map_non = {name: i for i, name in enumerate(feature_names_non)}
    X_ant_aligned = np.zeros((X_ant.shape[0], len(all_feature_names)))
    X_non_aligned = np.zeros((X_non.shape[0], len(all_feature_names)))
    
    for i, feature in enumerate(all_feature_names):
        if feature in feature_map_ant:
            X_ant_aligned[:, i] = X_ant[:, feature_map_ant[feature]]
        if feature in feature_map_non:
            X_non_aligned[:, i] = X_non[:, feature_map_non[feature]]
    
    X_ant_processed = variance_selector.transform(X_ant_aligned)
    X_non_processed = variance_selector.transform(X_non_aligned)
    X_ant_scaled = scaler.transform(X_ant_processed)
    X_non_scaled = scaler.transform(X_non_processed)
    X_ant_selected = feature_selector.transform(X_ant_scaled)
    X_non_selected = feature_selector.transform(X_non_scaled)
    
    scores_ant = model.decision_function(X_ant_selected)
    scores_non = model.decision_function(X_non_selected)

    # Convert decision function to probability-like scores [0,1]
    # SVM: negative = antigen, positive = non-antigen
    # For consistent interpretation: higher score = more antigenic
    from scipy.special import expit
    scores_ant_norm = expit(-scores_ant)  # Negative because SVM: negative = antigen
    scores_non_norm = expit(-scores_non)

    antigens_df['IApred'] = scores_ant_norm
    non_antigens_df['IApred'] = scores_non_norm

    # Track and remove duplicates
    removed_duplicates = []

    if antigens_df['Antigen'].duplicated().any():
        duplicate_mask = antigens_df['Antigen'].duplicated()
        duplicate_rows = antigens_df[duplicate_mask].copy()
        # Rename 'Antigen' column to 'Sequence' for consistency
        duplicate_rows = duplicate_rows.rename(columns={'Antigen': 'Sequence'})
        duplicate_rows['Dataset'] = 'antigens'
        removed_duplicates.extend(duplicate_rows.to_dict('records'))
        print(f"  ⚠ Warning: Found {len(duplicate_rows)} duplicate sequences in antigens, removing...")
        antigens_df = antigens_df.drop_duplicates(subset=['Antigen'], keep='first').reset_index(drop=True)

    if non_antigens_df['Non-Antigen'].duplicated().any():
        duplicate_mask = non_antigens_df['Non-Antigen'].duplicated()
        duplicate_rows = non_antigens_df[duplicate_mask].copy()
        # Rename 'Non-Antigen' column to 'Sequence' for consistency
        duplicate_rows = duplicate_rows.rename(columns={'Non-Antigen': 'Sequence'})
        duplicate_rows['Dataset'] = 'non-antigens'
        removed_duplicates.extend(duplicate_rows.to_dict('records'))
        print(f"  ⚠ Warning: Found {len(duplicate_rows)} duplicate sequences in non-antigens, removing...")
        non_antigens_df = non_antigens_df.drop_duplicates(subset=['Non-Antigen'], keep='first').reset_index(drop=True)

    return antigens_df, non_antigens_df, removed_duplicates

def load_tabpfn_predictions(project_root, model_name, model_path, exclude_helminths=False):
    """Load TabPFN model and generate predictions"""
    model = load(os.path.join(model_path, 'IApred_TabPFN.joblib'))

    variance_selector = load(os.path.join(model_path, 'IApred_variance_selector.joblib'))
    all_feature_names = load(os.path.join(model_path, 'IApred_all_feature_names.joblib'))

    # Load feature selector if it exists (for 529 and 100 features models)
    feature_selector = None
    selector_path = os.path.join(model_path, 'IApred_feature_selector.joblib')
    if os.path.exists(selector_path):
        feature_selector = load(selector_path)
    
    # Load data
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))
    
    sequences_ant = [str(s) for s in antigens_df['Antigen'].tolist() if pd.notna(s)]
    sequences_non = [str(s) for s in non_antigens_df['Non-Antigen'].tolist() if pd.notna(s)]
    
    X_ant, feature_names_ant, failed_ant = sequences_to_vectors(sequences_ant)
    X_non, feature_names_non, failed_non = sequences_to_vectors(sequences_non)
    
    # Remove failed sequences
    if len(failed_ant) > 0:
        failed_ant = failed_ant.astype(int)
        antigens_df = antigens_df.drop(antigens_df.index[failed_ant]).reset_index(drop=True)
        valid_indices_ant = np.array([i for i in range(len(sequences_ant)) if i not in failed_ant])
        X_ant = X_ant[valid_indices_ant]
    if len(failed_non) > 0:
        failed_non = failed_non.astype(int)
        non_antigens_df = non_antigens_df.drop(non_antigens_df.index[failed_non]).reset_index(drop=True)
        valid_indices_non = np.array([i for i in range(len(sequences_non)) if i not in failed_non])
        X_non = X_non[valid_indices_non]
    
    # Align features
    feature_map_ant = {name: i for i, name in enumerate(feature_names_ant)}
    feature_map_non = {name: i for i, name in enumerate(feature_names_non)}
    X_ant_aligned = np.zeros((X_ant.shape[0], len(all_feature_names)))
    X_non_aligned = np.zeros((X_non.shape[0], len(all_feature_names)))
    
    for i, feature in enumerate(all_feature_names):
        if feature in feature_map_ant:
            X_ant_aligned[:, i] = X_ant[:, feature_map_ant[feature]]
        if feature in feature_map_non:
            X_non_aligned[:, i] = X_non[:, feature_map_non[feature]]
    
    X_ant_aligned = np.nan_to_num(X_ant_aligned, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    X_ant_aligned = np.clip(X_ant_aligned, -1e6, 1e6)
    X_non_aligned = np.nan_to_num(X_non_aligned, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    X_non_aligned = np.clip(X_non_aligned, -1e6, 1e6)
    
    # Apply variance selector
    X_ant_filtered = variance_selector.transform(X_ant_aligned)
    X_non_filtered = variance_selector.transform(X_non_aligned)

    # Apply feature selector if it exists (for 529 and 100 features models)
    if feature_selector is not None:
        X_ant_filtered = feature_selector.transform(X_ant_filtered)
        X_non_filtered = feature_selector.transform(X_non_filtered)
    
    # TabPFN: class 0 = 'antigen', class 1 = 'non-antigen'
    # Get probabilities for class 0 (antigen) - no inversion needed
    # Higher probability of class 0 = more antigenic
    probabilities_ant = model.predict_proba(X_ant_filtered)[:, 0]
    probabilities_non = model.predict_proba(X_non_filtered)[:, 0]
    antigens_df['IApred'] = probabilities_ant
    non_antigens_df['IApred'] = probabilities_non
    
    # Exclude helminths if requested
    if exclude_helminths:
        antigens_df = antigens_df.iloc[:-5].copy() if len(antigens_df) >= 5 else antigens_df.copy()
        non_antigens_df = non_antigens_df.iloc[:-5].copy() if len(non_antigens_df) >= 5 else non_antigens_df.copy()
    
    # Check for and remove duplicates
    seq_col_ant = 'Antigen' if 'Antigen' in antigens_df.columns else antigens_df.columns[0]
    seq_col_non = 'Non-Antigen' if 'Non-Antigen' in non_antigens_df.columns else non_antigens_df.columns[0]

    removed_duplicates = []

    if antigens_df[seq_col_ant].duplicated().any():
        duplicate_mask = antigens_df[seq_col_ant].duplicated()
        duplicate_rows = antigens_df[duplicate_mask].copy()
        # Rename sequence column to 'Sequence' for consistency
        duplicate_rows = duplicate_rows.rename(columns={seq_col_ant: 'Sequence'})
        duplicate_rows['Dataset'] = 'antigens'
        removed_duplicates.extend(duplicate_rows.to_dict('records'))
        print(f"  ⚠ Warning: Found {len(duplicate_rows)} duplicate sequences in antigens, removing...")
        antigens_df = antigens_df.drop_duplicates(subset=[seq_col_ant], keep='first').reset_index(drop=True)

    if non_antigens_df[seq_col_non].duplicated().any():
        duplicate_mask = non_antigens_df[seq_col_non].duplicated()
        duplicate_rows = non_antigens_df[duplicate_mask].copy()
        # Rename sequence column to 'Sequence' for consistency
        duplicate_rows = duplicate_rows.rename(columns={seq_col_non: 'Sequence'})
        duplicate_rows['Dataset'] = 'non-antigens'
        removed_duplicates.extend(duplicate_rows.to_dict('records'))
        print(f"  ⚠ Warning: Found {len(duplicate_rows)} duplicate sequences in non-antigens, removing...")
        non_antigens_df = non_antigens_df.drop_duplicates(subset=[seq_col_non], keep='first').reset_index(drop=True)

    return antigens_df, non_antigens_df, removed_duplicates

def calculate_metrics(antigens_df, non_antigens_df, model_type='SVM'):
    """Calculate all performance metrics"""
    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    y_scores = np.concatenate([antigens_df['IApred'].values, non_antigens_df['IApred'].values])

    # All models now output scores in [0,1] range, use 0.5 as threshold
    y_pred = (y_scores >= 0.5).astype(int)
    
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
        'Model': model_type,
        'True Positive': tp, 'True Negative': tn, 'False Positive': fp, 'False Negative': fn,
        'Accuracy': accuracy_score(y_true, y_pred), 'Precision': precision, 'Sensitivity': sensitivity,
        'Specificity': specificity, 'MCC': mcc, "Youden's J": j_index, 'F1-score': f1,
        'F0.5-score': f05, 'Brier Score': brier, 'ECE': ece, 'AUC-ROC': roc_auc,
        'Sample Size': len(y_true)
    }

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    ensure_dir_exists(os.path.join(project_root, 'comparison/results'))
    
    print("="*80)
    print("GENERATING ALL MODEL EVALUATIONS")
    print("="*80)

    all_results = []
    all_antigens_predictions = {}  # Store predictions by model name
    all_non_antigens_predictions = {}  # Store predictions by model name
    all_removed_duplicates = []  # Track all removed duplicates

    # 1. SVM
    print("\n[1/4] Processing SVM...")
    try:
        antigens_df, non_antigens_df, removed_duplicates = load_svm_predictions(project_root)
        all_removed_duplicates.extend(removed_duplicates)
        metrics = calculate_metrics(antigens_df, non_antigens_df, 'SVM')
        all_results.append(metrics)
        # Store predictions for merging (already deduplicated)
        all_antigens_predictions['SVM'] = antigens_df.copy()
        all_non_antigens_predictions['SVM'] = non_antigens_df.copy()
        print(f"    Antigens: {len(antigens_df)} unique sequences")
        print(f"    Non-antigens: {len(non_antigens_df)} unique sequences")
        print(f"  ✓ SVM: AUC={metrics['AUC-ROC']:.3f}, MCC={metrics['MCC']:.3f}")
    except Exception as e:
        print(f"  ✗ Error with SVM: {str(e)}")

    # 2. TabPFN (all features)
    print("\n[2/4] Processing TabPFN (all features)...")
    try:
        model_path = os.path.join(project_root, 'tabpfn_training/models/all_features')
        antigens_df, non_antigens_df, removed_duplicates = load_tabpfn_predictions(project_root, 'TabPFN', model_path, exclude_helminths=False)
        all_removed_duplicates.extend(removed_duplicates)
        metrics = calculate_metrics(antigens_df, non_antigens_df, 'TabPFN-all')
        all_results.append(metrics)
        # Store predictions for merging (already deduplicated)
        all_antigens_predictions['TabPFN-all'] = antigens_df.copy()
        all_non_antigens_predictions['TabPFN-all'] = non_antigens_df.copy()
        print(f"    Antigens: {len(antigens_df)} unique sequences")
        print(f"    Non-antigens: {len(non_antigens_df)} unique sequences")
        print(f"  ✓ TabPFN-all: AUC={metrics['AUC-ROC']:.3f}, MCC={metrics['MCC']:.3f}")
    except Exception as e:
        print(f"  ✗ Error with TabPFN-all: {str(e)}")
        import traceback
        traceback.print_exc()

    # 3. TabPFN (529 features)
    print("\n[3/4] Processing TabPFN (529 features)...")
    try:
        model_path = os.path.join(project_root, 'tabpfn_training/models/529_features')
        antigens_df, non_antigens_df, removed_duplicates = load_tabpfn_predictions(project_root, 'TabPFN', model_path, exclude_helminths=False)
        all_removed_duplicates.extend(removed_duplicates)
        metrics = calculate_metrics(antigens_df, non_antigens_df, 'TabPFN-529')
        all_results.append(metrics)
        # Store predictions for merging (already deduplicated)
        all_antigens_predictions['TabPFN-529'] = antigens_df.copy()
        all_non_antigens_predictions['TabPFN-529'] = non_antigens_df.copy()
        print(f"    Antigens: {len(antigens_df)} unique sequences")
        print(f"    Non-antigens: {len(non_antigens_df)} unique sequences")
        print(f"  ✓ TabPFN-529: AUC={metrics['AUC-ROC']:.3f}, MCC={metrics['MCC']:.3f}")
    except Exception as e:
        print(f"  ✗ Error with TabPFN-529: {str(e)}")
        import traceback
        traceback.print_exc()

    # 4. TabPFN (100 features)
    print("\n[4/4] Processing TabPFN (100 features)...")
    try:
        model_path = os.path.join(project_root, 'tabpfn_training/models/100_features')
        antigens_df, non_antigens_df, removed_duplicates = load_tabpfn_predictions(project_root, 'TabPFN', model_path, exclude_helminths=False)
        all_removed_duplicates.extend(removed_duplicates)
        metrics = calculate_metrics(antigens_df, non_antigens_df, 'TabPFN-100')
        all_results.append(metrics)
        # Store predictions for merging (already deduplicated)
        all_antigens_predictions['TabPFN-100'] = antigens_df.copy()
        all_non_antigens_predictions['TabPFN-100'] = non_antigens_df.copy()
        print(f"    Antigens: {len(antigens_df)} unique sequences")
        print(f"    Non-antigens: {len(non_antigens_df)} unique sequences")
        print(f"  ✓ TabPFN-100: AUC={metrics['AUC-ROC']:.3f}, MCC={metrics['MCC']:.3f}")
    except Exception as e:
        print(f"  ✗ Error with TabPFN-100: {str(e)}")
        import traceback
        traceback.print_exc()
    
    # Save results
    if all_results:
        results_df = pd.DataFrame(all_results)
        output_path = os.path.join(project_root, 'comparison/results/all_models_comparison.csv')
        results_df.to_csv(output_path, index=False)
        print(f"\n{'='*80}")
        print(f"Results saved to: {output_path}")
        print(f"{'='*80}")
        print("\nSummary:")
        print(results_df[['Model', 'AUC-ROC', 'MCC', 'Sensitivity', 'Specificity', 'True Positive', 'True Negative', 'False Positive', 'False Negative']].to_string(index=False))
        
        # Create combined prediction CSVs
        print("\n" + "="*80)
        print("CREATING COMBINED PREDICTION FILES")
        print("="*80)
        
        # Load original data to ensure all sequences are included
        try:
            original_antigens = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
            original_non_antigens = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))
        except:
            print("  ⚠ Warning: Could not load original data files")
            original_antigens = None
            original_non_antigens = None
        
        # Combine antigens predictions
        if all_antigens_predictions:
            # Start with original data or first model's dataframe
            if original_antigens is not None and 'Antigen' in original_antigens.columns:
                combined_antigens = original_antigens[['Antigen']].copy()
                combined_antigens.rename(columns={'Antigen': 'Sequence'}, inplace=True)
                # Remove duplicates from original data
                combined_antigens = combined_antigens.drop_duplicates(subset=['Sequence']).reset_index(drop=True)
            else:
                first_model = list(all_antigens_predictions.keys())[0]
                seq_col = 'Antigen' if 'Antigen' in all_antigens_predictions[first_model].columns else all_antigens_predictions[first_model].columns[0]
                combined_antigens = all_antigens_predictions[first_model][[seq_col]].copy()
                combined_antigens.rename(columns={seq_col: 'Sequence'}, inplace=True)
                # Remove duplicates
                combined_antigens = combined_antigens.drop_duplicates(subset=['Sequence']).reset_index(drop=True)
            
            # Add scores from each model
            for model_name, df in all_antigens_predictions.items():
                # Get the sequence column name
                seq_col = 'Antigen' if 'Antigen' in df.columns else df.columns[0]
                # Create a dataframe with sequence and score
                model_scores = df[[seq_col, 'IApred']].copy()
                model_scores.rename(columns={seq_col: 'Sequence', 'IApred': f'{model_name}_score'}, inplace=True)
                # Remove duplicates - keep first occurrence if duplicates exist
                model_scores = model_scores.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True)
                # Merge on sequence (left join to keep all original sequences)
                combined_antigens = combined_antigens.merge(model_scores, on='Sequence', how='left')
            
            # Final deduplication to ensure no duplicates in final output
            initial_count = len(combined_antigens)
            combined_antigens = combined_antigens.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True)
            final_count = len(combined_antigens)
            
            if initial_count != final_count:
                print(f"  ⚠ Warning: Removed {initial_count - final_count} duplicate sequences from final antigens output")
            
            # Save combined antigens CSV
            antigens_output = os.path.join(project_root, 'comparison/results/all_models_antigens_predictions.csv')
            combined_antigens.to_csv(antigens_output, index=False)
            print(f"  ✓ Combined antigens predictions saved: {antigens_output}")
            print(f"    Total rows: {combined_antigens.shape[0]} (should be {final_count})")
            print(f"    Unique sequences: {combined_antigens['Sequence'].nunique()} (should match row count)")
            print(f"    Columns: {', '.join(combined_antigens.columns.tolist())}")
        
        # Combine non-antigens predictions
        if all_non_antigens_predictions:
            # Start with original data or first model's dataframe
            if original_non_antigens is not None and 'Non-Antigen' in original_non_antigens.columns:
                combined_non_antigens = original_non_antigens[['Non-Antigen']].copy()
                combined_non_antigens.rename(columns={'Non-Antigen': 'Sequence'}, inplace=True)
                # Remove duplicates from original data
                combined_non_antigens = combined_non_antigens.drop_duplicates(subset=['Sequence']).reset_index(drop=True)
            else:
                first_model = list(all_non_antigens_predictions.keys())[0]
                seq_col = 'Non-Antigen' if 'Non-Antigen' in all_non_antigens_predictions[first_model].columns else all_non_antigens_predictions[first_model].columns[0]
                combined_non_antigens = all_non_antigens_predictions[first_model][[seq_col]].copy()
                combined_non_antigens.rename(columns={seq_col: 'Sequence'}, inplace=True)
                # Remove duplicates
                combined_non_antigens = combined_non_antigens.drop_duplicates(subset=['Sequence']).reset_index(drop=True)
            
            # Add scores from each model
            for model_name, df in all_non_antigens_predictions.items():
                # Get the sequence column name
                seq_col = 'Non-Antigen' if 'Non-Antigen' in df.columns else df.columns[0]
                # Create a dataframe with sequence and score
                model_scores = df[[seq_col, 'IApred']].copy()
                model_scores.rename(columns={seq_col: 'Sequence', 'IApred': f'{model_name}_score'}, inplace=True)
                # Remove duplicates - keep first occurrence if duplicates exist
                model_scores = model_scores.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True)
                # Merge on sequence (left join to keep all original sequences)
                combined_non_antigens = combined_non_antigens.merge(model_scores, on='Sequence', how='left')
            
            # Final deduplication to ensure no duplicates in final output
            initial_count = len(combined_non_antigens)
            combined_non_antigens = combined_non_antigens.drop_duplicates(subset=['Sequence'], keep='first').reset_index(drop=True)
            final_count = len(combined_non_antigens)
            
            if initial_count != final_count:
                print(f"  ⚠ Warning: Removed {initial_count - final_count} duplicate sequences from final non-antigens output")
            
            # Save combined non-antigens CSV
            non_antigens_output = os.path.join(project_root, 'comparison/results/all_models_non-antigens_predictions.csv')
            combined_non_antigens.to_csv(non_antigens_output, index=False)
            print(f"  ✓ Combined non-antigens predictions saved: {non_antigens_output}")
            print(f"    Total rows: {combined_non_antigens.shape[0]} (should be {final_count})")
            print(f"    Unique sequences: {combined_non_antigens['Sequence'].nunique()} (should match row count)")
            print(f"    Columns: {', '.join(combined_non_antigens.columns.tolist())}")

        # Save removed duplicates to CSV if any exist
        if all_removed_duplicates:
            # Ensure all records have consistent column names
            processed_duplicates = []
            for dup in all_removed_duplicates:
                processed_dup = {}
                # Copy all columns except sequence columns
                for key, value in dup.items():
                    if key not in ['Antigen', 'Non-Antigen']:
                        processed_dup[key] = value

                # Set the sequence column consistently
                if 'Sequence' in dup:
                    processed_dup['Sequence'] = dup['Sequence']
                elif 'Antigen' in dup:
                    processed_dup['Sequence'] = dup['Antigen']
                elif 'Non-Antigen' in dup:
                    processed_dup['Sequence'] = dup['Non-Antigen']
                else:
                    processed_dup['Sequence'] = ''

                processed_duplicates.append(processed_dup)

            duplicates_df = pd.DataFrame(processed_duplicates)
            # Reorder columns to put Sequence first
            cols = ['Class', 'Organism', 'Sequence', 'IApred', 'Dataset'] + [col for col in duplicates_df.columns if col not in ['Class', 'Organism', 'Sequence', 'IApred', 'Dataset']]
            duplicates_df = duplicates_df[cols]

            duplicates_output = os.path.join(project_root, 'comparison/results/removed_duplicates.csv')
            duplicates_df.to_csv(duplicates_output, index=False)
            print(f"\n📋 Removed duplicates saved to: {duplicates_output}")
            print(f"   Total duplicates removed: {len(processed_duplicates)}")
            print(f"   Columns: {', '.join(duplicates_df.columns.tolist())}")
        else:
            print("\n📋 No duplicates were found and removed.")

    else:
        print("\n✗ No results were generated!")

if __name__ == "__main__":
    main()

