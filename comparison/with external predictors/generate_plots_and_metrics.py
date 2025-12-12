#!/usr/bin/env python3
"""COMPREHENSIVE PLOTS AND METRICS GENERATION FOR ANTIGEN PREDICTORS"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, matthews_corrcoef, brier_score_loss
from sklearn.calibration import calibration_curve
from math import pi
import logging

logging.basicConfig(level=logging.INFO)

def clean_predictor_names(predictors):
    """Clean predictor names by removing '_score' suffix"""
    cleaned = []
    for pred in predictors:
        if pred.endswith('_score'):
            cleaned.append(pred.replace('_score', ''))
        else:
            cleaned.append(pred)
    return cleaned

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
            ece += np.abs(bin_acc - bin_conf)
            valid_bins += 1
    return ece / valid_bins if valid_bins > 0 else 0

def generate_plots_and_metrics():
    """Generate plots and comprehensive metrics for all predictors"""
    
    # Load data
    antigens = pd.read_csv('unified_antigens.csv')
    non_antigens = pd.read_csv('unified_non_antigens.csv')
    
    antigens['label'] = 1
    non_antigens['label'] = 0
    combined = pd.concat([antigens, non_antigens], ignore_index=True)
    
    predictors = ['SVM_score', 'TabPFN-all_score', 'TabPFN-529_score', 'TabPFN-100_score', 
                  'ANTIGENpro', 'Vaxijen 3.0', 'VaxiJen 2.0']
    cleaned_predictors = clean_predictor_names(predictors)
    
    y_true = combined['label']
    
    print(f'Full dataset shape: {combined.shape}')
    
    # Generate ROC plot with cleaned labels
    plt.figure(figsize=(8, 8))
    
    for i, pred in enumerate(predictors):
        if pred in combined.columns:
            # Use all available data for this predictor
            valid_mask = combined[pred].notna()
            y_scores = combined.loc[valid_mask, pred]
            y_true_valid = y_true[valid_mask]
            
            if len(y_scores) > 0:
                fpr, tpr, _ = roc_curve(y_true_valid, y_scores)
                roc_auc = auc(fpr, tpr)
                
                plt.plot(fpr, tpr, lw=2, 
                        label='{} (AUC = {:.3f})'.format(cleaned_predictors[i], roc_auc))
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.5)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curves Comparison - All Predictors', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.savefig('roc_auc_comparison_unified.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate comprehensive metrics
    results = {'Predictor': []}
    metrics_to_calculate = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'Specificity', 'Sensitivity', 
                           'MCC', 'Youden_J', 'Brier_Score', 'ECE', 'TN', 'FP', 'FN', 'TP']
    
    for metric in metrics_to_calculate:
        results[metric] = []
    
    for predictor in predictors:
        if predictor in combined.columns:
            # Use all available data for this predictor
            valid_mask = combined[predictor].notna()
            y_pred_proba = combined.loc[valid_mask, predictor].values
            y_true_valid = y_true[valid_mask].values
            
            if len(y_pred_proba) == 0:
                continue
            
            # Special handling for VaxiJen 2.0
            if predictor == 'VaxiJen 2.0':
                class_data = combined.loc[valid_mask, 'Class'].values
                gram_classes = ['Gram+', 'Gram-']
                threshold_mask = np.isin(class_data, gram_classes)
                y_pred = np.zeros(len(y_pred_proba))
                y_pred[threshold_mask] = (y_pred_proba[threshold_mask] >= 0.4).astype(int)
                y_pred[~threshold_mask] = (y_pred_proba[~threshold_mask] >= 0.5).astype(int)
            else:
                y_pred = (y_pred_proba >= 0.5).astype(int)
            
            # Basic metrics
            accuracy = np.mean(y_pred == y_true_valid)
            precision = np.sum((y_pred == 1) & (y_true_valid == 1)) / np.sum(y_pred == 1) if np.sum(y_pred == 1) > 0 else 0
            recall = np.sum((y_pred == 1) & (y_true_valid == 1)) / np.sum(y_true_valid == 1) if np.sum(y_true_valid == 1) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            fpr_roc, tpr_roc, _ = roc_curve(y_true_valid, y_pred_proba)
            auc_score = auc(fpr_roc, tpr_roc)
            
            # Confusion matrix
            tn, fp, fn, tp = confusion_matrix(y_true_valid, y_pred).ravel()
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            # Additional metrics
            mcc = matthews_corrcoef(y_true_valid, y_pred)
            youden_j = sensitivity + specificity - 1
            
            # Normalize scores for Brier score
            y_pred_proba_norm = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min()) if y_pred_proba.max() > y_pred_proba.min() else y_pred_proba
            brier = brier_score_loss(y_true_valid, y_pred_proba_norm)
            
            ece = calculate_ece(y_true_valid, y_pred_proba_norm)
            
            results['Predictor'].append(predictor)
            results['Accuracy'].append(accuracy)
            results['Precision'].append(precision)
            results['Recall'].append(recall)
            results['F1-Score'].append(f1)
            results['AUC'].append(auc_score)
            results['Specificity'].append(specificity)
            results['Sensitivity'].append(sensitivity)
            results['MCC'].append(mcc)
            results['Youden_J'].append(youden_j)
            results['Brier_Score'].append(brier)
            results['ECE'].append(ece)
            results['TN'].append(tn)
            results['FP'].append(fp)
            results['FN'].append(fn)
            results['TP'].append(tp)
    
    metrics_df = pd.DataFrame(results)
    
    # Print data completeness
    print('\nData completeness per predictor:')
    for predictor in predictors:
        if predictor in combined.columns:
            valid_count = combined[predictor].notna().sum()
            total_count = len(combined)
            print(f'  {predictor}: {valid_count}/{total_count} ({100*valid_count/total_count:.1f}%)')
    
    # Save comprehensive metrics CSV
    metrics_df.to_csv('all_evaluation_metrics_comprehensive.csv', index=False)
    print(f'\n✅ Saved comprehensive metrics CSV with {len(metrics_df)} predictors and {len(metrics_to_calculate)} metrics')
    
    # Generate radial plot with cleaned labels
    radar_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'AUC', 'MCC']
    num_metrics = len(radar_metrics)
    angles = [n / float(num_metrics) * 2 * pi for n in range(num_metrics)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(projection='polar'))
    
    for idx, pred in enumerate(predictors):
        row = metrics_df[metrics_df['Predictor'] == pred]
        if not row.empty:
            values = [row[metric].values[0] for metric in radar_metrics]
            values += values[:1]
            ax.plot(angles, values, 'o-', linewidth=2, label=cleaned_predictors[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(radar_metrics, fontsize=10)
    ax.set_ylim(0.2, 1.0)
    ax.set_title('Performance Metrics Comparison - All Predictors', size=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0), fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.savefig('radial_metrics_comparison_unified.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print('\n✅ Generated plots and comprehensive metrics successfully!')

if __name__ == '__main__':
    generate_plots_and_metrics()
