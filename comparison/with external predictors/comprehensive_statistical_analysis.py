#!/usr/bin/env python3
"""COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS FOR ANTIGEN PREDICTORS"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc
from statsmodels.stats.contingency_tables import mcnemar
import itertools
from scipy import stats
from matplotlib.patches import Rectangle
import matplotlib.colors as mcolors

def clean_predictor_names(predictors):
    """Clean predictor names by removing '_score' suffix"""
    cleaned = []
    for pred in predictors:
        if pred.endswith('_score'):
            cleaned.append(pred.replace('_score', ''))
        else:
            cleaned.append(pred)
    return cleaned

def delong_test(y_true, y_scores1, y_scores2, alpha=0.05):
    """Perform DeLong's test for comparing two AUCs""" 
    # Convert to numpy arrays to avoid pandas index issues
    y_true = np.asarray(y_true)
    y_scores1 = np.asarray(y_scores1)
    y_scores2 = np.asarray(y_scores2)
    
    fpr1, tpr1, _ = roc_curve(y_true, y_scores1)
    auc1 = auc(fpr1, tpr1)
    
    fpr2, tpr2, _ = roc_curve(y_true, y_scores2)
    auc2 = auc(fpr2, tpr2)
    
    n = len(y_true)
    q1 = auc1 * (1 - auc1)
    q2 = auc2 * (1 - auc2)
    cov = 0
    
    se = np.sqrt((q1 + q2 - 2*cov) / n)
    
    if se == 0:
        return auc1, auc2, 1.0, 'Cannot compute'
    
    z = abs(auc1 - auc2) / se
    p_value = 2 * (1 - stats.norm.cdf(z))
    
    significance = 'Significant' if p_value < alpha else 'Not significant'
    
    return auc1, auc2, p_value, significance

def mcnemar_test_predictions(y_true, pred1, pred2):
    """Perform McNemar's test on predictions"""
    # Convert to numpy arrays to avoid pandas index issues
    y_true = np.asarray(y_true)
    pred1 = np.asarray(pred1)
    pred2 = np.asarray(pred2)
    
    both_correct = np.sum((pred1 == y_true) & (pred2 == y_true))
    model1_correct_only = np.sum((pred1 == y_true) & (pred2 != y_true))
    model2_correct_only = np.sum((pred1 != y_true) & (pred2 == y_true))
    both_wrong = np.sum((pred1 != y_true) & (pred2 != y_true))
    
    contingency_table = np.array([[both_correct, model1_correct_only],
                                  [model2_correct_only, both_wrong]])
    
    try:
        result = mcnemar(contingency_table, exact=True, correction=True)
        p_value = result.pvalue
        statistic = result.statistic
    except:
        result = mcnemar(contingency_table, exact=False, correction=True)
        p_value = result.pvalue
        statistic = result.statistic
    
    significance = 'Significant' if p_value < 0.05 else 'Not significant'
    
    return contingency_table, statistic, p_value, significance

def create_combined_significance_heatmap(mcnemar_df, delong_df, predictors):
    """Create a combined heatmap with corrected color scheme"""
    cleaned_predictors = clean_predictor_names(predictors)
    
    # Create matrices
    mcnemar_matrix = pd.DataFrame(index=predictors, columns=predictors, dtype=float)
    delong_matrix = pd.DataFrame(index=predictors, columns=predictors, dtype=float)
    
    for _, row in mcnemar_df.iterrows():
        i = predictors.index(row['Predictor_1'])
        j = predictors.index(row['Predictor_2'])
        mcnemar_matrix.iloc[i, j] = row['p_value']
        mcnemar_matrix.iloc[j, i] = row['p_value']
    
    for _, row in delong_df.iterrows():
        i = predictors.index(row['Predictor_1'])
        j = predictors.index(row['Predictor_2'])
        delong_matrix.iloc[i, j] = row['p_value']
        delong_matrix.iloc[j, i] = row['p_value']
    
    # Create combined matrix
    combined_matrix = pd.DataFrame(index=predictors, columns=predictors, dtype=float)
    
    for i in range(len(predictors)):
        for j in range(len(predictors)):
            if i == j:
                combined_matrix.iloc[i, j] = np.nan
            elif i > j:
                combined_matrix.iloc[i, j] = mcnemar_matrix.iloc[i, j]
            else:
                combined_matrix.iloc[i, j] = delong_matrix.iloc[i, j]
    
    # Red for p < 0.05 (significant), light yellow for p >= 0.05 (non-significant)
    colors = ['#FF4444', '#FFFF99']
    cmap = mcolors.ListedColormap(colors)
    bounds = [0, 0.05, 1.0]
    norm = mcolors.BoundaryNorm(bounds, cmap.N)
    
    # Mask for diagonal
    mask = np.zeros_like(combined_matrix, dtype=bool)
    mask[np.diag_indices_from(mask)] = True
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Heatmap without colorbar
    sns.heatmap(combined_matrix, annot=combined_matrix.round(4), fmt='.4f', 
                cmap=cmap, norm=norm, cbar=False,
                ax=ax, square=True, linewidths=0.5, mask=mask,
                annot_kws={'size': 10, 'weight': 'bold'})
    
    # Triangle backgrounds
    n = len(predictors)
    for i in range(n):
        for j in range(n):
            if i == j:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color='white', alpha=1.0, zorder=2))
            elif i > j:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color='lightblue', alpha=0.08, zorder=1))
            else:
                ax.add_patch(Rectangle((j, i), 1, 1, fill=True, color='lightcoral', alpha=0.08, zorder=1))
    
    # Title and labels with better spacing
    ax.set_title('Statistical Significance', fontsize=16, fontweight='bold', pad=30)
    ax.text(0.5, 1.02, "Lower: McNemar's | Upper: DeLong's", 
            transform=ax.transAxes, ha='center', fontsize=14, fontweight='bold')
    ax.set_xticklabels(cleaned_predictors, rotation=45, ha='right', fontsize=12, fontweight='bold')
    ax.set_yticklabels(cleaned_predictors, rotation=0, fontsize=12, fontweight='bold')
    
    # Legend
    legend_elements = [
        plt.Rectangle((0,0),1,1, facecolor='lightblue', alpha=0.3, label="McNemar's"),
        plt.Rectangle((0,0),1,1, facecolor='lightcoral', alpha=0.3, label="DeLong's"),
        plt.Rectangle((0,0),1,1, facecolor='#FF4444', alpha=1.0, label='p < 0.05'),
        plt.Rectangle((0,0),1,1, facecolor='#FFFF99', alpha=1.0, label='p ≥ 0.05')
    ]
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
             ncol=4, fontsize=10, frameon=True)
    
    plt.tight_layout()
    plt.savefig('combined_statistical_significance_heatmap_simple.png', dpi=300, bbox_inches='tight')
    plt.close()
    print('✅ Saved combined statistical significance heatmap')

def main():
    """Main function to run complete statistical analysis"""
    print('🚀 COMPREHENSIVE STATISTICAL SIGNIFICANCE ANALYSIS')
    print('=' * 60)
    
    print('Loading data...')
    antigens = pd.read_csv('unified_antigens.csv')
    non_antigens = pd.read_csv('unified_non_antigens.csv')
    
    antigens['label'] = 1
    non_antigens['label'] = 0
    combined = pd.concat([antigens, non_antigens], ignore_index=True)
    
    predictors = ['SVM_score', 'TabPFN-all_score', 'TabPFN-529_score', 'TabPFN-100_score', 
                  'ANTIGENpro', 'Vaxijen 3.0', 'VaxiJen 2.0']
    
    print('Full dataset shape:', combined.shape)
    print('Predictors included:', len(predictors))
    print('Total pairs to compare:', len(list(itertools.combinations(predictors, 2))))
    
    y_true = combined['label']
    
    print('Performing McNemar test...')
    mcnemar_results = []
    for pred1, pred2 in itertools.combinations(predictors, 2):
        if pred1 not in combined.columns or pred2 not in combined.columns:
            continue
            
        scores1 = combined[pred1]
        scores2 = combined[pred2]
        
        valid_mask = scores1.notna() & scores2.notna()
        sample_size = np.sum(valid_mask)
        
        if sample_size < 10:
            continue
            
        scores1_valid = scores1[valid_mask].values
        scores2_valid = scores2[valid_mask].values
        y_true_valid = y_true[valid_mask].values
        
        pred1_binary = (scores1_valid >= 0.5).astype(int)
        pred2_binary = (scores2_valid >= 0.5).astype(int)
        
        if pred1 == 'VaxiJen 2.0':
            class_data = combined.loc[valid_mask, 'Class'].values
            gram_classes = ['Gram+', 'Gram-']
            threshold_mask = np.isin(class_data, gram_classes)
            pred1_binary = np.zeros(len(scores1_valid))
            pred1_binary[threshold_mask] = (scores1_valid[threshold_mask] >= 0.4).astype(int)
            pred1_binary[~threshold_mask] = (scores1_valid[~threshold_mask] >= 0.5).astype(int)
        
        if pred2 == 'VaxiJen 2.0':
            class_data = combined.loc[valid_mask, 'Class'].values
            gram_classes = ['Gram+', 'Gram-']
            threshold_mask = np.isin(class_data, gram_classes)
            pred2_binary = np.zeros(len(scores2_valid))
            pred2_binary[threshold_mask] = (scores2_valid[threshold_mask] >= 0.4).astype(int)
            pred2_binary[~threshold_mask] = (scores2_valid[~threshold_mask] >= 0.5).astype(int)
        
        contingency_table, statistic, p_value, significance = mcnemar_test_predictions(y_true_valid, pred1_binary, pred2_binary)
        
        mcnemar_results.append({
            'Predictor_1': pred1,
            'Predictor_2': pred2,
            'Sample_Size': sample_size,
            'Statistic': statistic,
            'p_value': p_value,
            'Significance': significance,
            'Both_Correct': contingency_table[0,0],
            'Pred1_Correct_Only': contingency_table[0,1],
            'Pred2_Correct_Only': contingency_table[1,0],
            'Both_Wrong': contingency_table[1,1]
        })
    
    print('Performing DeLong test...')
    delong_results = []
    for pred1, pred2 in itertools.combinations(predictors, 2):
        if pred1 not in combined.columns or pred2 not in combined.columns:
            continue
            
        scores1 = combined[pred1]
        scores2 = combined[pred2]
        
        valid_mask = scores1.notna() & scores2.notna()
        sample_size = np.sum(valid_mask)
        
        if sample_size < 10:
            continue
            
        scores1_valid = scores1[valid_mask].values
        scores2_valid = scores2[valid_mask].values
        y_true_valid = y_true[valid_mask].values
        
        auc1, auc2, p_value, significance = delong_test(y_true_valid, scores1_valid, scores2_valid)
        
        delong_results.append({
            'Predictor_1': pred1,
            'Predictor_2': pred2,
            'Sample_Size': sample_size,
            'AUC_1': auc1,
            'AUC_2': auc2,
            'AUC_Difference': auc1 - auc2,
            'p_value': p_value,
            'Significance': significance
        })
    
    mcnemar_df = pd.DataFrame(mcnemar_results)
    delong_df = pd.DataFrame(delong_results)
    
    mcnemar_df.to_csv('mcnemar_test_all_predictors_436.csv', index=False)
    delong_df.to_csv('delong_test_all_predictors_436.csv', index=False)
    print('Saved CSV results successfully!')
    
    print('Creating heatmaps...')
    create_combined_significance_heatmap(mcnemar_df, delong_df, predictors)
    
    significant_mcnemar = mcnemar_df[mcnemar_df['Significance'] == 'Significant']
    significant_delong = delong_df[delong_df['Significance'] == 'Significant']
    
    print()
    print('STATISTICAL SIGNIFICANCE SUMMARY:')
    # Use format() instead of f-strings to avoid apostrophe issues
    print('McNemar test: {} out of {} pairs significantly different'.format(len(significant_mcnemar), len(mcnemar_df)))
    print('DeLong test: {} out of {} pairs significantly different'.format(len(significant_delong), len(delong_df)))
    
    print()
    print('🎉 ANALYSIS COMPLETE!')

if __name__ == '__main__':
    main()
