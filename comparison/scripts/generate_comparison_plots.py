#!/usr/bin/env python3
"""
Script to generate comparison plots from the all_models_comparison.csv file
Plots:
1. Radar plot with 6 metrics (Sensitivity, Specificity, Accuracy, MCC, Youden's J, ROC-AUC)
2. Calibration plot with all 5 models
3. ROC curve with all 5 models
All plots use matching colors across all visualizations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.calibration import calibration_curve
from scipy.special import expit
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['font.weight'] = 'bold'

def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def load_predictions_for_plots(project_root):
    """Load predictions for all models to generate ROC and calibration plots"""
    from joblib import load
    from functions_for_training import sequences_to_vectors
    
    results = {}
    
    # Load external evaluation data
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))

    # Remove duplicates from original data before processing
    original_antigens_count = len(antigens_df)
    original_non_antigens_count = len(non_antigens_df)

    if antigens_df['Antigen'].duplicated().any():
        antigens_df = antigens_df.drop_duplicates(subset=['Antigen'], keep='first').reset_index(drop=True)
        print(f"Removed {original_antigens_count - len(antigens_df)} duplicate antigens")

    if non_antigens_df['Non-Antigen'].duplicated().any():
        non_antigens_df = non_antigens_df.drop_duplicates(subset=['Non-Antigen'], keep='first').reset_index(drop=True)
        print(f"Removed {original_non_antigens_count - len(non_antigens_df)} duplicate non-antigens")

    print(f"Using {len(antigens_df)} unique antigens and {len(non_antigens_df)} unique non-antigens for ROC plots")
    
    # 1. SVM
    try:
        model_path = os.path.join(project_root, 'svm_training/models')
        model = load(os.path.join(model_path, 'IApred_SVM.joblib'))
        scaler = load(os.path.join(model_path, 'IApred_scaler.joblib'))
        variance_selector = load(os.path.join(model_path, 'IApred_variance_selector.joblib'))
        feature_selector = load(os.path.join(model_path, 'IApred_feature_selector.joblib'))
        all_feature_names = load(os.path.join(model_path, 'IApred_all_feature_names.joblib'))
        
        sequences_ant = [str(s) for s in antigens_df['Antigen'].tolist() if pd.notna(s)]
        sequences_non = [str(s) for s in non_antigens_df['Non-Antigen'].tolist() if pd.notna(s)]
        
        X_ant, fn_ant, failed_ant = sequences_to_vectors(sequences_ant)
        X_non, fn_non, failed_non = sequences_to_vectors(sequences_non)
        
        if len(failed_ant) > 0:
            failed_ant = failed_ant.astype(int)
            antigens_df_svm = antigens_df.drop(antigens_df.index[failed_ant]).reset_index(drop=True)
            valid_indices_ant = np.array([i for i in range(len(sequences_ant)) if i not in failed_ant])
            X_ant = X_ant[valid_indices_ant]
        else:
            antigens_df_svm = antigens_df.copy()
        
        if len(failed_non) > 0:
            failed_non = failed_non.astype(int)
            non_antigens_df_svm = non_antigens_df.drop(non_antigens_df.index[failed_non]).reset_index(drop=True)
            valid_indices_non = np.array([i for i in range(len(sequences_non)) if i not in failed_non])
            X_non = X_non[valid_indices_non]
        else:
            non_antigens_df_svm = non_antigens_df.copy()
        
        # Process and predict
        feature_map_ant = {name: i for i, name in enumerate(fn_ant)}
        feature_map_non = {name: i for i, name in enumerate(fn_non)}
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
        # For SVM: negative scores = antigen, positive = non-antigen
        # Convert to probability-like scores (0-1) where higher = more antigenic
        # Use sigmoid transformation to map decision function to [0,1]
        from scipy.special import expit
        scores_ant_norm = expit(-scores_ant)  # Negative because SVM: negative = antigen
        scores_non_norm = expit(-scores_non)
        
        results['SVM'] = {
            'antigens_scores': scores_ant_norm,
            'non_antigens_scores': scores_non_norm,
            'antigens_df': antigens_df_svm,
            'non_antigens_df': non_antigens_df_svm
        }
    except Exception as e:
        print(f"Error loading SVM: {str(e)}")
    
    # 2-4. TabPFN models
    tabpfn_models = [
        ('TabPFN-all', 'tabpfn_training/models/all_features', 'TabPFN', False, False),
        ('TabPFN-529', 'tabpfn_training/models/529_features', 'TabPFN', False, True),
        ('TabPFN-100', 'tabpfn_training/models/100_features', 'TabPFN', False, True)
    ]
    
    for model_name, model_dir, model_file, exclude_helminths, use_feature_selector in tabpfn_models:
        try:
            model_path = os.path.join(project_root, model_dir)
            model = load(os.path.join(model_path, 'IApred_TabPFN.joblib'))
            variance_selector = load(os.path.join(model_path, 'IApred_variance_selector.joblib'))
            all_feature_names = load(os.path.join(model_path, 'IApred_all_feature_names.joblib'))

            feature_selector = None
            if use_feature_selector:
                feature_selector = load(os.path.join(model_path, 'IApred_feature_selector.joblib'))
            
            sequences_ant = [str(s) for s in antigens_df['Antigen'].tolist() if pd.notna(s)]
            sequences_non = [str(s) for s in non_antigens_df['Non-Antigen'].tolist() if pd.notna(s)]
            
            X_ant, fn_ant, failed_ant = sequences_to_vectors(sequences_ant)
            X_non, fn_non, failed_non = sequences_to_vectors(sequences_non)
            
            if len(failed_ant) > 0:
                failed_ant = failed_ant.astype(int)
                antigens_df_tabpfn = antigens_df.drop(antigens_df.index[failed_ant]).reset_index(drop=True)
                valid_indices_ant = np.array([i for i in range(len(sequences_ant)) if i not in failed_ant])
                X_ant = X_ant[valid_indices_ant]
            else:
                antigens_df_tabpfn = antigens_df.copy()
            
            if len(failed_non) > 0:
                failed_non = failed_non.astype(int)
                non_antigens_df_tabpfn = non_antigens_df.drop(non_antigens_df.index[failed_non]).reset_index(drop=True)
                valid_indices_non = np.array([i for i in range(len(sequences_non)) if i not in failed_non])
                X_non = X_non[valid_indices_non]
            else:
                non_antigens_df_tabpfn = non_antigens_df.copy()
            
            # Exclude helminths if needed
            if exclude_helminths:
                antigens_df_tabpfn = antigens_df_tabpfn.iloc[:-5].copy() if len(antigens_df_tabpfn) >= 5 else antigens_df_tabpfn.copy()
                non_antigens_df_tabpfn = non_antigens_df_tabpfn.iloc[:-5].copy() if len(non_antigens_df_tabpfn) >= 5 else non_antigens_df_tabpfn.copy()
                # Adjust X arrays accordingly
                if len(antigens_df_tabpfn) < X_ant.shape[0]:
                    X_ant = X_ant[:len(antigens_df_tabpfn)]
                if len(non_antigens_df_tabpfn) < X_non.shape[0]:
                    X_non = X_non[:len(non_antigens_df_tabpfn)]
            
            # Align features
            feature_map_ant = {name: i for i, name in enumerate(fn_ant)}
            feature_map_non = {name: i for i, name in enumerate(fn_non)}
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
            
            if variance_selector is not None:
                X_ant_filtered = variance_selector.transform(X_ant_aligned)
                X_non_filtered = variance_selector.transform(X_non_aligned)
            else:
                X_ant_filtered = X_ant_aligned
                X_non_filtered = X_non_aligned
            
            if use_feature_selector:
                X_ant_filtered = feature_selector.transform(X_ant_filtered)
                X_non_filtered = feature_selector.transform(X_non_filtered)
            
            # TabPFN: class 0 = 'antigen', class 1 = 'non-antigen'
            # Get probabilities for class 0 (antigen) - no inversion needed
            # Higher probability of class 0 = more antigenic
            probabilities_ant = model.predict_proba(X_ant_filtered)[:, 0]
            probabilities_non = model.predict_proba(X_non_filtered)[:, 0]
            
            results[model_name] = {
                'antigens_scores': probabilities_ant,
                'non_antigens_scores': probabilities_non,
                'antigens_df': antigens_df_tabpfn,
                'non_antigens_df': non_antigens_df_tabpfn
            }
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
    
    return results

def create_heatmap_plot(df, output_path):
    """Create heatmap matrix showing performance metrics across all models"""
    
    # Metrics for heatmap - all are "higher is better"
    metrics = ['Sensitivity', 'Specificity', 'Accuracy', 'MCC', "Youden's J", 'AUC-ROC']
    
    # Prepare data matrix
    models = df['Model'].tolist()
    data_matrix = df[metrics].values
    
    # Create DataFrame for seaborn heatmap
    heatmap_df = pd.DataFrame(data_matrix, index=models, columns=metrics)
    
    # Determine appropriate vmin/vmax based on data
    # MCC can range from -1 to 1, others typically 0 to 1
    data_min = data_matrix.min()
    data_max = data_matrix.max()
    
    # Check if MCC is in the metrics
    has_mcc = 'MCC' in metrics
    
    # Set reasonable bounds
    if has_mcc:
        # For MCC, allow full range but pad based on data
        vmin = min(-1.0, data_min - 0.1) if data_min < 0 else max(0.0, data_min - 0.05)
    else:
        vmin = max(0.0, data_min - 0.05)
    
    vmax = min(1.0, data_max + 0.05)
    
    # Create figure with better size
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Use a professional colormap - 'RdYlGn_r' (reversed Red-Yellow-Green) 
    # Green = high performance, Red = low performance
    # Alternative: 'YlOrRd' for warm colors, 'viridis' for modern look
    cmap = 'RdYlGn_r'
    
    # Create heatmap using seaborn for better styling
    sns.heatmap(heatmap_df, 
                annot=True,  # Show values
                fmt='.3f',   # Format to 3 decimal places
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                center=None,  # No center for asymmetric ranges
                square=False,
                linewidths=2,  # Grid lines
                linecolor='white',
                cbar_kws={
                    'label': 'Performance Score',
                    'shrink': 0.8,
                    'pad': 0.02,
                    'aspect': 20
                },
                ax=ax,
                annot_kws={
                    'size': 12,
                    'weight': 'bold',
                    'color': 'black'
                })
    
    # Improve colorbar
    cbar = ax.collections[0].colorbar
    cbar.set_label('Performance Score', rotation=270, labelpad=25, fontsize=14, fontweight='bold')
    cbar.ax.tick_params(labelsize=12)
    
    # Improve labels
    ax.set_xlabel('Performance Metrics', fontsize=15, fontweight='bold', labelpad=15)
    ax.set_ylabel('Models', fontsize=15, fontweight='bold', labelpad=15)
    
    # Rotate x-axis labels for better readability
    ax.set_xticklabels(metrics, fontsize=13, fontweight='bold', rotation=45, ha='right')
    ax.set_yticklabels(models, fontsize=13, fontweight='bold', rotation=0)
    
    # Add title
    ax.set_title('Model Performance Comparison Heatmap', 
                 fontsize=18, fontweight='bold', pad=25)
    
    # Highlight best and worst performers for each metric
    for j, metric in enumerate(metrics):
        col_values = data_matrix[:, j]
        max_idx = np.argmax(col_values)
        min_idx = np.argmin(col_values)
        
        # Highlight best performer with a subtle border
        ax.add_patch(plt.Rectangle((j, max_idx), 1, 1, 
                                 fill=False, edgecolor='darkgreen', 
                                 linewidth=2.5, alpha=0.9))
        # Highlight worst performer
        ax.add_patch(plt.Rectangle((j, min_idx), 1, 1, 
                                 fill=False, edgecolor='darkred', 
                                 linewidth=2.5, alpha=0.9))
    
    # Adjust layout
    plt.tight_layout()
    
    # Save figure
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  ✓ Heatmap matrix saved: {output_path}")

def create_calibration_plot(predictions_dict, output_path):
    """Create calibration plot with all models"""
    colors = {
        'SVM': '#1f77b4',
        'TabPFN-all': '#ff7f0e',
        'TabPFN-529': '#2ca02c',
        'TabPFN-100': '#d62728'
    }
    
    # Display name mapping
    display_names = {
        'TabPFN-838': 'TabPFN-all',
        'TabPFN-529': 'TabPFN-529',
        'TabPFN-100': 'TabPFN-100',
        'SVM': 'SVM'
    }

    fig, ax = plt.subplots(figsize=(10, 8))

    for model_name, data in predictions_dict.items():
        display_name = display_names.get(model_name, model_name)
        y_true = np.concatenate([
            np.ones(len(data['antigens_df'])),
            np.zeros(len(data['non_antigens_df']))
        ])
        y_scores = np.concatenate([
            data['antigens_scores'],
            data['non_antigens_scores']
        ])

        prob_true, prob_pred = calibration_curve(y_true, y_scores, n_bins=10)
        ax.plot(prob_pred, prob_true, marker='o', label=f'{display_name}',
                color=colors.get(display_name, 'gray'), linewidth=2, markersize=6)
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Perfect calibration')
    ax.set_xlabel('Mean Predicted Probability', fontsize=14, fontweight='bold')
    ax.set_ylabel('Fraction of Positives', fontsize=14, fontweight='bold')
    ax.set_title('Calibration Curves - All Models', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Calibration plot saved: {output_path}")

def create_roc_plot(predictions_dict, output_path):
    """Create ROC curve plot with all models"""
    colors = {
        'SVM': '#1f77b4',
        'TabPFN-all': '#ff7f0e',
        'TabPFN-529': '#2ca02c',
        'TabPFN-100': '#d62728'
    }
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Display name mapping
    display_names = {
        'TabPFN-838': 'TabPFN-all',
        'TabPFN-529': 'TabPFN-529',
        'TabPFN-100': 'TabPFN-100',
        'SVM': 'SVM'
    }

    for model_name, data in predictions_dict.items():
        display_name = display_names.get(model_name, model_name)
        y_true = np.concatenate([
            np.ones(len(data['antigens_df'])),
            np.zeros(len(data['non_antigens_df']))
        ])
        y_scores = np.concatenate([
            data['antigens_scores'],
            data['non_antigens_scores']
        ])

        fpr, tpr, _ = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{display_name} (AUC = {roc_auc:.3f})',
                color=colors.get(display_name, 'gray'), linewidth=2)
    
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--', linewidth=2, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=14, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=14, fontweight='bold')
    ax.set_title('IApred models ROC curves comparison', fontsize=16, fontweight='bold', pad=15)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  ✓ ROC plot saved: {output_path}")

def main():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    results_dir = os.path.join(project_root, 'comparison/results')
    ensure_dir_exists(results_dir)
    
    # Load comparison CSV
    csv_path = os.path.join(results_dir, 'all_models_comparison.csv')
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found. Please run generate_all_evaluations.py first.")
        return
    
    df = pd.read_csv(csv_path)
    print(f"\nLoaded comparison data: {len(df)} models")
    print(df[['Model', 'AUC-ROC', 'MCC', 'Sensitivity', 'Specificity']].to_string(index=False))
    
    # Create heatmap matrix
    print("\nGenerating heatmap matrix...")
    create_heatmap_plot(df, os.path.join(results_dir, 'performance_heatmap.png'))
    
    # Load predictions for ROC and calibration plots
    print("\nLoading predictions for ROC and calibration plots...")
    predictions_dict = load_predictions_for_plots(project_root)
    
    if predictions_dict:
        # Create calibration plot
        print("\nGenerating calibration plot...")
        create_calibration_plot(predictions_dict, os.path.join(results_dir, 'calibration_comparison.png'))
        
        # Create ROC plot
        print("\nGenerating ROC plot...")
        create_roc_plot(predictions_dict, os.path.join(results_dir, 'roc_comparison.png'))
        
        print(f"\n{'='*80}")
        print("All plots generated successfully!")
        print(f"{'='*80}")
    else:
        print("\n✗ No predictions loaded for plotting!")

if __name__ == "__main__":
    main()

