import pandas as pd
import numpy as np
from joblib import load
import os
import logging
from functions_for_training import sequences_to_vectors
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc,
    confusion_matrix, matthews_corrcoef, brier_score_loss
)
from sklearn.calibration import calibration_curve

def calculate_chi_square(prob_true, prob_pred, epsilon=1e-10):
    valid_mask = prob_pred > epsilon
    if not np.any(valid_mask):
        return 0
    chi_square = np.sum(((prob_true[valid_mask] - prob_pred[valid_mask])**2) / 
                       (prob_pred[valid_mask] + epsilon))
    return chi_square / np.sum(valid_mask)

def classify_vaxijen2(row):
    if row['Class'] in ['Gram+', 'Gram-']:
        return 1 if row['VaxiJen 2.0'] >= 0.4 else 0
    else:
        return 1 if row['VaxiJen 2.0'] >= 0.5 else 0

def process_dataframe(df, is_antigen=True):
    df['ANTIGENpro_pred'] = (df['ANTIGENpro'] >= 0.5).astype(int)
    df['VaxiJen2_pred'] = df.apply(classify_vaxijen2, axis=1)
    # Use the new 'Vaxijen 3.0-all' for the main VaxiJen3 prediction
    df['VaxiJen3_pred'] = (df['Vaxijen 3.0-all'] >= 0.5).astype(int)
    # Additionally compute a predicate for the bacteria-specific column when present
    if 'Vaxijen 3.0-bacteria' in df.columns:
        df['VaxiJen3_bact_pred'] = (df['Vaxijen 3.0-bacteria'] >= 0.5).astype(int)
    else:
        df['VaxiJen3_bact_pred'] = np.nan
    df['IApred_pred'] = (df['IApred'] >= 0).astype(int)
    
    if is_antigen:
        df['predictor_agreements'] = df['ANTIGENpro_pred'] + df['VaxiJen2_pred'] + df['VaxiJen3_pred']
    else:
        df['predictor_agreements'] = 3 - (df['ANTIGENpro_pred'] + df['VaxiJen2_pred'] + df['VaxiJen3_pred'])
    
    df['is_antigen'] = is_antigen
    return df

def update_csv_with_predictions(csv_path, model, scaler, variance_selector, feature_selector, feature_mask, all_feature_names, sequence_column):
    logging.info(f"Attempting to read {os.path.abspath(csv_path)}")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        logging.error(f"File not found: {csv_path}")
        logging.info(f"Current working directory: {os.getcwd()}")
        logging.info(f"Files in directory: {os.listdir('.')}")
        return

    sequences = df[sequence_column].tolist()
    logging.info(f"Processing {len(sequences)} sequences from {sequence_column} column")
    
    X_new, feature_names, failed_indices = sequences_to_vectors(sequences)
    
    feature_map = {name: i for i, name in enumerate(feature_names)}
    X_new_aligned = np.zeros((X_new.shape[0], len(all_feature_names)))
    
    for i, feature in enumerate(all_feature_names):
        if feature in feature_map:
            X_new_aligned[:, i] = X_new[:, feature_map[feature]]
            
    X_new_aligned = np.nan_to_num(X_new_aligned, nan=0.0,
                                 posinf=np.finfo(np.float64).max,
                                 neginf=np.finfo(np.float64).min)
    X_new_aligned = np.clip(X_new_aligned, -1e6, 1e6)
    X_new_var_selected = variance_selector.transform(X_new_aligned)
    X_new_scaled = scaler.transform(X_new_var_selected)
    X_new_selected = feature_selector.transform(X_new_scaled)
    
    scores = model.decision_function(X_new_selected)
    df['IApred'] = -scores
    
    df.to_csv(csv_path, index=False)
    logging.info(f"Updated {csv_path} with predictions")

def create_agreement_plot():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))
    
    antigens_df = process_dataframe(antigens_df, True)
    non_antigens_df = process_dataframe(non_antigens_df, False)
    combined_df = pd.concat([antigens_df, non_antigens_df])

    plt.figure(figsize=(12, 7))
    colors = {
        (True, 3): '#004400',   # Darker green
        (True, 2): '#006400',   # Dark green
        (True, 1): '#3CB371',   # Medium sea green
        (True, 0): '#E8F5E9',   # Very light green
        (False, 3): '#660000',  # Darker red
        (False, 2): '#8B0000',  # Dark red
        (False, 1): '#CD5C5C',  # Indian red
        (False, 0): '#FFEBEE'   # Very light red
    }

    scatter_objects = []
    labels = []
    for is_antigen in [True, False]:
        for agreement in [3, 2, 1, 0]:
            mask = (combined_df['is_antigen'] == is_antigen) & (combined_df['predictor_agreements'] == agreement)
            points = combined_df[mask]
            
            if len(points) > 0:
                x = points['IApred'].values
                kde = gaussian_kde(x)
                density = kde(x)
                y_jitter = np.random.normal(0, 0.1 * density, len(points))
                
                scatter = plt.scatter(points['IApred'], 
                                    y_jitter,
                                    c=colors[(is_antigen, agreement)],
                                    edgecolors='black',
                                    alpha=0.8,
                                    s=75,
                                    label=f'{"Antigens" if is_antigen else "Non-antigens"} ({agreement} predictors)')
                
                scatter_objects.append(scatter)
                labels.append(f'Predicted by {agreement} other predictors')

    plt.title('IApred Prediction Agreement', fontsize=24, fontweight='bold')
    plt.xlabel('IAscore', fontsize=18, fontweight='bold')
    plt.ylabel('')
    plt.yticks([])
    plt.xlim(-2.5, 2.5)
    plt.xticks(np.arange(-2.5, 3.0, 0.5), fontsize=14)
    plt.axvline(x=0, color='black', linestyle=':', alpha=0.5)

    n = len(scatter_objects)//2
    first_column = plt.legend(scatter_objects[:n], labels[:n], 
                            loc='upper center', bbox_to_anchor=(0.25, -0.15),
                            title='Antigens', fontsize=14)
    plt.gca().add_artist(first_column)
    plt.legend(scatter_objects[n:], labels[n:], 
              loc='upper center', bbox_to_anchor=(0.75, -0.15),
              title='Non-antigens', fontsize=14)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    plt.savefig('../results/PredictionAgreement.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_calibration_plot():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))
    
    score_ranges = {
        'IApred': (-2.12, 2),
        'ANTIGENpro': (0, 1),
        'VaxiJen 2.0': (0, 1.5),
        'VaxiJen 3.0-all': (0, 1),
        'VaxiJen 3.0-bacteria': (0, 1)
    }

    def normalize_scores(scores, min_score, max_score):
        return np.clip((scores - min_score) / (max_score - min_score), 0, 1)

    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    ordered_predictors = ['IApred', 'ANTIGENpro', 'VaxiJen 3.0-bacteria', 'VaxiJen 3.0-all', 'VaxiJen 2.0']
    predictors = {
        'IApred': ('IApred', 'IApred_pred'),
        'ANTIGENpro': ('ANTIGENpro', 'ANTIGENpro_pred'),
        'VaxiJen 3.0-all': ('Vaxijen 3.0-all', 'VaxiJen3_pred'),
        'VaxiJen 3.0-bacteria': ('Vaxijen 3.0-bacteria', 'VaxiJen3_bact_pred'),
        'VaxiJen 2.0': ('VaxiJen 2.0', 'VaxiJen2_pred')
    }
    
    # Plot calibration curves
    plt.figure(figsize=(7, 6))
    
    # Colors: keep current green for bacteria, light green for -all
    for predictor_name, color in zip(ordered_predictors, ['blue', 'orange', 'green', 'lightgreen', 'red']):
        score_col, _ = predictors[predictor_name]
        
        # Normalize scores
        min_score, max_score = score_ranges[predictor_name]
        antigens_normalized = normalize_scores(antigens_df[score_col], min_score, max_score)
        non_antigens_normalized = normalize_scores(non_antigens_df[score_col], min_score, max_score)
        
        # Use only rows where this predictor has a value
        ant_mask = antigens_df[score_col].notna()
        non_mask = non_antigens_df[score_col].notna()
        y_true_sub = np.concatenate([
            np.ones(np.sum(ant_mask)),
            np.zeros(np.sum(non_mask))
        ])
        y_scores = np.concatenate([
            antigens_normalized[ant_mask],
            non_antigens_normalized[non_mask]
        ])
        
        if y_scores.size == 0:
            continue
        
        prob_true, prob_pred = calibration_curve(y_true_sub, y_scores, n_bins=10)
        chi_square = calculate_chi_square(prob_true, prob_pred)
        
        plt.plot(prob_pred, prob_true, marker='o', color=color,
                label=f'{predictor_name} (χ² = {chi_square:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlabel('Mean Predicted Probability', fontsize=18, fontweight='bold')
    plt.ylabel('Fraction of Positives', fontsize=18, fontweight='bold')
    plt.title('Calibration Curves for Antigen Predictors', fontsize=24, fontweight='bold')
    plt.legend(loc='lower right', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('../results/CalibrationPlot.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_matrix():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))
    
    antigens_df = process_dataframe(antigens_df, True)
    non_antigens_df = process_dataframe(non_antigens_df, False)
    
    metrics_list = []
    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    
    ordered_predictors = ['IApred', 'ANTIGENpro', 'VaxiJen 3.0-bacteria', 'VaxiJen 3.0-all', 'VaxiJen 2.0']
    predictors = {
        'IApred': ('IApred', 'IApred_pred'),
        'ANTIGENpro': ('ANTIGENpro', 'ANTIGENpro_pred'),
        'VaxiJen 3.0-all': ('Vaxijen 3.0-all', 'VaxiJen3_pred'),
        'VaxiJen 3.0-bacteria': ('Vaxijen 3.0-bacteria', 'VaxiJen3_bact_pred'),
        'VaxiJen 2.0': ('VaxiJen 2.0', 'VaxiJen2_pred')
    }
    
    for predictor_name in ordered_predictors:
        score_col, pred_col = predictors[predictor_name]

        # Build masks where this predictor has a real score and prediction
        ant_mask = antigens_df[score_col].notna()
        non_mask = non_antigens_df[score_col].notna()
        # Some prediction columns may be NaN as well (e.g., bacteria). Require both present
        if pred_col in antigens_df.columns:
            ant_mask = ant_mask & antigens_df[pred_col].notna()
        if pred_col in non_antigens_df.columns:
            non_mask = non_mask & non_antigens_df[pred_col].notna()

        # Skip if no data
        if not (np.any(ant_mask) or np.any(non_mask)):
            continue
        
        # Get predictions
        y_pred = np.concatenate([
            antigens_df.loc[ant_mask, pred_col].astype(int),
            non_antigens_df.loc[non_mask, pred_col].astype(int)
        ])
        
        # Get and normalize scores using dynamic ranges from available values only
        all_scores = np.concatenate([
            antigens_df.loc[ant_mask, score_col].astype(float).values,
            non_antigens_df.loc[non_mask, score_col].astype(float).values
        ])
        min_score, max_score = np.min(all_scores), np.max(all_scores)
        # Avoid zero division
        if max_score == min_score:
            max_score = min_score + 1e-12
        
        antigens_normalized = (antigens_df.loc[ant_mask, score_col] - min_score) / (max_score - min_score)
        non_antigens_normalized = (non_antigens_df.loc[non_mask, score_col] - min_score) / (max_score - min_score)
        y_scores = np.concatenate([
            antigens_normalized.values,
            non_antigens_normalized.values
        ])
        y_scores = np.clip(y_scores, 0, 1)

        # Build matching ground truth labels for the filtered rows
        y_true_sub = np.concatenate([
            np.ones(np.sum(ant_mask)),
            np.zeros(np.sum(non_mask))
        ])
        
        # Calculate metrics (unchanged)
        tn, fp, fn, tp = confusion_matrix(y_true_sub, y_pred).ravel()
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = precision_score(y_true_sub, y_pred)
        recall = recall_score(y_true_sub, y_pred)
        f1 = f1_score(y_true_sub, y_pred)
        f05 = (1 + 0.5**2) * (precision * recall) / ((0.5**2 * precision) + recall)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        j_index = sensitivity + specificity - 1
        fdr = fp / (fp + tp) if (fp + tp) > 0 else 0
        mcc = matthews_corrcoef(y_true_sub, y_pred)
        fpr_curve, tpr_curve, _ = roc_curve(y_true_sub, y_scores)
        roc_auc = auc(fpr_curve, tpr_curve)
        prob_true, prob_pred = calibration_curve(y_true_sub, y_scores, n_bins=10)
        chi_square = calculate_chi_square(prob_true, prob_pred)
        brier = brier_score_loss(y_true_sub, y_scores)
        bin_boundaries = np.linspace(0, 1, 11)
        bin_indices = np.digitize(y_scores, bin_boundaries) - 1
        n_bins = len(bin_boundaries) - 1
        ece = 0
        valid_bins = 0
        for bin_idx in range(n_bins):
            mask = bin_indices == bin_idx
            if np.any(mask):
                bin_conf = np.mean(y_scores[mask])
                bin_acc = np.mean(y_true_sub[mask])
                ece += np.abs(bin_acc - bin_conf)
                valid_bins += 1
        ece = ece / valid_bins if valid_bins > 0 else 0
        metrics_list.append({
            'Metric': predictor_name,
            'True Positive': tp,
            'True Negative': tn,
            'False Positive': fp,
            'False Negative': fn,
            'Accuracy': accuracy_score(y_true_sub, y_pred),
            'Precision': precision,
            'Sensitivity': sensitivity,
            'Specificity': specificity,
            'FPR': fpr,
            'FDR': fdr,
            'MCC': mcc,
            "Youden's J": j_index,
            'F1-score': f1,
            'F0.5-score': f05,
            'Brier Score': brier,
            'ECE': ece,
            'AUC-ROC': roc_auc,
            'Chi-Square': chi_square,
            'Score Range': f'{min_score:.3f} to {max_score:.3f}'
        })
    metrics_df = pd.DataFrame(metrics_list)
    metrics_df = metrics_df.set_index('Metric').transpose()
    metrics_df = metrics_df[ordered_predictors]
    metrics_df.round(3).to_csv('../results/PerformanceMetrics.csv')

def create_roc_comparison():
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_antigens.csv'))
    non_antigens_df = pd.read_csv(os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'))

    # Process dataframes
    antigens_df = process_dataframe(antigens_df, True)
    non_antigens_df = process_dataframe(non_antigens_df, False)
    
    # Create true labels and combine predictions
    y_true = np.concatenate([np.ones(len(antigens_df)), np.zeros(len(non_antigens_df))])
    
    predictors = {
        'IApred': ('IApred', 'IApred_pred'),
        'ANTIGENpro': ('ANTIGENpro', 'ANTIGENpro_pred'),
        'VaxiJen 3.0-all': ('Vaxijen 3.0-all', 'VaxiJen3_pred'),
        'VaxiJen 3.0-bacteria': ('Vaxijen 3.0-bacteria', 'VaxiJen3_bact_pred'),
        'VaxiJen 2.0': ('VaxiJen 2.0', 'VaxiJen2_pred')
    }
    
    ordered_predictors = ['IApred', 'ANTIGENpro', 'VaxiJen 3.0-bacteria', 'VaxiJen 3.0-all', 'VaxiJen 2.0']

    # Plot ROC curves
    plt.figure(figsize=(9, 8))
    # Colors: keep current green for bacteria, light green for -all
    colors = ['purple', 'blue', 'green', 'lightgreen', 'red']

    for predictor_name, color in zip(ordered_predictors, colors):
        score_col, _ = predictors[predictor_name]
        
        # Use only rows where this predictor has a value
        ant_mask = antigens_df[score_col].notna()
        non_mask = non_antigens_df[score_col].notna()
        if not (np.any(ant_mask) or np.any(non_mask)):
            continue
        
        y_scores = np.concatenate([
            antigens_df.loc[ant_mask, score_col].fillna(0),
            non_antigens_df.loc[non_mask, score_col].fillna(0)
        ])
        y_true_sub = np.concatenate([
            np.ones(np.sum(ant_mask)),
            np.zeros(np.sum(non_mask))
        ])
        
        fpr, tpr, _ = roc_curve(y_true_sub, y_scores)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=color,
                label=f'{predictor_name} (AUC = {roc_auc:.3f})')

    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    plt.ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    plt.title('ROC Curves for Different Antigen Predictors', fontsize=24, fontweight='bold')
    plt.legend(loc="lower right", fontsize=14)
    plt.grid(True)
    
    plt.savefig('../results/PredictorROCcomparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    logging.info(f"Starting script from {os.getcwd()}")
    
    models_dir = 'models'
    try:
        svm_model = load(os.path.join(models_dir, 'IApred_SVM.joblib'))
        scaler = load(os.path.join(models_dir, 'IApred_scaler.joblib'))
        variance_selector = load(os.path.join(models_dir, 'IApred_variance_selector.joblib'))
        feature_selector = load(os.path.join(models_dir, 'IApred_feature_selector.joblib'))
        feature_mask = load(os.path.join(models_dir, 'IApred_feature_mask.joblib'))
        all_feature_names = load(os.path.join(models_dir, 'IApred_all_feature_names.joblib'))
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        return

    files_to_process = [
        (os.path.join(project_root, 'data/External_evaluation_antigens.csv'), 'Antigen'),
        (os.path.join(project_root, 'data/External_evaluation_non-antigens.csv'), 'Non-Antigen')
    ]

    for file_path, sequence_column in files_to_process:
        update_csv_with_predictions(file_path, 
                                  svm_model, scaler, variance_selector, 
                                  feature_selector, feature_mask, 
                                  all_feature_names, sequence_column)
    
    create_agreement_plot()
    create_roc_comparison()
    create_calibration_plot()
    create_performance_matrix()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
