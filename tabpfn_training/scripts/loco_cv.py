import os
import argparse
import numpy as np
import pickle
import joblib
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../'))
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from Bio import SeqIO
from functions_for_training import sequences_to_vectors, remove_constant_features
from tabpfn import TabPFNClassifier
import torch

def get_pathogen_classes():
    return {
        'gram-': [
            'A_pleuropneumoniae',
            'B_pertussis',
            'C_pneumoniae'
        ],
        'gram+': [
            'B_anthracis',
            'C_pseudotuberculosis',
            'M_bovis',
            'S_aureus',
            'S_pseudintermedius',
            'T_whipplei'
        ],
        'fungi': [
            'C_albicans',
            'A_fumigatus',
            'C_gattii',
            'C_neoformans',
            'C_posadasii',
            'P_brasiliensis',
            'T_marneffei',
            'H_capsulatum'
        ],
        'protozoa': [
            'P_vivax',
            'T_cruzi',
            'T_gondii'
        ],
        'helminth': [
            'A_suum',
            'E_granulosus',
            'F_hepatica',
            'S_mansoni',
            'T_spp'
        ]
    }


def ensure_dir_exists(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def get_class_splits(category_a_files, category_b_files):
    # Always include these files in training
    always_include = ['antigensP.fasta', 'non-antigensP.fasta',
                     'nr_virus_antigens.fasta', 'nr_virus_non-antigens.fasta']

    pathogen_classes = get_pathogen_classes()
    splits = []
    for class_name, class_pathogens in pathogen_classes.items():
        test_files = []
        for pathogen in class_pathogens:
            antigen_file = os.path.join("antigens", f"{pathogen}_antigens.fasta")
            non_antigen_file = os.path.join("non-antigens", f"{pathogen}_non-antigens.fasta")
            if os.path.basename(antigen_file) in [os.path.basename(f) for f in category_a_files]:
                test_files.extend([antigen_file, non_antigen_file])


        train_files = {
            'antigens': [f for f in category_a_files if os.path.basename(f) not in [os.path.basename(tf) for tf in test_files]],
            'non-antigens': [f for f in category_b_files if os.path.basename(f) not in [os.path.basename(tf) for tf in test_files]]
        }

        splits.append((train_files, test_files, class_name))

    return splits


def read_multiple_fasta(file_list):
    sequences = []
    sequence_ids = []

    for file_path in file_list:
        try:
            for record in SeqIO.parse(file_path, "fasta"):
                sequences.append(str(record.seq))
                sequence_ids.append(f"{file_path}|{record.id}")
        except Exception as e:
            print(f"Error reading file {file_path}: {str(e)}")

    return sequences, sequence_ids


def cross_validate_model_loco(X, y, model, category_a_files, category_b_files, sequence_ids, results_dir):
    splits = get_class_splits(category_a_files, category_b_files)
    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, 100)
    class_results = {}

    fig = plt.figure(figsize=(10, 8))
    ax = plt.subplot(111)

    for train_files, test_files, class_name in splits:

        train_indices = []
        test_indices = []

        test_pathogens = set()
        for test_file in test_files:
            pathogen = os.path.basename(test_file).split('_antigens.fasta')[0]
            if pathogen:
                test_pathogens.add(pathogen)
            pathogen = os.path.basename(test_file).split('_non-antigens.fasta')[0]
            if pathogen:
                test_pathogens.add(pathogen)

        for idx, seq_id in enumerate(sequence_ids):
            is_test = False
            for pathogen in test_pathogens:
                if f"/{pathogen}_" in seq_id:
                    test_indices.append(idx)
                    is_test = True
                    break

            if not is_test:
                train_indices.append(idx)

        train_indices = np.array(train_indices)
        test_indices = np.array(test_indices)

        if len(test_indices) == 0:
            continue

        print(f"\nClass: {class_name}")
        print(f"Training set - Total: {len(train_indices)}, "
              f"Antigens: {np.sum(y[train_indices] == 0)}, "
              f"Non-antigens: {np.sum(y[train_indices] == 1)}")
        print(f"Testing set - Total: {len(test_indices)}, "
              f"Antigens: {np.sum(y[test_indices] == 0)}, "
              f"Non-antigens: {np.sum(y[test_indices] == 1)}")

        unique_classes = np.unique(y[test_indices])
        if len(unique_classes) < 2:
            print(f"Warning: Test set for {class_name} does not contain both classes. Skipping.")
            continue


        model.fit(X[train_indices], y[train_indices])
        y_pred_proba = model.predict_proba(X[test_indices])[:, 0]  # TabPFN antigen class (inverted)
        y_pred_proba = 1 - y_pred_proba  # Invert scores so higher = more antigenic
        y_test = y[test_indices]
        y_test_roc = 1 - y_test  # Invert labels: 1 for antigens (positive class), 0 for non-antigens
        try:
            fpr, tpr, _ = roc_curve(y_test_roc, y_pred_proba)
            roc_auc = auc(fpr, tpr)
            if not np.isnan(roc_auc):
                tprs.append(np.interp(mean_fpr, fpr, tpr))
                tprs[-1][0] = 0.0
                aucs.append(roc_auc)
                class_results[class_name] = {
                    'auc': roc_auc,
                    'n_samples': len(test_indices),
                    'n_antigens': np.sum(y_test == 0),
                    'n_non_antigens': np.sum(y_test == 1)
                }
                ax.plot(fpr, tpr, alpha=0.7,
                       label=f'{class_name.capitalize()} (AUC = {roc_auc:.2f}, n={len(test_indices)})')

        except Exception as e:
            print(f"Warning: Could not calculate ROC curve for {class_name}: {str(e)}")
            continue

    if len(aucs) == 0:
        print("No valid AUC scores were calculated.")
        return None, None, class_results

    # Plot mean ROC curve
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    ax.plot(mean_fpr, mean_tpr, color='b',
            label=r'Mean ROC (AUC = %0.2f ± %0.2f)' % (mean_auc, std_auc),
            lw=2)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    ax.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2)
    ax.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r', alpha=.8)
    ax.set_xlabel('False Positive Rate', fontsize=18, fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontsize=18, fontweight='bold')
    ax.set_title('Leave-One-Class-Out Cross-Validation', fontsize=24, fontweight='bold', pad=20)

    ax.legend(loc='lower right',
             fontsize=14,
             frameon=True,
             facecolor='white',
             edgecolor='gray',
             framealpha=0.8)

    ax.grid(True, linestyle='--', alpha=0.3)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.tick_params(axis='both', which='major', labelsize=14)

    # Save plot
    plot_path = os.path.join(results_dir, 'loco_cv_roc.png')
    plt.savefig(plot_path, bbox_inches='tight', dpi=300)
    print(f"ROC curve has been saved in {plot_path}")
    plt.close()
    print("\nClass-specific results:")
    for class_name, results in sorted(class_results.items()):
        print(f"{class_name.capitalize()}: AUC = {results['auc']:.3f} "
              f"(n={results['n_samples']}, antigens={results['n_antigens']}, "
              f"non-antigens={results['n_non_antigens']})")

    return mean_auc, std_auc, class_results


def load_tabpfn_model_and_preprocessing(model_dir, k):
    """Load TabPFN model and preprocessing components"""
    print(f"Loading TabPFN model from {model_dir}")

    # Load the model
    model_path = os.path.join(model_dir, 'IApred_TabPFN.joblib')
    model = joblib.load(model_path)

    # Load preprocessing components
    variance_selector_path = os.path.join(model_dir, 'IApred_variance_selector.joblib')
    feature_selector_path = os.path.join(model_dir, 'IApred_feature_selector.joblib')

    variance_selector = joblib.load(variance_selector_path) if os.path.exists(variance_selector_path) else None
    feature_selector = joblib.load(feature_selector_path) if os.path.exists(feature_selector_path) else None

    return model, variance_selector, feature_selector


def main():
    parser = argparse.ArgumentParser(description='Perform LOCO-CV for TabPFN model')
    parser.add_argument('--k', type=str, default='all',
                      help='Number of features to use (default: all, options: all, 529, 100)')
    args = parser.parse_args()

    model_suffix = args.k if args.k != 'all' else 'all_features'
    results_dir = f'../results/{model_suffix}'
    ensure_dir_exists(results_dir)


    # Get the correct path to data directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    antigens_dir = os.path.join(project_root, "antigens")
    non_antigens_dir = os.path.join(project_root, "non-antigens")

    category_a_files = [f for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    category_b_files = [f for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]

    category_a_files = [os.path.join(antigens_dir, f) for f in category_a_files]
    category_b_files = [os.path.join(non_antigens_dir, f) for f in category_b_files]

    try:
        print("Reading sequences from FASTA files...")
        category_a_sequences, category_a_ids = read_multiple_fasta(category_a_files)
        category_b_sequences, category_b_ids = read_multiple_fasta(category_b_files)
        print(f"Total sequences in Category A (antigens): {len(category_a_sequences)}")
        print(f"Total sequences in Category B (non-antigens): {len(category_b_sequences)}")

        all_sequences = category_a_sequences + category_b_sequences
        sequence_ids = category_a_ids + category_b_ids
        labels = np.array(['antigens'] * len(category_a_sequences) +
                        ['non-antigens'] * len(category_b_sequences))

        print("\nExtracting sequence features...")
        X, feature_names, failed_indices = sequences_to_vectors(all_sequences)
        print(f"Initial feature matrix shape: {X.shape}")

        # Remove failed sequences
        if len(failed_indices) > 0:
            print(f"Removing {len(failed_indices)} sequences that failed feature extraction")
            mask = np.ones(len(labels), dtype=bool)
            mask[failed_indices] = False
            labels = labels[mask]
            sequence_ids = [sid for i, sid in enumerate(sequence_ids) if i not in failed_indices]
            print(f"Feature matrix shape after removing failed sequences: {X.shape}")

        print("\nPreparing for model evaluation...")

        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)
        print(f"Unique labels: {le.classes_}")
        print(f"Label counts: {np.bincount(labels_encoded)}")

        print("\nFiltering constant features...")
        X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)
        print(f"Features after removing constant features: {X_filtered.shape[1]}")

        # Apply feature selection based on k
        if args.k == 'all':
            X_selected = X_filtered
            selected_feature_names = feature_names_filtered
            print("Using all features (no additional selection)")
        else:
            k = int(args.k)
            print(f"\nPerforming feature selection (k={k})...")
            selector = SelectKBest(f_classif, k=k)
            X_selected = selector.fit_transform(X_filtered, labels_encoded)
            selected_feature_mask = selector.get_support()
            selected_feature_names = [name for name, selected in zip(feature_names_filtered, selected_feature_mask) if selected]
            print(f"Features after selection: {X_selected.shape[1]}")

        # Load TabPFN model and preprocessing
        model_dir = f'../models/{model_suffix}'
        model, variance_selector, feature_selector = load_tabpfn_model_and_preprocessing(model_dir, args.k)

        # Apply the same preprocessing as used during training
        if variance_selector is not None:
            X_selected = variance_selector.transform(X_selected)
            print("Applied variance threshold filtering")

        if feature_selector is not None and args.k != 'all':
            X_selected = feature_selector.transform(X_selected)
            print("Applied feature selection")

        print("\nEvaluating model with LOCO-CV...")
        mean_auc, std_auc, class_results = cross_validate_model_loco(
            X_selected, labels_encoded, model,
            category_a_files, category_b_files, sequence_ids, results_dir
        )

        if mean_auc is not None:
            print(f"\nOverall LOCO-CV ROC AUC: {mean_auc:.4f} (±{std_auc:.4f})")
        else:
            print("\nCould not calculate overall ROC AUC due to insufficient data")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise

if __name__ == "__main__":
    main()
