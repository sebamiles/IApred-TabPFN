import os
import sys
import numpy as np
from Bio import SeqIO

# Add shared directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from functions_for_training import sequences_to_vectors, remove_constant_features
from Bio import SeqIO

def read_fasta(file_path):
    sequences = []
    amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
    with open(file_path, 'r') as file:
        for record in SeqIO.parse(file, "fasta"):
            cleaned_sequence = ''.join(aa for aa in str(record.seq).upper() if aa in amino_acids)
            if cleaned_sequence:
                sequences.append(cleaned_sequence)
    return sequences

def get_data_paths():
    """Get paths to antigens and non-antigens directories relative to project root"""
    # Get project root (IApred-PFN directory)
    project_root = os.path.dirname(os.path.abspath(__file__))
    antigens_dir = os.path.join(project_root, 'antigens')
    non_antigens_dir = os.path.join(project_root, 'non-antigens')
    return antigens_dir, non_antigens_dir

def load_training_data():
    """Load all training sequences from antigens and non-antigens directories"""
    antigens_dir, non_antigens_dir = get_data_paths()
    
    antigen_files = [os.path.join(antigens_dir, f) for f in os.listdir(antigens_dir) if f.endswith('.fasta')]
    non_antigen_files = [os.path.join(non_antigens_dir, f) for f in os.listdir(non_antigens_dir) if f.endswith('.fasta')]
    
    print("Reading antigen files...")
    antigens = []
    for file_name in antigen_files:
        try:
            sequences = read_fasta(file_name)
            antigens.extend(sequences)
            print(f"  Loaded {len(sequences)} sequences from {os.path.basename(file_name)}")
        except Exception as e:
            print(f"Warning: Could not read file {file_name}: {str(e)}")
    
    print(f"\nReading non-antigen files...")
    non_antigens = []
    for file_name in non_antigen_files:
        try:
            sequences = read_fasta(file_name)
            non_antigens.extend(sequences)
            print(f"  Loaded {len(sequences)} sequences from {os.path.basename(file_name)}")
        except Exception as e:
            print(f"Warning: Could not read file {file_name}: {str(e)}")
    
    print(f"\nTotal sequences: {len(antigens)} antigens, {len(non_antigens)} non-antigens")
    
    all_sequences = antigens + non_antigens
    labels = np.array(['antigen'] * len(antigens) + ['non-antigen'] * len(non_antigens))
    
    return all_sequences, labels, antigen_files, non_antigen_files

def load_and_extract_features():
    """Load sequences and extract features"""
    all_sequences, labels, antigen_files, non_antigen_files = load_training_data()
    
    print("\nExtracting features...")
    X, feature_names, failed_indices = sequences_to_vectors(all_sequences)
    
    if len(failed_indices) > 0:
        failed_indices = failed_indices.astype(int)
        labels = np.delete(labels, failed_indices)
        print(f"Removed {len(failed_indices)} sequences with failed feature extraction")
    
    print("Filtering constant features...")
    X_filtered, feature_mask, feature_names_filtered = remove_constant_features(X, feature_names)
    
    return X_filtered, labels, feature_names_filtered, antigen_files, non_antigen_files

def read_external_evaluation_data():
    """Load external evaluation CSV files"""
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    antigens_csv = os.path.join(project_root, 'External_evaluation_antigens.ods')
    non_antigens_csv = os.path.join(project_root, 'External_evaluation_non-antigens.ods')
    
    # Try CSV first, then ODS
    import pandas as pd
    try:
        if os.path.exists(antigens_csv.replace('.ods', '.csv')):
            antigens_df = pd.read_csv(antigens_csv.replace('.ods', '.csv'))
        else:
            antigens_df = pd.read_excel(antigens_csv, engine='odf')
    except:
        antigens_df = None
    
    try:
        if os.path.exists(non_antigens_csv.replace('.ods', '.csv')):
            non_antigens_df = pd.read_csv(non_antigens_csv.replace('.ods', '.csv'))
        else:
            non_antigens_df = pd.read_excel(non_antigens_csv, engine='odf')
    except:
        non_antigens_df = None
    
    return antigens_df, non_antigens_df

