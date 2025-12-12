#!/usr/bin/env python3
"""
IApred Prediction Script

This script allows users to predict antigenicity of protein sequences using
pre-trained SVM or TabPFN models.

Usage:
    python IApred.py --input sequences.fasta --model svm --output predictions.csv
    python IApred.py --input sequences.fasta --model tabpfn --output predictions.csv
    python IApred.py --input sequences.fasta --model both --output comparison.csv
    python IApred.py  # Interactive mode
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
from joblib import load
from Bio import SeqIO
import logging
from prettytable import PrettyTable

# Add current directory to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from functions_for_training import sequences_to_vectors

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_single_sequence(input_text):
    """Parse a single sequence that may have a header and/or be multi-line"""
    lines = input_text.strip().split('\n')
    
    # Remove empty lines
    lines = [line.strip() for line in lines if line.strip()]
    
    if not lines:
        return None, None
    
    # Check if first line is a header (starts with >)
    if lines[0].startswith('>'):
        sequence_id = lines[0][1:].strip() if len(lines[0]) > 1 else "sequence_1"
        # Combine remaining lines as sequence
        sequence = ''.join(lines[1:])
    else:
        sequence_id = "sequence_1"
        # Combine all lines as sequence
        sequence = ''.join(lines)
    
    # Remove any whitespace from sequence
    sequence = ''.join(sequence.split())
    
    return sequence, sequence_id

def read_sequences_from_file(file_path):
    """Read sequences from FASTA or text file, handling multi-line sequences"""
    sequences = []
    sequence_ids = []
    
    # Try FASTA format first (handles multi-line sequences automatically)
    try:
        with open(file_path, 'r') as f:
            for record in SeqIO.parse(f, "fasta"):
                sequences.append(str(record.seq))
                sequence_ids.append(record.id)
        if sequences:
            logging.info(f"Read {len(sequences)} sequences from FASTA file")
            return sequences, sequence_ids
    except:
        pass
    
    # Try text file (one sequence per line, or multi-line FASTA-like)
    try:
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Try to parse as multi-line FASTA
        current_id = None
        current_seq = []
        
        for line in content.split('\n'):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_seq:
                    sequences.append(''.join(current_seq))
                    sequence_ids.append(current_id if current_id else f"seq_{len(sequences)}")
                # Start new sequence
                current_id = line[1:].strip() if len(line) > 1 else None
                current_seq = []
            else:
                # Add to current sequence
                current_seq.append(line)
        
        # Save last sequence
        if current_seq:
            sequences.append(''.join(current_seq))
            sequence_ids.append(current_id if current_id else f"seq_{len(sequences)}")
        
        if sequences:
            logging.info(f"Read {len(sequences)} sequences from text file")
            return sequences, sequence_ids
        
        # Fallback: one sequence per line
        for idx, line in enumerate(content.split('\n')):
            line = line.strip()
            if line and not line.startswith('#'):
                sequences.append(line)
                sequence_ids.append(f"seq_{idx+1}")
        
        if sequences:
            logging.info(f"Read {len(sequences)} sequences from text file (one per line)")
            return sequences, sequence_ids
            
    except Exception as e:
        logging.error(f"Error reading file: {str(e)}")
        return [], []
    
    return [], []

def load_svm_model(model_dir):
    """Load SVM model and preprocessing components"""
    model_path = os.path.join(model_dir, 'IApred_SVM.joblib')
    scaler_path = os.path.join(model_dir, 'IApred_scaler.joblib')
    variance_selector_path = os.path.join(model_dir, 'IApred_variance_selector.joblib')
    feature_selector_path = os.path.join(model_dir, 'IApred_feature_selector.joblib')
    all_feature_names_path = os.path.join(model_dir, 'IApred_all_feature_names.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"SVM model not found at {model_path}")
    
    model = load(model_path)
    scaler = load(scaler_path)
    variance_selector = load(variance_selector_path)
    feature_selector = load(feature_selector_path)
    all_feature_names = load(all_feature_names_path)
    
    logging.info("SVM model loaded successfully")
    return model, scaler, variance_selector, feature_selector, all_feature_names

def load_tabpfn_model(model_dir):
    """Load TabPFN model and preprocessing components"""
    model_path = os.path.join(model_dir, 'IApred_TabPFN.joblib')
    variance_selector_path = os.path.join(model_dir, 'IApred_variance_selector.joblib')
    all_feature_names_path = os.path.join(model_dir, 'IApred_all_feature_names.joblib')
    feature_selector_path = os.path.join(model_dir, 'IApred_feature_selector.joblib')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"TabPFN model not found at {model_path}")
    
    model = load(model_path)
    variance_selector = load(variance_selector_path) if os.path.exists(variance_selector_path) else None
    all_feature_names = load(all_feature_names_path)
    feature_selector = load(feature_selector_path) if os.path.exists(feature_selector_path) else None
    
    logging.info("TabPFN model loaded successfully")
    return model, variance_selector, feature_selector, all_feature_names

def predict_svm(sequences, model_dir):
    """Make predictions using SVM model"""
    model, scaler, variance_selector, feature_selector, all_feature_names = load_svm_model(model_dir)
    
    logging.info("Extracting features from sequences...")
    X, feature_names, failed_indices = sequences_to_vectors(sequences)
    
    if len(failed_indices) > 0:
        logging.warning(f"Failed to extract features from {len(failed_indices)} sequences")
        # Remove failed sequences
        valid_indices = [i for i in range(len(sequences)) if i not in failed_indices]
        sequences = [sequences[i] for i in valid_indices]
        X = X[valid_indices]
    
    # Align features with model's expected features
    feature_map = {name: i for i, name in enumerate(feature_names)}
    X_aligned = np.zeros((X.shape[0], len(all_feature_names)))
    
    for i, feature in enumerate(all_feature_names):
        if feature in feature_map:
            X_aligned[:, i] = X[:, feature_map[feature]]
    
    # Apply preprocessing
    X_processed = variance_selector.transform(X_aligned)
    X_scaled = scaler.transform(X_processed)
    X_selected = feature_selector.transform(X_scaled)
    
    # Make predictions
    logging.info("Making predictions...")
    scores = model.decision_function(X_selected)
    
    # Convert decision function scores to probabilities [0, 1]
    # Using sigmoid transformation
    probabilities = 1 / (1 + np.exp(-scores))
    
    return probabilities, sequences

def predict_tabpfn(sequences, model_dir):
    """Make predictions using TabPFN model"""
    model, variance_selector, feature_selector, all_feature_names = load_tabpfn_model(model_dir)
    
    logging.info("Extracting features from sequences...")
    X, feature_names, failed_indices = sequences_to_vectors(sequences)
    
    if len(failed_indices) > 0:
        logging.warning(f"Failed to extract features from {len(failed_indices)} sequences")
        # Remove failed sequences
        valid_indices = [i for i in range(len(sequences)) if i not in failed_indices]
        sequences = [sequences[i] for i in valid_indices]
        X = X[valid_indices]
    
    # Align features with model's expected features
    feature_map = {name: i for i, name in enumerate(feature_names)}
    X_aligned = np.zeros((X.shape[0], len(all_feature_names)))
    
    for i, feature in enumerate(all_feature_names):
        if feature in feature_map:
            X_aligned[:, i] = X[:, feature_map[feature]]
    
    # Handle NaN and infinite values
    X_aligned = np.nan_to_num(X_aligned, nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
    X_aligned = np.clip(X_aligned, -1e6, 1e6)
    
    # Apply preprocessing
    if variance_selector is not None:
        X_processed = variance_selector.transform(X_aligned)
    else:
        X_processed = X_aligned
    
    if feature_selector is not None:
        X_final = feature_selector.transform(X_processed)
    else:
        X_final = X_processed
    
    # Make predictions
    logging.info("Making predictions...")
    # TabPFN: class 0 = 'antigen', class 1 = 'non-antigen'
    # We want antigens to have higher scores, so use class 0 probability
    probabilities = model.predict_proba(X_final)[:, 0]
    # Invert so higher = more antigenic (consistent with SVM)
    probabilities = 1 - probabilities
    
    return probabilities, sequences

def interactive_mode():
    """Interactive mode for user-friendly prediction"""
    print("\n" + "="*60)
    print("  IApred: Interactive Antigenicity Prediction")
    print("="*60 + "\n")
    
    # Ask for input type
    while True:
        print("What would you like to predict?")
        print("  1. Single sequence")
        print("  2. FASTA file (multiple sequences)")
        choice = input("\nEnter your choice (1 or 2): ").strip()
        
        if choice in ['1', '2']:
            break
        print("Invalid choice. Please enter 1 or 2.\n")
    
    sequences = []
    sequence_ids = []
    
    if choice == '1':
        # Single sequence input
        print("\n" + "-"*60)
        print("Single Sequence Input")
        print("-"*60)
        print("You can paste:")
        print("  - Just the sequence (can be on multiple lines, 60-100 aa per line)")
        print("  - Sequence with FASTA header (starting with >)")
        print("\nPaste your sequence below:")
        print("(You can paste multi-line sequences. Press Ctrl+D (Linux/Mac) or Ctrl+Z then Enter (Windows) when done)")
        print("Or press Enter twice on an empty line to finish:\n")
        
        lines = []
        empty_count = 0
        try:
            while True:
                try:
                    line = input()
                    if line.strip() == '':
                        empty_count += 1
                        if empty_count >= 2 and lines:
                            break
                    else:
                        empty_count = 0
                        lines.append(line)
                except (EOFError, KeyboardInterrupt):
                    if lines:
                        break
                    else:
                        print("\nNo input received. Exiting.")
                        return
        except KeyboardInterrupt:
            print("\n\nInput cancelled. Exiting.")
            return
        
        if not lines:
            print("Error: No input received.")
            return
        
        input_text = '\n'.join(lines)
        sequence, seq_id = parse_single_sequence(input_text)
        
        if not sequence:
            print("Error: No valid sequence found in input.")
            return
        
        if len(sequence) < 10:
            print(f"Warning: Sequence is very short ({len(sequence)} amino acids). Results may be less reliable.")
        
        sequences = [sequence]
        sequence_ids = [seq_id]
        print(f"\n✓ Sequence loaded: {len(sequence)} amino acids")
        print(f"  Sequence ID: {seq_id}")
        
    else:
        # FASTA file input
        print("\n" + "-"*60)
        print("FASTA File Input")
        print("-"*60)
        while True:
            file_path = input("Enter the path to your FASTA file: ").strip()
            
            # Remove quotes if present
            if file_path.startswith('"') and file_path.endswith('"'):
                file_path = file_path[1:-1]
            elif file_path.startswith("'") and file_path.endswith("'"):
                file_path = file_path[1:-1]
            
            if os.path.exists(file_path):
                sequences, sequence_ids = read_sequences_from_file(file_path)
                if sequences:
                    print(f"\n✓ Loaded {len(sequences)} sequence(s) from file")
                    break
                else:
                    print("Error: No valid sequences found in file. Please try again.")
            else:
                print(f"Error: File not found: {file_path}")
                retry = input("Try again? (y/n): ").strip().lower()
                if retry != 'y':
                    return
    
    if not sequences:
        print("Error: No sequences to predict.")
        return
    
    # Ask for model selection
    print("\n" + "-"*60)
    print("Model Selection")
    print("-"*60)
    print("Available options:")
    print("  1. SVM (Support Vector Machine)")
    print("  2. TabPFN - All features (recommended, best performance)")
    print("  3. Both models (compare predictions)")
    
    while True:
        model_choice = input("\nSelect option (1-3): ").strip()
        if model_choice == '1':
            model = 'svm'
            use_both = False
            break
        elif model_choice == '2':
            model = 'tabpfn'
            use_both = False
            break
        elif model_choice == '3':
            model = 'both'
            use_both = True
            break
        print("Invalid choice. Please enter 1, 2, or 3.")
    
    # Use default threshold
    threshold = 0.5
    
    # Ask for output file
    print("\n" + "-"*60)
    print("Output File")
    print("-"*60)
    output_file = input("Enter output CSV file path (default: predictions.csv): ").strip()
    if not output_file:
        output_file = "predictions.csv"
    
    # Remove quotes if present
    if output_file.startswith('"') and output_file.endswith('"'):
        output_file = output_file[1:-1]
    elif output_file.startswith("'") and output_file.endswith("'"):
        output_file = output_file[1:-1]
    
    # Get model directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svm_model_dir = os.path.join(script_dir, 'models', 'svm')
    tabpfn_model_dir = os.path.join(script_dir, 'models', 'tabpfn', 'all_features')
    
    # Make predictions
    print("\n" + "="*60)
    print("  Making Predictions...")
    print("="*60 + "\n")
    
    try:
        if use_both:
            # Predict with both models
            print("Running SVM model...")
            svm_probs, valid_sequences = predict_svm(sequences, svm_model_dir)
            print("\nRunning TabPFN model...")
            tabpfn_probs, _ = predict_tabpfn(sequences, tabpfn_model_dir)
            
            # Create comparison results dataframe
            results = pd.DataFrame({
                'Sequence_ID': sequence_ids[:len(valid_sequences)],
                'Sequence': valid_sequences,
                'SVM_Score': svm_probs,
                'SVM_Prediction': ['Antigen' if p >= threshold else 'Non-Antigen' for p in svm_probs],
                'TabPFN_Score': tabpfn_probs,
                'TabPFN_Prediction': ['Antigen' if p >= threshold else 'Non-Antigen' for p in tabpfn_probs],
                'Agreement': ['Agree' if (svm_probs[i] >= threshold) == (tabpfn_probs[i] >= threshold) else 'Disagree' 
                             for i in range(len(valid_sequences))],
                'Score_Difference': np.abs(svm_probs - tabpfn_probs)
            })
            
            # Save results
            results.to_csv(output_file, index=False)
            
            # Display results
            print("\n" + "="*60)
            print("  Comparison Results")
            print("="*60 + "\n")
            print(f"✓ Predictions saved to: {output_file}")
            print(f"✓ Total sequences predicted: {len(results)}")
            print(f"\nSVM Results:")
            print(f"  - Antigens predicted: {sum(results['SVM_Prediction'] == 'Antigen')}")
            print(f"  - Non-antigens predicted: {sum(results['SVM_Prediction'] == 'Non-Antigen')}")
            print(f"\nTabPFN Results:")
            print(f"  - Antigens predicted: {sum(results['TabPFN_Prediction'] == 'Antigen')}")
            print(f"  - Non-antigens predicted: {sum(results['TabPFN_Prediction'] == 'Non-Antigen')}")
            print(f"\nAgreement:")
            print(f"  - Agreeing predictions: {sum(results['Agreement'] == 'Agree')} ({100*sum(results['Agreement'] == 'Agree')/len(results):.1f}%)")
            print(f"  - Disagreeing predictions: {sum(results['Agreement'] == 'Disagree')} ({100*sum(results['Agreement'] == 'Disagree')/len(results):.1f}%)")
            print(f"\nAverage score difference: {results['Score_Difference'].mean():.4f}")
            print(f"Max score difference: {results['Score_Difference'].max():.4f}")
            print("\n" + "-"*60)
            print("Detailed Results:")
            print("-"*60)
            
            # Create pretty table for comparison results
            table = PrettyTable()
            table.field_names = ["Sequence_ID", "SVM_Score", "SVM_Pred", "TabPFN_Score", "TabPFN_Pred", "Agreement"]
            table.align["Sequence_ID"] = "l"
            table.align["SVM_Score"] = "r"
            table.align["TabPFN_Score"] = "r"
            table.align["Agreement"] = "c"
            
            for idx, row in results.iterrows():
                table.add_row([
                    row['Sequence_ID'][:30] + "..." if len(row['Sequence_ID']) > 30 else row['Sequence_ID'],
                    f"{row['SVM_Score']:.4f}",
                    row['SVM_Prediction'],
                    f"{row['TabPFN_Score']:.4f}",
                    row['TabPFN_Prediction'],
                    row['Agreement']
                ])
            
            print(table)
            print()
            
        else:
            # Single model prediction
            if model == 'svm':
                probabilities, valid_sequences = predict_svm(sequences, svm_model_dir)
            else:  # tabpfn
                probabilities, valid_sequences = predict_tabpfn(sequences, tabpfn_model_dir)
            
            # Create results dataframe
            results = pd.DataFrame({
                'Sequence_ID': sequence_ids[:len(valid_sequences)],
                'Sequence': valid_sequences,
                'Antigenicity_Score': probabilities,
                'Prediction': ['Antigen' if p >= threshold else 'Non-Antigen' for p in probabilities]
            })
            
            # Save results
            results.to_csv(output_file, index=False)
            
            # Display results
            print("\n" + "="*60)
            print("  Prediction Results")
            print("="*60 + "\n")
            print(f"✓ Predictions saved to: {output_file}")
            print(f"✓ Total sequences predicted: {len(results)}")
            print(f"✓ Antigens predicted: {sum(results['Prediction'] == 'Antigen')}")
            print(f"✓ Non-antigens predicted: {sum(results['Prediction'] == 'Non-Antigen')}")
            print("\n" + "-"*60)
            print("Detailed Results:")
            print("-"*60)
            
            # Create pretty table for single model results
            table = PrettyTable()
            table.field_names = ["Sequence_ID", "Score", "Prediction"]
            table.align["Sequence_ID"] = "l"
            table.align["Score"] = "r"
            table.align["Prediction"] = "c"
            
            for idx, row in results.iterrows():
                table.add_row([
                    row['Sequence_ID'][:40] + "..." if len(row['Sequence_ID']) > 40 else row['Sequence_ID'],
                    f"{row['Antigenicity_Score']:.4f}",
                    row['Prediction']
                ])
            
            print(table)
            print()
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(
        description='IApred: Predict protein antigenicity using pre-trained models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict using SVM model
  python IApred.py --input sequences.fasta --model svm --output predictions.csv
  
  # Predict using TabPFN model
  python IApred.py --input sequences.fasta --model tabpfn --output predictions.csv
  
  # Compare both models
  python IApred.py --input sequences.fasta --model both --output comparison.csv
  
  # Interactive mode (run without arguments)
  python IApred.py
        """
    )
    
    parser.add_argument('--input', '-i',
                       help='Input file with protein sequences (FASTA or text file, one sequence per line)')
    parser.add_argument('--model', '-m', choices=['svm', 'tabpfn', 'both'],
                       help='Model to use for prediction: svm, tabpfn, or both (for comparison)')
    parser.add_argument('--output', '-o',
                       help='Output CSV file with predictions')
    parser.add_argument('--threshold', '-t', type=float, default=0.5,
                       help='Threshold for binary classification (default: 0.5)')
    
    args = parser.parse_args()
    
    # If no arguments provided, run interactive mode
    if not args.input or not args.model or not args.output:
        interactive_mode()
        return
    
    # Command-line mode
    # Get model directories
    script_dir = os.path.dirname(os.path.abspath(__file__))
    svm_model_dir = os.path.join(script_dir, 'models', 'svm')
    tabpfn_model_dir = os.path.join(script_dir, 'models', 'tabpfn', 'all_features')
    
    # Read sequences
    logging.info(f"Reading sequences from {args.input}...")
    sequences, sequence_ids = read_sequences_from_file(args.input)
    
    if len(sequences) == 0:
        logging.error("No sequences found in input file")
        sys.exit(1)
    
    # Make predictions
    try:
        if args.model == 'both':
            # Predict with both models
            logging.info("Running SVM model...")
            svm_probs, valid_sequences = predict_svm(sequences, svm_model_dir)
            logging.info("Running TabPFN model...")
            tabpfn_probs, _ = predict_tabpfn(sequences, tabpfn_model_dir)
            
            # Create comparison results dataframe
            results = pd.DataFrame({
                'Sequence_ID': sequence_ids[:len(valid_sequences)],
                'Sequence': valid_sequences,
                'SVM_Score': svm_probs,
                'SVM_Prediction': ['Antigen' if p >= args.threshold else 'Non-Antigen' for p in svm_probs],
                'TabPFN_Score': tabpfn_probs,
                'TabPFN_Prediction': ['Antigen' if p >= args.threshold else 'Non-Antigen' for p in tabpfn_probs],
                'Agreement': ['Agree' if (svm_probs[i] >= args.threshold) == (tabpfn_probs[i] >= args.threshold) else 'Disagree' 
                             for i in range(len(valid_sequences))],
                'Score_Difference': np.abs(svm_probs - tabpfn_probs)
            })
            
            # Save results
            results.to_csv(args.output, index=False)
            logging.info(f"Comparison results saved to {args.output}")
            logging.info(f"Total sequences predicted: {len(results)}")
            logging.info(f"SVM - Antigens: {sum(results['SVM_Prediction'] == 'Antigen')}, Non-antigens: {sum(results['SVM_Prediction'] == 'Non-Antigen')}")
            logging.info(f"TabPFN - Antigens: {sum(results['TabPFN_Prediction'] == 'Antigen')}, Non-antigens: {sum(results['TabPFN_Prediction'] == 'Non-Antigen')}")
            logging.info(f"Agreement: {sum(results['Agreement'] == 'Agree')} ({100*sum(results['Agreement'] == 'Agree')/len(results):.1f}%)")
            logging.info(f"Disagreement: {sum(results['Agreement'] == 'Disagree')} ({100*sum(results['Agreement'] == 'Disagree')/len(results):.1f}%)")
            logging.info(f"Average score difference: {results['Score_Difference'].mean():.4f}")
            
        elif args.model == 'svm':
            probabilities, valid_sequences = predict_svm(sequences, svm_model_dir)
            
            # Create results dataframe
            results = pd.DataFrame({
                'Sequence_ID': sequence_ids[:len(valid_sequences)],
                'Sequence': valid_sequences,
                'Antigenicity_Score': probabilities,
                'Prediction': ['Antigen' if p >= args.threshold else 'Non-Antigen' for p in probabilities]
            })
            
            # Save results
            results.to_csv(args.output, index=False)
            logging.info(f"Predictions saved to {args.output}")
            logging.info(f"Total sequences predicted: {len(results)}")
            logging.info(f"Antigens predicted: {sum(results['Prediction'] == 'Antigen')}")
            logging.info(f"Non-antigens predicted: {sum(results['Prediction'] == 'Non-Antigen')}")
            
        else:  # tabpfn
            probabilities, valid_sequences = predict_tabpfn(sequences, tabpfn_model_dir)
            
            # Create results dataframe
            results = pd.DataFrame({
                'Sequence_ID': sequence_ids[:len(valid_sequences)],
                'Sequence': valid_sequences,
                'Antigenicity_Score': probabilities,
                'Prediction': ['Antigen' if p >= args.threshold else 'Non-Antigen' for p in probabilities]
            })
            
            # Save results
            results.to_csv(args.output, index=False)
            logging.info(f"Predictions saved to {args.output}")
            logging.info(f"Total sequences predicted: {len(results)}")
            logging.info(f"Antigens predicted: {sum(results['Prediction'] == 'Antigen')}")
            logging.info(f"Non-antigens predicted: {sum(results['Prediction'] == 'Non-Antigen')}")
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()

