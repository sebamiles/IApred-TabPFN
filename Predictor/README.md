# IApred Predictor

Standalone prediction tool for protein antigenicity prediction using pre-trained SVM and TabPFN models.

## Installation

1. Install Python 3.8 or higher
2. Install required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### Interactive Mode (Recommended for Beginners)

Simply run the script without any arguments for a guided, user-friendly experience:

```bash
python IApred.py
```

The interactive mode will guide you through:
1. **Input type selection**: Choose between single sequence or FASTA file
2. **Sequence input**: 
   - For single sequence: Paste your sequence (with or without FASTA header, multi-line supported)
   - For FASTA file: Provide the file path
3. **Model selection**: Choose from SVM, TabPFN, or both (for comparison)
4. **Output file**: Specify where to save results

Note: Classification threshold is set to 0.5 by default (can be changed via command-line `--threshold` option)

**Single Sequence Input Examples:**
- Just the sequence (multi-line is fine):
  ```
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG
  ```
- With FASTA header:
  ```
  >my_protein
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG
  ```
- Multi-line sequence (as typically formatted in FASTA files):
  ```
  MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVG
  DGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLG
  QHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVM
  GDGERQFSTLKSTVEAIWYIKKETG
  ```

### Command-Line Mode (For Advanced Users)

Predict antigenicity of sequences from a FASTA file using command-line arguments:

```bash
python IApred.py --input sequences.fasta --model svm --output predictions.csv
```

### Available Models

- **SVM**: Traditional Support Vector Machine model
- **TabPFN**: Tabular Prior-data Fitted Networks (foundation model) with all features
- **Both**: Run both models and compare predictions (recommended for comprehensive analysis)

### Examples

**Using SVM model:**
```bash
python IApred.py --input my_sequences.fasta --model svm --output results.csv
```

**Using TabPFN model (recommended for best performance):**
```bash
python IApred.py --input my_sequences.fasta --model tabpfn --output results.csv
```

**Compare both models:**
```bash
python IApred.py --input my_sequences.fasta --model both --output comparison.csv
```

**Custom threshold:**
```bash
python IApred.py --input my_sequences.fasta --model svm --output results.csv --threshold 0.6
```

### Input Format

The input file can be in one of two formats:

1. **FASTA format:**
```
>sequence1
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG
>sequence2
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG
```

2. **Text format (one sequence per line):**
```
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG
MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWYIKKETG
```

### Output Format

**Single Model Output:**
The output CSV file contains the following columns:

- `Sequence_ID`: Identifier from FASTA file or auto-generated (seq_1, seq_2, etc.)
- `Sequence`: The protein sequence
- `Antigenicity_Score`: Probability score (0-1), where higher values indicate higher antigenicity
- `Prediction`: Binary prediction (Antigen/Non-Antigen) based on threshold (default: 0.5)

**Both Models Comparison Output:**
When using `--model both`, the output CSV contains:

- `Sequence_ID`: Identifier from FASTA file or auto-generated
- `Sequence`: The protein sequence
- `SVM_Score`: SVM prediction score (0-1)
- `SVM_Prediction`: SVM binary prediction (Antigen/Non-Antigen)
- `TabPFN_Score`: TabPFN prediction score (0-1)
- `TabPFN_Prediction`: TabPFN binary prediction (Antigen/Non-Antigen)
- `Agreement`: Whether both models agree (Agree/Disagree)
- `Score_Difference`: Absolute difference between SVM and TabPFN scores

### Command Line Arguments

- `--input`, `-i`: Input file with protein sequences (required in command-line mode)
- `--model`, `-m`: Model to use: `svm`, `tabpfn`, or `both` (required in command-line mode)
- `--output`, `-o`: Output CSV file path (required in command-line mode)
- `--threshold`, `-t`: Threshold for binary classification (default: 0.5)

### Model Performance

Based on external validation (n=436 sequences):

| Model | ROC-AUC | MCC | Sensitivity | Specificity |
|-------|---------|-----|-------------|-------------|
| **TabPFN (All Features)** | **0.816** | **0.518** | **0.766** | **0.752** |
| SVM (Optimized) | 0.782 | 0.432 | 0.703 | 0.729 |

**Recommendation**: 
- Use TabPFN for best performance
- Use "both" option to compare predictions and get consensus results

## File Structure

```
Predictor/
├── IApred.py                  # Main prediction script
├── functions_for_training.py   # Feature extraction functions
├── protein_motifs.txt         # Protein motif patterns
├── requirements.txt           # Python dependencies
├── README.md                  # This file
└── models/                    # Pre-trained models
    ├── svm/                   # SVM model files
    └── tabpfn/                # TabPFN model files
        └── all_features/      # TabPFN model with all features
```

## Notes

- All sequences are automatically cleaned (only standard 20 amino acids are kept)
- Sequences that fail feature extraction are skipped with a warning
- The script supports both FASTA and plain text input formats
- Predictions are saved in CSV format for easy analysis

## Citation

If you use IApred in your research, please cite:

```bibtex
@article{miles2025iapred,
  title={IApred: A versatile open-source tool for predicting protein antigenicity across diverse pathogens},
  author={Miles, Sebasti{\'a}n and others},
  journal={ImmunoInformatics},
  volume={20},
  pages={100061},
  year={2025}
}
```

## License

MIT License

