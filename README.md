# 🧬 IApred: Advanced Antigenicity Prediction

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

**IApred** is a state-of-the-art antigenicity prediction tool that showcases the evolution from traditional machine learning to modern foundation models. Compare SVM's complex hyperparameter optimization with TabPFN's zero-shot learning approach.

## 📊 Performance Comparison

| Model | ROC-AUC | MCC | Training Time | Code Complexity |
|-------|---------|-----|---------------|----------------|
| **SVM (Optimized)** | 0.761 | 0.399 | ~45 minutes | 🔴 High (719 lines) |
| **TabPFN (All Features)** | **0.780** | **0.446** | **<30 seconds** | 🟢 Low (64 lines core) |
| TabPFN (529 Features) | 0.772 | 0.418 | <30 seconds | 🟢 Low |
| TabPFN (100 Features) | 0.772 | 0.408 | <30 seconds | 🟢 Low |

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Repository Structure](#️-repository-structure)
- [🚀 Quick Start](#-quick-start)
- [📦 Installation](#-installation)
- [🔧 Usage](#-usage)
- [📊 Data](#-data)
- [📈 Results](#-results)
- [🤝 Contributing](#-contributing)
- [📚 Citation](#-citation)
- [📄 License](#-license)

## ✨ Features

### 🔬 Dual Implementation
- **Traditional SVM**: Comprehensive hyperparameter optimization pipeline
- **Modern TabPFN**: Foundation model with zero-shot learning capabilities

### ⚡ Streamlined Workflow
- **One-command training**: Complete pipelines for both approaches
- **Automated CV**: 10-fold, Leave-One-Class-Out (LOCO)
- **Multi-model comparison**: Automated evaluation and visualization

### 🎯 Production Ready
- **GPU acceleration** support for TabPFN
- **Scalable architecture** for different feature sets
- **Comprehensive validation** across multiple cross-validation schemes

## 🏗️ Repository Structure

```
IApred-PFN/
├── 📄 README.md                          # Comprehensive documentation
├── 📦 requirements.txt                   # Python dependencies
├── 🔧 train_svm.py              # Complete SVM training pipeline
├── 🚀 train_tabpfn.py           # TabPFN training with CV
├── 📊 compare_models.py                  # Multi-model comparison
├── 🔄 data_loader.py                     # Data loading utilities
└── 🧬 functions_for_training.py          # Feature extraction & processing

├── 🔧 svm_training/                      # Traditional SVM implementation
│   ├── 📜 scripts/                       # Training & evaluation scripts
│   │   ├── Find_best_k.py               # Feature selection optimization
│   │   ├── Optimize_C_and_gamma.py      # Hyperparameter tuning
│   │   ├── generate_and_save_models.py  # Final model training
│   │   ├── run_training.py              # Pipeline orchestrator
│   │   ├── 10fold_CV.py                 # 10-fold cross-validation
│   │   ├── LOCO-CV.py                   # Leave-One-Class-Out
│   │   └── External_Evaluation.py       # External validation
│   ├── 🤖 models/                        # Trained SVM model files
│   └── 📈 results/                       # Evaluation outputs & plots

├── 🚀 tabpfn_training/                   # TabPFN implementations
│   ├── 📜 scripts/                       # Unified scripts (--k parameter)
│   │   ├── train_model.py               # Model training
│   │   ├── 10fold_cv.py                 # 10-fold cross-validation
│   │   ├── loco_cv.py                   # Leave-One-Class-Out
│   │   └── external_evaluation.py       # External validation
│   ├── 🤖 models/                        # Models by feature count
│   │   ├── all_features/                # Full feature set
│   │   ├── 529_features/                # Optimized features
│   │   └── 100_features/                # Minimal features
│   └── 📈 results/                       # Results by feature count

├── 📊 comparison/                       # Model comparison framework
│   ├── 📜 scripts/                      # Comparison scripts
│   │   ├── generate_all_evaluations.py # Multi-model evaluation
│   │   ├── generate_comparison_plots.py # Visualization creation
│   │   └── retrain_all_models.py        # Batch retraining
│   └── 📈 results/                      # Comparison outputs

├── 📋 data/                             # External evaluation datasets
│   ├── External_evaluation_antigens.csv      # 218 antigen sequences
│   └── External_evaluation_non_antigens.csv  # 218 non-antigens

└── 📄 docs/                             # Documentation
    └── IApred-TabPFN.md                # Complete scientific paper
```

## 🚀 Quick Start

### Complete Pipelines

```bash
# 1. Train SVM model with full pipeline (feature selection, hyperparameter tuning, CV)
python train_svm.py

# alteratevly, if no feature selection and hyperparameter turning want to be done:
python train_svm.py --k 529 --c 1 --gamma 0.001 --skip-optimization


# 2. Train TabPFN model with specified features and CV
python train_tabpfn.py --k all    # All features (recommended)
python train_tabpfn.py --k 529    # 529 features
python train_tabpfn.py --k 100    # 100 features

# 3. Compare SVM with multiple TabPFN models
python compare_models.py --tabpfn all 529 100
```

### Example Workflow

```bash
# Train models
python train_svm_pipeline.py
python train_tabpfn_pipeline.py --k all

# Compare performance
python compare_models.py --tabpfn all

# Results saved in comparison/results/
```

## 📦 Installation

1. **Clone the repository**:
```bash
git clone https://github.com/yourusername/IApred-TabPFN.git
cd IApred-TabPFN
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **GPU acceleration** (recommended for TabPFN):
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 🔧 Usage

### Advanced Usage

#### SVM Implementation

```bash
# Full optimization pipeline (default)
python train_svm_pipeline.py

# Skip optimization, use specific parameters
python train_svm_pipeline.py --k 529 --c 1.0 --gamma 0.001 --skip-optimization

# Individual steps (if needed)
cd svm_training/scripts
python Find_best_k.py                    # Feature selection
python Optimize_C_and_gamma.py --k 529   # Hyperparameter tuning
python generate_and_save_models.py --k 529 --c 1.0 --gamma 0.01
python 10fold_CV.py                      # Cross-validation
```

#### TabPFN Implementation

```bash
# Full pipeline with different feature sets
python train_tabpfn_pipeline.py --k all   # Recommended
python train_tabpfn_pipeline.py --k 529   # Optimized
python train_tabpfn_pipeline.py --k 100   # Minimal

# Individual steps
cd tabpfn_training/scripts
python train_model.py --k all            # Training only
python 10fold_cv.py --k all              # CV only
```

#### Model Comparison

```bash
# Compare specific model combinations
python compare_models.py --tabpfn all 529    # SVM vs TabPFN-all + TabPFN-529
python compare_models.py --tabpfn 100        # SVM vs TabPFN-100 only

# Generate plots from existing evaluations
cd comparison/scripts
python generate_comparison_plots.py
```

## 📊 Data

### Training Data
- **918 antigens** from bacteria, viruses, fungi, protozoa, helminths
- **918 non-antigens** (balanced dataset)
- **838 features** extracted per sequence:
  - Physicochemical properties (isoelectric point, GRAVY, etc.)
  - Secondary structure predictions
  - Compositional features
  - E-descriptors
  - Amino acid dimers
  - Short Linear Motifs (SLiMs)

### External Validation
- **436 sequences** (218 antigens + 218 non-antigens)
- Independent dataset from Protegen
- <90% similarity to training data

## 📈 Results

### Performance Comparison

| Model | ROC-AUC | MCC | Specificity | Training Time | Lines of Code |
|-------|---------|-----|-------------|---------------|---------------|
| **SVM (Optimized)** | 0.761 | 0.399 | 0.702 | ~45 minutes | 719 total |
| **TabPFN (All Features)** | **0.780** | **0.446** | **0.748** | **<30 seconds** | 64 (core) |
| TabPFN (529 Features) | 0.772 | 0.418 | 0.720 | <30 seconds | 67 (core) |
| TabPFN (100 Features) | 0.772 | 0.408 | 0.711 | <30 seconds | 67 (core) |

### Key Improvements with TabPFN
- **+2.5% ROC-AUC** improvement
- **+11.5% MCC** improvement
- **99% reduction** in training time
- **100% elimination** of hyperparameter optimization
- **49% reduction** in core training code

### Output Files
After running comparisons, you'll find:
- `comparison/results/all_models_comparison.csv` - Comprehensive metrics table
- `comparison/results/performance_heatmap.png` - Metrics heatmap visualization
- `comparison/results/roc_comparison.png` - ROC curves comparison
- `comparison/results/calibration_comparison.png` - Calibration plots

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📚 Citation

If you use this code or data, please cite:

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

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📞 Contact

For questions or issues, please open a GitHub issue or contact the maintainers.

---

## 📊 Repository Statistics

- **Total Files**: 33 files (streamlined and organized)
- **Python Scripts**: 20 files (~4,600 lines total)
- **Documentation**: 4 files (README, paper, data docs)
- **Data Files**: 2 evaluation datasets
- **Code Reduction**: Removed 8 unnecessary files while maintaining full functionality

**Note**: This repository demonstrates the paradigm shift from traditional machine learning optimization (SVM: 35-45 hours development) to foundation model approaches (TabPFN: <30 seconds training). The TabPFN implementation achieves superior performance with dramatically simplified development workflows.

