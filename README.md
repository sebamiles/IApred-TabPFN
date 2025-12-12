# 🧬 IApred: Advanced Antigenicity Prediction Tool

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![DOI](https://img.shields.io/badge/DOI-10.1016/j.immuno.2025.100061-blue)](https://doi.org/10.1016/j.immuno.2025.100061)

**IApred** is an intrinsic antigenicity predictor. The original implementation uses traditional machine learning approaches (SVM), while in this implementation we use modern foundation models (TabPFN). This project demonstrates the transition from complex hyperparameter optimization to efficient zero-shot learning for protein antigenicity prediction across diverse pathogens.

## 📊 Key Features

- **Dual ML Approaches**: Original IApred-SVM (with extensive optimization) vs. modern TabPFN foundation models
- **Training**: 918 antigens + 918 non-antigens from 22 diverse pathogens
- **Validation**: 10-fold CV, Leave-One-Class-Out (LOCO) validation, and external validation (n=436)
- **Feature Engineering**: 838 physicochemical, structural, protein motifs and sequence-based descriptors
- **User-Friendly Predictor**: Standalone tool for antigenicity prediction with multiple input formats
- **GPU Acceleration**: PyTorch-based TabPFN implementation need CUDA for fast inference

## 📈 Performance Comparison (External Validation, n=436)

| Model | Accuracy | ROC-AUC | MCC | Sensitivity | Specificity | F1-Score | Precision |
|-------|----------|---------|-----|-------------|-------------|----------|-----------|
| **TabPFN (All Features)** | **0.745** | **0.807** | **0.491** | **0.775** | **0.715** | **0.756** | **0.738** |
| TabPFN (529 Features) | 0.748 | 0.803 | 0.495 | 0.775 | 0.720 | 0.758 | 0.741 |
| TabPFN (100 Features) | 0.720 | 0.797 | 0.440 | 0.730 | 0.710 | 0.726 | 0.723 |
| SVM (Optimized) | 0.693 | 0.775 | 0.387 | 0.766 | 0.617 | 0.717 | 0.675 |
| ANTIGENpro | 0.659 | 0.752 | 0.336 | 0.851 | 0.457 | 0.719 | 0.623 |
| VaxiJen 3.0 | 0.658 | 0.691 | 0.320 | 0.604 | 0.715 | 0.643 | 0.687 |
| VaxiJen 2.0 | 0.603 | 0.681 | 0.225 | 0.838 | 0.360 | 0.683 | 0.576 |

<img width="2113" height="2113" alt="roc_auc_comparison_unified" src="https://github.com/user-attachments/assets/b27378a9-4e00-49a3-a558-07ebee0f8cb0" />

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/IApred-PFN.git
cd IApred-PFN

# Install dependencies
pip install -r requirements.txt
```

### Training Models

```bash
# Train SVM model with full optimization pipeline
python train_svm.py

# Train TabPFN models (recommended - no hyperparameter tuning needed)
python train_tabpfn.py --k all         # Best performance (all 838 features)
python train_tabpfn.py --k 529         # Optimized feature subset
python train_tabpfn.py --k "number"    # Use a custom number of features for training
```

### Model Comparison

```bash
# Compare all models and generate performance plots
python compare_models.py --tabpfn all 529 "numbers"
```

### Prediction

```bash
# Use the standalone predictor (recommended for users)
cd Predictor

IApred can be run in one step:
python IApred.py --input your_sequences.fasta --model tabpfn --output predictions.csv

Or can be run with an interactive UX
python IApred.py

```

## 📁 Repository Structure

```
IApred-PFN/
├── 🧬 Core Training Framework
│   ├── train_svm.py              # SVM full training pipeline with hyperparameter optimization
│   ├── train_tabpfn.py           # TabPFN training pipeline
│   ├── compare_models.py         # Model comparison framework
│   ├── functions_for_training.py # Feature extraction utilities
│   └── data_loader.py           # Data loading utilities
├── 📊 Training Implementations
│   ├── svm_training/            # SVM training scripts & results
│   │   ├── scripts/            # Individual training scripts
│   │   ├── models/             # Trained SVM models
│   │   └── results/            # Training metrics & plots
│   └── tabpfn_training/        # TabPFN training scripts & results
│       ├── scripts/            # Individual training scripts
│       ├── models/             # Trained TabPFN models
│       └── results/            # Training metrics & plots
├── 🔍 Model Comparison
│   └── comparison/             # Comparative analysis framework
│       ├── scripts/            # Analysis scripts
│       └── results/            # Comparison plots & metrics
├── 🧪 Standalone Predictor
│   └── Predictor/              # User-friendly prediction tool
│       ├── IApred.py           # Main prediction script
│       ├── models/             # Pre-trained models
│       └── README.md           # Predictor documentation
├── 📋 Data
│   ├── antigens/               # Training antigen sequences (22 pathogens)
│   ├── non-antigens/           # Training non-antigen sequences
│   ├── data/                   # External evaluation datasets
│   └── protein_features.pkl    # Pre-computed feature matrix
├── 📄 Documentation
│   ├── Paper/                  # Scientific manuscript & figures
│   ├── README.md               # This file
│   └── requirements.txt        # Python dependencies
└── 🧪 Legacy Scripts
    ├── run_all_cv.py           # Cross-validation runner
    └── compare_models.py       # Model comparison (legacy)
```

## 📊 Data Overview

### Training Data
- **918 antigens + 918 non-antigens** from **22 diverse pathogens**
- **Pathogens covered**: *Aspergillus fumigatus*, *Actinobacillus pleuropneumoniae*, *Ascaris suum*, *Bacillus anthracis*, *Bordetella pertussis*, *Candida albicans*, *Cryptococcus neoformans*, *Coccidioides posadasii*, *Fasciola hepatica*, *Histoplasma capsulatum*, *Mycobacterium bovis*, viruses, *Paracoccidioides brasiliensis*, *Plasmodium vivax*, *Staphylococcus aureus*, *Schistosoma mansoni*, *Toxoplasma gondii*, and more

### Feature Engineering
- **838 physicochemical descriptors**: Amino acid composition, physicochemical properties, structural features
- **Advanced features**: Sequence motifs, evolutionary conservation, secondary structure predictions
- **Feature selection**: Variance threshold, correlation-based, and model-based selection

### Validation Strategy
- **10-fold cross-validation** on training data
- **Leave-One-Class-Out (LOCO)** validation for pathogen-specific performance
- **External validation** on independent dataset (436 sequences, <90% similarity to training data)

## 🔬 Methodology

### SVM Approach (Traditional ML)
1. **Feature Selection**: Recursive feature elimination with cross-validation
2. **Hyperparameter Optimization**: Grid search for C and gamma parameters
3. **Model Training**: RBF kernel SVM with probability calibration
4. **Feature Reduction**: From 838 to optimal subset (529 features)

### TabPFN Approach (Foundation Model)
1. **Zero-shot Learning**: No hyperparameter tuning required
2. **Multiple Feature Sets**: All features (791), optimized (529), minimal (100)
3. **Prior-data Fitting**: Leverages transformer architecture trained on tabular data
4. **GPU Acceleration**: PyTorch implementation for efficient training/inference

## 📈 Key Results

- **TabPFN (All Features)** achieves **0.807 ROC-AUC** and **0.491 MCC** on external validation
- **+3.2% ROC-AUC improvement** over optimized SVM baseline
- **+46% MCC improvement** over existing antigenicity predictors (ANTIGENpro, VaxiJen)
- **Zero hyperparameter tuning** for TabPFN vs. extensive optimization for SVM
- **Superior calibration** and reduced overfitting compared to traditional approaches


## 🤝 Usage Guidelines

### For Researchers
1. **Training**: Use `train_tabpfn.py --k all` for best performance with minimal setup
2. **Prediction**: Use the `Predictor/` folder for standalone predictions
3. **Comparison**: Run `compare_models.py` to generate comprehensive performance reports
4. **Validation**: Always validate on independent datasets before deployment

### For Developers
- Models are saved in scikit-learn/TabPFN format for easy integration
- Feature extraction functions are modular and reusable
- All scripts include comprehensive logging and error handling
- GPU acceleration available for TabPFN inference

## 🔧 System Requirements

- **Python**: 3.8 or higher
- **RAM**: 16GB recommended (32GB for large-scale training)
- **GPU**: CUDA-compatible GPU recommended for TabPFN training
- **Storage**: 5GB for models and datasets
- **OS**: Linux/macOS/Windows (Linux recommended for training)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Setup
```bash
# Fork and clone the repository
git clone https://github.com/yourusername/IApred-PFN.git
cd IApred-PFN

# Create virtual environment
python -m venv iapred_env
source iapred_env/bin/activate  # On Windows: iapred_env\Scripts\activate

# Install in development mode
pip install -e .
pip install -r requirements-dev.txt  # For testing and development
```

### Testing
```bash
# Run unit tests
python -m pytest tests/

# Run integration tests
python -m pytest tests/integration/

# Check code quality
flake8 . --max-line-length=100
black --check .
```

## 📚 Citation

If you use IApred in your research, please cite our paper:

```bibtex
@article{miles2025iapred,
  title={IApred: A versatile open-source tool for predicting protein antigenicity across diverse pathogens},
  author={Miles, Sebasti{\'a}n and others},
  journal={ImmunoInformatics},
  volume={20},
  pages={100061},
  year={2025},
  doi={10.1016/j.immuno.2025.100061}
}
```

### Additional References
- **TabPFN**: Hollmann et al. "TabPFN: A Transformer That Solves Small Tabular Classification Problems in a Second." arXiv preprint arXiv:2207.01848 (2022)
- **ANTIGENpro**: Magnan et al. "ANTIGENpro: Machine learning pipeline for predicting immunogenic peptides." bioRxiv (2021)
- **VaxiJen**: Doytchinova & Flower. "VaxiJen: a server for prediction of protective antigens, tumour antigens and subunit vaccines." BMC Bioinformatics 8, 4 (2007)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Training Data**: Compiled from multiple public databases and literature sources
- **Feature Extraction**: Built upon established bioinformatics tools and methods
- **TabPFN Implementation**: Based on the open-source TabPFN library
- **Community**: Thanks to the bioinformatics and machine learning communities for foundational work

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/IApred-PFN/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/IApred-PFN/discussions)
- **Email**: [Contact Information]

---

<div align="center">

**IApred: Bridging Traditional ML and Foundation Models for Antigenicity Prediction**

[🚀 Quick Start](#-quick-start) • [📖 Documentation](#-repository-structure) • [🤝 Contributing](#-contributing) • [📚 Citation](#-citation)

</div>

