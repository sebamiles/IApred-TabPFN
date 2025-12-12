# SVM Training Pipeline

This directory contains the complete SVM (Support Vector Machine) training pipeline for the IApred antigenicity predictor. The SVM approach uses traditional machine learning with extensive hyperparameter optimization and feature selection.

## 📊 Pipeline Overview

The SVM training pipeline consists of several automated steps:

1. **Feature Selection Optimization** (`Find_best_k.py`) - Determines optimal number of features
2. **Hyperparameter Optimization** (`Optimize_C_and_gamma.py`) - Grid search for SVM parameters
3. **Model Training** (`generate_and_save_models.py`) - Final model training with best parameters
4. **Cross-Validation** (`10fold_CV.py`, `LOCO-CV.py`) - Rigorous validation
5. **External Evaluation** (`External_Evaluation.py`) - Independent validation

## 🗂️ Directory Structure

```
svm_training/
├── scripts/                    # Individual training scripts
│   ├── Find_best_k.py         # Feature selection optimization
│   ├── Optimize_C_and_gamma.py # Hyperparameter optimization
│   ├── generate_and_save_models.py # Final model training
│   ├── 10fold_CV.py           # 10-fold cross-validation
│   ├── LOCO-CV.py             # Leave-One-Class-Out validation
│   ├── External_Evaluation.py # External validation
│   └── run_training.py       # Pipeline orchestrator
├── models/                    # Trained models and preprocessing
│   ├── IApred_SVM.joblib      # Final SVM model
│   ├── IApred_scaler.joblib   # Feature scaler
│   ├── IApred_feature_selector.joblib # Feature selector
│   ├── IApred_variance_selector.joblib # Variance threshold selector
│   ├── IApred_feature_mask.joblib # Selected feature mask
│   └── IApred_all_feature_names.joblib # All feature names
└── results/                   # Training results and plots
    ├── 10fold_cv_results.txt  # 10-fold CV metrics
    ├── 10fold_cv_roc.png      # 10-fold CV ROC curve
    └── loco_cv_roc.png        # LOCO CV ROC curve
```

## 🚀 Usage

### Automated Pipeline (Recommended)

Run the complete pipeline automatically:

```bash
# From project root
python train_svm.py
```

This will execute all steps in sequence and save results to the appropriate directories.

### Manual Execution

Run individual components if needed:

```bash
cd svm_training/scripts

# Step 1: Feature selection optimization
python Find_best_k.py

# Step 2: Hyperparameter optimization
python Optimize_C_and_gamma.py

# Step 3: Final model training
python generate_and_save_models.py

# Step 4: Cross-validation
python 10fold_CV.py
python LOCO-CV.py

# Step 5: External evaluation
python External_Evaluation.py
```

## 🔧 Technical Details

### Feature Selection Strategy
- **Initial filtering**: Variance threshold (>0.01) removes low-variance features
- **Optimization**: Recursive feature elimination with cross-validation
- **Optimal k**: 529 features selected from original 838 features
- **Selection criteria**: Maximizes 10-fold CV ROC-AUC

### Hyperparameter Optimization
- **Kernel**: RBF (Radial Basis Function)
- **C parameter**: Regularization parameter [0.01, 0.1, 1, 10, 100]
- **γ parameter**: Kernel coefficient [0.001, 0.01, 0.1, 1, 'scale']
- **Optimization method**: Grid search with 10-fold CV
- **Best parameters**: C=1.0, γ=0.001

### Model Configuration
- **SVM implementation**: scikit-learn SVC with probability=True
- **Feature scaling**: StandardScaler (zero mean, unit variance)
- **Probability calibration**: Platt scaling for reliable probability estimates
- **Random state**: 42 for reproducibility

## 📈 Performance Metrics

### Training Performance (10-fold CV)
- **ROC-AUC**: 0.832 ± 0.024
- **Accuracy**: 0.742 ± 0.025
- **MCC**: 0.484 ± 0.050
- **Sensitivity**: 0.754 ± 0.042
- **Specificity**: 0.730 ± 0.041

### External Validation (n=436)
- **ROC-AUC**: 0.775
- **Accuracy**: 0.693
- **MCC**: 0.387
- **Sensitivity**: 0.766
- **Specificity**: 0.617

## 📊 Output Files

### Models (`models/` directory)
- **IApred_SVM.joblib**: Trained SVM model with optimal parameters
- **IApred_scaler.joblib**: Feature standardization parameters
- **IApred_feature_selector.joblib**: Feature selection transformer
- **IApred_feature_mask.joblib**: Boolean mask for selected features
- **IApred_all_feature_names.joblib**: Complete list of feature names

### Results (`results/` directory)
- **10fold_cv_results.txt**: Detailed 10-fold CV metrics and confusion matrices
- **10fold_cv_roc.png**: ROC curves for all 10 folds
- **loco_cv_roc.png**: ROC curves for Leave-One-Class-Out validation

## 🔍 Validation Strategies

### 10-Fold Cross-Validation
- Stratified splitting maintains class balance
- Performance averaged across all folds
- Provides robust estimate of generalization performance

### Leave-One-Class-Out (LOCO) Validation
- Tests performance on each pathogen class independently
- Ensures model works across diverse pathogens
- Identifies potential class-specific biases

### External Validation
- Independent dataset with 436 sequences
- Less than 90% similarity to training data
- Provides unbiased performance estimate

## ⚙️ Configuration

The pipeline can be customized by modifying parameters in the individual scripts:

- **Feature selection range**: Modify `k_values` in `Find_best_k.py`
- **Hyperparameter grid**: Update `param_grid` in `Optimize_C_and_gamma.py`
- **Cross-validation folds**: Change `cv` parameter in validation scripts
- **Random seed**: Modify `random_state` for different runs

## 🐛 Troubleshooting

### Common Issues
1. **Memory errors**: Reduce feature selection range or use smaller grid
2. **Long training time**: Start with smaller parameter grids
3. **Convergence warnings**: Adjust SVM parameters or increase iterations
4. **Feature selection fails**: Check input data format and feature quality

### Performance Optimization
- Use multiple CPU cores for grid search (`n_jobs=-1`)
- Consider feature preprocessing to reduce dimensionality
- Use smaller parameter grids for initial testing
- Monitor memory usage during feature selection

## 📚 References

- **SVM Theory**: Cortes & Vapnik. "Support-vector networks." Machine Learning 20, 273–297 (1995)
- **Feature Selection**: Guyon & Elisseeff. "An introduction to variable and feature selection." JMLR 3, 1157–1182 (2003)
- **Hyperparameter Optimization**: Bergstra & Bengio. "Random search for hyper-parameter optimization." JMLR 13, 281–305 (2012)

## 🔗 Integration

The trained SVM model can be loaded and used independently:

```python
from joblib import load

# Load trained model
svm_model = load('models/IApred_SVM.joblib')
scaler = load('models/IApred_scaler.joblib')
feature_selector = load('models/IApred_feature_selector.joblib')

# Use for prediction (after feature extraction)
features_scaled = scaler.transform(features)
features_selected = feature_selector.transform(features_scaled)
predictions = svm_model.predict_proba(features_selected)
```

