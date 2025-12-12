# Model Comparison Framework

This directory contains comprehensive tools for comparing machine learning models for antigenicity prediction. The framework evaluates multiple approaches including IApred's SVM and TabPFN models against state-of-the-art external predictors.

## 🎯 Purpose

The comparison framework serves multiple purposes:

- **Internal validation**: Compare different IApred model configurations
- **External benchmarking**: Evaluate against established antigenicity predictors
- **Performance analysis**: Generate comprehensive metrics and visualizations
- **Statistical testing**: Assess significance of performance differences

## 📊 Comparison Types

### Internal Model Comparison
- **IApred SVM** vs **IApred TabPFN** variants
- **Feature ablation studies**: All features vs. optimized subsets
- **Cross-validation consistency**: 10-fold CV and LOCO validation

### External Predictor Benchmarking
- **ANTIGENpro**: State-of-the-art antigenicity predictor
- **VaxiJen 2.0 & 3.0**: Established vaccine design tools
- **IApred variants**: Our SVM and TabPFN implementations

## 🗂️ Directory Structure

```
comparison/
├── scripts/                    # Comparison scripts
│   ├── generate_all_evaluations.py    # Internal model comparison
│   ├── generate_comparison_plots.py   # Visualization generation
│   └── retrain_all_models.py          # Model retraining utility
├── results/                   # Comparison outputs
│   ├── all_models_comparison.csv      # Internal comparison metrics
│   ├── all_models_antigens_predictions.csv    # Antigen predictions
│   ├── all_models_non-antigens_predictions.csv # Non-antigen predictions
│   ├── performance_heatmap.png        # Performance comparison heatmap
│   ├── roc_comparison.png             # ROC curves comparison
│   ├── calibration_comparison.png     # Calibration plots
│   ├── removed_duplicates.csv         # Duplicate sequence handling
│   └── delete/                        # Legacy results (can be removed)
└── with external predictors/  # External benchmarking
    ├── all_evaluation_metrics_comprehensive.csv # Complete metrics
    ├── generate_plots_and_metrics.py  # External comparison script
    ├── comprehensive_statistical_analysis.py # Statistical tests
    ├── roc_auc_comparison_unified.png # Unified ROC comparison
    ├── radial_metrics_comparison_unified.png # Radial metrics plot
    ├── combined_statistical_significance_heatmap_simple.png
    ├── delong_test_all_predictors_436.csv # DeLong test results
    ├── mcnemar_test_all_predictors_436.csv # McNemar test results
    ├── unified_antigens.csv            # Standardized antigen data
    └── unified_non-antigens.csv        # Standardized non-antigen data
```

## 🚀 Usage

### Internal Model Comparison

Compare IApred models on internal validation:

```bash
# From project root
python compare_models.py --tabpfn all 529 100

# Or run comparison script directly
cd comparison/scripts
python generate_all_evaluations.py
```

### External Predictor Benchmarking

Compare against external tools:

```bash
cd comparison/with\ external\ predictors
python generate_plots_and_metrics.py
python comprehensive_statistical_analysis.py
```

### Visualization Generation

Generate comparison plots:

```bash
cd comparison/scripts
python generate_comparison_plots.py
```

## 📈 Performance Metrics

### Comprehensive Evaluation Metrics

The framework evaluates models using multiple performance measures:

| Metric | Description | Range | Interpretation |
|--------|-------------|-------|----------------|
| **ROC-AUC** | Area Under ROC Curve | 0-1 | Discrimination ability |
| **Accuracy** | Overall correct predictions | 0-1 | Classification accuracy |
| **MCC** | Matthews Correlation Coefficient | -1 to +1 | Balanced performance |
| **Sensitivity** | True Positive Rate | 0-1 | Antigen detection |
| **Specificity** | True Negative Rate | 0-1 | Non-antigen detection |
| **Precision** | Positive Predictive Value | 0-1 | Prediction reliability |
| **F1-Score** | Harmonic mean of precision/recall | 0-1 | Balanced accuracy |
| **Brier Score** | Mean squared probability error | 0-∞ | Calibration quality |
| **ECE** | Expected Calibration Error | 0-1 | Probability calibration |

### Key Results Summary

**External Validation Performance (n=436):**

| Predictor | ROC-AUC | MCC | Accuracy | Sensitivity | Specificity |
|-----------|---------|-----|----------|-------------|-------------|
| **TabPFN (All Features)** | **0.807** | **0.491** | **0.745** | **0.775** | **0.715** |
| TabPFN (529 Features) | 0.803 | 0.495 | 0.748 | 0.775 | 0.720 |
| TabPFN (100 Features) | 0.797 | 0.440 | 0.720 | 0.730 | 0.710 |
| SVM (Optimized) | 0.775 | 0.387 | 0.693 | 0.766 | 0.617 |
| ANTIGENpro | 0.752 | 0.336 | 0.659 | 0.851 | 0.457 |
| VaxiJen 3.0 | 0.691 | 0.320 | 0.658 | 0.604 | 0.715 |
| VaxiJen 2.0 | 0.681 | 0.225 | 0.603 | 0.838 | 0.360 |

## 📊 Visualization Outputs

### Performance Heatmap
- **File**: `performance_heatmap.png`
- **Content**: Color-coded performance comparison across all metrics
- **Purpose**: Quick visual comparison of model strengths/weaknesses

### ROC Comparison
- **File**: `roc_comparison.png`
- **Content**: Overlaid ROC curves for all models
- **Purpose**: Visual assessment of discrimination ability

### Calibration Comparison
- **File**: `calibration_comparison.png`
- **Content**: Reliability curves showing probability calibration
- **Purpose**: Evaluate how well predicted probabilities match actual outcomes

### Statistical Significance
- **DeLong Test**: Compares ROC-AUC differences statistically
- **McNemar Test**: Tests classification agreement significance
- **Heatmaps**: Visual representation of statistical significance

## 🔬 Statistical Analysis

### Significance Testing

The framework includes rigorous statistical evaluation:

#### DeLong Test for ROC-AUC
- Tests whether ROC-AUC differences are statistically significant
- Accounts for correlation between ROC curves
- Provides p-values for pairwise comparisons

#### McNemar Test for Classification
- Tests whether classification disagreements are significant
- Chi-square based test for paired nominal data
- Evaluates consistency of predictions

### Confidence Intervals
- **Bootstrap resampling**: 1000 iterations for robust estimates
- **95% confidence intervals**: Reported for all metrics
- **Standard errors**: Provided for statistical inference

## 📋 Data Standardization

### Sequence Processing
- **Duplicate removal**: Identical sequences removed to avoid bias
- **Format standardization**: Consistent FASTA processing
- **Quality control**: Invalid sequences filtered out

### Prediction Standardization
- **Threshold normalization**: 0.5 probability threshold for binary classification
- **Output format**: Consistent CSV structure across all predictors
- **Missing data handling**: Robust error handling for prediction failures

## 🛠️ Technical Details

### Script Dependencies
- **pandas/numpy**: Data manipulation and analysis
- **matplotlib/seaborn**: Visualization and plotting
- **scipy.stats**: Statistical testing functions
- **scikit-learn**: Performance metrics calculation

### Performance Optimization
- **Vectorized operations**: Efficient computation on large datasets
- **Memory management**: Streaming processing for large prediction files
- **Parallel processing**: Multi-core utilization where possible

## 📁 Output Formats

### CSV Files Structure

#### `all_models_comparison.csv`
```csv
Model,ROC-AUC,Accuracy,MCC,Sensitivity,Specificity,Precision,F1-Score,Brier_Score,ECE
TabPFN_all,0.807,0.745,0.491,0.775,0.715,0.738,0.756,0.180,0.084
SVM,0.775,0.693,0.387,0.766,0.617,0.675,0.717,0.194,0.068
```

#### `all_evaluation_metrics_comprehensive.csv`
Contains complete metrics for all predictors including confidence intervals and additional statistics.

### Visualization Specifications
- **Resolution**: High DPI (300) for publication quality
- **Color scheme**: Colorblind-friendly palettes
- **Font sizes**: Optimized for readability
- **Aspect ratios**: Suitable for various publication formats

## 🎯 Best Practices

### For Researchers
1. **Use multiple metrics**: Don't rely on single performance measures
2. **Consider calibration**: Well-calibrated probabilities are crucial for applications
3. **Statistical significance**: Always check if differences are meaningful
4. **Cross-validation**: Validate on independent datasets

### For Benchmarking
1. **Fair comparison**: Use identical test sets and preprocessing
2. **Multiple runs**: Account for stochasticity with multiple evaluations
3. **Confidence intervals**: Report uncertainty in performance estimates
4. **Computational efficiency**: Consider inference speed for production use

## 🐛 Troubleshooting

### Common Issues
1. **Memory errors**: Process data in smaller batches
2. **File not found**: Ensure models are trained before comparison
3. **Import errors**: Check all required packages are installed
4. **Plot display issues**: Use appropriate matplotlib backend

### Performance Debugging
- **Slow execution**: Check for memory leaks or inefficient loops
- **Inconsistent results**: Verify random seeds and data preprocessing
- **Visualization errors**: Update matplotlib/seaborn versions

## 📚 References

### Statistical Methods
- **DeLong et al.**: "Comparing the areas under two or more correlated receiver operating characteristic curves: a nonparametric approach" Biometrics (1988)
- **McNemar**: "Note on the sampling error of the difference between correlated proportions or percentages" Psychometrika (1947)

### Evaluation Metrics
- **Hand & Till**: "A simple generalisation of the area under the ROC curve for multiple class classification problems" Machine Learning (2001)
- **Matthews**: "Differentiation of protein in immune sera by a precipitation reaction" Biochem J (1975)

## 🔗 Integration

The comparison framework can be extended to include new predictors:

```python
# Example: Add custom predictor to comparison
import pandas as pd
from comparison_framework import ModelComparator

# Load predictions
custom_predictions = pd.read_csv('custom_predictor_results.csv')

# Initialize comparator
comparator = ModelComparator()

# Add to comparison
comparator.add_predictor('Custom Predictor', custom_predictions)

# Generate comparison
results = comparator.generate_comparison()
plots = comparator.generate_plots()
```

---

**Note**: This framework provides comprehensive evaluation tools essential for rigorous model comparison in antigenicity prediction research.

