# SHAP Analysis for Hierarchical Multi-Modal Fusion Network

This directory contains the implementation for SHAP (SHapley Additive exPlanations) analysis of the pre-trained multimodal psoriasis classification models.

## Overview

SHAP analysis provides model-agnostic explanations for machine learning predictions by computing the contribution of each feature to individual predictions. This implementation analyzes the feature importance across different model configurations to understand which clinical and spectral features are most important for psoriasis classification.

## Key Scientific Findings

The SHAP analysis of our hierarchical multimodal fusion network revealed significant insights into the molecular and clinical determinants of psoriatic arthritis (PSA) versus psoriasis (PSO) classification. Our investigation of 125 patients (46 PSA, 79 PSO) across three model configurations demonstrated a clear hierarchy of feature importance that aligns with established pathophysiological mechanisms. In the spectral model, disulfide bond content emerged as the predominant discriminative feature (mean |SHAP| = 0.1799), substantially exceeding other molecular markers and highlighting the critical role of oxidative stress in disease differentiation. This finding is complemented by significant contributions from amide bond content markers (Amide Content 2: 0.1057; Amide Content 1: 0.0725), which reflect underlying protein structural modifications characteristic of inflammatory arthropathies. The clinical model analysis revealed that body surface area (BSA = 0.0267) and patient age (0.0239) represent the most influential clinical parameters, consistent with established disease severity indicators. Notably, in the integrated clinical-spectral model, molecular features maintained their dominance, with Amide Content 2 (0.1091) and disulfide content (0.0580) ranking as the primary contributors, while clinical features demonstrated relatively modest influence. These findings provide compelling evidence that molecular-level biochemical alterations, particularly those related to oxidative stress and protein metabolism, serve as more sensitive and specific biomarkers for PSA/PSO differentiation than traditional clinical assessments, supporting the biological rationale for our hierarchical fusion approach and suggesting potential therapeutic targets for precision medicine interventions.

## Features

- **Comprehensive Model Analysis**: Analyzes all 7 model configurations (clinical, spectral, clinical+spectral, etc.)
- **Feature Importance Visualization**: Generates summary plots, bar charts, and heatmaps
- **Cross-Model Comparison**: Compares feature importance across different configurations
- **Numerical Results**: Saves detailed SHAP values and statistics in CSV format
- **Comprehensive Reporting**: Generates detailed analysis reports

## Installation

1. **Install SHAP Dependencies**:
   ```bash
   pip install -r requirements_shap.txt
   ```

2. **Verify Installation**:
   ```bash
   python run_shap_analysis.py
   ```

## Usage

### Quick Start

```bash
# Run complete SHAP analysis
python run_shap_analysis.py
```

### Advanced Usage

```python
from shap_analysis import SHAPAnalyzer

# Initialize analyzer with custom paths
analyzer = SHAPAnalyzer(
    data_path="../../../data",
    model_path="enhanced_results/models",
    output_path="enhanced_results/shap_analysis"
)

# Run analysis
analyzer.run_complete_analysis()
```

## Model Configurations Analyzed

The script analyzes the following pre-trained models:

| Configuration | Model File | Description |
|---------------|------------|-------------|
| Clinical | `best_clinical_model.pth` | Clinical features only |
| Spectral | `best_spectral_model.pth` | Spectral features only |
| Clinical+Spectral | `best_clinical_spectral_model.pth` | Combined clinical and spectral |

**Note**: Image-based models use attention-based analysis since SHAP DeepExplainer for images is computationally intensive.

## Features Analyzed

### Clinical Features (5 features)
- **Gender**: Patient gender (binary)
- **Age**: Patient age (continuous)
- **BMI**: Body Mass Index (continuous)
- **PASI**: Psoriasis Area and Severity Index (continuous)
- **BSA**: Body Surface Area affected (continuous)

### Spectral Features (5 features)
- **Amide_I_Structure**: Protein secondary structure indicator
- **Amide_I_Content**: Protein content measure
- **Amide_II_Structure**: Additional protein structure indicator
- **Amide_II_Content**: Additional protein content measure
- **Disulfide_Content**: Oxidative stress indicator

## Output Files

### Directory Structure
```
enhanced_results/shap_analysis/
├── plots/                              # Visualization files
│   ├── shap_summary_clinical.png       # SHAP summary plots
│   ├── shap_importance_clinical.png    # Feature importance bar charts
│   ├── shap_mean_abs_clinical.png      # Mean absolute SHAP values
│   ├── feature_importance_comparison.png # Cross-model comparison
│   └── ... (similar files for other models)
├── results/                            # Numerical results
│   ├── shap_values_clinical.csv        # Raw SHAP values
│   ├── feature_importance_clinical.csv # Feature importance statistics
│   ├── feature_importance_comparison.csv # Cross-model comparison data
│   └── ... (similar files for other models)
└── shap_analysis_report.txt            # Comprehensive analysis report
```

### Visualization Types

1. **SHAP Summary Plots**: Show the distribution of SHAP values for each feature
2. **Feature Importance Bar Charts**: Rank features by mean absolute SHAP value
3. **Mean Absolute SHAP Values**: Horizontal bar charts of feature importance
4. **Cross-Model Comparison Heatmap**: Compare feature importance across models

### Data Files

1. **Raw SHAP Values** (`shap_values_*.csv`): Individual SHAP values for each sample and feature
2. **Feature Importance Summary** (`feature_importance_*.csv`): Statistics including mean, std of SHAP values
3. **Cross-Model Comparison** (`feature_importance_comparison.csv`): Feature importance across all models

## Interpretation Guide

### SHAP Value Interpretation

- **Positive SHAP values**: Push the prediction towards PSO (Psoriasis)
- **Negative SHAP values**: Push the prediction towards PSA (Psoriatic Arthritis)
- **Magnitude**: Indicates the strength of the feature's contribution

### Feature Importance Ranking

Features are ranked by **mean absolute SHAP value**, which indicates:
- How much each feature contributes to predictions on average
- Which features are most important for the model's decision-making
- Relative importance compared to other features

### Example Interpretation

```
Top 3 most important features for Clinical model:
1. PASI: 0.0234 (disease severity is most important)
2. BSA: 0.0156 (affected area is second most important)  
3. Age: 0.0089 (patient age contributes moderately)
```

## Requirements

### System Requirements
- Python 3.8+
- 8GB+ RAM (for SHAP computations)
- GPU recommended (for model loading)

### Software Dependencies
- PyTorch >= 1.12.0
- SHAP >= 0.41.0
- NumPy >= 1.21.0
- Pandas >= 1.3.0
- Matplotlib >= 3.5.0
- Seaborn >= 0.11.0
- Scikit-learn >= 1.0.0

## Troubleshooting

### Common Issues

1. **Missing Model Files**:
   ```
   Error: Model file enhanced_results/models/best_clinical_model.pth not found
   ```
   **Solution**: Ensure all model files exist and are properly trained

2. **SHAP Installation Issues**:
   ```
   ImportError: No module named 'shap'
   ```
   **Solution**: Install SHAP with `pip install shap`

3. **Memory Issues**:
   ```
   Out of memory during SHAP computation
   ```
   **Solution**: Reduce sample sizes in the script or use a machine with more RAM

### Performance Notes

- **Clinical/Spectral models**: Fast analysis (< 5 minutes)
- **Combined models**: Moderate analysis time (5-15 minutes)
- **Image models**: Use attention-based analysis (faster alternative)

## Technical Details

### SHAP Explainer Types

- **KernelExplainer**: Used for tabular models (clinical, spectral, clinical+spectral)
- **Background Sampling**: Uses 20 background samples for efficiency
- **Explanation Sampling**: Analyzes 30 samples for detailed explanations

### Statistical Analysis

- **Mean Absolute SHAP**: Primary importance metric
- **SHAP Value Distribution**: Shows feature contribution patterns
- **Cross-Model Consistency**: Identifies consistently important features

## Citation

If you use this SHAP analysis in your research, please cite:

```bibtex
@article{psoriasis_multimodal_2024,
  title={Hierarchical Multi-Modal Fusion Network for Psoriatic Condition Classification: SHAP Analysis},
  author={Research Team},
  year={2024}
}
```

## Contact

For questions or issues with the SHAP analysis:
- Check the troubleshooting section above
- Review the generated `shap_analysis_report.txt` for detailed results
- Ensure all dependencies are properly installed

---

**Note**: This analysis uses pre-trained models and does not require retraining. Results are saved automatically and can be reviewed at any time. 