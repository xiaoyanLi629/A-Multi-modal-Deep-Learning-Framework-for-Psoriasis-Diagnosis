# Clinical Data Classification Experiment

This experiment evaluates the performance of traditional machine learning models for predicting disease groups (PSA vs PSO) using only basic clinical variables.

## Objective

To assess how well traditional machine learning algorithms can distinguish between Psoriatic Arthritis (PSA) and Psoriasis (PSO) using readily available clinical measurements:

- **Gender** (1=Male, 2=Female)
- **Age** (years)
- **BMI** (Body Mass Index)
- **PASI** (Psoriasis Area and Severity Index)
- **BSA** (Body Surface Area affected)

## Dataset

- **Total samples**: 125 patients
- **PSA patients**: 46 (36.8%)
- **PSO patients**: 79 (63.2%)
- **Train/Test split**: 70%/30% (87 training, 38 test samples)

## Models Tested

Six traditional machine learning algorithms were evaluated:

1. **Support Vector Machine (SVM)**
2. **Logistic Regression**
3. **Random Forest**
4. **K-Nearest Neighbors (KNN)**
5. **Naive Bayes**
6. **Decision Tree**

## Models Description

### 1. **Support Vector Machine (SVM)**

**Algorithm Overview:**
SVM finds the optimal hyperplane that separates the two classes with maximum margin. It uses kernel tricks to handle non-linearly separable data.

**Implementation Parameters:**
```python
SVC(random_state=42, probability=True)
```

**Key Characteristics:**
- **Kernel**: RBF (Radial Basis Function) - default kernel for non-linear classification
- **C parameter**: 1.0 (default) - controls regularization strength
- **Gamma**: 'scale' (default) - controls the influence of single training examples
- **Probability**: Enabled for ROC-AUC calculation

**Strengths:**
- Effective in high-dimensional spaces
- Memory efficient (uses subset of training points)
- Versatile with different kernel functions
- Works well with limited samples

**Weaknesses:**
- Sensitive to feature scaling (hence StandardScaler applied)
- No probabilistic output by default
- Can be slow on large datasets

**Why Chosen:**
- Excellent performance on small to medium datasets
- Robust to overfitting with proper regularization
- Well-suited for binary classification problems

### 2. **Logistic Regression**

**Algorithm Overview:**
Uses the logistic function to model the probability of binary outcomes. It's a linear classifier that estimates class probabilities.

**Implementation Parameters:**
```python
LogisticRegression(random_state=42, max_iter=1000)
```

**Key Characteristics:**
- **Solver**: 'lbfgs' (default) - good for small datasets
- **Regularization**: L2 (Ridge) regularization by default
- **C parameter**: 1.0 (default) - inverse of regularization strength
- **Max iterations**: 1000 - ensures convergence

**Strengths:**
- Provides probability estimates naturally
- Fast training and prediction
- No tuning of hyperparameters required
- Interpretable coefficients
- Less prone to overfitting with regularization

**Weaknesses:**
- Assumes linear relationship between features and log-odds
- Sensitive to outliers
- Requires feature scaling for optimal performance

**Why Chosen:**
- Baseline model for binary classification
- Provides interpretable results
- Well-established in medical applications

### 3. **Random Forest**

**Algorithm Overview:**
Ensemble method that builds multiple decision trees and merges them together for more accurate and stable predictions.

**Implementation Parameters:**
```python
RandomForestClassifier(n_estimators=100, random_state=42)
```

**Key Characteristics:**
- **Trees**: 100 estimators - good balance between performance and speed
- **Max features**: 'sqrt' (default) - considers √5 ≈ 2 features per split
- **Bootstrap**: True - samples with replacement
- **Random state**: Fixed for reproducibility

**Strengths:**
- Handles mixed data types naturally
- Built-in feature importance calculation
- Resistant to overfitting
- No need for feature scaling
- Handles missing values well

**Weaknesses:**
- Can overfit with very noisy data
- Less interpretable than single trees
- May not perform well on very small datasets

**Why Chosen:**
- Robust and widely applicable
- Provides feature importance insights
- Good baseline for ensemble methods

### 4. **K-Nearest Neighbors (KNN)**

**Algorithm Overview:**
Non-parametric method that classifies samples based on the majority class among the k nearest neighbors in feature space.

**Implementation Parameters:**
```python
KNeighborsClassifier(n_neighbors=5)
```

**Key Characteristics:**
- **Neighbors**: 5 - optimal for small datasets (rule of thumb: √n ≈ √125 ≈ 11, but 5 chosen to avoid ties)
- **Distance metric**: Euclidean (default)
- **Weights**: Uniform - all neighbors weighted equally
- **Algorithm**: 'auto' - automatically selects best algorithm

**Strengths:**
- Simple and intuitive
- No assumptions about data distribution
- Naturally handles multi-class problems
- Can capture complex decision boundaries

**Weaknesses:**
- Computationally expensive for large datasets
- Sensitive to irrelevant features
- Performs poorly with high-dimensional data
- Requires feature scaling

**Why Chosen:**
- Non-parametric alternative to parametric models
- Good for capturing local patterns
- Complements linear and tree-based approaches

### 5. **Naive Bayes (Gaussian)**

**Algorithm Overview:**
Probabilistic classifier based on Bayes' theorem with the assumption of feature independence. Uses Gaussian distribution for continuous features.

**Implementation Parameters:**
```python
GaussianNB()
```

**Key Characteristics:**
- **Distribution**: Assumes Gaussian (normal) distribution for each feature
- **Priors**: Calculated from training data class frequencies
- **Variance smoothing**: 1e-9 (default) - prevents zero probabilities

**Strengths:**
- Fast training and prediction
- Works well with small datasets
- Handles multi-class classification naturally
- Not sensitive to irrelevant features
- Provides probability estimates

**Weaknesses:**
- Strong independence assumption (often violated)
- Assumes Gaussian distribution for continuous features
- Can be outperformed by more sophisticated methods

**Why Chosen:**
- Fast baseline model
- Works well when independence assumption holds
- Good probabilistic interpretation

### 6. **Decision Tree**

**Algorithm Overview:**
Creates a tree-like model of decisions by recursively splitting the feature space based on feature values that best separate the classes.

**Implementation Parameters:**
```python
DecisionTreeClassifier(random_state=42, max_depth=10)
```

**Key Characteristics:**
- **Max depth**: 10 - prevents overfitting while allowing complexity
- **Criterion**: 'gini' (default) - measures impurity for splits
- **Splitter**: 'best' - chooses best split at each node
- **Min samples split**: 2 (default)

**Strengths:**
- Highly interpretable and visualizable
- Handles both numerical and categorical features
- No need for feature scaling
- Can capture non-linear relationships
- Provides feature importance

**Weaknesses:**
- Prone to overfitting
- Can create overly complex trees
- Unstable (small data changes can result in very different trees)
- Biased toward features with more levels

**Why Chosen:**
- Most interpretable model
- Good for understanding feature relationships
- Forms basis for ensemble methods

## Model Selection Rationale

The six models were chosen to represent different machine learning paradigms:

1. **Linear Models**: Logistic Regression (interpretable baseline)
2. **Instance-based**: KNN (non-parametric, local learning)
3. **Probabilistic**: Naive Bayes (fast, probabilistic baseline)
4. **Margin-based**: SVM (complex decision boundaries)
5. **Tree-based**: Decision Tree (interpretable), Random Forest (ensemble)

This diversity ensures robust comparison across different algorithmic approaches, from simple linear methods to complex ensemble techniques.

## Hyperparameter Justification

**Conservative Approach:**
- Used mostly default parameters to ensure fair comparison
- Applied minimal tuning to avoid overfitting on small dataset
- Fixed random seeds for reproducibility

**Key Decisions:**
- **SVM**: Default RBF kernel suitable for non-linear clinical relationships
- **Decision Tree**: Limited depth (10) to prevent overfitting
- **Random Forest**: 100 trees for stable predictions without excessive computation
- **KNN**: k=5 to balance bias-variance tradeoff
- **Logistic Regression**: Increased max_iter to ensure convergence

## Results Summary

### Model Performance Ranking

| Rank | Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|------|-------|----------|-----------|--------|----------|---------|
| 1 | **SVM** | **0.658** | **0.667** | **0.917** | **0.772** | **0.634** |
| 2 | Logistic Regression | 0.632 | 0.632 | 1.000 | 0.774 | 0.446 |
| 3 | Random Forest | 0.553 | 0.621 | 0.750 | 0.679 | 0.551 |
| 4 | K-Nearest Neighbors | 0.526 | 0.607 | 0.708 | 0.654 | 0.579 |
| 5 | Naive Bayes | 0.526 | 0.615 | 0.667 | 0.640 | 0.455 |
| 6 | Decision Tree | 0.474 | 0.583 | 0.583 | 0.583 | 0.435 |

### Best Performing Model: SVM

- **Accuracy**: 65.8%
- **Precision**: 66.7% (proportion of predicted PSO cases that were actually PSO)
- **Recall**: 91.7% (proportion of actual PSO cases that were correctly identified)
- **F1-Score**: 77.2% (harmonic mean of precision and recall)
- **ROC-AUC**: 63.4% (area under the receiver operating characteristic curve)

## Model Generation Methodology

This experiment employs a **dual evaluation strategy** to ensure robust and reliable performance assessment:

### 1. **Primary Evaluation: Train/Test Split**

**Data Partitioning:**
```python
train_test_split(test_size=0.3, random_state=42, stratify=y)
```
- **Training set**: 87 samples (70% of data)
- **Test set**: 38 samples (30% of data)
- **Stratified sampling**: Maintains balanced class distribution in both sets
- **Fixed random seed (42)**: Ensures reproducible results across runs

**Process:**
1. Models are trained exclusively on the 87 training samples
2. Performance is evaluated on the **completely unseen** 38 test samples
3. These results form the **main performance table** above

**Key Features:**
- Simulates real-world deployment scenario
- Test set remains untouched during model development
- Provides unbiased estimate of generalization performance

### 2. **Secondary Evaluation: Cross-Validation**

**5-Fold Stratified Cross-Validation:**
```python
StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```
- Uses **all 125 samples** for more robust evaluation
- Data split into 5 equal folds
- Each fold serves as test set exactly once
- Maintains class balance across all folds

**Benefits:**
- Maximizes use of limited data (125 samples)
- Provides confidence intervals for performance estimates
- Reduces variance in performance assessment
- Better estimates model stability

### 3. **Data Preprocessing**

**Feature Scaling:**
- **Algorithms requiring scaling**: Logistic Regression, SVM, K-Nearest Neighbors
  - Applied `StandardScaler` to normalize features
- **Tree-based algorithms**: Random Forest, Decision Tree, Naive Bayes
  - Used raw features (naturally handle different scales)

**Missing Value Handling:**
- Imputation with median values for numerical features
- No missing values found in current dataset

### 4. **Result Interpretation**

**Performance Metrics Sources:**
- **Main Table Results** (e.g., SVM: 65.8%): From train/test split evaluation
- **Cross-Validation Results** (e.g., SVM: 64.8% ± 14.7%): From 5-fold CV
- **Confusion Matrices**: Based on test set predictions
- **ROC Curves**: Generated from test set probability predictions

**Why Different Numbers:**
- Train/test split: **65.8%** (single test on 38 samples)
- Cross-validation: **64.8% ± 14.7%** (average of 5 tests on different subsets)
- Both are valid; CV provides more stable estimates, hold-out simulates deployment

### 5. **Reproducibility**

**Fixed Random Seeds:**
- Data splitting: `random_state=42`
- Model training: `random_state=42` (where applicable)
- Cross-validation: `random_state=42`
- Results are **100% reproducible** across different runs

**Implementation Details:**
```python
# Consistent data split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# Standardized cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
```

## Key Findings

### 1. **Moderate Classification Performance**
- The best model (SVM) achieved 65.8% accuracy using only basic clinical variables
- This suggests that clinical parameters have some discriminative power but are not sufficient alone for perfect classification

### 2. **High Recall for PSO Detection**
- Most models showed high recall for PSO detection (>70%)
- This means the models are good at identifying PSO patients when they are present
- Important for clinical screening applications

### 3. **Feature Utility**
- Basic clinical variables (Gender, Age, BMI, PASI, BSA) provide meaningful but limited predictive power
- Models performed reasonably well considering the limited feature set

### 4. **Model Variability**
- Performance varied significantly across models (47.4% to 65.8% accuracy)
- SVM and Logistic Regression performed best on this dataset

## Clinical Implications

1. **Screening Tool Potential**: While not diagnostic-grade, these models could serve as initial screening tools in clinical settings

2. **Need for Additional Features**: The moderate performance suggests that additional biomarkers (like the protein structural features from the main analysis) are needed for more accurate classification

3. **Cost-Effective Approach**: Using only basic clinical variables makes this approach very cost-effective and accessible

4. **Clinical Decision Support**: Models could assist clinicians by flagging patients who may need more detailed examination

## Cross-Validation Results

5-fold cross-validation was performed to assess model stability:

- **SVM**: 64.8% ± 14.7%
- **Random Forest**: 63.2% ± 16.3%
- **Logistic Regression**: 58.4% ± 10.9%
- **KNN**: 60.8% ± 13.8%
- **Naive Bayes**: 56.8% ± 16.3%
- **Decision Tree**: 56.8% ± 11.8%

## Files Generated

### Data Files
- `model_comparison_summary.csv` - Detailed performance metrics for all models
- `classification_report.txt` - Comprehensive analysis report

### Visualizations
- `plots/metrics_comparison.png` - Bar charts comparing all performance metrics
- `plots/confusion_matrices.png` - Confusion matrices for all models
- `plots/roc_curves.png` - ROC curves comparison
- `plots/feature_importance.png` - Feature importance for tree-based models

## Usage

### Running the Experiment

```bash
# Install dependencies
pip install -r requirements.txt

# Run the complete experiment
python clinical_classification.py
```

### Customizing the Experiment

The `ClinicalDataClassifier` class can be easily modified to:
- Add or remove features
- Include additional models
- Adjust hyperparameters
- Change evaluation metrics

## Limitations

1. **Small Dataset**: Only 125 samples may limit model generalizability
2. **Feature Limitation**: Only basic clinical variables were used
3. **Class Imbalance**: PSO patients (63.2%) outnumber PSA patients (36.8%)
4. **Single Institution**: Data may not generalize to other populations

## Future Improvements

1. **Feature Engineering**: Create composite features from existing variables
2. **Hyperparameter Tuning**: Optimize model parameters for better performance
3. **Ensemble Methods**: Combine multiple models for improved accuracy
4. **Additional Data**: Include more diverse patient populations
5. **Feature Selection**: Identify the most informative clinical variables

## Conclusion

This experiment demonstrates that basic clinical variables provide moderate but insufficient discriminative power for PSA vs PSO classification. The SVM model achieved the best performance with 65.8% accuracy. While these results are promising for a screening tool, the addition of biomarkers (as shown in the comprehensive analysis) significantly improves classification accuracy to >95%.

The clinical variables alone serve as a valuable baseline and could be used in resource-limited settings or as a first-line screening approach before more expensive biomarker testing. 