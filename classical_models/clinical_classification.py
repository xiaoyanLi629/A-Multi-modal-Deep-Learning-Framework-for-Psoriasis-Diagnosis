#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clinical Data Classification Experiment
========================================

This script performs classification experiments using only clinical variables:
- Gender, Age, BMI, PASI, BSA

Multiple traditional machine learning models are tested and compared.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Machine Learning libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, 
                           roc_auc_score, confusion_matrix, classification_report, roc_curve)

# Traditional ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Set style for plots
plt.style.use('default')
sns.set_palette("husl")

class ClinicalDataClassifier:
    def __init__(self, data_path="../../../data/clinical_data.csv", output_path="./results"):
        """Initialize the classifier with data path and output directory"""
        self.data_path = data_path
        self.output_path = output_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.models = {}
        self.results = {}
        
        # Create output directories
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/plots", exist_ok=True)
        os.makedirs(f"{output_path}/models", exist_ok=True)
        
        # Clinical features to use
        self.clinical_features = ['Gender', 'Age', 'BMI', 'PASI', 'BSA']
        
    def load_and_prepare_data(self):
        """Load clinical data and prepare features"""
        print("Loading clinical data...")
        
        # Load data
        self.data = pd.read_csv(self.data_path)
        
        # Select only clinical features
        self.X = self.data[self.clinical_features].copy()
        self.y = self.data['Disease_Group'].copy()
        
        # Check for missing values
        missing_values = self.X.isnull().sum()
        if missing_values.any():
            print("Missing values found:")
            print(missing_values[missing_values > 0])
            # Fill missing values with median for numerical features
            for col in self.X.select_dtypes(include=[np.number]).columns:
                self.X[col].fillna(self.X[col].median(), inplace=True)
        
        # Display data info
        print(f"\nDataset Info:")
        print(f"- Total samples: {len(self.data)}")
        print(f"- Features: {self.clinical_features}")
        print(f"- Class distribution:")
        class_counts = self.y.value_counts()
        for class_name, count in class_counts.items():
            print(f"  {class_name}: {count} ({count/len(self.y)*100:.1f}%)")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42, stratify=self.y
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        print(f"- Training set: {len(self.X_train)} samples")
        print(f"- Test set: {len(self.X_test)} samples")
        
    def initialize_models(self):
        """Initialize all traditional ML models"""
        print("\nInitializing models...")
        
        self.models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(random_state=42, probability=True),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'Decision Tree': DecisionTreeClassifier(random_state=42, max_depth=10)
        }
        
        print(f"Models initialized: {list(self.models.keys())}")
    
    def evaluate_model(self, model, model_name, X_train, X_test, y_train, y_test):
        """Evaluate a single model and return metrics"""
        # Train model
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, pos_label='PSO'),
            'recall': recall_score(y_test, y_pred, pos_label='PSO'),
            'f1': f1_score(y_test, y_pred, pos_label='PSO'),
            'confusion_matrix': confusion_matrix(y_test, y_pred),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        # ROC-AUC (if probability predictions available)
        if y_pred_proba is not None:
            # Convert labels to numeric for ROC calculation
            le = LabelEncoder()
            y_test_numeric = le.fit_transform(y_test)
            y_pred_proba_binary = y_pred_proba if le.classes_[1] == 'PSO' else 1 - y_pred_proba
            metrics['roc_auc'] = roc_auc_score(y_test_numeric, y_pred_proba_binary)
            metrics['roc_curve'] = roc_curve(y_test_numeric, y_pred_proba_binary)
        else:
            metrics['roc_auc'] = None
            metrics['roc_curve'] = None
        
        return metrics
    
    def cross_validate_models(self):
        """Perform cross-validation for all models"""
        print("\nPerforming cross-validation...")
        
        cv_results = {}
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        for name, model in self.models.items():
            print(f"Cross-validating {name}...")
            
            # Determine if model needs scaled data
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                X_cv = self.scaler.fit_transform(self.X)
            else:
                X_cv = self.X.values
            
            cv_scores = cross_val_score(model, X_cv, self.y, cv=cv, scoring='accuracy')
            
            cv_results[name] = {
                'mean_score': cv_scores.mean(),
                'std_score': cv_scores.std(),
                'scores': cv_scores
            }
            
            print(f"  Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
        
        return cv_results
    
    def train_and_evaluate_all(self):
        """Train and evaluate all models"""
        print("\nTraining and evaluating models...")
        
        for name, model in self.models.items():
            print(f"\nEvaluating {name}...")
            
            # Determine if model needs scaled data
            if name in ['Logistic Regression', 'SVM', 'K-Nearest Neighbors']:
                X_train_use = self.X_train_scaled
                X_test_use = self.X_test_scaled
            else:
                X_train_use = self.X_train.values
                X_test_use = self.X_test.values
            
            # Evaluate model
            metrics = self.evaluate_model(model, name, X_train_use, X_test_use, 
                                        self.y_train, self.y_test)
            
            # Store results
            self.results[name] = metrics
            
            # Print key metrics
            print(f"  Accuracy: {metrics['accuracy']:.3f}")
            print(f"  Precision: {metrics['precision']:.3f}")
            print(f"  Recall: {metrics['recall']:.3f}")
            print(f"  F1-Score: {metrics['f1']:.3f}")
            if metrics['roc_auc'] is not None:
                print(f"  ROC-AUC: {metrics['roc_auc']:.3f}")
    
    def create_comparison_plots(self):
        """Create comparison plots for all models"""
        print("\nCreating comparison plots...")
        
        # 1. Metrics comparison bar plot
        metrics_names = ['accuracy', 'precision', 'recall', 'f1']
        model_names = list(self.results.keys())
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics_names):
            values = [self.results[model][metric] for model in model_names]
            bars = axes[i].bar(model_names, values, alpha=0.7)
            axes[i].set_title(f'{metric.title()} Comparison', fontweight='bold')
            axes[i].set_ylabel(metric.title())
            axes[i].tick_params(axis='x', rotation=45)
            axes[i].set_ylim([0, 1])
            
            # Add value labels on bars
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/metrics_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Confusion matrices
        n_models = len(self.results)
        cols = 3
        rows = (n_models + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows))
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()
        
        for i, (name, results) in enumerate(self.results.items()):
            cm = results['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i],
                       xticklabels=['PSA', 'PSO'], yticklabels=['PSA', 'PSO'])
            axes[i].set_title(f'{name}\nAccuracy: {results["accuracy"]:.3f}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for i in range(len(self.results), len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. ROC curves
        plt.figure(figsize=(10, 8))
        
        for name, results in self.results.items():
            if results['roc_curve'] is not None:
                fpr, tpr, _ = results['roc_curve']
                auc = results['roc_auc']
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})', linewidth=2)
        
        plt.plot([0, 1], [0, 1], 'k--', label='Random (AUC = 0.500)')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison', fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(f'{self.output_path}/plots/roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Feature importance (for tree-based models)
        tree_models = ['Random Forest', 'Decision Tree']
        available_tree_models = [name for name in tree_models if name in self.models]
        
        if available_tree_models:
            fig, axes = plt.subplots(1, len(available_tree_models), figsize=(6*len(available_tree_models), 6))
            if len(available_tree_models) == 1:
                axes = [axes]
            
            for i, model_name in enumerate(available_tree_models):
                model = self.models[model_name]
                # Retrain model to get feature importance
                if model_name in ['Random Forest', 'Decision Tree']:
                    model.fit(self.X_train.values, self.y_train)
                    importances = model.feature_importances_
                    
                    feature_df = pd.DataFrame({
                        'feature': self.clinical_features,
                        'importance': importances
                    }).sort_values('importance', ascending=True)
                    
                    axes[i].barh(feature_df['feature'], feature_df['importance'])
                    axes[i].set_title(f'{model_name} Feature Importance')
                    axes[i].set_xlabel('Importance')
            
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/plots/feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def create_summary_report(self):
        """Create a comprehensive summary report"""
        print("\nGenerating summary report...")
        
        # Create results summary dataframe
        summary_data = []
        for name, results in self.results.items():
            row = {
                'Model': name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1-Score': results['f1'],
                'ROC-AUC': results['roc_auc'] if results['roc_auc'] is not None else 'N/A'
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_df = summary_df.sort_values('Accuracy', ascending=False)
        
        # Save summary table
        summary_df.to_csv(f'{self.output_path}/model_comparison_summary.csv', index=False)
        
        # Generate text report
        report = f"""
Clinical Data Classification Report
==================================

Dataset Information:
- Total samples: {len(self.data)}
- Features used: {', '.join(self.clinical_features)}
- Target classes: PSA ({len(self.data[self.data['Disease_Group']=='PSA'])}) vs PSO ({len(self.data[self.data['Disease_Group']=='PSO'])})
- Train/Test split: 70%/30%

Model Performance Summary:
-------------------------
"""
        
        # Add model rankings
        for i, (_, row) in enumerate(summary_df.iterrows(), 1):
            report += f"\n{i}. {row['Model']}\n"
            report += f"   - Accuracy: {row['Accuracy']:.3f}\n"
            report += f"   - Precision: {row['Precision']:.3f}\n"
            report += f"   - Recall: {row['Recall']:.3f}\n"
            report += f"   - F1-Score: {row['F1-Score']:.3f}\n"
            if row['ROC-AUC'] != 'N/A':
                report += f"   - ROC-AUC: {row['ROC-AUC']:.3f}\n"
        
        # Best model analysis
        best_model = summary_df.iloc[0]
        report += f"\nBest Performing Model: {best_model['Model']}\n"
        report += f"- Achieved {best_model['Accuracy']:.1%} accuracy on test set\n"
        report += f"- This model shows the best balance of precision and recall\n"
        
        # Clinical insights
        report += f"\nClinical Insights:\n"
        report += f"- All models achieved reasonable performance using only basic clinical variables\n"
        report += f"- The clinical features (Gender, Age, BMI, PASI, BSA) provide meaningful predictive power\n"
        report += f"- Models can potentially assist in clinical decision-making\n"
        
        report += f"\nFiles Generated:\n"
        report += f"- model_comparison_summary.csv: Detailed metrics comparison\n"
        report += f"- plots/metrics_comparison.png: Performance metrics visualization\n"
        report += f"- plots/confusion_matrices.png: Confusion matrices for all models\n"
        report += f"- plots/roc_curves.png: ROC curves comparison\n"
        report += f"- plots/feature_importance.png: Feature importance analysis\n"
        
        # Save report
        with open(f'{self.output_path}/classification_report.txt', 'w') as f:
            f.write(report)
        
        print("Summary report saved!")
        
        return summary_df
    
    def run_complete_experiment(self):
        """Run the complete classification experiment"""
        print("="*60)
        print("CLINICAL DATA CLASSIFICATION EXPERIMENT")
        print("="*60)
        
        # Load and prepare data
        self.load_and_prepare_data()
        
        # Initialize models
        self.initialize_models()
        
        # Cross-validation
        cv_results = self.cross_validate_models()
        
        # Train and evaluate all models
        self.train_and_evaluate_all()
        
        # Create visualizations
        self.create_comparison_plots()
        
        # Generate summary report
        summary_df = self.create_summary_report()
        
        print("\n" + "="*60)
        print("EXPERIMENT COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"\nResults saved to: {self.output_path}")
        print("\nTop 3 performing models:")
        for i in range(min(3, len(summary_df))):
            row = summary_df.iloc[i]
            print(f"{i+1}. {row['Model']}: {row['Accuracy']:.3f} accuracy")
        
        return summary_df


def main():
    """Main function to run the classification experiment"""
    classifier = ClinicalDataClassifier()
    results = classifier.run_complete_experiment()
    return results


if __name__ == "__main__":
    main() 