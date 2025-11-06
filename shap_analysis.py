#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis for Hierarchical Multi-Modal Fusion Network
========================================================
Author: AI Assistant  
Date: 2024

This script performs SHAP (SHapley Additive exPlanations) analysis on pre-trained 
multimodal models to provide feature importance explanations and interpretability.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import os
import glob
import warnings
warnings.filterwarnings('ignore')

# SHAP imports
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    print("Warning: SHAP not installed. Please install with: pip install shap")
    SHAP_AVAILABLE = False

from sklearn.preprocessing import StandardScaler

# Import model architecture
from enhanced_multimodal_fusion import HierarchicalMultiModalNet, PsoriasisDataset

class SHAPAnalyzer:
    """Comprehensive SHAP analysis for multimodal psoriasis classification models"""
    
    def __init__(self, data_path="../../../data", model_path="enhanced_results/models", 
                 output_path="enhanced_results/shap_analysis"):
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directories
        os.makedirs(f"{output_path}/plots", exist_ok=True)
        os.makedirs(f"{output_path}/results", exist_ok=True)
        
        # Model configurations
        self.model_configs = {
            'clinical': 'best_clinical_model.pth',
            'spectral': 'best_spectral_model.pth', 
            'image': 'best_image_model.pth',
            'clinical_spectral': 'best_clinical_spectral_model.pth',
            'clinical_image': 'best_clinical_image_model.pth',
            'spectral_image': 'best_spectral_image_model.pth',
            'tri_modal': 'best_tri_modal_model.pth'
        }
        
        # Feature names (matching actual column names in data)
        self.clinical_features = ['Gender', 'Age', 'BMI', 'PASI', 'BSA']
        self.spectral_features = ['Amide_Bond_1_Structure', 'Amide_Bond_1_Content', 
                                'Amide_Bond_2_Structure', 'Amide_Bond_2_Content', 'Disulfide_Bond_Content']
        
        # Clean feature names for display (remove numbers, underscores, and Roman numerals)
        self.clinical_features_display = ['Gender', 'Age', 'BMI', 'PASI', 'BSA']
        self.spectral_features_display = ['Amide Structure 1', 'Amide Content 1', 
                                        'Amide Structure 2', 'Amide Content 2', 'Disulfide Content']
        
        # Initialize SHAP results storage
        self.shap_results = {}
        
        # Load data
        self._load_data()
    
    def _clean_feature_names_for_display(self, feature_names):
        """Clean feature names for better display in plots"""
        cleaned_names = []
        for name in feature_names:
            # Create mapping for clean display names
            if name in self.clinical_features:
                idx = self.clinical_features.index(name)
                cleaned_names.append(self.clinical_features_display[idx])
            elif name in self.spectral_features:
                idx = self.spectral_features.index(name)
                cleaned_names.append(self.spectral_features_display[idx])
            else:
                # Fallback: general cleaning
                clean_name = name.replace('_', ' ').replace('Bond', '').replace('1', 'I').replace('2', 'II')
                clean_name = ' '.join(clean_name.split())  # Remove extra spaces
                cleaned_names.append(clean_name)
        return cleaned_names
        
    def _load_data(self):
        """Load and preprocess the dataset"""
        print("Loading dataset for SHAP analysis...")
        
        # Load clinical data - it already contains all the data we need
        clinical_data = pd.read_csv(f"{self.data_path}/clinical_data.csv")
        
        # The clinical_data.csv already contains both clinical and spectral features
        # No need to merge with PSAPSO.xlsx since all data is in one file
        merged_data = clinical_data.copy()
        
        # Remove rows with missing values
        merged_data = merged_data.dropna()
        
        print(f"Loaded {len(merged_data)} complete samples")
        
        # Prepare features using correct column names
        self.clinical_data = merged_data[['Gender', 'Age', 'BMI', 'PASI', 'BSA']].values
        
        # Use correct spectral column names from the data file
        self.spectral_data = merged_data[['Amide_Bond_1_Structure', 'Amide_Bond_1_Content', 
                                        'Amide_Bond_2_Structure', 'Amide_Bond_2_Content', 
                                        'Disulfide_Bond_Content']].values
        
        # Use correct group column name
        self.labels = merged_data['Disease_Group'].map({'PSA': 0, 'PSO': 1}).values
        
        # Standardize features
        self.clinical_scaler = StandardScaler()
        self.spectral_scaler = StandardScaler()
        
        self.clinical_data_scaled = self.clinical_scaler.fit_transform(self.clinical_data)
        self.spectral_data_scaled = self.spectral_scaler.fit_transform(self.spectral_data)
        
        print("Data preprocessing completed")
    
    def _load_model(self, config_name):
        """Load a pre-trained model"""
        model_file = f"{self.model_path}/{self.model_configs[config_name]}"
        
        if not os.path.exists(model_file):
            print(f"Warning: Model file {model_file} not found")
            return None
            
        # Initialize model
        model = HierarchicalMultiModalNet(mode=config_name)
        
        # Load state dict
        try:
            checkpoint = torch.load(model_file, map_location=self.device)
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
            
            model.to(self.device)
            model.eval()
            print(f"Successfully loaded {config_name} model")
            return model
            
        except Exception as e:
            print(f"Error loading {config_name} model: {e}")
            return None
    
    def _create_model_wrapper(self, model, mode):
        """Create a wrapper function for SHAP analysis"""
        
        def model_wrapper(data_array):
            """Wrapper function that handles different input modalities"""
            model.eval()
            predictions = []
            
            with torch.no_grad():
                for i in range(len(data_array)):
                    inputs = {}
                    
                    # Prepare inputs based on mode
                    if mode == 'clinical':
                        inputs['clinical'] = torch.FloatTensor(data_array[i:i+1]).to(self.device)
                    elif mode == 'spectral':
                        inputs['spectral'] = torch.FloatTensor(data_array[i:i+1]).to(self.device)
                    elif mode == 'clinical_spectral':
                        clinical_part = data_array[i:i+1, :5]
                        spectral_part = data_array[i:i+1, 5:]
                        inputs['clinical'] = torch.FloatTensor(clinical_part).to(self.device)
                        inputs['spectral'] = torch.FloatTensor(spectral_part).to(self.device)
                    
                    # Get prediction
                    output = model(**inputs)
                    prob = torch.sigmoid(output).cpu().numpy()
                    predictions.append(prob[0, 0])
            
            return np.array(predictions)
        
        return model_wrapper
    
    def analyze_tabular_models(self):
        """Analyze models with tabular data (clinical, spectral, clinical+spectral)"""
        if not SHAP_AVAILABLE:
            print("SHAP not available, skipping tabular analysis")
            return
            
        print("\n" + "="*60)
        print("SHAP Analysis for Tabular Models")
        print("="*60)
        
        tabular_configs = ['clinical', 'spectral', 'clinical_spectral']
        
        for config in tabular_configs:
            print(f"\nAnalyzing {config} model...")
            
            # Load model
            model = self._load_model(config)
            if model is None:
                continue
            
            # Prepare data
            if config == 'clinical':
                X = self.clinical_data_scaled
                feature_names = self.clinical_features
            elif config == 'spectral':
                X = self.spectral_data_scaled
                feature_names = self.spectral_features
            elif config == 'clinical_spectral':
                X = np.concatenate([self.clinical_data_scaled, self.spectral_data_scaled], axis=1)
                feature_names = self.clinical_features + self.spectral_features
            
            # Create model wrapper
            model_func = self._create_model_wrapper(model, config)
            
            try:
                # Create SHAP explainer
                background_size = min(20, len(X))
                background_indices = np.random.choice(len(X), background_size, replace=False)
                background = X[background_indices]
                
                explainer = shap.KernelExplainer(model_func, background)
                
                # Calculate SHAP values for a subset of samples
                explain_size = min(30, len(X))
                explain_indices = np.random.choice(len(X), explain_size, replace=False)
                explain_data = X[explain_indices]
                
                print(f"Computing SHAP values for {explain_size} samples...")
                shap_values = explainer.shap_values(explain_data)
                
                # Store results
                self.shap_results[config] = {
                    'shap_values': shap_values,
                    'feature_names': feature_names,
                    'feature_names_display': self._clean_feature_names_for_display(feature_names),
                    'data': explain_data,
                    'expected_value': explainer.expected_value
                }
                
                # Generate visualizations
                feature_names_display = self._clean_feature_names_for_display(feature_names)
                self._create_tabular_visualizations(config, shap_values, feature_names, feature_names_display, explain_data)
                
            except Exception as e:
                print(f"Error in SHAP analysis for {config}: {e}")
                continue
    
    def _create_tabular_visualizations(self, config, shap_values, feature_names, feature_names_display, data):
        """Create SHAP visualizations for tabular data"""
        try:
            plt.style.use('default')
            
            # 1. Summary plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data, feature_names=feature_names_display, show=False)
            plt.title(f'SHAP Summary Plot - {config.replace("_", " ").title()} Model')
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/plots/shap_summary_{config}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 2. Feature importance plot
            plt.figure(figsize=(10, 6))
            shap.summary_plot(shap_values, data, feature_names=feature_names_display, plot_type="bar", show=False)
            plt.title(f'SHAP Feature Importance - {config.replace("_", " ").title()} Model')
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/plots/shap_importance_{config}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 3. Mean absolute SHAP values
            mean_shap = np.abs(shap_values).mean(axis=0)
            
            plt.figure(figsize=(10, 6))
            indices = np.argsort(mean_shap)[::-1]
            plt.barh(range(len(mean_shap)), mean_shap[indices])
            plt.yticks(range(len(mean_shap)), [feature_names_display[i] for i in indices])
            plt.xlabel('Mean |SHAP Value|')
            plt.title(f'Mean Absolute SHAP Values - {config.replace("_", " ").title()} Model')
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/plots/shap_mean_abs_{config}.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # 4. Save numerical results
            shap_df = pd.DataFrame(shap_values, columns=feature_names_display)
            shap_df.to_csv(f'{self.output_path}/results/shap_values_{config}.csv', index=False)
            
            # Feature importance summary
            importance_df = pd.DataFrame({
                'feature': feature_names_display,
                'mean_abs_shap': mean_shap,
                'mean_shap': shap_values.mean(axis=0),
                'std_shap': shap_values.std(axis=0)
            }).sort_values('mean_abs_shap', ascending=False)
            
            importance_df.to_csv(f'{self.output_path}/results/feature_importance_{config}.csv', index=False)
            
            print(f"SHAP analysis completed for {config} model")
            
        except Exception as e:
            print(f"Error creating visualizations for {config}: {e}")
    
    def create_comparative_analysis(self):
        """Create comparative analysis across all models"""
        print("\n" + "="*60)
        print("Creating Comparative SHAP Analysis")
        print("="*60)
        
        # Compare feature importance across tabular models
        tabular_models = ['clinical', 'spectral', 'clinical_spectral']
        available_models = [m for m in tabular_models if m in self.shap_results]
        
        if len(available_models) >= 2:
            self._compare_feature_importance(available_models)
        
        # Create comprehensive report
        self._generate_shap_report()
    
    def _compare_feature_importance(self, models):
        """Compare feature importance across models"""
        
        # Collect feature importance data
        importance_data = []
        
        for model in models:
            if model in self.shap_results:
                shap_values = self.shap_results[model]['shap_values']
                feature_names_display = self.shap_results[model]['feature_names_display']
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                
                for i, feature in enumerate(feature_names_display):
                    importance_data.append({
                        'model': model,
                        'feature': feature,
                        'importance': mean_abs_shap[i]
                    })
        
        if importance_data:
            importance_df = pd.DataFrame(importance_data)
            
            # Create comparison plot
            plt.figure(figsize=(15, 8))
            
            # Pivot for heatmap
            pivot_df = importance_df.pivot(index='feature', columns='model', values='importance')
            
            sns.heatmap(pivot_df, annot=True, cmap='viridis', fmt='.3f')
            plt.title('Feature Importance Comparison Across Models')
            plt.ylabel('Features')
            plt.xlabel('Models')
            plt.tight_layout()
            plt.savefig(f'{self.output_path}/plots/feature_importance_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
            
            # Save comparison data
            importance_df.to_csv(f'{self.output_path}/results/feature_importance_comparison.csv', index=False)
            
            print("Feature importance comparison completed")
    
    def _generate_shap_report(self):
        """Generate comprehensive SHAP analysis report"""
        
        report = []
        report.append("# SHAP Analysis Report")
        report.append("=" * 50)
        report.append(f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        report.append("## Summary")
        report.append(f"- Total models analyzed: {len(self.shap_results)}")
        report.append(f"- Dataset size: {len(self.labels)} samples")
        report.append(f"- Class distribution: PSA: {sum(self.labels == 0)}, PSO: {sum(self.labels == 1)}")
        report.append("")
        
        report.append("## Model-Specific Results")
        
        for model_name, results in self.shap_results.items():
            report.append(f"\n### {model_name.replace('_', ' ').title()} Model")
            
            if 'shap_values' in results:
                shap_values = results['shap_values']
                feature_names_display = results['feature_names_display']
                
                # Top 3 most important features
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_indices = np.argsort(mean_abs_shap)[::-1][:3]
                
                report.append("Top 3 most important features:")
                for i, idx in enumerate(top_indices):
                    report.append(f"{i+1}. {feature_names_display[idx]}: {mean_abs_shap[idx]:.4f}")
                
                report.append(f"- SHAP values shape: {shap_values.shape}")
                report.append(f"- Expected value: {results['expected_value']:.4f}")
        
        report.append("\n## Files Generated")
        report.append("### Plots:")
        
        if os.path.exists(f"{self.output_path}/plots"):
            plot_files = os.listdir(f"{self.output_path}/plots")
            for file in sorted(plot_files):
                if file.endswith('.png'):
                    report.append(f"- {file}")
        
        report.append("\n### Data Files:")
        if os.path.exists(f"{self.output_path}/results"):
            result_files = os.listdir(f"{self.output_path}/results")
            for file in sorted(result_files):
                if file.endswith('.csv'):
                    report.append(f"- {file}")
        
        # Save report
        with open(f"{self.output_path}/shap_analysis_report.txt", 'w') as f:
            f.write('\n'.join(report))
        
        print(f"\nSHAP analysis report saved to {self.output_path}/shap_analysis_report.txt")
    
    def run_complete_analysis(self):
        """Run complete SHAP analysis for all models"""
        print("Starting Comprehensive SHAP Analysis")
        print("=" * 60)
        
        if not SHAP_AVAILABLE:
            print("Error: SHAP library not installed. Please install with: pip install shap")
            return
        
        # Analyze tabular models
        self.analyze_tabular_models()
        
        # Create comparative analysis
        self.create_comparative_analysis()
        
        print("\n" + "="*60)
        print("SHAP Analysis Complete!")
        print("="*60)
        print(f"Results saved to: {self.output_path}")

def main():
    """Main function to run SHAP analysis"""
    
    # Check if SHAP is installed
    try:
        import shap
        print(f"SHAP version: {shap.__version__}")
    except ImportError:
        print("Error: SHAP library not found. Please install it using:")
        print("pip install shap")
        return
    
    # Initialize analyzer
    analyzer = SHAPAnalyzer()
    
    # Run analysis
    analyzer.run_complete_analysis()

if __name__ == "__main__":
    main() 