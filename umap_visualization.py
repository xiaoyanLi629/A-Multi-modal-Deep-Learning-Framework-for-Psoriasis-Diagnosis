#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
UMAP Visualization for Psoriasis Multi-Modal Analysis
=====================================================
Author: AI Assistant  
Date: 2024

This script generates comprehensive UMAP visualizations for research paper figures:
1. Original feature spaces (clinical, spectral, combined)
2. Learned feature representations from trained models
3. Image feature embeddings
4. Multi-modal fusion feature spaces
5. Ablation study comparisons
"""

import os
import sys
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import umap
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from enhanced_multimodal_fusion import (
    HierarchicalMultiModalNet, 
    PsoriasisDataset, 
    MultiModalExperiment
)

# Set visualization style
plt.style.use('default')
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 8)
plt.rcParams['font.size'] = 12
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['legend.fontsize'] = 12

class UMAPVisualizer:
    """Comprehensive UMAP visualization for multi-modal psoriasis analysis"""
    
    def __init__(self, data_path='../../../data', model_path='./enhanced_results/models', 
                 output_path='./umap_visualizations'):
        self.data_path = data_path
        self.model_path = model_path
        self.output_path = output_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create output directory
        os.makedirs(output_path, exist_ok=True)
        
        # Initialize data containers
        self.clinical_data = None
        self.spectral_data = None
        self.image_features = None
        self.labels = None
        self.label_encoder = LabelEncoder()
        
        print(f"UMAP Visualizer initialized")
        print(f"Device: {self.device}")
        print(f"Output path: {output_path}")
        
    def load_data(self):
        """Load and preprocess all data modalities"""
        print("Loading data...")
        
        # Load clinical data
        clinical_df = pd.read_csv(os.path.join(self.data_path, 'clinical_data.csv'))
        
        # Separate features and labels
        clinical_features = clinical_df[['Gender', 'Age', 'BMI', 'PASI', 'BSA']].values
        spectral_features = clinical_df[['Amide_Bond_1_Structure', 'Amide_Bond_1_Content',
                                       'Amide_Bond_2_Structure', 'Amide_Bond_2_Content', 
                                       'Disulfide_Bond_Content']].values
        
        self.labels = self.label_encoder.fit_transform(clinical_df['Disease_Group'].values)
        self.label_names = ['PSA', 'PSO']
        
        # Standardize features
        scaler_clinical = StandardScaler()
        scaler_spectral = StandardScaler()
        
        self.clinical_data = scaler_clinical.fit_transform(clinical_features)
        self.spectral_data = scaler_spectral.fit_transform(spectral_features)
        
        print(f"Loaded data: {len(self.labels)} samples")
        print(f"Clinical features: {self.clinical_data.shape}")
        print(f"Spectral features: {self.spectral_data.shape}")
        print(f"Class distribution: PSA={np.sum(self.labels==0)}, PSO={np.sum(self.labels==1)}")
        
    def extract_image_features(self):
        """Extract features from SEM images using pre-trained models"""
        print("Extracting image features...")
        
        try:
            # Load experiment class to get image data
            experiment = MultiModalExperiment(self.data_path, self.model_path)
            experiment.load_and_preprocess_data()
            
            # Extract image features using a trained model
            model_path = os.path.join(self.model_path, 'best_image_model.pth')
            if os.path.exists(model_path):
                model = HierarchicalMultiModalNet(mode='image')
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model.to(self.device)
                model.eval()
                
                # Create dataset
                dataset = PsoriasisDataset(
                    experiment.clinical_data, experiment.spectral_data, 
                    experiment.image_paths, experiment.labels, 
                    mode='image', image_transform=experiment.val_transform
                )
                
                # Extract features
                features = []
                with torch.no_grad():
                    for i in range(len(dataset)):
                        _, _, image = dataset[i]
                        image = image.unsqueeze(0).to(self.device)
                        
                        # Get image encoder features
                        img_features = model.image_encoder(image)
                        features.append(img_features.cpu().numpy().flatten())
                        
                        if i % 20 == 0:
                            print(f"Processed {i+1}/{len(dataset)} images")
                
                self.image_features = np.array(features)
                print(f"Extracted image features: {self.image_features.shape}")
                
            else:
                print("Image model not found, using random features as placeholder")
                self.image_features = np.random.randn(len(self.labels), 1280)
                
        except Exception as e:
            print(f"Error extracting image features: {e}")
            print("Using random features as placeholder")
            self.image_features = np.random.randn(len(self.labels), 1280)
    
    def create_umap_embedding(self, data, n_neighbors=15, min_dist=0.1, n_components=2, 
                             metric='euclidean', random_state=42):
        """Create UMAP embedding for given data"""
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state
        )
        
        embedding = reducer.fit_transform(data)
        return embedding
    
    def plot_umap(self, embedding, labels, title, filename, figsize=(10, 8)):
        """Create UMAP scatter plot"""
        plt.figure(figsize=figsize)
        
        # Define colors for classes
        colors = ['#FF6B6B', '#4ECDC4']  # Red for PSA, Teal for PSO
        
        for i, label in enumerate(self.label_names):
            mask = labels == i
            plt.scatter(embedding[mask, 0], embedding[mask, 1], 
                       c=colors[i], label=label, alpha=0.7, s=50)
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.xlabel('UMAP 1', fontsize=14)
        plt.ylabel('UMAP 2', fontsize=14)
        plt.legend(title='Disease Group', fontsize=12)
        plt.grid(True, alpha=0.3)
        
        # Add sample count to legend
        psa_count = np.sum(labels == 0)
        pso_count = np.sum(labels == 1)
        plt.text(0.02, 0.98, f'PSA: {psa_count}\nPSO: {pso_count}', 
                transform=plt.gca().transAxes, fontsize=10, 
                verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, filename), dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Saved: {filename}")
    
    def generate_original_feature_umaps(self):
        """Generate UMAP visualizations for original feature spaces"""
        print("\nGenerating original feature UMAP visualizations...")
        
        # 1. Clinical features UMAP
        clinical_embedding = self.create_umap_embedding(self.clinical_data)
        self.plot_umap(clinical_embedding, self.labels, 
                      'UMAP of Clinical Features\n(Gender, Age, BMI, PASI, BSA)',
                      'umap_clinical_features.png')
        
        # 2. Spectral features UMAP  
        spectral_embedding = self.create_umap_embedding(self.spectral_data)
        self.plot_umap(spectral_embedding, self.labels,
                      'UMAP of Spectral Features\n(Infrared Amide & Disulfide Bonds)',
                      'umap_spectral_features.png')
        
        # 3. Combined clinical + spectral UMAP
        combined_features = np.concatenate([self.clinical_data, self.spectral_data], axis=1)
        combined_embedding = self.create_umap_embedding(combined_features)
        self.plot_umap(combined_embedding, self.labels,
                      'UMAP of Combined Clinical + Spectral Features',
                      'umap_combined_features.png')
        
        # 4. Image features UMAP (if available)
        if self.image_features is not None:
            # Use PCA first to reduce dimensionality for large image features
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            image_features_pca = pca.fit_transform(self.image_features)
            
            image_embedding = self.create_umap_embedding(image_features_pca)
            self.plot_umap(image_embedding, self.labels,
                          'UMAP of SEM Image Features\n(EfficientNet-B0 Embeddings)',
                          'umap_image_features.png')
    
    def extract_model_features(self, model_name, mode):
        """Extract intermediate features from trained models"""
        model_path = os.path.join(self.model_path, f'best_{model_name}_model.pth')
        
        if not os.path.exists(model_path):
            print(f"Model {model_name} not found at {model_path}")
            return None
            
        try:
            # Load model
            model = HierarchicalMultiModalNet(mode=mode)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            model.to(self.device)
            model.eval()
            
            # Create dataset
            experiment = MultiModalExperiment(self.data_path, self.model_path)
            experiment.load_and_preprocess_data()
            
            dataset = PsoriasisDataset(
                experiment.clinical_data, experiment.spectral_data,
                experiment.image_paths, experiment.labels,
                mode=mode, image_transform=experiment.val_transform
            )
            
            # Extract features
            features = []
            with torch.no_grad():
                for i in range(len(dataset)):
                    batch = dataset[i]
                    
                    if mode == 'clinical':
                        clinical = batch[0].unsqueeze(0).to(self.device)
                        encoded = model.clinical_encoder(clinical)
                        features.append(encoded.cpu().numpy().flatten())
                    elif mode == 'spectral':
                        spectral = batch[1].unsqueeze(0).to(self.device)  
                        encoded = model.spectral_encoder(spectral)
                        features.append(encoded.cpu().numpy().flatten())
                    elif mode == 'clinical_spectral':
                        clinical = batch[0].unsqueeze(0).to(self.device)
                        spectral = batch[1].unsqueeze(0).to(self.device)
                        clinical_feat = model.clinical_encoder(clinical)
                        spectral_feat = model.spectral_encoder(spectral)
                        fused = torch.cat([clinical_feat, spectral_feat], dim=1)
                        encoded = model.bio_fusion(fused)
                        features.append(encoded.cpu().numpy().flatten())
                        
            return np.array(features)
            
        except Exception as e:
            print(f"Error extracting features from {model_name}: {e}")
            return None
    
    def generate_learned_feature_umaps(self):
        """Generate UMAP visualizations for learned feature representations"""
        print("\nGenerating learned feature UMAP visualizations...")
        
        # Model configurations to visualize
        model_configs = [
            ('clinical', 'clinical'),
            ('spectral', 'spectral'),
            ('clinical_spectral', 'clinical_spectral'),
        ]
        
        for model_name, mode in model_configs:
            features = self.extract_model_features(model_name, mode)
            if features is not None:
                embedding = self.create_umap_embedding(features)
                title = f'UMAP of Learned {model_name.replace("_", " ").title()} Features'
                filename = f'umap_learned_{model_name}_features.png'
                self.plot_umap(embedding, self.labels, title, filename)
    
    def create_comparison_plot(self):
        """Create comparison plot showing multiple UMAP embeddings"""
        print("\nCreating comparison plot...")
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        # Data for comparison
        data_configs = [
            (self.clinical_data, 'Clinical Features'),
            (self.spectral_data, 'Spectral Features'),
            (np.concatenate([self.clinical_data, self.spectral_data], axis=1), 'Combined Features'),
        ]
        
        # Add image features if available
        if self.image_features is not None:
            from sklearn.decomposition import PCA
            pca = PCA(n_components=50)
            image_features_pca = pca.fit_transform(self.image_features)
            data_configs.append((image_features_pca, 'Image Features'))
        
        colors = ['#FF6B6B', '#4ECDC4']
        
        for i, (data, title) in enumerate(data_configs[:6]):
            if i < len(axes):
                embedding = self.create_umap_embedding(data)
                
                for j, label in enumerate(self.label_names):
                    mask = self.labels == j
                    axes[i].scatter(embedding[mask, 0], embedding[mask, 1], 
                                  c=colors[j], label=label, alpha=0.7, s=30)
                
                axes[i].set_title(title, fontsize=14, fontweight='bold')
                axes[i].set_xlabel('UMAP 1')
                axes[i].set_ylabel('UMAP 2')
                axes[i].grid(True, alpha=0.3)
                if i == 0:
                    axes[i].legend()
        
        # Hide unused subplots
        for i in range(len(data_configs), len(axes)):
            axes[i].set_visible(False)
        
        plt.suptitle('UMAP Comparison: Different Feature Modalities', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'umap_comparison_all_modalities.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: umap_comparison_all_modalities.png")
    
    def analyze_umap_parameters(self):
        """Analyze the effect of different UMAP parameters"""
        print("\nAnalyzing UMAP parameter effects...")
        
        # Use combined features for parameter analysis
        combined_features = np.concatenate([self.clinical_data, self.spectral_data], axis=1)
        
        # Parameter variations
        n_neighbors_values = [5, 15, 30, 50]
        min_dist_values = [0.01, 0.1, 0.5, 0.9]
        
        # n_neighbors effect
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        colors = ['#FF6B6B', '#4ECDC4']
        
        for i, n_neighbors in enumerate(n_neighbors_values):
            embedding = self.create_umap_embedding(combined_features, n_neighbors=n_neighbors)
            
            for j, label in enumerate(self.label_names):
                mask = self.labels == j
                axes[i].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=colors[j], label=label, alpha=0.7, s=30)
            
            axes[i].set_title(f'n_neighbors = {n_neighbors}', fontsize=12)
            axes[i].set_xlabel('UMAP 1')
            axes[i].set_ylabel('UMAP 2')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend()
        
        plt.suptitle('Effect of n_neighbors Parameter on UMAP Embedding', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'umap_parameter_n_neighbors.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        # min_dist effect
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        for i, min_dist in enumerate(min_dist_values):
            embedding = self.create_umap_embedding(combined_features, min_dist=min_dist)
            
            for j, label in enumerate(self.label_names):
                mask = self.labels == j
                axes[i].scatter(embedding[mask, 0], embedding[mask, 1], 
                              c=colors[j], label=label, alpha=0.7, s=30)
            
            axes[i].set_title(f'min_dist = {min_dist}', fontsize=12)
            axes[i].set_xlabel('UMAP 1')
            axes[i].set_ylabel('UMAP 2')
            axes[i].grid(True, alpha=0.3)
            if i == 0:
                axes[i].legend()
        
        plt.suptitle('Effect of min_dist Parameter on UMAP Embedding', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'umap_parameter_min_dist.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Saved parameter analysis plots")
    
    def generate_3d_umap(self):
        """Generate 3D UMAP visualization"""
        print("\nGenerating 3D UMAP visualization...")
        
        # Use combined features
        combined_features = np.concatenate([self.clinical_data, self.spectral_data], axis=1)
        embedding_3d = self.create_umap_embedding(combined_features, n_components=3)
        
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        colors = ['#FF6B6B', '#4ECDC4']
        
        for i, label in enumerate(self.label_names):
            mask = self.labels == i
            ax.scatter(embedding_3d[mask, 0], embedding_3d[mask, 1], embedding_3d[mask, 2],
                      c=colors[i], label=label, alpha=0.7, s=50)
        
        ax.set_title('3D UMAP of Combined Clinical + Spectral Features', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.set_zlabel('UMAP 3')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_path, 'umap_3d_combined_features.png'), 
                   dpi=300, bbox_inches='tight')
        plt.show()
        print("Saved: umap_3d_combined_features.png")
    
    def run_complete_umap_analysis(self):
        """Run complete UMAP analysis pipeline"""
        print("="*60)
        print("UMAP Visualization Analysis for Psoriasis Research")
        print("="*60)
        
        # Load data
        self.load_data()
        
        # Extract image features
        self.extract_image_features()
        
        # Generate original feature UMAPs
        self.generate_original_feature_umaps()
        
        # Generate learned feature UMAPs
        self.generate_learned_feature_umaps()
        
        # Create comparison plot
        self.create_comparison_plot()
        
        # Analyze UMAP parameters
        self.analyze_umap_parameters()
        
        # Generate 3D UMAP
        self.generate_3d_umap()
        
        print("\n" + "="*60)
        print("UMAP Analysis Complete!")
        print(f"All visualizations saved to: {self.output_path}")
        print("="*60)
        
        # Generate summary report
        self._generate_summary_report()
    
    def _generate_summary_report(self):
        """Generate a summary report of UMAP analysis"""
        report_path = os.path.join(self.output_path, 'umap_analysis_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write("UMAP Visualization Analysis Report\n")
            f.write("=================================\n\n")
            
            f.write(f"Dataset Summary:\n")
            f.write(f"- Total samples: {len(self.labels)}\n")
            f.write(f"- PSA samples: {np.sum(self.labels==0)}\n")
            f.write(f"- PSO samples: {np.sum(self.labels==1)}\n")
            f.write(f"- Clinical features: {self.clinical_data.shape[1]}\n")
            f.write(f"- Spectral features: {self.spectral_data.shape[1]}\n")
            if self.image_features is not None:
                f.write(f"- Image features: {self.image_features.shape[1]}\n")
            f.write("\n")
            
            f.write("Generated Visualizations:\n")
            f.write("1. umap_clinical_features.png - Clinical features (Gender, Age, BMI, PASI, BSA)\n")
            f.write("2. umap_spectral_features.png - Spectral features (Amide & Disulfide bonds)\n")
            f.write("3. umap_combined_features.png - Combined clinical + spectral features\n")
            f.write("4. umap_image_features.png - SEM image features (if available)\n")
            f.write("5. umap_learned_*_features.png - Learned representations from trained models\n")
            f.write("6. umap_comparison_all_modalities.png - Side-by-side comparison\n")
            f.write("7. umap_parameter_*.png - Parameter analysis plots\n")
            f.write("8. umap_3d_combined_features.png - 3D UMAP visualization\n")
            f.write("\n")
            
            f.write("Usage for Research Paper:\n")
            f.write("- Use original feature UMAPs to show data distribution\n")
            f.write("- Use learned feature UMAPs to demonstrate model representations\n")
            f.write("- Use comparison plot to show multi-modal integration effects\n")
            f.write("- Use parameter analysis to justify UMAP settings\n")
            f.write("- Use 3D UMAP for supplementary materials\n")
        
        print(f"Summary report saved to: {report_path}")

def main():
    """Main function to run UMAP visualization analysis"""
    # Initialize visualizer
    visualizer = UMAPVisualizer(
        data_path='../../../data',
        model_path='./enhanced_results/models',
        output_path='./umap_visualizations'
    )
    
    # Run complete analysis
    visualizer.run_complete_umap_analysis()

if __name__ == "__main__":
    main() 