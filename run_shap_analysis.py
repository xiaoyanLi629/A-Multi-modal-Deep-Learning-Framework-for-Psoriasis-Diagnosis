#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SHAP Analysis Runner Script
==========================
This script runs SHAP analysis on pre-trained multimodal models.

Usage:
    python run_shap_analysis.py

Requirements:
    pip install -r requirements_shap.txt
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = {
        'torch': 'PyTorch',
        'numpy': 'NumPy', 
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'sklearn': 'Scikit-learn',
        'PIL': 'Pillow',
        'shap': 'SHAP'
    }
    
    missing_packages = []
    
    for package, name in required_packages.items():
        try:
            __import__(package)
            print(f"✓ {name} installed")
        except ImportError:
            missing_packages.append(name)
            print(f"✗ {name} not found")
    
    if missing_packages:
        print(f"\nMissing packages: {', '.join(missing_packages)}")
        print("Please install missing packages with:")
        print("pip install -r requirements_shap.txt")
        return False
    
    return True

def main():
    """Main function to run SHAP analysis"""
    print("SHAP Analysis Runner")
    print("=" * 40)
    
    # Check dependencies
    print("Checking dependencies...")
    if not check_dependencies():
        return
    
    print("\nAll dependencies found!")
    print("Starting SHAP analysis...")
    
    # Import and run SHAP analysis
    try:
        from shap_analysis import SHAPAnalyzer
        
        # Initialize and run analyzer
        analyzer = SHAPAnalyzer()
        analyzer.run_complete_analysis()
        
        print("\n" + "="*50)
        print("SHAP Analysis Completed Successfully!")
        print("="*50)
        print(f"Results saved to: {analyzer.output_path}")
        print("\nGenerated files:")
        print("- SHAP visualizations in /plots/")
        print("- Numerical results in /results/")
        print("- Comprehensive report: shap_analysis_report.txt")
        
    except Exception as e:
        print(f"Error running SHAP analysis: {e}")
        print("Please check that:")
        print("1. All model files exist in enhanced_results/models/")
        print("2. Data files exist in ../../../data/")
        print("3. All dependencies are properly installed")

if __name__ == "__main__":
    main() 