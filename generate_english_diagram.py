#!/usr/bin/env python3

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_model_architecture_diagram():
    """Generate hierarchical multimodal fusion network architecture diagram in English"""
    
    # Create figure
    fig, ax = plt.subplots(1, 1, figsize=(16, 12))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 12)
    ax.axis('off')
    
    # Define colors
    colors = {
        'clinical': '#FF6B6B',      # Red - Clinical
        'spectral': '#4ECDC4',      # Cyan - Spectral  
        'image': '#45B7D1',         # Blue - Image
        'fusion': '#96CEB4',        # Green - Fusion
        'classifier': '#FFEAA7',    # Yellow - Classifier
        'attention': '#DDA0DD'      # Purple - Attention
    }
    
    # Draw input data layer
    def draw_input_box(x, y, width, height, text, color, details):
        # Main box
        box = FancyBboxPatch((x, y), width, height, 
                           boxstyle="round,pad=0.1", 
                           facecolor=color, edgecolor='black', linewidth=2)
        ax.add_patch(box)
        
        # Title
        ax.text(x + width/2, y + height - 0.3, text, 
               ha='center', va='center', fontsize=12, fontweight='bold')
        
        # Details
        for i, detail in enumerate(details):
            ax.text(x + width/2, y + height - 0.8 - i*0.3, detail,
                   ha='center', va='center', fontsize=9)
    
    # Draw processing layer
    def draw_process_box(x, y, width, height, text, color, details=None):
        box = FancyBboxPatch((x, y), width, height,
                           boxstyle="round,pad=0.05",
                           facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        
        ax.text(x + width/2, y + height/2, text,
               ha='center', va='center', fontsize=10, fontweight='bold')
        
        if details:
            ax.text(x + width/2, y + height/2 - 0.2, details,
                   ha='center', va='center', fontsize=8)
    
    # Draw connection arrows
    def draw_arrow(start_x, start_y, end_x, end_y, color='black', style='->', linewidth=2):
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle=style, color=color, linewidth=linewidth))
    
    # Title
    ax.text(8, 11.5, 'Hierarchical Multimodal Fusion Network Architecture', 
           ha='center', va='center', fontsize=16, fontweight='bold')
    
    # Input data layer (top)
    draw_input_box(1, 9, 3, 1.5, 'Clinical Data', colors['clinical'],
                  ['5 Features', 'Gender, Age, BMI', 'PASI, BSA'])
    
    draw_input_box(6, 9, 3, 1.5, 'Spectral Data', colors['spectral'],
                  ['5 Features', 'Amide Bond Content', 'Disulfide Bond'])
    
    draw_input_box(11, 9, 3, 1.5, 'SEM Images', colors['image'],
                  ['224×224×3', 'Dorsal + Ventral', 'Dual-view Fusion'])
    
    # Encoder layers
    draw_process_box(1, 7, 3, 1, 'Clinical Encoder', colors['clinical'], '32→16 dim')
    draw_process_box(6, 7, 3, 1, 'Spectral Encoder', colors['spectral'], '64→32→16 dim')
    draw_process_box(11, 6.5, 3, 2, 'Image Encoder', colors['image'], 
                    'EfficientNet-B0\n+ Attention\n→ 64 dim')
    
    # Attention module
    draw_process_box(11, 5, 3, 1, 'Attention Weights', colors['attention'], '1280→64 dim')
    
    # Single-modal classifiers
    draw_process_box(0.5, 5, 2, 0.8, 'Clinical\nClassifier', colors['classifier'])
    draw_process_box(3, 5, 2, 0.8, 'Spectral\nClassifier', colors['classifier'])
    draw_process_box(10.5, 3.5, 2, 0.8, 'Image\nClassifier', colors['classifier'])
    
    # Bi-modal fusion layers
    draw_process_box(2, 3, 3, 1, 'Clinical+Spectral Fusion', colors['fusion'], '32→24 dim')
    draw_process_box(6, 3, 3, 1, 'Clinical+Image Fusion', colors['fusion'], '80→48 dim')
    draw_process_box(10, 3, 3, 1, 'Spectral+Image Fusion', colors['fusion'], '80→48 dim')
    
    # Tri-modal hierarchical fusion
    draw_process_box(7, 1.5, 4, 1, 'Biological Fusion Layer', colors['fusion'], 
                    'Spectral(16) + Image(64) → 32 dim')
    draw_process_box(7, 0.2, 4, 1, 'Final Fusion Layer', colors['fusion'],
                    'Bio Features(32) + Clinical(16) → 1 dim')
    
    # Draw connection arrows
    # Input to encoders
    draw_arrow(2.5, 9, 2.5, 8, colors['clinical'])
    draw_arrow(7.5, 9, 7.5, 8, colors['spectral'])
    draw_arrow(12.5, 9, 12.5, 8.5, colors['image'])
    
    # Encoders to single-modal classifiers
    draw_arrow(1.5, 7, 1.5, 5.8, colors['clinical'])
    draw_arrow(4, 7, 4, 5.8, colors['spectral'])
    draw_arrow(11.5, 6.5, 11.5, 4.3, colors['image'])
    
    # Encoders to bi-modal fusion
    draw_arrow(2.5, 7, 3.5, 4, colors['clinical'])  # clinical to clinical+spectral
    draw_arrow(7.5, 7, 6.5, 4, colors['spectral'])  # spectral to clinical+spectral
    
    draw_arrow(2.5, 7, 7.5, 4, colors['clinical'])  # clinical to clinical+image
    draw_arrow(12.5, 6.5, 7.5, 4, colors['image'])  # image to clinical+image
    
    draw_arrow(7.5, 7, 11.5, 4, colors['spectral'])  # spectral to spectral+image
    draw_arrow(12.5, 6.5, 11.5, 4, colors['image'])  # image to spectral+image
    
    # Tri-modal hierarchical fusion arrows
    draw_arrow(7.5, 7, 8.5, 2.5, colors['spectral'])  # spectral to bio fusion
    draw_arrow(12.5, 6.5, 9.5, 2.5, colors['image'])  # image to bio fusion
    draw_arrow(9, 1.5, 9, 1.2, colors['fusion'])      # bio to final fusion
    draw_arrow(2.5, 7, 8, 1.2, colors['clinical'])     # clinical to final fusion
    
    # Attention connections
    draw_arrow(12.5, 6.5, 12.5, 6, colors['attention'])
    draw_arrow(12.5, 5, 12.5, 4.3, colors['attention'])
    
    # Add legend
    legend_x = 0.5
    legend_y = 2.5
    legend_items = [
        ('Clinical Data', colors['clinical']),
        ('Spectral Data', colors['spectral']),
        ('Image Data', colors['image']),
        ('Fusion Layer', colors['fusion']),
        ('Classifier', colors['classifier']),
        ('Attention', colors['attention'])
    ]
    
    ax.text(legend_x, legend_y + 0.5, 'Legend:', fontsize=12, fontweight='bold')
    for i, (label, color) in enumerate(legend_items):
        rect = patches.Rectangle((legend_x, legend_y - i*0.3), 0.3, 0.2, 
                               facecolor=color, edgecolor='black')
        ax.add_patch(rect)
        ax.text(legend_x + 0.4, legend_y - i*0.3 + 0.1, label, 
               va='center', fontsize=10)
    
    # Add ablation configurations
    config_x = 14.5
    config_y = 8
    configs = [
        'Ablation Configs:',
        '1. Clinical',
        '2. Spectral', 
        '3. Image',
        '4. Clinical+Spectral',
        '5. Clinical+Image',
        '6. Spectral+Image',
        '7. Tri-Modal'
    ]
    
    for i, config in enumerate(configs):
        if i == 0:
            ax.text(config_x, config_y - i*0.4, config, fontsize=12, fontweight='bold')
        else:
            ax.text(config_x, config_y - i*0.4, config, fontsize=10)
    
    plt.tight_layout()
    plt.savefig('model_architecture.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.savefig('enhanced_results/model_architecture.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ English model architecture diagram generated: model_architecture.png")
    print("✅ Also saved to: enhanced_results/model_architecture.png")
    
    return fig

def create_ablation_flow_diagram():
    """Create ablation study flow diagram in English"""
    
    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Title
    ax.text(7, 9.5, 'Ablation Study Experimental Flow', ha='center', va='center', 
           fontsize=16, fontweight='bold')
    
    # Define flow steps
    steps = [
        (2, 8, 'Data Loading\n124 Samples', '#FFE5B4'),
        (7, 8, 'Stratified Split\nTrain:79 Val:20 Test:25', '#B4E5FF'),
        (12, 8, 'Feature Preprocessing\nStandardization+Normalization', '#B4FFE5'),
        
        (1, 6, 'Clinical\n68.0%', '#FF6B6B'),
        (3.5, 6, 'Spectral\n84.0%', '#4ECDC4'),
        (6, 6, 'Image\n64.0%', '#45B7D1'),
        
        (1, 4, 'Clinical+\nSpectral\n92.0%', '#96CEB4'),
        (4, 4, 'Clinical+\nImage\n68.0%', '#FFEAA7'),
        (7, 4, 'Spectral+\nImage\n88.0%', '#DDA0DD'),
        
        (10, 6, 'Tri-Modal\n84.0%', '#FFA07A'),
        
        (7, 2, 'Performance\nComparison\nVisualization', '#E6E6FA'),
        (7, 0.5, 'Comprehensive\nReport\nGeneration', '#F0E68C')
    ]
    
    # Draw step boxes
    for x, y, text, color in steps:
        box = FancyBboxPatch((x-0.6, y-0.4), 1.2, 0.8,
                           boxstyle="round,pad=0.1",
                           facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(box)
        ax.text(x, y, text, ha='center', va='center', fontsize=9, fontweight='bold')
    
    # Draw connecting arrows
    arrows = [
        # Data flow
        (2.6, 8, 6.4, 8),    # Data loading -> Stratified split
        (7.6, 8, 11.4, 8),   # Stratified split -> Preprocessing
        
        # Preprocessing to models
        (12, 7.6, 1, 6.4),   # Preprocessing -> Clinical
        (12, 7.6, 3.5, 6.4), # Preprocessing -> Spectral  
        (12, 7.6, 6, 6.4),   # Preprocessing -> Image
        (12, 7.6, 10, 6.4),  # Preprocessing -> Tri-Modal
        
        # Single-modal to bi-modal
        (1, 5.6, 1, 4.4),    # Clinical -> Clinical+Spectral
        (3.5, 5.6, 1, 4.4),  # Spectral -> Clinical+Spectral
        (1, 5.6, 4, 4.4),    # Clinical -> Clinical+Image
        (6, 5.6, 4, 4.4),    # Image -> Clinical+Image
        (3.5, 5.6, 7, 4.4),  # Spectral -> Spectral+Image
        (6, 5.6, 7, 4.4),    # Image -> Spectral+Image
        
        # To result analysis
        (7, 3.6, 7, 2.4),    # Model results -> Comparison
        (7, 1.6, 7, 0.9),    # Comparison -> Report generation
    ]
    
    for start_x, start_y, end_x, end_y in arrows:
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y),
                   arrowprops=dict(arrowstyle='->', color='gray', linewidth=1.5))
    
    plt.tight_layout()
    plt.savefig('ablation_flow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.savefig('enhanced_results/ablation_flow.png', dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    print("✅ English ablation study flow diagram generated: ablation_flow.png")
    print("✅ Also saved to: enhanced_results/ablation_flow.png")
    
    return fig

if __name__ == "__main__":
    print("🎨 Generating English model architecture diagrams...")
    
    # Create results directory
    import os
    os.makedirs('enhanced_results', exist_ok=True)
    
    # Generate architecture diagram
    fig1 = create_model_architecture_diagram()
    
    # Generate flow diagram
    fig2 = create_ablation_flow_diagram()
    
    plt.show()
    
    print("🎉 All English diagrams generated successfully!") 