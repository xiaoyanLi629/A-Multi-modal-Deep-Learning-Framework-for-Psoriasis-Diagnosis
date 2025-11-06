# Hierarchical Multi-Modal Fusion Network: Detailed Architecture Specification

## Table of Contents
1. [Overview](#overview)
2. [Network Architecture](#network-architecture)
3. [Component Specifications](#component-specifications)
4. [Ablation Study Configurations](#ablation-study-configurations)
5. [Training and Optimization](#training-and-optimization)
6. [Implementation Details](#implementation-details)
7. [Performance Specifications](#performance-specifications)

## Overview

The Hierarchical Multi-Modal Fusion Network is a novel deep learning architecture designed for psoriatic condition classification using three heterogeneous data modalities: clinical data, infrared spectroscopy data, and scanning electron microscopy (SEM) images. The network implements a biologically-inspired hierarchical fusion strategy that mirrors the pathophysiological progression from molecular changes to clinical manifestations.

### Key Design Principles
- **Biological Hierarchy**: Two-stage fusion reflecting molecular → morphological → clinical progression
- **Modal Independence**: Each modality processed by specialized encoders
- **Attention Mechanism**: Interpretable attention for image features
- **Ablation Support**: Seven configurations for comprehensive evaluation

## Network Architecture

### High-Level Architecture Diagram

```
Input Layer:
├── Clinical Data [5D] ────────┐
├── Spectral Data [5D] ────────┼─────► Stage 1: Biological Fusion
└── Image Data [224×224×3] ────┘      [Spectral + Image → Bio Features]
                                                    │
                                                    ▼
                                      Stage 2: Clinical Integration
                                      [Bio Features + Clinical → Final]
                                                    │
                                                    ▼
                                            Classification Layer
                                                [PSA/PSO]
```

### Mathematical Formulation

The complete network can be expressed as:
```
f(c, s, I) = σ(W_cls · ReLU(BN(W_final[h_bio; h_c] + b_final)) + b_cls)
```

Where:
- `h_bio = ReLU(BN(W_bio[h_s; h_i] + b_bio))` (biological features)
- `h_c = Encoder_clinical(c)` (clinical features)
- `h_s = Encoder_spectral(s)` (spectral features)
- `h_i = Encoder_image(I)` (image features with attention)

## Component Specifications

### 1. Clinical Data Encoder

**Purpose**: Process structured demographic and clinical severity features

**Architecture**:
```python
Clinical Encoder:
Input: [batch_size, 5] → (Gender, Age, BMI, PASI, BSA)
├── Linear(5 → 32) + BatchNorm1d + ReLU + Dropout(0.3)
├── Linear(32 → 16) + ReLU
Output: [batch_size, 16]
```

**Parameters**:
- Total Parameters: 0.58K
- Input Features: 5 (Gender, Age, BMI, PASI, BSA)
- Output Dimension: 16
- Regularization: Dropout (0.3), BatchNorm

**Design Rationale**: 
Shallow architecture prevents overfitting on limited clinical variables while capturing non-linear relationships between demographic and severity indicators.

### 2. Spectral Data Encoder

**Purpose**: Extract molecular-level features from infrared spectroscopy data

**Architecture**:
```python
Spectral Encoder:
Input: [batch_size, 5] → (Amide I/II ratios, Disulfide content)
├── Linear(5 → 64) + BatchNorm1d + ReLU + Dropout(0.3)
├── Linear(64 → 32) + ReLU + Dropout(0.2)
├── Linear(32 → 16) + ReLU
Output: [batch_size, 16]
```

**Parameters**:
- Total Parameters: 2.13K
- Input Features: 5 molecular biomarkers
- Output Dimension: 16
- Regularization: Progressive dropout (0.3 → 0.2), BatchNorm

**Design Rationale**: 
Deeper architecture captures complex molecular signatures while progressive dropout prevents overfitting on spectral measurements.

### 3. Image Encoder with Attention

**Purpose**: Extract morphological features from SEM images with interpretable attention

**Architecture**:
```python
Image Encoder:
Input: [batch_size, 3, 224, 224] → SEM Image Pairs
├── EfficientNet-B0 (Pretrained) → [batch_size, 1280]
├── GlobalAveragePooling
├── Multi-Head Attention Branch:
│   ├── Linear(1280 → 128) + ReLU
│   └── Linear(128 → 64) + Sigmoid → Attention Weights
├── Feature Projection Branch:
│   ├── Linear(1280 → 128) + ReLU + Dropout(0.3)
│   └── Linear(128 → 64) + ReLU
├── Attention Application: Features × Attention_Weights
Output: [batch_size, 64]
```

**Parameters**:
- Total Parameters: 4.06M (including EfficientNet-B0)
- Input Resolution: 224×224×3
- Output Dimension: 64
- Attention Mechanism: 64-dimensional attention weights

**Design Rationale**:
- EfficientNet-B0: Optimal efficiency-performance balance for medical imaging
- Transfer Learning: Pretrained weights provide crucial inductive biases
- Attention Mechanism: Enables interpretability and focus on diagnostically relevant regions

### 4. Hierarchical Fusion Strategy

#### Stage 1: Biological Feature Fusion

**Purpose**: Combine molecular (spectral) and morphological (image) information

**Architecture**:
```python
Biological Fusion:
Input: Concat([spectral_features, image_features]) → [batch_size, 80]
├── Linear(80 → 48) + BatchNorm1d + ReLU + Dropout(0.3)
├── Linear(48 → 32) + ReLU
Output: [batch_size, 32] → Biological Feature Representation
```

**Mathematical Formulation**:
```
h_bio = ReLU(BN(W_bio2 · ReLU(BN(W_bio1 · [h_s; h_i] + b_bio1)) + b_bio2))
```

#### Stage 2: Clinical Integration Fusion

**Purpose**: Integrate biological features with clinical context

**Architecture**:
```python
Final Fusion:
Input: Concat([bio_features, clinical_features]) → [batch_size, 48]
├── Linear(48 → 24) + BatchNorm1d + ReLU + Dropout(0.3)
├── Linear(24 → 8) + ReLU
├── Linear(8 → 1) → Classification Output
Output: [batch_size, 1] → Logit Score
```

**Mathematical Formulation**:
```
h_final = ReLU(BN(W_final · [h_bio; h_c] + b_final))
y_pred = σ(W_cls · h_final + b_cls)
```

### 5. Classification Layer

**Architecture**:
```python
Classifier:
Input: [batch_size, final_dim] → Context-dependent
├── Linear(final_dim → 1) + Sigmoid
Output: [batch_size, 1] → Probability(PSO)
```

**Output Interpretation**:
- Output ≈ 0: PSA (Psoriatic Arthritis)
- Output ≈ 1: PSO (Psoriasis)

## Ablation Study Configurations

The network supports seven distinct configurations for comprehensive modal contribution analysis:

### Configuration Details

| Config ID | Name | Input Modalities | Final Layer Input | Parameters |
|-----------|------|------------------|-------------------|------------|
| C1 | Clinical-only | Clinical | 16D | 0.58K |
| C2 | Spectral-only | Spectral | 16D | 2.13K |
| C3 | Image-only | Image | 64D | 4.06M |
| C4 | Clinical+Spectral | Clinical, Spectral | 32D → 24D | 2.71K |
| C5 | Clinical+Image | Clinical, Image | 80D → 48D | 4.06M |
| C6 | Spectral+Image | Spectral, Image | 80D → 48D | 4.06M |
| C7 | Tri-Modal | All three | 48D → 24D | 4.06M |

### Configuration-Specific Architectures

**Uni-Modal Configurations (C1-C3)**:
```python
# Clinical-only
classifier = Linear(16 → 8) + ReLU + Dropout(0.3) + Linear(8 → 1)

# Spectral-only  
classifier = Linear(16 → 8) + ReLU + Dropout(0.3) + Linear(8 → 1)

# Image-only
classifier = Linear(64 → 32) + ReLU + Dropout(0.3) + Linear(32 → 8) + ReLU + Linear(8 → 1)
```

**Bi-Modal Configurations (C4-C6)**:
```python
# Clinical+Spectral
fusion = Linear(32 → 24) + BatchNorm + ReLU + Dropout(0.3)
classifier = Linear(24 → 8) + ReLU + Linear(8 → 1)

# Clinical+Image / Spectral+Image
fusion = Linear(80 → 48) + BatchNorm + ReLU + Dropout(0.3)
classifier = Linear(48 → 16) + ReLU + Linear(16 → 1)
```

**Tri-Modal Configuration (C7)**:
```python
# Hierarchical fusion as described above
bio_fusion = Linear(80 → 48) + BatchNorm + ReLU + Dropout(0.3) + Linear(48 → 32) + ReLU
final_fusion = Linear(48 → 24) + BatchNorm + ReLU + Dropout(0.3) + Linear(24 → 8) + ReLU + Linear(8 → 1)
```

## Training and Optimization

### Loss Function Design

**Primary Classification Loss**:
```python
L_cls = -1/N Σ[y_i * log(σ(f(x_i))) + (1-y_i) * log(1-σ(f(x_i)))]
```

**Regularization Components**:
```python
L_reg = λ * Σ||θ||²₂  # L2 regularization, λ = 1e-4
L_total = L_cls + L_reg
```

### Optimization Configuration

**Optimizer**: Adam
```python
learning_rate = 0.001
beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-8
weight_decay = 1e-4
```

**Learning Rate Scheduling**: ReduceLROnPlateau
```python
patience = 10 epochs
factor = 0.1
min_lr = 1e-7
```

**Early Stopping**:
```python
patience = 50 epochs
monitor = 'validation_accuracy'
mode = 'max'
```

### Training Configuration

**Batch Processing**:
- Batch Size: 8
- Maximum Epochs: 500
- Gradient Clipping: None

**Data Augmentation** (Image Only):
```python
transforms = [
    RandomHorizontalFlip(p=0.5),
    RandomRotation(degrees=15),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
]
```

## Implementation Details

### Weight Initialization

**Linear Layers**:
```python
nn.init.xavier_uniform_(weight)
nn.init.constant_(bias, 0)
```

**Pretrained Components**:
- EfficientNet-B0: ImageNet pretrained weights
- Feature extraction layers: Fine-tuned during training

### Memory and Computational Requirements

**Training Requirements**:
- GPU Memory: ~8GB (batch_size=8)
- Training Time: 15-52 minutes per configuration
- Peak Memory Usage: 256-267 MB

**Inference Requirements**:
- Inference Time: <100ms per sample
- Memory Footprint: ~2GB GPU memory
- Model File Size: ~16MB

### Input/Output Specifications

**Input Format**:
```python
clinical_input = [gender, age, bmi, pasi, bsa]  # shape: [batch, 5]
spectral_input = [amide1_struct, amide1_content, amide2_struct, 
                 amide2_content, disulfide]     # shape: [batch, 5]
image_input = tensor                            # shape: [batch, 3, 224, 224]
```

**Output Format**:
```python
logits = model(clinical, spectral, image)      # shape: [batch, 1]
probabilities = torch.sigmoid(logits)          # range: [0, 1]
predictions = (probabilities > 0.5).int()     # 0: PSA, 1: PSO
```

## Performance Specifications

### Ablation Study Results

| Configuration | Accuracy | Precision | Recall | F1-Score | AUC | Training Time |
|---------------|----------|-----------|--------|----------|-----|---------------|
| Clinical+Spectral | **92.0%** | **93.8%** | **93.8%** | **93.8%** | **0.993** | 5.8 min |
| Spectral+Image | **92.0%** | **93.8%** | **93.8%** | **93.8%** | **0.979** | 48.2 min |
| Tri-Modal | 88.0% | 84.2% | 100.0% | 91.4% | 0.965 | 52.1 min |
| Spectral-only | 84.0% | 87.5% | 87.5% | 87.5% | 0.958 | 4.1 min |
| Clinical+Image | 76.0% | 72.7% | 100.0% | 84.2% | 0.819 | 45.7 min |
| Image-only | 72.0% | 71.4% | 93.8% | 81.1% | 0.646 | 45.7 min |
| Clinical-only | 52.0% | 64.3% | 56.2% | 60.0% | 0.514 | 2.3 min |

### Computational Efficiency Analysis

**Parameter Efficiency**:
- Most efficient: Clinical+Spectral (2.71K parameters, 92.0% accuracy)
- Most complex: Image-containing configurations (4.06M parameters)
- Best trade-off: Clinical+Spectral configuration

**Training Convergence**:
- Fastest convergence: Spectral-based configurations
- Most stable: Clinical+Spectral (170 epochs)
- Most volatile: Image-only configurations

### Attention Analysis

**Attention Pattern Statistics**:
- PSA: More dispersed attention (entropy: 2.31 ± 0.15)
- PSO: More focused attention (entropy: 1.87 ± 0.12)
- Expert correlation: IoU = 0.73 ± 0.09, r = 0.82 (p < 0.001)

## Software Dependencies

### Core Requirements
```python
torch >= 1.12.0
torchvision >= 0.13.0
numpy >= 1.21.0
pandas >= 1.3.0
scikit-learn >= 1.0.0
matplotlib >= 3.5.0
seaborn >= 0.11.0
PIL >= 8.3.0
opencv-python >= 4.5.0
```

### Hardware Requirements
```
Minimum: NVIDIA GTX 1080 Ti (11GB)
Recommended: NVIDIA Tesla V100 (32GB)
CPU: Intel Xeon or AMD EPYC (16+ cores)
RAM: 32GB+ system memory
Storage: 100GB+ SSD space
```

## Reproducibility Guidelines

### Random Seed Management
```python
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Model Checkpointing
```python
checkpoint = {
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'scheduler_state_dict': scheduler.state_dict(),
    'epoch': epoch,
    'best_val_acc': best_val_acc,
    'config': model_config
}
```

### Evaluation Protocol
1. Stratified train/validation/test split (79/20/25)
2. 5-fold cross-validation for robustness assessment
3. Statistical significance testing (McNemar's test)
4. Attention visualization and expert validation

---

**Document Version**: 2.0  
**Last Updated**: December 2024  
**Compatibility**: PyTorch 1.12+, Python 3.8+  
**License**: MIT License  
**Contact**: Research Team 