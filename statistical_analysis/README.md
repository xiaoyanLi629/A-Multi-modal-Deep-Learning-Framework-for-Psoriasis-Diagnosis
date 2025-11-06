# 银屑病数据集统计分析项目

本项目对银屑病临床数据进行全面的统计分析，包括描述性统计、相关性分析、假设检验、机器学习分类和可视化。

## 数据概述

- **临床数据**: 125个样本，包含46个PSA（银屑病关节炎）和79个PSO（银屑病）病例
- **SEM图像**: 249张扫描电镜图像，分为背面和腹面两个视角
- **分析变量**: 
  - 人口统计学特征：年龄、性别、BMI
  - 疾病严重程度：PASI评分、BSA（体表面积）
  - 生化指标：酰胺键结构与含量、二硫键含量

## 项目结构

```
code/statistical_analysis/
├── comprehensive_analysis.py    # 主分析脚本
├── requirements.txt            # Python依赖包
├── README.md                  # 项目说明文档
└── results/                   # 分析结果目录
    ├── analysis_report.txt    # 综合分析报告
    ├── plots/                 # 可视化图表
    │   ├── correlation_heatmap_all.png        # Overall correlation heatmap
    │   ├── correlation_heatmap_by_group.png   # Group-wise correlation heatmap
    │   ├── confusion_matrices.png             # Confusion matrices
    │   ├── feature_importance.png             # Feature importance plot
    │   └── distribution_boxplots.png          # Data distribution boxplots
    └── statistics/            # 详细统计数据
        ├── basic_statistics.csv               # 基础描述性统计
        ├── grouped_statistics.xlsx            # 按疾病组分组统计
        ├── correlation_matrices.xlsx          # 相关性矩阵
        ├── statistical_tests.csv              # 统计假设检验结果
        ├── feature_importance.csv             # 特征重要性排序
        └── sem_image_counts.csv               # SEM图像统计
```

## 运行环境

### 环境要求
- Python 3.7+
- 建议使用虚拟环境

### 安装依赖
```bash
pip install -r requirements.txt
```

### 运行分析
```bash
python comprehensive_analysis.py
```

## 关键发现

### 1. 机器学习分类性能
- **随机森林准确率**: 97.4%
- **逻辑回归准确率**: 92.1%

### 2. 最重要的区分特征（按重要性排序）
1. **二硫键含量** (30.9%) - 最重要的区分因子
2. **酰胺键2含量** (29.0%) - 第二重要的区分因子  
3. **酰胺键1含量** (20.2%) - 第三重要的区分因子
4. **酰胺键1结构** (4.4%)
5. **PASI评分** (3.6%)

### 3. 统计显著性差异（p<0.05）
以下参数在PSA和PSO组间存在显著差异：
- **酰胺键1结构** (p = 0.0095)
- **酰胺键1含量** (p < 0.0001)
- **酰胺键2结构** (p = 0.0012)
- **酰胺键2含量** (p < 0.0001) 
- **二硫键含量** (p < 0.0001)

### 4. 生化指标差异
- **PSO组**的酰胺键和二硫键含量明显高于PSA组
- **PSA组平均值**：
  - 酰胺键1含量：6.07
  - 酰胺键2含量：3.79
  - 二硫键含量：0.09
- **PSO组平均值**：
  - 酰胺键1含量：12.54
  - 酰胺键2含量：8.21
  - 二硫键含量：0.29

## 分析功能

### 1. 描述性统计
- 基础统计量（均值、标准差、分位数等）
- 按疾病组分组统计
- 性别和年龄分布分析

### 2. 相关性分析
- 总体相关性矩阵
- 按疾病组别的相关性分析
- 相关性热力图可视化

### 3. 统计假设检验
- t检验（独立样本）
- Mann-Whitney U检验（非参数）
- Shapiro-Wilk正态性检验
- 卡方检验（分类变量关联性）

### 4. 机器学习分析
- 随机森林分类器
- 逻辑回归分类器
- 混淆矩阵分析
- 特征重要性评估

### 5. 数据可视化
- 相关性热力图
- 混淆矩阵热力图
- 特征重要性条形图
- 数据分布箱线图

## 结论

本分析表明：

1. **生化指标是区分PSA和PSO的关键因素**，特别是二硫键含量和酰胺键含量
2. **PSO患者的蛋白质结构特征更为显著**，表现为更高的酰胺键和二硫键含量
3. **机器学习模型具有很高的分类准确性**，说明这些生化指标具有很好的诊断价值
4. **传统临床指标**（如PASI、BSA）的区分能力相对较弱

这些发现为银屑病亚型的诊断和治疗提供了重要的生物标志物参考。

## 输出文件说明

- `analysis_report.txt`: 完整的分析报告，包含所有关键发现
- `statistics/`: 详细的统计数据表格，支持进一步分析
- `plots/`: 高质量的可视化图表，适合用于学术报告和论文

## 许可证

本项目仅用于学术研究目的。 