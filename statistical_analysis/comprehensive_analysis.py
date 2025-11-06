#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
银屑病数据集统计分析
Comprehensive Statistical Analysis for Psoriasis Dataset

包括临床数据和SEM图像的统计分析
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

# 设置字体支持
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

class PsoriasisDataAnalyzer:
    def __init__(self, data_path="../../data", output_path="./results"):
        """初始化分析器"""
        self.data_path = data_path
        self.output_path = output_path
        self.clinical_data = None
        self.image_counts = None
        self.stats_results = {}
        
        # 创建输出目录
        os.makedirs(output_path, exist_ok=True)
        os.makedirs(f"{output_path}/plots", exist_ok=True)
        os.makedirs(f"{output_path}/statistics", exist_ok=True)
    
    def load_data(self):
        """加载临床数据"""
        print("正在加载数据...")
        
        # 加载临床数据
        clinical_file = os.path.join(self.data_path, "clinical_data.csv")
        self.clinical_data = pd.read_csv(clinical_file)
        
        # 统计SEM图像数量
        self.count_sem_images()
        
        print(f"数据加载完成:")
        print(f"- 临床数据: {len(self.clinical_data)} 个样本")
        print(f"- 疾病组别: {self.clinical_data['Disease_Group'].value_counts().to_dict()}")
        
    def count_sem_images(self):
        """统计SEM图像数量"""
        sem_path = os.path.join(self.data_path, "SEM")
        image_counts = {"PSA": {"背面": 0, "腹面": 0}, "PSO": {"背面": 0, "腹面": 0}}
        
        for disease in ["PSA", "PSO"]:
            for side in ["背面", "腹面"]:
                folder_path = os.path.join(sem_path, disease, side)
                if os.path.exists(folder_path):
                    tif_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.tif', '.tiff'))]
                    image_counts[disease][side] = len(tif_files)
        
        self.image_counts = image_counts
        
    def basic_statistics(self):
        """基础描述性统计"""
        print("\n=== 基础描述性统计 ===")
        
        # 数值列
        numeric_cols = ['Age', 'BMI', 'PASI', 'BSA', 'Amide_Bond_1_Structure', 
                       'Amide_Bond_1_Content', 'Amide_Bond_2_Structure', 
                       'Amide_Bond_2_Content', 'Disulfide_Bond_Content']
        
        # 基础统计
        basic_stats = self.clinical_data[numeric_cols].describe()
        
        # 按疾病组别分组统计
        grouped_stats = self.clinical_data.groupby('Disease_Group')[numeric_cols].describe()
        
        # 性别分布
        gender_dist = self.clinical_data['Gender'].value_counts()
        gender_by_disease = pd.crosstab(self.clinical_data['Disease_Group'], 
                                       self.clinical_data['Gender'])
        
        # 保存统计结果
        self.stats_results['basic_stats'] = basic_stats
        self.stats_results['grouped_stats'] = grouped_stats
        self.stats_results['gender_distribution'] = gender_dist
        self.stats_results['gender_by_disease'] = gender_by_disease
        
        # 打印关键统计
        print(f"总样本数: {len(self.clinical_data)}")
        print(f"PSA组: {len(self.clinical_data[self.clinical_data['Disease_Group']=='PSA'])}")
        print(f"PSO组: {len(self.clinical_data[self.clinical_data['Disease_Group']=='PSO'])}")
        print(f"性别分布 (1=男, 2=女): {gender_dist.to_dict()}")
        
        return basic_stats, grouped_stats
    
    def correlation_analysis(self):
        """相关性分析"""
        print("\n=== 相关性分析 ===")
        
        numeric_cols = ['Age', 'BMI', 'PASI', 'BSA', 'Amide_Bond_1_Structure', 
                       'Amide_Bond_1_Content', 'Amide_Bond_2_Structure', 
                       'Amide_Bond_2_Content', 'Disulfide_Bond_Content']
        
        # 计算相关系数矩阵
        correlation_matrix = self.clinical_data[numeric_cols].corr()
        
        # 分组相关性分析
        psa_corr = self.clinical_data[self.clinical_data['Disease_Group']=='PSA'][numeric_cols].corr()
        pso_corr = self.clinical_data[self.clinical_data['Disease_Group']=='PSO'][numeric_cols].corr()
        
        self.stats_results['correlation_all'] = correlation_matrix
        self.stats_results['correlation_psa'] = psa_corr
        self.stats_results['correlation_pso'] = pso_corr
        
        return correlation_matrix, psa_corr, pso_corr
    
    def statistical_tests(self):
        """统计假设检验"""
        print("\n=== 统计假设检验 ===")
        
        numeric_cols = ['Age', 'BMI', 'PASI', 'BSA', 'Amide_Bond_1_Structure', 
                       'Amide_Bond_1_Content', 'Amide_Bond_2_Structure', 
                       'Amide_Bond_2_Content', 'Disulfide_Bond_Content']
        
        test_results = {}
        
        psa_data = self.clinical_data[self.clinical_data['Disease_Group']=='PSA']
        pso_data = self.clinical_data[self.clinical_data['Disease_Group']=='PSO']
        
        for col in numeric_cols:
            # t检验 (独立样本)
            t_stat, t_pval = stats.ttest_ind(psa_data[col], pso_data[col])
            
            # Mann-Whitney U检验 (非参数)
            u_stat, u_pval = stats.mannwhitneyu(psa_data[col], pso_data[col])
            
            # Shapiro-Wilk正态性检验
            shapiro_psa = stats.shapiro(psa_data[col])
            shapiro_pso = stats.shapiro(pso_data[col])
            
            test_results[col] = {
                't_statistic': t_stat,
                't_pvalue': t_pval,
                'u_statistic': u_stat,
                'u_pvalue': u_pval,
                'shapiro_psa_pvalue': shapiro_psa.pvalue,
                'shapiro_pso_pvalue': shapiro_pso.pvalue,
                'significant_diff': t_pval < 0.05
            }
        
        # 卡方检验 - 性别与疾病组的关联
        chi2_stat, chi2_pval, dof, expected = stats.chi2_contingency(
            pd.crosstab(self.clinical_data['Disease_Group'], self.clinical_data['Gender'])
        )
        
        test_results['gender_disease_chi2'] = {
            'chi2_statistic': chi2_stat,
            'chi2_pvalue': chi2_pval,
            'degrees_of_freedom': dof
        }
        
        self.stats_results['statistical_tests'] = test_results
        
        return test_results
    
    def machine_learning_analysis(self):
        """机器学习分析和混淆矩阵"""
        print("\n=== 机器学习分析 ===")
        
        # 准备特征和标签
        feature_cols = ['Age', 'BMI', 'PASI', 'BSA', 'Amide_Bond_1_Structure', 
                       'Amide_Bond_1_Content', 'Amide_Bond_2_Structure', 
                       'Amide_Bond_2_Content', 'Disulfide_Bond_Content', 'Gender']
        
        X = self.clinical_data[feature_cols]
        y = self.clinical_data['Disease_Group']
        
        # 数据标准化
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # 划分训练测试集
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=0.3, random_state=42, stratify=y
        )
        
        # 随机森林分类器
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_pred = rf_model.predict(X_test)
        
        # 逻辑回归
        lr_model = LogisticRegression(random_state=42)
        lr_model.fit(X_train, y_train)
        lr_pred = lr_model.predict(X_test)
        
        # 混淆矩阵
        rf_cm = confusion_matrix(y_test, rf_pred)
        lr_cm = confusion_matrix(y_test, lr_pred)
        
        # 分类报告
        rf_report = classification_report(y_test, rf_pred, output_dict=True)
        lr_report = classification_report(y_test, lr_pred, output_dict=True)
        
        # 特征重要性
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        ml_results = {
            'rf_confusion_matrix': rf_cm,
            'lr_confusion_matrix': lr_cm,
            'rf_classification_report': rf_report,
            'lr_classification_report': lr_report,
            'feature_importance': feature_importance,
            'feature_names': feature_cols,
            'rf_accuracy': rf_report['accuracy'],
            'lr_accuracy': lr_report['accuracy']
        }
        
        self.stats_results['ml_results'] = ml_results
        
        return ml_results
    
    def create_visualizations(self):
        """创建可视化图表"""
        print("\n=== 创建可视化图表 ===")
        
        # 定义标签清理函数
        def clean_label(label):
            """清理标签：移除下划线，替换数字为罗马数字"""
            label = str(label)
            label = label.replace('_', ' ')
            label = label.replace('1', 'I')
            label = label.replace('2', 'II')
            return label
        
        # 1. 相关性热力图
        plt.figure(figsize=(12, 10))
        # 创建带有清理标签的相关性矩阵副本
        correlation_cleaned = self.stats_results['correlation_all'].copy()
        correlation_cleaned.index = [clean_label(idx) for idx in correlation_cleaned.index]
        correlation_cleaned.columns = [clean_label(col) for col in correlation_cleaned.columns]
        
        sns.heatmap(correlation_cleaned, 
                   annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Correlation Heatmap of All Parameters', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/correlation_heatmap_all.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. 按疾病组分别的相关性热力图
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # 清理PSA组相关性标签
        correlation_psa_cleaned = self.stats_results['correlation_psa'].copy()
        correlation_psa_cleaned.index = [clean_label(idx) for idx in correlation_psa_cleaned.index]
        correlation_psa_cleaned.columns = [clean_label(col) for col in correlation_psa_cleaned.columns]
        
        sns.heatmap(correlation_psa_cleaned, 
                   annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=axes[0])
        axes[0].set_title('PSA Group Correlation Heatmap', fontsize=14, fontweight='bold')
        
        # 清理PSO组相关性标签
        correlation_pso_cleaned = self.stats_results['correlation_pso'].copy()
        correlation_pso_cleaned.index = [clean_label(idx) for idx in correlation_pso_cleaned.index]
        correlation_pso_cleaned.columns = [clean_label(col) for col in correlation_pso_cleaned.columns]
        
        sns.heatmap(correlation_pso_cleaned, 
                   annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f', ax=axes[1])
        axes[1].set_title('PSO Group Correlation Heatmap', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/correlation_heatmap_by_group.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. 混淆矩阵可视化
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # 随机森林混淆矩阵
        sns.heatmap(self.stats_results['ml_results']['rf_confusion_matrix'], 
                   annot=True, fmt='d', cmap='Blues',
                   xticklabels=['PSA', 'PSO'], yticklabels=['PSA', 'PSO'],
                   ax=axes[0])
        axes[0].set_title(f'Random Forest Confusion Matrix\nAccuracy: {self.stats_results["ml_results"]["rf_accuracy"]:.3f}')
        axes[0].set_ylabel('True Labels')
        axes[0].set_xlabel('Predicted Labels')
        
        # 逻辑回归混淆矩阵
        sns.heatmap(self.stats_results['ml_results']['lr_confusion_matrix'], 
                   annot=True, fmt='d', cmap='Greens',
                   xticklabels=['PSA', 'PSO'], yticklabels=['PSA', 'PSO'],
                   ax=axes[1])
        axes[1].set_title(f'Logistic Regression Confusion Matrix\nAccuracy: {self.stats_results["ml_results"]["lr_accuracy"]:.3f}')
        axes[1].set_ylabel('True Labels')
        axes[1].set_xlabel('Predicted Labels')
        
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/confusion_matrices.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. 特征重要性图
        plt.figure(figsize=(10, 6))
        feature_imp = self.stats_results['ml_results']['feature_importance'].copy()
        # 清理特征名称
        feature_imp['feature'] = feature_imp['feature'].apply(clean_label)
        sns.barplot(data=feature_imp, x='importance', y='feature', palette='viridis')
        plt.title('Random Forest Feature Importance', fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score')
        plt.ylabel('Features')
        plt.tight_layout()
        plt.savefig(f'{self.output_path}/plots/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. 数据分布小提琴图
        numeric_cols = ['Age', 'BMI', 'PASI', 'BSA', 'Amide_Bond_1_Content', 
                       'Amide_Bond_2_Content', 'Disulfide_Bond_Content']
        
        # 美化的标签名称
        col_labels = {
            'Age': 'Age (years)',
            'BMI': 'BMI (kg/m²)',
            'PASI': 'PASI Score',
            'BSA': 'BSA (%)',
            'Amide_Bond_1_Content': 'Amide Bond I Content',
            'Amide_Bond_2_Content': 'Amide Bond II Content',
            'Disulfide_Bond_Content': 'Disulfide Bond Content'
        }
        
        # 应用清理函数到所有标签
        col_labels = {key: clean_label(value) for key, value in col_labels.items()}
        
        # 设置专业的颜色方案
        colors = ['#2E86AB', '#A23B72']  # 专业的蓝色和紫红色
        
        # 设置matplotlib参数以获得更好的外观
        plt.rcParams.update({
            'font.size': 12,
            'font.family': 'serif',
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.size': 6,
            'ytick.major.size': 6,
            'xtick.direction': 'out',
            'ytick.direction': 'out'
        })
        
        fig, axes = plt.subplots(2, 4, figsize=(20, 12))
        axes = axes.flatten()
        
        for i, col in enumerate(numeric_cols):
            # 获取两组数据
            psa_data = self.clinical_data[self.clinical_data['Disease_Group'] == 'PSA'][col].dropna()
            pso_data = self.clinical_data[self.clinical_data['Disease_Group'] == 'PSO'][col].dropna()
            
            # 创建小提琴图
            violin_parts = axes[i].violinplot([psa_data, pso_data],
                                            positions=[1, 2], widths=0.6, showmeans=True, showmedians=True)
            
            # 设置小提琴图颜色
            for pc, color in zip(violin_parts['bodies'], colors):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)  # 稍微降低透明度以便看到数据点
                pc.set_edgecolor('black')
                pc.set_linewidth(1)
            
            # 设置统计线条颜色
            for partname in ('cbars', 'cmins', 'cmaxes', 'cmedians', 'cmeans'):
                if partname in violin_parts:
                    violin_parts[partname].set_edgecolor('black')
                    violin_parts[partname].set_linewidth(1.5)
            
            # 添加箱线图叠加以显示四分位数
            box_parts = axes[i].boxplot([psa_data, pso_data],
                                      positions=[1, 2], widths=0.15, patch_artist=True,
                                      boxprops=dict(facecolor='white', alpha=0.8, linewidth=1.2),
                                      medianprops=dict(color='red', linewidth=2),
                                      whiskerprops=dict(linewidth=1.2),
                                      capprops=dict(linewidth=1.2))
            
            # 添加数据点 - PSA组
            np.random.seed(42)  # 确保重现性
            x_psa = np.random.normal(1, 0.04, size=len(psa_data))  # 在x=1周围添加轻微抖动
            axes[i].scatter(x_psa, psa_data, alpha=0.7, s=25, c=colors[0], 
                          edgecolors='white', linewidth=0.5, zorder=3)
            
            # 添加数据点 - PSO组
            x_pso = np.random.normal(2, 0.04, size=len(pso_data))  # 在x=2周围添加轻微抖动
            axes[i].scatter(x_pso, pso_data, alpha=0.7, s=25, c=colors[1], 
                          edgecolors='white', linewidth=0.5, zorder=3)
            
            # 设置标题和标签
            axes[i].set_title(f'{col_labels[col]}', fontsize=14, fontweight='bold', pad=15)
            axes[i].set_xlabel('Disease Group', fontsize=12, fontweight='bold')
            axes[i].set_ylabel(col_labels[col], fontsize=12, fontweight='bold')
            axes[i].set_xticks([1, 2])
            axes[i].set_xticklabels(['PSA', 'PSO'], fontsize=11)
            
            # 美化网格线
            axes[i].grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            axes[i].set_axisbelow(True)
            
            # 调整y轴标签
            axes[i].tick_params(axis='y', labelsize=10)
            axes[i].tick_params(axis='x', labelsize=11)
        
        # 隐藏最后一个空白子图
        axes[-1].set_visible(False)
        
        # 添加整体标题
        fig.suptitle('Distribution of Clinical and Spectral Parameters by Disease Group', 
                    fontsize=18, fontweight='bold', y=0.98)
        
        # 添加图例
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[0], alpha=0.7, label='PSA'),
                          Patch(facecolor=colors[1], alpha=0.7, label='PSO')]
        fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.92), 
                  fontsize=12, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90)
        plt.savefig(f'{self.output_path}/plots/distribution_violinplots.png', dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close()
        
        # 重置matplotlib参数
        plt.rcdefaults()
        
        print("All plots have been saved to plots/ directory")
    
    def save_statistics(self):
        """保存统计结果到文件"""
        print("\n=== 保存统计结果 ===")
        
        # 1. 基础统计
        self.stats_results['basic_stats'].to_csv(f'{self.output_path}/statistics/basic_statistics.csv')
        
        # 2. 分组统计
        with pd.ExcelWriter(f'{self.output_path}/statistics/grouped_statistics.xlsx') as writer:
            self.stats_results['grouped_stats'].to_excel(writer, sheet_name='按疾病组统计')
            self.stats_results['gender_by_disease'].to_excel(writer, sheet_name='性别疾病交叉表')
        
        # 3. 相关性矩阵
        with pd.ExcelWriter(f'{self.output_path}/statistics/correlation_matrices.xlsx') as writer:
            self.stats_results['correlation_all'].to_excel(writer, sheet_name='全体相关性')
            self.stats_results['correlation_psa'].to_excel(writer, sheet_name='PSA组相关性')
            self.stats_results['correlation_pso'].to_excel(writer, sheet_name='PSO组相关性')
        
        # 4. 统计检验结果
        test_results_df = pd.DataFrame(self.stats_results['statistical_tests']).T
        test_results_df.to_csv(f'{self.output_path}/statistics/statistical_tests.csv')
        
        # 5. 机器学习结果
        self.stats_results['ml_results']['feature_importance'].to_csv(
            f'{self.output_path}/statistics/feature_importance.csv', index=False
        )
        
        # 6. 图像统计
        image_stats_df = pd.DataFrame(self.image_counts).T
        image_stats_df.to_csv(f'{self.output_path}/statistics/sem_image_counts.csv')
        
        # 7. 综合报告
        self.generate_summary_report()
        
        print("所有统计结果已保存到 statistics/ 目录")
    
    def generate_summary_report(self):
        """生成综合分析报告"""
        report = f"""
银屑病数据集统计分析报告
========================

## 数据概览
- 总样本数: {len(self.clinical_data)}
- PSA组样本: {len(self.clinical_data[self.clinical_data['Disease_Group']=='PSA'])}
- PSO组样本: {len(self.clinical_data[self.clinical_data['Disease_Group']=='PSO'])}

## SEM图像统计
- PSA背面图像: {self.image_counts['PSA']['背面']} 张
- PSA腹面图像: {self.image_counts['PSA']['腹面']} 张  
- PSO背面图像: {self.image_counts['PSO']['背面']} 张
- PSO腹面图像: {self.image_counts['PSO']['腹面']} 张

## 关键发现

### 1. 人口统计学特征
{self.clinical_data.groupby('Disease_Group')[['Age', 'Gender', 'BMI']].describe()}

### 2. 疾病严重程度指标
{self.clinical_data.groupby('Disease_Group')[['PASI', 'BSA']].describe()}

### 3. 生化指标
{self.clinical_data.groupby('Disease_Group')[['Amide_Bond_1_Content', 'Amide_Bond_2_Content', 'Disulfide_Bond_Content']].describe()}

### 4. 机器学习分类性能
- 随机森林准确率: {self.stats_results['ml_results']['rf_accuracy']:.3f}
- 逻辑回归准确率: {self.stats_results['ml_results']['lr_accuracy']:.3f}

### 5. 最重要的区分特征
{self.stats_results['ml_results']['feature_importance'].head()}

## 统计显著性检验
显著差异的参数 (p<0.05):
"""
        
        # 添加显著性检验结果
        significant_params = []
        for param, results in self.stats_results['statistical_tests'].items():
            if isinstance(results, dict) and 'significant_diff' in results:
                if results['significant_diff']:
                    significant_params.append(f"- {param}: p = {results['t_pvalue']:.4f}")
        
        if significant_params:
            report += "\n".join(significant_params)
        else:
            report += "无参数显示显著性差异"
        
        report += "\n\n## 文件说明\n"
        report += "- basic_statistics.csv: 基础描述性统计\n"
        report += "- grouped_statistics.xlsx: 按疾病组分组统计\n" 
        report += "- correlation_matrices.xlsx: 相关性矩阵\n"
        report += "- statistical_tests.csv: 统计假设检验结果\n"
        report += "- feature_importance.csv: 特征重要性排序\n"
        report += "- plots/: 所有可视化图表\n"
        
        with open(f'{self.output_path}/analysis_report.txt', 'w', encoding='utf-8') as f:
            f.write(report)
    
    def run_complete_analysis(self):
        """运行完整的统计分析流程"""
        print("开始银屑病数据集统计分析...")
        print("="*50)
        
        # 加载数据
        self.load_data()
        
        # 基础统计
        self.basic_statistics()
        
        # 相关性分析
        self.correlation_analysis()
        
        # 统计检验
        self.statistical_tests()
        
        # 机器学习分析
        self.machine_learning_analysis()
        
        # 创建可视化
        self.create_visualizations()
        
        # 保存结果
        self.save_statistics()
        
        print("\n" + "="*50)
        print("统计分析完成！")
        print(f"结果已保存到: {self.output_path}")
        print("请查看以下文件:")
        print("- analysis_report.txt: 综合分析报告")
        print("- statistics/: 详细统计数据")
        print("- plots/: 可视化图表")


if __name__ == "__main__":
    # 创建分析器实例并运行分析
    analyzer = PsoriasisDataAnalyzer()
    analyzer.run_complete_analysis() 