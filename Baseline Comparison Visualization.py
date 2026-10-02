import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# 设置绘图风格 (符合 IEEE/SAGE 期刊标准)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 12
sns.set_style("ticks")

# 数据准备
# Micro-F1 数据
baselines = ['Baseline 1', 'Baseline 2', 'Baseline 3', 'Proposed']
micro_f1 = [0.9476, 0.9591, 0.9509, 0.9617]

# 核心困难类别数据 (已将拼音替换为标准英文缩写)
# NTRF (原 jinduan): 0.8396, 0.8575, 0.8596, 0.8763
# DCGF (原 jiedier): 0.9384, 0.9556, 0.9318, 0.9598
hard_faults = pd.DataFrame({
    'Category': ['NTRF', 'NTRF', 'NTRF', 'NTRF',
                 'DCGF', 'DCGF', 'DCGF', 'DCGF'],
    'Model': ['B1', 'B2', 'B3', 'Proposed',
              'B1', 'B2', 'B3', 'Proposed'],
    'F1-Score': [0.8396, 0.8575, 0.8596, 0.8763,
                 0.9384, 0.9556, 0.9318, 0.9598]
})

# 创建画布 (1行2列)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 图 A: Micro-F1 演进
axes[0].bar(baselines, micro_f1, color=['#d9d9d9', '#bdbdbd', '#969696', '#525252'], edgecolor='black')
axes[0].set_ylim(0.94, 0.965)
axes[0].set_ylabel('Micro-F1 Score', fontweight='bold')
axes[0].set_title('(a) Overall Performance Trend', fontweight='bold')
axes[0].grid(axis='y', linestyle='--', alpha=0.7)

# 图 B: 困难类别对比
sns.barplot(data=hard_faults, x='Category', y='F1-Score', hue='Model',
            palette='Blues_d', ax=axes[1], edgecolor='black')
axes[1].set_ylim(0.8, 1.0)
axes[1].set_title('(b) Weak Fault Diagnosis Performance', fontweight='bold')
axes[1].set_ylabel('F1-Score', fontweight='bold')
axes[1].set_xlabel('Category', fontweight='bold')
axes[1].legend(title='Model', loc='upper left')

plt.tight_layout()

# 同时保存高清 PNG 和 PDF (供论文排版使用)
plt.savefig('Fig10_Ablation_Comparison.png', dpi=300, bbox_inches='tight')
plt.savefig('Fig10_Ablation_Comparison.pdf', bbox_inches='tight')
print("✅ 图10（消融实验与弱故障识别对比图）已成功生成全英文版。")
plt.show()
