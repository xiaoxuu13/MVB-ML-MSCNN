# baseline_results.py
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score
import os

from sklearn.preprocessing import MultiLabelBinarizer

from train3 import get_classes

# 汇总4组基线的Val F1和Test F1
baseline_metrics = []
classes = get_classes()  # 复用train.py中的get_classes函数
mlb = MultiLabelBinarizer(classes=classes)
mlb.fit([classes])

for baseline in [1, 2, 3, 4]:
    # 读取训练日志（Val F1）
    log_path = f"training_log_baseline{baseline}.csv"
    log_df = pd.read_csv(log_path)
    best_val_f1 = log_df['val_f1'].max()

    # 读取测试集预测结果（需提前保存test_preds/test_targets，或重新计算）
    # 简易版：直接读取classification_report中的micro-F1（手动提取也可）
    report_path = f"compound_classification_report_baseline{baseline}.txt"
    with open(report_path, 'r') as f:
        report = f.read()
    # 提取micro-F1
    test_f1_line = [line for line in report.split('\n') if 'micro avg' in line][0]
    test_f1 = float(test_f1_line.split()[-2])

    baseline_metrics.append({
        'Baseline': f'Baseline{baseline}',
        'Multi-Scale': 'Yes' if baseline in [3, 4] else 'No',
        'CBAM': 'Yes' if baseline in [2, 4] else 'No',
        'Best Val F1': round(best_val_f1, 4),
        'Test Micro F1': round(test_f1, 4)
    })

# 保存对比表格
df = pd.DataFrame(baseline_metrics)
df.to_csv("baseline_comparison.csv", index=False)
print("基线对比结果已保存至 baseline_comparison.csv")
print(df)