import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import re
import os
from sklearn.metrics import precision_recall_curve, average_precision_score
from itertools import cycle
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import os
import numpy as np
from tqdm import tqdm  # 进度条库，建议 pip install tqdm

from config import CONFIG
from dataset import MVBDiskDataset
from model import ML_MSCNN
from torch.amp import autocast, GradScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
# 全局设置顶刊学术排版格式
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.unicode_minus'] = False


def plot_figure_3():
    """图 3：真实的损失与 F1 曲线 (双 Y 轴设计)"""
    if not os.path.exists('training_log.csv'):
        print("未找到 training_log.csv，请先运行修改后的 train.py")
        return

    data = pd.read_csv('training_log.csv')

    fig, ax1 = plt.subplots(figsize=(7, 5), dpi=300)

    # 左轴：Training Loss
    color_loss = '#D32F2F'  # 深红色
    lns1 = ax1.plot(data['epoch'], data['train_loss'], color=color_loss, marker='o',
                    markersize=5, label='Training Loss', linewidth=2)
    ax1.set_xlabel('Epochs', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Loss Value', fontsize=12, color=color_loss, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_loss)

    # 右轴：Validation F1-score
    ax2 = ax1.twinx()
    color_f1 = '#1976D2'  # 深蓝色
    lns2 = ax2.plot(data['epoch'], data['val_f1'], color=color_f1, marker='s',
                    markersize=5, label='Validation Micro-F1', linewidth=2)
    ax2.set_ylabel('F1-score', fontsize=12, color=color_f1, fontweight='bold')
    ax2.tick_params(axis='y', labelcolor=color_f1)
    ax2.set_ylim(0, 1.05)

    # 合并图例并放在图表内部
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='center right', frameon=True, fontsize=11, edgecolor='black')

    plt.grid(axis='both', linestyle='--', alpha=0.6)
    plt.title('Training Convergence Analysis', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('Fig3_Training_Curves.png', bbox_inches='tight')
    plt.savefig('Fig3_Training_Curves.pdf', bbox_inches='tight')  # 顶刊要求的PDF矢量图
    print("✅ 图3 (训练曲线) 已生成。")


def plot_figure_4():
    """图 4：基于分类报告解析的性能热力图"""
    if not os.path.exists('classification_report.txt'):
        print("未找到 classification_report.txt")
        return

    with open('compound_classification_report.txt', 'r') as f:
        lines = f.readlines()

    classes = []
    metrics = []

    # 正则提取真实的 Precision, Recall, F1-score
    for line in lines:
        parts = re.split(r'\s{2,}', line.strip())
        # 排除非类别行，提取具体的 8 个故障类
        if len(parts) >= 4 and parts[0] not in ['precision', 'micro avg', 'macro avg', 'weighted avg', 'samples avg',
                                                '']:
            classes.append(parts[0])
            metrics.append([float(parts[1]), float(parts[2]), float(parts[3])])

    if not metrics:
        print("解析分类报告失败，请检查 classification_report.txt 的格式。")
        return

    df = pd.DataFrame(metrics, index=classes, columns=['Precision', 'Recall', 'F1-score'])

    plt.figure(figsize=(8, 6), dpi=300)
    # 使用 YlGnBu 渐变色系，顶刊最爱，黑白打印也清晰
    ax = sns.heatmap(df, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=1, linecolor='white',
                     cbar_kws={'label': 'Performance Metric Score'},
                     annot_kws={"size": 12, "family": "Times New Roman", "weight": "bold"})

    ax.figure.axes[-1].yaxis.label.set_size(12)
    ax.figure.axes[-1].yaxis.label.set_family('Times New Roman')

    plt.title('Classification Performance Matrix', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(rotation=0, fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Fig4_Classification_Heatmap.png', bbox_inches='tight')
    plt.savefig('Fig4_Classification_Heatmap.pdf', bbox_inches='tight')
    print("✅ 图4 (性能热力图) 已生成。")


def plot_figure_5():
    """图 5：真实的消融实验柱状图"""
    if not os.path.exists('ablation_results.csv'):
        print("未找到 ablation_results.csv，请先运行 run_ablation.py")
        return

    data = pd.read_csv('ablation_results.csv')
    models = ['Baseline\n(Only 3x3 Conv)', 'MS-CNN\n(w/o CBAM)', 'ML-MSCNN\n(Proposed)']
    f1_scores = data['Micro_F1'].values

    x = np.arange(len(models))
    width = 0.5

    fig, ax = plt.subplots(figsize=(7, 5), dpi=300)
    rects1 = ax.bar(x, f1_scores, width, color='#4575B4', edgecolor='black', linewidth=1.2)

    ax.set_ylabel('Micro F1-score', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=12, fontweight='bold')
    ax.set_ylim(min(f1_scores) - 0.05, 1.02)  # 动态截断 Y 轴

    for rect in rects1:
        height = rect.get_height()
        ax.annotate(f'{height:.4f}', xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 5), textcoords="offset points", ha='center', va='bottom',
                    fontsize=12, fontweight='bold', family='Times New Roman')

    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.title('Ablation Study of Model Components', fontsize=14, fontweight='bold', pad=15)
    plt.tight_layout()
    plt.savefig('Fig5_Ablation_Study.pdf', bbox_inches='tight')
    print("✅ 图5 (真实消融实验图) 已生成。")


# def plot_figure_6():
#     """图 6：多标签分类的 Precision-Recall 曲线"""
#     if not os.path.exists('pr_curve_data.npz'):
#         print("未找到 pr_curve_data.npz，请先运行修改后的 train1.py")
#         return
#
#     # 加载真实概率数据
#     data = np.load('pr_curve_data.npz')
#     y_test = data['targets']
#     y_score = data['probs']
#     classes = data['classes']
#     n_classes = len(classes)
#
#     # 计算每一类的 PR 曲线和 AP 值
#     precision = dict()
#     recall = dict()
#     average_precision = dict()
#
#     for i in range(n_classes):
#         precision[i], recall[i], _ = precision_recall_curve(y_test[:, i], y_score[:, i])
#         average_precision[i] = average_precision_score(y_test[:, i], y_score[:, i])
#
#     # 计算微平均 (Micro-average) PR 曲线
#     precision["micro"], recall["micro"], _ = precision_recall_curve(y_test.ravel(), y_score.ravel())
#     average_precision["micro"] = average_precision_score(y_test, y_score, average="micro")
#
#     # 开始绘图
#     fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
#
#     # 绘制 Micro-average 曲线 (加粗虚线，突出整体性能)
#     plt.plot(recall["micro"], precision["micro"],
#              label=f'Micro-average PR curve (AP = {average_precision["micro"]:0.3f})',
#              color='red', linestyle=':', linewidth=3)
#
#     # 为不同类别分配顶刊常用的高级色系
#     colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f'])
#
#     for i, color in zip(range(n_classes), colors):
#         plt.plot(recall[i], precision[i], color=color, linewidth=1.5,
#                  label=f'{classes[i]} (AP = {average_precision[i]:0.3f})')
#
#     # 图表格式设置
#     plt.xlim([0.0, 1.05])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall', fontsize=12, fontweight='bold')
#     plt.ylabel('Precision', fontsize=12, fontweight='bold')
#     plt.title('Precision-Recall Curves for MVB Fault Diagnosis', fontsize=14, fontweight='bold', pad=15)
#
#     # 将图例放在图外或合适位置，避免遮挡曲线
#     plt.legend(loc="lower left", fontsize=10, edgecolor='black', framealpha=0.9)
#     plt.grid(axis='both', linestyle='--', alpha=0.6)
#
#     plt.tight_layout()
#     plt.savefig('Fig6_PR_Curve.png', bbox_inches='tight')
#     plt.savefig('Fig6_PR_Curve.pdf', bbox_inches='tight')
#     print("✅ 图6 (PR曲线图) 已生成。")
# def plot_figure_6_compound_optimized():
#     """
#     优化后的图 6：不仅展示基础标签，更通过重构概率展示模型对“复合故障”的解耦识别能力。
#     """
#     if not os.path.exists('pr_curve_data.npz'):
#         print("❌ 未找到 pr_curve_data.npz，请先运行 train1.py")
#         return
#
#     # 1. 加载数据
#     data = np.load('pr_curve_data.npz', allow_pickle=True)
#     y_probs_base = data['probs']  # (N, 8)
#     y_true_base = data['targets']  # (N, 8)
#     base_classes = list(data['classes'])
#     num_samples = y_true_base.shape[0]
#
#     # 2. 识别测试集中真实存在的“具体复合状态”
#     # 将 (N, 8) 转换为 ['duanyi+jinduan', 'zhengchang', ...] 字符串列表
#     true_state_names = []
#     for row in y_true_base:
#         active_indices = np.where(row == 1)[0]
#         if len(active_indices) == 0:
#             true_state_names.append('zhengchang')
#         else:
#             names = sorted([base_classes[idx] for idx in active_indices])
#             true_state_names.append('+'.join(names))
#
#     # 获取所有独特的状态（包括单项和复合）
#     distinct_states = sorted(list(set(true_state_names)))
#     print(f">>> 检测到 {len(distinct_states)} 种具体的故障状态（含复合）。")
#
#     # 3. 构建具体状态的“替代概率矩阵”
#     y_true_specific = np.zeros((num_samples, len(distinct_states)))
#     y_probs_specific = np.zeros((num_samples, len(distinct_states)))
#
#     for s_idx, state_name in enumerate(distinct_states):
#         # 填充真实标签 (One-hot 形式用于各状态曲线计算)
#         for i in range(num_samples):
#             if true_state_names[i] == state_name:
#                 y_true_specific[i, s_idx] = 1
#
#         # --- 核心：复合概率重构逻辑 ---
#         if state_name == 'zhengchang':
#             # 【修复】由于模型本身有 zhengchang 神经元，直接读取原生概率即可！
#             idx = base_classes.index('zhengchang')
#             y_probs_specific[:, s_idx] = y_probs_base[:, idx]
#         elif '+' in state_name:
#             # 复合故障概率 = 该组合内所有基础神经元概率的最小值 (逻辑与)
#             constituent_names = state_name.split('+')
#             indices = [base_classes.index(n) for n in constituent_names]
#             y_probs_specific[:, s_idx] = np.min(y_probs_base[:, indices], axis=1)
#         else:
#             # 单项故障状态直接对应其神经元概率
#             idx = base_classes.index(state_name)
#             y_probs_specific[:, s_idx] = y_probs_base[:, idx]
#
#     # 4. 计算指标
#     precision, recall, ap = dict(), dict(), dict()
#     for i in range(len(distinct_states)):
#         precision[i], recall[i], _ = precision_recall_curve(y_true_specific[:, i], y_probs_specific[:, i])
#         ap[i] = average_precision_score(y_true_specific[:, i], y_probs_specific[:, i])
#
#     # 【修复】计算全局 Micro-average 时，必须使用最原始的基础标签输出矩阵 (N, 8)
#     precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_base.ravel(), y_probs_base.ravel())
#     ap["micro"] = average_precision_score(y_true_base, y_probs_base, average="micro")
#     # 5. 绘图
#     plt.rcParams['font.family'] = 'serif'
#     fig, ax = plt.subplots(figsize=(9, 7), dpi=300)
#
#     # A. 绘制 Micro-average (加粗黑虚线作为基准)
#     plt.plot(recall["micro"], precision["micro"], color='black', linestyle='--', lw=3,
#              label=f'Micro-average (AP = {ap["micro"]:.3f})')
#
#     # B. 挑选要展示的类别 (避免线太多太乱)
#     # 策略：展示所有的复合故障(含'+')，以及整体表现最好的几个单项
#     colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
#
#     # 你在报告中提到的“胜出”类（F1接近1.0的复合故障）
#     target_compounds = [s for s in distinct_states if '+' in s]
#
#     for i, state_name in enumerate(distinct_states):
#         if state_name in target_compounds:
#             # 复合故障用彩色实线，加粗
#             plt.plot(recall[i], precision[i], color=next(colors), lw=2,
#                      label=f'Compound: {state_name} (AP = {ap[i]:.3f})')
#         elif state_name == 'zhengchang' or ap[i] > 0.98:  # 优秀的单项
#             # 优秀单项用彩色细线
#             plt.plot(recall[i], precision[i], color=next(colors), lw=1, alpha=0.6,
#                      label=f'Single: {state_name} (AP = {ap[i]:.3f})')
#
#     # 6. 视觉细节优化
#     plt.xlim([0.0, 1.0])
#     plt.ylim([0.0, 1.05])
#     plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
#     plt.ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
#     plt.title('PR Curves for Multi-label Decoupling Performance', fontsize=14, fontweight='bold', pad=20)
#
#     # 调整图例位置，放在右侧或下方
#     plt.legend(loc="lower left", fontsize=9, frameon=True, edgecolor='black')
#     plt.grid(True, linestyle=':', alpha=0.5)
#
#     plt.tight_layout()
#     plt.savefig('Fig6_Compound_PR_Curve.png', bbox_inches='tight')
#     plt.savefig('Fig6_Compound_PR_Curve.pdf', bbox_inches='tight')
#     print("✅ 优化后的图6（含复合故障分析）已生成。")
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from sklearn.metrics import precision_recall_curve, average_precision_score


def plot_figure_9_compound_optimized():
    """
    优化后的PR曲线绘制代码：已将拼音标签映射为标准英文缩写
    """
    if not os.path.exists('pr_curve_data_baseline4.npz') and not os.path.exists('pr_curve_data.npz'):
        print("❌ 未找到 pr_curve_data.npz，请先运行 train_final.py")
        return

    # 尝试加载最新的 npz 文件
    file_name = 'pr_curve_data_baseline4.npz' if os.path.exists('pr_curve_data_baseline4.npz') else 'pr_curve_data.npz'
    data = np.load(file_name, allow_pickle=True)
    y_probs_base = data['probs']  # (N, 8)
    y_true_base = data['targets']  # (N, 8)
    base_classes = list(data['classes'])
    num_samples = y_true_base.shape[0]

    # ================= 新增：拼音到英文缩写的映射 =================
    ENG_MAPPING = {
        'zhengchang': 'Normal',
        'duanyi': 'SCOC',
        'duaner': 'DCOC',
        'jiediyi': 'SCGF',
        'jiedier': 'DCGF',
        'jinduan': 'NTRF',
        'yuanduan': 'FTRF',
        'shuangduan': 'DTRF'
    }

    def get_eng_name(state_str):
        if state_str == 'zhengchang':
            return 'Normal'
        parts = state_str.split('+')
        return '+'.join([ENG_MAPPING.get(p, p) for p in parts])

    # ==========================================================

    # 2. 识别测试集中真实存在的“具体复合状态”
    true_state_names = []
    for row in y_true_base:
        active_indices = np.where(row == 1)[0]
        if len(active_indices) == 0:
            true_state_names.append('zhengchang')
        else:
            names = sorted([base_classes[idx] for idx in active_indices])
            true_state_names.append('+'.join(names))

    # 获取所有独特的状态（包括单项和复合）
    distinct_states = sorted(list(set(true_state_names)))
    print(f">>> 检测到 {len(distinct_states)} 种具体的故障状态（含复合）。")

    # 3. 构建具体状态的“替代概率矩阵”
    y_true_specific = np.zeros((num_samples, len(distinct_states)))
    y_probs_specific = np.zeros((num_samples, len(distinct_states)))

    for s_idx, state_name in enumerate(distinct_states):
        # 填充真实标签 (One-hot 形式用于各状态曲线计算)
        for i in range(num_samples):
            if true_state_names[i] == state_name:
                y_true_specific[i, s_idx] = 1

        # --- 核心：复合概率重构逻辑 ---
        if state_name == 'zhengchang':
            idx = base_classes.index('zhengchang')
            y_probs_specific[:, s_idx] = y_probs_base[:, idx]
        elif '+' in state_name:
            # 复合故障概率 = 该组合内所有基础神经元概率的最小值 (逻辑与)
            constituent_names = state_name.split('+')
            indices = [base_classes.index(n) for n in constituent_names]
            y_probs_specific[:, s_idx] = np.min(y_probs_base[:, indices], axis=1)
        else:
            # 单项故障状态直接对应其神经元概率
            idx = base_classes.index(state_name)
            y_probs_specific[:, s_idx] = y_probs_base[:, idx]

    # 4. 计算指标
    precision, recall, ap = dict(), dict(), dict()
    for i in range(len(distinct_states)):
        precision[i], recall[i], _ = precision_recall_curve(y_true_specific[:, i], y_probs_specific[:, i])
        ap[i] = average_precision_score(y_true_specific[:, i], y_probs_specific[:, i])

    # 计算全局 Micro-average
    precision["micro"], recall["micro"], _ = precision_recall_curve(y_true_base.ravel(), y_probs_base.ravel())
    ap["micro"] = average_precision_score(y_true_base, y_probs_base, average="micro")

    # 5. 绘图
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(9, 7), dpi=300)

    # A. 绘制 Micro-average (加粗黑虚线作为基准)
    plt.plot(recall["micro"], precision["micro"], color='black', linestyle='--', lw=3,
             label=f'Micro-average (AP = {ap["micro"]:.3f})')

    # B. 挑选要展示的类别
    colors = cycle(['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2'])
    target_compounds = [s for s in distinct_states if '+' in s]

    for i, state_name in enumerate(distinct_states):
        eng_state_name = get_eng_name(state_name)  # 转换为英文名

        if state_name in target_compounds:
            # 复合故障用彩色实线，加粗
            plt.plot(recall[i], precision[i], color=next(colors), lw=2,
                     label=f'Compound: {eng_state_name} (AP = {ap[i]:.3f})')
        elif state_name == 'zhengchang' or ap[i] > 0.98:
            # 优秀单项用彩色细线
            plt.plot(recall[i], precision[i], color=next(colors), lw=1, alpha=0.6,
                     label=f'Single: {eng_state_name} (AP = {ap[i]:.3f})')

    # 6. 视觉细节优化
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall (Sensitivity)', fontsize=12, fontweight='bold')
    plt.ylabel('Precision (PPV)', fontsize=12, fontweight='bold')
    plt.title('PR Curves for Multi-label Decoupling Performance', fontsize=14, fontweight='bold', pad=20)

    plt.legend(loc="lower left", fontsize=9, frameon=True, edgecolor='black')
    plt.grid(True, linestyle=':', alpha=0.5)

    plt.tight_layout()
    plt.savefig('Fig9_Compound_PR_Curve.png', bbox_inches='tight')
    plt.savefig('Fig9_Compound_PR_Curve.pdf', bbox_inches='tight')
    print("✅ 优化后的图9（全英文标签）已生成。")

# 使用说明：
# 在你的主环境中，需要准备好这些变量
# plot_figure_6_compound_optimized(mlb, test_loader, device, model, save_path='Fig6_Compound_Decomposition.png')
if __name__ == "__main__":
    # plot_figure_3()
    # plot_figure_4()
    # plot_figure_5()
    plot_figure_9_compound_optimized()
    print("\n🎉 所有论文用图已生成完毕，提供 .png 预览与 .pdf 矢量图供论文插入！")