import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as patches


def plot_real_diagnostic_logic():
    # 1. 填入你提供的真实数据
    classes = ['duaner', 'duanyi', 'jiedier', 'jiediyi', 'jinduan', 'shuangduan', 'yuanduan', 'zhengchang']
    # 真实概率值 (处理极小值为0，方便可视化显示)
    raw_probs = [2.45e-19, 1.0, 6.81e-22, 8.02e-30, 1.0, 8.30e-16, 3.18e-12, 9.64e-15]
    probs = np.array(raw_probs)

    # 2. 设置画布与全局样式
    plt.rcParams['font.family'] = 'serif'
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # --- A. 绘制概率热力图矩阵 (Real Logits Matrix) ---
    start_x = 2
    box_w = 1.0
    box_h = 0.8
    y_pos = 4.2

    ax.text(start_x - 0.2, y_pos + 0.4, "Output Probabilities (Sigmoid)",
            fontsize=12, fontweight='bold', ha='right', color='#2c3e50')

    for i, p in enumerate(probs):
        # 映射颜色：1.0 对应深蓝色，趋于 0 对应浅灰色
        color = plt.cm.Blues(p * 0.8 + 0.1) if p > 0.1 else '#f5f5f5'
        edge_color = '#2980b9' if p > 0.5 else '#bdc3c7'
        lw = 2 if p > 0.5 else 1

        rect = patches.Rectangle((start_x + i * box_w, y_pos), box_w, box_h,
                                 linewidth=lw, edgecolor=edge_color, facecolor=color)
        ax.add_patch(rect)

        # 标注类别名称
        ax.text(start_x + i * box_w + box_w / 2, y_pos + box_h + 0.2, classes[i],
                ha='center', va='bottom', fontsize=10, rotation=30)

        # 标注具体数值 (如果是 1.0 加粗显示)
        text_val = "1.00" if p > 0.99 else f"{p:.1e}"
        weight = 'bold' if p > 0.5 else 'normal'
        ax.text(start_x + i * box_w + box_w / 2, y_pos + box_h / 2, text_val,
                ha='center', va='center', fontsize=9, fontweight=weight)

    # --- B. 绘制独立判定阈值线 (Decoupling Logic) ---
    threshold_y = 3.2
    ax.axhline(y=threshold_y, xmin=0.18, xmax=0.82, color='#e74c3c', linestyle='--', alpha=0.6)
    ax.text(start_x + 8.2, threshold_y, "Threshold = 0.5", color='#e74c3c', fontsize=10, va='center')

    # --- C. 绘制诊断报告结果框 (Final Report) ---
    report_rect = patches.FancyBboxPatch((4, 0.5), 4, 1.8, boxstyle="round,pad=0.2",
                                         linewidth=2, edgecolor='#2980b9', facecolor='#ebf5fb')
    ax.add_patch(report_rect)

    # 提取判定结果
    detected_faults = [classes[i] for i, p in enumerate(probs) if p > 0.5]
    result_str = " + ".join(detected_faults)

    ax.text(6, 2.0, "DIAGNOSTIC REPORT", fontsize=12, fontweight='bold', ha='center', color='#2980b9')
    ax.text(4.5, 1.3, "Status:", fontsize=10, fontweight='bold')
    ax.text(5.5, 1.3, "Compound Fault Detected", fontsize=10, color='#c0392b')
    ax.text(4.5, 0.8, "Result:", fontsize=10, fontweight='bold')
    ax.text(5.5, 0.8, f"{result_str}", fontsize=11, fontweight='bold', color='#1f618d', family='monospace')

    # --- D. 绘制连接箭头 ---
    for i, p in enumerate(probs):
        if p > 0.5:
            # 从激活的格子指向结果框
            ax.annotate('', xy=(6, 2.3), xytext=(start_x + i * box_w + box_w / 2, y_pos),
                        arrowprops=dict(arrowstyle='->', color='#2980b9', lw=1.5, connectionstyle="arc3,rad=-0.2"))

    plt.tight_layout()
    plt.savefig('real_diagnostic_output.svg', format='svg', bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    plot_real_diagnostic_logic()