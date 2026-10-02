import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_multilabel_logits():
    # 1. 设置基础标签（基于你代码中的 unique_faults 逻辑）
    # 假设你的 8 个基础类别如下：
    base_classes = ['duaner', 'duanyi', 'jiedier', 'jiediyi', 'jinduan', 'shuangduan', 'yuanduan', 'zhengchang']

    # 2. 模拟一个复合故障样本的输出概率 (Logits 经过 Sigmoid 后)
    # 场景：检测到了 duanyi + jiediyi
    sample_probs = np.array([0.02, 0.98, 0.05, 0.96, 0.11, 0.03, 0.01, 0.01])

    # 设置画布
    fig, ax = plt.subplots(figsize=(10, 3))

    # 3. 绘制热力图矩阵 (1x8)
    sns.heatmap(sample_probs.reshape(1, -1),
                annot=True,
                fmt=".2f",
                cmap="YlGnBu",
                cbar_kws={'label': 'Probability'},
                xticklabels=base_classes,
                yticklabels=False,
                annot_kws={"size": 12, "weight": "bold"})

    # 4. 视觉装饰
    plt.title("Output Layer Logic: Independent Probability Matrix (Decoupling)",
              fontsize=14, pad=20, fontweight='bold', family='serif')
    plt.xticks(rotation=45, ha='right', fontsize=10, family='serif')

    # 5. 添加阈值判定标注 (Threshold=0.5)
    for i, p in enumerate(sample_probs):
        if p > 0.5:
            # 在高概率格子上加一个红框高亮
            rect = plt.Rectangle((i, 0), 1, 1, fill=False, edgecolor='red', lw=3)
            ax.add_patch(rect)
            # 添加指向结论的箭头标注
            ax.annotate('Active', xy=(i + 0.5, 0), xytext=(i + 0.5, -0.3),
                        arrowprops=dict(facecolor='black', arrowstyle='->'),
                        ha='center', color='red', fontweight='bold')

    # 6. 最终合成结论展示
    final_res = " + ".join([base_classes[i] for i, p in enumerate(sample_probs) if p > 0.5])
    plt.text(4, 1.8, f"Final Diagnosis: {final_res}",
             bbox=dict(facecolor='white', alpha=0.8, edgecolor='blue', boxstyle='round,pad=0.5'),
             fontsize=14, ha='center', color='blue', fontweight='bold')

    plt.tight_layout()
    plt.savefig('multilabel_logits_matrix.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    plot_multilabel_logits()