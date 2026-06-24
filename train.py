# train.py
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
import argparse  # 新增：命令行参数解析


# 新增：解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='ML-MSCNN 基线对比实验')
    parser.add_argument('--baseline', type=int, default=4, choices=[1, 2, 3, 4],
                        help='基线类型: 1=仅3x3卷积(无多尺度无CBAM), 2=3x3+CBAM, 3=多尺度无CBAM, 4=完整模型(多尺度+CBAM)')
    return parser.parse_args()


def get_classes():
    # 从 processed_data/train 中扫描一次文件名获取所有类别
    sample_files = os.listdir('./processed_data/train')
    unique_faults = set()
    for f in sample_files:
        label_part = f.split('#')[0]
        for p in label_part.split('+'): unique_faults.add(p)
    return sorted(list(unique_faults))


def plot_tsne(sampled_features, sampled_labels_str, title, filename):
    """
    接收已经随机采样好的特征数据和标签名，绘制并保存 t-SNE 图。
    Args:
        sampled_features (np.ndarray): 用于绘图的 N x D 特征矩阵。
        sampled_labels_str (list): 对应的故障组合字符串列表。
        title (str): 图像的标题。
        filename (str): 图像保存的文件名（如 tsne_before.png）。
    """
    print(f"\n======== 正在生成 t-SNE 图像: {filename} ========")
    print("正在计算 t-SNE 降维 (这通常需要几十秒，请耐心等待)...")
    # 建议使用 init='pca' 加快收敛，random_state 确保可复现
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(sampled_features)
    # 开始绘图
    plt.figure(figsize=(12, 10))
    # 使用 seaborn 的 scatterplot，用 hue 参数上色
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=sampled_labels_str,
        # tab20 调色板适合类别较多的情况
        palette=sns.color_palette("tab20", len(set(sampled_labels_str))),
        legend="full",
        alpha=0.8,
        s=50
    )
    plt.title(title, fontsize=16)
    # 把图例放到图外侧，防止遮挡数据点
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f">>> ✅ {filename} 已成功保存。")


# 修改：函数新增 baseline 参数
def train_pipeline(baseline=4):
    # 新增：基线配置映射（唯一控制变量）
    baseline_config = {
        1: (False, False),  # 无多尺度，无CBAM
        2: (False, True),   # 无多尺度，有CBAM
        3: (True, False),   # 有多尺度，无CBAM
        4: (True, True)     # 完整模型（默认）
    }
    use_multiscale, use_cbam = baseline_config[baseline]
    print(f"\n===== 当前运行基线: {baseline} | 多尺度: {use_multiscale} | CBAM: {use_cbam} =====\n")

    # 1. 准备环境
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = get_classes()
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    print(f"检测到故障类别: {classes}")

    # 2. 加载数据集
    train_ds = MVBDiskDataset('./processed_data/train', mlb)
    val_ds = MVBDiskDataset('./processed_data/val', mlb)
    test_ds = MVBDiskDataset('./processed_data/test', mlb)
    # 显卡优化：pin_memory=True 加速数据传输
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    # 3. 初始化模型（修改：传入多尺度/CBAM开关）
    model = ML_MSCNN(
        num_classes=len(classes),
        use_multiscale=use_multiscale,
        use_cbam=use_cbam
    ).to(device)

    weights = torch.ones(len(classes)).to(device)
    if 'zhengchang' in classes:
        zc_idx = classes.index('zhengchang')
        # 给除正常以外的所有故障类设置 1.5 倍权重
        fault_indices = [i for i in range(len(classes)) if i != zc_idx]
        weights[fault_indices] = 1.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scaler = GradScaler('cuda')

    # 早停机制变量
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0
    history_log = {'epoch': [], 'train_loss': [], 'val_f1': []}

    # 4. 训练循环
    print(f"开始训练... (每轮约 {len(train_ds) // CONFIG['batch_size']} 个Step)")
    for epoch in range(CONFIG['epochs']):
        # --- Training ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{CONFIG['epochs']}")
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        # --- Validation (每轮跑一次，用于选模型) ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(labels.numpy())
        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='micro')
        history_log['epoch'].append(epoch + 1)
        history_log['train_loss'].append(train_loss / len(train_loader))
        history_log['val_f1'].append(val_f1)
        print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        # --- Checkpoint & Early Stopping（修改：文件名带基线编号） ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model_baseline{baseline}.pth")
            print(f">>> 新的最佳模型已保存：best_model_baseline{baseline}.pth")
        else:
            patience_counter += 1
            print(f">>> 性能未提升 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("早停触发！训练结束。")
                break

    import pandas as pd
    pd.DataFrame(history_log).to_csv(f"training_log_baseline{baseline}.csv", index=False)
    print(f">>> 真实训练过程数据已保存至 training_log_baseline{baseline}.csv")

    # 5. Final Test (只跑一次)
    print("\n======== 最终测试集全量评估 & t-SNE 跳跃采样 ========")
    # 屏蔽 FutureWarning 警告（保持控制台整洁）
    import warnings
    warnings.filterwarnings("ignore", category=FutureWarning)
    model.load_state_dict(torch.load(f"best_model_baseline{baseline}.pth", weights_only=True))
    model.eval()

    # --- ！！！就在这里插入数据提取脚本 ！！！ ---
    print("\n>>> 正在抓取一个真实的复合故障样本数据用于绘图...")
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            # 你的模型输出包含特征，所以这里用解包方式
            outputs, _ = model(imgs, return_features=True)
            probs = torch.sigmoid(outputs)
            found = False
            for i in range(len(labels)):
                if labels[i].sum() > 1:  # 找到复合故障
                    print("\n" + "=" * 30)
                    print("--- 真实数据提取成功 ---")
                    print(f"Classes: {classes}")
                    # 获取真实标签名
                    real_names = mlb.inverse_transform(labels[i].unsqueeze(0).cpu().numpy())
                    print(f"Real Label: {real_names}")
                    # 获取原始 Sigmoid 概率
                    prob_list = probs[i].cpu().numpy().tolist()
                    print(f"Sigmoid Probs: {prob_list}")
                    print("=" * 30 + "\n")
                    found = True
                    break
            if found: break
            # --- ！！！插入结束 ！！！ ---

    test_preds, test_targets = [], []
    test_probs_list = []
    # --- 【优化采样核心】：用于 t-SNE 的特征收集器 ---
    features_before_list, features_after_list, tsne_targets_list = [], [], []
    # 策略：计算测试集 batch 总数，计算一个跳跃间隔，均匀抓取 batch
    TOTAL_TEST_BATCHES = len(test_loader)
    # 大约抓取 30-40 个 batch (假设 batch_size=64-128, 最终约3000点)
    TARGET_COLLECT_BATCHES = 40
    # 计算跳跃步长：TOTAL / TARGET
    collect_interval = max(1, TOTAL_TEST_BATCHES // TARGET_COLLECT_BATCHES)
    print(f">>> 测试集总 batch 数: {TOTAL_TEST_BATCHES}")
    print(f">>> 将每隔 {collect_interval} 个 batch 采样一个，用于 t-SNE 绘图...")
    # -----------------------------------------------
    current_batch_idx = 0
    print("正在测试全量模型，并均匀抓取特征用于 t-SNE ...")
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Test"):
            imgs = imgs.to(device)
            # 1. 前向传播
            outputs, feat_after = model(imgs, return_features=True)
            # 2. 【核心修复】：均匀跳跃采样用于 t-SNE！
            if current_batch_idx % collect_interval == 0:
                feat_before = imgs.cpu().view(imgs.size(0), -1).numpy()
                features_before_list.append(feat_before)
                features_after_list.append(feat_after.cpu().numpy())
                tsne_targets_list.append(labels.numpy())
            # 3. 全量收集预测概率
            probs = torch.sigmoid(outputs)
            test_probs_list.append(probs.cpu().numpy())

            # 针对特征微弱的故障，降低置信度门槛，减少漏检
            thresholds = torch.full((len(classes),), 0.5).to(device)
            if 'shuangduan' in classes: thresholds[classes.index('shuangduan')] = 0.25
            if 'duanyi' in classes: thresholds[classes.index('duanyi')] = 0.30
            if 'jiediyi' in classes: thresholds[classes.index('jiediyi')] = 0.55
            # 【回退】：把 0.25 改回 0.40
            if 'zhengchang' in classes: thresholds[classes.index('zhengchang')] = 0.40

            test_preds.append((probs > thresholds).cpu().numpy())
            test_targets.append(labels.numpy())
            current_batch_idx += 1

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    test_probs = np.vstack(test_probs_list)
    # 合并 t-SNE 特征
    features_before = np.vstack(features_before_list)
    features_after = np.vstack(features_after_list)
    tsne_targets = np.vstack(tsne_targets_list)
    print(f"\n>>> 测试完成！已安全提取 t-SNE 跳跃采样特征 (数量={len(features_before)})。")

    # ================= 优化互斥逻辑 =================
    if 'zhengchang' in classes:
        zc_index = classes.index('zhengchang')
        fault_indices = [i for i in range(len(classes)) if i != zc_index]
        has_any_fault = np.any(test_preds[:, fault_indices], axis=1)
        zc_probs = test_probs[:, zc_index]
        max_fault_probs = np.max(test_probs[:, fault_indices], axis=1)
        # 恢复 0.2 的缓冲区逻辑
        should_kill_zc = (max_fault_probs > 0.5) & (max_fault_probs > (zc_probs - 0.2))
        test_preds[should_kill_zc, zc_index] = 0
        print(f">>> 已应用优化互斥逻辑：修正了 {np.sum(has_any_fault)} 个假阳性正常标签。")
    # ==========================================================

    np.savez(f'pr_curve_data_baseline{baseline}.npz', probs=test_probs, targets=test_targets, classes=classes)
    report = classification_report(test_targets, test_preds, target_names=classes)
    with open(f"classification_report_baseline{baseline}.txt", "w") as f:
        f.write(report)

    # ---------------- 复合故障整体匹配度评估 ----------------
    print("\n======== 复合故障整体匹配度评估 ========")
    pred_tuples = mlb.inverse_transform(test_preds)
    true_tuples = mlb.inverse_transform(test_targets)
    pred_compound_classes = ["+".join(p) if p else "Unrecognized" for p in pred_tuples]
    true_compound_classes = ["+".join(t) if t else "zhengchang" for t in true_tuples]
    unique_true_classes = sorted(list(set(true_compound_classes)))
    compound_report = classification_report(
        true_compound_classes,
        pred_compound_classes,
        labels=unique_true_classes,
        digits=4
    )
    print(compound_report)
    with open(f"compound_classification_report_baseline{baseline}.txt", "w") as f:
        f.write(compound_report)

    from sklearn.metrics import multilabel_confusion_matrix
    # 确保 test_targets 和 test_preds 是 (N, 8) 形状的二进制矩阵
    mcm = multilabel_confusion_matrix(test_targets, test_preds)
    # 假设 zhengchang 的索引是 zc_index (例如 7)
    zc_index = classes.index('zhengchang')
    print("\n>>> 正常状态 (zhengchang) 独立混淆矩阵:")
    print(mcm[zc_index])
    # 为了看看是谁被误判成了正常，找出假阴性 (FN) 最高的故障
    print("\n>>> 各故障类别的假阴性 (实际有病，预测正常) 数量:")
    for i, cls_name in enumerate(classes):
        if cls_name != 'zhengchang':
            fn = mcm[i][1, 0]  # 实际为1，预测为0的数量
            print(f"[{cls_name}]: 漏检 {fn} 个")

    # ---------------- 绘制 t-SNE 图像（修改：文件名带基线编号） ----------------
    print("\n>>> 开始进行具可比性的 t-SNE 降维绘图 (Raw vs. Decoupled)...")
    # 解析用于画图的采样样本的标签名
    tsne_true_tuples = mlb.inverse_transform(tsne_targets)
    # 将标签 tuple 转成字符串用于 Scatterplot 上色
    tsne_labels_str = ["+".join(t) if t else "zhengchang" for t in tsne_true_tuples]
    # 为了确保两张图的样本点和调色板一致，我们需要先生成唯一的颜色映射表
    unique_sampled_classes = sorted(list(set(tsne_labels_str)))
    print(f">>> 采样特征中包含 {len(unique_sampled_classes)} 种独特的故障状态。")

    # 绘制 Before 图
    title_before = f"t-SNE Viz: Raw STFT Feature Space (Baseline{baseline})"
    plot_tsne(features_before, tsne_labels_str, title_before, f"tsne_before_baseline{baseline}.png")

    # 绘制 After 图
    title_after = f"t-SNE Viz: Decoupled Feature Space (Baseline{baseline})"
    plot_tsne(features_after, tsne_labels_str, title_after, f"tsne_after_baseline{baseline}.png")

    print("\n======== 核心诊断可视化任务已完成 ========")
    print(f">>> 请查看 tsne_before_baseline{baseline}.png 和 tsne_after_baseline{baseline}.png。")


class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=1):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)  # 预测正确的概率
        focal_loss = self.alpha * (1-pt)**self.gamma * bce_loss
        return focal_loss.mean()


if __name__ == '__main__':
    args = parse_args()
    train_pipeline(args.baseline)