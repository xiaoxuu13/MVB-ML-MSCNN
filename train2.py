# train.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report
import os
import numpy as np
from tqdm import tqdm
import pandas as pd
import warnings
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

from config import CONFIG
from dataset import MVBDiskDataset
from model import ML_MSCNN
from torch.amp import autocast, GradScaler

# ---------- 辅助函数 ----------
def get_classes():
    sample_files = os.listdir('./processed_data/train')
    unique_faults = set()
    for f in sample_files:
        label_part = f.split('#')[0]
        for p in label_part.split('+'):
            unique_faults.add(p)
    return sorted(list(unique_faults))

def plot_tsne(sampled_features, sampled_labels_str, title, filename):
    print(f"\n======== 正在生成 t-SNE 图像: {filename} ========")
    print("正在计算 t-SNE 降维 (这通常需要几十秒，请耐心等待)...")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(sampled_features)

    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=sampled_labels_str,
        palette=sns.color_palette("tab20", len(set(sampled_labels_str))),
        legend="full",
        alpha=0.8,
        s=50
    )
    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f">>> ✅ {filename} 已成功保存。")

class MultiLabelFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0):
        super(MultiLabelFocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, inputs, targets):
        bce_loss = self.bce(inputs, targets)
        pt = torch.exp(-bce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        return focal_loss.mean()

def find_best_thresholds(val_loader, model, device, classes, steps=20):
    """在验证集上为每个类别搜索最佳阈值"""
    model.eval()
    all_probs = []
    all_targets = []
    with torch.no_grad():
        for imgs, labels in tqdm(val_loader, desc="Searching thresholds"):
            imgs = imgs.to(device)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(labels.numpy())
    all_probs = np.vstack(all_probs)
    all_targets = np.vstack(all_targets)

    best_thresholds = np.full(len(classes), 0.5)
    for i in range(len(classes)):
        best_f1 = 0
        for th in np.linspace(0.3, 0.7, steps):
            pred = (all_probs[:, i] > th).astype(int)
            f1 = f1_score(all_targets[:, i], pred)
            if f1 > best_f1:
                best_f1 = f1
                best_thresholds[i] = th
    print(f"最佳阈值: {dict(zip(classes, best_thresholds))}")
    return best_thresholds

def apply_mutual_exclusion(test_preds, test_probs, classes):
    """后处理：处理故障对之间的互斥（如 duaner 与 duanyi 不能共存）"""
    # 定义互斥对（根据你的故障含义添加）
    mutual_exclusion_pairs = [
        ('duaner', 'duanyi'),
        ('jiedier', 'jiediyi'),
        # 如果还有其他互斥关系，请在此添加
    ]
    for a, b in mutual_exclusion_pairs:
        if a in classes and b in classes:
            idx_a = classes.index(a)
            idx_b = classes.index(b)
            mask = (test_preds[:, idx_a] == 1) & (test_preds[:, idx_b] == 1)
            if mask.any():
                # 保留概率高的，另一个置0
                probs_a = test_probs[mask, idx_a]
                probs_b = test_probs[mask, idx_b]
                test_preds[mask, idx_a] = (probs_a > probs_b).astype(int)
                test_preds[mask, idx_b] = (probs_b >= probs_a).astype(int)
                print(f"互斥对 ({a}, {b}) 修正了 {mask.sum()} 个样本")
    return test_preds

# ---------- 主训练流程 ----------
def train_pipeline():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = get_classes()
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])
    print(f"检测到故障类别: {classes}")

    # 加载数据集（训练集启用增强）
    train_ds = MVBDiskDataset('./processed_data/train', mlb, augment=CONFIG['use_augmentation'])
    val_ds = MVBDiskDataset('./processed_data/val', mlb, augment=False)
    test_ds = MVBDiskDataset('./processed_data/test', mlb, augment=False)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True,
                              num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                            num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False,
                             num_workers=4)

    # 初始化模型
    model = ML_MSCNN(num_classes=len(classes)).to(device)

    # ---------- 计算类别权重 ----------
    def get_pos_weight(dataset, num_classes):
        """使用温和的权重策略"""
        pos_counts = torch.zeros(num_classes)
        for _, label in dataset:
            pos_counts += label
        neg_counts = len(dataset) - pos_counts

        # 使用平方根缩放，避免权重过大
        pos_weight = (neg_counts / (pos_counts + 1e-8)) ** 0.5
        pos_weight = torch.clamp(pos_weight, max=5.0)  # 限制最大权重

        return pos_weight

    # 使用更温和的权重
    pos_weight = get_pos_weight(train_ds, len(classes)).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print("使用带权重的 BCEWithLogitsLoss (温和权重)")
    print(f"类别权重: {pos_weight.cpu().numpy()}")

    # 降低学习率
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0005)  # 从0.001降低到0.0005
    scaler = GradScaler('cuda')

    # 学习率调度器
    scheduler = None
    if CONFIG['use_scheduler']:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='max', factor=CONFIG['scheduler_factor'],
            patience=CONFIG['scheduler_patience'], verbose=True
        )

    # 早停
    best_val_f1 = 0.0
    patience_counter = 0
    history_log = {'epoch': [], 'train_loss': [], 'val_f1': [], 'lr': []}

    print(f"开始训练... (每轮约 {len(train_ds) // CONFIG['batch_size']} 个Step)")

    for epoch in range(CONFIG['epochs']):
        # --- Training ---
        model.train()
        train_loss = 0
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{CONFIG['epochs']}")

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

        # --- Validation ---
        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                probs = torch.sigmoid(outputs)
                # 使用当前固定阈值 0.5 做验证（后续会搜索最优阈值）
                preds = (probs > 0.5).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(labels.numpy())

        val_preds = np.vstack(val_preds)
        val_targets = np.vstack(val_targets)
        val_f1 = f1_score(val_targets, val_preds, average='micro')
        avg_train_loss = train_loss / len(train_loader)

        history_log['epoch'].append(epoch+1)
        history_log['train_loss'].append(avg_train_loss)
        history_log['val_f1'].append(val_f1)
        history_log['lr'].append(optimizer.param_groups[0]['lr'])

        print(f"Epoch {epoch+1} Summary: Train Loss: {avg_train_loss:.4f} | Val F1: {val_f1:.4f} | LR: {optimizer.param_groups[0]['lr']:.6f}")

        # --- 学习率调度（根据验证 F1）---
        if scheduler:
            scheduler.step(val_f1)

        # --- 模型保存与早停 ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> 新的最佳模型已保存！")
        else:
            patience_counter += 1
            print(f">>> 性能未提升 ({patience_counter}/{CONFIG['patience']})")
            if patience_counter >= CONFIG['patience']:
                print("早停触发！训练结束。")
                break

    # 保存训练日志
    pd.DataFrame(history_log).to_csv("training_log.csv", index=False)
    print(">>> 训练日志已保存至 training_log.csv")

    # ========== 最终测试 ==========
    print("\n======== 最终测试集评估 ========")
    warnings.filterwarnings("ignore", category=FutureWarning)

    model.load_state_dict(torch.load("best_model.pth", weights_only=True))
    model.eval()

    # --- 可选：在验证集上搜索最佳阈值 ---
    if CONFIG['threshold_search']:
        best_thresholds = find_best_thresholds(val_loader, model, device, classes, steps=CONFIG['threshold_steps'])
    else:
        best_thresholds = np.full(len(classes), 0.5)

    # --- 收集测试集预测结果与特征 ---
    test_preds, test_targets = [], []
    test_probs_list = []
    features_before_list, features_after_list, tsne_targets_list = [], [], []

    TOTAL_TEST_BATCHES = len(test_loader)
    TARGET_COLLECT_BATCHES = 40
    collect_interval = max(1, TOTAL_TEST_BATCHES // TARGET_COLLECT_BATCHES)
    print(f">>> 测试集总 batch 数: {TOTAL_TEST_BATCHES}")
    print(f">>> 将每隔 {collect_interval} 个 batch 采样一个，用于 t-SNE 绘图...")

    current_batch_idx = 0
    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Test"):
            imgs = imgs.to(device)

            # 前向传播（同时获取特征）
            outputs, feat_after = model(imgs, return_features=True)
            probs = torch.sigmoid(outputs).cpu().numpy()

            # 使用搜索到的最佳阈值进行预测
            preds = (probs > best_thresholds).astype(int)

            test_probs_list.append(probs)
            test_preds.append(preds)
            test_targets.append(labels.numpy())

            # 均匀采样用于 t-SNE
            if current_batch_idx % collect_interval == 0:
                feat_before = imgs.cpu().view(imgs.size(0), -1).numpy()
                features_before_list.append(feat_before)
                features_after_list.append(feat_after.cpu().numpy())
                tsne_targets_list.append(labels.numpy())

            current_batch_idx += 1

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    test_probs = np.vstack(test_probs_list)

    # 合并 t-SNE 特征
    features_before = np.vstack(features_before_list) if features_before_list else np.array([])
    features_after = np.vstack(features_after_list) if features_after_list else np.array([])
    tsne_targets = np.vstack(tsne_targets_list) if tsne_targets_list else np.array([])

    # ========== 后处理：互斥逻辑 ==========
    # 1. 正常与故障的互斥（已有）
    # if 'zhengchang' in classes:
    #     zc_index = classes.index('zhengchang')
    #     fault_indices = [i for i in range(len(classes)) if i != zc_index]
    #     has_any_fault = np.any(test_preds[:, fault_indices], axis=1)
    #     test_preds[has_any_fault, zc_index] = 0
    # if 'zhengchang' in classes:
    #     zc_index = classes.index('zhengchang')
    #     fault_indices = [i for i in range(len(classes)) if i != zc_index]
    #     has_any_fault = np.any(test_preds[:, fault_indices], axis=1)
    #     test_preds[has_any_fault, zc_index] = 0
    #     print(f">>> 已应用安全互斥逻辑：修正了 {np.sum(has_any_fault)} 个假阳性正常标签。")

    # 2. 故障对之间的互斥
    # test_preds = apply_mutual_exclusion(test_preds, test_probs, classes)

    # ========== 评估与报告 ==========
    np.savez('pr_curve_data.npz', probs=test_probs, targets=test_targets, classes=classes)

    report = classification_report(test_targets, test_preds, target_names=classes)
    with open("classification_report.txt", "w") as f:
        f.write(report)
    print("\n分类报告（单标签）:")
    print(report)

    # 复合故障评估
    print("\n======== 复合故障整体匹配度评估 ========")
    pred_tuples = mlb.inverse_transform(test_preds)
    true_tuples = mlb.inverse_transform(test_targets)

    pred_compound_classes = ["+".join(p) if p else "zhengchang" for p in pred_tuples]
    true_compound_classes = ["+".join(t) if t else "zhengchang" for t in true_tuples]
    unique_true_classes = sorted(list(set(true_compound_classes)))

    compound_report = classification_report(
        true_compound_classes,
        pred_compound_classes,
        labels=unique_true_classes,
        digits=4
    )
    print(compound_report)
    with open("compound_classification_report.txt", "w") as f:
        f.write(compound_report)

    # ========== t-SNE 可视化 ==========
    if len(features_before) > 0:
        print("\n>>> 开始进行具可比性的 t-SNE 降维绘图 (Raw vs. Decoupled)...")
        tsne_true_tuples = mlb.inverse_transform(tsne_targets)
        tsne_labels_str = ["+".join(t) if t else "zhengchang" for t in tsne_true_tuples]
        unique_sampled_classes = sorted(list(set(tsne_labels_str)))
        print(f">>> 采样特征中包含 {len(unique_sampled_classes)} 种独特的故障状态。")

        plot_tsne(features_before, tsne_labels_str, "t-SNE Viz: Raw STFT Image Feature Space", "tsne_before.png")
        plot_tsne(features_after, tsne_labels_str, "t-SNE Viz: Decoupled MSCNN-CBAM Feature Space", "tsne_after.png")
    else:
        print(">>> 警告：未收集到足够的 t-SNE 采样特征，跳过绘图。")

    print("\n======== 核心诊断可视化任务已完成 ========")
    print(">>> 请查看 tsne_before.png 和 tsne_after.png。")

if __name__ == '__main__':
    train_pipeline()