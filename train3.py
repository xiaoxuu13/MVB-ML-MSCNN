import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score, classification_report, silhouette_score
import os
import numpy as np
from tqdm import tqdm
from config import CONFIG
from dataset import MVBDiskDataset
from model import ML_MSCNN
from torch.amp import autocast, GradScaler
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import random

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


def to_eng(label_str):
    if not label_str: return "Normal"
    parts = label_str.split('+')
    eng_parts = [ENG_MAPPING.get(p, p) for p in parts]
    return "+".join(eng_parts)


def parse_args():
    parser = argparse.ArgumentParser(description='ML-MSCNN 基线对比实验')
    parser.add_argument('--baseline', type=int, default=4, choices=[1, 2, 3, 4],
                        help='基线类型: 1=仅3x3卷积, 2=3x3+CBAM, 3=多尺度无CBAM, 4=完整模型')
    parser.add_argument('--kernels', type=str, default="3,5,7", help='卷积核组合，如 3,5,7 或 3,3,3')
    parser.add_argument('--seed', type=int, default=42, help='随机种子，用于复现和统计')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_classes():
    sample_files = os.listdir('./processed_data/train')
    unique_faults = set()
    for f in sample_files:
        label_part = f.split('#')[0]
        for p in label_part.split('+'): unique_faults.add(p)
    return sorted(list(unique_faults))


def plot_tsne(sampled_features, sampled_labels_str, title, filename):
    print(f"\n======== 正在生成 t-SNE 图像: {filename} ========")
    tsne = TSNE(n_components=2, random_state=42, init='pca', learning_rate='auto')
    tsne_results = tsne.fit_transform(sampled_features)
    plt.figure(figsize=(12, 10))
    sns.scatterplot(
        x=tsne_results[:, 0], y=tsne_results[:, 1],
        hue=sampled_labels_str,
        palette=sns.color_palette("tab20", len(set(sampled_labels_str))),
        legend="full", alpha=0.8, s=50
    )
    plt.title(title, fontsize=16)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize=10)
    plt.tight_layout()
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f">>> ✅ {filename} 已成功保存。")


def train_pipeline(args):
    baseline = args.baseline
    k_str = args.kernels.split(',')
    kernel_sizes = (int(k_str[0]), int(k_str[1]), int(k_str[2]))

    baseline_config = {
        1: (False, False),
        2: (False, True),
        3: (True, False),
        4: (True, True)
    }
    use_multiscale, use_cbam = baseline_config[baseline]
    print(f"\n=== 当前基线: {baseline} | 卷积核: {kernel_sizes} | 种子: {args.seed} ===")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classes = get_classes()
    eng_classes = [ENG_MAPPING.get(c, c) for c in classes]  # 映射为英文

    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])

    train_ds = MVBDiskDataset('./processed_data/train', mlb)
    val_ds = MVBDiskDataset('./processed_data/val', mlb)
    test_ds = MVBDiskDataset('./processed_data/test', mlb)

    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=4)

    model = ML_MSCNN(
        num_classes=len(classes),
        use_multiscale=use_multiscale,
        use_cbam=use_cbam,
        kernel_sizes=kernel_sizes
    ).to(device)

    # ================= 新增：FLOPs 与 参数量计算 =================
    try:
        from thop import profile
        dummy_input = torch.randn(1, 1, 64, 64).to(device)
        macs, params = profile(model, inputs=(dummy_input,), verbose=False)
        print(f"\n>>> [计算复杂度] FLOPs (MACs): {macs}, Params: {params}")
    except ImportError:
        print("\n>>> [计算复杂度] 提示：请在终端执行 pip install thop 以自动计算 FLOPs。")
    # ==============================================================

    weights = torch.ones(len(classes)).to(device)
    if 'zhengchang' in classes:
        zc_idx = classes.index('zhengchang')
        fault_indices = [i for i in range(len(classes)) if i != zc_idx]
        weights[fault_indices] = 1.5
    criterion = nn.BCEWithLogitsLoss(pos_weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scaler = GradScaler('cuda')

    best_val_f1 = 0.0
    patience, patience_counter = 5, 0

    print("开始训练...")
    for epoch in range(CONFIG['epochs']):
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

        model.eval()
        val_preds, val_targets = [], []
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                outputs = model(imgs)
                preds = (torch.sigmoid(outputs) > 0.5).cpu().numpy()
                val_preds.append(preds)
                val_targets.append(labels.numpy())
        val_f1 = f1_score(np.vstack(val_targets), np.vstack(val_preds), average='micro')
        print(f"Epoch {epoch + 1} | Val F1: {val_f1:.4f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), f"best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= patience: break

    # --- 测试集评估 ---
    model.load_state_dict(torch.load(f"best_model.pth", weights_only=True))
    model.eval()

    test_preds, test_targets, test_probs_list = [], [], []
    features_before_list, features_after_list, tsne_targets_list = [], [], []

    collect_interval = max(1, len(test_loader) // 40)
    current_batch_idx = 0

    with torch.no_grad():
        for imgs, labels in tqdm(test_loader, desc="Final Test"):
            imgs = imgs.to(device)
            outputs, feat_after = model(imgs, return_features=True)

            if current_batch_idx % collect_interval == 0:
                features_before_list.append(imgs.cpu().view(imgs.size(0), -1).numpy())
                features_after_list.append(feat_after.cpu().numpy())
                tsne_targets_list.append(labels.numpy())

            probs = torch.sigmoid(outputs)
            test_probs_list.append(probs.cpu().numpy())
            test_preds.append((probs > 0.5).cpu().numpy())
            test_targets.append(labels.numpy())
            current_batch_idx += 1

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    test_probs = np.vstack(test_probs_list)
    features_before = np.vstack(features_before_list)
    features_after = np.vstack(features_after_list)
    tsne_targets = np.vstack(tsne_targets_list)

    # ================= 新增：阈值敏感性分析 =================
    print("\n======== 阈值敏感性分析 (Threshold Sensitivity) ========")
    for t in [0.3, 0.4, 0.5, 0.6, 0.7]:
        preds_t = (test_probs > t).astype(int)
        f1_t = f1_score(test_targets, preds_t, average='micro')
        print(f">>> Threshold: {t} -> Micro-F1: {f1_t:.4f}")
    # ==========================================================

    report = classification_report(test_targets, test_preds, target_names=eng_classes)
    print("\n" + report)

    # t-SNE 英文标签生成
    tsne_true_tuples = mlb.inverse_transform(tsne_targets)
    tsne_labels_str = [to_eng("+".join(t)) for t in tsne_true_tuples]

    # ================= 新增：轮廓系数 (Silhouette Score) =================
    print("\n======== 聚类特征定量评估 (Silhouette Score) ========")
    sil_before = silhouette_score(features_before, tsne_labels_str)
    sil_after = silhouette_score(features_after, tsne_labels_str)
    print(f">>> 原始特征 (Raw STFT): {sil_before:.4f}")
    print(f">>> 模型解耦深层特征 (Decoupled): {sil_after:.4f}")
    # ====================================================================

    plot_tsne(features_before, tsne_labels_str, "Raw STFT Feature Space", f"tsne_before.png")
    plot_tsne(features_after, tsne_labels_str, "Decoupled Deep Feature Space", f"tsne_after.png")
    print("\n>>> 恭喜，所有数据提取完毕！")


if __name__ == '__main__':
    args = parse_args()
    set_seed(args.seed)
    train_pipeline(args)