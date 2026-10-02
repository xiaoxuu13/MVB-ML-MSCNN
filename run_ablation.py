import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import f1_score
import pandas as pd
from tqdm import tqdm

from config import CONFIG
from dataset import MVBDiskDataset
from model import ML_MSCNN
from train1 import get_classes  # 复用你的类别获取函数
from torch.amp import autocast, GradScaler


def run_single_experiment(exp_name, use_multiscale, use_cbam, train_loader, test_loader, classes):
    print(f"\n{'=' * 50}")
    print(f"🚀 开始消融实验: {exp_name}")
    print(f"{'=' * 50}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ML_MSCNN(num_classes=len(classes), use_multiscale=use_multiscale, use_cbam=use_cbam).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scaler = GradScaler('cuda')

    # 为了消融实验快速见效，我们采用稍激进的早停策略
    best_f1, patience, patience_counter = 0.0, 3, 0

    # 简化的训练循环 (省略验证集，直接在训练中看趋势)
    for epoch in range(CONFIG['epochs']):
        model.train()
        for imgs, labels in tqdm(train_loader, desc=f"Epoch {epoch + 1}", leave=False):
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with autocast('cuda'):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

    # 最终在测试集上评估
    model.eval()
    test_preds, test_targets = [], []
    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            probs = torch.sigmoid(model(imgs))
            thresholds = torch.full((len(classes),), 0.5).to(device)
            test_preds.append((probs > thresholds).cpu().numpy())
            test_targets.append(labels.numpy())

    import numpy as np
    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    micro_f1 = f1_score(test_targets, test_preds, average='micro')

    print(f"✅ {exp_name} 训练完成, 测试集 Micro-F1: {micro_f1:.4f}")
    return micro_f1


if __name__ == '__main__':
    classes = get_classes()
    mlb = MultiLabelBinarizer(classes=classes)
    mlb.fit([classes])

    # 消融实验使用稍大的 batch 以加速
    train_ds = MVBDiskDataset('./processed_data/train', mlb)
    test_ds = MVBDiskDataset('./processed_data/test', mlb)
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=128, shuffle=False, num_workers=4)

    results = []

    # 实验 1: 基础 CNN (无多尺度，无注意力)
    f1_base = run_single_experiment("Base CNN (Only 3x3)", False, False, train_loader, test_loader, classes)
    results.append({'Model': 'Base CNN', 'Micro_F1': f1_base})

    # 实验 2: 多尺度 CNN (有多尺度，无注意力)
    f1_ms = run_single_experiment("MS-CNN (w/o CBAM)", True, False, train_loader, test_loader, classes)
    results.append({'Model': 'MS-CNN (w/o CBAM)', 'Micro_F1': f1_ms})

    # 实验 3: 本文提出模型 (全模块)
    f1_proposed = run_single_experiment("ML-MSCNN (Proposed)", True, True, train_loader, test_loader, classes)
    results.append({'Model': 'ML-MSCNN (Proposed)', 'Micro_F1': f1_proposed})

    # 导出真实的对比数据
    pd.DataFrame(results).to_csv("ablation_results.csv", index=False)
    print("\n🎉 真实消融实验数据已生成至 ablation_results.csv")