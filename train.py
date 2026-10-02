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

def get_classes():
    # 从 processed_data/train 中扫描一次文件名获取所有类别
    sample_files = os.listdir('./processed_data/train')
    unique_faults = set()
    for f in sample_files:
        label_part = f.split('#')[0]
        for p in label_part.split('+'): unique_faults.add(p)
    return sorted(list(unique_faults))


def train_pipeline():
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

    # 3. 初始化模型
    model = ML_MSCNN(num_classes=len(classes)).to(device)
    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
    scaler = GradScaler('cuda')

    # 早停机制变量
    best_val_f1 = 0.0
    patience = 5
    patience_counter = 0

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

        print(f"Epoch {epoch + 1} Summary: Train Loss: {train_loss / len(train_loader):.4f} | Val F1: {val_f1:.4f}")

        # --- Checkpoint & Early Stopping ---
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
            print(">>> 新的最佳模型已保存！")
        else:
            patience_counter += 1
            print(f">>> 性能未提升 ({patience_counter}/{patience})")
            if patience_counter >= patience:
                print("早停触发！训练结束。")
                break

    # 5. Final Test (只跑一次)
    print("\n======== 最终测试集评估 (使用最佳模型) ========")
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    test_preds, test_targets = [], []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            # test_preds.append((torch.sigmoid(outputs) > 0.5).cpu().numpy())
            probs = torch.sigmoid(outputs)
            thresholds = torch.full((len(classes),), 0.5).to(device)
            test_preds.append((probs > thresholds).cpu().numpy())
            test_targets.append(labels.numpy())

    test_preds = np.vstack(test_preds)
    test_targets = np.vstack(test_targets)
    # ================= 插入优化第二步：互斥逻辑后处理 =================
    # 逻辑：如果模型预测出任意一种故障（非'zhengchang'类的其他列为1），
    # 那么强制将'zhengchang'这一列置为0。防止出现“即是故障又是正常”的笑话。

    # 1. 找到 'zhengchang' 在 classes 列表中的索引位置
    if 'zhengchang' in classes:
        zc_index = classes.index('zhengchang')

        # 2. 获取所有“故障类”的索引（除了 zhengchang 以外的所有列）
        fault_indices = [i for i in range(len(classes)) if i != zc_index]

        # 3. 找出哪些样本被预测出了故障 (只要故障列里有一个是 True/1，该样本就是故障)
        # np.any(..., axis=1) 检查每一行，如果故障列有任意一个为True，返回True
        has_any_fault = np.any(test_preds[:, fault_indices], axis=1)

        # 4. 强制修正：对于确诊故障的样本，把它的 'zhengchang' 列强制设为 0 (False)
        test_preds[has_any_fault, zc_index] = 0

        print(f">>> 已应用互斥逻辑：修正了 {np.sum(has_any_fault)} 个样本的正常标签冲突。")
    # ==============================================================

    # 生成详细报告
    report = classification_report(test_targets, test_preds, target_names=classes)
    print(report)

    # 保存报告到文件 (方便写论文复制)
    with open("classification_report.txt", "w") as f:
        f.write(report)
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
    train_pipeline()