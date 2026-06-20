import torch
from torch.utils.data import Dataset
import os
import glob
import numpy as np


class MVBDiskDataset(Dataset):
    def __init__(self, data_dir, mlb):
        """
        不直接加载图片，只记录每个样本所属的文件路径和索引
        """
        self.mlb = mlb
        self.file_info = []  # 存储格式：[(file_path, sample_idx_in_file, label_vec), ...]

        print(f"正在扫描数据集目录: {data_dir} ...")
        file_paths = glob.glob(os.path.join(data_dir, '*.npy'))

        for path in file_paths:
            # 1. 仅解析标签（不读数据内容）
            filename = os.path.basename(path)
            label_str = filename.split('#')[0]
            faults = label_str.split('+')
            label_vec = self.mlb.transform([faults])[0].astype(np.float32)

            # 2. 预读一下这个文件有多少个样本（这步很快）
            # 使用 mmap_mode 可以不占内存地读取数组形状
            data_mmap = np.load(path, mmap_mode='r')
            num_samples = data_mmap.shape[0]

            # 3. 记录索引
            for i in range(num_samples):
                self.file_info.append((path, i, label_vec))

        print(f"扫描完成，总计样本数: {len(self.file_info)}")

    def __len__(self):
        return len(self.file_info)

    def __getitem__(self, idx):
        path, sample_idx, label_vec = self.file_info[idx]

        # 实时从磁盘读取该文件的该样本
        # 注意：np.load 配合 mmap_mode='r' 非常快，且不吃内存
        data_block = np.load(path, mmap_mode='r')
        img = data_block[sample_idx].astype(np.float32)

        # 归一化或转为 Tensor
        img_tensor = torch.from_numpy(img).unsqueeze(0)  # 变为 (1, H, W)
        label_tensor = torch.from_numpy(label_vec)

        return img_tensor, label_tensor
# import torch
# from torch.utils.data import Dataset
# import os
# import glob
# import numpy as np
# from torchvision import transforms
#
# class MVBDiskDataset(Dataset):
#     def __init__(self, data_dir, mlb, augment=False):
#         """
#         参数:
#             data_dir: 数据目录
#             mlb: MultiLabelBinarizer
#             augment: 是否启用数据增强（仅训练集）
#         """
#         self.mlb = mlb
#         self.augment = augment
#         self.file_info = []  # 存储格式：[(file_path, sample_idx_in_file, label_vec), ...]
#
#         print(f"正在扫描数据集目录: {data_dir} ...")
#         file_paths = glob.glob(os.path.join(data_dir, '*.npy'))
#
#         for path in file_paths:
#             filename = os.path.basename(path)
#             label_str = filename.split('#')[0]
#             faults = label_str.split('+')
#             label_vec = self.mlb.transform([faults])[0].astype(np.float32)
#
#             data_mmap = np.load(path, mmap_mode='r')
#             num_samples = data_mmap.shape[0]
#
#             for i in range(num_samples):
#                 self.file_info.append((path, i, label_vec))
#
#         print(f"扫描完成，总计样本数: {len(self.file_info)}")
#
#         # 定义数据增强变换（仅当 augment=True 时使用）
#         if self.augment:
#             self.transform = transforms.Compose([
#                 transforms.RandomHorizontalFlip(p=0.5),
#                 transforms.RandomRotation(10),
#                 transforms.RandomAffine(degrees=0, translate=(0.05, 0.05)),
#                 transforms.Normalize(mean=[0.5], std=[0.5])  # 将像素值归一化到 [-1,1]
#             ])
#         else:
#             self.transform = transforms.Compose([
#                 transforms.Normalize(mean=[0.5], std=[0.5])
#             ])
#
#     def __len__(self):
#         return len(self.file_info)
#
#     def __getitem__(self, idx):
#         path, sample_idx, label_vec = self.file_info[idx]
#
#         data_block = np.load(path, mmap_mode='r')
#         img = data_block[sample_idx].astype(np.float32)
#
#         # 转换为 Tensor 并添加通道维度 (1, H, W)
#         img_tensor = torch.from_numpy(img).unsqueeze(0)
#
#         # 应用变换（包括归一化）
#         img_tensor = self.transform(img_tensor)
#
#         label_tensor = torch.from_numpy(label_vec)
#         return img_tensor, label_tensor