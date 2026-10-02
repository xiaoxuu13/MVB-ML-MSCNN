# preprocess.py
import os
import pandas as pd
import numpy as np
from scipy.signal import stft
import cv2
import shutil
from config import CONFIG

RAW_ROOT = './raw_data'
SAVE_ROOT = './processed_data'





def process_and_save_aggregated(file_path, save_dir, label_name):
    """读取一个CSV，切片并保存为一个聚合的.npy文件"""
    file_basename = os.path.basename(file_path).replace('.csv', '')

    try:
        # 读取数据
        df = pd.read_csv(file_path, usecols=['AnalogChnB'])
        signal = np.nan_to_num(df['AnalogChnB'].values)

        samples_list = []

        # 循环切片
        for i in range(0, len(signal) - CONFIG['window_size'], CONFIG['step_size']):
            segment = signal[i: i + CONFIG['window_size']]

            # STFT变换
            # _, _, Zxx = stft(segment, fs=1.0, nperseg=CONFIG['nperseg'])
            # mag = np.abs(Zxx)
            # mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
            # 删掉所有 np.log1p 和 GLOBAL_MAX_LOG 的逻辑
            # 严格恢复成你最初这三行：
            _, _, Zxx = stft(segment, fs=1.0, nperseg=CONFIG['nperseg'])
            mag = np.abs(Zxx)
            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)

            # # 方案 B: 保留底噪阈值的相对归一化 (防止微小噪声被放大)
            # noise_threshold = 0.1  # 根据实际信号能量调整
            # mag_range = mag.max() - mag.min()
            # if mag_range > noise_threshold:
            #     mag = (mag - mag.min()) / mag_range
            # else:
            #     # 如果极差很小（纯正常底噪），则不进行拉伸，直接赋低值或零
            #     mag = np.zeros_like(mag)
            img = cv2.resize(mag, CONFIG['image_size'])

            samples_list.append(img.astype(np.float16))  # 暂存到列表

        if len(samples_list) > 0:
            # 聚合：将列表堆叠成一个大数组 (N, 64, 64)
            data_block = np.stack(samples_list)

            # 保存为一个文件：Label#OriginalName.npy
            save_name = f"{label_name}#{file_basename}.npy"
            np.save(os.path.join(save_dir, save_name), data_block)
            print(f"已保存: {save_name} (包含 {len(samples_list)} 个样本)")
            return len(samples_list)

    except Exception as e:
        print(f"Error: {e}")
        return 0

def main():
    # 清理旧数据
    if os.path.exists(SAVE_ROOT): shutil.rmtree(SAVE_ROOT)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(SAVE_ROOT, split), exist_ok=True)

    # 扫描文件夹
    folders = [d for d in os.listdir(RAW_ROOT) if os.path.isdir(os.path.join(RAW_ROOT, d))]

    for folder in folders:
        folder_path = os.path.join(RAW_ROOT, folder)
        csv_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.csv')])

        # --- 关键策略：只取前 N 个文件 ---
        limit = CONFIG['files_per_class']
        if len(csv_files) < limit:
            print(f"警告: {folder} 只有 {len(csv_files)} 个文件，将全部使用。")
            selected_files = csv_files
        else:
            selected_files = csv_files[:limit]

        print(f"正在处理 {folder}: 选中 {len(selected_files)} 个文件...")

        # --- 分配 Train/Val/Test ---
        # 按照 config 中的 [3, 1, 1] 逻辑分配
        n_train = CONFIG['split_ratio'][0]
        n_val = CONFIG['split_ratio'][1]

        train_fs = selected_files[:n_train]
        val_fs = selected_files[n_train: n_train + n_val]
        test_fs = selected_files[n_train + n_val:]

        # 执行处理
        for f in train_fs: process_and_save_aggregated(os.path.join(folder_path, f), os.path.join(SAVE_ROOT, 'train'), folder)
        for f in val_fs: process_and_save_aggregated(os.path.join(folder_path, f), os.path.join(SAVE_ROOT, 'val'), folder)
        for f in test_fs: process_and_save_aggregated(os.path.join(folder_path, f), os.path.join(SAVE_ROOT, 'test'), folder)

    print("✅ 预处理完成！")


if __name__ == '__main__':
    main()