import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
import cv2
from mpl_toolkits.mplot3d import Axes3D


# --- 1. 模拟/读取真实数据 ---
# 建议替换为你本地的一个真实csv路径
def get_real_data(file_path):
    df = pd.read_csv(file_path, usecols=['AnalogChnB'])
    return np.nan_to_num(df['AnalogChnB'].values[:5000])  # 取前5000个点展示


# --- 2. 绘制原始信号 (Raw Signal) ---
def plot_raw_signal(signal):
    plt.figure(figsize=(10, 2))
    plt.plot(signal, color='#1f77b4', linewidth=0.8)
    plt.axis('on')  # 去掉坐标轴，方便后期拼图
    plt.savefig('asset_1_raw.png', bbox_inches='tight', transparent=True)
    plt.close()


# --- 3. 绘制 3D 时频山峦图 (The "Pro" Look) ---
def plot_3d_stft(signal):
    # 匹配你的 config 参数
    window_size = 2048
    segment = signal[:window_size]
    f, t, Zxx = stft(segment, fs=1.0, nperseg=256)
    mag = np.abs(Zxx)

    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # 建立网格
    T, F = np.meshgrid(t, f)
    # 使用 Viridis 或 Plasma 这种学术常用色系
    surf = ax.plot_surface(T, F, mag, cmap='viridis', edgecolor='none', alpha=0.9)

    ax.view_init(elev=30, azim=45)  # 调整视角
    plt.axis('off')
    plt.savefig('asset_2_3d_stft.svg', bbox_inches='tight', transparent=True)

    plt.close()


# --- 4. 绘制最终输入图 (Final 64x64 Input) ---
def plot_final_input(signal):
    window_size = 2048
    segment = signal[:window_size]
    _, _, Zxx = stft(segment, fs=1.0, nperseg=256)
    mag = np.abs(Zxx)
    # 匹配你的归一化和缩放逻辑
    mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
    img = cv2.resize(mag, (64, 64))

    plt.figure(figsize=(4, 4))
    plt.imshow(img, cmap='viridis')
    plt.axis('on')
    plt.savefig('asset_3_final.png', bbox_inches='tight', transparent=True)
    plt.close()


# 执行生成 (请确保路径下有csv文件)
signal = get_real_data('./raw_data/jiediyi+yuanduan/jiediyi+yuanduan0.csv')
plot_raw_signal(signal)
# plot_3d_stft(signal)
# plot_final_input(signal)
print("素材生成完毕：asset_1, 2, 3 .pdf")