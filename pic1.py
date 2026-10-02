import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy import signal

import os

# 设置顶刊字体标准
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['figure.dpi'] = 300  # 矢量图设置，生成时使用savefig，dpi=300

# 1. 数据选取：模拟或读取真实数据 (示例数据，请替换为您的 preprocess.py 读取路径)
# 假设您已经挑选好 4 个 CSV 文件路径
data_paths = {
    'zhengchang': './raw_data/zhengchang/zheng0.csv',
    'duanyi': './raw_data/duanyi/duanB+0.csv',
    'jiediyi': './raw_data/jiediyi/B+jiedi0.csv',
    'jiediyi+daunyi' : './raw_data/jiediyi+duanyi/jiediyi+duanyi0.csv'
}
titles = ['zhengchang', 'duanyi', 'jiediyi', 'jiediyi+daunyi']
cmaps = ['viridis', 'jet']  # 示例使用，建议选用一致的 cmap
sample_data = np.random.randn(2048)  # 模拟原始信号，请替换为从您的csv读取的数据


def create_stft_contrast_chart(output_filename='figure1_stft_contrast.pdf'):
    fig, axes = plt.subplots(4, 1, figsize=(6, 10))  # figsize根据 Visio 布局调整
    fig.subplots_adjust(hspace=0.4)  # 调整子图间距

    for i, (state, path) in enumerate(data_paths.items()):
        ax = axes[i]

        # 1. 读取真实信号 (用真实读取逻辑替换 numpy 随机数)
        # 假设原始信号存储在 csv 中，需要提取差分电压信号
        raw_signal = load_signal_from_csv(path)
        # raw_signal = sample_data  # 使用示例数据

        # 2. 执行 preprocess.py 中的核心逻辑：STFT
        # 严格对齐 preprocess.py 的参数 (例如 nperseg, noverlap, fs)
        f, t, Zxx = signal.stft(raw_signal, fs=1.0, nperseg=256, noverlap=128)
        # 提取幅度谱
        amplitude_spectrogram = np.abs(Zxx)

        # 3. 绘图脚本要点实现：时频图
        # 设置坐标
        im = ax.pcolormesh(t, f / 1e6, amplitude_spectrogram, cmap=cmaps[0], shading='auto')  # f/1e6 转化为 MHz

        # 设置顶刊标准坐标轴
        ax.set_title(titles[i])
        ax.set_ylabel('Frequency (MHz)')

        # 统一 Times New Roman 字体在 plt.rcParams 中设置

        # 最后一图设置X轴
        if i == 3:
            ax.set_xlabel('Time (s)')
        else:
            ax.set_xticklabels([])  # 去除其他子图X轴

        # 4. 必须加上 Colorbar，通过ax设置能量强度
        cbar = fig.colorbar(im, ax=ax, orientation='vertical')
        cbar.set_label('Energy Magnitude')
        cbar.ax.tick_params(labelsize=8)  # 调整Colorbar字体

    # 去除多余白色边框并保存
    plt.savefig(output_filename, bbox_inches='tight', format='pdf', dpi=300)
    plt.close()
    print(f"矢量时频对比图已保存为 {output_filename}")


if __name__ == '__main__':
    create_stft_contrast_chart()