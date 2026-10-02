# config.py
CONFIG = {
    # STFT与切片参数
    'window_size': 2048,
    'step_size': 1024,  # 50% 重叠
    'nperseg': 256,  # 频率分辨率
    'image_size': (64, 64),  # 64x64 对 2060 最友好，且足够看清故障

    # 训练参数
    'batch_size': 64,  # 64x64图片下，6GB显存可以开到64甚至128
    'learning_rate': 0.001,
    'epochs': 30,

    # 数据选择策略
    'files_per_class': 5,  # 每个故障文件夹只取前5个文件
    'split_ratio': [3, 1, 1]  # 3个训练，1个验证，1个测试
}
# config.py
# CONFIG = {
#     # STFT与切片参数
#     'window_size': 2048,
#     'step_size': 1024,          # 50% 重叠
#     'nperseg': 256,             # 频率分辨率
#     'image_size': (64, 64),     # 64x64 对 2060 最友好，且足够看清故障
#
#     # 训练参数
#     'batch_size': 64,           # 64x64图片下，6GB显存可以开到64甚至128
#     'learning_rate': 0.001,
#     'epochs': 30,
#     'patience': 5,              # 早停耐心值
#
#     # 损失函数与调度器
#     'use_focal_loss': False,     # 是否使用 Focal Loss
#     'focal_alpha': 0.25,        # Focal Loss alpha 参数
#     'focal_gamma': 2.0,         # Focal Loss gamma 参数（调大更关注难分样本）
#     'use_scheduler': True,      # 是否使用 ReduceLROnPlateau
#     'scheduler_patience': 5,    # 调度器耐心值
#     'scheduler_factor': 0.5,    # 学习率衰减因子
#
#     # 数据增强（仅训练集）
#     'use_augmentation': False,
#     'augmentation': {
#         'random_horizontal_flip': 0.5,
#         'random_rotation': 10,      # 度
#         'random_affine_translate': 0.05,
#     },
#
#     # 阈值调优
#     'threshold_search': False,       # 是否在验证集上搜索最佳阈值
#     'threshold_steps': 20,          # 搜索步数（在 0.3~0.7 之间）
#
#     # 数据选择策略
#     'files_per_class': 5,           # 每个故障文件夹只取前5个文件
#     'split_ratio': [3, 1, 1]        # 3个训练，1个验证，1个测试
# }