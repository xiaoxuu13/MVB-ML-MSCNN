import torch.nn as nn
import torch
import torch.nn.functional as F

# --- 新增：CBAM 注意力模块 ---
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1   = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes)
        self.sa = SpatialAttention()
    def forward(self, x):
        x = self.ca(x) * x
        x = self.sa(x) * x
        return x
# class ML_MSCNN(nn.Module):
#     def __init__(self, num_classes):
#         super(ML_MSCNN, self).__init__()
#
#         # --- 多尺度特征提取 (Multi-Scale) ---
#         # 分支1：小卷积核 (捕捉细微毛刺)
#         self.branch1 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=3, padding=1),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         # 分支2：中卷积核
#         self.branch2 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=5, padding=2),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         # 分支3：大卷积核 (捕捉整体趋势)
#         self.branch3 = nn.Sequential(
#             nn.Conv2d(1, 16, kernel_size=7, padding=3),
#             nn.BatchNorm2d(16),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#
#         # --- 特征融合 (Fusion) ---
#         # 拼接后的通道数 = 16 + 16 + 16 = 48
#         self.fusion_layer = nn.Sequential(
#             nn.Conv2d(48, 64, kernel_size=3, padding=1),
#             nn.BatchNorm2d(64),
#             nn.ReLU(),
#             nn.MaxPool2d(2)
#         )
#         # --- 新增：在这里插入注意力 ---
#         self.attention = CBAM(64)
#
#         # --- 全连接与分类 ---
#         # 64x64 输入 -> 经过两次 MaxPool(2) -> 尺寸变为 16x16
#         self.flatten_dim = 64 * 16 * 16
#
#         self.fc = nn.Sequential(
#             nn.Linear(self.flatten_dim, 256),
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#             # 注意：这里不加 Sigmoid，因为 Loss 函数里会加
#         )
#     class ML_MSCNN(nn.Module):
#         # 【新增】支持传入参数控制模块开关
#         def __init__(self, num_classes=8, use_multiscale=True, use_cbam=True):
#             super(ML_MSCNN, self).__init__()
#             self.use_multiscale = use_multiscale
#             self.use_cbam = use_cbam
#
#             # 基础分支 (3x3)
#             self.branch1 = nn.Sequential(
#                 nn.Conv2d(1, 16, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(16),
#                 nn.ReLU(),
#                 nn.MaxPool2d(2)
#             )
#
#             # 仅在启用多尺度时实例化其他分支
#             if self.use_multiscale:
#                 self.branch2 = nn.Sequential(
#                     nn.Conv2d(1, 16, kernel_size=5, padding=2),
#                     nn.BatchNorm2d(16),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2)
#                 )
#                 self.branch3 = nn.Sequential(
#                     nn.Conv2d(1, 16, kernel_size=7, padding=3),
#                     nn.BatchNorm2d(16),
#                     nn.ReLU(),
#                     nn.MaxPool2d(2)
#                 )
#                 fusion_in_channels = 48
#             else:
#                 fusion_in_channels = 16  # 单分支时融合层输入通道数为16
#
#             self.fusion_layer = nn.Sequential(
#                 nn.Conv2d(fusion_in_channels, 64, kernel_size=3, padding=1),
#                 nn.BatchNorm2d(64),
#                 nn.ReLU()
#             )
#
#             if self.use_cbam:
#                 self.cbam = CBAM(64)
#
#             self.flatten_dim = 64 * 16 * 16
#
#             self.fc = nn.Sequential(
#                 nn.Linear(self.flatten_dim, 256),
#                 nn.ReLU(),
#                 nn.Dropout(0.5),
#                 nn.Linear(256, num_classes)
#             )
#
#         def forward(self, x):
#             x1 = self.branch1(x)
#
#             # 动态拼接特征
#             if self.use_multiscale:
#                 x2 = self.branch2(x)
#                 x3 = self.branch3(x)
#                 x_concat = torch.cat([x1, x2, x3], dim=1)
#             else:
#                 x_concat = x1
#
#             x_fused = self.fusion_layer(x_concat)
#
#             # 动态应用注意力
#             if self.use_cbam:
#                 x_fused = self.cbam(x_fused)
#
#             x_fused = F.max_pool2d(x_fused, 2)
#             x_flat = x_fused.view(x_fused.size(0), -1)
#             output = self.fc(x_flat)
#             return output
#     def forward(self, x):
#         # 并行通过三个分支
#         x1 = self.branch1(x)
#         x2 = self.branch2(x)
#         x3 = self.branch3(x)
#
#         # 在通道维度拼接 (Dim 1)
#         x_concat = torch.cat([x1, x2, x3], dim=1)
#
#         # 融合与降维
#         x_fused = self.fusion_layer(x_concat)
#         # --- 应用注意力 ---
#         x_fused = self.attention(x_fused)
#
#         # 展平
#         x_flat = x_fused.view(x_fused.size(0), -1)
#
#         # 输出 Logits
#         output = self.fc(x_flat)
#         return output
class ML_MSCNN(nn.Module):
    # 【新增】支持传入参数控制模块开关
    def __init__(self, num_classes=8, use_multiscale=True, use_cbam=True):
        super(ML_MSCNN, self).__init__()
        self.use_multiscale = use_multiscale
        self.use_cbam = use_cbam

        # 基础分支 (3x3)
        self.branch1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        # 仅在启用多尺度时实例化其他分支
        if self.use_multiscale:
            self.branch2 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=5, padding=2),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            self.branch3 = nn.Sequential(
                nn.Conv2d(1, 16, kernel_size=7, padding=3),
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )
            fusion_in_channels = 48
        else:
            fusion_in_channels = 16  # 单分支时融合层输入通道数为16

        self.fusion_layer = nn.Sequential(
            nn.Conv2d(fusion_in_channels, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )

        if self.use_cbam:
            self.cbam = CBAM(64)

        self.flatten_dim = 64 * 16 * 16

        self.fc = nn.Sequential(
            nn.Linear(self.flatten_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x,return_features=False):
        x1 = self.branch1(x)

        # 动态拼接特征
        if self.use_multiscale:
            x2 = self.branch2(x)
            x3 = self.branch3(x)
            x_concat = torch.cat([x1, x2, x3], dim=1)
        else:
            x_concat = x1

        x_fused = self.fusion_layer(x_concat)

        # 动态应用注意力
        if self.use_cbam:
            x_fused = self.cbam(x_fused)

        x_fused = F.max_pool2d(x_fused, 2)
        x_flat = x_fused.view(x_fused.size(0), -1)
        output = self.fc(x_flat)
        if return_features:
            return output, x_flat  # 测试时返回特征用于 t-SNE

        return output