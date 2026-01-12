import torch
import torch.nn as nn
import torch.nn.functional as F


class MPCI(nn.Module):
    def __init__(self, in_channels, reduction=3):
        super(MPCI, self).__init__()
        total_channels = in_channels * 3

        # 确保 total_channels 大于 0
        if total_channels <= 0:
            raise ValueError(
                f"Total channels should be greater than 0, but got {total_channels}. Please check in_channels and num_parts.")

        # 卷积模块和通道注意力机制，结合部位间的联系
        self.conv1 = nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm2d(total_channels)
        self.relu = nn.ReLU(inplace=True)

        # 通道注意力机制（类似SE模块）
        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(total_channels, total_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(total_channels // reduction, total_channels, kernel_size=1, bias=False),
            nn.Sigmoid()
        )

        # 进一步增强特征
        self.conv2 = nn.Conv2d(total_channels, total_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(total_channels)
        self.last_attention_weights = None  # 

    def forward(self, local_features):
        # 打印 local_features 的形状

        # 第一层卷积处理 + BatchNorm + ReLU
        x = self.relu(self.bn1(self.conv1(local_features)))

        # 通道注意力机制
        attention_weights = self.attention(x)
        self.last_attention_weights = attention_weights.detach().cpu()  
        x = x * attention_weights

        # 进一步卷积处理
        x = self.relu(self.bn2(self.conv2(x)))

        return x


# 测试
if __name__ == "__main__":

    model = MPCI(in_channels=3)

    # 创建一个随机输入张量，形状为 (batch_size, 9, 64, 64)
    input_tensor = torch.randn(32, 9, 64, 64)

    # 进行前向传播
    output = model(input_tensor)

    # 打印输出的形状
    print(f"MultiPartDynamicFeatureEnhancer Output shape: {output.shape}")
