import torch
import torch.nn as nn
import torch.nn.functional as F

#1 Conv1x1
class Conv1x1(nn.Module):
    def __init__(self, in_channels=9, mid_channels=32, out_features=512):
        super(Conv1x1, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mid_channels, out_features)

    def forward(self, x):
        x = F.relu(self.conv1x1(x))
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#2 ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels=9, mid_channels=32, out_features=512):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.shortcut = nn.Conv2d(in_channels, mid_channels, kernel_size=1)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mid_channels, out_features)

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = x + shortcut
        x = F.relu(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

#3 GAP
class GAP(nn.Module):
    def __init__(self, in_channels=9, out_features=512):
        super(GAP, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_channels, out_features)

    def forward(self, x):
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# 4. 空间金字塔池化 (Spatial Pyramid Pooling, SPP)
class SPP(nn.Module):
    def __init__(self, in_channels=9, mid_channels=32, out_features=512):
        super(SPP, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.fc = nn.Linear(mid_channels * (1*1 + 2*2 + 4*4), out_features)

    def forward(self, x):
        x = F.relu(self.conv(x))
        spp = [F.adaptive_avg_pool2d(x, s).view(x.size(0), -1) for s in (1, 2, 4)]
        x = torch.cat(spp, dim=1)
        x = self.fc(x)
        return x

# 5. 循环神经网络 (RNN)
class RNN(nn.Module):
    def __init__(self, input_size=9 * 64, hidden_size=128, out_features=512):
        super(RNN, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, hidden_size=hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, out_features)

    def forward(self, x):
        x = x.view(x.size(0), 64, -1)
        output, _ = self.rnn(x)
        x = output[:, -1, :]
        x = self.fc(x)
        return x

# 6. trainsformer LocalSelfAttention
class LocalSelfAttention(nn.Module):
    def __init__(self, in_channels=9, mid_channels=32, out_features=512):
        super(LocalSelfAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.qkv_conv = nn.Conv2d(mid_channels, 3 * mid_channels, kernel_size=1)
        self.out_conv = nn.Conv2d(mid_channels, mid_channels, kernel_size=1)
        self.scale = mid_channels ** -0.5
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(mid_channels, out_features)

    def forward(self, x):
        # Extract features
        x = F.relu(self.conv(x))  # (batch_size, mid_channels, 64, 64)

        # Generate Q, K, V from the input feature map
        qkv = self.qkv_conv(x).chunk(3, dim=1)  # Split into Q, K, V
        q, k, v = map(lambda t: t.flatten(2), qkv)  # Flatten spatial dimensions

        # Compute attention scores
        dots = torch.einsum('bci,bcj->bij', q, k) * self.scale
        attn = torch.softmax(dots, dim=-1)

        # Apply attention weights to values
        out = torch.einsum('bij,bcj->bci', attn, v)
        out = out.view_as(qkv[0])  # Reshape back to original spatial dimensions

        # Final convolution to integrate attended features
        out = self.out_conv(out)

        # Flatten and pass through fully connected layer
        out = self.gap(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


#MPFD
class MPFD(nn.Module):
    def __init__(self, in_channels, reduction=3):
        super(MPFD, self).__init__()
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
        self.last_attention_weights = None  # ⭐️ 新增行

    def forward(self, local_features):
        # 打印 local_features 的形状

        # 第一层卷积处理 + BatchNorm + ReLU
        x = self.relu(self.bn1(self.conv1(local_features)))

        # 通道注意力机制
        attention_weights = self.attention(x)
        self.last_attention_weights = attention_weights.detach().cpu()  # ⭐️ 保存注意力权重
        x = x * attention_weights

        # 进一步卷积处理
        x = self.relu(self.bn2(self.conv2(x)))

        return x


# 测试
if __name__ == "__main__":
    # 初始化 MultiPartDynamicFeatureEnhancer 模块
    model = MultiPartDynamicFeatureEnhancer(in_channels=3, num_parts=3, output_dim=512, reduction=3, num_blocks=3)

    # 创建一个随机输入张量，形状为 (batch_size, 9, 64, 64)
    input_tensor = torch.randn(32, 9, 64, 64)

    # 进行前向传播
    output = model(input_tensor)

    # 打印输出的形状
    print(f"MultiPartDynamicFeatureEnhancer Output shape: {output.shape}")

    modules = [
        Conv1x1(),
        ResidualBlock(),
        GAP(),
        SPP(),
        RNN(),
        LocalSelfAttention()
    ]

    # 随机输入数据
    input_tensor = torch.randn(32, 9, 64, 64)

    # 测试每个模块
    for module in modules:
        output = module(input_tensor)
        print(f"{module}Output shape: {output.shape}\n")