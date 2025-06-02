import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft
from typing import Optional, Dict



class EMAModule(nn.Module):
    def __init__(self, channels, num_heads=4, gamma=2, b=1):
        super(EMAModule, self).__init__()
        self.num_heads = num_heads
        self.channels_per_head = channels // num_heads
        assert self.channels_per_head * num_heads == channels, "Channels should be divisible by num_heads"

        # Multi-scale feature extraction
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=num_heads)
        self.conv3x3_2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=num_heads)

        # Cross-spatial learning
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Channel-wise attention mechanism (similar to ECA)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_attention = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        # Multi-scale feature extraction
        out = self.conv1x1(x)
        out = self.conv3x3_1(out) + self.conv3x3_2(out)

        # Cross-spatial learning
        out_h = self.pool_h(out)
        out_w = self.pool_w(out).permute(0, 1, 3, 2)

        # Concatenate and apply 1x1 convolution to reduce dimensionality
        hw = torch.cat([out_h, out_w], dim=2)
        hw = self.conv1x1(hw).chunk(2, dim=2)

        out_h, out_w = hw[0], hw[1].permute(0, 1, 3, 2)

        # Apply sigmoid activation for attention weights
        weight_h = out_h.sigmoid()
        weight_w = out_w.sigmoid()

        # Apply attention weights
        out = residual * weight_h * weight_w

        # Channel-wise attention mechanism (similar to ECA)
        y = self.avg_pool(out)
        y = self.conv_attention(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return out * y.expand_as(out) + residual


class FcaBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(FcaBlock, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        # 对每个通道执行2D FFT并取绝对值作为频谱图
        spectrum = fft.fftshift(fft.fft2(x, dim=(-2, -1)), dim=(-2, -1)).abs().mean(dim=(2, 3))
        # 将频谱图展平并送入全连接层
        y = self.fc(spectrum).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SENet(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SENet, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ECA(nn.Module):
    """Constructs an ECA module.

    Args:
        channel: Number of channels of the input feature map
        gamma: Parameter for determining the size of the convolutional kernel. Default: 2
        b: Parameter for determining the size of the convolutional kernel. Default: 1
    """

    def __init__(self, channel, gamma=2, b=1):
        super(ECA, self).__init__()
        t = int(abs((math.log(channel, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Squeeze
        y = self.avg_pool(x)
        # Excitation
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Reweight
        y = self.sigmoid(y)
        return x * y.expand_as(x)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 全连接层实现通道注意力，采用 Conv2d 实现 1x1 卷积
        self.fc1 = nn.Conv2d(in_planes, in_planes // reduction, kernel_size=1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // reduction, in_planes, kernel_size=1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        # 空间注意力的卷积核
        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 沿着通道维度计算平均池化和最大池化
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class CBAM(nn.Module):
    def __init__(self, in_planes, reduction=16, spatial_kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_planes, reduction=reduction)
        self.spatial_attention = SpatialAttention(kernel_size=spatial_kernel_size)

    def forward(self, x):
        # 通道注意力
        x = x * self.channel_attention(x)
        # 空间注意力
        x = x * self.spatial_attention(x)
        return x

class NonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, sub_sample=True, bn_layer=True):
        super(NonLocalBlock, self).__init__()

        self.sub_sample = sub_sample

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv_nd = nn.Conv2d
        max_pool_layer = nn.MaxPool2d(kernel_size=(2, 2)) if sub_sample else None
        bn = nn.BatchNorm2d

        self.g = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            conv_nd(in_channels=self.inter_channels, out_channels=self.in_channels,
                    kernel_size=1, stride=1, padding=0),
            bn(self.in_channels)
        )
        nn.init.constant_(self.W[1].weight, 0)
        nn.init.constant_(self.W[1].bias, 0)

        self.theta = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                             kernel_size=1, stride=1, padding=0)
        self.phi = conv_nd(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)

        if sub_sample:
            self.g = nn.Sequential(self.g, max_pool_layer)
            self.phi = nn.Sequential(self.phi, max_pool_layer)

    def forward(self, x):
        batch_size = x.size(0)

        g_x = self.g(x).view(batch_size, self.inter_channels, -1)
        g_x = g_x.permute(0, 2, 1)

        theta_x = self.theta(x).view(batch_size, self.inter_channels, -1)
        theta_x = theta_x.permute(0, 2, 1)
        phi_x = self.phi(x).view(batch_size, self.inter_channels, -1)
        f = torch.matmul(theta_x, phi_x)

        f_div_C = F.softmax(f, dim=-1)

        y = torch.matmul(f_div_C, g_x)
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, self.inter_channels, *x.size()[2:])
        W_y = self.W(y)
        z = W_y + x

        return z

class EMA(nn.Module):
    def __init__(self, channels, num_heads=4, gamma=2, b=1):
        super(EMA, self).__init__()
        self.num_heads = num_heads
        self.channels_per_head = channels // num_heads
        assert self.channels_per_head * num_heads == channels, "Channels should be divisible by num_heads"

        # Multi-scale feature extraction
        self.conv1x1 = nn.Conv2d(channels, channels, kernel_size=1)
        self.conv3x3_1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=num_heads)
        self.conv3x3_2 = nn.Conv2d(channels, channels, kernel_size=5, padding=2, groups=num_heads)

        # Cross-spatial learning
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # Channel-wise attention mechanism (similar to ECA)
        t = int(abs((math.log(channels, 2) + b) / gamma))
        k_size = t if t % 2 else t + 1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_attention = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        residual = x

        # Multi-scale feature extraction
        out = self.conv1x1(x)
        out = self.conv3x3_1(out) + self.conv3x3_2(out)

        # Cross-spatial learning
        out_h = self.pool_h(out)
        out_w = self.pool_w(out).permute(0, 1, 3, 2)

        # Concatenate and apply 1x1 convolution to reduce dimensionality
        hw = torch.cat([out_h, out_w], dim=2)
        hw = self.conv1x1(hw).chunk(2, dim=2)

        out_h, out_w = hw[0], hw[1].permute(0, 1, 3, 2)

        # Apply sigmoid activation for attention weights
        weight_h = out_h.sigmoid()
        weight_w = out_w.sigmoid()

        # Apply attention weights
        out = residual * weight_h * weight_w

        # Channel-wise attention mechanism (similar to ECA)
        y = self.avg_pool(out)
        y = self.conv_attention(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return out * y.expand_as(out) + residual

class ConvModule(nn.Module):
    """A simple ConvModule using only PyTorch components."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1,
                 norm_cfg: Optional[Dict] = None, act_cfg: Optional[Dict] = None):
        super(ConvModule, self).__init__()
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups, bias=False))

        if norm_cfg and norm_cfg['type'] == 'BN':
            layers.append(nn.BatchNorm2d(out_channels, **norm_cfg.get('kwargs', {})))

        if act_cfg and act_cfg['type'] == 'SiLU':
            layers.append(nn.SiLU())
        elif act_cfg and act_cfg['type'] == 'ReLU':
            layers.append(nn.ReLU())
        # 可以根据需要添加更多的激活函数支持

        self.conv_module = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_module(x)


class CAA(nn.Module):
    """Context Anchor Attention"""

    def __init__(
            self,
            channels: int,
            h_kernel_size: int = 11,
            v_kernel_size: int = 11,
            norm_cfg: Optional[Dict] = dict(type='BN', momentum=0.03, eps=0.001),
            act_cfg: Optional[Dict] = dict(type='SiLU'),
            init_cfg: Optional[Dict] = None,
    ):
        super(CAA, self).__init__()
        self.avg_pool = nn.AvgPool2d(7, 1, 3)
        self.conv1 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.h_conv = ConvModule(channels, channels, (1, h_kernel_size), 1, (0, h_kernel_size // 2), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.v_conv = ConvModule(channels, channels, (v_kernel_size, 1), 1, (v_kernel_size // 2, 0), groups=channels,
                                 norm_cfg=None, act_cfg=None)
        self.conv2 = ConvModule(channels, channels, 1, 1, 0, norm_cfg=norm_cfg, act_cfg=act_cfg)
        self.act = nn.Sigmoid()

    def forward(self, x):
        attn_factor = self.act(self.conv2(self.v_conv(self.h_conv(self.conv1(self.avg_pool(x))))))
        return attn_factor * x  # 注意这里乘以输入x，以实现注意力机制的效果


class ELA(nn.Module):
    def __init__(self, in_channels, phi='T'):
        super(ELA, self).__init__()

        # 不同模式下的默认 kernel_size 和推荐 group 值（只用于建议）
        kernel_size_dict = {'T': 5, 'B': 7, 'S': 5, 'L': 7}
        suggested_groups = {'T': 32, 'B': 16, 'S': 16, 'L': 16}

        # 设置 kernel size
        self.kernel_size = kernel_size_dict.get(phi, 5)
        padding = self.kernel_size // 2

        # 自动适配 group 设置（防止不能整除）
        raw_groups = suggested_groups.get(phi, 16)
        self.groups = self._get_compatible_group(in_channels, raw_groups)

        # 定义模块
        self.conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=self.kernel_size,
                                padding=padding, groups=self.groups, bias=False)

        self.gn = nn.GroupNorm(self.groups, in_channels)
        self.sigmoid = nn.Sigmoid()

    def _get_compatible_group(self, in_channels, suggested):
        # 返回一个能被 in_channels 整除，且接近建议值的 group 数
        for g in reversed(range(1, suggested + 1)):
            if in_channels % g == 0:
                return g
        return 1  # fallback（最小为1）

    def forward(self, x):
        b, c, h, w = x.size()

        # 平均池化 + 转为 (B, C, H or W)
        x_h = torch.mean(x, dim=3, keepdim=True).view(b, c, h)
        x_w = torch.mean(x, dim=2, keepdim=True).view(b, c, w)

        # 一维卷积
        x_h = self.conv1d(x_h)
        x_w = self.conv1d(x_w)

        # GroupNorm + Sigmoid + reshape
        x_h = self.sigmoid(self.gn(x_h)).view(b, c, h, 1)
        x_w = self.sigmoid(self.gn(x_w)).view(b, c, 1, w)

        return x * x_h * x_w

class MultiPartELA(nn.Module):
    def __init__(self, phi='T'):
        super(MultiPartELA, self).__init__()
        # 为每个部位定义一个独立的 ELA 模块
        self.ela_head = ELA(in_channels=3, phi=phi)
        self.ela_left_hand = ELA(in_channels=3, phi=phi)
        self.ela_right_hand = ELA(in_channels=3, phi=phi)

    def forward(self, x):
        # x 的形状为 (batch_size, 9, height, width)
        # 将 x 拆分为三个部分，每个部分对应一个部位
        x_head = x[:, 0:3, :, :]
        x_left_hand = x[:, 3:6, :, :]
        x_right_hand = x[:, 6:9, :, :]

        # 分别应用对应的 ELA 模块
        out_head = self.ela_head(x_head)
        out_left_hand = self.ela_left_hand(x_left_hand)
        out_right_hand = self.ela_right_hand(x_right_hand)

        # 将处理后的三个部分合并
        out = torch.cat([out_head, out_left_hand, out_right_hand], dim=1)
        return out


# 测试代码
if __name__ == "__main__":
    # 创建输入张量，形状为 (batch_size, channels, height, width)
    x = torch.randn(32, 9, 64, 64)  # 示例输入

    # 实例化 MultiPartELA 模块
    model = MultiPartELA(phi='T')  # 选择合适的 phi 值：'T', 'B', 'S', 'L'

    # 前向传播
    out = model(x)

    # 输出结果形状
    print("输出张量的形状:", out.shape)
