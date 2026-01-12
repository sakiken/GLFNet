import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft

# from src.endmethod.GlobalContextAttention import GlobalContextAttention


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


class CoTAttention(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(CoTAttention, self).__init__()
        self.kernel_size = kernel_size
        self.key_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )
        self.value_embed = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )
        self.global_weight = nn.Parameter(torch.zeros(1))  # 控制全局特征的融合强度

    def forward(self, x):
        bs, c, h, w = x.shape
        # 局部特征捕获
        k1 = self.key_embed(x)  # bs, c, h, w
        v = self.value_embed(x).view(bs, c, -1)  # bs, c, h*w

        # 全局特征引导（轻量级）
        global_context = x.mean(dim=(2, 3), keepdim=True)  # bs, c, 1, 1
        global_context = global_context.expand_as(x)  # bs, c, h, w

        # 融合局部和全局特征
        k2 = F.softmax(v, dim=-1).view(bs, c, h, w) * global_context
        attn = k1 + self.global_weight * k2
        return attn




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

# class ImageConvNet(nn.Module):
#     def __init__(self):
#         super(ImageConvNet, self).__init__()
#         self.pool = nn.MaxPool2d(2, stride=2)
#
#         self.cnn1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
#         self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
#         self.bat10 = nn.BatchNorm2d(64)
#         self.bat11 = nn.BatchNorm2d(64)
#
#         self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
#         self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
#         self.bat20 = nn.BatchNorm2d(128)
#         self.bat21 = nn.BatchNorm2d(128)
#
#         self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
#         self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
#         self.bat30 = nn.BatchNorm2d(256)
#         self.bat31 = nn.BatchNorm2d(256)
#
#         self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
#         self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
#         self.bat40 = nn.BatchNorm2d(512)
#         self.bat41 = nn.BatchNorm2d(512)
#
#         self.GCEB1 = GCEB(64)
#         self.GCEB2 = GCEB(64)
#         self.GCEB3 = GCEB(128)
#         self.GCEB4 = GCEB(128)
#         self.GCEB5 = GCEB(256)
#         self.GCEB6 = GCEB(256)
#         self.GCEB7 = GCEB(512)
#         self.GCEB8 = GCEB(512)
#
#         self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
#
#     def forward(self, inp):
#         c = F.relu(self.GCEB1(self.bat10(self.cnn1(inp))))
#         c = F.relu(self.GCEB2(self.bat11(self.cnn2(c))))
#         c = self.pool(c)
#
#         c = F.relu(self.GCEB3(self.bat20(self.cnn3(c))))
#         c = F.relu(self.GCEB4(self.bat21(self.cnn4(c))))
#         c = self.pool(c)
#
#         c = F.relu(self.GCEB5(self.bat30(self.cnn5(c))))
#         c = F.relu(self.GCEB6(self.bat31(self.cnn6(c))))
#         c = self.pool(c)
#
#         c = F.relu(self.GCEB7(self.bat40(self.cnn7(c))))
#         c = F.relu(self.GCEB8(self.bat41(self.cnn8(c))))
#
#         # 全局平均池化
#         c = self.global_avg_pool(c)
#         c = c.view(c.size(0), -1)  # 将特征图展平
#
#         return c


class GlobalContextAttention1(nn.Module):

    def __init__(self, dim, kernel_size=3):
        super(GlobalContextAttention, self).__init__()
        self.kernel_size = kernel_size

        # 局部特征提取器（Local Feature Extractor）
        self.local_feature_extractor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        # 全局特征提取器（Global Feature Extractor）
        self.global_feature_extractor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        # 全局上下文权重（Global Context Weight），用于控制全局特征的融合强度
        self.global_context_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, c, h, w = x.shape

        # 提取局部特征
        local_features = self.local_feature_extractor(x)  # bs, c, h, w

        # 提取并展平全局特征
        global_features_flattened = self.global_feature_extractor(x).view(bs, c, -1)  # bs, c, h*w
        global_features_softmax = F.softmax(global_features_flattened, dim=-1).view(bs, c, h, w)
        # 计算全局上下文
        global_context = x.mean(dim=(2, 3), keepdim=True)  # bs, c, 1, 1
        global_context_expanded = global_context.expand_as(x)  # bs, c, h, w

        # 融合局部特征与全局特征
        weighted_global_features = global_features_softmax * global_context_expanded
        fused_features = local_features + self.global_context_weight * weighted_global_features

        return fused_features

class GlobalContextAttention(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(GlobalContextAttention, self).__init__()
        self.kernel_size = kernel_size

        # 使用 Depthwise Separable Convolution 替代标准卷积
        self.local_feature_extractor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=kernel_size, padding=kernel_size // 2, groups=dim, bias=False),  # Depthwise
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),  # Pointwise
            nn.BatchNorm2d(dim),
            nn.ReLU()
        )

        self.global_feature_extractor = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=1, bias=False),
            nn.BatchNorm2d(dim)
        )

        self.global_context_weight = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        bs, c, h, w = x.shape

        local_features = self.local_feature_extractor(x)  # bs, c, h, w

        global_features_flattened = self.global_feature_extractor(x).view(bs, c, -1)  # bs, c, h*w
        global_features_softmax = F.softmax(global_features_flattened, dim=-1).view(bs, c, h, w)

        global_context = x.mean(dim=(2, 3), keepdim=True)  # bs, c, 1, 1
        global_context_expanded = global_context.expand_as(x)  # bs, c, h, w

        weighted_global_features = global_features_softmax * global_context_expanded
        fused_features = local_features + self.global_context_weight * weighted_global_features

        return fused_features

class ImageConvNet(nn.Module):
    def __init__(self):
        super(ImageConvNet, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2)

        self.cnn1 = nn.Conv2d(3, 64, 3, stride=2, padding=1)
        self.cnn2 = nn.Conv2d(64, 64, 3, padding=1)
        self.bat10 = nn.BatchNorm2d(64)
        self.bat11 = nn.BatchNorm2d(64)

        self.cnn3 = nn.Conv2d(64, 128, 3, stride=1, padding=1)
        self.cnn4 = nn.Conv2d(128, 128, 3, padding=1)
        self.bat20 = nn.BatchNorm2d(128)
        self.bat21 = nn.BatchNorm2d(128)

        self.cnn5 = nn.Conv2d(128, 256, 3, stride=1, padding=1)
        self.cnn6 = nn.Conv2d(256, 256, 3, padding=1)
        self.bat30 = nn.BatchNorm2d(256)
        self.bat31 = nn.BatchNorm2d(256)

        self.cnn7 = nn.Conv2d(256, 512, 3, stride=1, padding=1)
        self.cnn8 = nn.Conv2d(512, 512, 3, padding=1)
        self.bat40 = nn.BatchNorm2d(512)
        self.bat41 = nn.BatchNorm2d(512)

        self.GCA1 = GlobalContextAttention(64)
        self.GCA2 = GlobalContextAttention(64)
        self.GCA3 = GlobalContextAttention(128)
        self.GCA4 = GlobalContextAttention(128)
        self.GCA5 = GlobalContextAttention(256)
        self.GCA6 = GlobalContextAttention(256)
        self.GCA7 = GlobalContextAttention(512)
        self.GCA8 = GlobalContextAttention(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inp):
        c = F.relu(self.GCA1(self.bat10(self.cnn1(inp))))
        c = F.relu(self.GCA2(self.bat11(self.cnn2(c))))
        c = self.pool(c)

        c = F.relu(self.GCA3(self.bat20(self.cnn3(c))))
        c = F.relu(self.GCA4(self.bat21(self.cnn4(c))))
        c = self.pool(c)

        c = F.relu(self.GCA5(self.bat30(self.cnn5(c))))
        c = F.relu(self.GCA6(self.bat31(self.cnn6(c))))
        c = self.pool(c)

        c = F.relu(self.GCA7(self.bat40(self.cnn7(c))))
        c = F.relu(self.GCA8(self.bat41(self.cnn8(c))))

        # 全局平均池化
        c = self.global_avg_pool(c)
        c = c.view(c.size(0), -1)  # 将特征图展平

        return c


if __name__ == "__main__":
    model = ImageConvNet().cuda()
    print("Model loaded.")
    image = torch.rand(32, 3, 224, 224).cuda()
    print("Image loaded.")

    # Run a feedforward and check shape
    c = model(image)
    print(image.shape)
    print(c.shape)