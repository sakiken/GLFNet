import torch
import torch.nn as nn

class LocalRelationModule(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LocalRelationModule, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        # 可根据需要添加更多层或关系建模机制

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        return out
class LRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=64, num_regions=3):
        super(LRNet, self).__init__()
        self.num_regions = num_regions
        self.local_modules = nn.ModuleList([
            LocalRelationModule(in_channels, out_channels) for _ in range(num_regions)
        ])
        self.fusion_conv = nn.Conv2d(out_channels * num_regions, out_channels, kernel_size=1)
        # 可根据需要添加更多层

    def forward(self, x):
        # x shape: (batch_size, num_regions * in_channels, H, W)
        batch_size, _, H, W = x.size()
        region_channels = x.size(1) // self.num_regions
        region_features = []

        for i in range(self.num_regions):
            region_input = x[:, i * region_channels:(i + 1) * region_channels, :, :]
            region_out = self.local_modules[i](region_input)
            region_features.append(region_out)

        # 将所有区域的特征在通道维度上拼接
        fused_features = torch.cat(region_features, dim=1)
        # 通过融合卷积层
        out = self.fusion_conv(fused_features)
        return out


class AdaptiveWeightedResidualBottleneck(nn.Module):
    def __init__(self, in_channels, bottleneck_channels, groups=3):
        super(AdaptiveWeightedResidualBottleneck, self).__init__()
        self.groups = groups
        self.group_channels = bottleneck_channels // groups

        # 三组 GAP -> 1x1 Conv -> Grouped 1x1 Conv 分支
        self.branches = nn.ModuleList()
        for _ in range(groups):
            self.branches.append(nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, self.group_channels, kernel_size=1),
                nn.Conv2d(self.group_channels, self.group_channels, kernel_size=1, groups=self.group_channels)
            ))

        # 中间 3x3 卷积
        self.conv3x3 = nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=1)

        # 最后 1x1 卷积 + 残差连接
        self.final_conv = nn.Conv2d(bottleneck_channels, in_channels, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        print(x.shape)
        branches_out = [branch(x) for branch in self.branches]
        concat = torch.cat(branches_out, dim=1)
        out = self.conv3x3(concat)
        out = self.final_conv(out)
        return self.relu(out + x)

class MultiPartAWRB(nn.Module):
    def __init__(self, part_channels=3, bottleneck_channels=24, num_parts=3):
        super(MultiPartAWRB, self).__init__()
        self.num_parts = num_parts
        self.part_channels = part_channels

        self.blocks = nn.ModuleList([
            AdaptiveWeightedResidualBottleneck(
                in_channels=part_channels,
                bottleneck_channels=bottleneck_channels
            ) for _ in range(num_parts)
        ])

    def forward(self, x):
        # x: (B, 9, H, W) → 分为3个 (B, 3, H, W)
        parts = torch.chunk(x, self.num_parts, dim=1)
        outputs = [self.blocks[i](parts[i]) for i in range(self.num_parts)]
        return torch.cat(outputs, dim=1)  # 拼接回 (B, 9, H, W)
