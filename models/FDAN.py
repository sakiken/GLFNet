# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class GaussianMembershipFunction(nn.Module):
#     def __init__(self, num_classes, in_channels):
#         super(GaussianMembershipFunction, self).__init__()
#         self.num_classes = num_classes
#         self.in_channels = in_channels
#         # Initialize mean and variance for Gaussian MFs with learnable parameters
#         self.mu = nn.Parameter(torch.randn(num_classes, in_channels))
#         self.sigma = nn.Parameter(torch.abs(torch.randn(num_classes, in_channels)))
#
#     def forward(self, x):
#         b, c, h, w = x.size()
#         mu = self.mu.unsqueeze(-1).unsqueeze(-1)
#         sigma = self.sigma.unsqueeze(-1).unsqueeze(-1)
#         # Compute Gaussian MF
#         mf = torch.exp(-((x.unsqueeze(1) - mu) ** 2) / (2 * sigma ** 2))
#         # Apply 'OR' operation across classes to avoid gradient vanishing
#         mf_max, _ = torch.max(mf, dim=1)
#         return mf_max
#
# class SEBlock(nn.Module):
#     def __init__(self, channel, reduction=16):
#         super(SEBlock, self).__init__()
#         self.squeeze = nn.AdaptiveAvgPool2d(1)
#         self.excitation = nn.Sequential(
#             nn.Linear(channel, channel // reduction, bias=False),
#             nn.ReLU(inplace=True),
#             nn.Linear(channel // reduction, channel, bias=False),
#             nn.Sigmoid()
#         )
#
#     def forward(self, x):
#         b, c, _, _ = x.size()
#         y = self.squeeze(x).view(b, c)
#         y = self.excitation(y).view(b, c, 1, 1)
#         return x * y.expand_as(x)
#
# class BottleneckBlockWithSE(nn.Module):
#     expansion = 4
#
#     def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
#         super(BottleneckBlockWithSE, self).__init__()
#         self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_channels)
#         self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_channels)
#         self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
#         self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
#         self.se = SEBlock(out_channels * self.expansion, reduction)
#         self.relu = nn.ReLU(inplace=True)
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         residual = x
#
#         out = self.conv1(x)
#         out = self.bn1(out)
#         out = self.relu(out)
#
#         out = self.conv2(out)
#         out = self.bn2(out)
#         out = self.relu(out)
#
#         out = self.conv3(out)
#         out = self.bn3(out)
#         out = self.se(out)
#
#         if self.downsample is not None:
#             residual = self.downsample(x)
#
#         out += residual
#         out = self.relu(out)
#
#         return out
#
# class FDAN(nn.Module):
#     def __init__(self, num_classes=10):
#         super(FDAN, self).__init__()
#         self.fuzzification = GaussianMembershipFunction(num_classes, in_channels=3)
#         self.feature_extraction = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
#         )
#         # Define a stack of Bottleneck Blocks with SE
#         self.layer1 = self._make_layer(64, 64, blocks=3, stride=1)
#         self.layer2 = self._make_layer(256, 128, blocks=4, stride=2)
#         self.layer3 = self._make_layer(512, 256, blocks=6, stride=2)
#         self.layer4 = self._make_layer(1024, 512, blocks=3, stride=2)
#         self.classifier = nn.Sequential(
#             nn.AdaptiveAvgPool2d((1, 1)),
#             nn.Flatten(),
#             nn.Linear(2048, num_classes),
#             nn.Softmax(dim=1)
#         )
#
#     def _make_layer(self, in_channels, out_channels, blocks, stride):
#         downsample = None
#         if stride != 1 or in_channels != out_channels * BottleneckBlockWithSE.expansion:
#             downsample = nn.Sequential(
#                 nn.Conv2d(in_channels, out_channels * BottleneckBlockWithSE.expansion, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_channels * BottleneckBlockWithSE.expansion),
#             )
#
#         layers = []
#         layers.append(BottleneckBlockWithSE(in_channels, out_channels, stride, downsample))
#         for i in range(1, blocks):
#             layers.append(BottleneckBlockWithSE(out_channels * BottleneckBlockWithSE.expansion, out_channels))
#
#         return nn.Sequential(*layers)
#
#     def forward(self, x):
#         x = self.fuzzification(x)
#         x = self.feature_extraction(x)
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         x = self.classifier(x)
#         return x
#
# # Example of how to use the model
# if __name__ == "__main__":
#     # Assuming we have 10 classes for driver behavior recognition
#     model = FDAN()
#     input_tensor = torch.rand(32, 3, 224, 224)  # Batch size of 32, 3 channels, 224x224 image
#     output = model(input_tensor)
#     print(output.shape)

import torch
import torch.nn as nn
import torch.nn.functional as F

class GaussianMembershipFunction(nn.Module):
    def __init__(self, num_classes, in_channels):
        super(GaussianMembershipFunction, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        # Initialize mean and variance for Gaussian MFs with learnable parameters
        self.mu = nn.Parameter(torch.randn(num_classes, in_channels))
        self.sigma = nn.Parameter(torch.abs(torch.randn(num_classes, in_channels)))

    def forward(self, x):
        b, c, h, w = x.size()
        mu = self.mu.unsqueeze(-1).unsqueeze(-1)
        sigma = self.sigma.unsqueeze(-1).unsqueeze(-1)
        # Compute Gaussian MF
        mf = torch.exp(-((x.unsqueeze(1) - mu) ** 2) / (2 * sigma ** 2))
        # Apply 'OR' operation across classes to avoid gradient vanishing
        mf_max, _ = torch.max(mf, dim=1)
        return mf_max

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResidualBlockWithSE(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, reduction=16):
        super(ResidualBlockWithSE, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SEBlock(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

class FDAN(nn.Module):
    def __init__(self, num_classes=10):
        super(FDAN, self).__init__()
        self.fuzzification = GaussianMembershipFunction(num_classes, in_channels=3)
        self.feature_extraction = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        # Define a stack of Residual Blocks with SE
        self.layer1 = self._make_layer(64, 64, blocks=2, stride=1)
        self.layer2 = self._make_layer(64, 128, blocks=2, stride=2)
        self.layer3 = self._make_layer(128, 256, blocks=2, stride=2)
        self.layer4 = self._make_layer(256, 512, blocks=2, stride=2)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, num_classes),
            # nn.Softmax(dim=1)
        )

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        layers = []
        layers.append(ResidualBlockWithSE(in_channels, out_channels, stride, downsample))
        for i in range(1, blocks):
            layers.append(ResidualBlockWithSE(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.fuzzification(x)
        x = self.feature_extraction(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.classifier(x)
        return x

# Example of how to use the model
if __name__ == "__main__":

    model = FDAN(num_classes=10)
    input_tensor = torch.rand(32, 3, 224, 224)  # Batch size of 32, 3 channels, 224x224 image
    output = model(input_tensor)
    print(output.shape)  # Should print: torch.Size([32, 10])