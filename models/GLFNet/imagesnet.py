import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch.fft as fft

class GCEB(nn.Module):
    def __init__(self, dim, kernel_size=3):
        super(GCEB, self).__init__()
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

        self.GCEB1 = GCEB(64)
        self.GCEB2 = GCEB(64)
        self.GCEB3 = GCEB(128)
        self.GCEB4 = GCEB(128)
        self.GCEB5 = GCEB(256)
        self.GCEB6 = GCEB(256)
        self.GCEB7 = GCEB(512)
        self.GCEB8 = GCEB(512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, inp):
        c = F.relu(self.GCEB1(self.bat10(self.cnn1(inp))))
        c = F.relu(self.GCEB2(self.bat11(self.cnn2(c))))
        c = self.pool(c)

        c = F.relu(self.GCEB3(self.bat20(self.cnn3(c))))
        c = F.relu(self.GCEB4(self.bat21(self.cnn4(c))))
        c = self.pool(c)

        c = F.relu(self.GCEB5(self.bat30(self.cnn5(c))))
        c = F.relu(self.GCEB6(self.bat31(self.cnn6(c))))
        c = self.pool(c)

        c = F.relu(self.GCEB7(self.bat40(self.cnn7(c))))
        c = F.relu(self.GCEB8(self.bat41(self.cnn8(c))))

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
