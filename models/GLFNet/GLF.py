import torch
import torch.nn as nn

from Dgli import GLFB
from dwrb import *
from imagesnet import ImageConvNet


class DriverNet(nn.Module):
    def __init__(self, num_classes=10, global_dim=512):
        super(DriverNet, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.global_dim = global_dim
        self.ImageConvNet = ImageConvNet().cuda()

        # 动态 AWRb 模块
        self.MPFD = MPFD(3, 3)
        self.pooling_layer = nn.AdaptiveAvgPool2d(1)
        self.linear_layer = nn.Linear(9, 512)

        # self.MultiPartDynamicFeatureEnhancer = MPFAE(in_channels=3, num_parts=3, output_dim=512, reduction=3)

        #对比其他融合模块
        # self.LGMA = LGMA(global_dim)

        # GLIBlock 多模态特征融合
        self.GLFB = GLFB(global_dim, ratio=16, dropout_rate=0.1)

        # 第一层全连接层 (global_dim * height * width -> 256)
        self.fc1 = nn.Linear(global_dim, 256)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.5)

        # 第二层全连接层 (256 -> 128)
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.5)

        # # 第三层全连接层 (128 -> num_classes)
        self.fc3 = nn.Linear(128, num_classes)

        self.last_global_weight = None  # ⭐️ 保存权重
        self.last_local_weight = None

        # Grad-CAM hook 临时缓存
        self.gradcam_features = {}
        self.gradcam_gradients = {}

        # 注册 Grad-CAM hooks（ImageConvNet 的 cnn8 层）
        self.ImageConvNet.cnn8.register_forward_hook(self.save_features_hook)
        self.ImageConvNet.cnn8.register_full_backward_hook(self.save_gradients_hook)

    def save_features_hook(self, module, input, output):
        self.gradcam_features['value'] = output.detach()

    def save_gradients_hook(self, module, grad_input, grad_output):
        self.gradcam_gradients['value'] = grad_output[0].detach()

    # def forward(self, global_img, local_features):
    def forward(self, global_img, local_features):
        global_feat = self.ImageConvNet(global_img)
        # print(f'global_feat shape ', global_feat.shape)
        local_features = local_features.squeeze(1)
        # print(f'local_features shape ', local_features.shape)
        # 使用 MultiPartDynamicFeatureEnhancer 模块增强局部特征
        enhanced_local_features = self.MPFD(local_features)

        pooled_features = self.pooling_layer(enhanced_local_features).view(enhanced_local_features.size(0), -1)
        enhanced_local_features = self.linear_layer(pooled_features)
        # print(f'enhanced_local_features ', enhanced_local_features.shape)
        fused_feat, global_weight, local_weight = self.GLFB(global_feat, enhanced_local_features)
        # fused_feat = self.LGMA(global_feat, enhanced_local_features)

        self.last_global_weight = global_weight.cpu()
        self.last_local_weight = local_weight.cpu()

        # 第一层全连接层
        x = self.fc1(fused_feat)
        x = self.relu1(x)
        x = self.dropout1(x)
        # print(f'x1 ', x.shape)
        # 第二层全连接层
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.dropout2(x)
        # print(f'x2 ', x.shape)
        # 第三层全连接层
        x = self.fc3(x)
        # print(f'x3 ', x.shape)
        return x

