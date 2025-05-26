import torch
import torch.nn as nn
import kornia as K
from skimage.feature import hog

class FastHOGProcessor(nn.Module):
    """GPU加速的批量HOG处理（兼容新版Kornia）"""

    def __init__(self, orientations=8, pixels_per_cell=(2, 2)):
        super().__init__()
        # 灰度转换
        self.grayscale = K.color.RgbToGrayscale()

        # 创建HOG配置
        self.config = {
            'orientations': orientations,
            'pixels_per_cell': pixels_per_cell,
            'cells_per_block': (1, 1),
        }

        # 使用spatial_gradient替代Sobel
        self.gaussian = K.filters.GaussianBlur2d((5, 5), (1.5, 1.5))

    def _hog_visualization(self, grad_mag, grad_ori):
        # 简化版HOG可视化
        return torch.clamp(grad_mag * grad_ori / (grad_ori.max() + 1e-6), 0, 1)

    def forward(self, x):
        # 输入: [B, 3, H, W]
        # 1. 转灰度
        gray = self.grayscale(x)  # [B, 1, H, W]

        # 2. 高斯平滑
        blurred = self.gaussian(gray)

        # 3. 使用spatial_gradient计算梯度
        gradients = K.filters.spatial_gradient(blurred)  # 输出形状 [B, 2, H, W]
        grad_x = gradients[:, :, 0]  # 提取x方向梯度 [B, H, W]
        grad_y = gradients[:, :, 1]  # 提取y方向梯度 [B, H, W]

        # 4. 计算梯度幅值和方向
        grad_mag = torch.sqrt(grad_x ** 2 + grad_y  ** 2 + 1e-6)
        grad_ori = torch.atan2(grad_y, grad_x) * (180 / torch.pi)  # 转角度制

        # 5. 简化版HOG特征可视化
        hog_vis = self._hog_visualization(grad_mag, grad_ori)

        return hog_vis.unsqueeze(1)  # [B, 1, H, W]


class D_HCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.hog_processor = FastHOGProcessor()

        # 主网络保持不变
        self.net = nn.Sequential(
            nn.Conv2d(1, 32, 12, padding=6),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 6, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),

            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        with torch.no_grad():
            hog_features = self.hog_processor(x)
        hog_features = hog_features.squeeze(1)
        return self.net(hog_features)


# 速度测试对比 ---------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = 32


    # 新版模型测试
    optimized_model = D_HCNN().to(device)
    optimized_input = torch.rand(batch_size, 3, 224, 224).to(device)

    output = optimized_model(optimized_input)
    print('output.shape',output.shape)
