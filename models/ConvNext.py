import torch
import torch.nn as nn
from torchvision import models
from torchvision.models.convnext import ConvNeXt_Tiny_Weights

class ConvNextModel(nn.Module):
    def __init__(self, num_classes=10, freeze_stages=4, freeze_classifier=False):
        super(ConvNextModel, self).__init__()

        # 加载预训练的 ConvNeXt-T 模型，并指定使用 ImageNet1K V1 权重
        weights = ConvNeXt_Tiny_Weights.IMAGENET1K_V1
        self.convnext = models.convnext_tiny(weights=weights)

        # 修改最后的全连接层，适应我们自己的分类任务
        in_features = self.convnext.classifier[2].in_features  # 获取最后一层输入的特征数量
        self.convnext.classifier[2] = nn.Linear(in_features, num_classes)

        # 冻结指定数量的阶段
        if freeze_stages > 0:
            for stage_idx in range(min(freeze_stages, 4)):  # ConvNeXt-Tiny 有 4 个阶段
                for param in self.convnext.features[stage_idx].parameters():
                    param.requires_grad = False

        # 冻结分类器层
        if freeze_classifier:
            for param in self.convnext.classifier.parameters():
                param.requires_grad = False

        # # 打印哪些参数将被更新
        # print("Freezing parameters:")
        # for name, param in self.named_parameters():
        #     if param.requires_grad:
        #         print(f"Will update: {name}")
        #     else:
        #         print(f"Frozen: {name}")

    def forward(self, x):
        return self.convnext(x)

# 测试代码
if __name__ == "__main__":
    # 创建一个 ConvNeXt 模型实例，冻结前两个阶段和分类器
    model = ConvNextModel(num_classes=10, freeze_stages=4, freeze_classifier=True)

    # 创建一个模拟输入张量（batch_size=32, 3通道, 224x224 图像）
    dummy_input = torch.randn(32, 3, 224, 224)

    # 获取模型输出
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 输出形状应为 torch.Size([32, 10])