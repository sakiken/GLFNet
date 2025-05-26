import torch
import torch.nn as nn
from ConvNext_v2.Conv2model import convnextv2_tiny  # 确保你的模型路径正确


class ConvNeXtV2Model(nn.Module):
    def __init__(self, num_classes=10, freeze_stages=4, freeze_classifier=True):
        super(ConvNeXtV2Model, self).__init__()

        # 加载 ConvNeXt-V2 模型结构
        self.convnext_v2 = convnextv2_tiny()

        # 加载预训练权重
        weight_path = '../weights/convnextv2_tiny_1k_224_ema.pt'
        checkpoint = torch.load(weight_path, map_location='cpu')  # 确保将权重加载到 CPU

        # 注意：这里假设检查点包含一个名为 'model' 的键，根据实际情况调整
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']

        missing_keys, unexpected_keys = self.convnext_v2.load_state_dict(checkpoint, strict=False)
        print(f"Missing keys: {missing_keys}")
        print(f"Unexpected keys: {unexpected_keys}")

        # 修改最后的全连接层，适应我们自己的分类任务
        in_features = self.convnext_v2.head.in_features  # 获取最后一层输入的特征数量
        self.convnext_v2.head = nn.Linear(in_features, num_classes)

        # 冻结指定数量的阶段
        if freeze_stages > 0:
            for stage_idx in range(min(freeze_stages, 4)):  # ConvNeXt-V2-Tiny 有 4 个阶段
                for name, param in self.convnext_v2.named_parameters():
                    if f'downsample_layers.{stage_idx}' in name or f'stages.{stage_idx}' in name:
                        param.requires_grad = False
                        print(f"Frozen: {name}")  # 打印冻结的参数名

        # 冻结分类器层
        if freeze_classifier:
            for name, param in self.convnext_v2.named_parameters():
                if 'head' in name:
                    param.requires_grad = False
                    print(f"Frozen: {name}")  # 打印冻结的参数名

        # 打印可训练参数以验证冻结效果
        print("Parameters to be trained:")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name)

    def forward(self, x):
        return self.convnext_v2(x)


# 测试代码
if __name__ == "__main__":
    # 创建一个 ConvNeXt-V2 模型实例，冻结前两个阶段和分类器
    model = ConvNeXtV2Model(num_classes=10, freeze_stages=2, freeze_classifier=True)

    # 创建一个模拟输入张量（batch_size=32, 3通道, 224x224 图像）
    dummy_input = torch.randn(32, 3, 224, 224)

    # 获取模型输出
    output = model(dummy_input)
    print("Output shape:", output.shape)  # 输出形状应为 torch.Size([32, 10])