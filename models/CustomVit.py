import torch
import torch.nn as nn
from transformers import ViTModel, ViTConfig

class CustomViT(nn.Module):
    def __init__(self, num_classes=10, pretrained=True, model_name="google/vit-base-patch16-224"):
        super().__init__()
        if pretrained:
            self.vit = ViTModel.from_pretrained(model_name)
        else:
            config = ViTConfig.from_pretrained(model_name)
            self.vit = ViTModel(config)

        self.hidden_size = self.vit.config.hidden_size
        self.classifier = nn.Linear(self.hidden_size, num_classes)

    def forward(self, x):
        outputs = self.vit(pixel_values=x, output_hidden_states=True)
        cls_token = outputs.last_hidden_state[:, 0]  # [CLS] token
        logits = self.classifier(cls_token)
        return logits  # 返回 hidden_states 以支持 Grad-CAM