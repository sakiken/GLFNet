# l_tla_singleframe.py
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # torchvision optional: use MobileNetV2 as lightweight backbone if available
    from torchvision.models import mobilenet_v2
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

# -------------------------
# Channel Attention (paper-style)
# -------------------------
class ChannelAttention(nn.Module):
    """
    Channel attention like SE/CBAM variant used in many lightweight attention designs.
    Use both AvgPool and MaxPool branches followed by a small MLP (shared) and sigmoid gating.
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(8, channels // reduction)
        # shared MLP implemented with conv1x1 layers for spatial invariance
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg = F.adaptive_avg_pool2d(x, 1)  # (B,C,1,1)
        max_ = F.adaptive_max_pool2d(x, 1)  # (B,C,1,1)
        out = self.mlp(avg) + self.mlp(max_)
        scale = self.sigmoid(out)
        return x * scale


# -------------------------
# Spatial Attention (paper-style)
# -------------------------
class SpatialAttention(nn.Module):
    """
    Spatial attention via compressing channel dimension (avg + max) then conv -> sigmoid.
    Kernel size 7 used in the CBAM paper; paper may use similar large receptive kernel.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: (B, C, H, W)
        avg = torch.mean(x, dim=1, keepdim=True)   # (B,1,H,W)
        max_, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        cat = torch.cat([avg, max_], dim=1)  # (B,2,H,W)
        attn = self.sigmoid(self.conv(cat))  # (B,1,H,W)
        return x * attn


# -------------------------
# Lightweight backbone options
# -------------------------
class SmallConvBackbone(nn.Module):
    """
    A small lightweight conv backbone inspired by lightweight networks.
    Produces feature map ~ (B, feat_dim, 7, 7) for input 224x224.
    This is provided so the model can run without torchvision.
    """
    def __init__(self, out_channels=1280):
        super().__init__()
        # design: conv stem + 4 blocks with downsampling to reach 7x7
        # 224 -> 112 -> 56 -> 28 -> 14 -> 7  (we do strides accordingly)
        layers = []
        c = 3
        cfg = [
            (32, 3, 2),  # /2 -> 112
            (64, 3, 2),  # /2 -> 56
            (128, 3, 2), # /2 -> 28
            (256, 3, 2), # /2 -> 14
            (out_channels, 3, 2)  # /2 -> 7
        ]
        for out_ch, k, s in cfg:
            layers += [
                nn.Conv2d(c, out_ch, kernel_size=k, stride=s, padding=k//2, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            ]
            c = out_ch
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# -------------------------
# L-TLA single-frame network
# -------------------------
class L_TLA(nn.Module):
    """
    Single-frame L-TLA: backbone -> Channel Attention -> Spatial Attention -> GAP -> FC
    - backbone_type: "mobile" uses torchvision's MobileNetV2.features (if available)
                     "small" uses a custom small conv backbone (faster to run)
    - num_classes: number of output classes
    """
    def __init__(self, num_classes=10, backbone_type='mobile', feat_dim=512, use_pretrained=False):
        super().__init__()

        self.backbone_type = backbone_type.lower()
        # if self.backbone_type == 'mobile' and _HAS_TORCHVISION:
        if self.backbone_type == 'mobile':
            # MobileNetV2 features produces feature maps with 1280 channels at the end
            backbone = mobilenet_v2(weights=None).features  # weights=None to avoid auto-download
            # Optionally use pretrained weights if available in your env; left as argument for user
            self.backbone = backbone
            self.feat_dim = 1280
        else:
            # fallback small backbone so code runs without torchvision
            self.backbone = SmallConvBackbone(out_channels=feat_dim)
            self.feat_dim = feat_dim

        # attention modules
        self.channel_attn = ChannelAttention(self.feat_dim, reduction=16)
        self.spatial_attn = SpatialAttention(kernel_size=7)

        # final classifier head: global average pool + dropout + fc
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=0.35)
        self.fc = nn.Linear(self.feat_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        # initialize conv and fc layers (standard)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if getattr(m, 'bias', None) is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        x: (B,3,224,224)
        returns: logits (B, num_classes)
        """
        B = x.size(0)
        feat = self.backbone(x)  # expect (B, C, Hf, Wf) e.g. (B,1280,7,7) for MobileNetV2
        # Channel attention
        feat = self.channel_attn(feat)
        # Spatial attention
        feat = self.spatial_attn(feat)
        # Classification head
        pooled = self.global_pool(feat).view(B, -1)  # (B, C)
        out = self.dropout(pooled)
        logits = self.fc(out)
        return logits


# -------------------------
# Quick self-test
# -------------------------
if __name__ == "__main__":
    # emulate your input shape (32,3,224,224)
    B = 32
    x = torch.randn(B, 3, 224, 224)

    # Option A: MobileNetV2 backbone (if torchvision available)
    if _HAS_TORCHVISION:
        print("Testing with MobileNetV2 backbone...")
        model = L_TLA(num_classes=10, backbone_type='mobile')
    else:
        print("Testing with small custom backbone...")
        model = L_TLA(num_classes=10, backbone_type='small', feat_dim=512)

    # for speed on CPU tests, switch model to eval and use no grad
    model.eval()
    with torch.no_grad():
        out = model(x)
    print("Input shape:", x.shape)
    print("Output shape:", out.shape)  # expect (32, num_classes)
