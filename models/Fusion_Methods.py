import torch
import torch.nn as nn
import torch.nn.functional as F

class LGMA(nn.Module):
    def __init__(self, in_channels):
        super(LGMA, self).__init__()
        self.in_channels = in_channels
        self.mu = nn.Parameter(torch.ones(1))   # 可学习参数 μ
        self.lam = nn.Parameter(torch.ones(1))  # 可学习参数 λ
        self.sigmoid = nn.Sigmoid()
        self.fc = nn.Linear(in_channels, in_channels)  # 等价于 1×1 Conv

    def forward(self, x_global, x_local):
        # x_global, x_local: [B, C]

        # 通道注意权重（式 (3)、(4) 的一维版本）
        v1 = self.mu * self.sigmoid(x_global)   # [B, C]
        v2 = self.lam * self.sigmoid(x_local)   # [B, C]

        # 互注意加权
        x_local_att = x_local * v1              # [B, C]
        x_global_att = x_global * v2            # [B, C]

        # 融合并映射
        fused = self.fc(x_local_att + x_global_att)  # [B, C]

        return fused


class CrossAttentionFusion(nn.Module):
    def __init__(self, in_channels, hidden_dim=None):
        super(CrossAttentionFusion, self).__init__()
        if hidden_dim is None:
            hidden_dim = in_channels // 2

        self.query_proj = nn.Linear(in_channels, hidden_dim)
        self.key_proj   = nn.Linear(in_channels, hidden_dim)
        self.value_proj = nn.Linear(in_channels, in_channels)
        self.out_proj   = nn.Linear(in_channels, in_channels)

    def forward(self, global_feat, local_feat):
        # 输入形状：[B, C]
        # 输出：融合后的特征，[B, C]

        # Local 特征作为 Query，Global 特征作为 Key & Value
        Q = self.query_proj(local_feat)    # [B, D]
        K = self.key_proj(global_feat)     # [B, D]
        V = self.value_proj(global_feat)   # [B, C]

        # Cross-Attention 权重计算
        attn_weights = torch.bmm(Q.unsqueeze(1), K.unsqueeze(2))  # [B, 1, 1]
        attn_weights = F.softmax(attn_weights, dim=-1)            # [B, 1, 1]

        # 加权 Value
        attended = attn_weights.squeeze(-1) * V  # [B, C]

        # 投影输出
        fused = self.out_proj(attended + local_feat)  # 残差 + 映射

        return fused




# 测试代码
if __name__ == "__main__":
    import torch

    batch_size = 32
    in_channels = 512
    global_feat = torch.randn(batch_size, in_channels)
    local_feat = torch.randn(batch_size, in_channels)

    model = CrossAttentionFusion(in_channels=in_channels)
    fused_feat = model(global_feat, local_feat)

    print("Fused Feature Shape:", fused_feat.shape)
