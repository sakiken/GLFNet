import math
import torch
import torch.nn as nn

class GLFB(nn.Module):
    """

    Global-LocalInteractionBlock(GLIB).
    The Global-Local Interaction Block (GLIBlock) aims to achieve efficient fusion of global and local features.
    The module first enhances the global and local features individually using a residual fully connected layer. Then, it employs a weight modulation mechanism,
    implemented with learnable parameters, to dynamically compute the weighting coefficients for the global and local features. During the weighted fusion process,.
    the module retains the critical information from both global and local features while incorporating residual connections to strengthen feature representation.
     By leveraging local features for fine-grained details and global features for contextual information, this module effectively improves the overall performance of the model.

    """


    def __init__(self, in_channels, ratio=16, dropout_rate=0.1):
        super(GLFB, self).__init__()

        # 残差全连接层
        self.residual_fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // ratio, bias=False),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(in_channels // ratio, in_channels, bias=False),
        )

        # 参数化权重
        self.fusion_weights = nn.Parameter(torch.zeros(in_channels * 2, 2))
        # print(f'fusion_weights',self.fusion_weights.shape)
        nn.init.xavier_uniform_(self.fusion_weights)

        # 归一化和激活函数
        self.feature_norm = nn.LayerNorm(in_channels * 2)
        self.sigmoid = nn.Sigmoid()
        self.feature_dropout = nn.Dropout(dropout_rate)

    def _modulate_weights(self, concatenated_feat):
        # 参数化权重融合
        # print(f'concatenated_feat',concatenated_feat.shape)
        z = concatenated_feat * self.fusion_weights[None, :, :]
        # print(f'z',z.shape)
        z = torch.sum(z, dim=2)
        # print(f'z',z.shape)
        z_hat = self.feature_norm(z)
        z_hat = self.sigmoid(z_hat)
        return z_hat.unsqueeze(-1).unsqueeze(-1)

    def forward(self, global_feat, local_feat):
        b, c_global = global_feat.shape
        b, c_local = local_feat.shape

        if c_global != c_local:
            raise ValueError("Global and local features must have the same number of channels")

        # 残差全连接处理
        residual_global_feat = global_feat + self.residual_fc(global_feat)
        # print(f'residual_global_feat',residual_global_feat.shape)
        residual_local_feat = local_feat + self.residual_fc(local_feat)
        # print(f'residual_local_feat',residual_local_feat.shape)

        # 特征拼接与权重融合
        concatenated_feat = torch.cat(
            (residual_global_feat.unsqueeze(2), residual_local_feat.unsqueeze(2)), dim=1
        )
        modulated_weights = self._modulate_weights(concatenated_feat)
        # 加权融合（加入残差连接）
        global_weight = modulated_weights[:, :c_global, 0, 0]
        local_weight = modulated_weights[:, c_global:c_global + c_local, 0, 0]

        fused_features = (
            global_feat * global_weight + local_feat * local_weight
        ) + (global_feat + local_feat)
        fused_features = self.feature_dropout(fused_features)

        return fused_features, global_weight.detach(), local_weight.detach()  # ⭐️ 返回权重


# 测试代码
if __name__ == "__main__":

    # 创建随机输入数据
    batch_size = 32
    in_channels = 512
    global_feat = torch.randn(batch_size, in_channels)
    local_feat = torch.randn(batch_size, in_channels)

    # 创建 GlobalLocalInteractionBlock 实例
    model = GLFB(in_channels=in_channels)

    # 前向传播
    fused_feat = model(global_feat, local_feat)

    # 打印输出形状
    print("Fused Feature Shape:", fused_feat.shape)