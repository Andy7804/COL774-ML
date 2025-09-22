# modules/fusion.py
import torch.nn as nn

class CrossAttentionFusion(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8):
        super(CrossAttentionFusion, self).__init__()
        self.cross_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)

    def forward(self, text_features, image_features):
        # text_features: [B, 1, 768] (only [CLS])
        # image_features: [B, num_patches, 768]

        out, _ = self.cross_attention(
            query=text_features,
            key=image_features,
            value=image_features
        )
        return out.squeeze(1)  # [B, 768]
