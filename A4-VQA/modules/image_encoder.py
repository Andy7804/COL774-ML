# modules/image_encoder.py
import torch.nn as nn
import torchvision.models as models

class ImageEncoder(nn.Module):
    def __init__(self, output_dim=768):
        super(ImageEncoder, self).__init__()
        
        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)
        modules = list(resnet.children())[:-2]  # Remove avgpool and fc
        self.resnet_backbone = nn.Sequential(*modules)
        
        for param in self.resnet_backbone.parameters():
            param.requires_grad = False  # Initially freeze ResNet
        
        self.linear_proj = nn.Linear(2048, output_dim)  # Project to 768

    def forward(self, images):
        features = self.resnet_backbone(images)  # [B, 2048, H, W]
        B, C, H, W = features.shape
        features = features.view(B, C, -1)        # [B, 2048, H*W]
        features = features.permute(0, 2, 1)       # [B, H*W, 2048]
        features = self.linear_proj(features)      # [B, H*W, 768]
        return features  # (sequence of image patches)
