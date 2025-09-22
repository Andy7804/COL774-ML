# modules/vqa_model.py
import torch.nn as nn
from modules.image_encoder import ImageEncoder
from modules.text_encoderBERT import BERTTextEncoder 
from modules.fusion import CrossAttentionFusion
from modules.classifier import Classifier

class VQAModelBERT(nn.Module):
    def __init__(self, vocab_size, num_classes, max_len=32):
        super(VQAModelBERT, self).__init__()
        self.image_encoder = ImageEncoder(output_dim=768)
        self.text_encoder = BERTTextEncoder(vocab_size=30522, max_len=32, embed_dim=768, num_layers=6, num_heads=8) # Modify according to requirement!
        self.fusion = CrossAttentionFusion()
        self.classifier = Classifier(num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask):
        image_feats = self.image_encoder(images)
        text_feats = self.text_encoder(input_ids, attention_mask)
        fused_feats = self.fusion(text_feats.unsqueeze(1), image_feats)
        logits = self.classifier(fused_feats)
        return logits