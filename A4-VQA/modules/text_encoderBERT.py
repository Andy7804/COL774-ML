# modules/text_encoder.py
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

class BERTTextEncoder(nn.Module):
    def __init__(self, vocab_size, max_len=32, embed_dim=768, num_layers=6, num_heads=8):
        super(BERTTextEncoder, self).__init__()
        
        # Initialize with BERT embeddings
        bert_model = BertModel.from_pretrained("bert-base-uncased")
        
        # Use BERT's word embeddings
        self.token_embeddings = nn.Embedding.from_pretrained(
            bert_model.embeddings.word_embeddings.weight,
            freeze=False  # Allow fine-tuning
        )
        
        # Rest of your existing initialization
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.position_embeddings = nn.Parameter(torch.randn(1, max_len+1, embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=2048, 
            activation='relu'
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        # Free up memory by removing the full BERT model after initialization
        del bert_model
        torch.cuda.empty_cache()

    def forward(self, input_ids, attention_mask):
        embeddings = self.token_embeddings(input_ids)  # [B, L, 768]
        B, L, _ = embeddings.shape

        cls_tokens = self.cls_token.expand(B, -1, -1)  # [B, 1, 768]
        embeddings = torch.cat((cls_tokens, embeddings), dim=1)  # [B, L+1, 768]

        embeddings = embeddings + self.position_embeddings[:, :L+1, :]

        # Transformer expects (seq_len, batch, embed_dim)
        embeddings = embeddings.permute(1, 0, 2)

        # Prepare attention mask
        extended_attention_mask = torch.cat(
            (torch.ones((attention_mask.size(0), 1), device=attention_mask.device), attention_mask),
            dim=1
        )
        key_padding_mask = (extended_attention_mask == 0)  # (batch_size, seq_len)

        out = self.transformer_encoder(embeddings, src_key_padding_mask=key_padding_mask)

        # [CLS] token output
        return out[0]  # [batch_size, embed_dim]