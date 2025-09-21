import torch
import torch.nn as nn

class DummyTransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()

    def forward(self, x):
        return x

## Layer Normalizer
class DummyLayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-5):
        super().__init__()

    def forward(self, x):
        return x

class DummyGPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embed_dimension"])
        self.pos_embedding = nn.Embedding(config["context_length"], config["embed_dimension"])
        self.dropout_embed = nn.Dropout(config["dropout_rate"])
        self.transformer_blocks = nn.Sequential(
            # Placeholder transformer block.
            *[DummyTransformerBlock(config)
              for _ in range(config["layer_count"])]
        )

        # Placeholder Layer Norm
        self.final_norm = DummyLayerNorm(config["embed_dimension"])
        self.out_head = nn.Linear(
            config["embed_dimension"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, sequence_length = in_idx.shape
        token_embeds = self.token_embedding(in_idx)
        pos_embeds   = self.pos_embedding(torch.arange(
            sequence_length, device=in_idx.device
        ))

        x      = token_embeds + pos_embeds
        x      = self.dropout_embed(x)
        x      = self.transformer_blocks(x)
        x      = self.final_norm(x)
        logits = self.out_head(x) # Linear output.

        return logits




