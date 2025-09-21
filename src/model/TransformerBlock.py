from torch import nn

from src.attention.MultiHeadAttention import MultiHeadAttention
from src.model.FeedForward import FeedForward
from src.model.LayerNormalizer import LayerNormalizer


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(
            dimension_in    = config["embed_dimension"],
            dimension_out   = config["embed_dimension"],
            context_length  = config["context_length"],
            dropout_percent = config["dropout_rate"],
            head_count      = config["head_count"],
            qkv_bias        = config["qkv_bias"]
        )
        self.feed_forward  = FeedForward(config)
        self.normal1       = LayerNormalizer(config["embed_dimension"])
        self.normal2       = LayerNormalizer(config["embed_dimension"])
        self.drop_shortcut = nn.Dropout(config["dropout_rate"])

    def forward(self, x):
        shortcut = x ## Shortcut for attention block.
        x = self.normal1(x)
        x = self.attention(x)
        x = self.drop_shortcut(x)
        x = x + shortcut ## Add back the original.

        shortcut = x ## Shortcut to feed forward block.
        x = self.normal2(x)
        x = self.feed_forward(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
