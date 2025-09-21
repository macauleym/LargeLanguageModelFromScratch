import torch
from torch import nn

from src.model.LayerNormalizer import LayerNormalizer
from src.model.TransformerBlock import TransformerBlock


class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        ## Reminder:
        ## "Embedding" is basically just a fancy dictionary; it is a matrx
        ## that holds whatever data we like.
        self.token_embedding = nn.Embedding(config["vocab_size"], config["embed_dimension"])
        self.position_embedding = nn.Embedding(config["context_length"], config["embed_dimension"])
        self.drop_embedding = nn.Dropout(config["dropout_rate"])

        self.transformer_blocks = nn.Sequential(
            *[TransformerBlock(config) for _ in range(config["layer_count"])])

        self.final_normalizer = LayerNormalizer(config["embed_dimension"])
        self.out_head = nn.Linear(
            config["embed_dimension"], config["vocab_size"], bias=False
        )

    def forward(self, in_idx):
        batch_size, sequence_length = in_idx.shape
        token_embeddings = self.token_embedding(in_idx)

        ## The device setting allows us to train the model on
        ## either the CPU or the GPU
        position_embeddings = self.position_embedding(
            torch.arange(sequence_length, device=in_idx.device)
        )

        x = token_embeddings + position_embeddings
        x = self.drop_embedding(x)
        x = self.transformer_blocks(x)
        x = self.final_normalizer(x)

        logits = self.out_head(x)

        return logits