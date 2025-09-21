import torch
from torch import nn

class LayerNormalizer(nn.Module):
    def __init__(self, embed_dimension):
        super().__init__()
        self.epsilon   = 1e-5
        self.scale = nn.Parameter(torch.ones(embed_dimension))
        self.shift = nn.Parameter(torch.zeros(embed_dimension))

    def forward(self, x):
        mean         = x.mean(dim=-1, keepdim=True)
        variance     = x.var(dim=-1, keepdim=True)
        normalized_x = (x - mean) / torch.sqrt(variance + self.epsilon)

        return self.scale * normalized_x + self.shift
