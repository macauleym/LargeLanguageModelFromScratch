import torch
from torch import nn

##
# [G]aussian [E]rror [L]inear [U]nit Activation Module
##
class GELUActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        ## Cheaper approximation of the standard GELU function.
        ## Note: The original GPT-2 model was trained with this model,
        ## and found it via curve fitting.
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(1.0 / torch.pi))
            * (x + 0.044715 * torch.pow(x, 3))
        ))
