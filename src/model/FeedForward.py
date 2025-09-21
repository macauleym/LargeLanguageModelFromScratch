from torch import nn

from src.model.GELUActivation import GELUActivation

class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(config["embed_dimension"], 4 * config["embed_dimension"]),
            GELUActivation(),
            nn.Linear(4 * config["embed_dimension"], config["embed_dimension"])
        )

    def forward(self, x):
        return self.layers(x)
