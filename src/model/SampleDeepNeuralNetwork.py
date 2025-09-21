import torch
from torch import nn

from GELUActivation import GELUActivation


class SampleDeepNeuralNetwork(nn.Module):
    def __init__(self, layer_sizes, use_shortcut):
        super().__init__()
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(layer_sizes[0], layer_sizes[1]), GELUActivation()),
            nn.Sequential(nn.Linear(layer_sizes[1], layer_sizes[2]), GELUActivation()),
            nn.Sequential(nn.Linear(layer_sizes[2], layer_sizes[3]), GELUActivation()),
            nn.Sequential(nn.Linear(layer_sizes[3], layer_sizes[4]), GELUActivation()),
            nn.Sequential(nn.Linear(layer_sizes[4], layer_sizes[5]), GELUActivation()),
        ])

    def forward(self, x):
        for layer in self.layers:
            # Compute current layer output.
            layer_output = layer(x)

            # Can we apply the shortcut?
            if self.use_shortcut and x.shape == layer_output.shape:
                x = x + layer_output
            else:
                x = layer_output

        return x
