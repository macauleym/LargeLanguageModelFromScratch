import matplotlib.pyplot as plt
import torch
from torch import nn

from GELUActivation import GELUActivation

gelu = GELUActivation()
relu = nn.ReLU()

x      = torch.linspace(-3, 3, 100) # Create 100 sample points, between -3 and 3.
y_gelu = gelu(x)
y_relu = relu(x)
plt.figure(figsize=(8, 3))

for i, (y, label) in enumerate(zip([y_gelu, y_relu], ["GELU", "ReLU"]), 1):
    plt.subplot(1,2,i)
    plt.plot(x, y)
    plt.title(f"{label} activation function")
    plt.xlabel("x")
    plt.ylabel(f"{label}(x)")
    plt.grid(True)

plt.tight_layout()
plt.show()
