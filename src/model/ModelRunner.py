import torch
import tiktoken

from configuration.GptConfig import GPT_CONFIG_124M
from DummyGPTModel import DummyGPTModel
from GPTModel import GPTModel

from LayerNormalizer import LayerNormalizer
from FeedForward import FeedForward
from SampleDeepNeuralNetwork import SampleDeepNeuralNetwork
from TransformerBlock import TransformerBlock

print()
print("##########")
print("GPT Model Runner")
print("##########")

##
# Some convenience functions.
##
def meanOf(t):
    return t.mean(dim=-1, keepdim=True)

def varianceOf(t):
    return t.var(dim=-1, keepdim=True)

def print_gradients(model, x):
    output = model(x) # Forward pass.
    target = torch.tensor([[0.]])

    loss = torch.nn.MSELoss()
    loss = loss(output, target) # Calculate loss, based on difference between target and output.

    loss.backward() # Backward pass, to get the gradients.

    for name, param in model.named_parameters():
        if 'weight' in name:
            print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")

tokenizer = tiktoken.get_encoding("gpt2")
batch = []
text1 = "Every effort moves you"
text2 = "Every day holds a"

batch.append(torch.tensor(tokenizer.encode(text1)))
batch.append(torch.tensor(tokenizer.encode(text2)))
batch = torch.stack(batch, dim=0)
print("Tokenized batch:\n", batch)

torch.manual_seed(123)
model = DummyGPTModel(GPT_CONFIG_124M) # 124 million parameter model.
logits = model(batch)
print("Output shape: ", logits.shape)
print("Logits: \n", logits)

##
# Implementing the layer normalization, standard in most
# Transformer architectures.
##
torch.manual_seed(123)
batch_example = torch.randn(2, 5) ## 2 sets with 5 dimensions.
layer = torch.nn.Sequential(
      torch.nn.Linear(in_features=5, out_features=6)
    , torch.nn.ReLU()) ## ReLU = [Re]ctified [L]inear [U]nit. Clamps negative values to 0.
out = layer(batch_example)
print("Layer normalization example:\n", out)

##
# Using -1 for the dimension here will calculate across the columns,
# aggregating for each row. If we instead used 0 here it would
# calculate across rows. In our case, since we have a multi-dimensional
# tensor (matrix), we want to sum across the columns.
##
mean = out.mean(dim=-1, keepdim=True)
variance = out.var(dim=-1, keepdim=True)
print("Mean:\n", mean)
print("Variance:\n", variance)

out_norm = (out - mean) / torch.sqrt(variance)
mean = meanOf(out_norm)
variance = varianceOf(out_norm)
print("Normalized layer outputs:\n", out_norm)

# Disable scientific notation for better readability.
torch.set_printoptions(sci_mode=False)
print("Layer Mean:\n", mean)
print("Layer Variance:\n", variance)

# Using the new LayerNormalizer class.
normalizer = LayerNormalizer(embed_dimension=5)
out_ln     = normalizer(batch_example)
mean       = meanOf(out_ln)
variance   = varianceOf(out_ln)
print("Normalizer'd Mean:\n", mean)
print("Normalizer'd Variance:\n", variance)

# Using the FeedForward class.
ffn = FeedForward(GPT_CONFIG_124M)
x   = torch.rand(2, 3, 768)
out = ffn(x)
print("FeedForward output tensor:\n", out.shape)

# Testing out a sample deep Neural Network to illustrate
# the use of the shortcut module.
layer_sizes = [3, 3, 3, 3, 3, 1]
sample_input = torch.tensor([[1., 0., -1.]])
torch.manual_seed(123)
model_without_shortcut = SampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=False
)
print("\nGradients for Model without Shortcut")
print_gradients(model_without_shortcut, sample_input)

##
# The output of this illustrates the _vanishing gradients problem_.
# Where the gradient becomes smaller and smaller as we traverse back
# from the last layer (4 in this case) to the first layer (0).
# Next we'll create a model that uses the skip connections (shortcuts)
# to show how these help retain the gradients when traversing back
# through the NN's layers.
##

torch.manual_seed(123)
model_with_shortcut = SampleDeepNeuralNetwork(
    layer_sizes, use_shortcut=True
)
print("\nGradients for Model using the Shortcut")
print_gradients(model_with_shortcut, sample_input)

##
# Now we have all the pieces we need to build up the transformer block.
##
torch.manual_seed(123)
x = torch.rand(2, 4, 768) # 768 is the dimensions we want for training.
block = TransformerBlock(GPT_CONFIG_124M)
output = block(x)
print("Input shape to TransformerBlock: ", x.shape)
print("Output shape from TransformerBlock: ", output.shape)

##
# Now initialize the 124-million GPT model and see what it do.
##
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)

out = model(batch)
print("Input Batch to GPT Model:\n", batch)
print("\nOutput shape: \n", out.shape)
print("\nOutput from the GPT model:\n", out)

total_params = sum(p.numel() for p in model.parameters())
print(f"Total number of parameters in this GPT model: {total_params:,}")

## The params should show 163,009,536, which is an increased number from
## the 124-million that we referenced earlier.
## This is due to _weight tying_. The model ties the weight from the token
## embedding layer to the output layer.
print("Token embedding layer shape: ", model.token_embedding.weight.shape)
print("Output layer shape: ", model.out_head.weight.shape)

## Remove the output layer param count from the total model count.
## This will remove the weight tying amount, and should give us the
## expected 124-million parameter count.
total_params_gpt2 = (
    total_params - sum(p.numel()
                       for p in model.out_head.parameters())
)
print(f"Number of trainable parameters in the GPT model considering weight tying: {total_params_gpt2:,}")

## Calculate the memory requirement for the 163-million parameters.
total_size_bytes = total_params * 4 # float32 will need 4 bytes, per parameter.
total_size_mb = total_size_bytes / (1024 * 1024) # Convert to megabytes
print(f"Total memory footprint of the model: {total_size_mb:.2f}MB")
