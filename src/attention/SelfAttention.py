import torch
from torch import nn

from data import SampleAttentionData

def softmax_naive(x):
    return torch.exp(x) / torch.exp(x).sum(dim=0)

print()
print("##########")
print("Simplified Self-Attention")
print("##########")

inputs = SampleAttentionData.inputs

##
# Getting the attention score and weights from only
# a single input. We will later alter this to operate
# on a full input array.
##


query = inputs[1] ## <-- Second token serves as the query.
attn_scores_2 = torch.empty(inputs.shape[0])
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i, query) ## <-- Dot product; higher values mean
                                             ##     closer relationship between
                                             ##     the input vectors.
print("Attention scores from query: ", attn_scores_2)

##
# The attention scores must be normalized, so their combinations sum to 1.
# "Normalize" here is done using the "softmax" function.
#
## Basic normalize
attn_weights_2_tmp = attn_scores_2 / attn_scores_2.sum()
print("Attention weights: ", attn_weights_2_tmp)
print("Sum: ", attn_weights_2_tmp.sum())

## Naive normalize
attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights with naive softmax: ", attn_weights_2_naive)
print("Sum naive: ", attn_weights_2_naive.sum())

## Torch normalize
attn_weights_2_torch = torch.softmax(attn_scores_2, dim=0)
print("Attention weights with torch softmax: ", attn_weights_2_torch)
print("Sum torch: ", attn_weights_2_torch.sum())

context_vector_2 = torch.zeros(query.shape)
for i,x_i in enumerate(inputs):
    context_vector_2 += attn_weights_2_torch[i] * x_i
print("Context vector from torch weights: ", context_vector_2)

##
# Now we generalize the above into something that can be
# run over a full input collection.
##
attn_scores = torch.empty(6, 6)
for i,x_i in enumerate(inputs):
    for j,x_j in enumerate(inputs):
        attn_scores[i, j] = torch.dot(x_i, x_j)
print("Full attention scores for all inputs:\n", attn_scores)

##
# Achieve the above result without for, but with matrix multiplication
##
attn_scores = inputs @ inputs.T # <-- Multiply the input matrix by its transpose.
print("Attention scores after matrix multiplication:\n", attn_scores)

##
# Normalize the scores to get the full suite of weights.
# The `dim` parameter is the dimension  of the input along which
# the softmax function will apply.
# Setting dim=-1 instructs it to run along the last dimension.
# This will normalize across columns, ensuring each row sums to 1.
##
attn_weights = torch.softmax(attn_scores, dim=-1)
print("Normalised scores to get the weights:\n", attn_weights)
print("All row sums: ", attn_scores.sum(dim=-1))

# Now compute all context vectors via matrix multiplication.
all_context_vectors = attn_weights @ inputs
print("All context vectors:\n", all_context_vectors)
print("Previous second context vector: ", context_vector_2)

##########
# Extended Self-Attention
##########

##
# Begin similar to the above.
# Grab the second element to test with first, then generalize.
##
x_2 = SampleAttentionData.x_2
d_in = SampleAttentionData.dimension_in
d_out = SampleAttentionData.dimension_out

# Dimensions are normally the same in standard GPT models.
# Now we initialize the weight matrices.
torch.manual_seed(123)

# requires_grad=False here to de-clutter the outputs at this point.
# For training, we will set this to True.
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# Now compute each vector.
query_2 = x_2 @ W_query
key_2   = x_2 @ W_key
value_2 = x_2 @ W_value
print("Query vector from the single input: ", query_2)

keys = inputs @ W_key
values = inputs @ W_value
print("keys.shape: ", keys.shape)
print("values.shape: ", values.shape)

##
# Now that we have the embedding space, we can compute
# the unnormalized attention scores.
##
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print("Attention scores for omega22, ", attn_score_22)

attn_scores_2 = query_2 @ keys.T
print("Generalized computation for all attention scores: ", attn_scores_2)

##
# Now we want the weights from the scores.
# We scale the attention scores using the softmax.
# Now, however, we also scale the scores by the sqrt
# of the embedding dimension of the keys.
##
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
print("Attention weights after scaling the scores: ", attn_weights_2)

##
# Normalization by scaling using the dimension is to avoid small gradients.
# As the dimension grows large dot products can result in small gradients during
# backpropagation due to the softmax function. As the dot product increases softmax
# behaves more like a step, which nears the gradients to zero. This smallness
# can result in drastically slow computation/learning.
##

##
# Now we have all we need, we can compute the context vector.
# We multiply each value vector with the respective attention weight,
# then summing them all to get the context vector.
##
context_vector_2 = attn_weights_2 @ values
print("Context vector after applying scaling using the dimension: ", context_vector_2)

##
# Query, Key, Value  naming scheme.
# These names are taken from traditional database naming.
# As they are associated with concepts of storing, searching,
# and retrieving information.
# "query" is associated with searching. It represents the item
# the model focuses on or tries to understand. The query is
# the source of binding the other parts of the attention.
# "key" is associated with indexing and searching. In the
# attention mechanism, each item in the input has a key. These
# keys are used to match the query.
# "value" is similar to the value in a key-value pair. It is
# the actual content of the input items. Once the model determines
# which keys are the most relevant given the query, it pulls the
# associated values.
##

##
# Compact self-attention
##
class SelfAttentionV1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key   = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))

    def forward(self, x):
        keys         = x @ self.W_key
        queries      = x @ self.W_query
        values       = x @ self.W_value
        attn_scores  = queries @ keys.T ## The omega value
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attn_weights @ values

        return context_vector

torch.manual_seed(123)
sa_v1 = SelfAttentionV1(d_in, d_out)
print("Compute the context vectors for the input, now using our SelfAttentionV1 class:\n", sa_v1(inputs))

##
# Now improve the attention mechanism by leveraging
# PyTorch's nn.Linear layers. They perform efficient
# matrix multiplication whet the bias units are disabled.
# nn.Linear is also better than nn.Parameter as it has
# an optimized weight initialization scheme built-in.
# This means that we get a much more stable and effective
# model training.
##
class SelfAttentionV2(nn.Module):
    def __init__(self, d_in, d_out, qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)

    def forward(self, x):
        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)
        attn_scores = queries @ keys.T
        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )
        context_vector = attn_weights @ values

        return context_vector

##
# Use v2 similar to v1.
# Weights will be different here, due to nn.Linear having
# that optimized weight initialization scheme over nn.Parameter.
##
torch.manual_seed(789)
sa_v2 = SelfAttentionV2(d_in, d_out)
print("Context vectors from the updated SelfAttentionV2 class:\n", sa_v2(inputs))

##
# Trying out v2 weight in v1.
##
sa_v1.W_query = nn.Parameter(sa_v2.W_query.weight.T)
sa_v1.W_key   = nn.Parameter(sa_v2.W_key.weight.T)
sa_v1.W_value = nn.Parameter(sa_v2.W_value.weight.T)
print("Compute the context vectors for the input, now using our SelfAttentionV1 class but with the v2 weights:\n", sa_v1(inputs))
