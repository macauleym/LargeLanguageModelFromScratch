import torch
from torch import nn

from data import SampleAttentionData

from CasualAttention import CasualAttention

print()
print("##########")
print("MultiHeadAttention")
print("##########")

##
# Now we extend the casual self-attention over multiple heads.
# This involves dividing the single casual attention into multiple "heads".
# Essentially, creating multiple instances of the attention mechanism, with
# their own weights, and combining their output.
# We create weight matrices equal to the number of "heads" we want processing.
# The output then also has 2 attention weight matrices, as well as two
# casual and dropout masks.
# The core idea behind this is to get the attention mechanism to run multiple
# times in parallel. With different learned projections, the results of multiplying
# the input by the weight.
##

class MultiHeadAttentionWrapper(nn.Module):
    def __init__( self
                , dimension_in
                , dimension_out
                , context_length
                , dropout_percent
                , head_count
                , qkv_bias=False
                ):
        super().__init__()
        self.heads = nn.ModuleList([CasualAttention(
              dimension_in
            , dimension_out
            , context_length
            , dropout_percent
            , qkv_bias
            ) for _ in range(head_count)])

    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)

##
# The shape of the context vector that we get back is equal
# to (dimension_out * head_count). If we want a dimension of 2
# and 2 heads, we would get a context vector of dimension 4.
##

torch.manual_seed(123)
batch       = SampleAttentionData.batch
context_len = batch.shape[1] # The number of tokens
d_in        = 3
d_out       = 2

mha = MultiHeadAttentionWrapper(
      d_in
    , d_out
    , context_len
    , dropout_percent=0.0
    , head_count=2
    )

context_vectors = mha(batch)
print("Context vectors:\n", context_vectors)
print("Context vector shape: ", context_vectors.shape)

## This should result in a shape of [2,6,4]
## 2 is the number of input texts.
## 6 is the nuber of tokens for each input.
## 4 is the resulting dimension of the embedding.

##
# Next, we will merge the MultiHeadAttentionWrapper and the CasualAttention classes.
# The wrapper instantiated many CasualAttention classes for each head needed.
# The new class will more naturally merge the concepts to improve efficiency.
# It does this by reshaping the input qkv tensors then combining the result.
##
class MultiHeadAttention(nn.Module):
    def __init__(self, dimension_in, dimension_out, context_length, dropout_percent, head_count, qkv_bias=False):
        super().__init__()
        assert (dimension_out % head_count == 0), \
            "dimension_out must be divisible by head_count"

        self.dimension_out = dimension_out
        self.head_count = head_count
        self.head_dimension = dimension_out // head_count ## Reduce the projected dimension to match the desired output dimension.
        self.W_query = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.W_key = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.W_value = nn.Linear(dimension_in, dimension_out, bias=qkv_bias)
        self.out_projection = nn.Linear(dimension_out, dimension_out)
        self.dropout = nn.Dropout(dropout_percent)

        self.register_buffer(
              "mask"
            , torch.triu(torch.ones(context_length, context_length), diagonal=1)
        )

    def forward(self, x):
        b, num_tokens, dimension_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        ## We implicitly split the matrix by adding a head_count dimension.
        ## Then unroll the last dimension.
        keys = keys.view(b, num_tokens, self.head_count, self.head_dimension)
        values = values.view(b, num_tokens, self.head_count, self.head_dimension)
        queries = queries.view(b, num_tokens, self.head_count, self.head_dimension)

        ## Transpose from shape of
        ## (b, num_tokens, head_count, head_dimension)
        ## to
        ## (b, head_count, num_tokens, head_dimension)
        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        ## Compute the dot product for each head.
        attention_scores = queries @ keys.transpose(2, 3)

        ## Trunkate masks to the number of tokens.
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        ## Use the mask to fill the scores.
        attention_scores.masked_fill_(mask_bool, -torch.inf)

        ## Standard affair to get the weights from the scores.
        ## Normalize the scores.
        attention_weights = torch.softmax(attention_scores / keys.shape[-1]**0.5, dim=-1)

        ## Apply Dropout mask.
        attention_weights = self.dropout(attention_weights)

        ## Tensor shape: (b, num_tokens, head_count, head_dimension)
        context_vector = (attention_weights @ values).transpose(1, 2)
        context_vector = context_vector.contiguous().view(
            b, num_tokens, self.dimension_out
        )

        ## Adds an optional linear projection.
        context_vector = self.out_projection(context_vector)

        return context_vector

## To illustrate how the transpose and reshaping of matrices does what we want, consider the following:
a = torch.tensor([[[[0.2745 , 0.6584 , 0.2775 , 0.8573],
                    [0.8993 , 0.0390 , 0.9268 , 0.7388],
                    [0.7179 , 0.7058 , 0.9156 , 0.4340]],

                   [[0.0772 , 0.3565 , 0.1479 , 0.5331],
                    [0.4066 , 0.2318 , 0.4545 , 0.5331],
                    [0.4606 , 0.5159 , 0.4220 , 0.5786]]]])

## We can perform a batched matrix multiplication with the tensor and itself,
## and a view of the tensor. We first transpose the last two dimensions, num_tokens and head_dimension:
## 2 -> The number of inputs.
## 3 -> The number of each input's tokens.
print(a @ a.transpose(2, 3))

## The above is a very compact way of performing the following.
first_head = a[0,0,:,:]
first_res = first_head @ first_head.T
print("First head:\n", first_res)

second_head = a[0,1,:,:]
second_res = second_head @ second_head.T
print("\nSecond head:\n", second_res)
print()

torch.manual_seed(123)
batch_size, context_length, dimension_in = batch.shape
dimension_out = 2
mha = MultiHeadAttention(dimension_in, dimension_out, context_length, dropout_percent=0.0, head_count=2)
context_vectors = mha(batch)
print("Context Vectors from MHA:\n", context_vectors)
print("Context Vector shape from MHA:\n", context_vectors.shape)

##
# This now concludes writing the core components necessary when
# we finally get around to implementing and training our model.
# Note: The sizes for dimensions and head count here are significantly
# smaller that what we will actually use when training.
# GPT-2 has 12 attention heads and a context vector embedding size of 768.
# The largest GPT-2 model has 25 heads and a context vector embedding size of 1,600.
##
