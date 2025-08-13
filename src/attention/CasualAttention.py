import torch
from torch import nn

from data import SampleAttentionData

from SelfAttention import SelfAttentionV2

print()
print("##########")
print("Casual/Masked Attention Mechanism")
print("##########")

inputs = SampleAttentionData.inputs
d_in   = SampleAttentionData.dimension_in
d_out  = SampleAttentionData.dimension_out

##
# General steps to achieve the masking:
# 1. Get the (unnormalized) attention scores.
# 2. Apply the softmax to get the weights.
# 3. Normalize the weights so they sum to 1.
# 4. Mask values above the diagonal of the matrix with '0'.
# 5. Mask the attention scores (unnormalized).
# 6. Re-normalize the now masked rows, so the new rows sum to 1 again.
##
torch.manual_seed(789)
sa_v2        = SelfAttentionV2(d_in, d_out)
queries      = sa_v2.W_query(inputs)
keys         = sa_v2.W_key(inputs)
attn_scores  = queries @ keys.T
attn_weights = torch.softmax(
    attn_scores / keys.shape[-1]**0.5, dim=-1
)
print("Steps 1, 2, and 3 of masking the attention weights:\n", attn_weights)

# Use torch.tril to apply the default mask to the tensor.
context_length = attn_scores.shape[0]
mask_simple = torch.tril(torch.ones(context_length, context_length))
print("Step 4 of masking, creating the initial mask weights to 0 above the diagonal:\n", mask_simple)

masked_simple = attn_weights * mask_simple
print("Step 5 of masking, merging the weights with the mask:\n", masked_simple)

# Re-normalize each row, by dividing each element in each row by the sum of that row.
row_sums = masked_simple.sum(dim=-1, keepdim=True)
mask_simple_norm = masked_simple / row_sums
print("Step 6 of masking, renormalizing each row back to summing to 1:\n", mask_simple_norm)

##
# Alternate masking implementation.
# Leveraging -infinity as the mask value instead of 0.
##
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
print("Alternate masking implementation, leveraging -infinity:\n", masked)

attn_weights = torch.softmax(masked / keys.shape[-1]**0.5, dim=1)
print("Now renormalized weights of the -infinity implementation:\n", attn_weights)

##
# Apply the "dropout" mask to the current masked weights.
# The non-dropped values are scaled by a factor of 1/dropout_percent (1/0.5=2 in the below case).
# This scaling ensure the matrix is still properly
# balanced for attention.
##
torch.manual_seed(123)
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6, 6)
print("Sample dropout tensor:\n", dropout(example))

# Now apply the dropout to our current masked weights.
torch.manual_seed(123)
print("Dropout applied to our masked attention weights:\n", dropout(attn_weights))

# Simulate a batch of inputs.
batch = SampleAttentionData.batch
print("Shape of our sample batch: ", batch.shape)

##
# Implementing the CasualAttention class.
# Similar to the above SelfAttentionV1 and V2 classes, but
# now with the added masking and dropout mechanisms.
##
class CasualAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            'mask',
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x):
        b, num_tokens, d_in = x.shape

        keys    = self.W_key(x)
        queries = self.W_query(x)
        values  = self.W_value(x)

        attn_scores = queries @ keys.transpose(1,2) ## Transpose dimensions 1 & 2 keeping batch dimension at first position.
        attn_scores.masked_fill(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(
            attn_scores / keys.shape[-1]**0.5, dim=-1
        )

        attn_weights   = self.dropout(attn_weights)
        context_vector = attn_weights @ values

        return context_vector

# Now leverage the new attention class, as we have the previous ones.
torch.manual_seed(123)
context_length = batch.shape[1]
ca = CasualAttention(d_in, d_out, context_length, 0.0)
context_vectors = ca(batch)
print("Shape of the context vectors after calculating them using the new CasualAttention class: ", context_vectors.shape)
