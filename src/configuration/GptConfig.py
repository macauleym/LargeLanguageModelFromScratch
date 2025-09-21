GPT_CONFIG_124M = {
    "vocab_size"              : 50257, # Vocabulary size. Used in the BPE (byte pair encoder) tokenizer from chapter 2.
    "context_length"          : 1024,  # The max number of input tokens the model can handle with positional embedding (ch.2).
    "embed_dimension"         : 768,   # The embedding size, transforming each token into a 768-dimension vector.
    "head_count"              : 12,    # The number of heads we want to use in the multi-headed attention.
    "layer_count"             : 12,    # The number of transformer blocks to use in the model.
    "dropout_rate"            : 0.1,   # The dropout percent (0.1 is 10%) to prevent overfitting (ch.3).
    "embed_dropout"           : 0.1,
    "shortcut_dropout"        : 0.1,
    "multi_attention_dropout" : 0.1,
    "qkv_bias"                : False  # If we should include the bias vector in the Linear layers of the MHA for query, key, and value computations.
                                       # Disabled for now, but will be revisited in a later chapter of the book.
}
