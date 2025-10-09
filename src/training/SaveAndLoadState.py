import numpy as np
import torch

from src.generation.GenerativeText import generate
from src.model.GPTModel import GPTModel
from src.training.TrainingFunctions import text_to_token_ids
from training.TrainingFunctions import token_ids_to_text

##
# OpenAI training weight updates.
##
def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
    return torch.nn.Parameter(torch.tensor(right))

def load_weights_into_gpt(gpt, params):
    gpt.position_embedding.weight = assign(gpt.position_embedding.weight, params['wpe'])
    gpt.token_embedding.weight    = assign(gpt.token_embedding.weight, params['wte'])

    for block in range(len(params["blocks"])):
        query_weight, key_weight, value_weight = np.split(
            (params["blocks"][block]["attn"]["c_attn"])["w"], 3, axis=-1
        )
        gpt.transformer_blocks[block].attention.W_query.weight = assign(
              gpt.transformer_blocks[block].attention.W_query.weight
            , query_weight.T)
        gpt.transformer_blocks[block].attention.W_key.weight   = assign(
              gpt.transformer_blocks[block].attention.W_key.weight
            , key_weight.T)
        gpt.transformer_blocks[block].attention.W_value.weight = assign(
              gpt.transformer_blocks[block].attention.W_value.weight
            , value_weight.T)

        query_bias, key_bias, value_bias = np.split(
            (params["blocks"][block]["attn"]["c_attn"])["b"], 3, axis=-1
        )
        gpt.transformer_blocks[block].attention.W_query.bias = assign(
              gpt.transformer_blocks[block].attention.W_query.bias
            , query_bias)
        gpt.transformer_blocks[block].attention.W_key.bias   = assign(
              gpt.transformer_blocks[block].attention.W_key.bias
            , key_bias)
        gpt.transformer_blocks[block].attention.W_value.bias = assign(
              gpt.transformer_blocks[block].attention.W_value.bias
            , value_bias)

        gpt.transformer_blocks[block].attention.out_projection.weight = assign(
              gpt.transformer_blocks[block].attention.out_projection.weight
            , params["blocks"][block]["attn"]["c_proj"]["w"].T)
        gpt.transformer_blocks[block].attention.out_projection.bias = assign(
              gpt.transformer_blocks[block].attention.out_projection.bias
            , params["blocks"][block]["attn"]["c_proj"]["b"])

        gpt.transformer_blocks[block].feed_forward.layers[0].weight = assign(
              gpt.transformer_blocks[block].feed_forward.layers[0].weight
            , params["blocks"][block]["mlp"]["c_fc"]["w"].T)
        gpt.transformer_blocks[block].feed_forward.layers[0].bias = assign(
              gpt.transformer_blocks[block].feed_forward.layers[0].bias
            , params["blocks"][block]["mlp"]["c_fc"]["b"])
        gpt.transformer_blocks[block].feed_forward.layers[2].weight = assign(
              gpt.transformer_blocks[block].feed_forward.layers[2].weight
            , params["blocks"][block]["mlp"]["c_proj"]["w"].T)
        gpt.transformer_blocks[block].feed_forward.layers[2].bias = assign(
              gpt.transformer_blocks[block].feed_forward.layers[2].bias
            , params["blocks"][block]["mlp"]["c_proj"]["b"])

        gpt.transformer_blocks[block].normal1.scale = assign(
              gpt.transformer_blocks[block].normal1.scale
            , params["blocks"][block]["ln_1"]["g"])
        gpt.transformer_blocks[block].normal1.shift = assign(
              gpt.transformer_blocks[block].normal1.shift
            , params["blocks"][block]["ln_1"]["b"])
        gpt.transformer_blocks[block].normal2.scale = assign(
              gpt.transformer_blocks[block].normal2.scale
            , params["blocks"][block]["ln_2"]["g"])
        gpt.transformer_blocks[block].normal2.shift = assign(
              gpt.transformer_blocks[block].normal2.shift
            , params["blocks"][block]["ln_2"]["b"])

    ## The original model by OpenAI reused the token embeddings weights
    ## in the output to reduce the total parameter count, known as "weight tying".
    gpt.final_normalizer.scale = assign(gpt.final_normalizer.scale, params["g"])
    gpt.final_normalizer.shift = assign(gpt.final_normalizer.shift, params["b"])
    gpt.out_head.weight        = assign(gpt.out_head.weight,      params["wte"])

def run(model, params, tokenizer, optimizer, device, config) :
    # Save the model's state_dict to a file, so we don't
    # have to re-train it every time we want to use it.
    torch.save(model.state_dict(), "my_llm.pth")

    model = GPTModel(config)
    model.load_state_dict(torch.load("my_llm.pth", map_location=device))
    model.eval()

    # Save the model's state, and optimizer.
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        },
        "my_llm_and_optimizer.pth"
    )

    # We can then restore both the model and optimizer.
    checkpoint = torch.load("my_llm_and_optimizer.pth", map_location=device)
    model = GPTModel(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    model.train()

    model_configs = {
          "gpt2-small (124M)": {"embed_dimension": 768, "layer_count": 12, "head_count": 12}
        , "gpt2-medium (355M)": {"embed_dimension": 1024, "layer_count": 24, "head_count": 16}
        , "gpt2-large (774M)": {"embed_dimension": 1280, "layer_count": 36, "head_count": 20}
        , "gpt2-xl (1558M)": {"embed_dimension": 1600, "layer_count": 63, "head_count": 25}
    }

    model_name = "gpt2-small (124M)"
    NEW_CONFIG = config.copy()
    NEW_CONFIG.update(model_configs[model_name])
    NEW_CONFIG.update({"context_length": 1024})
    NEW_CONFIG.update({"qkv_bias": True}) # Enabled for backward compatibility.

    # Now, leverage the updated config to create a new model instance.
    gpt = GPTModel(NEW_CONFIG)

    # Load the weights we pulled from OpenAI
    load_weights_into_gpt(gpt, params)
    gpt.to(device)
    gpt.eval()

    torch.manual_seed(123)
    token_ids = generate(
          model=gpt
        , idx=text_to_token_ids("Every effort moves you", tokenizer).to(device)
        , max_new_tokens=25
        , context_size=NEW_CONFIG["context_length"]
        , top_k=50
        , temperature=1.0
        )
    print("Output text from loading OpenAI weights:\n", token_ids_to_text(token_ids, tokenizer))
