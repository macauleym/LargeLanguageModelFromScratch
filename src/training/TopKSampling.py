import torch

from src.configuration.GptConfig import GPT_CONFIG_124M
from src.generation.GenerativeText import generate
from src.training.TrainingFunctions import text_to_token_ids, token_ids_to_text

def run(next_token_logits, model, tokenizer):
    ##
    # Top-k Sampling
    ##
    top_k = 3
    top_logits, top_pos = torch.topk(next_token_logits, top_k)
    print("Top Logits: ", top_logits)
    print("Top Positions: ", top_pos)

    # Set the lowest values to -inf to prep for running softmax.
    new_logits = torch.where(
          condition = next_token_logits < top_logits[-1]
        , input = torch.tensor(float('-inf'))
        , other = next_token_logits
        )
    print("New Logits for Top-k Sampling: ", new_logits)

    topk_probabs = torch.softmax(new_logits, dim = 0)
    print("Probabilities for Top-k: ", topk_probabs)

    # Leveraging the new generate function.
    torch.manual_seed(123)
    token_ids = generate(
          model          = model
        , idx            = text_to_token_ids("Ever effort moves you", tokenizer)
        , max_new_tokens = 15
        , context_size   = GPT_CONFIG_124M["context_length"]
        , top_k          = 25
        , temperature    = 1.4
        )
    print("Output text from new generate function:\n", token_ids_to_text(token_ids, tokenizer))

    return new_logits, topk_probabs, token_ids