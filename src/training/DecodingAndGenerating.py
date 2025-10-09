import tiktoken

from src.configuration.GptConfig import GPT_CONFIG_124M
from src.generation.GenerativeText import generate_text_simple
from src.training.TrainingFunctions import text_to_token_ids, token_ids_to_text

def run(model):
    ##
    # Decoding and generation strategies
    ##
    model.to("cpu")
    model.eval()
    tokenizer = tiktoken.get_encoding("gpt2")
    token_ids = generate_text_simple(
          model          = model
        , idx            = text_to_token_ids("Every effort moves you", tokenizer)
        , max_new_tokens = 25
        , context_size   = GPT_CONFIG_124M["context_length"]
        )
    print("Output text:\n", token_ids_to_text(token_ids, tokenizer))

    return model
