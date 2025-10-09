import tiktoken
import torch

from src.configuration.GptConfig import GPT_CONFIG_124M
from src.model.GPTModel import GPTModel

print()
print("##########")
print("Generating Text")
print("##########")

def generate_text_simple(model, idx, max_new_tokens, context_size):
    # idx is a (batch, n_tokens) array of the current context indices.
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)

        ## Here, we now focus only on the last time step.
        ## (batch, n_token, vocab_size)
        ## becomes
        ## (batch, vocab_size)
        logits = logits[:, -1, :]
        probabilities = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probabilities, dim=-1, keepdim=True)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def generate(
      model
    , idx
    , max_new_tokens
    , context_size
    , temperature = 0.0
    , top_k = None
    , eos_id = None
    ):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                  logits < min_val
                , torch.tensor(float('-inf')).to(logits.device)
                , logits
                )
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim = -1)
            idx_next = torch.multinomial(probs, num_samples = 1)
        else:
            idx_next = torch.argmax(logits, dim = -1, keepdim = True)
        if idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim = 1)

    return idx

torch.manual_seed(123)

# First, encode the input context.
start_context = "Hello, I am"
tokenizer = tiktoken.get_encoding("gpt2")
encoded = tokenizer.encode(start_context)
print("encoded: ", encoded)
encoded_tensor = torch.tensor(encoded).unsqueeze(0) # Adds the batch dimension
print("encoded_tensor.shape: ", encoded_tensor.shape)

model = GPTModel(GPT_CONFIG_124M)
model.eval() # Disables dropout here, since we're not concerned with training.
out = generate_text_simple(
      model=model
    , idx=encoded_tensor
    , max_new_tokens=6
    , context_size=GPT_CONFIG_124M["context_length"]
    )
print("Generated output: ", out)
print("Output length: ", len(out[0]))

# Now decode the tokens.
decoded_text = tokenizer.decode(out.squeeze(0).tolist())
print("Decoded tokens: ", decoded_text)
