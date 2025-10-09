import torch
import tiktoken

import src.training.SaveAndLoadState
from src.generation.GenerativeText import generate_text_simple
from src.model.GPTModel import GPTModel
from src.data.TrainingData import read_file_content, the_verdict_path
from src.training.TrainingFunctions import text_to_token_ids, token_ids_to_text, plot_losses, calc_loss_loader, train_model_simple
from src.training.pull_gpt2_model import openai_settings, openai_params
from src.tokenizing.Tokenizing import create_dataloader_v1

print()
print("##########")
print("Pretraining and Evaluating Output")
print("##########")

def run_initial_model_training():
    ##
    # Step 1:
    # Map the inputs and targets to token IDs.
    ##
    inputs = torch.tensor([[16833, 3626, 6100],
                           [40, 1107, 588]])
    targets = torch.tensor([[3626, 6100, 345],
                            [1107, 588, 11311]])

    ##
    # Step 2:
    # Calculate logit vectors and probability scores with softmax.
    ##
    with torch.no_grad():
        logits = model(inputs)
    probabilities = torch.softmax(logits, dim=-1)
    print("Probabilities shape: ", probabilities.shape)

    ##
    # Steps 3 & 4:
    # Apply argmax to the probabilities to get the corresponding token IDs.
    ##
    token_ids = torch.argmax(probabilities, dim=-1, keepdim=True)
    print("Token IDs:\n", token_ids)

    ##
    # Step 5:
    # Convert the token IDs back into text.
    ##
    print(f"Targets batch 1: {token_ids_to_text(targets[0], tokenizer)}")
    print(f"Outputs batch 1:"
          f" {token_ids_to_text(token_ids[0].flatten(), tokenizer)}")

    ######

    text_idx = 0
    target_probabilities_1 = probabilities[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 1: ", target_probabilities_1)

    text_idx = 1
    target_probabilities_2 = probabilities[text_idx, [0, 1, 2], targets[text_idx]]
    print("Text 2: ", target_probabilities_2)

    log_probabilities = torch.log(torch.cat((target_probabilities_1, target_probabilities_2)))
    print("Log Probabilities: ", log_probabilities)

    average_log_probabilities = torch.mean(log_probabilities)
    print("Average Logs: ", average_log_probabilities)

    negative_average = average_log_probabilities * -1
    print("Inverse Average: ", negative_average)

    print("Logits shape: ", logits.shape)
    print("Targets shape: ", targets.shape)

    logits_flat = logits.flatten(0, 1)
    targets_flat = targets.flatten()
    print("Flattened logits: ", logits_flat.shape)
    print("Flattened targets: ", targets_flat.shape)

    entropy_loss = torch.nn.functional.cross_entropy(logits_flat, targets_flat)
    print("Cross Entropy Loss: ", entropy_loss)

    ##
    # Perplexity is also another way to calculate how well the model
    # understands what word to generate next.
    ##
    perplexity = torch.exp(entropy_loss)

    ##
    # We now begin the process of training the model.
    ##
    text_data = read_file_content(the_verdict_path)

    total_characters = len(text_data)
    total_tokens = len(tokenizer.encode(text_data))
    print("Character count: ", total_characters)
    print("Token count: ", total_tokens)

    ##
    # Split the data into the training chunks to use.
    ##
    train_ratio = 0.90
    split_idx = int(train_ratio * len(text_data))
    train_data = text_data[:split_idx]
    val_data = text_data[split_idx:]

    ##
    # Now load the data sets for training.
    ##
    torch.manual_seed(123)
    train_loader = create_dataloader_v1(
        train_data
        , batch_size=2
        , max_length=GPT_CONFIG_124M["context_length"]
        , stride=GPT_CONFIG_124M["context_length"]
        , drop_last=True
        , shuffle=True
        , num_workers=0
    )
    validation_loader = create_dataloader_v1(
        val_data
        , batch_size=2
        , max_length=GPT_CONFIG_124M["context_length"]
        , stride=GPT_CONFIG_124M["context_length"]
        , drop_last=False
        , shuffle=False
        , num_workers=0
    )

    # Show that the loaders were setup correctly.
    print("Training loader: ")
    for x, y in train_loader:
        print(x.shape, y.shape)

    print("\nValidation loader: ")
    for x, y in validation_loader:
        print(x.shape, y.shape)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device)
        val_loss = calc_loss_loader(validation_loader, model, device)
    print("Training loss: ", train_loss)
    print("Validation loss: ", val_loss)

    return device, train_loader, validation_loader

def run_for_epochs(num_epochs, device, train_loader, validation_loader):
    ##
    # Put everything together and start training a GPTModel for 10 epochs.
    # This will use everything we've built so far, using an AdamW optimizer.
    ##
    torch.manual_seed(123)
    model = GPTModel(GPT_CONFIG_124M)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters()
        , lr=0.0004
        , weight_decay=0.1
    )
    train_losses, val_losses, tokens_seen = train_model_simple(
        model
        , train_loader
        , validation_loader
        , optimizer
        , device
        , num_epochs=num_epochs
        , eval_frequency=5
        , eval_iter=5
        , start_context="Every effort moves you"
        , tokenizer=tokenizer
    )

    # Use new helper to plot losses from training.
    epochs_tensor = torch.linspace(0, num_epochs, len(train_losses))
    plot_losses(epochs_tensor, tokens_seen, train_losses, val_losses)

    return model, optimizer

##
# Model setup
##
GPT_CONFIG_124M = {
    "vocab_size"        : 50257,
    "context_length"    : 256,
    "embed_dimension"   : 768,
    "head_count"        : 12,
    "layer_count"       : 12,
    "embed_dropout"     : 0.1,
    "shortcut_dropout"  : 0.1,
    "attention_dropout" : 0.1,
    "qkv_bias"          : False
}
torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
model.eval()

##
# Using the model to generate text.
##
start_context = "Every effort moves you"
tokenizer = tiktoken.get_encoding("gpt2")

token_ids = generate_text_simple(
    model=model,
    idx=text_to_token_ids(start_context, tokenizer),
    max_new_tokens=10,
    context_size=GPT_CONFIG_124M["context_length"]
)
print("Output text: ", token_ids_to_text(token_ids, tokenizer))

device, train_loader, validation_loader = run_initial_model_training()
epochs = 1 #10
model, optimizer = run_for_epochs(epochs, device, train_loader, validation_loader)

#inverse_vocab, next_token_logits = src.training.TempratureScaling.run()
#new_logits, topk_probabs, token_ids = src.training.TopKSampling.run(next_token_logits, model, tokenizer)

print("Settings: ", openai_settings)
print("Parameter dictionary keys: ", openai_params.keys())

print(openai_params["wte"])
print("Token embedding weight tensor dimensions: ", openai_params["wte"].shape)

src.training.SaveAndLoadState.run(model, openai_params, tokenizer, optimizer, device, GPT_CONFIG_124M)

