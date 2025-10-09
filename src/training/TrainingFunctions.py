import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from src.generation.GenerativeText import generate_text_simple

##
# Functions to handle to/from token ids.
##
def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={'<|endoftext|>'})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)

    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)

    return tokenizer.decode(flat.tolist())

##
# Functions for loss calculation
##
def calc_loss_batch(input_batch, target_batch, model, device):
    input_device  = input_batch.to(device)
    target_device = target_batch.to(device)
    logits        = model(input_device)
    loss          = torch.nn.functional.cross_entropy(
        logits.flatten(0,1), target_device.flatten()
    )

    return loss

def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batche) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(
                input_batch, target_batche, model, device
            )
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

##
# Training
##
def evaluate_model(model, train_loader, validation_loader, device, eval_iter):
    model.eval()
    with torch.no_grad(): # Gradients are not needed during evaluation.
        train_loss = calc_loss_loader(
            train_loader, model, device, num_batches=eval_iter
        )
        val_loss = calc_loss_loader(
            validation_loader, model, device, num_batches=eval_iter
        )

    model.train()

    return train_loss, val_loss

def generate_and_print_sample(model, tokenizer, device, start_context):
    model.eval()
    context_size = model.position_embedding.weight.shape[0]
    encoded = text_to_token_ids(start_context, tokenizer).to(device)
    with torch.no_grad():
        token_ids = generate_text_simple(
              model=model
            , idx=encoded
            , max_new_tokens=50
            , context_size=context_size
            )
    decoded_text = token_ids_to_text(token_ids, tokenizer)
    print("Printing training sample:\n", decoded_text.replace("\n", " "))
    model.train()

def train_model_simple(
      model
    , train_loader
    , validation_loader
    , optimizer
    , device
    , num_epochs
    , eval_frequency
    , eval_iter
    , start_context
    , tokenizer
    ):
    train_losses = []
    validation_losses = []
    track_tokens_seen = []

    tokens_seen = 0
    global_step = -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad() ## Resets the gradients from previous batches.
            loss = calc_loss_batch(
                input_batch, target_batch, model, device
            )
            loss.backward()  ## Calculate the loss gradients.
            optimizer.step() ## Updates the weights with gradients.
            tokens_seen += input_batch.numel()
            global_step += 1

            ## Optional evaluation step.
            if global_step % eval_frequency == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, validation_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                validation_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(f"Epoch {epoch+1} (Step {global_step:06d}): "
                      f"Train loss: {train_loss:.3f}"
                      f"Validation loss: {val_loss:.3f}"
                      )

        generate_and_print_sample(
            model, tokenizer, device, start_context
        )

    return train_losses, validation_losses, track_tokens_seen

##
# Training Visualization
##
def plot_losses(epochs_seen, tokens_seen, train_losses, val_losses):
    fig, ax1 = plt.subplots(figsize=(5,3))
    ax1.plot(epochs_seen, train_losses, label = "Training Loss")
    ax1.plot(
        epochs_seen, val_losses, linestyle = "-.", label = "Validation Loss"
    )
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss")
    ax1.legend(loc="upper right")
    ax1.xaxis.set_major_locator(MaxNLocator(integer = True))

    ax2 = ax1.twiny()
    ax2.plot(tokens_seen, train_losses, alpha = 0)
    ax2.set_xlabel("Tokens seen")

    fig.tight_layout()
    plt.show()
