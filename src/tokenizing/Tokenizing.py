import torch
import tiktoken
from torch.utils.data import DataLoader

from src.data.TrainingData import read_file_content
from src.data.GPTDatasetV1 import GPTDatasetV1


"""
def tokenize(text: str) -> list[str]:
    token_pattern = r'([,.:?!_"()\\']|--|\\s)'
    result = re.split(token_pattern, text)

    ## Including whitespaces for LLM training usually
    ## depends on the purpose of the LLM and what it will
    ## be used for.
    return [item for item in result if item.strip()]

source = "Hello, world! Is this-- a test?"
token_source = tokenize(source)
print(token_source)

the_verdict = TrainingData.read_file_content("the-verdict.txt")
preprocessed = tokenize(the_verdict)
print(len(preprocessed))
print(preprocessed[:30])

all_words = sorted(set(preprocessed))

## Extending our vocabulary to include tokens
## representing the end of a text, and unknown words.
## Some other special tokens that can be used are:
## [BOS] (beginning of sequence) -- Used to signify to an LLM where content begins.
## [EOS] (end of sequence) -- Similar to <|endoftext|>, it tells the LLM where content ends.
## [PAD] (padding) -- Used to ensure that all given texts to an LLM is of the same length.
##      Shorter content is extended with this token up to the largest text size in a batch.
## GPT models don't need any of these, only <|endoftext|> is sufficient. For both [EOS] and [PAD].
## GPT models also don't need an <|unk|> token, as they leverage _byte pair encoding_ to break words down into parts.
all_words.extend(["<|endoftext|>","<|unk|>"])
vocab_size = len(all_words)
print(vocab_size)

vocab = {token:integer for integer,token in enumerate(all_words)}
for i,item in enumerate(list(vocab.items())[-5:]):
    print(item)

tokenizer = SimpleTokenizerV1.SimpleTokenizerV1(vocab)
"""
text = """"It's the last he painted, you know,"
          Mrs.Gisburn said with pardonable pride."""
"""
ids = tokenizer.encode(text)
print(ids)
print(tokenizer.decode(ids))

text_part1 = "Hello, do you like tea?"
text_part2 = "In the sunlit terraces of the palace."
unrelated_text = " <|endoftext|> ".join((text_part1, text_part2))

tokenizer2 = SimpleTokenizerV2.SimpleTokenizerV2(vocab)
print(tokenizer2.encode(unrelated_text))
print(tokenizer2.decode(tokenizer2.encode(unrelated_text)))

## Leveraging "tiktoken" to encode text using the byte pair encoding algorythm.
tiktokenizer = tiktoken.get_encoding("gpt2")
tiktext = (
    "Hello, do you like tea? <|endoftext|> In the sunlit terraces"
    " of someunknownPlace."
)
tikintegers = tiktokenizer.encode(tiktext, allowed_special={"<|endoftext|>"})
print(tikintegers)
tikstrings = tiktokenizer.decode(tikintegers)
print(tikstrings)
"""

"""
tiktokenizer = tiktoken.get_encoding("gpt2")
enc_text = tiktokenizer.encode(TrainingData.read_file_content("the-verdict.txt"))
print(len(enc_text))
enc_sample = enc_text[50:]

context_size = 4
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")

for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(tiktokenizer.decode(context), "---->", tiktokenizer.decode([desired]))
"""

verdict_text = read_file_content("../../data/the-verdict.txt")
def create_dataloader_v1(
      txt
    , batch_size  = 4
    , max_length  = 256
    , stride      = 128
    , shuffle     = True
    , drop_last   = True
    , num_workers = 0
    ):
    tokenizer = tiktoken.get_encoding("gpt2")
    dataset   = GPTDatasetV1(
          txt
        , tokenizer
        , max_length
        , stride
        )
    dataloader = DataLoader(
          dataset
        , batch_size=batch_size
        , shuffle=shuffle
        , drop_last=drop_last ## If we should drop the last batch if it's size is smaller than the given batch_size, to prevent loss spikes during training.
        , num_workers=num_workers
        )

    return dataloader

#dataloader = create_dataloader_v1(
#      verdict_text
#    , batch_size=8
#    , max_length=4
#    , stride=4
#    , shuffle=False
#    )
#data_iter = iter(dataloader)
#inputs, targets = next(data_iter)
#print("Inputs:\n", inputs)
#print("Targets:\n", targets)

## Hands-on vector embedding example.
#input_ids = torch.tensor([2,3,5,1])
#vocab_size = 6
#output_dim = 3
#
#torch.manual_seed(43893)
#embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
#print(embedding_layer.weight)
#print(embedding_layer(torch.tensor([3])))
#print(embedding_layer(input_ids))

## Using our own data
vocab_size            = 50257
output_dim            = 256
token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)

max_length = 4
dataloader = create_dataloader_v1(
  verdict_text
, batch_size = 8
, max_length = max_length
, stride     = max_length
, shuffle    = False
)
data_iter       = iter(dataloader)
inputs, targets = next(data_iter)
print("Token Ids:\n", inputs)
print("\nInptus shape:\n", inputs.shape)

token_embeddings = token_embedding_layer(inputs)
print(token_embeddings.shape)

## Apply GPT-style absolute embedding approach.
context_length      = max_length
pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
pos_embeddings      = pos_embedding_layer(torch.arange(context_length))
print(pos_embeddings.shape)

## Apply the positional embeddings to our token embeddings.
input_embeddings = token_embeddings + pos_embeddings
print(input_embeddings.shape)
