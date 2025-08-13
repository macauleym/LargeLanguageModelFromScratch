import re

class SimpleTokenizerV1:
    def __init__(self, vocab):
        ## Stores the vocabulary as a class attribute to
        ## be used in the encode/decode methods.
        self.str_to_int = vocab

        ## Inverse vocabulary. To map the Ids back to the words/tokens.
        self.int_to_str = {i:s for s,i in vocab.items()}

    ###
    # Process the input text into token Ids.
    ###
    def encode(self, text):
        preprocessed = re.split(r'([,.?!_"()\']|--|\s)', text)
        preprocessed = [
            item.strip() for item in preprocessed if item.strip()
        ]
        ids = [
            self.str_to_int[s] for s in preprocessed
        ]

        return ids

    ###
    # Converts given token Ids back into the corresponding words.
    ###
    def decode(self, ids):
        text = " ".join([
            self.int_to_str[i] for i in ids
        ])

        ## Remove spaces before punctuation.
        text = re.sub(r'\s+([,.?!"()\'])', r'\1', text)

        return text
