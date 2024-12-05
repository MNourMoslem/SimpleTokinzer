"""
This file contains the SimpleTokenizer class, which implements the tokenizer with helper functions.
"""

import regex
import json
import os

class SimpleTokinzer:
    """
    SimpleTokinzer class is responsible for tokenizing text data using regular expressions.
    It also allows saving and loading vocabularies.
    """

    def __init__(self, vocab_dir=None):
        """
        Initializes the SimpleTokinzer with default values.
        If a vocabulary directory is provided, it loads the vocabulary from that directory.
        """
        # Regex pattern for tokenizing text
        self.pattern = r"<\|[^\|]*\|>|'s|'t|'ll|'ve|'r|\s*[^\d\W]+|[\d]+|[^\w]"
        self.special_tokens = set()  # Set to store special tokens
        self.unknown_token = "<|unknown|>"  # Used to handle unknown characters

        if vocab_dir:
            self.load(vocab_dir)  # Load vocabulary if directory is provided
            return

        self.num_tokens = None  # Total number of tokens in the vocabulary
        self.vocab = {i: chr(i) for i in range(256)}  # Initialize vocabulary with ASCII characters
        self.word2token = {}  # Dictionary to map words to token IDs

    def set_special_tokens(self, tokens: list, unknown_token: str = None):
        """
        Sets special tokens that should be included in the vocabulary.
        """
        self.special_tokens = set(tokens)  # Update the special tokens set
        if unknown_token:
            self.unknown_token = unknown_token

    def train(self, data: str, num_tokens):
        """
        Trains the tokenizer by analyzing the given data and creating a vocabulary with the specified number of tokens.
        """
        adjusted_data = self._apply_pattern(self.pattern, data)  # Apply regex pattern to the data
        raws = [get_raw(item) for item in adjusted_data]  # Get the raw byte representation of each token
        epochs = num_tokens - 256 - len(self.special_tokens)  # Calculate the number of iterations needed

        # Iterate through the training process to create the vocabulary
        for i in range(epochs):
            print(end='\r')
            print(f"Processing: {i+1:>6} / {epochs:<6}", end="")
            pairs = {}

            # Count the frequency of each pair of tokens
            for raw in raws:
                if len(raw) > 1:
                    for pair in zip(raw, raw[1:]):
                        pairs[pair] = pairs.get(pair, 0) + 1
            if not pairs:
                i -= 1
                break

            # Get the most frequent pair and add it to the vocabulary
            top = get_most_pair(pairs)
            token_id = 256 + i
            self.vocab[token_id] = self.decode(top)

            # Merge the pair into a single token in the raw data
            for j in range(len(raws)):
                raws[j] = merge(raws[j], top, token_id)

        # Add special tokens to the vocabulary
        for i2, s_token in enumerate(self.special_tokens, start=1):
            self.vocab[i + 256 + i2] = s_token

        # Add special unknown token to the vocabulary
        self.vocab[len(self.vocab)] = self.unknown_token

        # Create a mapping from words to token IDs
        self.word2token = {v: int(k) for k, v in self.vocab.items()}
        self.num_tokens = len(self.vocab)  # Update the total number of tokens
        print()
        print('Done!')

    def _apply_pattern(self, pattern: str, data: str):
        """
        Applies the given regex pattern to the input data and returns the matches.
        """
        return regex.findall(pattern, data)

    def decode_no_join(self, token_ids: list) -> list:
        """
        Decodes a sequence of token IDs but does not join them into a single text.
        Instead, it returns a list where each token is converted to its character equivalent.
        Handles special tokens properly.
        """
        return [self.vocab.get(token, self.unknown_token) for token in token_ids]

    def decode(self, token_ids: list):
        """
        Decodes a sequence of token IDs back into the original text.
        Handles special tokens properly.
        """
        return "".join(self.decode_no_join(token_ids))

    def encode(self, text: str):
        """
        Encodes a sequence of characters into token IDs using the current vocabulary.
        Prioritizes special tokens to prevent splitting.
        """
        raw = []
        i = 0

        while i < len(text):
            # Check for special tokens first
            matched = False
            for token in sorted(self.special_tokens, key=len, reverse=True):  # Sort by length to match longest first
                if text[i:i+len(token)] == token:
                    raw.append(self.word2token.get(token, self.word2token[self.unknown_token]))
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue

            # Fallback to character-based encoding if no special token is matched
            char = text[i]
            raw.append(self.word2token.get(char, self.word2token[self.unknown_token]))
            i += 1

        return raw

    def save(self, dir: str):
        """
        Saves the current vocabulary to a file in the specified directory.
        """
        json_vocab = json.dumps(self.vocab)
        with open(dir, "w") as vocab_file:
            vocab_file.write(json_vocab)

    def load(self, dir: str):
        """
        Loads a vocabulary from the specified file and updates the tokenizer's vocabulary.
        """
        with open(dir, 'r') as f:
            vocab = json.load(f)

        self.vocab = {int(k): v for k, v in vocab.items()}
        self.word2token = {v: int(k) for k, v in self.vocab.items()}
        self.num_tokens = len(self.vocab)  # Update the total number of tokens


def get_raw(text: str, encoding="utf-8"):
    """
    Converts a string into its raw byte representation using the specified encoding.
    """
    return list(text.encode(encoding))

def get_pairs(raw: list):
    """
    Counts the frequency of adjacent token pairs in the input list.
    """
    pairs = {}
    for pair in zip(raw, raw[1:]):
        pairs[pair] = pairs.get(pair, 0) + 1
    return pairs

def sort_pairs(pairs: dict):
    """
    Sorts the token pairs by frequency in descending order.
    """
    return sorted(((v, k) for k, v in pairs.items()), reverse=True)

def merge(raw: list, pair: tuple, token_id: int):
    """
    Merges the most frequent pair in the raw list into a single token.
    """
    new_raw = []
    i = 0

    while i < len(raw):
        if i < len(raw) - 1 and (raw[i], raw[i+1]) == pair:
            new_raw.append(token_id)
            i += 2
        else:
            new_raw.append(raw[i])
            i += 1

    return new_raw

def get_most_pair(pairs: dict):
    """
    Returns the token pair with the highest frequency from the input dictionary.
    """
    return max(pairs, key=pairs.get)

