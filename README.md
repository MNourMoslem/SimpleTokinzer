__SimpleTokenizer__

Overview
The SimpleTokenizer is a custom tokenizer implemented in Python. It is designed to break down a given text into a sequence of tokens based on a specified regular expression pattern. The tokenizer can be trained on a corpus of text to generate a vocabulary that maps characters or sequences of characters to token IDs. This allows for efficient text encoding and decoding, which can be useful in various natural language processing tasks.

----
How to train the tokenizer?
What the tokenizer does is find the most frequent pair of characters in a text, create a token for that pair, and then replace the pair in the text with that token. This process generates one token, and to create more tokens, the process is repeated.

Steps:

1. Find the most frequent pair of characters.
2. Convert the pair to a token.
3. Merge the pair in the text into that token.
4. Repeat the process to find more tokens.

----
example:

text = "aabcabc"
We find that "ab" appears the most, so we convert it to a new token called "Z". Then, we merge "a" and "b" to get "Z" and return the new text:

text = "aZcaZ"
If we repeat the process and want to find another token, we see that "aZ" now appears the most. We convert it into a token, merge "a" and "Z" into the new token, and the text becomes:

text = "XcX"

----
----
Important Note:
It's important that we don't apply this directly to our text because sometimes we don't want certain characters to be merged together, such as punctuation and words or numbers and words.

For example, instead of merging "dog." "dog?" "dog!" into a single token, we want "dog" as one token and ".", "?", "!" as separate tokens.

The same applies to numbers. For example, instead of merging "someword23", we want two tokens like "someword" and "23".

This is why we apply a regular expression to separate characters that we don't want to concatenate.

example:
text = "my email is somename234@gmail and my nickname is somename999."

after regular expression:
text = ("my", " email", " is", "somename", "234", "@", "gmail", " and", " my", " nickname", " is", " somename", "999", ".")

then we basically apply same steps
most pair -> create token -> merge pair to the token -> repeate for more tokens if needed

(notice also that we don't like to concatenate space (" ") unless it comes in the beginning of the token)

---

Tokenizer Algorithm
1. Pattern Matching
The tokenizer uses a predefined regular expression pattern to identify tokens in the text. This pattern is designed to capture contractions, words, digits, and punctuation (inspired from gpt2 tokinzer and almost the same pattern).
The pattern is: r"'s|'t|'ll|'ve|'r|\s*[^\d\W]+|[\d]+|[^\w]".
This pattern matches:
Contractions (e.g., 's, 't)
Words (sequences of non-digit, non-whitespace characters)
Digits (numbers)
Punctuation and other non-word characters

2. Training
During training, the algorithm analyzes a large text corpus and repeatedly merges the most frequent pairs of tokens.
Initially, the vocabulary contains ASCII characters (0-255).
The training process adds new tokens by merging the most frequent adjacent token pairs in the text.
This process continues until the desired number of tokens is reached, accounting for special tokens.

3. Encoding
Encoding converts text into a sequence of token IDs using the generated vocabulary.
The algorithm iterates through the text, finding the longest match in the vocabulary for each sequence of characters.
If a match is found, the corresponding token ID is added to the encoded sequence.

4. Decoding
Decoding converts a sequence of token IDs back into the original text.
The algorithm looks up each token ID in the vocabulary and concatenates the corresponding text.

5. Saving and Loading Vocabulary
The vocabulary can be saved to a file and loaded later for encoding or decoding new text.
This allows for consistent tokenization across different datasets or sessions.
Code Overview
SimpleTokenizer Class
Attributes:

pattern: The regular expression pattern used to tokenize text.
special_tokens: A set of special tokens that are reserved in the vocabulary.
num_tokens: The total number of tokens in the vocabulary.
vocab: A dictionary mapping token IDs to their corresponding text.
word2token: A dictionary mapping text to its corresponding token ID.
Methods:

* init(self, vocab_dir=None): Initializes the tokenizer. If a vocabulary directory is provided, it loads the vocabulary from that directory.
* set_special_tokens(self, tokens: list): Sets special tokens that should be included in the vocabulary.
* train(self, data: str, num_tokens): Trains the tokenizer on the given data to generate a vocabulary with the specified number of tokens.
* _apply_pattern(self, pattern: str, data: str): Applies the regular expression pattern to the input data and returns the matches.
* decode(self, token: tuple): Decodes a sequence of token IDs back into the original text.
* encode(self, text: list): Encodes a sequence of characters into token IDs using the current vocabulary.
* save(self, dir: str, vocab_name: str): Saves the current vocabulary to a file in the specified directory.
* load(self, dir: str): Loads a vocabulary from the specified file and updates the tokenizer's vocabulary.
  
Helper Functions
* get_raw(text: str, encoding="utf-8"): Converts a string into its raw byte representation using the specified encoding.
* get_pairs(raw: list): Counts the frequency of adjacent token pairs in the input list.
* sort_pairs(pairs: dict): Sorts the token pairs by frequency in descending order.
* get_pairs_str(pairs: list, decoding="utf-8", errors="replace"): Converts a list of byte token pairs into a string representation.
* merge(raw: list, pair: tuple, token_id: int): Merges the most frequent pair in the raw list into a single token.
* get_most_pair(pairs: dict): Returns the token pair with the highest frequency from the input dictionary.
