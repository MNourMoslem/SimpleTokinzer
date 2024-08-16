from tokenizer.simple_tokenizer import SimpleTokinzer

with open('TinyStories10k.txt', 'r') as f:
    tiny = f.read()

tokenizer = SimpleTokinzer()
tokenizer.set_special_tokens(["<|startoftext|>", "<|endoftext|>"])

tokenizer.train(tiny, 1000)

print("Tokinzer vocab:")
print(tokenizer.vocab)

print("Tokinzer special_tokens:")
print(tokenizer.special_tokens)

tokenizer.save('vocab2.json')

tokenizer2 = SimpleTokinzer("vocab2.json")

print("Tokinzer2 vocab:")
print(tokenizer2.vocab)

print("Tokinzer special_tokens:")
print(tokenizer2.special_tokens) # it would be empty

example = "example text to test the tokinzer encoding and decoding."
enc = tokenizer2.encode(example)
print(f"Encoded Text: {enc}")

dec = tokenizer2.decode(enc)
print(f"Decoded Text: {dec}")