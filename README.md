# First-Step-to-My-Copilot

A simple implementation of GPT, the first step toward an model that can help write codes

---
## The purpose of this try

Through this implementation of GPT, you will have a clearer understanding of:
- the mechanism behind `Tokenizer`, `Self-Attention`, `Multi-head-self-attention`...
- how training and generating texts works on a **pratical level**

**If** you also have the **concern** on how to further the exploration on LLM, this is a good try
-  only **ten** minutes needed to train on `cpu`

---

## Generate Demo
```
=== generation mode ===
Model and tokenizer loaded successfully.
input 'quit' to exit
input your prompt

User:data = [1, 2, 3
Model:, 4, 5]
for num in numbers:
    total = total + number
print(total)

x = 15
if x > 10:
    print("Sm
```
**Actually** it is copying the data in `corpus.txt` and sometimes making some changes, due to the dataset is **tooo small**

---

## GPT structure

### SimpleTokenizer

- encode char to int and decode int to char
- classmethod `load` to create Tokenizer from params in file

### scaled_dot_product_attention

- calculating the weights of each token to other tokens
- `out`: the output has the shape of (..., seq_len, embed_dim) implementing the attention output of each token

### class MultiHeadSelfAttention

- create `qkv` and train together as a tensor, chunk it when to use, it enhance speed
- `o_proj` is the output projection: to combine multi heads together but not concat seperatly
- `attn_mask` to introduce casual mask

### class FeedForward

- `Linear(d_model, d_ff)`: to increase model capacity
- `GELU()`: to introduce non-linearity

### class DecoderBlock

- `ln1/ln2`: LayerNorm for each token
- `attn`: multiHeadSelfAttention
- `ffn`: FeedForward layer
- has residual connection after attn and ffn layers

## project structure

```bash
First-Step-to-My-Copilot/
|--paras/               # incuding the paragrams of  tokenizer and GPT
    |--tiny_gpt.pth 
    |--tokenizer.pth
|--Tiny-GPT.py          # GPT model implementation
|--corpus.txt           # Training corpus
```

## CLI Usage

This project provides a command-line interface to train and generate text with the TinyGPT model.

### Command-Line Arguments

* `--device`: Set device (`cpu` or `cuda` for GPU). Default is `cpu`.
* `--epochs`: Number of training epochs. Default is `20`.
* `--train`: Flag to start training.
* `--generate`: Flag to generate text using the trained model.
* `--model_path`: Path to save/load the trained model. Default is `'paras/tiny_gpt.pth'`.
* `--tokenizer_path`: Path to save/load the tokenizer. Default is `'paras/tokenizer.pth'`.



