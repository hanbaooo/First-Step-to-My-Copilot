# First-Step-to-My-Copilot
a simple implementation of GPT, the first step toward an model that can help write codes

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
|--load-corpus.py       # load corpus from wikitext
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



