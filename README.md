# First-Step-to-My-Copilot
a simple implementation of GPT, the first step toward an model that can help write codes

## GPT structure
### SimpleTokenizer
- encode char to int and decode int to char
- classmethod `load` to create Tokenizer from params in file

### scaled_dot_product_attention
- calculating the weights of each token to other tokens
- the output has the shape of (..., seq_len, embed_dim) implementing the attention output of each token



## project structure
```bash
First-Step-to-My-Copilot/
|--paras/           # incuding the paragrams of  tokenizer and GPT
|--Tiny-GPT.py      # GPT model implementation
|--load-corpus.py   # load corpus from wikitext
|--corpus.txt       # Training corpus

```

