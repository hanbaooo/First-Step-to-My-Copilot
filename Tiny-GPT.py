"""
toy_gpt_demo.py

A minimal, self-contained GPT-like model demo in PyTorch.
The file demonstrates:
- a tiny tokenizer (char-level)
- positional embeddings
- masked scaled dot-product multi-head self-attention
- transformer decoder block (pre-LN)
- language modeling training loop on a tiny toy corpus
- simple generation (autoregressive sampling)

This is educational code: readable, commented, and runnable on CPU/GPU.
"""
import os
import math
import random
import argparse
from typing import List, Tuple

import torch
from torch import nn
from torch.nn import functional as F


# -------------------------
# Simple char-level tokenizer
# -------------------------
class SimpleTokenizer:
    """
    self.stoi: a dict mapping str -> int
    self.itos: a dict mapping int -> str
    """
    def __init__(self, data: str):
        chars = sorted(list(set(data)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.vocab_size = len(self.stoi)

    def encode(self, s: str) -> List[int]:
        return [self.stoi[ch] for ch in s]

    def decode(self, ids: List[int]) -> str:
        return ''.join(self.itos[i] for i in ids)

    @classmethod
    def load(cls, filepath):
        """Load tokenizer from file."""
        data = torch.load(filepath)
        tokenizer = cls.__new__(cls) # skip __init__ and save attributes directly
        tokenizer.stoi = data['stoi']
        tokenizer.itos = data['itos']
        tokenizer.vocab_size = data['vocab_size']
        return tokenizer

    def save(self, filepath):
        """Save tokenizer to file."""
        torch.save({
            'stoi': self.stoi,
            'itos': self.itos,
            'vocab_size': self.vocab_size
        }, filepath)


# -------------------------
# Scaled Dot-Product Attention
# -------------------------

def scaled_dot_product_attention(q, k, v, mask=None):
    """Scaled dot-product attention mechanism.
    Args:
        q,k,v: (..., seq_len, head_dim)
        mask: (..., seq_len, seq_len) with True at positions to mask
    Returns:
        out: (..., seq_len, head_dim)
        weights: (..., seq_len, seq_len) for visualization
    """
    dk = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(dk)
    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))
    weights = F.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    return out, weights


# -------------------------
# Multi-head Self-Attention
# -------------------------
class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_head, dropout=0.1):
        """Multi-head self-attention mechanism.
        Args:
            d_model (int): embedding dim
            n_head (int): Number of attention heads
            dropout (float, optional): Dropout rate. Defaults to 0.1
        """
        super().__init__()
        assert d_model % n_head == 0, "d_model must be divisible by n_head"
        self.d_model = d_model
        self.n_head = n_head
        self.d_head = d_model // n_head

        # one linear for QKV for efficiency, then split, faster than 3 separate linears
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model) # output projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        """Multi-head self-attention mechanism.
        Args:
            x: (B, T, d_model) T: sequence length
            attn_mask: (B, T, T) with True at positions to mask
        """
        B, T, _ = x.shape
        qkv = self.qkv(x)  # (B, T, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)

        # reshape for heads
        # (B, n_head, T, d_head)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)

        # attn_mask expected shape: (B, T, T) or (1, T, T) ; expand to heads
        if attn_mask is not None:
            # attn_mask True where to mask
            mask = attn_mask.unsqueeze(1).expand(B, self.n_head, T, T)
        else:
            mask = None

        out, weights = scaled_dot_product_attention(q, k, v, mask)
        # out: (B, n_head, T, d_head) -> concat
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)
        out = self.o_proj(out) # o_proj for combining heads outputs
        out = self.dropout(out)
        return out, weights


# -------------------------
# Feed Forward
# -------------------------
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        """Feed Forward Network (FFN) with two linear layers and GELU activation.

        Args:
            d_model (int): Input dimension.
            d_ff (int): Hidden dimension.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        Porpuse:
            To introduce non-linearity and increase model capacity.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# -------------------------
# Transformer Decoder Block (pre-LN)
# -------------------------
class DecoderBlock(nn.Module):
    def __init__(self, d_model, n_head, d_ff, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model) # normalization for every token
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, n_head, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout)

    def forward(self, x, attn_mask=None):
        # pre-LN
        x_norm = self.ln1(x)
        attn_out, weights = self.attn(x_norm, attn_mask) # call forward of MultiHeadSelfAttention
        x = x + attn_out

        x_norm = self.ln2(x)
        ffn_out = self.ffn(x_norm)
        x = x + ffn_out
        return x, weights


# -------------------------
# Tiny GPT-like model
# -------------------------
class TinyGPT(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_head=4, n_layer=4, d_ff=512, max_len=256, dropout=0.1):
        """A tiny GPT-like model for demonstration.
        Args:
            vocab_size (int): Vocabulary size.
            d_model (int, optional): Embedding dimension. Defaults to 128.
            n_head (int, optional): Number of attention heads. Defaults to 4.
            n_layer (int, optional): Number of transformer layers. Defaults to 4.
            d_ff (int, optional): Feed forward hidden dimension. Defaults to 512.
            max_len (int, optional): Maximum sequence length. Defaults to 256.
            dropout (float, optional): Dropout rate. Defaults to 0.1.
        """
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.drop = nn.Dropout(dropout)

        self.layers = nn.ModuleList([
            DecoderBlock(d_model, n_head, d_ff, dropout)
            for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False) # output : probabilities over vocab

        self.max_len = max_len
        self.d_model = d_model

    def forward(self, idx):
        """Forward pass for the model.

        Args:
            idx (torch.Tensor): Input token indices of shape (B, T).
        """
        B, T = idx.shape
        assert T <= self.max_len, "Sequence length exceeds model max_len"

        tok_emb = self.token_emb(idx)  # (B, T, d_model)
        pos = torch.arange(0, T, device=idx.device).unsqueeze(0)
        pos_emb = self.pos_emb(pos)  # (1, T, d_model) same for all batches

        x = tok_emb + pos_emb
        x = self.drop(x)

        # causal mask: mask out future tokens, the i-th token can only attend to [0, i]
        # mask True where to mask
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=idx.device), diagonal=1)
        mask = mask.unsqueeze(0).expand(B, T, T)  # (B, T, T)

        attn_weights_all = []
        for layer in self.layers:
            x, weights = layer(x, mask)
            attn_weights_all.append(weights)

        x = self.ln_f(x)
        logits = self.head(x)  # (B, T, vocab) output logits for each token
        return logits, attn_weights_all

    @torch.no_grad()
    def generate(self, idx, max_new_tokens=100, temperature=1.0, top_k=None):
        """generate mechanism: autoregressive sampling
        Args:
            idx (torch.Tensor): (B, T) input token indices
            max_new_tokens (int, optional): number of tokens to generate. Defaults to 100.
            temperature (float, optional): sampling temperature. Defaults to 1.0.
            top_k (int, optional): if specified, sample from top_k tokens only. Defaults to None.  
        """
        B, T = idx.shape
        device = idx.device
        for _ in range(max_new_tokens):
            t = idx.shape[1]
            # crop idx if longer than max_len
            if t > self.max_len:
                # crop to last max_len tokens
                idx_cond = idx[:, -self.max_len:]
            else:
                idx_cond = idx
            logits, _ = self.forward(idx_cond)
            # take the last token logits
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                topvals, _ = torch.topk(logits, top_k)
                min_top = topvals[:, -1].unsqueeze(-1) # adding dimension for broadcasting
                logits = torch.where(logits < min_top, torch.full_like(logits, -1e9), logits)

            probs = F.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1) # sample the next token randomly
            idx = torch.cat([idx, next_id], dim=1)
        return idx

    @classmethod
    def load(cls, filepath, device='cpu'):
        """Load model from checkpoint."""
        checkpoint = torch.load(filepath, map_location=device) # a dict with model hyperparameters and state_dict
        model = cls(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],
            n_head=checkpoint['n_head'],
            n_layer=checkpoint['n_layer'],
            d_ff=checkpoint['d_ff'],
            max_len=checkpoint['max_len']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        return model
    
    def save(self, filepath):
        """Save model to checkpoint."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'vocab_size': self.token_emb.num_embeddings,
            'd_model': self.d_model,
            'n_head': self.layers[0].attn.n_head if len(self.layers) > 0 else 4,  # 可能出错！
            'd_ff': self.layers[0].ffn.net[0].out_features if len(self.layers) > 0 else 512,  # 可能出错！
            'max_len': self.max_len
        }, filepath)


# -------------------------
# Toy training loop
# -------------------------

def build_dataset(corpus: str, tokenizer: SimpleTokenizer, block_size: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """char-level sliding windows -> next-token prediction
    Args:
        corpus (str): input text corpus
        tokenizer (SimpleTokenizer): tokenizer instance
        block_size (int): sequence length
    """
    enc = tokenizer.encode(corpus) # replace every char with its token id
    inputs = []
    targets = []
    for i in range(0, len(enc) - block_size):
        inputs.append(enc[i:i+block_size])
        targets.append(enc[i+1:i+block_size+1])
    X = torch.tensor(inputs, dtype=torch.long)
    Y = torch.tensor(targets, dtype=torch.long)
    return X, Y


def train_toy(model: TinyGPT, data_X, data_Y, epochs=10, batch_size=16, lr=3e-4, device='cpu'):
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    dataset_size = data_X.size(0)

    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(dataset_size) # shuffle dataset each epoch
        total_loss = 0.0
        for i in range(0, dataset_size, batch_size): # i from 0 to dataset_size with step=batch_size
            idx = perm[i:i+batch_size]
            xb = data_X[idx].to(device)
            yb = data_Y[idx].to(device)

            logits, _ = model(xb)
            logits = logits.view(-1, logits.size(-1))
            loss = F.cross_entropy(logits, yb.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # gradient clipping
            opt.step()

            total_loss += loss.item() * xb.size(0)
        avg_loss = total_loss / dataset_size
        print(f"Epoch {epoch+1}/{epochs} - loss: {avg_loss:.4f}")

def interactive_mode(model, tokenizer, device="cpu"):
    print("input 'quit' to exit")
    print("input your prompt")
    
    while True:
        user_input = input("\nUser:").strip()
        if user_input.lower() == "quit":
            break
        if not user_input:
            continue

        # tokenize input
        context = user_input
        idx = torch.tensor([tokenizer.encode(context)], dtype=torch.long).to(device)

        with torch.no_grad():
            out_ids = model.generate(
                idx,
                max_new_tokens=100,
                temperature=0.8,
                top_k=10
            )

        generated_text = tokenizer.decode(out_ids[0].tolist()[len(idx[0]):]) # decode only generated part
        print(f"Model:{generated_text}")


# -------------------------
# Example usage
# -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser() # for command line arguments
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--generate', action='store_true', help='Generate text')
    parser.add_argument('--model_path', default='paras/tiny_gpt.pth', help='Path to save/load the model')
    parser.add_argument('--tokenizer_path', default='paras/tokenizer.pth', help='Path to save/load the tokenizer')
    args = parser.parse_args()

    file_path = "corpus.txt"

    if args.train or not os.path.exists(args.model_path):
        print("=== training mode ===")
        with open(file_path, "r", encoding="utf-8") as f:
            corpus = f.read()
        corpus = corpus.strip()

        tokenizer = SimpleTokenizer(corpus)
        print(f"Vocab size: {tokenizer.vocab_size}")

        block_size = 256 # the context length for training
        X, Y = build_dataset(corpus, tokenizer, block_size)
        print(f"Dataset size: {X.size(0)} sequences")

        model = TinyGPT(vocab_size=tokenizer.vocab_size,
                        d_model=512,
                        n_head=8,
                        n_layer=8,
                        d_ff=2048,
                        max_len=block_size)

        device = torch.device(args.device)
        train_toy(model, X, Y, epochs=args.epochs, batch_size=32, lr=1e-3, device=device)

        print(f"Saving model to {args.model_path} and tokenizer to {args.tokenizer_path}")
        model.save(args.model_path)
        tokenizer.save(args.tokenizer_path)

    if args.generate:
        print("=== generation mode ===")
        device = torch.device(args.device)

        # load tokenizer
        tokenizer = SimpleTokenizer.load(args.tokenizer_path)
        model = TinyGPT.load(args.model_path, device=device)

        print("Model and tokenizer loaded successfully.")
        interactive_mode(model, tokenizer, device)
