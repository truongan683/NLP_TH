"""Post-norm encoder built from Linear, softmax, LayerNorm and Dropout."""
import math
import torch
from torch import nn

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError('d_model must be divisible by n_heads')
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, return_attention=False):
        b, t, d = x.shape
        def split(layer):
            return layer(x).reshape(b, t, self.n_heads, self.d_k).transpose(1, 2)
        q, k, v = split(self.q), split(self.k), split(self.v)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if padding_mask is not None:
            if padding_mask.all(dim=1).any():
                raise ValueError('Every sequence needs at least one unmasked token')
            scores = scores.masked_fill(padding_mask[:, None, None, :], float('-inf'))
        weights = scores.softmax(dim=-1)
        context = (self.dropout(weights) @ v).transpose(1, 2).contiguous().reshape(b, t, d)
        result = self.out(context)
        if padding_mask is not None:
            result = result.masked_fill(padding_mask.unsqueeze(-1), 0)
        return (result, weights) if return_attention else result

class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
        self.norm1, self.norm2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, padding_mask=None, return_attention=False):
        attention, weights = self.attention(x, padding_mask, True)
        x = self.norm1(x + self.dropout(attention))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        if padding_mask is not None:
            x = x.masked_fill(padding_mask.unsqueeze(-1), 0)
        return (x, weights) if return_attention else x
