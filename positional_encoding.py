"""Sinusoidal position information, with no trainable parameters."""
import math
import torch
from torch import nn

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_length=4096):
        super().__init__()
        position = torch.arange(max_length).float().unsqueeze(1)
        frequency = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_length, d_model)
        pe[:, 0::2] = torch.sin(position * frequency)
        pe[:, 1::2] = torch.cos(position * frequency[:d_model // 2])
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        if x.shape[1] > self.pe.shape[1]:
            raise ValueError('Sequence exceeds positional encoding capacity')
        return x + self.pe[:, :x.shape[1]].to(dtype=x.dtype)
