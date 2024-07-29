import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # dimension of each token
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

        def forward(self, x):
            return self.embedding(x) * math.sqrt(self.d_model)


#! ****Positional encoding****
# ? seq_len: Maximun length of the sentence, because we need to create one vector for each position. if your input sequence has 50 tokens, seq_len would be 50 --> of each training batch.
    # Batch 1: Tokens 1 to 50
    # Batch 2: Tokens 51 to 100
    # Batch 3: Tokens 101 to 150
    # Batch 4: Tokens 151 to 200

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.droptout = nn.Dropout(dropout)

        # create a matrix of (seq_len, d_model)
        pe = torch.zeros(seq_len, d_model)

        # create a matrix of (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqeeze(1)

        # denominator
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (- math.log(1000.0)/d_model))
        # applying

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe.pe.unsqueeze(0)  # (1, seq_len, d_model)

        self.register_buffer('pe', pe)  # save in a file

    def forward(self, x):
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(1))  # multiplier
        self.bias = nn.Parameter(torch.zeros(1))  # added

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedFowardBlock(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def foward(self, x):
        # (Batch, seq_len, d_model) --> (Batch, seq_len, d_ff) --> (Batch, seq_len, d_model)
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


# ? @params h: number of self attention blocks we want
class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model: int, h: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.h = h
        assert d_model % h == 0, "d_model is not divisible by h"

        self.d_k = d_model // h

        self.w_q = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask):
