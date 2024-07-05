import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model:int, vocab_size:int):
        super().__init__()
        self.d_model = d_model # dimension of each token
        self.vocab_size = vocab_size # 
        self.embedding = nn.Embedding(vocab_size, d_model)

        def forward(self, x):
            return self.embedding(x) * math.sqrt(self.d_model)


# Positional encoding
class PositionalEncoding(nn.Module):
    def __init__(self, d_model:int, seq_len: int, dropout:float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.droptout = nn.Dropout(dropout)     
