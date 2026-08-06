from __future__ import annotations

import torch
from torch import nn


class TinyCharLM(nn.Module):
    """A tiny character LM with a visible learned embedding matrix.

    This is intentionally simple: embedding -> GRU -> linear head.
    The embedding matrix is the main object of inspection.
    """

    def __init__(self, vocab_size: int, embedding_dim: int = 64, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.head = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x: torch.Tensor, hidden=None):
        z = self.embedding(x)
        out, hidden = self.rnn(z, hidden)
        logits = self.head(out)
        return logits, hidden
