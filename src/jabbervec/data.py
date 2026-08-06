from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import Dataset


@dataclass(frozen=True)
class Vocab:
    chars: list[str]
    stoi: dict[str, int]
    itos: dict[int, str]

    @classmethod
    def from_text(cls, text: str) -> "Vocab":
        chars = sorted(set(text))
        return cls(chars=chars, stoi={c: i for i, c in enumerate(chars)}, itos={i: c for i, c in enumerate(chars)})

    def encode(self, text: str) -> list[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.itos[int(i)] for i in ids)


class CharSequenceDataset(Dataset):
    def __init__(self, ids: list[int], block_size: int):
        if len(ids) <= block_size:
            raise ValueError("Text is too short for the selected block_size.")
        self.ids = torch.tensor(ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        return len(self.ids) - self.block_size

    def __getitem__(self, idx: int):
        x = self.ids[idx : idx + self.block_size]
        y = self.ids[idx + 1 : idx + self.block_size + 1]
        return x, y


def load_text(path: str | Path, lowercase: bool = True) -> str:
    text = Path(path).read_text(encoding="utf-8")
    if lowercase:
        text = text.lower()
    return text
