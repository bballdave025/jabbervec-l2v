from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import CharSequenceDataset, Vocab, load_text
from .model import TinyCharLM


def train(args: argparse.Namespace) -> None:
    text = load_text(args.input, lowercase=not args.keep_case)
    vocab = Vocab.from_text(text)
    ids = vocab.encode(text)
    dataset = CharSequenceDataset(ids, block_size=args.block_size)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    model = TinyCharLM(len(vocab.chars), args.embedding_dim, args.hidden_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        for x, y in loader:
            opt.zero_grad(set_to_none=True)
            logits, _ = model(x)
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
        print(f"epoch {epoch:03d} loss={total_loss / max(1, len(loader)):.4f}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "vocab": {"chars": vocab.chars},
            "config": {
                "embedding_dim": args.embedding_dim,
                "hidden_dim": args.hidden_dim,
                "block_size": args.block_size,
            },
        },
        outdir / "model.pt",
    )
    (outdir / "vocab.json").write_text(json.dumps({"chars": vocab.chars}, indent=2), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train a tiny JabberVec character language model.")
    p.add_argument("--input", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--embedding-dim", type=int, default=64)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--block-size", type=int, default=32)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--keep-case", action="store_true")
    return p


def main() -> None:
    train(build_parser().parse_args())


if __name__ == "__main__":
    main()
