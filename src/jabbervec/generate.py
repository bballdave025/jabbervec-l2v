from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F

from .data import Vocab
from .model import TinyCharLM


def load_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu")
    chars = ckpt["vocab"]["chars"]
    vocab = Vocab(chars=chars, stoi={c: i for i, c in enumerate(chars)}, itos={i: c for i, c in enumerate(chars)})
    cfg = ckpt["config"]
    model = TinyCharLM(len(chars), cfg["embedding_dim"], cfg["hidden_dim"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, vocab


@torch.no_grad()
def generate_text(model: TinyCharLM, vocab: Vocab, prompt: str, length: int, temperature: float) -> str:
    ids = vocab.encode(prompt.lower()) or [0]
    x = torch.tensor([ids], dtype=torch.long)
    hidden = None
    out_ids = ids[:]
    for _ in range(length):
        logits, hidden = model(x[:, -1:], hidden)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = F.softmax(logits, dim=-1)
        nxt = int(torch.multinomial(probs, num_samples=1).item())
        out_ids.append(nxt)
        x = torch.tensor([[nxt]], dtype=torch.long)
    return vocab.decode(out_ids)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate text from a tiny JabberVec model.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt", default="j")
    p.add_argument("--length", type=int, default=120)
    p.add_argument("--temperature", type=float, default=0.9)
    args = p.parse_args()
    model, vocab = load_checkpoint(args.checkpoint)
    print(generate_text(model, vocab, args.prompt, args.length, args.temperature))


if __name__ == "__main__":
    main()
