from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import torch
from sklearn.decomposition import PCA


def main() -> None:
    p = argparse.ArgumentParser(description="Visualize learned character embeddings with PCA.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--components", type=int, default=2, choices=[2, 3])
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    chars = ckpt["vocab"]["chars"]
    emb = ckpt["model_state"]["embedding.weight"].numpy()
    coords = PCA(n_components=args.components).fit_transform(emb)

    fig = plt.figure(figsize=(8, 6))
    if args.components == 3:
        ax = fig.add_subplot(111, projection="3d")
        ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2])
        for ch, (x, y, z) in zip(chars, coords):
            ax.text(x, y, z, repr(ch)[1:-1])
    else:
        ax = fig.add_subplot(111)
        ax.scatter(coords[:, 0], coords[:, 1])
        for ch, (x, y) in zip(chars, coords):
            ax.text(x, y, repr(ch)[1:-1])
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    ax.set_title("JabberVec character embeddings")
    fig.tight_layout()
    fig.savefig(args.out, dpi=160)
    print(args.out)


if __name__ == "__main__":
    main()
