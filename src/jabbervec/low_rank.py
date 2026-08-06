from __future__ import annotations

import argparse

import numpy as np
import torch


def main() -> None:
    p = argparse.ArgumentParser(description="Inspect singular values of a learned embedding matrix.")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--k", type=int, default=8, help="rank for reconstruction error report")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    emb = ckpt["model_state"]["embedding.weight"].numpy()
    u, s, vt = np.linalg.svd(emb, full_matrices=False)
    k = min(args.k, len(s))
    approx = (u[:, :k] * s[:k]) @ vt[:k, :]
    rel_err = np.linalg.norm(emb - approx) / max(np.linalg.norm(emb), 1e-12)

    print("singular_values:")
    print(s)
    print(f"rank_k={k} relative_frobenius_error={rel_err:.6f}")
    print("Note: formal floating-point rank is less interesting than effective rank / degradation behavior.")


if __name__ == "__main__":
    main()
