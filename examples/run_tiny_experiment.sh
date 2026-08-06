#!/usr/bin/env bash
set -euo pipefail

python -m jabbervec.train \
  --input examples/tiny_corpus.txt \
  --outdir runs/tiny \
  --embedding-dim 64 \
  --epochs 30

python -m jabbervec.visualize \
  --checkpoint runs/tiny/model.pt \
  --out runs/tiny/char_embeddings_pca.png

python -m jabbervec.generate \
  --checkpoint runs/tiny/model.pt \
  --prompt kr \
  --length 120
