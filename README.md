# JabberVec-l2v

*Looking through the embedding glass.*

**JabberVec-l2v** is a playful Letter2Vec laboratory inspired by Lewis Carroll, especially the tradition of **literary nonsense**: words that do not quite exist, yet somehow look like they could.

The serious learning goal is small and concrete:

> Train tiny character-level models, inspect their embedding matrices, and ask what geometry appears when letters learn from English-shaped text.

The playful question is:

> If every plausible English-looking word were allowed to exist, what would the vector space look like?

Those not literature-ly inclined may read `l2v` as **letter-to-vec**. Those willing to step through the looking glass can read the project as **JabberVec**: a place where *frabjous*, *krumpets*, and other not-quite-English strings can be studied as geometry.

Hidden project motto: like a mischievous cousin of Feynman's path-integral imagination, this repo explores a world where anything English-like that can happen, does happen — then asks which paths cluster together.

## Two starting questions

### 1. Can an embedding be an encoding? Can it be lossless?

Yes. An embedding is a kind of encoding: it maps discrete things into vectors.

A learned embedding usually differs from a plain ID or one-hot encoding because its geometry is trained to be useful. It is not just saying “this is token 17.” It is trying to make token 17 land somewhere useful relative to other tokens.

An embedding can be lossless in principle if the mapping is injective over the represented vocabulary. For a finite vocabulary, a learned embedding matrix can preserve token identity if every row is unique and the matrix is retained. In practice, neural embeddings are usually optimized for usefulness, not perfect invertibility.

### 2. Can embedding dimension `n` be greater than vocabulary size `V`?

Yes, especially for character-level models.

For lowercase English letters:

```text
V = 26
n = 64
E ∈ R^{26×64}
```

Each character gets a 64-dimensional learned vector. That is not forbidden. It can be useful because the model is not merely identifying letters; it is learning representations that support sequence prediction.

For mainstream word/subword LLMs, vocabulary size is often much larger than embedding dimension. But with small vocabularies — characters, bytes, toy domains, tiny symbolic systems — `n > V` is straightforward.

## What this repo does

This repo gives you a small, inspectable character-level language modeling experiment:

1. Load a text corpus.
2. Build a character vocabulary.
3. Train a tiny next-character model with a learned embedding matrix.
4. Generate English-ish strings.
5. Visualize the learned character embeddings with PCA.
6. Later, compare full-rank and low-rank approximations of the embedding matrix.

## Directory structure

```text
jabbervec-l2v/
├── README.md
├── LICENSE
├── pyproject.toml
├── dev_notes/
│   ├── context.md
│   └── admin_notes/
│       └── decisions.md
├── examples/
│   ├── tiny_corpus.txt
│   └── run_tiny_experiment.sh
├── scripts/
│   └── scaffold.py
├── src/
│   └── jabbervec/
│       ├── __init__.py
│       ├── data.py
│       ├── model.py
│       ├── train.py
│       ├── visualize.py
│       ├── generate.py
│       └── low_rank.py
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── tests/
└── runs/
```

## Install

From the repo root:

### **bash**

```bash
python -m venv .venv_jabber
. .venv_jabber/bin/activate
pip install -e .
```

### **Windows PowerShell**

```Windows PowerShell
python -m venv .venv_jabber
.\.venv_jabber\Scripts\Activate.ps1
pip install -e .
```

For CMD instead of PowerShell, replace `Activate.ps1` with `activate.bat`

If PowerShell blocks `Activate.ps1`, check the policy with `Get-ExecutionPolicy -List`. 
For this PowerShell session only, allow local scripts with 
`Set-ExecutionPolicy -Scope Process RemoteSigned`, then retry. The setting ends when 
you close the session.

### Note for any install

PyTorch is the heaviest dependency. If install fails, install the appropriate PyTorch 
wheel for your system, then rerun `pip install -e .`. If that doesn't make sense,
ask you AI Assistant of choice or use Google Search with AI Mode.

## Run the tiny experiment

### UNIX/Linux(R) `bash`

```bash
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
```

Or:

```bash
bash examples/run_tiny_experiment.sh
```

### Windows PowerShell

```Windows PowerShell
python -m jabbervec.train `
  --input examples/tiny_corpus.txt `
  --outdir runs/tiny `
  --embedding-dim 64 `
  --epochs 30

python -m jabbervec.visualize `
  --checkpoint runs/tiny/model.pt `
  --out runs/tiny/char_embeddings_pca.png

python -m jabbervec.generate `
  --checkpoint runs/tiny/model.pt `
  --prompt kr `
  --length 120
```

or

```Windows PowerShell
.\examples\run_tiny_experiment.ps1
```

If PowerShell blocks the experiment script (or if it earlier blocked `Activate.ps1`,) check 
the policy with `Get-ExecutionPolicy -List`. For this PowerShell session only, allow local 
scripts with `Set-ExecutionPolicy -Scope Process RemoteSigned`, then retry. The setting 
ends when you close the session.

## Dataset suggestions

### 1. Dictionary / word list

Good for spelling shapes and pseudo-word generation. Less good for natural sentence context.

Possible local source:

```bash
/usr/share/dict/words
```

A dictionary asks:

> What strings look like English words?

### 2. Lewis Carroll starter text

Use public-domain text such as *Jabberwocky* or *Through the Looking-Glass*. This is thematically perfect but small, so expect toy behavior.

A Carroll starter corpus asks:

> Can nonsense words still teach structure?

### 3. Project Gutenberg-style public domain corpora

Good for richer character transitions and style experiments.

Possible authors:

- Lewis Carroll
- Mark Twain
- Shakespeare
- Jane Austen
- Charles Dickens

A literary corpus asks:

> What character sequences appear in stylized English prose?

### 4. Mixed modern-ish corpus

Good for broader, less literary English.

Possible directions:

- Simple English Wikipedia dump
- WikiText-style datasets
- public domain or permissively licensed conversational corpora
- your own small mixed corpus assembled from allowed texts

A mixed corpus asks:

> What strings look plausible in broad modern English?

## Future experiments

### A. `n > V` embedding sweep

Train with:

```text
embedding_dim = 8, 16, 32, 64, 128
```

Compare validation loss, generated strings, and PCA plots.

### B. Dictionary vs. large corpus

Train once on a dictionary and once on a large mixed corpus. Compare whether the model learns word-shape only or broader character-flow behavior.

### C. Dimensionality reduction for visualization

Use PCA to project learned embeddings from `D` dimensions down to 2D or 3D.

This is not expected to improve model performance. It is mainly for understanding the geometry. The embedding space is probably not sparse in the way Latent Forge rating matrices might be sparse, so dimensionality reduction here is more diagnostic than obviously beneficial.

### D. Low-rank approximation

Take the learned embedding matrix:

```text
E ∈ R^{V×D}
```

Use truncated SVD to form a rank-`k` approximation:

```text
E_k ≈ E
```

Then ask:

- How low can `k` go before generated strings degrade?
- Do nearest-neighbor relationships among characters change?
- Does vowel/consonant structure survive?
- Are the singular values concentrated or flat?
- What is the **effective rank**, not merely the formal floating-point rank?

Important anti-perfectionism note: do not start by trying to “find the real rank.” First try low-rank approximations and see what breaks.

### E. Arbitrary English-to-JabberVec mapping

This is the strange future research direction.

Create an arbitrary length-preserving mapping from English words to JabberVec words, then transform a corpus through that mapping. Train embeddings on the transformed text and visualize the resulting geometry.

Questions:

- Do hidden source categories cluster after arbitrary renaming?
- If the corpus had many royalty concepts, do the corresponding JabberVec tokens cluster even though their surface forms are nonsense?
- Can we infer which source corpus or genre produced the geometry?
- What information survives when surface identity is scrambled but distributional context remains?

This is probably not a job-search portfolio priority, but it is an excellent “what does distributional meaning really mean?” experiment.

## Anti-perfectionism path

The smallest useful version:

1. Run the tiny corpus.
2. Generate a few strings.
3. Make one PCA plot.
4. Ask: “What does this make visible?”

Everything else is optional.
