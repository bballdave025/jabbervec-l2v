# Context / IP Notes: JabberVec-l2v

## Origin

This project came from MLU/linear algebra notes about embeddings for NLP and LLMs.

Two narrow questions started the experiment:

1. Can an embedding be an encoding, and can it be lossless?
2. Can embedding dimension `n` be greater than vocabulary size `V`, especially in character-level models?

The name **JabberVec** comes from Lewis Carroll-style literary nonsense: invented words that are not standard English, yet feel English-shaped. The `l2v` suffix keeps the Letter2Vec learning goal visible.

## Working conceptual answers

### Embedding as encoding

An embedding is an encoding in the general sense: it maps an input into a representation. A learned embedding is special because its vector geometry is trained to become useful for prediction.

### Lossless embeddings

A finite embedding can preserve identity if it is injective over the vocabulary. If every row of an embedding matrix is unique and the matrix is retained, exact row lookup can recover the token. But most learned embeddings are not trained for lossless recovery. They are trained for useful geometry.

### `n > V`

For a lowercase character model:

```text
V = 26
n = 64
E ∈ R^{26×64}
```

This is valid. It is unusual for mainstream subword LLMs because their vocabularies are large, but common or at least natural in toy/small-vocabulary character experiments.

## Personal learning intent

This is not a state-of-the-art model. It is a concept-solidifying laboratory.

Main learning targets:

- embedding matrix as learned lookup table,
- one-hot versus dense learned vectors,
- vocabulary size versus embedding dimension,
- PCA/SVD as ways to inspect vector spaces,
- low-rank approximation versus true/numerical/effective rank,
- distributional meaning under intentionally playful language transformations.

## IP boundary

Use public-domain, permissively licensed, or user-created corpora only. Do not include employer-confidential data, proprietary evaluation material, or workplace-specific notes.

## Good first milestone

Run the tiny experiment and generate one PCA plot. Stop there before expanding.
