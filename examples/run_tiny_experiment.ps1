# Run from any directory with the project's Python environment activated.
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent $PSScriptRoot
Push-Location $repoRoot
try {
    python -m jabbervec.train `
        --input examples/tiny_corpus.txt `
        --outdir runs/tiny `
        --embedding-dim 64 `
        --epochs 30
    if ($LASTEXITCODE -ne 0) { throw "Training failed (exit code $LASTEXITCODE)." }

    python -m jabbervec.visualize `
        --checkpoint runs/tiny/model.pt `
        --out runs/tiny/char_embeddings_pca.png
    if ($LASTEXITCODE -ne 0) { throw "Visualization failed (exit code $LASTEXITCODE)." }

    python -m jabbervec.generate `
        --checkpoint runs/tiny/model.pt `
        --prompt kr `
        --length 120
    if ($LASTEXITCODE -ne 0) { throw "Generation failed (exit code $LASTEXITCODE)." }
}
finally {
    Pop-Location
}
