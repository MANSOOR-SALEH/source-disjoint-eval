# Controlling test-partition contamination in source-disjoint dialect identification

Implementation for the IEEE Signal Processing Letters submission
*"Controlling Test-Partition Contamination in Source-Disjoint Dialect
Identification."*

Source-disjoint evaluation holds out whole recording sources to control
provenance. This repository implements the protocol and, crucially, the
**test-partition restriction** that keeps a training-source-matched comparison
honest, plus two manifest-level feasibility checks that run before any model is
trained.

## What this implements

- **The correction** (`src/folds.py`) — when training sources are subsampled to
  match the source-disjoint count, the test partition is restricted to the
  retained sources, so clips from discarded sources are not left in test as
  spurious source-disjoint items.
- **Layer probing** (`src/probe.py`) — per-layer profiles under three protocols
  (random, speaker-disjoint, source-disjoint), with both the convention layer
  (current practice) and nested selection (test condition never consulted).
- **Across-draw uncertainty** (`src/subsample_seeds.py`) — repeats the whole
  matched pipeline under several source subsamples and reports the drop with its
  across-draw mean and standard deviation.
- **Two preconditions**, computable from the manifest alone:
  `src/fold_coverage.py` (no fold starves a class of training data) and
  `src/fold_class_coverage.py` (every held-out fold is class-complete in test).
- **Synthetic leak test** (`tests/test_pipeline.py`).

## Manifest format

Each corpus is a CSV with columns:

```
utt_id, path, label, speaker_id, source_id
```

`source_id` is the provenance unit (broadcast programme, recording session,
device, or site). `label` is the dialect class.

## Quick start

```bash
python -m venv .venv
. .venv/bin/activate                 # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# 1. extract frozen-encoder features (cached under features/)
python src/extract.py --manifest manifests/aydid_clean.csv --encoder wavlm --out features/aydid.wavlm.npz

# 2. build corrected folds (size + source matching + test restriction)
python src/folds.py --manifest manifests/aydid_clean.csv --out folds/aydid_seed0.json \
    --n-folds 10 --n-repeats 20 --seed 0 --match-train-sources

# 3. probe
python src/probe.py --features features/aydid.wavlm.npz --manifest manifests/aydid_clean.csv \
    --folds folds/aydid_seed0.json --out results_fix/aydid.wavlm.json \
    --fixed-layer 12 --layer-stride 2 --no-linear --epochs 30 --seed 0

# 4. across-draw uncertainty (Table II)
python src/subsample_seeds.py --seeds 0 1 2 3 4 --reference results_fix
```

Full corpus list, exact commands, and the mapping from every reported number to
its output file are in [`docs/REPRODUCE.md`](docs/REPRODUCE.md). Feature format,
pooling, and audio normalisation are in [`docs/PIPELINE.md`](docs/PIPELINE.md).

## Data

The corpora are third-party: AYDID (Hugging Face Datasets,
doi:10.57967/hf/9014), SADA (Alharbi et al., ICASSP 2024), and KeSpeech (Tang et
al., NeurIPS D&B 2021). Manifests reference clips by `utt_id`; see
`docs/REPRODUCE.md` for how to obtain each corpus.

## Citation

See `CITATION.cff` (GitHub renders a "Cite this repository" button).

## License

MIT — see `LICENSE`.
