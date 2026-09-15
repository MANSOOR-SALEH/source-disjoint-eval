# Reproducing the reported numbers

This document maps every number in the Letter to the exact command and output
file that produces it. Results directories are `.gitignore`d because they are
re-derivable from the manifests and cached features; the steps below regenerate
them.

## 0. Environment

```bash
python -m venv .venv
. .venv/bin/activate            # Windows: .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

GPU is optional. On a 6 GB card, close other GPU applications and, if you hit a
CUDA out-of-memory error, lower the probe batch size (`bs` in `src/probe.py`).

## 1. Corpora and manifests

Three corpora, each described by a CSV manifest with columns
`utt_id, path, label, speaker_id, source_id`:

- **AYDID** — 17,460 clips, 7 Yemeni sub-dialects, 40 broadcast programmes;
  sources nested in the label. `manifests/aydid_clean.csv`.
- **SADA** — three-dialect subset, 7,499 clips, 63 programmes (52 multi-dialect);
  sources crossed with the label. `manifests/sada.csv`.
- **KeSpeech** — Mandarin sub-dialects, used only for the class-completeness
  example of Sec. V-B. `manifests/kespeech.csv`.

Audio is 16 kHz mono PCM, clips restricted to [1, 30] s. See `docs/PIPELINE.md`
for feature extraction, pooling, and normalisation.

## 2. Feature extraction (cached)

```bash
python src/extract.py --manifest manifests/aydid_clean.csv --encoder wavlm --out features/aydid.wavlm.npz
# ... repeat for each (corpus, encoder) pair; 7 encoders x 3 corpora
```

Encoders: `wavlm, xlsr, mhubert, mms, whisper, w2vbert, artst`. Features are
cached under `features/` so the probe steps below never re-extract.

## 3. Fold construction — the correction

`src/folds.py` builds the three protocols with **size matching**, **training-source
matching**, and **test-partition restriction** (the correction of Sec. III). The
`*_fix` / `*_seed{N}` fold files carry the `test_srcmatched` key that restricts
the test partition to retained sources; fold files without it reproduce the
uncorrected control.

```bash
# corrected folds, seed 0 (== the *_fix folds)
python src/folds.py --manifest manifests/aydid_clean.csv --out folds/aydid_seed0.json --n-folds 10 --n-repeats 20 --seed 0 --match-train-sources
python src/folds.py --manifest manifests/sada.csv        --out folds/sada_seed0.json  --n-folds 10 --seed 0 --match-train-sources
# repeat for --seed 1 2 3 4
```

## 4. Probing

`src/probe.py` scores every layer under every protocol and records, at the
convention layer, `spk_to_src_drop` (speaker-disjoint minus source-disjoint
macro-F1) and the nested-selection drop.

```bash
python src/probe.py --features features/aydid.wavlm.npz --manifest manifests/aydid_clean.csv \
    --folds folds/aydid_seed0.json --out results_fix/aydid.wavlm.json \
    --fixed-layer 12 --layer-stride 2 --no-linear --epochs 30 --seed 0
```

Classifier configuration (identical for every encoder, corpus and fold):
256-unit 1-hidden-layer MLP, ReLU, dropout 0.3, AdamW (lr 1e-3, weight decay
1e-2), batch 128, 30 epochs, cross-entropy, per-feature standardisation. See the
supplement, Table III.

## 5. Number-to-file map

**Everything below is read from the JSON field
`convention_layer.spk_to_src_drop` unless noted.**

| Reported in paper | Quantity | Where it comes from |
|---|---|---|
| **Table I, uncorrected column** (AYDID +0.008, SADA +0.169) | seed-0 WavLM drop, **no** test restriction | `results_seed0/{corpus}.wavlm.json` |
| **Table I, corrected column** (AYDID +0.014, SADA +0.208) | seed-0 WavLM drop, **with** test restriction | `results_fix/{corpus}.wavlm.json` |
| **Table I, stranded-clip fractions** | share of test clips whose source is dropped | `src/stats_report.py` (stranded-fraction report) |
| **Table II, corrected five-draw drop** (AYDID 0.024 (0.019), SADA 0.210 (0.020)) | corrected drop, per seed, averaged over 7 encoders, mean/SD over 5 seeds | `results_fix/` (seed 0) + `results_fix_seed{1,2,3,4}/`, aggregated by `src/subsample_seeds.py` |
| **§VI nested-vs-convention** (SADA +0.198 vs +0.172; AYDID +0.0099 vs −0.0014) | convention-layer drop vs nested-selection drop | `convention_layer.spk_to_src_drop` and `nested_drop.drop`, seed-0 corrected files |
| **§V-A** ρ_min (AYDID ≥0.60, SADA ≥0.47; nested-SADA 0.563) | per-fold class retention | `src/fold_coverage.py` |
| **§V-B** KeSpeech class-completeness, 2 ≤ f ≤ 5 | incidence-matrix analysis, no training | `src/fold_class_coverage.py` |

> **Important:** Table II is the **corrected** drop. It is aggregated from
> `results_fix` and `results_fix_seed1..4` (folds with `test_srcmatched`), **not**
> from `results_seed*`, which are the uncorrected diagnostic. Regenerating Table II
> from `results_seed*` will give the uncorrected five-draw numbers, which differ in
> construction from what the paper reports.

## 6. Aggregate

```bash
# corrected five-draw mean and SD (Table II)
python src/subsample_seeds.py --seeds 0 1 2 3 4 --reference results_fix

# intervals, omnibus test, corpus comparison (stats used in the text)
python src/stats_report.py
```

## 7. Preconditions (no training required)

```bash
python src/fold_coverage.py       --manifest manifests/aydid_clean.csv   # class retention in training
python src/fold_class_coverage.py --manifest manifests/kespeech.csv      # class completeness in test
```

## 8. Synthetic check

```bash
pytest tests/test_pipeline.py     # injects a known source->label leak and verifies detection
```
