# source-disjoint-eval

Code and manifests for the paper:

> **Reliable Source-Disjoint Evaluation for Dialect Identification: Sampling
> Variability, Partition Matching, and Fold Feasibility** — Pattern Recognition
> Letters (under review).

This is a **methods / evaluation-protocol** repository, not a recognizer. It
implements *partition-consistent matched grouped evaluation*: matching a
conventional split to a source-disjoint one in training-set size and source
count, restricting the comparison arm's **test** partition to the retained
sources, repeating over matched draws, and running two pre-training feasibility
checks (per-fold class retention and class-complete test folds). Frozen encoders
are used only as fixed probes.

## What is here

```
manifests/        fold manifests and retained-source sets per corpus/draw
src/              pipeline (feature extraction, folds, probe, draws, reports)
docs/             step-by-step runbooks
requirements.txt  Python dependencies
```

## What is NOT here

- **Raw corpora and audio/image files.** AYDID, SADA, KeSpeech and Camelyon17 are
  obtained from their own sources under their own licences (see below); nothing
  is redistributed here.
- **Cached features** (`features/*.npz`). These are large and are regenerated
  from the public corpora with the extraction scripts.

## Repository layout

```
src/
  extract.py              frozen speech-encoder features -> features/<corpus>.<encoder>.npz
  extract_vision.py       frozen ResNet-50 features for Camelyon17 (image analogue)
  encoders.py             encoder loaders / masked mean+std pooling
  folds.py                grouped-fold construction and source-count matching
  probe.py                256-unit MLP / linear probe, test-blind nested layer selection
  run_draws.py            end-to-end matched draws: corrected + uncorrected arms
  delta_report.py         paired delta across draws, bootstrap CIs, variance decomposition
  diagnose_folds.py       KeSpeech incidence-matrix feasibility (200 random partitions per f)
  diag_table.py           per-corpus q / seen-unseen gap diagnostics from saved predictions
  make_camelyon_manifest.py   WILDS Camelyon17 metadata -> balanced manifest
docs/
  RUN_DRAWS.md            how to reproduce the matched-draw tables
  CAMELYON_RUNBOOK.md     how to add the Camelyon17 cross-domain result
```

## Reproducing the results

Install:

```bash
pip install -r requirements.txt
```

Speech corpora (per corpus, per encoder):

```bash
# 1. features (frozen encoder, no fine-tuning)
python src/extract.py --manifest manifests/<corpus>.csv \
    --encoder wavlm --out features/<corpus>.wavlm.npz

# 2. matched draws: corrected (test restricted) + uncorrected (training-only)
python src/run_draws.py --corpus <corpus> --manifest manifests/<corpus>.csv \
    --features-dir features --encoders wavlm xlsr mhubert mms whisper w2vbert artst \
    --n-folds <f> --tag _main --seeds 0 1 2 3 4

# 3. paired delta, bootstrap CIs, variance decomposition
#    (run_draws.py calls delta_report.py; outputs results_*/<corpus>_*_delta_report.json)
```

KeSpeech feasibility (incidence-matrix check, no training):

```bash
python src/diagnose_folds.py --manifest manifests/kespeech.csv --max-folds 6
```

Camelyon17 cross-domain result: see `docs/CAMELYON_RUNBOOK.md`.

Fold sampling uses a resampling budget of 50 attempts per draw
(`max_tries` in `src/folds.py`); all reported contrasts use randomly sampled
class-complete folds, with direct construction as the stated fallback.

## Data sources

| Corpus | Where | Licence |
|--------|-------|---------|
| AYDID | Hugging Face Datasets, doi:10.57967/hf/9014 | see dataset card |
| SADA | Saudi Audio Dataset for Arabic (ICASSP 2024) | see source |
| KeSpeech | NeurIPS 2021 Datasets & Benchmarks | see source |
| Camelyon17 | WILDS (`get_dataset(dataset='camelyon17', download=True)`) | CC0 (CAMELYON17) |

## Citation

Add the BibTeX entry once the paper has a DOI / volume. Placeholder:

```bibtex
@article{bamahel_source_disjoint,
  title   = {Reliable Source-Disjoint Evaluation for Dialect Identification:
             Sampling Variability, Partition Matching, and Fold Feasibility},
  author  = {Ba Mahel, Mansoor S. M. and Wei, Jianguo and Yue, Xianghu and
             Awn, Norah Saeed and Bamahel, Abdulaziz S.},
  journal = {Pattern Recognition Letters},
  year    = {2026},
  note    = {under review}
}
```

## Archived release

A tagged release is archived on Zenodo (doi:10.5281/zenodo.22784722).

## Licence

Code is released under the MIT Licence (see `LICENSE`). The corpora remain under
their respective licences and are not redistributed here.
