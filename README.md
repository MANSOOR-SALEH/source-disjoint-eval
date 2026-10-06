# Reliable source-disjoint evaluation for dialect identification

Code, fold manifests, and analysis artifacts for the methods study
**"Reliable Source-Disjoint Evaluation for Dialect Identification: Partition
Matching, Sampling Variability, and Fold Feasibility"** (under review).

The study is a methodological contribution on how to construct and interpret
matched comparisons between **speaker-group-disjoint** and **source-disjoint**
evaluation when samples are grouped by recording provenance (program, session,
device, or site). It identifies three failure modes and gives checks that
remove them:

1. **Partition inconsistency.** Matching training-source counts in the
   comparison arm, but restricting only the *training* partition, leaves test
   clips whose source is no longer represented in training — silently turning
   part of the seen-source arm into a source-disjoint one. The fix restricts
   the test partition to the retained sources (Algorithm 1, Step 5).
2. **Matched-draw variability.** A single retained-source draw is unstable
   where subsampling is large; the protocol contrast should be reported as a
   mean over several matched draws with its across-draw SD.
3. **Fold feasibility.** Two manifest-level checks, computable before training:
   every training fold must retain adequate per-class data, and every test fold
   must be class-complete when macro-F1 is the metric.

All measurements are averaged over seven frozen speech encoders used only as
measurement probes, not as methods under comparison.

## Algorithm 1 — source-count matching with partition-consistent test restriction

```
Input : corpus manifest D = {(x_i, y_i, u_i, v_i)}; target training-source
        count m (from the source-disjoint arm); comparison arm a in
        {random, speaker-group-disjoint}
Output: matched arm whose partitions satisfy V_test  ⊆  V_train

1  Build candidate grouped folds: group by u_i (speaker-group-disjoint),
   label-stratified (random), and by v_i (source-disjoint).
2  repeat
3      Sample a retained-source set V_keep ⊆ V with |V_keep| = m at random.
4      D_train <- { i in candidate-train(a) : v_i in V_keep }   # source match
5      D_test  <- { i in candidate-test(a)  : v_i in V_keep }   # test restriction
       # Step 5 is omitted by the training-only control.
6      Size-match D_train to the smallest arm by label-stratified subsampling,
       keeping >= 1 sample per source in V_keep.
7  until every class is present in both D_train and D_test
8  Assert |V_train| = m and V_test ⊆ V_train; for the source-disjoint arm,
   V_train ∩ V_test = ∅ by construction.
9  Train the probe on D_train; score on D_test.
```

The corrected implementation differs from the training-only control by **Step 5
only**, so it can be applied post hoc to saved per-clip predictions without
retraining, provided every class remains present in the restricted test set.

## Repository layout

```
src/         feature extraction, fold construction, probe, matched-draw sweep,
             coverage and class-completeness checks
manifests/   fold manifests and retained-source sets per corpus and draw
docs/        REPRODUCE.md (end-to-end run) and PIPELINE.md (stage details)
tests/       pipeline sanity checks
requirements.txt   pinned dependencies
CITATION.cff       citation metadata
```

## Reproducing the results

```bash
pip install -r requirements.txt
# 1. extract frozen-encoder embeddings (per corpus, per encoder)
python -m src.extract      --corpus AYDID  --encoder wavlm
# 2. build matched grouped folds and retained-source draws
python -m src.folds        --corpus AYDID  --seeds 0 1 2 3 4
# 3. run the corrected and uncorrected sweeps over 5 draws x 7 encoders
python -m src.run_corrected_seeds --corpus AYDID
# 4. feasibility audits (class retention, class-complete test folds)
python -m src.fold_class_coverage --corpus AYDID
```

See `docs/REPRODUCE.md` for the exact fold manifests, retained-source sets,
seeds, environment metadata, and commands for every table and figure.

## Data

The study does not redistribute third-party audio. It uses AYDID, SADA, and
KeSpeech under their respective licences; only fold manifests and analysis
artifacts are distributed here.

## Citation

If you use this code or the fold manifests, please cite the paper (see
`CITATION.cff`). Archived release: Zenodo
[10.5281/zenodo.22784722](https://doi.org/10.5281/zenodo.22784722).

## Licence

MIT (see `LICENSE`).
