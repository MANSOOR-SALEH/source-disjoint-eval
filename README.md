# Reliable Source-Disjoint Evaluation for Dialect Identification

Implementation and reproducibility materials for:

**“Reliable Source-Disjoint Evaluation for Dialect Identification:  
Partition Matching, Sampling Variability, and Fold Feasibility”**

This repository accompanies a methodological study of grouped evaluation
protocols for dialect identification.

The work is not intended to introduce a new dialect classifier. Instead, it
examines whether comparisons between speaker-group-disjoint and
source-disjoint evaluation are constructed in a way that supports their
intended interpretation.

The repository focuses on three issues:

1. **Partition consistency after source-count matching**
2. **Matched-draw variability**
3. **Manifest-level fold feasibility**

---

## 1. Evaluation setting

Each utterance is represented by:

```text
utt_id, path, label, speaker_id, source_id
