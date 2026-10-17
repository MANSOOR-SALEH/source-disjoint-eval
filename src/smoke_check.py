"""
Sanity-check extracted features before trusting them.

    python src/smoke_check.py features/smoke_*.npz

Shape being right proves nothing: an adapter that reads the wrong tensor, or
returns the same hidden state for every layer, still produces a correctly shaped
array. This checks the three ways that happens.

  1. finite + non-degenerate: every layer must vary across utterances. A layer
     whose across-utterance SD is ~0 is a constant, i.e. nothing was read.
  2. layers must differ: cosine similarity between the first and last layer's
     mean vector should be well below 1.0. Near 1.0 means output_hidden_states
     returned copies, or only one layer is being indexed.
  3. mean and std halves must differ: the pooled vector is [mean || std]. If the
     two halves match, masked pooling collapsed.
"""

import sys
import glob

import numpy as np


def check(path):
    d = np.load(path, allow_pickle=True)
    e = d["emb"].astype(np.float32)            # [N, L, 2D]
    N, L, twoD = e.shape
    D = twoD // 2
    name = str(d["encoder"]) if "encoder" in d else path
    issues = []

    if not np.isfinite(e).all():
        issues.append(f"{int((~np.isfinite(e)).sum())} non-finite values")

    # 1. per-layer variation across utterances
    sd = e.std(axis=0).mean(axis=1)            # [L]
    dead = [int(i) for i, v in enumerate(sd) if v < 1e-4]
    if dead:
        issues.append(f"layers with ~zero across-utterance SD: {dead}")

    # 2. first vs last layer
    a, b = e[:, 0].mean(0), e[:, -1].mean(0)
    cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
    if cos > 0.99:
        issues.append(f"layer 0 vs layer {L-1} cosine = {cos:.4f} (near-identical)")

    # how many distinct layers are there really?
    flat = e.mean(0)                            # [L, 2D]
    nrm = flat / (np.linalg.norm(flat, axis=1, keepdims=True) + 1e-9)
    sim = nrm @ nrm.T
    dupes = int(((sim > 0.9999).sum() - L) / 2)
    if dupes:
        issues.append(f"{dupes} duplicated layer pair(s)")

    # 3. mean vs std halves
    m, s = e[:, :, :D], e[:, :, D:]
    if np.allclose(m, s, atol=1e-3):
        issues.append("mean and std halves are identical (pooling collapsed)")
    if (s < 0).any():
        issues.append("negative values in the std half")

    status = "FAIL" if issues else "ok  "
    print(f"{status} {name:10s} N={N:<6d} layers={L:<3d} dim={D:<5d} "
          f"L0~L{L-1} cos={cos:.3f}  SD range {sd.min():.3f}-{sd.max():.3f}")
    for i in issues:
        print(f"       -> {i}")
    return not issues


if __name__ == "__main__":
    args = sys.argv[1:] or ["features/smoke_*.npz"]
    paths = sorted({p for a in args for p in glob.glob(a)})
    if not paths:
        sys.exit("no files matched")
    ok = all(check(p) for p in paths)
    print("\nall clean" if ok else "\nFIX THE FAILURES before the full extraction run")
    sys.exit(0 if ok else 1)
