"""
Synthetic end-to-end validation.  Run: python tests/test_pipeline.py

Builds a corpus where the answer is known: layer 1 carries a unique per-source
signature (nested within class, so it is perfectly class-predictive on seen
sources), layer 3 carries a weak dialect signal that generalises across sources.

A correct pipeline must:
  * select layer 1 and score ~1.0 under random and speaker-disjoint,
  * select layer 3 and score well below that under source-disjoint.

If the source-disjoint arm still reports ~1.0, the folds are leaking. If it
selects layer 1, the inner splits are not grouped like the outer ones -- the
failure mode this project exists to avoid.

Source-count matching is disabled here on purpose. This test checks that the
FOLDS do not leak; source matching deliberately reduces the training sources in
the leakier arms, which suppresses the very leak the test is trying to detect.
The two are verified separately.

Calibration note: this passes cleanly at n_src >= 6 with 3 inner folds. At
n_src = 4 with 2 inner folds one fold in four selects the leaky layer, because
the inner grouped split has too few sources left to expose it. That is a real
constraint on the corpora, not a quirk of the toy -- see README section 3.
"""

import json
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src")


def build(tmp, n_cls=3, n_src=6, n_spk=5, n_utt=8, seed=3):
    rng = np.random.default_rng(seed)
    rows, meta = [], []
    for c in range(n_cls):
        for s in range(n_src):
            for k in range(n_spk):
                spk = f"s{c}_{s}_{k}"
                for u in range(n_utt):
                    rows.append(dict(utt_id=f"{spk}_{u}", path="x", label=f"C{c}",
                                     speaker_id=spk, source_id=f"src{c}_{s}"))
                    meta.append((c, s))
    df = pd.DataFrame(rows)
    man = os.path.join(tmp, "m.csv")
    df.to_csv(man, index=False)

    L, D = 5, 4 + n_cls * n_src
    emb = rng.normal(0, 1, (len(df), L, D)).astype(np.float32)
    for i, (c, s) in enumerate(meta):
        emb[i, 3, c] += 0.9                     # weak, generalising dialect signal
        emb[i, 1, 4 + c * n_src + s] += 8.0     # strong, source-specific signature
    feat = os.path.join(tmp, "f.npz")
    np.savez(feat, emb=emb.astype(np.float16),
             utt_id=df.utt_id.to_numpy().astype(str), layers=np.arange(L),
             encoder=np.array("synthetic"), hf_id=np.array("none"))
    return man, feat


def main():
    with tempfile.TemporaryDirectory() as tmp:
        man, feat = build(tmp)
        folds = os.path.join(tmp, "folds.json")
        res = os.path.join(tmp, "res.json")
        subprocess.run([sys.executable, f"{SRC}/folds.py", "--manifest", man,
                        "--out", folds, "--n-folds", "4", "--n-inner", "3", "--n-repeats", "6",
                        "--no-match-train-sources"], check=True)
        subprocess.run([sys.executable, f"{SRC}/probe.py", "--features", feat,
                        "--manifest", man, "--folds", folds, "--out", res,
                        "--fixed-layer", "3", "--epochs", "25", "--no-linear"],
                       check=True, stdout=subprocess.DEVNULL)
        r = json.load(open(res))

        def get(p):
            e = r["protocols"][p]
            return e["oracle_upper_bound"]["mean"], e["peak_layer_of_mean_profile"]

        ok = True
        for p in ("random", "speaker_disjoint"):
            f1, peak = get(p)
            hit = f1 > 0.95 and peak == 1
            ok &= hit
            print(f"{'PASS' if hit else 'FAIL'} {p:19s} best-layer F1={f1:.3f} peak=L{peak} "
                  f"(expect ~1.0, peak on the leaky layer 1)")

        for p in ("source_disjoint", "source_disjoint_rep"):
            if p not in r["protocols"]:
                continue
            f1, peak = get(p)
            hit = f1 < 0.75 and peak == 3
            ok &= hit
            print(f"{'PASS' if hit else 'FAIL'} {p:19s} best-layer F1={f1:.3f} peak=L{peak} "
                  f"(expect the generalising layer 3, F1 well below 1.0)")

        moved = len(set(r["peak_layer_by_protocol"].values())) > 1
        ok &= moved
        print(f"{'PASS' if moved else 'FAIL'} {'peak-shift detected':19s} "
              f"{r['peak_layer_by_protocol']}")

        print("\nALL PASS" if ok else "\nFAILURE -- do not run the real sweep until this passes")
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()