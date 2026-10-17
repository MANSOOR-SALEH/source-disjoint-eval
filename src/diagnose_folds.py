"""
Why can't grouped folds be built? Diagnose before spending compute.

    python src/diagnose_folds.py --manifest manifests/kespeech.csv

folds.py refuses to build a split that would drop a class from either side. That
refusal is correct, but it does not say WHICH class is binding or what the
maximum achievable fold count is. This does, from the manifest alone.

The binding quantity for a whole-group holdout across all classes is, for each
class, the number of sources carrying it. If a class has k sources, at most k-1
can be held out at once while leaving it represented in training, and the
held-out set must avoid taking all of any class. With f folds the sources are
partitioned f ways, so each fold removes roughly 1/f of every class's sources;
a class with k sources needs k >= f before a partition can leave it present on
both sides, and in practice needs more because the partition is random with
respect to class.
"""

import argparse
import itertools

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--group", default="source_id")
    ap.add_argument("--max-folds", type=int, default=10)
    a = ap.parse_args()

    df = pd.read_csv(a.manifest)
    ct = pd.crosstab(df[a.group], df["label"])
    labels = list(ct.columns)
    n_src = len(ct)

    print(f"{a.manifest}: {len(df)} utts, {len(labels)} classes, {n_src} sources\n")

    print("-- sources carrying each class --")
    carr = (ct > 0).sum(axis=0).sort_values()
    for lab, k in carr.items():
        excl = int(((ct[lab] > 0) & (ct.drop(columns=[lab]).sum(axis=1) == 0)).sum())
        print(f"  {str(lab):16s} {k:>4d} sources  ({excl} carry this class only)")
    print(f"\n  binding class: {carr.index[0]} with {carr.iloc[0]} sources")

    nested = int(ct.gt(0).sum(axis=1).max()) == 1
    print(f"  sources are {'NESTED IN' if nested else 'CROSSED WITH'} the label")

    # concentration: does one source dominate a class?
    print("\n-- largest source's share of each class --")
    share = (ct / ct.sum(axis=0)).max(axis=0).sort_values(ascending=False)
    for lab, v in share.items():
        flag = "  <-- dominates" if v > 0.5 else ""
        print(f"  {str(lab):16s} {v:>5.2f}{flag}")

    # what fold count is achievable? simulate the partition folds.py performs
    print(f"\n-- achievable fold counts (whole-group holdout, all classes both sides) --")
    rng = np.random.default_rng(0)
    srcs = np.array(ct.index)
    best = 0
    for f in range(2, a.max_folds + 1):
        ok_tries = 0
        for _ in range(200):
            perm = rng.permutation(srcs)
            parts = np.array_split(perm, f)
            good = True
            for part in parts:
                test_ct = ct.loc[list(part)].sum(axis=0)
                train_ct = ct.drop(index=list(part)).sum(axis=0)
                if (test_ct == 0).any() or (train_ct == 0).any():
                    good = False
                    break
            ok_tries += good
        rate = ok_tries / 200
        mark = "ok" if rate > 0.05 else "--"
        print(f"  {f:>2d} folds: {rate:5.0%} of random partitions valid  {mark}")
        if rate > 0.05:
            best = f
    if best:
        print(f"\n  -> up to {best} folds achievable")
    else:
        print(f"\n  -> NO fold count works with random partitioning.")

    # leave-one-source-out: the weakest usable protocol
    print(f"\n-- leave-one-source-out feasibility --")
    bad = []
    for s in ct.index:
        train_ct = ct.drop(index=[s]).sum(axis=0)
        if (train_ct == 0).any():
            bad.append(s)
    print(f"  {n_src - len(bad)} of {n_src} sources can be held out singly "
          f"while keeping every class in training")
    if bad:
        print(f"  cannot hold out: {bad[:6]}")
    print(f"  -> leave-one-source-out gives {n_src - len(bad)} folds, each holding "
          f"out ~{100/n_src:.0f}% of the corpus")

    # which classes would a dropped class buy back?
    print(f"\n-- effect of excluding the binding class --")
    for drop_lab in list(carr.index[:2]):
        sub = ct.drop(columns=[drop_lab])
        sub = sub[sub.sum(axis=1) > 0]
        best2 = 0
        for f in range(2, a.max_folds + 1):
            ok = 0
            for _ in range(200):
                parts = np.array_split(rng.permutation(np.array(sub.index)), f)
                good = all(
                    not ((sub.loc[list(p)].sum(axis=0) == 0).any() or
                         (sub.drop(index=list(p)).sum(axis=0) == 0).any())
                    for p in parts)
                ok += good
            if ok / 200 > 0.05:
                best2 = f
        print(f"  excluding {drop_lab}: up to {best2} folds "
              f"({len(sub)} sources, {sub.shape[1]} classes)")


if __name__ == "__main__":
    main()
