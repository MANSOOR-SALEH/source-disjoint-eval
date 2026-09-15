"""
How many classes appear in each source-disjoint TEST fold?

    python src/fold_class_coverage.py --manifest manifests/kespeech.csv \
        --folds folds/kespeech.json

fold_coverage.py answers whether a fold starves a class in TRAINING. This
answers a different and, for macro-F1, more dangerous question: whether every
class is present in the fold's TEST partition.

Why it matters. Macro-F1 averages per-class F1. If a held-out source carries
only k of K classes, the remaining K-k classes have no support in that fold, and
the score depends entirely on an undisclosed convention:

  * average over PRESENT classes only  -> the effective chance rate is 1/k, not
    1/K, so comparing the result against 1/K is wrong;
  * score absent classes as 0          -> macro-F1 is multiplied by roughly k/K
    regardless of how good the model is, and a "collapse to chance" can be
    forced arithmetically.

Under leave-one-source-out with sources CROSSED with the label this is a live
risk: one held-out city or programme need not carry every class. Under sources
NESTED in the label it is guaranteed to fire, since a source carries exactly one
class by definition -- which is why nested corpora must use within-class
grouping rather than whole-group holdout.

The script reports, per protocol: classes present per test fold, the implied
chance rate, and how the reported macro-F1 should be interpreted.
"""

import argparse
import json

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--protocol", default="source_disjoint")
    ap.add_argument("--show-folds", type=int, default=12,
                    help="how many folds to list individually")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest).reset_index(drop=True)
    spec = json.load(open(a.folds))
    K = df["label"].nunique()
    labels = sorted(df["label"].unique())

    print(f"{a.manifest}: {len(df)} utts, {K} classes")
    print(f"folds: {a.folds}\n")

    for prot in ([a.protocol] if a.protocol in spec["protocols"]
                 else list(spec["protocols"])):
        P = spec["protocols"][prot]
        rows = []
        for k, f in enumerate(P["folds"]):
            te = df.loc[f["test"], "label"]
            present = sorted(te.unique())
            counts = te.value_counts()
            rows.append({"fold": k, "n_test": len(f["test"]),
                         "k_present": len(present),
                         "min_class_n": int(counts.min()),
                         "present": present})
        R = pd.DataFrame(rows)
        full = int((R["k_present"] == K).sum())

        print("=" * 74)
        print(f"{prot}   {len(R)} folds, {K} classes")
        print("=" * 74)
        show = R.head(a.show_folds)
        print(f"{'fold':>4s} {'n_test':>7s} {'classes':>8s} {'min class n':>12s}  missing")
        print("-" * 74)
        for r in show.itertuples():
            miss = [l for l in labels if l not in r.present]
            print(f"{r.fold:>4d} {r.n_test:>7d} {r.k_present:>4d}/{K:<3d} "
                  f"{r.min_class_n:>12d}  {','.join(m[:10] for m in miss) if miss else '-'}")
        if len(R) > a.show_folds:
            print(f"  ... {len(R) - a.show_folds} more folds")
        print("-" * 74)
        print(f"folds containing every class: {full}/{len(R)}")
        print(f"classes per test fold: min {R['k_present'].min()}, "
              f"median {int(R['k_present'].median())}, max {R['k_present'].max()}")

        if full == len(R):
            print(f"\n  OK: every test fold contains all {K} classes, so macro-F1 is\n"
                  f"  averaged over the full label set in every fold and 1/{K} = "
                  f"{1/K:.3f} is the correct chance reference.")
        else:
            kmin = int(R["k_present"].min())
            print(f"\n  !! {len(R)-full} fold(s) are CLASS-INCOMPLETE.")
            print(f"     Macro-F1 in those folds is convention-dependent:")
            print(f"       - averaging over present classes only makes the effective\n"
                  f"         chance rate up to 1/{kmin} = {1/kmin:.3f}, not {1/K:.3f};")
            print(f"       - scoring absent classes as 0 scales macro-F1 by about\n"
                  f"         {R['k_present'].mean()/K:.2f} on average, independent of the model.")
            print(f"     Either report which convention was used and the corresponding\n"
                  f"     chance rate, or restrict the protocol to class-complete folds.")

        # how much do incomplete folds move the mean?
        if full < len(R):
            comp = R[R["k_present"] == K]
            print(f"\n     class-complete folds: {len(comp)}/{len(R)} "
                  f"({100*len(comp)/len(R):.0f}% of folds, "
                  f"{100*comp['n_test'].sum()/R['n_test'].sum():.0f}% of test utterances)")
        print()


if __name__ == "__main__":
    main()
