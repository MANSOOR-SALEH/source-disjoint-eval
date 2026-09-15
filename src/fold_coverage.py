"""
Per-fold class coverage: is a source-disjoint drop leakage, or starvation?

    python src/fold_coverage.py --manifest manifests/sada.csv --folds folds/sada.json
    python src/fold_coverage.py --manifest manifests/sada.csv --folds folds/sada.json \
        --results results/sada.wavlm.json

Why a threshold on "largest source's share of a class" is not enough
--------------------------------------------------------------------
Holding out a source removes a recording condition -- that is the point -- but
it also removes training data. If it removes most of ONE class's training data,
the model fails on that class for a reason unrelated to the confound, and the
resulting drop overstates leakage.

Whether that happens depends on the folds, not on a corpus-level summary. SADA's
largest Hijazi show is 52% of the class, which looks alarming, but that show also
carries Khaleeji and Najdi clips, so holding it out shrinks every class rather
than one. The nested SADA subset is the opposite case: its largest show is 82% of
Hijazi and nothing else, so holding it out starves that class alone.

This script therefore reports, for every fold:
  * retained_min -- the smallest fraction of any class's training data that
    survives, AFTER the size matching probe.py applies;
  * whether the training set still contains every class at a usable level.

The primary check is the threshold: if no fold retains less than 0.25 of any
class, the drop cannot be an artefact of a depleted split. When a results file
is given the script will also relate per-fold retention to per-fold score, but a
correlation is reported ONLY when there are enough folds and a wide enough
retention span to estimate a slope (>= 8 folds and span >= 0.20). Below that it
carries no information and is suppressed in either direction -- quoting it was
the source of an earlier misstatement. With few folds, report the retention
range and the fact that every fold clears the threshold, not a correlation.
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
    ap.add_argument("--results", help="results/<corpus>.<encoder>.json, to correlate "
                                      "retention against per-fold score")
    ap.add_argument("--matched", action="store_true", default=True,
                    help="use train_matched (what probe.py actually trains on)")
    a = ap.parse_args()

    df = pd.read_csv(a.manifest).reset_index(drop=True)
    spec = json.load(open(a.folds))
    if a.protocol not in spec["protocols"]:
        raise SystemExit(f"{a.protocol} not in {list(spec['protocols'])}")
    P = spec["protocols"][a.protocol]
    labels = sorted(df["label"].unique())
    full = df["label"].value_counts()

    print(f"{a.manifest}   protocol={a.protocol}   {len(P['folds'])} folds")
    print("=" * 78)
    hdr = f"{'fold':>4s} {'n_test':>7s} {'n_train':>8s} " + " ".join(f"{l[-6:]:>8s}" for l in labels) + f" {'min':>6s}"
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for k, f in enumerate(P["folds"]):
        key = "train_matched" if (a.matched and "train_matched" in f) else "train"
        tr = df.loc[f[key], "label"].value_counts()
        te = len(f["test"])
        frac = {l: tr.get(l, 0) / full[l] for l in labels}
        mn = min(frac.values())
        rows.append({"fold": k, "n_test": te, "n_train": len(f[key]),
                     "retained_min": mn,
                     **{f"ret_{l}": frac[l] for l in labels}})
        flag = "  <-- starved" if mn < 0.25 else ""
        print(f"{k:>4d} {te:>7d} {len(f[key]):>8d} "
              + " ".join(f"{frac[l]:>8.2f}" for l in labels)
              + f" {mn:>6.2f}{flag}")

    R = pd.DataFrame(rows)
    print("-" * len(hdr))
    print(f"retained_min: mean {R['retained_min'].mean():.2f}  "
          f"min {R['retained_min'].min():.2f}  max {R['retained_min'].max():.2f}")
    n_starved = int((R["retained_min"] < 0.25).sum())
    if n_starved:
        print(f"\n  {n_starved} of {len(R)} fold(s) retain under 25% of some class's "
              f"training data.\n  Those folds measure class coverage as much as "
              f"recording-source leakage.")

    if a.results:
        r = json.load(open(a.results))
        e = r["protocols"].get(a.protocol)
        if not e:
            raise SystemExit(f"{a.protocol} not in {a.results}")
        sc = np.array(e["fixed"]["per_fold"])
        if len(sc) != len(R):
            raise SystemExit(f"{len(sc)} scores vs {len(R)} folds -- mismatched files")
        R["score"] = sc
        rho = float(np.corrcoef(R["retained_min"], R["score"])[0, 1])
        print(f"\n{'='*78}")
        print(f"RETENTION vs SCORE   ({a.results})")
        print("=" * 78)
        for r_ in R.sort_values("retained_min").itertuples():
            bar = "#" * int(r_.score * 40)
            print(f"  retained_min {r_.retained_min:.2f}  score {r_.score:.3f}  {bar}")
        span = float(R["retained_min"].max() - R["retained_min"].min())
        n = len(R)
        print(f"\n  retention span {R['retained_min'].min():.2f}-"
              f"{R['retained_min'].max():.2f} (width {span:.2f}) over {n} fold(s)")
        print(f"  Pearson r = {rho:+.3f}")

        # A correlation on a handful of folds spanning a narrow retention range
        # carries almost no information, and quoting it either way is misleading.
        # State what the data supports: whether any fold is starved, and whether
        # the retention range is wide enough for the question to be answerable.
        unreliable = n < 8 or span < 0.20
        if unreliable:
            print("  -> NOT INTERPRETABLE as a slope: with fewer than 8 folds or a\n"
                  "     retention span under 0.20 there is no leverage to estimate one.\n"
                  "     Do NOT quote this r as evidence for or against an association.\n"
                  "     Report instead: the retention range and the fact that every\n"
                  "     fold clears the starvation threshold.")
        elif rho > 0.5:
            print("  -> folds that starve a class score worse. A substantial part of this\n"
                  "     protocol's drop is class coverage, not recording-source leakage.\n"
                  "     Report the drop restricted to non-starved folds.")
        elif rho > 0.2:
            print("  -> weak positive association; report it and check the non-starved subset.")
        else:
            print("  -> no association detected over an adequate retention range.")

        starved = int((R["retained_min"] < 0.25).sum())
        print(f"\n  SENTENCE FOR THE PAPER:")
        if starved == 0:
            print(f"    every fold retains at least "
                  f"{R['retained_min'].min():.2f} of each class's training data,\n"
                  f"    well clear of the 0.25 starvation threshold, so the reported\n"
                  f"    degradation is not attributable to a starved training split.")
        else:
            print(f"    {starved} of {n} folds retain under 0.25 of some class's training\n"
                  f"    data; the drop on those folds conflates coverage with provenance.")

        ok = R[R["retained_min"] >= 0.25]
        if len(ok) and len(ok) < len(R):
            print(f"\n  non-starved folds only ({len(ok)}/{len(R)}): "
                  f"mean score {ok['score'].mean():.3f} "
                  f"vs {R['score'].mean():.3f} overall")


if __name__ == "__main__":
    main()