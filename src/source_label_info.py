"""
Does the source variable carry the label? Measure it, then test whether it
predicts the source-disjoint drop.

    python src/source_label_info.py \
        --manifest aydid=manifests/aydid_clean.csv \
        --manifest sada=manifests/sada.csv \
        --manifest sada_nested=manifests/sada_nested.csv \
        --results results

Why this replaces the nested/crossed story
------------------------------------------
The nested-vs-crossed taxonomy predicted that nested corpora would show small
source-disjoint drops. The nested SADA subset falsified that: it is nested by
construction and drops ~0.55, the largest of the three. Nesting is not the
operative variable.

What separates the cases is HOW MUCH OF THE LABEL THE SOURCE VARIABLE CARRIES.
If knowing which show a clip came from tells you its dialect, then a model can
score by identifying the show, and holding shows out destroys that score. The
quantity is normalised mutual information between source and label,

    U(label | source) = I(source; label) / H(label),

the fraction of label entropy explained by source identity. It runs from 0 (the
source says nothing about the label) to 1 (the source determines it). It needs
no audio, no model and no training -- just the manifest -- so a practitioner can
compute it before deciding whether source-disjoint evaluation is necessary.

U alone does not separate the cases: AYDID and the nested SADA subset both have
U = 1.0 (every show is one dialect) yet drop 0.03 and 0.55 respectively. The
difference is not how much label information a source carries, but HOW MUCH OF
A CLASS A SINGLE SOURCE HOLDS.

    max_share(c) = largest source's fraction of class c

In the nested SADA subset one show is 82% of Hijazi. Holding it out does not
merely remove a recording condition -- it removes the class from TRAINING, so
the model cannot classify what it has barely seen. That is a class-coverage
failure, not recording-source leakage, and it inflates the drop for a reason
that has nothing to do with the confound under study.

AYDID has no such concentration: 50 speakers per class spread over 5-7 shows,
largest share ~25%. Full SADA is crossed, so no single show carries a class at
all.

The script therefore reports three quantities per corpus -- U(y|source),
max_share, and effective sources per class -- and flags any corpus whose
source-disjoint estimate is confounded by class coverage. A corpus with
max_share above ~0.5 cannot give an interpretable source-disjoint number, and
that is a precondition worth checking before running the protocol at all.
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd


def entropy(counts):
    p = np.asarray(counts, dtype=float)
    p = p[p > 0]
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def stats(df):
    """Label information carried by source_id, plus source-count diagnostics."""
    ct = pd.crosstab(df["source_id"], df["label"]).to_numpy(dtype=float)
    n = ct.sum()
    p_xy = ct / n
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    nz = p_xy > 0
    mi = float((p_xy[nz] * np.log(p_xy[nz] / (p_x @ p_y)[nz])).sum())
    h_label = entropy(p_y.ravel())
    h_source = entropy(p_x.ravel())

    # effective sources per class and, more importantly, the share of a class
    # held by its single largest source
    eff, raw, share = [], [], []
    for lab, sub in df.groupby("label"):
        sizes = sub["source_id"].value_counts().to_numpy(dtype=float)
        raw.append(len(sizes))
        eff.append(float(np.exp(entropy(sizes))))
        share.append(float(sizes.max() / sizes.sum()))

    # how well does source predict label, in accuracy terms
    best = ct.max(axis=1).sum() / n

    return {
        "n": int(n),
        "classes": int(ct.shape[1]),
        "sources": int(ct.shape[0]),
        "U_label_given_source": mi / h_label if h_label > 0 else float("nan"),
        "source_to_label_acc": float(best),
        "chance": 1.0 / ct.shape[1],
        "sources_per_class_min": int(min(raw)),
        "sources_per_class_mean": float(np.mean(raw)),
        "eff_sources_per_class_min": float(min(eff)),
        "eff_sources_per_class_mean": float(np.mean(eff)),
        "max_share_worst": float(max(share)),
        "max_share_mean": float(np.mean(share)),
        "coverage_ok": bool(max(share) < 0.5),
        "H_source": h_source,
    }


def drops(results_dir, prefix):
    """Mean source-disjoint drop across encoders for one corpus prefix."""
    out = []
    for p in glob.glob(os.path.join(results_dir, f"{prefix}.*.json")):
        stem = os.path.basename(p)[len(prefix) + 1:-5]
        if "." in stem:            # e.g. sada_nested.* matched by sada.*
            continue
        r = json.load(open(p))
        P = r.get("protocols", {})
        if "speaker_disjoint" in P and "source_disjoint" in P:
            out.append(P["speaker_disjoint"]["at_convention_layer"]
                       - P["source_disjoint"]["at_convention_layer"])
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", action="append", required=True,
                    metavar="NAME=PATH")
    ap.add_argument("--results", default="results")
    ap.add_argument("--csv", default="results/source_label_info.csv")
    a = ap.parse_args()

    rows = []
    for spec in a.manifest:
        name, path = spec.split("=", 1)
        df = pd.read_csv(path)
        s = stats(df)
        s["corpus"] = name
        d = drops(a.results, name)
        s["drop_mean"] = float(d.mean()) if len(d) else float("nan")
        s["drop_min"] = float(d.min()) if len(d) else float("nan")
        s["drop_max"] = float(d.max()) if len(d) else float("nan")
        s["n_encoders"] = len(d)
        rows.append(s)

    rows.sort(key=lambda r: r["U_label_given_source"])

    print("=" * 92)
    print("LABEL INFORMATION IN THE SOURCE VARIABLE  (manifest only -- no audio, no model)")
    print("=" * 92)
    print(f"{'corpus':14s} {'cls':>4s} {'src':>5s} {'U(y|src)':>9s} {'src/cls':>8s} "
          f"{'eff/cls':>8s} {'max share':>10s}  coverage")
    print("-" * 92)
    for r in rows:
        flag = "ok" if r["coverage_ok"] else "CONFOUNDED"
        print(f"{r['corpus']:14s} {r['classes']:>4d} {r['sources']:>5d} "
              f"{r['U_label_given_source']:>9.3f} "
              f"{r['sources_per_class_mean']:>8.1f} "
              f"{r['eff_sources_per_class_mean']:>8.1f} "
              f"{r['max_share_worst']:>10.2f}  {flag}")
    if any(not r["coverage_ok"] for r in rows):
        print("\n  CONFOUNDED: one source holds >50% of some class. Holding it out removes")
        print("  that class from TRAINING, so the source-disjoint score measures class")
        print("  coverage, not recording-source leakage. Such a corpus cannot give an")
        print("  interpretable source-disjoint number.")

    print()
    print("=" * 92)
    print("AGAINST THE MEASURED DROP")
    print("=" * 92)
    print(f"{'corpus':14s} {'U(y|src)':>9s} {'max share':>10s} {'coverage':>11s} {'drop':>18s}")
    print("-" * 92)
    for r in rows:
        rng = (f"{r['drop_mean']:.3f} [{r['drop_min']:.3f},{r['drop_max']:.3f}]"
               if r["n_encoders"] else "n/a")
        print(f"{r['corpus']:14s} {r['U_label_given_source']:>9.3f} "
              f"{r['max_share_worst']:>10.2f} "
              f"{('ok' if r['coverage_ok'] else 'CONFOUNDED'):>11s} {rng:>18s}")

    have = [r for r in rows if r["n_encoders"]]
    usable = [r for r in have if r["coverage_ok"]]
    print()
    if len(have) > len(usable):
        bad = [r["corpus"] for r in have if not r["coverage_ok"]]
        print(f"  {', '.join(bad)}: the drop is not interpretable as leakage -- see above.")
    if len(usable) >= 2:
        print(f"\n  Among corpora that pass the coverage check:")
        for r in sorted(usable, key=lambda r: r["U_label_given_source"]):
            print(f"    {r['corpus']:14s} U={r['U_label_given_source']:.3f}  "
                  f"drop={r['drop_mean']:.3f}")
        print(f"\n  Higher U means the source variable carries more of the label, so more")
        print(f"  of the reported score can be recovered from source identity alone.")

    os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(a.csv, index=False)
    print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
