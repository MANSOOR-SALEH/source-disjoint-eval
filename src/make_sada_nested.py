"""
Build a NESTED subset of SADA, to test whether nesting is the operative variable.

    python src/make_sada_nested.py ^
        --src "F:/All Pycharm Projects/SADA_CLEAN/sada_clean.csv" ^
        --out manifests/sada_nested.csv --purity 0.7

The problem this solves
-----------------------
AYDID (nested) drops 0.030 under source-disjoint evaluation; SADA (crossed)
drops 0.195. Attributing that to nesting is not yet warranted, because the two
corpora also differ in class count (7 vs 3), size, chance rate, language variety
and utterances per speaker. A reviewer will ask whether nesting matters or
whether SADA is simply the harder, noisier task.

This isolates the variable. Each show is assigned to the dialect that dominates
it, and only that dialect's utterances are kept, so every source maps to exactly
one label. The result is nested BY CONSTRUCTION while holding constant: corpus,
language, class count, chance rate, encoders, probe and protocol. Comparing
full SADA against nested SADA therefore varies nesting and little else.

Predicted outcome if nesting is the operative variable: the source-disjoint drop
on nested SADA falls toward AYDID's 0.03 rather than staying near 0.195.

The obvious objection, stated so the paper can answer it: discarding
off-dialect utterances also removes the shows' internal dialect variety, so the
subset is easier in a way that has nothing to do with nesting. The random and
speaker-disjoint arms control for this -- they are computed on the same subset,
so if the subset were simply easier those arms would rise too, and the
comparison of interest is the DROP, not the absolute level.
"""

import argparse
import os
import sys

import pandas as pd

OUT_COLS = ["utt_id", "path", "label", "speaker_id", "source_id",
            "content_type", "environment"]
MIN_DUR, MAX_DUR = 1.0, 30.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="manifests/sada_nested.csv")
    ap.add_argument("--source-col", default="ShowName")
    ap.add_argument("--purity", type=float, default=0.7,
                    help="keep a show only if its dominant dialect covers at least "
                         "this fraction of it; lower keeps more data, higher keeps "
                         "shows that were already close to single-dialect")
    ap.add_argument("--min-shows", type=int, default=4,
                    help="minimum shows per class for a grouped split to exist")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    print(f"read {len(df)} rows, {df[args.source_col].nunique()} sources")

    ct = pd.crosstab(df[args.source_col], df["dialect"])
    dominant = ct.idxmax(axis=1)
    purity = ct.max(axis=1) / ct.sum(axis=1)

    print(f"\n-- show purity (fraction of a show in its dominant dialect) --")
    print(pd.cut(purity, [0, .5, .7, .9, .99, 1.0]).value_counts().sort_index().to_string())

    keep_shows = purity[purity >= args.purity].index
    sub = df[df[args.source_col].isin(keep_shows)].copy()
    sub = sub[sub["dialect"] == sub[args.source_col].map(dominant)]
    print(f"\npurity >= {args.purity}: {len(sub)} utts from {sub[args.source_col].nunique()} shows "
          f"({100*len(sub)/len(df):.1f}% of the corpus)")

    if "SegmentLength" in sub.columns:
        d = sub["SegmentLength"]
        drop = (d < MIN_DUR) | (d > MAX_DUR)
        if drop.any():
            print(f"dropping {int(drop.sum())} clip(s) outside [{MIN_DUR}, {MAX_DUR}] s")
            sub = sub[~drop]

    g = sub.groupby("dialect").agg(utts=("dialect", "size"),
                                   shows=(args.source_col, "nunique"))
    print(f"\n-- nested subset --")
    print(g.to_string())
    if (g["shows"] < args.min_shows).any():
        sys.exit(f"\na class has fewer than {args.min_shows} shows: no usable grouped "
                 f"split. Lower --purity to keep more shows.")

    out = pd.DataFrame({
        "path": sub["file_path"].str.replace("\\", "/", regex=False),
        "label": sub["dialect"],
        "speaker_id": sub["speaker_id"],
    })
    out["utt_id"] = (out["path"].str.rsplit("/", n=1).str[-1]
                                .str.replace(r"\.wav$", "", regex=True))
    if out["utt_id"].duplicated().any():
        out["utt_id"] = out["path"].str.replace(r"\.wav$", "", regex=True)

    shows = sorted(sub[args.source_col].dropna().unique())
    mapping = {s: f"SHOW_{i:02d}" for i, s in enumerate(shows)}
    out["source_id"] = sub[args.source_col].map(mapping).values
    out["content_type"] = sub.get("Category", "unknown").values
    out["environment"] = sub.get("Environment", "unknown").values

    per = out.groupby("source_id")["label"].nunique()
    assert int(per.max()) == 1, "subset is not nested -- check the dominance mapping"
    print(f"\n  verified: every source maps to exactly one label (nested)")
    print(f"  source-disjoint folds will be capped at {int(g['shows'].min())}")

    ups = out.groupby("speaker_id").size()
    print(f"  {out['speaker_id'].nunique()} speakers, {ups.mean():.2f} utts/speaker")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out[OUT_COLS].to_csv(args.out, index=False, encoding="utf-8-sig")
    lut = os.path.splitext(args.out)[0] + "_sources.csv"
    pd.DataFrame({"source_id": list(mapping.values()),
                  args.source_col: list(mapping.keys()),
                  "dominant_dialect": [dominant[s] for s in shows],
                  "purity": [round(float(purity[s]), 3) for s in shows]}).to_csv(
        lut, index=False, encoding="utf-8-sig")
    print(f"\nwrote {args.out} ({len(out)} rows) and {lut}")
    print(f"\nnext:")
    print(f"  python src/folds.py --manifest {args.out} --out folds/sada_nested.json "
          f"--n-folds 4 --n-repeats 20")
    print(f"  then probe with --subset-of manifests/sada.csv so the existing SADA")
    print(f"  features are reused -- no re-extraction needed.")


if __name__ == "__main__":
    main()
