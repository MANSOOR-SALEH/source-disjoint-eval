"""
Convert the SADA manifest into the benchmark schema.

    python src/make_sada_manifest.py ^
        --src "F:/All Pycharm Projects/SADA_CLEAN/sada_clean.csv" ^
        --out manifests/sada.csv

Input  : file_path, dialect, speaker_id, ShowName, FileName, SegmentLength,
         Category, Environment, ...
Output : utt_id, path, label, speaker_id, source_id, content_type, environment

Why SADA is the right second corpus
-----------------------------------
Its provenance structure is the OPPOSITE of AYDID's, which is what makes the
pair informative rather than repetitive:

  AYDID  source NESTED in label   -- every show is one dialect, so the
                                     show/label correlation is a property of
                                     the corpus that no split can remove.
  SADA   source CROSSED with label -- most shows carry several dialects, so
                                     holding out a show removes a recording
                                     condition without removing label
                                     information.

That difference predicts different behaviour under source-disjoint evaluation,
and `folds.py` selects the holdout scheme accordingly: within-class partitioning
when nested, whole-group holdout across all classes when crossed. Using
within-class folds on a crossed corpus would place one show on both sides of the
split, which is a leak.

ShowName is Arabic text, so it is mapped to stable ASCII source ids
(SHOW_00 ...) with the lookup written alongside. Grouping does not care, but
Windows consoles and filenames do.
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
    ap.add_argument("--out", default="manifests/sada.csv")
    ap.add_argument("--source-col", default="ShowName",
                    choices=["ShowName", "FileName"],
                    help="provenance unit: ShowName (63 programmes, coarse) or "
                         "FileName (2535 episodes, fine, ~3 utts each)")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    need = {"file_path", "dialect", "speaker_id", args.source_col}
    if not need.issubset(df.columns):
        sys.exit(f"missing columns: {sorted(need - set(df.columns))}")
    print(f"read {len(df)} rows")

    out = pd.DataFrame({
        "path": df["file_path"].str.replace("\\", "/", regex=False),
        "label": df["dialect"],
        "speaker_id": df["speaker_id"],
    })
    out["utt_id"] = (out["path"].str.rsplit("/", n=1).str[-1]
                                .str.replace(r"\.wav$", "", regex=True))
    if out["utt_id"].duplicated().any():
        out["utt_id"] = out["path"].str.replace(r"\.wav$", "", regex=True)

    # Arabic show titles -> stable ascii ids, lookup preserved
    shows = sorted(df[args.source_col].dropna().unique())
    mapping = {s: f"SHOW_{i:02d}" for i, s in enumerate(shows)}
    out["source_id"] = df[args.source_col].map(mapping)
    lut = os.path.splitext(args.out)[0] + "_sources.csv"
    pd.DataFrame({"source_id": list(mapping.values()),
                  args.source_col: list(mapping.keys())}).to_csv(
        lut, index=False, encoding="utf-8-sig")

    out["content_type"] = df.get("Category", "unknown")
    out["environment"] = df.get("Environment", "unknown")

    if "SegmentLength" in df.columns:
        d = df["SegmentLength"]
        drop = (d < MIN_DUR) | (d > MAX_DUR)
        if drop.any():
            print(f"dropping {int(drop.sum())} clip(s) outside "
                  f"[{MIN_DUR}, {MAX_DUR}] s")
            out = out[~drop.values]

    blank = out["source_id"].isna()
    if blank.any():
        sys.exit(f"{int(blank.sum())} rows have no {args.source_col}")

    # --- provenance structure --------------------------------------------
    per_show = out.groupby("source_id")["label"].nunique()
    nested = int(per_show.max()) == 1
    print(f"\n-- provenance structure ({args.source_col}) --")
    print(f"  {out['source_id'].nunique()} sources, "
          f"{int((per_show > 1).sum())} spanning more than one class")
    print(f"  -> sources are {'NESTED IN' if nested else 'CROSSED WITH'} the label")
    if nested:
        print("     within-class holdout; the source/label correlation cannot be\n"
              "     removed by splitting, only measured.")
    else:
        print("     whole-group holdout across all classes; holding out a source\n"
              "     removes a recording condition without removing label\n"
              "     information. folds.py detects this automatically.")

    print("\n-- sources per class --")
    print(out.groupby("label")["source_id"].nunique().sort_values().to_string())
    print("\n-- class balance --")
    print(out["label"].value_counts().to_string())
    ups = out.groupby("speaker_id").size()
    print(f"\n-- utterances per speaker --\n  mean {ups.mean():.2f}  "
          f"max {ups.max()}  ({out['speaker_id'].nunique()} speakers)")
    if ups.mean() < 2:
        print("  !! under 2 utts/speaker: a random split is already close to\n"
              "     speaker-disjoint, so that protocol has little to do here.\n"
              "     Report the ratio; it bounds what speaker-disjoint can show.")
    cross = out.groupby("speaker_id")["source_id"].nunique()
    print(f"  speakers in >1 source: {int((cross > 1).sum())}")

    if out["content_type"].nunique() > 1:
        print(f"\n-- content_type x class (genre confound check) --")
        print(pd.crosstab(out["label"], out["content_type"]).to_string())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out[OUT_COLS].to_csv(args.out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {args.out} ({len(out)} rows) and {lut}")
    print(f"next: python src/check_manifest.py --manifest {args.out} "
          f"--audio-root <SADA audio root>")


if __name__ == "__main__":
    main()
