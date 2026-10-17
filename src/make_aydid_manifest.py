"""
Convert the AYDID manifest into the benchmark schema.

    python src/make_aydid_manifest.py ^
        --src "F:/PythonProject/caml_project/YEMEN_DIALECT_DATASET/aydid_manifest.csv" ^
        --out manifests/aydid.csv

Input  : file_path, dialect, speaker_id, show_name, content_type
Output : utt_id, path, label, speaker_id, source_id, content_type

Mapping is a rename plus a derived utt_id. The work this script actually does is
the diagnostics it prints, because two AYDID-specific things decide whether the
source-disjoint arm is viable:

  1. Whether show_name is populated for the field-recording stream. Media clips
     come from named programmes; field recordings may not, and a blank there is
     not a missing value you can impute -- it is a class of utterances with no
     recording provenance at all.
  2. How many distinct shows each dialect has. A "show" is a programme, not an
     episode, so a series with 40 episodes is ONE group. That is correct for
     source-disjoint evaluation and it is also why the group count is far
     smaller than the episode count.
"""

import argparse
import os
import sys

import pandas as pd

RENAME = {"file_path": "path", "dialect": "label", "show_name": "source_id"}
OUT_COLS = ["utt_id", "path", "label", "speaker_id", "source_id", "content_type"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", default="manifests/aydid.csv")
    ap.add_argument("--drop-no-source", action="store_true",
                    help="drop rows with a blank source_id instead of stopping")
    args = ap.parse_args()

    df = pd.read_csv(args.src)
    print(f"read {len(df)} rows from {args.src}")
    need = set(RENAME) | {"speaker_id"}
    if not need.issubset(df.columns):
        sys.exit(f"missing columns: {sorted(need - set(df.columns))}")

    df = df.rename(columns=RENAME)
    if "content_type" not in df.columns:
        df["content_type"] = "unknown"

    # utt_id from the path stem; paths are already unique per utterance
    df["utt_id"] = (df["path"].str.replace("\\", "/", regex=False)
                              .str.rsplit("/", n=1).str[-1]
                              .str.replace(r"\.wav$", "", regex=True))
    dup = int(df["utt_id"].duplicated().sum())
    if dup:
        print(f"  {dup} duplicate stems -> falling back to full-path utt_id")
        df["utt_id"] = df["path"].str.replace("\\", "/", regex=False).str.replace(r"\.wav$", "", regex=True)
    df["path"] = df["path"].str.replace("\\", "/", regex=False)

    # --- provenance coverage --------------------------------------------
    blank = df["source_id"].isna() | (df["source_id"].astype(str).str.strip() == "")
    if blank.any():
        print(f"\n!! {int(blank.sum())} rows ({100*blank.mean():.1f}%) have no show_name")
        print(df.loc[blank].groupby(["label", "content_type"]).size().to_string())
        print("\nThese are almost certainly the field-recording stream. Three options,\n"
              "in order of how well they survive review:\n"
              "  (a) recover a provenance label for them -- session, site, or recording\n"
              "      date. Best answer; Paper 2 flags exactly this gap in AYDID.\n"
              "  (b) drop them from the source-disjoint arm only, keep them in random\n"
              "      and speaker-disjoint, and report the different N per protocol.\n"
              "  (c) treat 'field' as one source per dialect. Cheapest, but it merges\n"
              "      recordings made in unrelated conditions into one group, which\n"
              "      understates the confound rather than removing it.\n"
              "Do NOT map each speaker to their own source: that silently turns the\n"
              "source-disjoint arm back into a speaker-disjoint one.")
        if args.drop_no_source:
            df = df[~blank].copy()
            print(f"  --drop-no-source given: kept {len(df)} rows")
        else:
            sys.exit("\nstopping. Decide, then re-run (--drop-no-source implements option b).")

    # --- diagnostics ------------------------------------------------------
    print("\n-- shows per dialect (a series = one group, however many episodes) --")
    g = df.groupby("label")["source_id"].nunique().sort_values()
    for label, n in g.items():
        flag = "ok  " if n >= 6 else ("WARN" if n >= 3 else "FATAL")
        print(f"  {flag} {label:8s} {n:3d} shows")
    print(f"  -> source-disjoint folds capped at {int(g.min())}; "
          f"{df['source_id'].nunique()} shows total")

    print("\n-- speakers per show --")
    sp = df.groupby("source_id")["speaker_id"].nunique()
    print(f"  mean {sp.mean():.1f}  min {sp.min()}  max {sp.max()}")
    cross = df.groupby("speaker_id")["source_id"].nunique()
    print(f"  speakers appearing in >1 show: {int((cross > 1).sum())}")

    if df["content_type"].nunique() > 1:
        print("\n-- content_type x dialect (candidate secondary grouping) --")
        print(pd.crosstab(df["label"], df["content_type"]).to_string())

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df[OUT_COLS].to_csv(args.out, index=False)
    print(f"\nwrote {args.out}  ({len(df)} rows)")
    print("next: python src/check_manifest.py --manifest %s --audio-root <AYDID root>" % args.out)


if __name__ == "__main__":
    main()
