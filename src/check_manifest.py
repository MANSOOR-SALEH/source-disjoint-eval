"""
Day-one gate. Validate a manifest BEFORE spending GPU time on extraction.

    python src/check_manifest.py --manifest manifests/aydid.csv --audio-root D:/AYDID

Checks, in the order that costs you time if you get them wrong:
  1. required columns present, no missing values in the grouping variables
  2. sources per class  -- the >= 6 gate from tests/test_pipeline.py
  3. utterances per speaker -- bounds what speaker-disjoint splitting can do
  4. speakers crossing sources -- if none, source-disjoint implies speaker-disjoint
  5. class balance, duration stats
  6. audio files exist, are 16 kHz mono (spot-checked, then all if --full-audio)

Nothing here needs a GPU or a model. Run it first.
"""

import argparse
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

REQUIRED = ["utt_id", "path", "label", "speaker_id", "source_id"]
MIN_SOURCES_PER_CLASS = 6      # from tests/test_pipeline.py calibration


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--full-audio", action="store_true",
                    help="check every file rather than a 200-file sample (slow)")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    problems, warnings = [], []

    print(f"\n{'='*66}\n{args.manifest}  --  {len(df)} rows\n{'='*66}")

    # 1. schema ------------------------------------------------------------
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        print(f"FATAL missing columns: {missing}")
        sys.exit(1)
    for c in ["label", "speaker_id", "source_id", "path"]:
        n = int(df[c].isna().sum())
        if n:
            problems.append(f"{n} rows have empty '{c}'")
    dup = int(df["utt_id"].duplicated().sum())
    if dup:
        problems.append(f"{dup} duplicate utt_id values")

    # 2. sources per class -------------------------------------------------
    print("\n-- sources per class (the >= 6 gate) --")
    src_counts = df.groupby("label")["source_id"].nunique().sort_values()
    for label, n in src_counts.items():
        if n < 2:
            flag, msg = "FATAL", "no grouped split exists for this class"
            problems.append(f"class {label} has {n} source(s): {msg}")
        elif n < 3:
            flag, msg = "FATAL", "leave-one-group-out only, no inner split possible"
            problems.append(f"class {label} has {n} sources: {msg}")
        elif n < MIN_SOURCES_PER_CLASS:
            flag, msg = "WARN ", "below 6: fix the layer instead of selecting it"
            warnings.append(f"class {label} has {n} sources: {msg}")
        else:
            flag, msg = "ok   ", ""
        print(f"  {flag} {str(label):12s} {n:3d} sources   {msg}")
    print(f"  -> source-disjoint folds will be capped at {int(src_counts.min())}")

    # 3. utterances per speaker -------------------------------------------
    ups = df.groupby("speaker_id").size()
    print(f"\n-- utterances per speaker --\n  mean {ups.mean():.1f}  median {ups.median():.0f}  "
          f"min {ups.min()}  max {ups.max()}  ({df['speaker_id'].nunique()} speakers)")
    if ups.mean() < 2:
        warnings.append("under 2 utts/speaker: a random split is already nearly "
                        "speaker-disjoint, so that protocol has little work to do "
                        "-- report the ratio in the paper (Paper 2 makes this point)")

    # 4. nesting -----------------------------------------------------------
    crossing = df.groupby("speaker_id")["source_id"].nunique()
    n_cross = int((crossing > 1).sum())
    print(f"\n-- nesting --\n  speakers appearing in >1 source: {n_cross}")
    if n_cross == 0:
        print("  -> speakers are nested in sources, so source-disjoint splits are\n"
              "     automatically speaker-disjoint. The source-disjoint drop is then\n"
              "     the MARGINAL effect of holding out sources given speakers are\n"
              "     already held out -- say this explicitly in the paper.")
    else:
        warnings.append(f"{n_cross} speakers cross sources: source-disjoint is NOT "
                        f"automatically speaker-disjoint here; state which you report")

    # 5. balance -----------------------------------------------------------
    print("\n-- class balance --")
    vc = df["label"].value_counts().sort_index()
    for label, n in vc.items():
        print(f"  {str(label):12s} {n:6d}  ({100*n/len(df):5.1f}%)")
    if vc.max() / max(vc.min(), 1) > 1.5:
        warnings.append(f"class imbalance {vc.max()/vc.min():.2f}x -- macro-F1 handles "
                        f"it but chance rate is no longer the majority-class rate")

    # 6. audio -------------------------------------------------------------
    print("\n-- audio --")
    try:
        import soundfile as sf
    except ImportError:
        print("  soundfile not installed, skipping")
        sf = None
    if sf is not None:
        paths = [os.path.join(args.audio_root, p) for p in df["path"]]
        sample = paths if args.full_audio else list(np.array(paths)[
            np.linspace(0, len(paths) - 1, min(200, len(paths))).astype(int)])
        bad_missing, bad_sr, bad_ch, durs = [], [], [], []
        for p in sample:
            if not os.path.exists(p):
                bad_missing.append(p); continue
            try:
                info = sf.info(p)
            except Exception as e:
                bad_missing.append(f"{p} ({e})"); continue
            if info.samplerate != 16000:
                bad_sr.append(f"{os.path.basename(p)}@{info.samplerate}")
            if info.channels != 1:
                bad_ch.append(os.path.basename(p))
            durs.append(info.frames / info.samplerate)
        print(f"  checked {len(sample)} file(s){'' if args.full_audio else ' (sample)'}")
        if bad_missing:
            problems.append(f"{len(bad_missing)} file(s) missing/unreadable, e.g. {bad_missing[:3]}")
        if bad_sr:
            problems.append(f"{len(bad_sr)} file(s) not 16 kHz, e.g. {bad_sr[:3]} "
                            f"-- resample the corpus offline, never in the extraction loop")
        if bad_ch:
            problems.append(f"{len(bad_ch)} file(s) not mono, e.g. {bad_ch[:3]}")
        if durs:
            d = np.array(durs)
            print(f"  duration  mean {d.mean():.2f}s  min {d.min():.2f}s  max {d.max():.2f}s  "
                  f"total(est) {d.mean()*len(df)/3600:.1f} h")
            if d.max() > 30:
                warnings.append(f"clips up to {d.max():.0f}s: Whisper truncates at 30 s, "
                                f"so those clips are not comparable across encoders")

    # verdict --------------------------------------------------------------
    print(f"\n{'='*66}")
    for w in warnings:
        print(f"WARN   {w}")
    for p in problems:
        print(f"FATAL  {p}")
    if problems:
        print("\nFix the FATAL items before running folds.py or extract.py.")
        sys.exit(1)
    print("\nManifest OK." + ("  Read the warnings before extraction." if warnings else ""))


if __name__ == "__main__":
    main()
