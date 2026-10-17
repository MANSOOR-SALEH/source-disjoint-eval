"""
Build a pipeline manifest for ANY corpus from an audio folder + a metadata CSV,
and report whether its recording provenance is a strong enough shortcut for the
correction to fire.

    python src/make_manifest.py \
        --audio-root /data/NEWCORPUS --metadata /data/NEWCORPUS/meta.csv \
        --id-col clip --path-col file --label-col dialect \
        --speaker-col spk --source-col channel \
        --out manifests/newcorpus.csv

Emits the columns the rest of the pipeline requires:
    utt_id, path, label, speaker_id, source_id   (+ duration)
`path` is stored relative to --audio-root. Clips outside [1, 30] s are dropped
(matching the SPL/PRL setup); pass --no-dur-filter to keep them.

Diagnostics printed (this is the Step-1 decision aid)
-----------------------------------------------------
* clips / classes / speakers / sources, sources per class
* whether sources are NESTED in label (each source carries one class) or CROSSED
* provenance predictivity: how well source identity alone predicts the label
  (normalised mutual information I(label; source) / H(label), 0..1). A corpus
  where this is high AND sources are crossed with labels is exactly where
  holding provenance out changes the score -- i.e. where delta != 0. AYDID/SADA
  sit near the low end, which is why their delta is ~0.
"""
import argparse
import math
import os
from collections import Counter

import numpy as np
import pandas as pd


def _nmi_label_given_source(labels, sources):
    """Normalised mutual information I(Y;S)/H(Y) in [0,1]."""
    lab = pd.Series(labels)
    src = pd.Series(sources)
    n = len(lab)
    hy = _entropy(lab.value_counts().values / n)
    if hy == 0:
        return 0.0
    mi = 0.0
    pS = src.value_counts() / n
    for s, ps in pS.items():
        sub = lab[src == s]
        pY_s = sub.value_counts() / len(sub)
        h_y_s = _entropy(pY_s.values)
        mi += ps * (hy - h_y_s)      # sum_s p(s)[H(Y)-H(Y|s)] = I(Y;S)
    return float(max(0.0, min(1.0, mi / hy)))


def _entropy(p):
    p = np.asarray([x for x in p if x > 0], float)
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def _duration(path):
    try:
        import soundfile as sf
        info = sf.info(path)
        return info.frames / float(info.samplerate)
    except Exception:
        return float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-root", default="")
    ap.add_argument("--metadata", required=True, help="CSV of per-clip metadata")
    ap.add_argument("--id-col", required=True)
    ap.add_argument("--path-col", required=True)
    ap.add_argument("--label-col", required=True)
    ap.add_argument("--speaker-col", required=True)
    ap.add_argument("--source-col", required=True,
                    help="recording-provenance key: channel / programme / session / device / site")
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--no-dur-filter", action="store_true")
    ap.add_argument("--no-measure-dur", action="store_true",
                    help="skip reading audio headers (no duration column, no filter)")
    a = ap.parse_args()

    md = pd.read_csv(a.metadata)
    for c in (a.id_col, a.path_col, a.label_col, a.speaker_col, a.source_col):
        if c not in md.columns:
            raise SystemExit(f"column '{c}' not in {a.metadata}; have {list(md.columns)}")

    df = pd.DataFrame({
        "utt_id": md[a.id_col].astype(str),
        "path": md[a.path_col].astype(str),
        "label": md[a.label_col].astype(str),
        "speaker_id": md[a.speaker_col].astype(str),
        "source_id": md[a.source_col].astype(str),
    })
    # normalise path to be relative to audio-root
    if a.audio_root:
        df["path"] = df["path"].apply(
            lambda p: os.path.relpath(p, a.audio_root) if os.path.isabs(p) else p)

    if not a.no_measure_dur:
        df["duration"] = [_duration(os.path.join(a.audio_root, p)) for p in df["path"]]
        if not a.no_dur_filter:
            before = len(df)
            df = df[(df.duration >= a.min_dur) & (df.duration <= a.max_dur)]
            df = df.reset_index(drop=True)
            print(f"duration filter [{a.min_dur},{a.max_dur}]s: kept {len(df)}/{before}")

    if df.utt_id.duplicated().any():
        raise SystemExit("duplicate utt_id in metadata")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    df.to_csv(a.out, index=False)

    # ---- Step-1 diagnostics ------------------------------------------------
    spc = df.groupby("label").source_id.nunique()
    nested = int(df.groupby("source_id").label.nunique().max()) == 1
    nmi = _nmi_label_given_source(df.label.values, df.source_id.values)
    print(f"\nwrote {a.out}: {len(df)} clips")
    print(f"  classes  : {df.label.nunique()}  ({dict(df.label.value_counts())})")
    print(f"  speakers : {df.speaker_id.nunique()}")
    print(f"  sources  : {df.source_id.nunique()}")
    print(f"  sources/class: min {int(spc.min())}, median {int(spc.median())}, "
          f"max {int(spc.max())}")
    print(f"  source-label structure: {'NESTED (source -> one class)' if nested else 'CROSSED'}")
    print(f"  provenance predictivity  I(label;source)/H(label) = {nmi:.2f}")
    verdict = ("HIGH - holding provenance out should move the score; good corpus "
               "for a non-null delta" if nmi >= 0.30 else
               "LOW - provenance barely predicts the label; expect delta ~ 0, "
               "like AYDID/SADA")
    print(f"  --> {verdict}")
    if spc.min() < 2:
        print("  !! a class has <2 sources: no grouped source-disjoint split exists "
              "for it. Merge that class or collect more provenance metadata.")


if __name__ == "__main__":
    main()
