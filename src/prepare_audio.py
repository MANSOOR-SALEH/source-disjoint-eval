"""
Audit and normalise corpus audio format.

    # 1. look before you touch anything
    python src/prepare_audio.py --manifest manifests/aydid.csv \
        --audio-root F:/.../WAVS --audit-only

    # 2. convert what needs converting
    python src/prepare_audio.py --manifest manifests/aydid.csv \
        --audio-root F:/.../WAVS --out-root F:/.../WAVS_16k

Why the audit comes first
-------------------------
Format variation is not a nuisance here, it is data. A 44.1 kHz recording
downsampled to 16 kHz has been through an anti-alias filter, so its spectral
content near the 8 kHz Nyquist edge differs from a natively-16 kHz recording.
That is a channel signature. If sample rate correlates with show or dialect --
and it usually does, because it tracks which capture chain produced the file --
then it is exactly the kind of recording-provenance artefact this project
measures, and it must be reported rather than silently normalised away.

So: audit, report the crosstabs, THEN convert, and state in the paper what
fraction of each class was resampled.

Conversion policy
-----------------
Files already at 16 kHz mono are copied byte-for-byte, not re-encoded. Only
non-conforming files are touched. Re-encoding everything would apply a filter
to files that never needed one and would blur the very distinction the audit
just measured.
"""

import argparse
import os
import shutil
import sys

import numpy as np
import pandas as pd
import soundfile as sf
from tqdm import tqdm

TARGET_SR = 16000
TARGET_SUBTYPE = "PCM_16"
MIN_DUR = 1.0     # Paper 1's stated segmentation floor
MAX_DUR = 30.0    # Whisper's context window; beyond this encoders see different audio


def _nonconforming(a):
    """A file conforms only if sample rate, channel count AND encoding match.
    Encoding matters as much as the other two: PCM_16 puts every sample on a
    1/32768 lattice, FLOAT does not, so the two are distinguishable from the
    waveform alone. In this corpus that distinction correlates with dialect."""
    return ((a["sr"] != TARGET_SR) | (a["ch"] != 1) | (a["subtype"] != TARGET_SUBTYPE))


def audit(df, root):
    rows = []
    for p, lab, src in tqdm(zip(df["path"], df["label"], df["source_id"]),
                            total=len(df), desc="audit"):
        full = os.path.join(root, p)
        try:
            i = sf.info(full)
            rows.append(dict(path=p, label=lab, source_id=src, sr=i.samplerate,
                             ch=i.channels, subtype=i.subtype,
                             dur=i.frames / i.samplerate, ok=True))
        except Exception as e:
            rows.append(dict(path=p, label=lab, source_id=src, sr=None, ch=None,
                             subtype=None, dur=None, ok=False, err=str(e)))
    return pd.DataFrame(rows)


def report(a):
    bad = a[~a["ok"]]
    if len(bad):
        print(f"\n{len(bad)} unreadable file(s), e.g. {bad['path'].head(3).tolist()}")
    a = a[a["ok"]].copy()

    print("\n-- sample rate --")
    print(a["sr"].value_counts().to_string())
    print("\n-- channels --")
    print(a["ch"].value_counts().to_string())
    print("\n-- encoding --")
    print(a["subtype"].value_counts().to_string())

    if a["sr"].nunique() > 1:
        print("\n-- sample rate x dialect (THIS IS THE ONE THAT MATTERS) --")
        ct = pd.crosstab(a["label"], a["sr"])
        print(ct.to_string())
        frac = ct.div(ct.sum(axis=1), axis=0)
        print(f"\n  classes where a single rate covers 100%: "
              f"{int((frac.max(axis=1) > 0.999).sum())} of {len(frac)}")
        if (frac.max(axis=1) > 0.999).any() and a["sr"].nunique() > 1:
            print("  !! at least one class is entirely one sample rate while others are\n"
                  "     mixed. Sample rate is then partially predictive of the label, and\n"
                  "     downsampling leaves an anti-alias signature above 8 kHz. Report\n"
                  "     the per-class resampled fraction in the paper.")

        print("\n-- sample rate x source (shows with mixed rates) --")
        by_src = pd.crosstab(a["source_id"], a["sr"])
        mixed = by_src[(by_src > 0).sum(axis=1) > 1]
        if len(mixed):
            print(mixed.to_string())
            print(f"\n  {len(mixed)} show(s) contain more than one sample rate -- so rate\n"
                  "  varies within a source too, not only between sources.")
        else:
            print("  none: sample rate is constant within every show, i.e. it is a\n"
                  "  property of the source. That makes it a clean provenance marker.")

    for col, note in [("ch", "channels"), ("subtype", "encoding")]:
        if a[col].nunique() > 1:
            print(f"\n-- {note} x dialect --")
            print(pd.crosstab(a["label"], a[col]).to_string())
            print(f"  {note} varies across the corpus. Anything that varies WITH the\n"
                  f"  label is a provenance marker, not a nuisance -- report it.")

    print("\n-- duration --")
    d = a["dur"]
    print(f"  mean {d.mean():.2f}s  min {d.min():.2f}s  max {d.max():.2f}s  "
          f"total {d.sum()/3600:.2f} h")
    short, long_ = a[d < MIN_DUR], a[d > MAX_DUR]
    if len(short) or len(long_):
        print(f"\n  outside the [{MIN_DUR}, {MAX_DUR}] s band: "
              f"{len(short)} short, {len(long_)} long")
        band = pd.DataFrame({
            "short": a[d < MIN_DUR].groupby("label").size(),
            "long": a[d > MAX_DUR].groupby("label").size(),
            "total": a.groupby("label").size()}).fillna(0).astype(int)
        print(band.to_string())
        if len(long_):
            print(f"\n  longest clips (Whisper truncates at 30 s, so these are NOT\n"
                  f"  comparable across encoders):")
            for r in long_.nlargest(min(10, len(long_)), "dur").itertuples():
                print(f"    {r.dur:8.1f}s  {r.label}  {r.path}")
        print(f"\n  pass --out-manifest to write a filtered manifest excluding these.")

    nonconf = _nonconforming(a)
    print(f"\n{int(nonconf.sum())} of {len(a)} file(s) differ from "
          f"{TARGET_SR} Hz / mono / {TARGET_SUBTYPE} ({100*nonconf.mean():.1f}%)")
    if nonconf.any():
        print(pd.DataFrame({
            "nonconforming": a[nonconf].groupby("label").size(),
            "total": a.groupby("label").size()}).fillna(0).astype(int).to_string())
        print("  Conversion makes the OUTPUT uniform, which is what the benchmark\n"
              "  needs: otherwise an encoder can score by noticing which pipeline\n"
              "  wrote the file. The correlation itself still belongs in the paper.")
    return a


def convert(a, root, out_root):
    try:
        import soxr
    except ImportError:
        sys.exit("pip install soxr --break-system-packages")

    conv = _nonconforming(a)
    n_copy = n_res = 0
    for r, needs in tqdm(zip(a.itertuples(), conv), total=len(a), desc="convert"):
        src_p = os.path.join(root, r.path)
        dst_p = os.path.join(out_root, r.path)
        os.makedirs(os.path.dirname(dst_p), exist_ok=True)
        if not needs:
            shutil.copyfile(src_p, dst_p)      # already conforming: leave it alone
            n_copy += 1
            continue
        x, sr = sf.read(src_p, dtype="float32", always_2d=False)
        if x.ndim > 1:
            x = x.mean(axis=1)
        if sr != TARGET_SR:
            x = soxr.resample(x, sr, TARGET_SR, quality="VHQ")
        peak = float(np.abs(x).max())
        if peak > 1.0:                          # resampler overshoot only
            x = x / peak * 0.999
        sf.write(dst_p, x, TARGET_SR, subtype=TARGET_SUBTYPE)
        n_res += 1
    print(f"\ncopied unchanged: {n_copy}   resampled: {n_res}")
    print(f"new audio root: {out_root}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--audio-root", required=True)
    ap.add_argument("--out-root")
    ap.add_argument("--audit-only", action="store_true")
    ap.add_argument("--audit-csv", default="results/audio_audit.csv")
    ap.add_argument("--out-manifest",
                    help="write a manifest filtered to the [MIN_DUR, MAX_DUR] band")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest)
    a = audit(df, args.audio_root)
    os.makedirs(os.path.dirname(args.audit_csv) or ".", exist_ok=True)
    a.to_csv(args.audit_csv, index=False)
    ok = report(a)
    print(f"\nper-file audit written to {args.audit_csv}")

    if args.out_manifest:
        keep = ok[(ok["dur"] >= MIN_DUR) & (ok["dur"] <= MAX_DUR)]["path"]
        out = df[df["path"].isin(set(keep))]
        out.to_csv(args.out_manifest, index=False)
        print(f"\nfiltered manifest: {len(out)} of {len(df)} rows -> {args.out_manifest}")
        print("  rebuild folds from this manifest, and re-extract against it.")
        print(pd.DataFrame({"kept": out.groupby("label").size(),
                            "dropped": df.groupby("label").size() - out.groupby("label").size()
                            }).to_string())

    if args.audit_only:
        return
    if not args.out_root:
        sys.exit("give --out-root to convert, or --audit-only to stop here")
    if not ok["ok"].all():
        sys.exit("unreadable files present -- resolve before converting")
    convert(ok, args.audio_root, args.out_root)


if __name__ == "__main__":
    main()