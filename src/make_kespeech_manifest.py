"""
Build a KeSpeech manifest for source-disjoint evaluation, with city as the
recording-provenance variable.

    python src/make_kespeech_manifest.py --root F:/DID_ASR_Datasets/KeSpeech \
        --out manifests/kespeech.csv --per-class 2000

KeSpeech gives this project its only non-Arabic corpus and its only
geographically-defined source variable. Both matter: two Arabic corpora that
differ 6x in the cost of source-disjoint evaluation is an observation; a third
corpus in another language turns it into an ordering.

What this script has to get right
---------------------------------
1. Only phase1 carries dialect labels (590,139 of 976,054 wavs). Phase2 is
   dropped.
2. Classes are kept only if they have >= --min-cities distinct cities, since a
   grouped split needs at least that many groups. Reported, not silently applied.
3. The subsample is stratified BY CITY WITHIN CLASS, with a cap on any one
   city's share. This is the fix for the failure diagnosed on the nested SADA
   subset, where one source held 82% of a class and holding it out starved that
   class out of training. Here no city may exceed --max-city-share of its class.
4. Duration comes from utt2dur; clips outside [1, 30] s are dropped so every
   encoder sees the same audio (Whisper truncates at 30 s).

A caveat the paper must state
-----------------------------
For Mandarin sub-dialects, city is partly CONSTITUTIVE of the label: Jiang-Huai
and Zhongyuan are geographic categories, so holding out a city removes a
recording location and also genuine within-dialect linguistic variation. The
source-disjoint drop therefore upper-bounds recording-source leakage rather than
isolating it. AYDID has the same property at show level, so the treatment is
consistent across corpora -- but it should be said, not assumed.

Expected KeSpeech layout (the script probes for variants):
    <root>/Tasks/ASR/{train_phase1,dev_phase1,test}/  wav.scp, utt2subdialect
    <root>/Metadata/  spk2city, utt2dur, spk2gender, spk2age
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

MIN_DUR, MAX_DUR = 1.0, 30.0


def find(root, *names, prefer=None):
    """Locate a metadata file by name under root.

    os.walk order is arbitrary, and KeSpeech ships the same filenames in
    train_phase1, dev_phase1 and test. Taking the first hit silently picked the
    2k-utterance dev split instead of the 590k train split, so candidates are
    collected and ranked: a path containing `prefer` wins, then the largest file.
    """
    hits = []
    for dirpath, _, files in os.walk(root):
        for n in names:
            if n in files:
                hits.append(os.path.join(dirpath, n))
    if not hits:
        return None
    if prefer:
        pref = [h for h in hits if prefer in h.replace("\\", "/")]
        if pref:
            hits = pref
    return max(hits, key=os.path.getsize)


def read_kv(path, cols):
    """KeSpeech metadata is whitespace-separated key/value, no header."""
    return pd.read_csv(path, sep=r"\s+", header=None, names=cols,
                       dtype=str, engine="python", quoting=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", default="manifests/kespeech.csv")
    ap.add_argument("--per-class", type=int, default=2000,
                    help="target clips per class after subsampling")
    ap.add_argument("--min-cities", type=int, default=4,
                    help="drop classes with fewer distinct cities than this")
    ap.add_argument("--max-city-share", type=float, default=0.35,
                    help="no single city may exceed this share of a class; the "
                         "guard against the class-starvation failure mode")
    ap.add_argument("--exclude", default="",
                    help="comma-separated labels to drop, e.g. Beijing")
    ap.add_argument("--split", default="train_phase1",
                    help="which KeSpeech split to read; only phase1 carries "
                         "dialect labels, and train_phase1 is the large one "
                         "(dev_phase1 has ~2k utts and will silently give you a "
                         "two-class corpus)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    # ---- locate metadata -------------------------------------------------
    paths = {}
    for key, names, pref in [("wav", ("wav.scp",), a.split),
                             ("dialect", ("utt2subdialect", "utt2dialect"), a.split),
                             ("spk", ("utt2spk",), None),
                             ("city", ("spk2city",), None),
                             ("dur", ("utt2dur",), None)]:
        p = find(a.root, *names, prefer=pref)
        if p is None:
            sys.exit(f"could not find {names[0]} under {a.root}")
        paths[key] = p
        print(f"  {key:8s} {p}")

    wav = read_kv(paths["wav"], ["utt_id", "path"])
    dia = read_kv(paths["dialect"], ["utt_id", "label"])
    spk = read_kv(paths["spk"], ["utt_id", "speaker_id"])
    city = read_kv(paths["city"], ["speaker_id", "source_id"])
    dur = read_kv(paths["dur"], ["utt_id", "dur"])
    dur["dur"] = pd.to_numeric(dur["dur"], errors="coerce")

    df = (wav.merge(dia, on="utt_id")
             .merge(spk, on="utt_id")
             .merge(city, on="speaker_id", how="left")
             .merge(dur, on="utt_id", how="left"))
    print(f"\njoined {len(df)} labelled utterances "
          f"({df['label'].nunique()} classes, {df['source_id'].nunique()} cities)")
    if len(df) < 50000:
        print(f"  !! only {len(df)} utterances -- phase1 train should be ~590k.\n"
              f"     Check that --split matched the intended directory above.")

    miss = df["source_id"].isna()
    if miss.any():
        print(f"  dropping {int(miss.sum())} utt(s) whose speaker has no city")
        df = df[~miss]

    if df["dur"].notna().any():
        bad = (df["dur"] < MIN_DUR) | (df["dur"] > MAX_DUR)
        print(f"  dropping {int(bad.sum())} clip(s) outside [{MIN_DUR}, {MAX_DUR}] s")
        df = df[~bad]

    # ---- class eligibility ----------------------------------------------
    print(f"\n-- cities per class --")
    per = df.groupby("label")["source_id"].nunique().sort_values()
    for lab, n in per.items():
        print(f"  {'ok  ' if n >= a.min_cities else 'DROP'} {str(lab):22s} {n:>3d} cities  "
              f"{int((df['label'] == lab).sum()):>7d} utts")
    keep = per[per >= a.min_cities].index
    if a.exclude:
        ex = {e.strip() for e in a.exclude.split(",") if e.strip()}
        keep = [k for k in keep if k not in ex]
        print(f"  excluded by request: {sorted(ex)}")
    df = df[df["label"].isin(keep)]
    print(f"  -> keeping {len(keep)} classes; source-disjoint folds capped at "
          f"{int(per[list(keep)].min())}")

    # ---- stratified subsample, capped per city ---------------------------
    # Draw evenly across a class's cities, but never let one city exceed
    # max_city_share of the class. This is what prevents the starvation seen on
    # the nested SADA subset.
    out, shrunk = [], []
    for lab, sub in df.groupby("label"):
        counts = sub["source_id"].value_counts()
        cities = list(counts.index)

        # The cap is a HARD constraint, so the achievable class size is bounded
        # by what the cap allows given the available cities:
        #     max_n such that  min(count_c, max_n * share) summed over c >= max_n
        # A class with too little data in its non-dominant cities is shrunk to
        # honour the cap rather than over-filled from the dominant one. This is
        # the fix for the nested-SADA failure: it is better to have a smaller
        # balanced class than a large one that a single held-out source empties.
        lo, hi = 0, int(counts.sum())
        while lo < hi:
            mid = (lo + hi + 1) // 2
            avail = int(np.minimum(counts.to_numpy(), int(mid * a.max_city_share)).sum())
            if avail >= mid:
                lo = mid
            else:
                hi = mid - 1
        target = min(a.per_class, lo)
        if target < a.per_class:
            shrunk.append((lab, a.per_class, target))

        cap = max(1, int(target * a.max_city_share))

        # Water-filling: every city contributes, and the allocation is levelled
        # rather than greedy. Filling greedily from the smallest cities hits the
        # target before reaching the large ones, which silently drops cities --
        # and cities are the unit the source-disjoint protocol holds out, so
        # losing them shrinks the number of folds and makes each held-out city a
        # larger share of the class. Keep every city; balance the counts.
        alloc = {c: 0 for c in cities}
        remaining = target
        while remaining > 0:
            open_c = [c for c in cities
                      if alloc[c] < min(int(counts[c]), cap)]
            if not open_c:
                break
            step = max(1, remaining // len(open_c))
            for c in open_c:
                room = min(int(counts[c]), cap) - alloc[c]
                take = min(step, room, remaining)
                alloc[c] += take
                remaining -= take
                if remaining <= 0:
                    break

        picked = []
        for c, n in alloc.items():
            if n <= 0:
                continue
            pool = sub[sub["source_id"] == c]
            picked.append(pool.sample(n=n, random_state=int(rng.integers(1e9))))
        out.append(pd.concat(picked))
    sam = pd.concat(out).reset_index(drop=True)
    for lab, want, got in shrunk:
        print(f"  {lab}: reduced {want} -> {got} to keep every city under "
              f"{a.max_city_share:.0%} of the class")

    # ---- report ----------------------------------------------------------
    print(f"\n-- sampled corpus: {len(sam)} utts --")
    rep = sam.groupby("label").agg(utts=("label", "size"),
                                   cities=("source_id", "nunique"),
                                   speakers=("speaker_id", "nunique"))
    rep["max_city_share"] = [
        sam[sam["label"] == l]["source_id"].value_counts(normalize=True).max()
        for l in rep.index]
    print(rep.to_string(float_format=lambda v: f"{v:.2f}"))
    worst = rep["max_city_share"].max()
    print(f"\n  worst max-city share {worst:.2f} "
          f"({'ok' if worst < 0.5 else 'TOO HIGH -- lower --max-city-share'})")
    nested = int(sam.groupby("source_id")["label"].nunique().max()) == 1
    print(f"  cities are {'NESTED IN' if nested else 'CROSSED WITH'} the label")
    ups = sam.groupby("speaker_id").size()
    print(f"  {sam['speaker_id'].nunique()} speakers, {ups.mean():.2f} utts/speaker")

    cross = sam.groupby("speaker_id")["source_id"].nunique()
    print(f"  speakers in >1 city: {int((cross > 1).sum())}")

    final = pd.DataFrame({
        "utt_id": sam["utt_id"],
        "path": sam["path"].str.replace("\\", "/", regex=False),
        "label": sam["label"],
        "speaker_id": sam["speaker_id"],
        "source_id": sam["source_id"],
        "content_type": "read",          # KeSpeech phase1 is read speech
        "environment": "unknown",
    })
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    final.to_csv(a.out, index=False, encoding="utf-8-sig")
    print(f"\nwrote {a.out} ({len(final)} rows)")
    print(f"\nnext:")
    print(f"  python src/check_manifest.py --manifest {a.out} --audio-root <KeSpeech audio root>")
    print(f"  python src/folds.py --manifest {a.out} --out folds/kespeech.json --n-folds 10")


if __name__ == "__main__":
    main()