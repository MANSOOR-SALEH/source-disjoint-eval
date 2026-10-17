"""
Build a pipeline manifest for WILDS Camelyon17 and report whether recording
provenance (hospital/center) is a strong shortcut there.

    python src/make_camelyon_manifest.py \
        --root /data/camelyon17_v1.0 --out manifests/camelyon.csv \
        --per-group 3000 --seed 0

Camelyon17 maps onto the source-disjoint protocol directly:
    label      = tumor            (binary: 0 normal / 1 tumor)
    speaker_id = patient          -> patient-group-disjoint arm (unseen patients)
    source_id  = center           -> center(hospital)-disjoint arm (unseen sites)
Center is the documented provenance shortcut (staining / scanner differences),
and tumor label is crossed with center (every center has both classes), so
holding center out separates recording source from label---the de-confounded
case the Arabic corpora cannot give.

Input: the WILDS Camelyon17 release directory, which contains `metadata.csv`
with columns (patient, node, x_coord, y_coord, tumor, slide, center, split) and
a `patches/` tree. `path` is written relative to that directory.

Output columns: utt_id, path, label, speaker_id, source_id  (schema shared with
make_aydid_manifest.py / make_kespeech_manifest.py).
"""
import argparse
import math
import os

import numpy as np
import pandas as pd


def _patch_path(r):
    # WILDS layout: patches/patient_XXX_node_Y/patch_patient_XXX_node_Y_x_..._y_....png
    p, n = int(r["patient"]), int(r["node"])
    stem = f"patient_{p:03d}_node_{n}"
    return (f"patches/{stem}/patch_{stem}_x_{int(r['x_coord'])}"
            f"_y_{int(r['y_coord'])}.png")


def _nmi_label_given_source(labels, sources):
    lab, src = pd.Series(labels), pd.Series(sources)
    n = len(lab)
    hy = _entropy(lab.value_counts().values / n)
    if hy == 0:
        return 0.0
    mi, pS = 0.0, src.value_counts() / n
    for s, ps in pS.items():
        sub = lab[src == s]
        mi += ps * (hy - _entropy(sub.value_counts().values / len(sub)))
    return float(max(0.0, min(1.0, mi / hy)))


def _entropy(p):
    p = np.asarray([x for x in p if x > 0], float)
    return float(-(p * np.log(p)).sum()) if len(p) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="WILDS camelyon17_v1.0 directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--per-group", type=int, default=3000,
                    help="max patches kept per (center, label) cell, for a "
                         "balanced and tractable subsample")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    meta = os.path.join(a.root, "metadata.csv")
    df = pd.read_csv(meta)
    df.columns = [c.strip() for c in df.columns]
    need = {"patient", "node", "x_coord", "y_coord", "tumor", "center"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"metadata.csv missing {sorted(missing)}; have {list(df.columns)}")

    out = pd.DataFrame({
        "utt_id": ("p" + df.patient.astype(int).astype(str)
                   + "_n" + df.node.astype(int).astype(str)
                   + "_x" + df.x_coord.astype(int).astype(str)
                   + "_y" + df.y_coord.astype(int).astype(str)),
        "path": df.apply(_patch_path, axis=1),
        "label": df.tumor.astype(int).map({0: "normal", 1: "tumor"}),
        "speaker_id": "pt" + df.patient.astype(int).astype(str),
        "source_id": "center" + df.center.astype(int).astype(str),
    })

    # balanced, tractable subsample: cap each (center, label) cell
    rng = np.random.default_rng(a.seed)
    keep = []
    for (_, _), g in out.groupby([df.center.values, out.label]):
        idx = g.index.to_numpy()
        if len(idx) > a.per_group:
            idx = rng.choice(idx, a.per_group, replace=False)
        keep.extend(idx.tolist())
    out = out.loc[sorted(keep)].reset_index(drop=True)
    if out.utt_id.duplicated().any():
        out = out.drop_duplicates("utt_id").reset_index(drop=True)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    out.to_csv(a.out, index=False)

    # ---- Step-1 diagnostics ----
    spc = out.groupby("label").source_id.nunique()
    nested = int(out.groupby("source_id").label.nunique().max()) == 1
    nmi = _nmi_label_given_source(out.label.values, out.source_id.values)
    print(f"wrote {a.out}: {len(out)} patches")
    print(f"  classes : {dict(out.label.value_counts())}")
    print(f"  patients: {out.speaker_id.nunique()}   centers: {out.source_id.nunique()}")
    print(f"  per-class center coverage: {dict(spc)}")
    print(f"  source-label structure: {'NESTED' if nested else 'CROSSED'}")
    print(f"  label-from-center MI  I(label;center)/H(label) = {nmi:.2f}")
    print("  NOTE: on Camelyon17 the shortcut is domain shift (staining/scanner),")
    print("        NOT center predicting the label, so a LOW MI here is correct and")
    print("        desirable---it means center is crossed with label (de-confounded).")
    print("        Whether provenance is a strong shortcut is decided by the run:")
    print("        a large center-disjoint macro-F1 drop vs patient-disjoint.")


if __name__ == "__main__":
    main()
