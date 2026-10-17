"""
diag_table.py  --  per-corpus diagnostic numbers for the PRL manuscript.

For each corpus it computes, from the UNCORRECTED runs you already have
(no retraining, no GPU):

    q        stranded fraction   = share of the comparison arm's TEST clips
                                    whose recording source was dropped from
                                    training (source not in V_keep)
    gap      seen-unseen gap      = macroF1(seen test clips) - macroF1(stranded
                                    test clips), in the uncorrected arm
    f1_seen  macroF1 on clips whose source is retained (in V_keep)
    f1_unseen macroF1 on stranded clips (source dropped)

These are exactly the two drivers in the simulation (Fig. 3a): the paper's
delta is small on a corpus precisely when `gap` is small. Pair the output with
the delta column already in Table 5 to fill the (q, gap, delta) diagnostic
table the reviewers asked for.

q is model-independent (a property of the split), so it is taken once per draw.
gap is model-dependent (it is an F1), so it is averaged over encoders within a
draw, then summarised (mean, SD) across draws.

----------------------------------------------------------------------------
WHAT THE SCRIPT NEEDS FROM YOUR FILES
----------------------------------------------------------------------------
For every (corpus, encoder, draw) in the UNCORRECTED runs, one per-clip
predictions table with, at minimum:

    utt_id      clip id            (any unique string)
    source_id   recording source v (program / city / center)
    y_true      gold label
    y_pred      predicted label
    arm         which comparison arm this row is scored under:
                "speaker" (speaker-disjoint) for the speech corpora,
                "random" for Camelyon17.   [optional -- see CONFIG]

AND a way to know which sources were retained (V_keep) for that draw, via
EITHER:
  (a) a boolean column in the predictions file saying the clip's source was
      retained  (e.g. `retained` / `seen` / `in_vkeep`), OR
  (b) a per-draw file listing the retained source ids
      (e.g. vkeep.json, retained_sources.txt) sitting in the run dir.

If your columns/filenames differ, change ONLY the CONFIG block below.
Run with --dry-run first: it prints every file it matched and the columns it
found, and computes nothing, so you can fix the mapping before a real run.

----------------------------------------------------------------------------
USAGE
----------------------------------------------------------------------------
    python src/diag_table.py --runs "results_uncorr*_seed*" --dry-run
    python src/diag_table.py --runs "results_uncorr*_seed*" --out diag.csv
    python src/diag_table.py --runs "results_uncorr*_seed*" --out diag.csv \
        --delta-from results_srcmatch     # optional: also pull delta if saved

Supported prediction formats: .csv, .tsv, .parquet, .json / .jsonl, .npz.
Only numpy and (for parquet) pandas are required; CSV/JSON use the stdlib.
"""
import argparse
import glob
import json
import os
import re
import sys
from collections import defaultdict

import numpy as np

# ===========================================================================
# CONFIG  --  edit here if your files use different names
# ===========================================================================

# Column names in the per-clip predictions table (first match wins).
COL = {
    "utt_id":    ["utt_id", "id", "clip_id", "utterance"],
    "source_id": ["source_id", "source", "v", "program", "city", "center"],
    "y_true":    ["y_true", "label", "gold", "target", "y"],
    "y_pred":    ["y_pred", "pred", "prediction", "yhat"],
    "arm":       ["arm", "protocol", "split"],         # optional
    "retained":  ["retained", "seen", "in_vkeep", "kept"],  # optional (bool)
}

# Which comparison arm to diagnose, by corpus. The conventional arm in the
# paper is speaker-disjoint for speech and the random split for Camelyon17.
# Matched by case-insensitive substring of the corpus name.
ARM_BY_CORPUS = {
    "camelyon": "random",
    "aydid":    "speaker",
    "sada":     "speaker",
    "kespeech": "speaker",
}
DEFAULT_ARM = "speaker"

# How to read (corpus, encoder, draw/seed) when they are NOT columns in the
# file: parse them from the file path. Named groups corpus/encoder/seed.
# Adjust to match your directory naming. Example path this matches:
#   results_uncorr_cam_seed3/camelyon.resnet50.preds.csv
PATH_META = re.compile(
    r"seed(?P<seed>\d+).*?[\\/](?P<corpus>[A-Za-z0-9]+)\.(?P<encoder>[A-Za-z0-9_\-]+)\."
)

# Per-draw retained-source file names to look for inside each run dir
# (used only if the predictions file has no `retained` column).
VKEEP_NAMES = ["vkeep.json", "retained_sources.json", "retained_sources.txt",
               "vkeep.txt"]

# macroF1 over: "present" = classes present in that subset's y_true (default,
# robust for the stranded subset which may miss classes); "fullK" = the corpus
# class set inferred from the union of y_true across the whole draw.
MACRO_F1_OVER = "present"

# ===========================================================================
# end CONFIG
# ===========================================================================


def _pick(colnames, keys):
    low = {c.lower(): c for c in colnames}
    for k in keys:
        if k.lower() in low:
            return low[k.lower()]
    return None


def _is_seq(v):
    return isinstance(v, (list, tuple, np.ndarray))


def _dict_of_arrays_to_rows(obj):
    """If obj is a flat dict whose values are equal-length sequences, expand to
    rows. Otherwise return None (not a per-clip table)."""
    if not isinstance(obj, dict) or not obj:
        return None
    vals = list(obj.values())
    if not all(_is_seq(v) for v in vals):
        return None
    lens = {len(v) for v in vals}
    if len(lens) != 1:
        return None
    n = lens.pop()
    keys = list(obj.keys())
    return [{k: obj[k][i] for k in keys} for i in range(n)]


def _structure(obj, depth=0):
    """Short human-readable sketch of a non-tabular object, for --dry-run."""
    if isinstance(obj, dict):
        items = list(obj.items())[:8]
        inner = ", ".join(f"{k}: {_structure(v, depth + 1)}" for k, v in items)
        more = "" if len(obj) <= 8 else f", +{len(obj) - 8} more"
        return "{" + inner + more + "}"
    if _is_seq(obj):
        return f"[{type(obj).__name__} len={len(obj)}]"
    return type(obj).__name__


def _load_table(path):
    """Return ("rows", list[dict]) for a per-clip table, or ("raw", obj) for a
    non-tabular file (never raises on structure)."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".csv", ".tsv"):
        import csv
        delim = "\t" if ext == ".tsv" else ","
        with open(path, newline="") as f:
            return "rows", list(csv.DictReader(f, delimiter=delim))
    if ext == ".parquet":
        import pandas as pd
        return "rows", pd.read_parquet(path).to_dict("records")
    if ext == ".jsonl":
        with open(path) as f:
            return "rows", [json.loads(line) for line in f if line.strip()]
    if ext == ".json":
        with open(path) as f:
            obj = json.load(f)
        if isinstance(obj, list):
            return "rows", obj
        rows = _dict_of_arrays_to_rows(obj)
        return ("rows", rows) if rows is not None else ("raw", obj)
    if ext == ".npz":
        z = np.load(path, allow_pickle=True)
        obj = {k: z[k] for k in z.keys()}
        rows = _dict_of_arrays_to_rows(obj)
        return ("rows", rows) if rows is not None else ("raw",
                {k: f"ndarray{z[k].shape}" for k in z.keys()})
    return "raw", f"unsupported extension {ext}"


def _meta_from_path(path, rows):
    """Resolve corpus/encoder/seed from columns if present, else from path."""
    if rows:
        c = _pick(rows[0].keys(), ["corpus"])
        e = _pick(rows[0].keys(), ["encoder"])
        s = _pick(rows[0].keys(), ["seed", "draw"])
        if c and e and s:
            r = rows[0]
            return str(r[c]), str(r[e]), str(r[s])
    m = PATH_META.search(path.replace("\\", "/"))
    if not m:
        return None
    g = m.groupdict()
    return g["corpus"], g["encoder"], g["seed"]


def _arm_for(corpus):
    cl = corpus.lower()
    for key, arm in ARM_BY_CORPUS.items():
        if key in cl:
            return arm
    return DEFAULT_ARM


def _load_vkeep(run_dir):
    for name in VKEEP_NAMES:
        p = os.path.join(run_dir, name)
        if os.path.exists(p):
            if p.endswith(".json"):
                with open(p) as f:
                    obj = json.load(f)
                vals = obj if isinstance(obj, list) else list(obj.values())[0]
                return set(map(str, vals))
            with open(p) as f:
                return set(x.strip() for x in f if x.strip())
    return None


def _macro_f1(y_true, y_pred, classes):
    f1s = []
    for c in classes:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == c and p == c)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != c and p == c)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == c and p != c)
        if tp + fp == 0 or tp + fn == 0:
            f1s.append(0.0)
            continue
        prec = tp / (tp + fp)
        rec = tp / (tp + fn)
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s)) if f1s else float("nan")


def _diagnose_one(rows, cols, vkeep, arm):
    """Return (q, f1_seen, f1_unseen) for one (corpus,encoder,draw) file."""
    # filter to the chosen comparison arm if an arm column exists
    if cols["arm"]:
        rows = [r for r in rows if str(r[cols["arm"]]).lower().startswith(arm[:4])]
    if not rows:
        return None

    def seen_flag(r):
        if cols["retained"]:
            return str(r[cols["retained"]]).lower() in ("1", "true", "yes", "t")
        return str(r[cols["source_id"]]) in vkeep

    seen, unseen = [], []
    for r in rows:
        (seen if seen_flag(r) else unseen).append(r)

    n = len(rows)
    q = len(unseen) / n if n else float("nan")

    if MACRO_F1_OVER == "fullK":
        classes = sorted({str(r[cols["y_true"]]) for r in rows})
    else:
        classes = None  # per-subset below

    def f1(subset):
        if not subset:
            return float("nan")
        yt = [str(r[cols["y_true"]]) for r in subset]
        yp = [str(r[cols["y_pred"]]) for r in subset]
        cl = classes if classes is not None else sorted(set(yt))
        return _macro_f1(yt, yp, cl)

    return q, f1(seen), f1(unseen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True,
                    help='glob for UNCORRECTED run dirs, e.g. "results_uncorr*_seed*"')
    ap.add_argument("--preds-glob", default="*.csv;*.tsv;*.parquet;*.json;*.jsonl;*.npz",
                    help="';'-separated filename patterns for prediction files inside each run dir")
    ap.add_argument("--out", default="diag_table.csv")
    ap.add_argument("--dry-run", action="store_true",
                    help="list matched files + detected columns, compute nothing")
    a = ap.parse_args()

    run_dirs = sorted(d for d in glob.glob(a.runs) if os.path.isdir(d))
    if not run_dirs:
        raise SystemExit(f"no run dirs matched: {a.runs}")

    pred_files = []
    for d in run_dirs:
        for pat in a.preds_glob.split(";"):
            pred_files.extend(glob.glob(os.path.join(d, "**", pat.strip()),
                                        recursive=True))
    # never treat a V_keep sidecar as a predictions table
    _vk = {n.lower() for n in VKEEP_NAMES}
    pred_files = sorted(set(p for p in pred_files
                            if os.path.basename(p).lower() not in _vk))
    if not pred_files:
        raise SystemExit("no prediction files found inside the run dirs; "
                         "check --preds-glob")

    # corpus -> {draw -> {"q": x, "gaps": [per-encoder gaps], "fs":[], "fu":[]}}
    agg = defaultdict(lambda: defaultdict(lambda: {"q": None, "gaps": [],
                                                   "fs": [], "fu": []}))
    problems = []

    for path in pred_files:
        try:
            kind, data = _load_table(path)
        except Exception as e:
            if a.dry_run:
                print(f"\n{path}\n  ERROR reading: {e}")
            else:
                problems.append(f"{path}: {e}")
            continue

        if kind == "raw":
            if a.dry_run:
                print(f"\n{path}\n  NOT a per-clip table. structure:")
                print("  " + _structure(data))
            else:
                problems.append(f"{path}: not a per-clip predictions table "
                                f"(aggregate/report file); narrow --preds-glob")
            continue

        rows = data
        if not rows or not isinstance(rows[0], dict):
            problems.append(f"{path}: empty or not row-records")
            continue

        cols = {k: _pick(rows[0].keys(), v) for k, v in COL.items()}
        meta = _meta_from_path(path, rows)

        if a.dry_run:
            print(f"\n{path}")
            print(f"  columns: {list(rows[0].keys())}")
            print(f"  mapped : {cols}")
            print(f"  meta   : {meta}")
            continue

        missing = [k for k in ("source_id", "y_true", "y_pred") if not cols[k]]
        if missing:
            problems.append(f"{path}: missing columns {missing}")
            continue
        if meta is None:
            problems.append(f"{path}: could not resolve corpus/encoder/seed "
                            f"(add columns or fix PATH_META)")
            continue
        corpus, encoder, seed = meta
        arm = _arm_for(corpus)

        vkeep = None
        if not cols["retained"]:
            vkeep = _load_vkeep(os.path.dirname(path)) or _load_vkeep(
                os.path.dirname(os.path.dirname(path)))
            if vkeep is None:
                problems.append(f"{path}: no `retained` column and no V_keep "
                                f"file ({VKEEP_NAMES}) in the run dir")
                continue

        res = _diagnose_one(rows, cols, vkeep, arm)
        if res is None:
            problems.append(f"{path}: no rows for arm '{arm}'")
            continue
        q, fs, fu = res
        slot = agg[corpus][seed]
        slot["q"] = q  # model-independent; last writer == same value
        if not (fs != fs or fu != fu):  # both finite
            slot["gaps"].append(fs - fu)
            slot["fs"].append(fs)
            slot["fu"].append(fu)

    if a.dry_run:
        return

    # summarise per corpus across draws (gap averaged over encoders per draw)
    def ms(xs):
        xs = [x for x in xs if x == x]
        if not xs:
            return float("nan"), float("nan")
        return float(np.mean(xs)), float(np.std(xs, ddof=1) if len(xs) > 1 else 0.0)

    rows_out = []
    for corpus in sorted(agg):
        draws = agg[corpus]
        qs = [d["q"] for d in draws.values() if d["q"] is not None]
        per_draw_gap = [float(np.mean(d["gaps"])) for d in draws.values() if d["gaps"]]
        per_draw_fs = [float(np.mean(d["fs"])) for d in draws.values() if d["fs"]]
        per_draw_fu = [float(np.mean(d["fu"])) for d in draws.values() if d["fu"]]
        q_m, q_s = ms(qs)
        g_m, g_s = ms(per_draw_gap)
        fs_m, _ = ms(per_draw_fs)
        fu_m, _ = ms(per_draw_fu)
        rows_out.append((corpus, len(draws), q_m, q_s, g_m, g_s, fs_m, fu_m))

    hdr = ["corpus", "n_draws", "q_mean", "q_sd", "gap_mean", "gap_sd",
           "f1_seen_mean", "f1_unseen_mean"]
    with open(a.out, "w") as f:
        f.write(",".join(hdr) + "\n")
        for r in rows_out:
            f.write(f"{r[0]},{r[1]}," + ",".join(f"{x:.4f}" for x in r[2:]) + "\n")

    print(f"wrote {a.out}\n")
    print(f"{'corpus':<12}{'draws':>6}{'q':>9}{'gap':>10}{'F1_seen':>10}{'F1_unseen':>11}")
    for r in rows_out:
        print(f"{r[0]:<12}{r[1]:>6}{r[2]:>9.3f}{r[4]:>10.3f}{r[6]:>10.3f}{r[7]:>11.3f}")

    if problems:
        print("\nSKIPPED / ISSUES (fix CONFIG and re-run):")
        for p in problems:
            print("  " + p)
        print("\nTip: run with --dry-run to see columns and path parsing.")


if __name__ == "__main__":
    main()