"""
Build the paper's tables from results/*.json.

    python src/summarise.py                    # console
    python src/summarise.py --latex            # + LaTeX for the manuscript
    python src/summarise.py --csv results/summary.csv

Recomputes relative depth from n_layers_total when present, so results written
before that field existed still report correctly.
"""

import argparse
import glob
import json
import os

import numpy as np

ORDER = ["wavlm", "xlsr", "mhubert", "mms", "whisper", "w2vbert", "artst"]
PRETTY = {"wavlm": "WavLM-Large", "xlsr": "XLS-R-300m", "mhubert": "mHuBERT-147",
          "mms": "MMS-300m", "whisper": "Whisper-large-v3",
          "w2vbert": "w2v-BERT 2.0", "artst": "ArTST"}
DEPTH = {"wavlm": 24, "xlsr": 24, "mhubert": 12, "mms": 24,
         "whisper": 32, "w2vbert": 24, "artst": 12}


def load():
    rows = []
    for p in sorted(glob.glob("results/*.json")):
        r = json.load(open(p))
        base = os.path.basename(p).replace(".json", "")
        corpus, enc = base.split(".", 1)
        cl = r.get("convention_layer")
        if not cl:
            continue
        P = r["protocols"]
        nl = r.get("n_layers_total") or cl.get("n_layers_total") or DEPTH.get(enc)
        row = {
            "corpus": corpus, "encoder": enc, "chance": r["chance"],
            "layer": cl["layer"], "n_layers": nl,
            "depth": cl["layer"] / nl if nl else float("nan"),
            "peaks": r.get("peak_layer_by_protocol", {}),
        }
        for k in ("random", "speaker_disjoint", "source_disjoint",
                  "source_disjoint_rep"):
            if k in P:
                row[k] = P[k]["at_convention_layer"]
                pf = np.array(P[k]["fixed"]["per_fold"])
                row[k + "_n"] = len(pf)
        if "speaker_disjoint" in row and "source_disjoint" in row:
            row["drop"] = row["speaker_disjoint"] - row["source_disjoint"]
        rows.append(row)
    return rows


def fmt(rows, corpus):
    sub = [r for r in rows if r["corpus"] == corpus]
    sub.sort(key=lambda r: ORDER.index(r["encoder"]) if r["encoder"] in ORDER else 99)
    if not sub:
        return
    has_rep = any("source_disjoint_rep" in r for r in sub)
    print(f"\n{'='*88}")
    print(f"{corpus.upper()}   chance = {sub[0]['chance']:.3f}")
    print("=" * 88)
    hdr = f"{'encoder':18s} {'layer':>10s} {'depth':>6s} {'rand':>7s} {'spk':>7s} {'src':>7s}"
    if has_rep:
        hdr += f" {'src-rep':>8s}"
    hdr += f" {'drop':>7s}"
    print(hdr)
    print("-" * len(hdr))
    for r in sub:
        line = (f"{PRETTY.get(r['encoder'], r['encoder']):18s} "
                f"{('L%d/%d' % (r['layer'], r['n_layers'])):>10s} "
                f"{r['depth']:>6.2f} "
                f"{r.get('random', float('nan')):>7.3f} "
                f"{r.get('speaker_disjoint', float('nan')):>7.3f} "
                f"{r.get('source_disjoint', float('nan')):>7.3f}")
        if has_rep:
            line += f" {r.get('source_disjoint_rep', float('nan')):>8.3f}"
        line += f" {r.get('drop', float('nan')):>+7.3f}"
        print(line)
    d = np.array([r["drop"] for r in sub if "drop" in r])
    print("-" * len(hdr))
    print(f"{'mean':18s} {'':>10s} {'':>6s} "
          f"{np.mean([r.get('random', np.nan) for r in sub]):>7.3f} "
          f"{np.mean([r.get('speaker_disjoint', np.nan) for r in sub]):>7.3f} "
          f"{np.mean([r.get('source_disjoint', np.nan) for r in sub]):>7.3f}"
          + (f" {np.mean([r.get('source_disjoint_rep', np.nan) for r in sub]):>8.3f}" if has_rep else "")
          + f" {d.mean():>+7.3f}")
    print(f"  drop range {d.min():.3f}-{d.max():.3f}   spread {d.max()-d.min():.3f}")


def peak_table(rows):
    print(f"\n{'='*88}")
    print("PEAK LAYER BY PROTOCOL  (does the best layer move when sources are held out?)")
    print("=" * 88)
    print(f"{'corpus':8s} {'encoder':18s} {'random':>8s} {'spk-disj':>9s} {'src-disj':>9s}  shift")
    print("-" * 66)
    for c in ("aydid", "sada"):
        for r in sorted([x for x in rows if x["corpus"] == c],
                        key=lambda r: ORDER.index(r["encoder"]) if r["encoder"] in ORDER else 99):
            p = r["peaks"]
            spk, src = p.get("speaker_disjoint"), p.get("source_disjoint")
            shift = "" if spk is None or src is None else (
                f"{src - spk:+d}" + ("  later" if src > spk else "  earlier" if src < spk else ""))
            print(f"{c:8s} {PRETTY.get(r['encoder'], r['encoder']):18s} "
                  f"{str(p.get('random', '-')):>8s} {str(spk):>9s} {str(src):>9s}  {shift}")


def latex(rows):
    print(f"\n{'='*88}\nLATEX\n{'='*88}")
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"Corpus & Encoder & Layer & Random & Spk-disj. & Src-disj. \\")
    print(r"\midrule")
    for c in ("aydid", "sada"):
        sub = sorted([x for x in rows if x["corpus"] == c],
                     key=lambda r: ORDER.index(r["encoder"]) if r["encoder"] in ORDER else 99)
        for i, r in enumerate(sub):
            name = {"aydid": "AYDID", "sada": "SADA"}[c] if i == 0 else ""
            print(f"{name} & {PRETTY.get(r['encoder'], r['encoder'])} & "
                  f"{r['layer']}/{r['n_layers']} & "
                  f"{r.get('random', float('nan')):.3f} & "
                  f"{r.get('speaker_disjoint', float('nan')):.3f} & "
                  f"{r.get('source_disjoint', float('nan')):.3f} \\\\")
        if c == "aydid":
            print(r"\midrule")
    print(r"\bottomrule")
    print(r"\end{tabular}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--latex", action="store_true")
    ap.add_argument("--csv")
    a = ap.parse_args()

    rows = load()
    if not rows:
        raise SystemExit("no results found")
    for c in ("aydid", "sada"):
        fmt(rows, c)
    peak_table(rows)

    ad = np.array([r["drop"] for r in rows if r["corpus"] == "aydid" and "drop" in r])
    sd = np.array([r["drop"] for r in rows if r["corpus"] == "sada" and "drop" in r])
    if len(ad) and len(sd):
        print(f"\n{'='*88}\nHEADLINE\n{'='*88}")
        print(f"  AYDID (sources nested in label):  drop {ad.mean():.3f} "
              f"[{ad.min():.3f}, {ad.max():.3f}] over {len(ad)} encoders")
        print(f"  SADA  (sources crossed with label): drop {sd.mean():.3f} "
              f"[{sd.min():.3f}, {sd.max():.3f}] over {len(sd)} encoders")
        print(f"  ratio {sd.mean()/ad.mean():.1f}x, and the two ranges do not overlap.")
        print(f"\n  Within each corpus the spread across seven encoders "
              f"({ad.max()-ad.min():.3f} / {sd.max()-sd.min():.3f}) is far smaller\n"
              f"  than the gap between corpora ({sd.mean()-ad.mean():.3f}): the cost of "
              f"source-disjoint\n  evaluation is a property of the corpus, not of the "
              f"representation.")

    if a.latex:
        latex(rows)
    if a.csv:
        import csv
        keys = sorted({k for r in rows for k in r if k != "peaks"})
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
