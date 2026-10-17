"""
Project status: what exists, what is missing, what to run next.

    python src/status.py

Reads the manifests, folds and feature files on disk and reports the state of
each (corpus x encoder) cell. Read-only -- it never writes or deletes.
"""

import glob
import json
import os
import sys

import numpy as np

ENCODERS = ["wavlm", "xlsr", "mhubert", "mms", "whisper", "w2vbert", "artst"]
CORPORA = {"aydid": "manifests/aydid_clean.csv", "sada": "manifests/sada.csv"}
GREEN, RED, DIM, OFF = "\033[32m", "\033[31m", "\033[90m", "\033[0m"


def gb(p):
    return os.path.getsize(p) / 1e9


def main():
    if not os.path.isdir("src"):
        sys.exit("run this from the project root (the folder containing src/)")

    print("=" * 72)
    print("MANIFESTS")
    print("=" * 72)
    import pandas as pd
    for c, m in CORPORA.items():
        if os.path.exists(m):
            d = pd.read_csv(m)
            srcs = d.groupby("label")["source_id"].nunique()
            nested = int(d.groupby("source_id")["label"].nunique().max()) == 1
            print(f"  {GREEN}OK{OFF}   {c:6s} {len(d):>6d} utts  {d['label'].nunique()} classes  "
                  f"{d['source_id'].nunique()} sources ({srcs.min()}-{srcs.max()}/class)  "
                  f"{'nested' if nested else 'crossed'}")
        else:
            print(f"  {RED}--{OFF}   {c:6s} missing: {m}")

    print()
    print("=" * 72)
    print("FOLDS")
    print("=" * 72)
    for c in CORPORA:
        p = f"folds/{c}.json"
        if not os.path.exists(p):
            print(f"  {RED}--{OFF}   {c:6s} missing: {p}")
            continue
        s = json.load(open(p))
        parts = "  ".join(f"{k}={v['n_folds']}" for k, v in s["protocols"].items())
        mt = s.get("matched_train_size")
        print(f"  {GREEN}OK{OFF}   {c:6s} {parts}"
              + (f"   matched_train={mt}" if mt else f"   {RED}no train matching{OFF}"))

    print()
    print("=" * 72)
    print("FEATURES")
    print("=" * 72)
    todo = []
    for c, m in CORPORA.items():
        n_expect = None
        if os.path.exists(m):
            n_expect = len(pd.read_csv(m))
        print(f"  {c}:")
        for e in ENCODERS:
            p = f"features/{c}.{e}.npz"
            if not os.path.exists(p):
                print(f"    {RED}--{OFF}  {e:8s} not extracted")
                todo.append((c, e))
                continue
            try:
                d = np.load(p, allow_pickle=True)
                shape = d["emb"].shape
                bad = n_expect is not None and shape[0] != n_expect
                flag = f"{RED}ROW COUNT != manifest ({n_expect}){OFF}" if bad else ""
                print(f"    {GREEN if not bad else RED}OK{OFF}  {e:8s} "
                      f"{shape[0]:>6d} x {shape[1]:>2d} layers x {shape[2]:>4d}  "
                      f"{gb(p):.2f} GB  {flag}")
            except Exception as ex:
                print(f"    {RED}!!{OFF}  {e:8s} unreadable: {ex}")
                todo.append((c, e))

    print()
    print("=" * 72)
    print("RESULTS")
    print("=" * 72)
    res = sorted(glob.glob("results/*.json"))
    if not res:
        print(f"  {DIM}none yet{OFF}")
    for p in res:
        try:
            r = json.load(open(p))
            cl = r.get("convention_layer")
            if not cl:
                print(f"  {DIM}{os.path.basename(p)} (no convention layer){OFF}")
                continue
            spk = r["protocols"].get("speaker_disjoint", {}).get("at_convention_layer")
            src = r["protocols"].get("source_disjoint", {}).get("at_convention_layer")
            line = (f"  {os.path.basename(p):28s} L{cl['layer']:<3d}"
                    f"(d={cl['relative_depth']:.2f})")
            if spk is not None:
                line += f"  spk {spk:.3f}"
            if src is not None:
                line += f"  src {src:.3f}  drop {spk - src:+.3f}"
            print(line)
        except Exception as ex:
            print(f"  {RED}!!{OFF} {p}: {ex}")

    print()
    print("=" * 72)
    print("NEXT")
    print("=" * 72)
    if todo:
        by_corpus = {}
        for c, e in todo:
            by_corpus.setdefault(c, []).append(e)
        for c, es in by_corpus.items():
            print(f"  extract {c}: {' '.join(es)}")
    elif not res:
        print("  all features present -> probe one encoder and time it:")
        print("  python src/probe.py --features features/aydid.wavlm.npz \\")
        print("      --manifest manifests/aydid_clean.csv --folds folds/aydid.json \\")
        print("      --out results/aydid.wavlm.json")
    else:
        done = {os.path.basename(p).replace(".json", "") for p in res}
        want = {f"{c}.{e}" for c in CORPORA for e in ENCODERS
                if os.path.exists(f"features/{c}.{e}.npz")}
        left = sorted(want - done)
        print("  probe: " + (" ".join(left) if left else "all done -> write the paper"))

    tot = sum(gb(p) for p in glob.glob("features/*.npz"))
    print(f"\n  features on disk: {tot:.1f} GB")


if __name__ == "__main__":
    main()
