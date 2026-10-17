"""
Fold construction for the three protocols.

    python src/folds.py --manifest manifests/aydid.csv --out folds/aydid.json \
        --n-folds 10 --n-inner 3 --seed 0

The single most important property here: the INNER split used for layer and
hyper-parameter selection is grouped by the SAME variable as the outer split.
Selecting the best layer on a speaker-disjoint inner split and then reporting a
source-disjoint outer score leaks provenance through the model-selection step,
and a reviewer will find it. Every protocol therefore gets its own inner
grouping, and folds.json records which variable was used so probe.py cannot
silently mismatch.

Protocols
---------
random          : stratified by label, no grouping. The leaky reference.
speaker_disjoint: grouped by speaker_id.
source_disjoint : grouped by source_id, groups drawn separately WITHIN each
                  class so every class is present in both train and test.
                  Classes with <= 3 sources fall back to leave-one-group-out and
                  the true fold count is recorded.
"""

import argparse
import json
from collections import defaultdict

import numpy as np
import pandas as pd


def _stratified_random(df, n_folds, rng):
    folds = [[] for _ in range(n_folds)]
    for _, sub in df.groupby("label"):
        idx = np.array(sub.index.to_numpy(), copy=True)
        rng.shuffle(idx)
        for k, chunk in enumerate(np.array_split(idx, n_folds)):
            folds[k].extend(chunk.tolist())
    return folds


def _groups_nested_in_label(df, group_col):
    """True when every group maps to exactly one label. Determines which
    holdout scheme is valid, and is itself a property worth reporting: a corpus
    whose recording sources are nested in the label cannot have that correlation
    removed by any split, whereas one whose sources are crossed with the label
    can."""
    return int(df.groupby(group_col)["label"].nunique().max()) == 1


def _grouped_global(df, group_col, n_folds, rng, max_tries=50):
    """Hold out whole groups across ALL classes at once. Required when groups
    span classes: a show cannot be in train for one dialect and test for
    another without its acoustic signature crossing the split."""
    groups = df[group_col].dropna().unique().tolist()
    if len(groups) < n_folds:
        n_folds = len(groups)

    # Leave-one-group-out. When the requested fold count reaches the number of
    # groups, each fold is a single group, and a random partition adds nothing
    # -- worse, array_split's all-or-nothing validity check rejects the whole
    # partition if any one group cannot be held out, discarding the many that
    # can. Build the folds directly and keep every group that is individually
    # valid. This is the protocol that works when several classes are carried by
    # only a handful of sources, since holding out one source at a time never
    # strips a thin class entirely.
    if n_folds >= len(groups):
        folds, skipped = [], []
        for g in groups:
            test = df.index[df[group_col] == g].tolist()
            train = sorted(set(range(len(df))) - set(test))
            if (df.loc[test, "label"].nunique() < 1 or
                    df.loc[train, "label"].nunique() < df["label"].nunique()):
                skipped.append(g)
                continue
            folds.append([int(i) for i in test])
        if not folds:
            raise SystemExit(
                f"no single '{group_col}' can be held out while keeping every "
                f"class in training")
        if skipped:
            print(f"  leave-one-out: skipped {len(skipped)} group(s) whose "
                  f"holdout would empty a class from training")
        return folds, len(folds), None

    for _ in range(max_tries):
        g = list(groups)
        rng.shuffle(g)
        parts = np.array_split(np.array(g, dtype=object), n_folds)
        folds, ok = [], True
        for part in parts:
            test = df.index[df[group_col].isin(set(part))].tolist()
            train = sorted(set(range(len(df))) - set(test))
            if (df.loc[test, "label"].nunique() < df["label"].nunique() or
                    df.loc[train, "label"].nunique() < df["label"].nunique()):
                ok = False
                break
            folds.append([int(i) for i in test])
        if ok:
            return folds, n_folds, None
    raise SystemExit(
        f"could not build {n_folds} global '{group_col}' folds keeping every class "
        f"on both sides. Reduce --n-folds, or pool a finer provenance unit.")


def _grouped_within_class(df, group_col, n_folds, rng):
    """Partition each class's groups across folds, so all classes appear in
    every test fold. Returns (folds, effective_n_folds, per_class_group_counts)."""
    per_class = {}
    for label, sub in df.groupby("label"):
        groups = sub[group_col].dropna().unique().tolist()
        rng.shuffle(groups)
        per_class[label] = groups

    counts = {str(k): len(v) for k, v in per_class.items()}
    min_groups = min(len(v) for v in per_class.values())
    if min_groups < 2:
        raise SystemExit(
            f"a class has {min_groups} distinct '{group_col}' value(s): no grouped "
            f"split exists. Either merge that class out of the protocol or collect "
            f"more provenance metadata."
        )
    eff = int(min(n_folds, min_groups))   # leave-one-group-out when groups are scarce

    assign = {}
    for label, groups in per_class.items():
        for k, chunk in enumerate(np.array_split(np.array(groups, dtype=object), eff)):
            for g in chunk:
                assign[(label, g)] = k

    folds = [[] for _ in range(eff)]
    for i, row in df.iterrows():
        k = assign.get((row["label"], row[group_col]))
        if k is not None:
            folds[k].append(int(i))
    return folds, eff, counts


def _inner_splits(df, train_idx, group_col, n_inner, rng):
    """Inner model-selection splits, grouped identically to the outer protocol."""
    sub = df.loc[train_idx]
    out = []
    if group_col is not None and not _groups_nested_in_label(df, group_col):
        f, eff, _ = _grouped_global(sub.reset_index(drop=True), group_col, n_inner, rng)
        pos = sub.index.to_numpy()
        for k in range(eff):
            val = [int(pos[i]) for i in f[k]]
            tr = [int(pos[i]) for j in range(eff) if j != k for i in f[j]]
            out.append({"train": tr, "val": val})
        return out
    if group_col is None:
        f = _stratified_random(sub, n_inner, rng)
        for k in range(n_inner):
            val = f[k]
            tr = [i for j in range(n_inner) if j != k for i in f[j]]
            out.append({"train": tr, "val": val})
        return out
    f, eff, _ = _grouped_within_class(sub, group_col, n_inner, rng)
    for k in range(eff):
        val = f[k]
        tr = [i for j in range(eff) if j != k for i in f[j]]
        out.append({"train": tr, "val": val})
    return out


PROTOCOLS = {
    "random": None,
    "speaker_disjoint": "speaker_id",
    "source_disjoint": "source_id",
}


def _repeated_grouped(df, group_col, n_repeats, rng, holdout=1):
    """Repeated grouped resampling: draw `holdout` group(s) per class at random,
    n_repeats times. Use when the number of groups caps the fold count so low
    that 4-6 folds cannot estimate the variance (Paper 4 saw +-0.137 SD on
    source-disjoint folds). Resamples are not disjoint from each other, so treat
    them as repeated estimates, not as cross-validation folds -- report the SD
    across resamples and say which they are."""
    per_class = {lab: sub[group_col].dropna().unique().tolist()
                 for lab, sub in df.groupby("label")}
    reps = []
    for _ in range(n_repeats):
        held = set()
        for lab, groups in per_class.items():
            k = min(holdout, max(1, len(groups) - 1))
            held.update((lab, g) for g in rng.choice(groups, size=k, replace=False))
        test = [int(i) for i, r in df.iterrows() if (r["label"], r[group_col]) in held]
        train = sorted(set(range(len(df))) - set(test))
        reps.append({"test": sorted(test), "train": train, "inner": []})
    return reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--n-inner", type=int, default=3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--match-train-size", action="store_true", default=True,
                    help="add a size-matched training subset to every fold so "
                         "protocols differ only in provenance, not in N")
    ap.add_argument("--no-match-train-size", dest="match_train_size",
                    action="store_false")
    ap.add_argument("--match-train-sources", action="store_true", default=True,
                    help="also subsample training SOURCES under the leakier "
                         "protocols to the source count of the source-disjoint "
                         "condition, so protocols differ only in provenance")
    ap.add_argument("--no-match-train-sources", dest="match_train_sources",
                    action="store_false")
    ap.add_argument("--retain", type=float, default=None,
                    help="fraction of the matched source count to retain, for a "
                         "dose-response sweep (e.g. 1.0 0.75 0.5 0.25)")
    ap.add_argument("--n-repeats", type=int, default=20,
                    help="repeated source-disjoint resamples, in addition to the "
                         "fixed folds; 0 to disable")
    args = ap.parse_args()

    df = pd.read_csv(args.manifest).reset_index(drop=True)
    rng = np.random.default_rng(args.seed)
    spec = {"manifest": args.manifest, "seed": args.seed, "protocols": {}}

    for name, gcol in PROTOCOLS.items():
        if gcol is not None and df[gcol].isna().any():
            print(f"skip {name}: '{gcol}' has missing values")
            continue
        if gcol is None:
            outer = _stratified_random(df, args.n_folds, rng)
            eff, counts, nested = args.n_folds, None, None
        else:
            nested = _groups_nested_in_label(df, gcol)
            if nested:
                outer, eff, counts = _grouped_within_class(df, gcol, args.n_folds, rng)
            else:
                outer, eff, counts = _grouped_global(df, gcol, args.n_folds, rng)
                print(f"  '{gcol}' groups span classes -> whole-group holdout across "
                      f"all classes (within-class partitioning would leak)")

        entry = {"group_col": gcol, "n_folds": eff, "groups_per_class": counts,
                 "nested_in_label": nested, "folds": []}
        all_idx = set(range(len(df)))
        for k in range(eff):
            test = sorted(outer[k])
            train = sorted(all_idx - set(test))
            entry["folds"].append({
                "test": test,
                "train": train,
                "inner": _inner_splits(df, train, gcol, args.n_inner, rng),
            })
        spec["protocols"][name] = entry

        # sanity: no group crosses the train/test boundary
        if gcol is not None:
            for k, f in enumerate(entry["folds"]):
                a = set(df.loc[f["train"], gcol])
                b = set(df.loc[f["test"], gcol])
                assert not (a & b), f"{name} fold {k} leaks {gcol}: {sorted(a & b)[:5]}"
        nsrc = [int(df.loc[f["train"], "source_id"].nunique()) for f in entry["folds"]] \
            if "source_id" in df.columns else []
        print(f"{name:17s} folds={eff:2d}  test sizes={[len(f['test']) for f in entry['folds']]}"
              + (f"  train sources {min(nsrc)}-{max(nsrc)}" if nsrc else ""))

    if (args.n_repeats and "source_id" in df.columns
            and not df["source_id"].isna().any()
            and _groups_nested_in_label(df, "source_id")):
        reps = _repeated_grouped(df, "source_id", args.n_repeats, rng)
        spec["protocols"]["source_disjoint_rep"] = {
            "group_col": "source_id", "n_folds": len(reps),
            "groups_per_class": None, "resampled": True, "folds": reps,
        }
        for k, f in enumerate(reps):
            a = set(df.loc[f["train"], "source_id"]); b = set(df.loc[f["test"], "source_id"])
            assert not (a & b), f"repeat {k} leaks source_id"
        print(f"{'source_disjoint_rep':17s} repeats={len(reps)}  "
              f"test sizes={[len(f['test']) for f in reps[:5]]}...")

    # ---- training-size matching -----------------------------------------
    # Source-disjoint folds hold out whole sources, so they hold out far more
    # utterances than a 10% random fold. Comparing protocols without matching
    # confounds provenance with training-set size -- the control Paper 4 applies
    # on ADI-17. Every fold gets a label-stratified training subset of the size
    # of the SMALLEST training partition anywhere in this file.
    # ---- training-source diversity matching -----------------------------
    # Size matching equalises N but leaves the NUMBER OF DISTINCT TRAINING
    # SOURCES free, and that number falls by construction under source-disjoint
    # evaluation. Fewer training channels means worse generalisation to any
    # unseen channel, independent of whether the test set is provenance-
    # disjoint, so an unmatched comparison conflates a test-side effect with a
    # train-side one. We therefore also subsample SOURCES under the leakier
    # protocols to the source count of the source-disjoint condition.
    #
    # CRITICAL: the test partition must be restricted at the same time. If a
    # source is dropped from training but its clips stay in test, those clips
    # become source-disjoint, and the leakier arm is silently converted into a
    # partial source-disjoint arm. Removing a fraction q of training sources
    # converts roughly q of the test items, so the measured drop is scaled by
    # about (1-q) for purely arithmetical reasons. On AYDID q = 13/40 = 0.325
    # and on SADA q = 7/63 = 0.111 -- large enough to account for the whole of
    # the movement the control is meant to attribute to channel diversity.
    #
    # We therefore write BOTH 'train_srcmatched' and 'test_srcmatched'. In the
    # leakier arms the test set keeps only clips whose source is retained in
    # training, so the arm remains what it claims to be. The source-disjoint arm
    # is untouched: none of its test sources appear in training by construction.
    # The arms then evaluate on different test subsets, which must be stated.
    if args.match_train_sources and "source_disjoint" in spec["protocols"]:
        tgt_src = min(
            df.loc[f["train"], "source_id"].nunique()
            for f in spec["protocols"]["source_disjoint"]["folds"])
        # --retain sweeps the control's strength for a dose-response curve:
        # retain=1.0 reproduces the standard matched setting, lower values
        # keep proportionally fewer training sources in EVERY arm, including
        # source-disjoint, so all arms stay comparable at each level.
        if args.retain is not None:
            tgt_src = max(2, int(round(tgt_src * args.retain)))
            print(f"  --retain {args.retain}: target lowered to {tgt_src} sources")
        srng = np.random.default_rng(args.seed + 4242)
        print(f"\nsource-count matching: target {tgt_src} training sources "
              f"(from the source-disjoint condition)")
        for pname, p in spec["protocols"].items():
            kept, dropped_te = [], []
            src_disjoint_arm = (pname == "source_disjoint"
                                or bool(p.get("resampled")))
            for f in p["folds"]:
                tr = np.array(f["train"])
                srcs = df.loc[tr, "source_id"].unique()
                if len(srcs) <= tgt_src:
                    f["train_srcmatched"] = sorted(int(i) for i in tr)
                    f["test_srcmatched"] = sorted(int(i) for i in f["test"])
                    kept.append(len(srcs))
                    continue
                # Draw sources at random, but never drop a class: sources are
                # added until every label in the training set is covered, and
                # the count may exceed the target only when coverage demands it.
                # Sources are drawn at random, but the retained set must keep
                # every class present on BOTH sides: in training (or the model
                # never sees the class) and in the restricted test partition (or
                # macro-F1 averages over an absent class). We therefore seed the
                # draw with the classes' requirements from both partitions
                # before filling to the target count.
                sub = df.loc[tr, ["source_id", "label"]]
                by_src = {sid: set(g["label"]) for sid, g in sub.groupby("source_id")}
                te_all = np.array(f["test"])
                te_sub = df.loc[te_all, ["source_id", "label"]]
                te_by_src = {sid: set(g["label"])
                             for sid, g in te_sub.groupby("source_id")}
                need_tr = set(sub["label"].unique())
                need_te = (set(te_sub["label"].unique())
                           if not src_disjoint_arm else set())
                chosen = []
                for sid in srng.permutation(list(by_src)):
                    done = (len(chosen) >= tgt_src and not need_tr and not need_te)
                    if done:
                        break
                    chosen.append(sid)
                    need_tr -= by_src[sid]
                    need_te -= te_by_src.get(sid, set())
                if need_tr or need_te:
                    raise SystemExit(
                        f"{pname}: cannot retain {tgt_src} sources while keeping "
                        f"every class in both the training and restricted test "
                        f"partitions. Reduce --n-folds, or run with "
                        f"--no-match-train-sources and report the uncontrolled "
                        f"comparison.")
                chosen = set(chosen)
                sel = tr[sub["source_id"].isin(chosen).to_numpy()]
                f["train_srcmatched"] = sorted(int(i) for i in sel)
                kept.append(len(chosen))

                te = np.array(f["test"])
                if src_disjoint_arm:
                    f["test_srcmatched"] = sorted(int(i) for i in te)
                else:
                    keep_te = te[df.loc[te, "source_id"].isin(chosen).to_numpy()]
                    if df.loc[keep_te, "label"].nunique() < df["label"].nunique():
                        raise SystemExit(
                            f"{pname} fold: restricting test to retained sources "
                            f"empties a class. Raise the target source count or "
                            f"reduce --n-folds.")
                    f["test_srcmatched"] = sorted(int(i) for i in keep_te)
                    dropped_te.append(1.0 - len(keep_te) / max(1, len(te)))
            msg = (f"  {pname:20s} training sources {min(kept)}-{max(kept)}  "
                   f"(unmatched: "
                   f"{min(df.loc[f['train'], 'source_id'].nunique() for f in p['folds'])}-"
                   f"{max(df.loc[f['train'], 'source_id'].nunique() for f in p['folds'])})")
            msg += ("; test unchanged" if src_disjoint_arm else
                    f"; test restricted, {100*np.mean(dropped_te):.1f}% of clips removed")
            print(msg)
        spec["matched_train_sources"] = int(tgt_src)

    if args.match_train_size:
        key = "train_srcmatched" if spec.get("matched_train_sources") else "train"
        sizes = [len(f.get(key, f["train"]))
                 for p in spec["protocols"].values() for f in p["folds"]]
        target = min(sizes)
        spec["matched_train_size"] = int(target)
        mrng = np.random.default_rng(args.seed + 991)
        for p in spec["protocols"].values():
            for f in p["folds"]:
                tr = np.array(f.get(key, f["train"]))
                if len(tr) == target:
                    f["train_matched"] = sorted(int(i) for i in tr)
                    continue
                lab = df.loc[tr, "label"].to_numpy()
                keep = []
                for L in np.unique(lab):
                    pool = tr[lab == L]
                    k = int(round(target * len(pool) / len(tr)))
                    keep.extend(mrng.choice(pool, size=min(k, len(pool)), replace=False))
                f["train_matched"] = sorted(int(i) for i in keep)
        print(f"\nmatched train size = {target} "
              f"(range before matching {min(sizes)}-{max(sizes)}); "
              f"probe.py uses 'train_matched' when present")

    with open(args.out, "w") as fh:
        json.dump(spec, fh)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()