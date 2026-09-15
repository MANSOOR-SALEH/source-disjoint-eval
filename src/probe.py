"""
Probe one (corpus, encoder) feature file under every protocol in folds.json.

    python src/probe.py --features features/aydid.wavlm.npz \
        --manifest manifests/aydid.csv --folds folds/aydid.json \
        --out results/aydid.wavlm.json --fixed-layer 8

Why this reports layer PROFILES rather than selecting a layer
-------------------------------------------------------------
Grouped inner layer selection needs >= 6 sources per class (see
tests/test_pipeline.py). AYDID has classes with 4 and 5 shows, so per-fold
selection is unstable there. Rather than mitigate that, we remove it: every
layer is scored on every outer fold, and three quantities are derived from one
pass.

  fixed   : an a-priori layer, declared before the run. This is the headline
            number. Precedent: Paper 2 fixed layers 4/4/8 from prior probing
            before any experiment and did not select post hoc.
  oracle  : best layer chosen ON THE TEST FOLD. An upper bound, always
            optimistic, reported as a bound and never as a system score.
  selected: inner-split selection, only when folds.json carries inner splits.

This is cheaper than the old inner-selection loop and answers a better question:
does the best layer MOVE when provenance is held out? A peak that shifts between
protocols is channel-borne -- the same signature as the Paper 2 layer audit.

Classifier capacity is fixed at the 256-unit MLP from Paper 4 so encoder
differences cannot be capacity in disguise. The linear probe runs alongside.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def fit_eval(Xtr, ytr, Xte, yte, n_cls, hidden=256, epochs=60, seed=0, lr=1e-3):
    torch.manual_seed(seed)
    sc = StandardScaler().fit(Xtr)
    Xtr_t = torch.tensor(sc.transform(Xtr), dtype=torch.float32, device=DEVICE)
    Xte_t = torch.tensor(sc.transform(Xte), dtype=torch.float32, device=DEVICE)
    ytr_t = torch.tensor(ytr, dtype=torch.long, device=DEVICE)

    if hidden:
        model = nn.Sequential(
            nn.Linear(Xtr.shape[1], hidden), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(hidden, n_cls)).to(DEVICE)
    else:
        model = nn.Linear(Xtr.shape[1], n_cls).to(DEVICE)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-2)
    lossf = nn.CrossEntropyLoss()
    n, bs = len(Xtr_t), 128
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            opt.zero_grad()
            lossf(model(Xtr_t[b]), ytr_t[b]).backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        pred = model(Xte_t).argmax(1).cpu().numpy()
    return float(f1_score(yte, pred, average="macro"))


def run(features, manifest, folds_path, out, hidden=256, seed=0,
        layer_stride=1, fixed_layer=None, linear_too=True, epochs=60,
        use_matched=True, subset_of=None):
    d = np.load(features, allow_pickle=True)
    emb = d["emb"].astype(np.float32)
    df = pd.read_csv(manifest).reset_index(drop=True)
    feat_ids = d["utt_id"].astype(str)

    if subset_of is not None:
        # Features were extracted against a larger manifest; select this
        # manifest's rows out of them by utt_id. Lets a subset experiment reuse
        # the parent corpus's features with no re-extraction.
        pos = {u: i for i, u in enumerate(feat_ids)}
        missing = [u for u in df["utt_id"].astype(str) if u not in pos]
        if missing:
            raise SystemExit(f"{len(missing)} utt_id(s) not in {features}, "
                             f"e.g. {missing[:3]}")
        emb = emb[[pos[u] for u in df["utt_id"].astype(str)]]
        print(f"selected {len(df)} of {len(feat_ids)} rows from the parent features")
    else:
        assert (feat_ids == df["utt_id"].astype(str).to_numpy()).all(), \
            "feature/manifest row order mismatch -- regenerate features from this " \
            "manifest, or pass --subset-of if this is a subset of a larger one"

    labels = sorted(df["label"].unique())
    y = df["label"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    n_cls, chance = len(labels), 1.0 / len(labels)
    spec = json.load(open(folds_path))
    layers = list(range(0, emb.shape[1], layer_stride))
    if fixed_layer is None:
        fixed_layer = layers[len(layers) // 2]
        print(f"!! no --fixed-layer given, defaulting to {fixed_layer}. Declare one "
              f"per encoder BEFORE the run and record it, or the headline number is "
              f"post hoc.")
    if fixed_layer not in layers:
        raise SystemExit(f"--fixed-layer {fixed_layer} not in {layers}")

    matched = spec.get("matched_train_size")
    matched_src = spec.get("matched_train_sources")
    if use_matched and matched:
        print(f"training-set size matched at {matched} across all protocols"
              + (f"; training sources matched at {matched_src}" if matched_src else ""))
    if use_matched and not matched_src:
        print("!! training-source count NOT matched: the leakier protocols train on\n"
              "   more distinct recording sources than the source-disjoint one, so the\n"
              "   measured drop mixes a test-side effect with reduced training\n"
              "   channel diversity. Rebuild folds with --match-train-sources.")
    if use_matched and matched_src:
        anyfold = next(iter(spec["protocols"].values()))["folds"][0]
        if "test_srcmatched" not in anyfold:
            print("!! these folds predate test-partition restriction. Source matching\n"
                  "   dropped training sources but left their clips in test, which\n"
                  "   converts part of each leakier arm into a source-disjoint arm and\n"
                  "   shrinks the measured drop for arithmetical reasons. Rebuild the\n"
                  "   folds before reporting these numbers.")
    elif not use_matched:
        print("!! --unmatched: protocols differ in training-set size as well as "
              "provenance; the source-disjoint drop will be partly sample size")

    n_layers_total = int(emb.shape[1]) - 1     # true depth, not the sampled count
    res = {"features": features, "encoder": str(d["encoder"]), "chance": chance,
           "n_layers_total": n_layers_total, "layer_stride": layer_stride,
           "matched_train_size": (matched if use_matched else None),
           "matched_train_sources": (matched_src if use_matched else None),
           "n_classes": n_cls, "layers": layers, "fixed_layer": int(fixed_layer),
           "protocols": {}}
    fi = layers.index(fixed_layer)

    for pname, p in spec["protocols"].items():
        prof_mlp = np.zeros((len(p["folds"]), len(layers)))
        prof_lin = np.zeros_like(prof_mlp)
        sel, inner_prof = [], []
        for k, f in enumerate(p["folds"]):
            tr_key = ("train_matched" if (use_matched and "train_matched" in f)
                      else ("train_srcmatched"
                            if (use_matched and "train_srcmatched" in f) else "train"))
            # When training sources are subsampled, the test partition must be
            # restricted to the retained sources too; otherwise clips from
            # dropped sources are source-disjoint and the leakier arm is
            # partially converted into the source-disjoint arm.
            te_key = ("test_srcmatched"
                      if (use_matched and "test_srcmatched" in f) else "test")
            tr, te = np.array(f[tr_key]), np.array(f[te_key])
            for li, L in enumerate(layers):
                prof_mlp[k, li] = fit_eval(emb[tr, L], y[tr], emb[te, L], y[te],
                                           n_cls, hidden=hidden, seed=seed, epochs=epochs)
                if linear_too:
                    prof_lin[k, li] = fit_eval(emb[tr, L], y[tr], emb[te, L], y[te],
                                               n_cls, hidden=0, seed=seed, epochs=epochs)
            if f.get("inner"):
                # NESTED SELECTION. The layer is chosen on inner splits carved
                # out of this fold's TRAINING partition and grouped identically
                # to the outer protocol, so no test score influences the choice.
                # Reported alongside the convention layer, which reproduces
                # current practice and is selected on speaker-disjoint scores.
                sc = np.zeros(len(layers))
                for inner in f["inner"]:
                    itr, iva = np.array(inner["train"]), np.array(inner["val"])
                    for li, L in enumerate(layers):
                        sc[li] += fit_eval(emb[itr, L], y[itr], emb[iva, L], y[iva],
                                           n_cls, hidden=hidden, seed=seed, epochs=25)
                inner_prof.append(sc / max(1, len(f["inner"])))
                sel.append(layers[int(sc.argmax())])
            print(f"{pname:20s} fold {k:2d}  fixed(L{fixed_layer}) {prof_mlp[k, fi]:.3f}"
                  f"  oracle(L{layers[int(prof_mlp[k].argmax())]}) {prof_mlp[k].max():.3f}")

        entry = {
            "group_col": p["group_col"],
            "resampled": bool(p.get("resampled", False)),
            "n_folds": len(p["folds"]),
            # full folds x layers matrix, so per-fold scores are available at
            # ANY layer afterwards -- required for exact CIs at the convention
            # layer, which is not known until every protocol has been scored.
            "profile_mlp_per_fold": prof_mlp.tolist(),
            "profile_mlp_mean": prof_mlp.mean(0).tolist(),
            "profile_mlp_sd": (prof_mlp.std(0, ddof=1).tolist() if len(prof_mlp) > 1 else None),
            "fixed": {"layer": int(fixed_layer),
                      "mean": float(prof_mlp[:, fi].mean()),
                      "sd": float(prof_mlp[:, fi].std(ddof=1)) if len(prof_mlp) > 1 else 0.0,
                      "per_fold": prof_mlp[:, fi].tolist()},
            "oracle_upper_bound": {
                "mean": float(prof_mlp.max(1).mean()),
                "per_fold_layer": [layers[i] for i in prof_mlp.argmax(1)],
                "note": "layer chosen on the test fold -- an upper bound, not a system score"},
            "peak_layer_of_mean_profile": int(layers[int(prof_mlp.mean(0).argmax())]),
        }
        if linear_too:
            entry["profile_linear_mean"] = prof_lin.mean(0).tolist()
            entry["fixed_linear_mean"] = float(prof_lin[:, fi].mean())
        if sel:
            entry["inner_selected_layers"] = sel
            # score each fold at ITS OWN nested-selected layer: a clean
            # train-selected estimate with no test-set influence
            nested = np.array([prof_mlp[k, layers.index(sel[k])]
                               for k in range(len(sel))])
            entry["nested"] = {
                "per_fold_layer": sel,
                "per_fold": nested.tolist(),
                "mean": float(nested.mean()),
                "sd": float(nested.std(ddof=1)) if len(nested) > 1 else 0.0,
            }
            entry["inner_profile_mean"] = np.mean(inner_prof, axis=0).tolist()
        res["protocols"][pname] = entry
        print(f"{pname:20s} peak layer of mean profile = L{entry['peak_layer_of_mean_profile']}"
              f"  fixed {entry['fixed']['mean']:.3f}"
              f"  oracle {entry['oracle_upper_bound']['mean']:.3f}\n")

    # --- the convention layer -------------------------------------------------
    # Standard practice sweeps layers and reports the best under speaker-disjoint
    # evaluation. We reproduce that, then score THAT SAME layer under every
    # protocol. This needs no external justification and no prior paper: it
    # measures the inflation a practitioner following current practice would get.
    conv_src = "speaker_disjoint" if "speaker_disjoint" in res["protocols"] else \
        next(iter(res["protocols"]))
    cl = res["protocols"][conv_src]["peak_layer_of_mean_profile"]
    ci = layers.index(cl)
    res["convention_layer"] = {"layer": int(cl), "chosen_on": conv_src,
                               "n_layers_total": n_layers_total,
                               "relative_depth": round(cl / n_layers_total, 3)}
    for pname, e in res["protocols"].items():
        prof = np.array(e["profile_mlp_mean"])
        e["at_convention_layer"] = float(prof[ci])
    print(f"\nconvention layer (best under {conv_src}) = L{cl} "
          f"(depth {cl}/{len(layers)-1})")
    for pname, e in res["protocols"].items():
        print(f"  {pname:20s} {e['at_convention_layer']:.3f}")
    if "speaker_disjoint" in res["protocols"] and "source_disjoint" in res["protocols"]:
        drop = (res["protocols"]["speaker_disjoint"]["at_convention_layer"]
                - res["protocols"]["source_disjoint"]["at_convention_layer"])
        res["convention_layer"]["spk_to_src_drop"] = float(drop)
        print(f"  -> holding out sources costs {drop:+.3f} macro-F1 at the layer "
              f"current practice would select")

        # nested-selection estimate: no test score entered the layer choice
        P = res["protocols"]
        if all("nested" in P[k] for k in ("speaker_disjoint", "source_disjoint")):
            ns = P["speaker_disjoint"]["nested"]["mean"]
            nr = P["source_disjoint"]["nested"]["mean"]
            res["nested_drop"] = {"speaker_disjoint": ns, "source_disjoint": nr,
                                  "drop": float(ns - nr)}
            print(f"\n  NESTED SELECTION (layer chosen on inner training splits only)")
            for k in ("random", "speaker_disjoint", "source_disjoint",
                      "source_disjoint_rep"):
                if k in P and "nested" in P[k]:
                    print(f"    {k:20s} {P[k]['nested']['mean']:.3f}  "
                          f"layers {sorted(set(P[k]['nested']['per_fold_layer']))}")
            print(f"    -> nested drop {ns - nr:+.3f}  "
                  f"(convention-layer drop {drop:+.3f}, "
                  f"difference {abs((ns - nr) - drop):.3f})")
            print(f"    The convention layer is selected on speaker-disjoint scores,")
            print(f"    so its drop is upward-biased; the nested figure is not.")

    peaks = {k: v["peak_layer_of_mean_profile"] for k, v in res["protocols"].items()}
    res["peak_layer_by_protocol"] = peaks
    if len(set(peaks.values())) > 1:
        print(f"PEAK LAYER MOVES ACROSS PROTOCOLS: {peaks}")
        print("  -> the layer that looks best under the leakier protocol is partly "
              "channel-borne. That belongs in the paper, not in a footnote.")
    else:
        print(f"peak layer stable across protocols at L{list(peaks.values())[0]}")

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(res, open(out, "w"), indent=1)
    print(f"wrote {out}")
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fixed-layer", type=int, default=None,
                    help="declare this per encoder BEFORE running")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layer-stride", type=int, default=1)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--no-linear", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--unmatched", action="store_true",
                    help="ignore train_matched and use the full training partition")
    ap.add_argument("--subset-of",
                    help="the manifest the features were extracted against, when "
                         "--manifest is a subset of it; rows are selected by utt_id")
    a = ap.parse_args()
    run(a.features, a.manifest, a.folds, a.out, hidden=a.hidden, seed=a.seed,
        layer_stride=a.layer_stride, fixed_layer=a.fixed_layer,
        linear_too=not a.no_linear, epochs=a.epochs, use_matched=not a.unmatched,
        subset_of=a.subset_of)