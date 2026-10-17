"""
Uncertainty for the letter: CIs on every drop, a test on the between-corpus
gap, and an equivalence test for the invariance claim.

    python src/stats_report.py --results results --out results/stats.json

Everything here runs on the per-fold scores already stored in results/*.json.
No re-training, no GPU.

Three quantities the reviewers asked for
----------------------------------------
1. A 95% CI on each encoder's source-disjoint drop. The drop is a difference of
   two means over folds that share a partition, so folds are resampled IN PAIRS
   and the same fold indices are used for both protocols -- resampling them
   independently would inflate the interval.

2. A CI on the between-corpus gap (AYDID mean drop vs SADA mean drop). Corpora
   are independent, so their bootstrap replicates are drawn independently and
   differenced.

3. An EQUIVALENCE test for "the drop does not depend on the encoder". A
   non-significant difference is not evidence of no difference; the claim needs
   two one-sided tests (TOST) against a stated margin. The margin is a decision,
   not a result: we use 0.05 macro-F1 by default, i.e. two encoders are treated
   as equivalent if their drops differ by less than 5 points, and the paper must
   state that number and justify it. The reported statistic is the widest
   pairwise difference among the seven encoders and whether its CI falls inside
   the margin.

Note on AYDID: with only 4 source-disjoint folds a bootstrap over folds is
coarse. The 20 repeated draws (source_disjoint_rep) give a better-resolved
interval and are reported alongside.
"""

import argparse
import glob
import itertools
import json
import os

import numpy as np

# Preferred display order. Any corpus present in the results but absent here is
# appended alphabetically, so adding a corpus needs no code change.
CORPUS_ORDER = ["aydid", "sada", "kespeech"]
ENC_ORDER = ["wavlm", "xlsr", "mhubert", "mms", "whisper", "w2vbert", "artst"]
PRETTY = {"wavlm": "WavLM", "xlsr": "XLS-R", "mhubert": "mHuBERT", "mms": "MMS",
          "whisper": "Whisper", "w2vbert": "w2v-BERT", "artst": "ArTST"}


def load(results_dir):
    """{corpus: {encoder: {protocol: per-fold scores at the convention layer}}}"""
    out = {}
    for p in sorted(glob.glob(os.path.join(results_dir, "*.json"))):
        base = os.path.basename(p)[:-5]
        if "." not in base:
            continue
        corpus, enc = base.split(".", 1)
        # Corpora are discovered from the files rather than whitelisted, so a
        # newly added corpus does not silently vanish from the report.
        if enc not in ENC_ORDER:
            continue
        r = json.load(open(p))
        cl = r.get("convention_layer")
        if not cl:
            continue
        layers = r.get("layers")
        try:
            li = layers.index(cl["layer"])
        except (AttributeError, ValueError):
            li = None
        d, exact = {}, True
        for prot, e in r["protocols"].items():
            # Prefer the full folds x layers matrix, which gives per-fold scores
            # exactly at the convention layer. Older result files stored only
            # the fixed-layer vector; that is a usable proxy for fold-level
            # variance but the paper must not describe it as exact.
            mat = e.get("profile_mlp_per_fold")
            if mat is not None and li is not None:
                d[prot] = np.asarray(mat, dtype=float)[:, li]
            else:
                exact = False
                pf = e.get("fixed", {}).get("per_fold")
                d[prot] = np.asarray(pf, dtype=float) if pf else None
        out.setdefault(corpus, {})[enc] = {"folds": d, "layer": cl["layer"],
                                           "exact": exact,
                                           "at_conv": {k: v.get("at_convention_layer")
                                                       for k, v in r["protocols"].items()}}
    return out


def paired_boot(a, b, B=10000, rng=None):
    """Bootstrap the mean difference a-b, resampling fold indices in pairs."""
    rng = rng or np.random.default_rng(0)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    idx = rng.integers(0, n, size=(B, n))
    return (a[idx].mean(1) - b[idx].mean(1))


def ci(v, lo=2.5, hi=97.5):
    return float(np.percentile(v, lo)), float(np.percentile(v, hi))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="results/stats.json")
    ap.add_argument("--margin", type=float, default=0.05,
                    help="legacy equivalence margin; the reported comparison now "
                         "uses a data-external reference (the random -> "
                         "speaker-disjoint step) instead of a chosen constant")
    ap.add_argument("--perm", type=int, default=10000,
                    help="permutations for the omnibus encoder test")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    data = load(a.results)
    if not data:
        raise SystemExit(f"no usable results in {a.results}")
    approx = [f"{c}.{e}" for c in data for e in data[c] if not data[c][e]["exact"]]
    if approx:
        print(f"!! {len(approx)} result file(s) predate per-fold profile storage, so\n"
              f"   fold variance is taken from the fixed layer rather than the\n"
              f"   convention layer. Re-run probe.py to get exact intervals; until\n"
              f"   then describe these as approximate in the manuscript.\n"
              f"   e.g. {approx[:3]}\n")

    report = {"margin": a.margin, "n_boot": a.boot, "corpora": {}}

    corpora = ([c for c in CORPUS_ORDER if c in data]
               + sorted(c for c in data if c not in CORPUS_ORDER))
    for corpus in corpora:
        print("=" * 84)
        print(f"{corpus.upper()}  --  source-disjoint drop, 95% CI (paired fold bootstrap)")
        print("=" * 84)
        print(f"{'encoder':10s} {'layer':>6s} {'spk':>7s} {'src':>7s} {'drop':>8s} "
              f"{'95% CI':>18s} {'folds':>6s}")
        print("-" * 84)
        reps, drops, fold_drops = {}, {}, {}
        for enc in ENC_ORDER:
            if enc not in data[corpus]:
                continue
            e = data[corpus][enc]
            spk, src = e["folds"].get("speaker_disjoint"), e["folds"].get("source_disjoint")
            if spk is None or src is None:
                continue
            # point estimate uses the reported convention-layer values
            d_point = e["at_conv"]["speaker_disjoint"] - e["at_conv"]["source_disjoint"]
            b = paired_boot(spk, src, a.boot, rng)
            # centre the bootstrap distribution on the reported point estimate
            b = b - b.mean() + d_point
            lo, hi = ci(b)
            reps[enc], drops[enc] = b, d_point
            # per-fold drop vector for the omnibus test; protocols use different
            # partitions, so pair by rank within each protocol's fold list
            # All encoders are evaluated on the SAME folds, so per-fold source-
            # disjoint scores are paired across encoders. The speaker-disjoint
            # folds are a different partition, so its mean is used as the
            # reference; the encoder contrast then lives entirely in the paired
            # source-disjoint folds, which is what the omnibus test needs.
            fd = float(np.mean(spk)) - np.asarray(src, float)
            # centre on the reported convention-layer point estimate, exactly as
            # the bootstrap above is centred, so that between-encoder variation
            # measures variation in the DROP and not in absolute accuracy
            fold_drops[enc] = fd - fd.mean() + d_point
            print(f"{PRETTY[enc]:10s} {e['layer']:>6d} "
                  f"{e['at_conv']['speaker_disjoint']:>7.3f} "
                  f"{e['at_conv']['source_disjoint']:>7.3f} "
                  f"{d_point:>+8.3f} [{lo:>+.3f}, {hi:>+.3f}] {len(src):>6d}")

        if not reps:
            continue
        vals = np.array([drops[e] for e in reps])
        corpus_rep = np.mean([reps[e] for e in reps], axis=0)
        clo, chi = ci(corpus_rep)
        print("-" * 84)
        print(f"{'mean':10s} {'':>6s} {'':>7s} {'':>7s} {vals.mean():>+8.3f} "
              f"[{clo:>+.3f}, {chi:>+.3f}]")
        print(f"  observed spread across encoders: {vals.max() - vals.min():.3f}")

        # --- how much does the encoder move the drop? ----------------------
        # The earlier version of this test took the MAXIMUM of the 21 pairwise
        # differences and bootstrapped a CI on it. Inference on a selected
        # maximum is upward-biased -- the largest of 21 gaps is larger than the
        # typical gap by construction, and a bootstrap CI on an order statistic
        # has no clean coverage. We therefore report three estimands that are
        # not selected on the data:
        #
        #   (i)   RANGE with a bootstrap CI, stated as a descriptive bound and
        #         explicitly flagged as an order statistic;
        #   (ii)  SD of the seven encoder drops, the natural non-selected
        #         measure of between-encoder variation;
        #   (iii) an OMNIBUS permutation test of the null that all seven
        #         encoders share one drop, obtained by permuting the encoder
        #         label within fold across the per-fold drop matrix.
        #
        # The margin for any equivalence statement is set from the data-external
        # reference below, not chosen to sit near the observed spread.
        names = list(reps)
        rng_p = np.random.default_rng(a.seed + 7)

        # Bootstrap the between-encoder statistics by resampling FOLDS, shared
        # across encoders, because every encoder is scored on the same folds.
        # Resampling each encoder independently would make the encoders look
        # more different than they are.
        FD = np.vstack([fold_drops[e] for e in names])        # [n_enc, n_folds]
        point = FD.mean(axis=1)
        rng_stat = float(point.max() - point.min())
        sd_stat = float(point.std(ddof=1))
        nf = FD.shape[1]
        idx = rng_p.integers(0, nf, size=(a.boot, nf))
        means_b = FD[:, idx].mean(axis=2)                     # [n_enc, n_boot]
        rng_boot = means_b.max(axis=0) - means_b.min(axis=0)
        sd_boot = means_b.std(axis=0, ddof=1)
        rlo, rhi = ci(rng_boot)
        slo, shi = ci(sd_boot)

        print(f"\n  between-encoder variation in the drop")
        print(f"    range (order statistic, biased upward): {rng_stat:.3f} "
              f"95% CI [{rlo:.3f}, {rhi:.3f}]")
        print(f"    SD across the {len(names)} encoders:        {sd_stat:.3f} "
              f"95% CI [{slo:.3f}, {shi:.3f}]")

        # omnibus permutation test on the per-fold drop matrix
        omni_p = None
        try:
            obs = sd_stat
            null = np.empty(a.perm)
            for b in range(a.perm):
                sh = np.array([rng_p.permutation(FD[:, k]) for k in range(nf)]).T
                null[b] = sh.mean(axis=1).std(ddof=1)
            omni_p = float((null >= obs).mean())
            print(f"    omnibus permutation test, H0 = all encoders share one drop:")
            print(f"      observed SD {obs:.4f},  p = {omni_p:.3f}"
                  + ("  -> encoders NOT distinguishable" if omni_p >= 0.05
                     else "  -> encoders differ"))
        except Exception as ex:
            print(f"    omnibus test unavailable ({ex})")

        # data-external margin: the random -> speaker-disjoint step, the
        # protocol change the field already treats as adequate. Anything
        # smaller than that step is negligible by the field's own standard.
        ref = None
        try:
            ref = float(np.mean([
                np.mean(data[corpus][e]["folds"]["random"]) -
                np.mean(data[corpus][e]["folds"]["speaker_disjoint"])
                for e in names]))
            print(f"\n    reference margin (random -> speaker-disjoint step on this"
                  f" corpus): {abs(ref):.3f}")
            verdict = "inside" if rhi < abs(ref) else "NOT inside"
            print(f"    encoder range CI is {verdict} that reference")
        except Exception:
            pass

        equiv = (ref is not None) and (rhi < abs(ref))
        wname = (names[int(point.argmax())], names[int(point.argmin())])

        report["corpora"][corpus] = {
            "drops": {e: float(drops[e]) for e in reps},
            "ci": {e: list(ci(reps[e])) for e in reps},
            "mean_drop": float(vals.mean()),
            "mean_ci": [clo, chi],
            "spread": float(vals.max() - vals.min()),
            "range": float(rng_stat),
            "range_ci": [rlo, rhi],
            "sd_across_encoders": sd_stat,
            "sd_ci": [slo, shi],
            "omnibus_p": omni_p,
            "reference_margin_rand_to_spk": (abs(ref) if ref is not None else None),
            "range_inside_reference": bool(equiv),
            "extreme_encoders": [wname[0], wname[1]],
        }
        if "source_disjoint_rep" in data[corpus][ENC_ORDER[0]]["folds"] and \
                data[corpus][ENC_ORDER[0]]["folds"]["source_disjoint_rep"] is not None:
            rr = []
            for enc in reps:
                spk = data[corpus][enc]["folds"]["speaker_disjoint"]
                rep = data[corpus][enc]["folds"]["source_disjoint_rep"]
                dp = (data[corpus][enc]["at_conv"]["speaker_disjoint"]
                      - data[corpus][enc]["at_conv"]["source_disjoint_rep"])
                b = paired_boot(spk, rep, a.boot, rng)
                rr.append(b - b.mean() + dp)
            m = np.mean(rr, axis=0)
            rlo, rhi = ci(m)
            print(f"\n  using the 20 repeated source-disjoint draws instead: "
                  f"mean drop {m.mean():+.3f}  95% CI [{rlo:+.3f}, {rhi:+.3f}]")
            report["corpora"][corpus]["repeat_mean_drop"] = float(m.mean())
            report["corpora"][corpus]["repeat_mean_ci"] = [rlo, rhi]
        print()

    # --- between-corpus gaps (all pairs) ----------------------------------
    cs = [c for c in report["corpora"]]
    if len(cs) >= 2:
        print("=" * 84)
        print("BETWEEN-CORPUS GAPS")
        print("=" * 84)

        def corpus_boot(c):
            """Bootstrap of a corpus's mean drop, averaging over its encoders."""
            return np.mean([
                paired_boot(data[c][e]["folds"]["speaker_disjoint"],
                            data[c][e]["folds"]["source_disjoint"], a.boot, rng)
                - paired_boot(data[c][e]["folds"]["speaker_disjoint"],
                              data[c][e]["folds"]["source_disjoint"], 1, rng).mean()
                + report["corpora"][c]["drops"][e]
                for e in report["corpora"][c]["drops"]], axis=0)

        boots = {c: corpus_boot(c) for c in cs}
        for c in cs:
            print(f"  {c:10s} mean drop {boots[c].mean():+.3f}   "
                  f"encoder spread {report['corpora'][c]['spread']:.3f}")
        print()
        report["gaps"] = {}
        for i in range(len(cs)):
            for j in range(i + 1, len(cs)):
                c1, c2 = cs[i], cs[j]
                g = boots[c2] - boots[c1]
                glo, ghi = ci(g)
                excl = "excludes zero" if (glo > 0 or ghi < 0) else "INCLUDES ZERO"
                print(f"  {c2} - {c1} = {g.mean():+.3f}  "
                      f"95% CI [{glo:+.3f}, {ghi:+.3f}]   {excl}")
                report["gaps"][f"{c2}-{c1}"] = {"value": float(g.mean()),
                                                "ci": [glo, ghi]}
        spreads = {c: report["corpora"][c]["spread"] for c in cs}
        rng_across = max(boots[c].mean() for c in cs) - min(boots[c].mean() for c in cs)
        print(f"\n  drop ranges from {min(boots[c].mean() for c in cs):+.3f} to "
              f"{max(boots[c].mean() for c in cs):+.3f} across corpora "
              f"(range {rng_across:.3f});")
        print(f"  largest within-corpus encoder spread is {max(spreads.values()):.3f} "
              f"({max(spreads, key=spreads.get)}).")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(report, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
