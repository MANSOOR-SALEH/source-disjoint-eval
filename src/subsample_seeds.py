"""
How sensitive is the source-count control to WHICH sources are kept?

    # cheap: 3 seeds, one encoder, both corpora  (~2 h)
    python src/subsample_seeds.py --seeds 0 1 2 --encoders wavlm

    # full: 5 seeds, all encoders  (~20 h)
    python src/subsample_seeds.py --seeds 0 1 2 3 4

Why this exists
---------------
Source-count matching keeps a random subset of the training sources -- 27 of 40
on AYDID, 56 of 63 on SADA. Which subset is a single random draw, so the
reported drop rides on one partition. Because the AYDID headline is a null,
that fragility is exactly where a reviewer will push: is -0.002 the answer, or
one lucky draw?

This re-runs the whole matched pipeline under several seeds and reports the
mean drop and its spread ACROSS DRAWS. The paper can then say "the drop varies
by +-X across five subsamples" instead of "the subsample is a single draw whose
sensitivity we did not quantify".

Note the two corpora are not equally exposed. AYDID discards 32% of its
training sources, SADA 11%, so AYDID's number should move more between draws.
That asymmetry is itself worth reporting -- it is the quantitative version of
the objection that matching a COUNT does not match acoustic DIVERSITY.

Outputs land in results_seed<N>/ and a summary in results_srcmatch/seed_sensitivity.json.
Nothing already computed is overwritten.
"""

import argparse
import json
import os
import subprocess
import sys

import numpy as np

CORPORA = {
    "aydid": ("manifests/aydid_clean.csv", 10, 20),
    "sada": ("manifests/sada.csv", 10, 0),
}
ENCODERS = ["wavlm", "xlsr", "mhubert", "mms", "whisper", "w2vbert", "artst"]


def run(cmd, dry):
    print("  " + " ".join(cmd))
    if dry:
        return True
    r = subprocess.run(cmd)
    if r.returncode != 0:
        print(f"  !! failed: {' '.join(cmd)}")
        return False
    return True


def probe_args(existing):
    """Recover --fixed-layer / --layer-stride / --no-linear from a prior result
    so the reruns reproduce the original configuration exactly."""
    if not os.path.exists(existing):
        return []
    r = json.load(open(existing))
    out = []
    cl = r.get("convention_layer") or {}
    fl = r.get("fixed_layer", cl.get("layer"))
    if fl is not None:
        out += ["--fixed-layer", str(int(fl))]
    layers = r.get("layers")
    if layers and len(layers) > 1:
        out += ["--layer-stride", str(int(layers[1] - layers[0]))]
    if "profile_linear_mean" not in next(iter(r.get("protocols", {}).values()), {}):
        out += ["--no-linear"]
    st = r.get("layer_stride")
    if st and "--layer-stride" not in out:
        out += ["--layer-stride", str(int(st))]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--corpora", nargs="+", default=list(CORPORA))
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--reference", default="results_fix",
                    help="existing matched results, used to recover probe flags")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    py = sys.executable
    made = {}

    for seed in a.seeds:
        rdir = f"results_seed{seed}"
        os.makedirs(rdir, exist_ok=True)
        for corpus in a.corpora:
            manifest, nfolds, nrep = CORPORA[corpus]
            folds = f"folds/{corpus}_seed{seed}.json"
            print(f"\n=== seed {seed} | {corpus} : folds ===")
            cmd = [py, "src/folds.py", "--manifest", manifest, "--out", folds,
                   "--n-folds", str(nfolds), "--seed", str(seed)]
            if nrep:
                cmd += ["--n-repeats", str(nrep)]
            if not run(cmd, a.dry_run):
                continue

            for enc in a.encoders:
                feat = f"features/{corpus}.{enc}.npz"
                if not a.dry_run and not os.path.exists(feat):
                    print(f"  skip {enc}: {feat} missing")
                    continue
                out = f"{rdir}/{corpus}.{enc}.json"
                if os.path.exists(out):
                    print(f"  have {out}")
                    made.setdefault(seed, []).append((corpus, enc))
                    continue
                cmd = [py, "src/probe.py", "--features", feat,
                       "--manifest", manifest, "--folds", folds,
                       "--out", out, "--epochs", str(a.epochs)]
                cmd += probe_args(f"{a.reference}/{corpus}.{enc}.json")
                if run(cmd, a.dry_run):
                    made.setdefault(seed, []).append((corpus, enc))

    if a.dry_run:
        print("\ndry run: nothing executed")
        return

    # ---- summarise across seeds -----------------------------------------
    # The quantity the main table needs is the drop with its ACROSS-DRAW
    # uncertainty. Fold bootstrapping conditions on one subsample and therefore
    # omits the dominant variance term when the control discards a large
    # fraction of sources.
    print("\n" + "=" * 78)
    print("SOURCE-SUBSAMPLE UNCERTAINTY")
    print("=" * 78)
    summary = {}
    for corpus in a.corpora:
        rows = {}
        for seed in a.seeds:
            for enc in a.encoders:
                p = f"results_seed{seed}/{corpus}.{enc}.json"
                if not os.path.exists(p):
                    continue
                r = json.load(open(p))
                P = r.get("protocols", {})
                if "speaker_disjoint" in P and "source_disjoint" in P:
                    rows.setdefault(enc, {})[seed] = (
                        P["speaker_disjoint"]["at_convention_layer"]
                        - P["source_disjoint"]["at_convention_layer"])
        if not rows:
            continue

        print(f"\n{corpus.upper()}")
        hdr = (f"{'encoder':10s}" + "".join(f"{'s'+str(s_):>9s}" for s_ in a.seeds)
               + f"{'mean':>9s}{'SD':>8s}{'range':>8s}")
        print(hdr); print("-" * len(hdr))
        per_seed = {s_: [] for s_ in a.seeds}
        enc_means = []
        for enc in a.encoders:
            if enc not in rows:
                continue
            vals = [rows[enc].get(s_) for s_ in a.seeds]
            got = np.array([v for v in vals if v is not None])
            for s_ in a.seeds:
                if rows[enc].get(s_) is not None:
                    per_seed[s_].append(rows[enc][s_])
            enc_means.append(got.mean())
            sd = got.std(ddof=1) if len(got) > 1 else 0.0
            print(f"{enc:10s}"
                  + "".join(f"{(f'{v:+.3f}' if v is not None else '--'):>9s}" for v in vals)
                  + f"{got.mean():>+9.3f}{sd:>8.3f}{np.ptp(got):>8.3f}")

        seed_means = np.array([np.mean(v) for v in per_seed.values() if v])
        print("-" * len(hdr))
        print(f"{'corpus':10s}"
              + "".join(f"{np.mean(per_seed[s_]):>+9.3f}" if per_seed[s_] else f"{'--':>9s}"
                        for s_ in a.seeds)
              + f"{seed_means.mean():>+9.3f}"
              + f"{(seed_means.std(ddof=1) if len(seed_means)>1 else 0):>8.3f}"
              + f"{np.ptp(seed_means):>8.3f}")

        # percentile interval over draws, the figure to quote
        lo, hi = (np.percentile(seed_means, [2.5, 97.5]) if len(seed_means) > 2
                  else (seed_means.min(), seed_means.max()))
        summary[corpus] = {
            "per_seed_corpus_mean": {int(k): float(np.mean(v))
                                     for k, v in per_seed.items() if v},
            "mean": float(seed_means.mean()),
            "sd_across_draws": float(seed_means.std(ddof=1)) if len(seed_means) > 1 else 0.0,
            "range_across_draws": float(np.ptp(seed_means)),
            "interval": [float(lo), float(hi)],
            "n_draws": int(len(seed_means)),
            "per_encoder": rows,
        }
        print(f"\n  across {len(seed_means)} draws: {seed_means.mean():+.3f} "
              f"(SD {seed_means.std(ddof=1) if len(seed_means)>1 else 0:.3f}, "
              f"range [{seed_means.min():+.3f}, {seed_means.max():+.3f}])")
        if seed_means.min() < 0 < seed_means.max():
            print("  -> the sign of the effect is not stable across subsamples")

    os.makedirs("results_srcmatch", exist_ok=True)
    out = "results_srcmatch/seed_sensitivity.json"
    json.dump({"seeds": a.seeds, "encoders": a.encoders, "corpora": summary},
              open(out, "w"), indent=1)
    print(f"\nwrote {out}")

    if len(summary) == 2:
        cs = list(summary)
        d = summary[cs[1]]["mean"] - summary[cs[0]]["mean"]
        print(f"\n  between-corpus difference of draw means: {abs(d):.3f}")

    print("\nFOR THE PAPER:")
    for c, r in summary.items():
        print(f"  {c}: drop {r['mean']:+.3f} (SD {r['sd_across_draws']:.3f} across "
              f"{r['n_draws']} independent source subsamples; range "
              f"{r['range_across_draws']:.3f})")


if __name__ == "__main__":
    main()