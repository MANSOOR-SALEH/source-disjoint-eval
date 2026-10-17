"""
End-to-end multi-draw driver: build folds, then score BOTH arms (corrected and
uncorrected) for every encoder, for every seed, into clearly named trees. Then
call delta_report.py for the paired analysis.

    python src/run_draws.py --corpus newcorpus \
        --manifest manifests/newcorpus.csv --n-folds 10 --n-repeats 20 \
        --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19

For each seed it:
  1. builds folds/{corpus}_seed{seed}.json  (folds.py, source matching ON, so
     both train_srcmatched and test_srcmatched exist),
  2. runs probe.py per encoder TWICE on those folds:
       corrected   -> results_corr_seed{seed}/   (uses test_srcmatched: Step 5),
       uncorrected -> results_uncorr_seed{seed}/  (--unmatched: full test),
  3. after all seeds, runs delta_report.py over the two trees.

The corrected/uncorrected split is thus explicit and symmetric -- no reliance on
a 'results_fix' naming convention. convention_layer and nested_drop are what the
paired analysis reads, so a per-encoder --fixed-layer is optional (probe.py will
default it with a warning; it does not affect the reported delta).

This is the only script that needs a GPU, and only through probe.py. 20 seeds x
7 encoders x 2 arms is cheap: the encoders are frozen and the probe is a 256-unit
MLP on cached features.
"""
import argparse
import os
import subprocess
import sys

ENCODERS = ["wavlm", "xlsr", "mhubert", "mms", "whisper", "w2vbert", "artst"]


def run(cmd, dry):
    print("  " + " ".join(str(c) for c in cmd))
    if dry:
        return True
    if subprocess.run([str(c) for c in cmd]).returncode != 0:
        print(f"  !! failed: {' '.join(str(c) for c in cmd)}")
        return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(20)))
    ap.add_argument("--encoders", nargs="+", default=ENCODERS)
    ap.add_argument("--n-folds", type=int, default=10)
    ap.add_argument("--n-inner", type=int, default=3)
    ap.add_argument("--tag", default="",
                    help="suffix for folds/result dirs so runs at different "
                         "settings (e.g. --tag _f3) don't overwrite each other")
    ap.add_argument("--n-repeats", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--features-dir", default="features")
    ap.add_argument("--fixed-layer", type=int, default=None,
                    help="optional; does not affect convention/nested delta")
    ap.add_argument("--no-analyse", action="store_true")
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    py = sys.executable
    corr_dirs, uncorr_dirs = [], []

    for seed in a.seeds:
        folds = f"folds/{a.corpus}{a.tag}_seed{seed}.json"
        print(f"\n=== seed {seed}: folds ===")
        cmd = [py, "src/folds.py", "--manifest", a.manifest, "--out", folds,
               "--n-folds", a.n_folds, "--n-inner", a.n_inner, "--seed", seed]
        if a.n_repeats:
            cmd += ["--n-repeats", a.n_repeats]
        if not run(cmd, a.dry_run):
            continue

        cdir = f"results_corr{a.tag}_seed{seed}"
        udir = f"results_uncorr{a.tag}_seed{seed}"
        os.makedirs(cdir, exist_ok=True)
        os.makedirs(udir, exist_ok=True)
        corr_dirs.append(cdir)
        uncorr_dirs.append(udir)

        for enc in a.encoders:
            feat = f"{a.features_dir}/{a.corpus}.{enc}.npz"
            if not a.dry_run and not os.path.exists(feat):
                print(f"  skip {enc}: {feat} missing")
                continue
            common = [py, "src/probe.py", "--features", feat,
                      "--manifest", a.manifest, "--folds", folds,
                      "--epochs", a.epochs]
            if a.fixed_layer is not None:
                common += ["--fixed-layer", a.fixed_layer]
            # corrected: default (uses test_srcmatched when present)
            run(common + ["--out", f"{cdir}/{a.corpus}.{enc}.json"], a.dry_run)
            # uncorrected: --unmatched (full test, no Step 5)
            run(common + ["--unmatched",
                          "--out", f"{udir}/{a.corpus}.{enc}.json"], a.dry_run)

    if a.dry_run or a.no_analyse:
        print("\n(stopping before analysis)")
        return
    if not corr_dirs:
        print("no results produced; nothing to analyse")
        return

    print("\n=== paired delta analysis ===")
    run([py, "src/delta_report.py",
         "--uncorr", *uncorr_dirs, "--corr", *corr_dirs,
         "--corpora", a.corpus, "--encoders", *a.encoders,
         "--boot", a.boot,
         "--out", f"results_srcmatch/{a.corpus}{a.tag}_delta_report.json",
         "--fig", f"fig_{a.corpus}{a.tag}_delta.pdf"], a.dry_run)


if __name__ == "__main__":
    main()
