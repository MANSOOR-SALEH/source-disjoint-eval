"""
Paired correction effect delta across many matched draws, with across-draw
bootstrap CIs and a draw x encoder variance decomposition.

    python src/delta_report.py \
        --uncorr results_seed0 results_seed1 results_seed2 results_seed3 results_seed4 \
        --corr   results_fix   results_fix_seed1 results_fix_seed2 results_fix_seed3 results_fix_seed4 \
        --corpora aydid sada --out results_srcmatch/delta_report.json --fig fig_delta.pdf

What it computes
----------------
For each corpus, each draw d (one uncorrected dir paired with one corrected dir)
and each encoder e, it reads the protocol contrast

    Delta = macro-F1(speaker_disjoint) - macro-F1(source_disjoint)

at TWO layer-selection rules already stored by probe.py:

    convention : convention_layer.spk_to_src_drop   (layer chosen on
                 speaker-disjoint scores -- current practice, test-informed)
    nested     : nested_drop.drop                   (layer chosen on inner
                 training splits only -- test-blind)

The paired correction effect is

    delta_correction = Delta_uncorrected - Delta_corrected          (per d, e)

i.e. how much the uncorrected (test-unrestricted) comparison inflates the drop
relative to the partition-consistent one. It reports, per corpus and selection
rule:

  * mean Delta_corrected and Delta_uncorrected with across-draw SD
  * mean delta_correction, across-draw SD, and a bootstrap 95% CI obtained by
    resampling DRAWS (the dominant variance term; a fold bootstrap conditions on
    one draw and misses it)
  * a balanced two-way random-effects variance decomposition of Delta_corrected
    into across-draw, across-encoder, and residual components, and the
    draw:encoder variance ratio.

It reads only the per-run JSONs that probe.py already writes, so it needs no GPU
and scales unchanged from 5 draws to 20-30. The same call produces the paper's
numbers on any corpus once its draws have been run.
"""
import argparse
import glob
import json
import os

import numpy as np

SELECTIONS = ("convention", "nested")


def _delta(run, selection):
    """Pull the stored speaker->source drop for one run JSON. None if absent."""
    if selection == "convention":
        cl = run.get("convention_layer") or {}
        return cl.get("spk_to_src_drop")
    nd = run.get("nested_drop") or {}
    return nd.get("drop")


def _load(dir_, corpus, enc):
    p = os.path.join(dir_, f"{corpus}.{enc}.json")
    if not os.path.exists(p):
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def _var_components(M):
    """Balanced two-way random-effects decomposition of M[draw, encoder].

    One observation per cell, so the residual term IS the draw x encoder
    interaction. Returns (sigma2_draw, sigma2_enc, sigma2_resid) with negative
    estimates clamped to 0 (standard for random-effects components).
    """
    M = np.asarray(M, float)
    nd, ne = M.shape
    grand = M.mean()
    row = M.mean(1)           # per-draw means
    col = M.mean(0)           # per-encoder means
    if nd < 2 or ne < 2:
        return float("nan"), float("nan"), float("nan")
    ms_d = ne * ((row - grand) ** 2).sum() / (nd - 1)
    ms_e = nd * ((col - grand) ** 2).sum() / (ne - 1)
    resid = M - row[:, None] - col[None, :] + grand
    ms_r = (resid ** 2).sum() / ((nd - 1) * (ne - 1))
    s2_d = max(0.0, (ms_d - ms_r) / ne)
    s2_e = max(0.0, (ms_e - ms_r) / nd)
    return s2_d, s2_e, ms_r


def _boot_ci(M, B=10000, seed=0, agg=np.mean):
    """Bootstrap CI for the grand mean of M[draw, encoder] by resampling DRAWS
    (rows) with replacement -- carries the across-draw uncertainty."""
    M = np.asarray(M, float)
    nd = M.shape[0]
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for b in range(B):
        rows = rng.integers(0, nd, nd)
        stats[b] = agg(M[rows])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def analyse(uncorr_dirs, corr_dirs, corpora, encoders, boot=10000):
    assert len(uncorr_dirs) == len(corr_dirs), \
        "give one corrected dir per uncorrected dir (same draw order)"
    out = {"draws": len(uncorr_dirs), "uncorr_dirs": uncorr_dirs,
           "corr_dirs": corr_dirs, "encoders": encoders, "corpora": {}}
    for corpus in corpora:
        cres = {}
        for sel in SELECTIONS:
            Du, Dc, enc_ok = [], [], []
            for e in encoders:
                col_u, col_c, ok = [], [], True
                for ud, cd in zip(uncorr_dirs, corr_dirs):
                    ru, rc = _load(ud, corpus, e), _load(cd, corpus, e)
                    du = _delta(ru, sel) if ru else None
                    dc = _delta(rc, sel) if rc else None
                    if du is None or dc is None:
                        ok = False
                        break
                    col_u.append(du)
                    col_c.append(dc)
                if ok and col_u:
                    Du.append(col_u)
                    Dc.append(col_c)
                    enc_ok.append(e)
            if not enc_ok:
                continue
            Du = np.array(Du).T          # [draw, encoder]
            Dc = np.array(Dc).T
            delta = Du - Dc              # paired correction effect
            draw_mean_dc = Dc.mean(1)    # per-draw encoder-mean (corrected)
            draw_mean_del = delta.mean(1)
            s2_d, s2_e, s2_r = _var_components(Dc)
            lo, hi = _boot_ci(delta, B=boot)
            cres[sel] = {
                "encoders": enc_ok,
                "n_draws": int(Dc.shape[0]),
                "Delta_corrected_mean": float(Dc.mean()),
                "Delta_corrected_sd_across_draws": float(draw_mean_dc.std(ddof=1)),
                "Delta_uncorrected_mean": float(Du.mean()),
                "delta_correction_mean": float(delta.mean()),
                "delta_correction_sd_across_draws": float(draw_mean_del.std(ddof=1)),
                "delta_correction_ci95": [lo, hi],
                "delta_sign_stable": bool(draw_mean_del.min() > 0 or draw_mean_del.max() < 0),
                "var_draw": s2_d, "var_encoder": s2_e, "var_resid": s2_r,
                "sd_draw": float(np.sqrt(s2_d)) if s2_d == s2_d else None,
                "sd_encoder": float(np.sqrt(s2_e)) if s2_e == s2_e else None,
                "draw_to_encoder_var_ratio": (float(s2_d / s2_e)
                                              if s2_e and s2_e > 0 else None),
                "per_draw_Delta_corrected": draw_mean_dc.tolist(),
                "per_draw_delta": draw_mean_del.tolist(),
            }
        out["corpora"][corpus] = cres
    return out


def _print(rep):
    print("=" * 78)
    print(f"PAIRED CORRECTION EFFECT  ({rep['draws']} draws, "
          f"{len(rep['encoders'])} encoders)")
    print("=" * 78)
    for corpus, cres in rep["corpora"].items():
        print(f"\n{corpus.upper()}")
        for sel, r in cres.items():
            ratio = r["draw_to_encoder_var_ratio"]
            print(f"  [{sel}]  n_draws={r['n_draws']}")
            print(f"    Delta uncorrected : {r['Delta_uncorrected_mean']:+.3f}")
            print(f"    Delta corrected   : {r['Delta_corrected_mean']:+.3f} "
                  f"(SD across draws {r['Delta_corrected_sd_across_draws']:.3f})")
            print(f"    delta (correction): {r['delta_correction_mean']:+.3f} "
                  f"(SD {r['delta_correction_sd_across_draws']:.3f}; "
                  f"95% CI [{r['delta_correction_ci95'][0]:+.3f}, "
                  f"{r['delta_correction_ci95'][1]:+.3f}])"
                  + ("" if r["delta_sign_stable"] else "  sign not stable across draws"))
            sd_d = r["sd_draw"]; sd_e = r["sd_encoder"]
            print(f"    variance: across-draw SD {sd_d:.3f}, across-encoder SD "
                  f"{sd_e:.3f}"
                  + (f", draw:encoder var ratio {ratio:.0f}x" if ratio else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uncorr", nargs="+", required=True,
                    help="uncorrected (test-unrestricted) result dirs, one per draw")
    ap.add_argument("--corr", nargs="+", required=True,
                    help="corrected (test-restricted) result dirs, same draw order")
    ap.add_argument("--corpora", nargs="+", default=["aydid", "sada"])
    ap.add_argument("--encoders", nargs="+",
                    default=["wavlm", "xlsr", "mhubert", "mms", "whisper",
                             "w2vbert", "artst"])
    ap.add_argument("--boot", type=int, default=10000)
    ap.add_argument("--out", default="results_srcmatch/delta_report.json")
    ap.add_argument("--fig", default=None, help="optional delta-distribution PDF")
    a = ap.parse_args()

    rep = analyse(a.uncorr, a.corr, a.corpora, a.encoders, boot=a.boot)
    _print(rep)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(rep, open(a.out, "w"), indent=1)
    print(f"\nwrote {a.out}")

    if a.fig:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            corpora = list(rep["corpora"])
            fig, ax = plt.subplots(1, len(corpora), figsize=(3.2 * len(corpora), 2.8),
                                   squeeze=False)
            for j, c in enumerate(corpora):
                r = rep["corpora"][c].get("nested") or rep["corpora"][c].get("convention")
                if not r:
                    continue
                pd_ = r["per_draw_delta"]
                ax[0][j].axhline(0, color="0.7", lw=0.8)
                ax[0][j].plot(range(1, len(pd_) + 1), pd_, "o", ms=4, color="black")
                ax[0][j].axhspan(r["delta_correction_ci95"][0],
                                 r["delta_correction_ci95"][1], color="0.85")
                ax[0][j].set_title(f"{c}  delta={r['delta_correction_mean']:+.3f}",
                                   fontsize=9)
                ax[0][j].set_xlabel("draw", fontsize=8)
                ax[0][j].set_ylabel("delta (uncorr - corr)", fontsize=8)
                ax[0][j].tick_params(labelsize=7)
            plt.tight_layout()
            plt.savefig(a.fig, bbox_inches="tight")
            print(f"wrote {a.fig}")
        except ImportError:
            print("matplotlib not available; skipped figure")


if __name__ == "__main__":
    main()
