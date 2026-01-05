from __future__ import annotations

# ============================================================
# validate_sod_violin_main.py  (CORRECTED)
#
# Purpose:
#   Produce violin-based, binned figures for:
#       y2(n) = log( sigma(n) / (2n) )
#   with deterministic prime-support envelopes shifted by -log 2.
#
# Outputs:
#   (A) By y2-quantile bins (approximately equal counts):
#       y2/L2/U2 violins and W/t violins
#   (B) Sister plots binned by k=ω(n):
#       same figures, with per-bin counts shown on x ticks
#   (C) Optional equal-count "k-quantile" bins (on ω(n)):
#       balances counts but mixes k values within bins.
# ============================================================

import argparse
from pathlib import Path

import numpy as np

from validate_sod_violin_helpers import (
    set_vis_params,
    sigma_sieve,
    spf_sieve,
    distinct_primes_from_spf,
    log_envelope_from_prime_support,
    quantile_bin_indices,
    filter_upper_tail,
    bin_by_k,
    compress_bins,
    make_two_figures,
)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--N", type=int, default=400_000, help="sieve max (increase for stronger stats)")
    ap.add_argument("--nsample", type=int, default=160_000, help="random sample size from {2..N}")

    ap.add_argument("--bins", type=int, default=7, help="number of y2-quantile bins")
    ap.add_argument("--qmin", type=float, default=0.60, help="tail filter quantile on y2 (<=0 disables)")

    ap.add_argument("--kmax", type=int, default=10, help="max k=ω(n) bin to show (clips higher into last bin)")
    ap.add_argument("--equalize_k", action="store_true", help="also produce equal-count bins by quantiles of ω(n)")

    ap.add_argument("--outdir", default="stats_outputs")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--fontsize", type=int, default=15)
    ap.add_argument("--lw", type=float, default=2.2)
    ap.add_argument("--violin_width", type=float, default=0.86)

    ap.add_argument("--show_points", action="store_true", help="overlay jittered sample points (recommended)")
    ap.add_argument("--max_points", type=int, default=300, help="max jittered points per violin (legacy cap)")
    ap.add_argument("--points_per_violin", type=int, default=36, help="target jitter points per violin (30–40 recommended)")
    ap.add_argument("--point_size", type=float, default=10.0, help="scatter point size")

    ap.add_argument("--min_bin_count", type=int, default=8, help="minimum samples per bin to plot")

    args = ap.parse_args()

    set_vis_params(args.fontsize, args.lw)

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    sig = sigma_sieve(args.N)
    spf = spf_sieve(args.N)

    rng = np.random.default_rng(args.seed)
    n_all = np.arange(2, args.N + 1, dtype=np.int32)
    ns = rng.choice(n_all, size=min(args.nsample, n_all.size), replace=False)

    log2 = float(np.log(2.0))

    y2 = np.empty(ns.size, dtype=np.float64)
    L2 = np.empty_like(y2)
    U2 = np.empty_like(y2)
    k = np.empty(ns.size, dtype=np.int32)

    for i, n in enumerate(ns):
        primes, omega = distinct_primes_from_spf(int(n), spf)
        L, U = log_envelope_from_prime_support(primes)

        # y2(n) = log( sigma(n)/(2n) ) = log(sigma(n)/n) - log(2)
        y = float(np.log(sig[n] / float(n)))
        y2[i] = y - log2
        L2[i] = L - log2
        U2[i] = U - log2
        k[i] = int(omega)

    keep = filter_upper_tail(y2, args.qmin)
    y2 = y2[keep]
    L2 = L2[keep]
    U2 = U2[keep]
    k = k[keep]

    W = np.maximum(U2 - L2, 1e-16)
    t = np.clip((y2 - L2) / W, 0.0, 1.0)

    # ----------------------------
    # (A) y2-quantile bins
    # ----------------------------
    bin_id_q, mids_q = quantile_bin_indices(y2, args.bins)
    bin_id_q, mids_q, (y2_q, L2_q, U2_q, W_q, t_q), counts_q = compress_bins(
        bin_id_q, mids_q, y2, L2, U2, W, t, min_count=args.min_bin_count
    )

    make_two_figures(
        outdir_path=outdir,
        tag="",
        bin_id=bin_id_q,
        mids=mids_q,
        counts=counts_q,
        y2=y2_q,
        L2=L2_q,
        U2=U2_q,
        W=W_q,
        t=t_q,
        dpi=args.dpi,
        show_points=args.show_points,
        lw=args.lw,
        width=args.violin_width,
        max_points=args.max_points,              # legacy cap
        point_size=args.point_size,
        xlabel_prefix="y2-quantile",
        label_mode="float",
        points_per_violin=args.points_per_violin,  # NEW knob
    )
    """
    # ----------------------------
    # (B) k=ω(n) bins
    # ----------------------------
    bin_id_k, mids_k = bin_by_k(k, kmax=args.kmax)
    bin_id_k, mids_k, (y2_k, L2_k, U2_k, W_k, t_k), counts_k = compress_bins(
        bin_id_k, mids_k, y2, L2, U2, W, t, min_count=args.min_bin_count
    )

    make_two_figures(
        outdir_path=outdir,
        tag="_by_k",
        bin_id=bin_id_k,
        mids=mids_k,
        counts=counts_k,
        y2=y2_k,
        L2=L2_k,
        U2=U2_k,
        W=W_k,
        t=t_k,
        dpi=args.dpi,
        show_points=args.show_points,
        lw=args.lw,
        width=args.violin_width,
        max_points=args.max_points,
        point_size=args.point_size,
        xlabel_prefix="k=ω(n)",
        label_mode="int",
        points_per_violin=args.points_per_violin,
    )

    # ----------------------------
    # (C) Optional equal-count k-quantile bins
    # ----------------------------
    if args.equalize_k:
        bin_id_kq, mids_kq = quantile_bin_indices(k.astype(float), args.bins)
        bin_id_kq, mids_kq, (y2_kq, L2_kq, U2_kq, W_kq, t_kq), counts_kq = compress_bins(
            bin_id_kq, mids_kq, y2, L2, U2, W, t, min_count=args.min_bin_count
        )

        make_two_figures(
            outdir_path=outdir,
            tag="_k_equalcount",
            bin_id=bin_id_kq,
            mids=mids_kq,
            counts=counts_kq,
            y2=y2_kq,
            L2=L2_kq,
            U2=U2_kq,
            W=W_kq,
            t=t_kq,
            dpi=args.dpi,
            show_points=args.show_points,
            lw=args.lw,
            width=args.violin_width,
            max_points=args.max_points,
            point_size=args.point_size,
            xlabel_prefix="k-quantile (on ω(n))",
            label_mode="float",
            points_per_violin=args.points_per_violin,
        )
    """
    # ----------------------------
    # Console summary
    # ----------------------------
    print(f"Wrote figures to: {outdir.resolve()}")
    print(f"Tail filter: qmin={args.qmin} (set --qmin 0 to disable).")
    print(f"y2-quantile bins plotted: {mids_q.size} (requested {args.bins})")
    print(f"Counts per y2-bin: {counts_q.tolist()}")
    #print(f"k-bins plotted: {mids_k.size} (kmax={args.kmax})")
    #print(f"Counts per k-bin: {counts_k.tolist()}")
    #if args.equalize_k:
       # print(f"Equal-count k-quantile bins plotted: {mids_kq.size}")
        #print(f"Counts per equal-count k-bin: {counts_kq.tolist()}")


if __name__ == "__main__":
    main()
