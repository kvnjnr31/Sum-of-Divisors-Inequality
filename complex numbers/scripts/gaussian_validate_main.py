# gaussian_validate_violin_main.py
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from gaussian_validate_helpers import (
    gaussian_primes_upto_norm,
    sample_gaussian_integers,
    stats_for_z,
    quantile_bin_indices,
    quantile_bins_on_integers,
    bin_counts,
    groups_from_bins,
    make_two_figures_from_groups,
)


def main() -> None:
    ap = argparse.ArgumentParser()

    # sampling
    ap.add_argument("--box", type=int, default=120, help="sample a,b uniformly from [-box, box]")
    ap.add_argument("--nsamples", type=int, default=80000, help="number of sampled Gaussian integers (nonzero)")
    ap.add_argument("--seed", type=int, default=0)

    # binning
    ap.add_argument("--bins_y", type=int, default=7, help="number of y-quantile bins")
    ap.add_argument("--bins_k", type=int, default=5, help="number of omega-quantile bins")
    ap.add_argument("--qmin_y", type=float, default=0.0, help="optional: keep y >= quantile(y, qmin_y); set 0 to disable")

    # plot style
    ap.add_argument("--show_points", action="store_true", help="overlay grey jitter points (behind violins)")
    ap.add_argument("--max_points", type=int, default=1200, help="legacy cap: max jitter points per bin (kept for compatibility)")
    ap.add_argument("--points_per_violin", type=int, default=36, help="number of jitter points to draw per violin (30–40 recommended)")
    ap.add_argument("--point_alpha", type=float, default=0.12)
    ap.add_argument("--point_size", type=float, default=8.0)
    ap.add_argument("--lw", type=float, default=2.2)
    ap.add_argument("--font", type=int, default=15)
    ap.add_argument("--tmin", type=float, default=0.4, help="lower ylim for t-plot")
    ap.add_argument("--tmax", type=float, default=1.0, help="upper ylim for t-plot")

    # outputs
    ap.add_argument("--outdir", default="outputs")
    ap.add_argument("--dpi", type=int, default=250)

    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------
    # Precompute Gaussian primes up to the largest norm in the box.
    # Norm(a+bi) <= 2*box^2
    # ------------------------------------------------------------
    max_norm = 2 * (args.box ** 2)
    primes = gaussian_primes_upto_norm(max_norm)

    # ------------------------------------------------------------
    # Sample z and compute stats
    # ------------------------------------------------------------
    zs = sample_gaussian_integers(args.box, args.nsamples, seed=args.seed)

    y = np.empty(zs.shape[0], dtype=float)
    L = np.empty_like(y)
    U = np.empty_like(y)
    W = np.empty_like(y)
    t = np.empty_like(y)
    omega = np.empty(zs.shape[0], dtype=int)

    for i, (a, b) in enumerate(zs):
        yi, Li, Ui, Wi, ti, wi = stats_for_z((int(a), int(b)), primes)
        y[i], L[i], U[i], W[i], t[i], omega[i] = yi, Li, Ui, Wi, ti, wi

    # ------------------------------------------------------------
    # Optional upper-tail filter on y
    # ------------------------------------------------------------
    if args.qmin_y and args.qmin_y > 0.0:
        thr = float(np.quantile(y, float(args.qmin_y)))
        keep = y >= thr
        y, L, U, W, t, omega = y[keep], L[keep], U[keep], W[keep], t[keep], omega[keep]

    # ============================================================
    # A) y-quantile bins (balanced counts)
    # ============================================================
    bin_id_y, mids_y = quantile_bin_indices(y, args.bins_y)
    B_y = mids_y.size
    counts_y = bin_counts(bin_id_y, B_y)

    y_bins = groups_from_bins(y, bin_id_y, B_y)
    L_bins = groups_from_bins(L, bin_id_y, B_y)
    U_bins = groups_from_bins(U, bin_id_y, B_y)
    W_bins = groups_from_bins(W, bin_id_y, B_y)
    t_bins = groups_from_bins(t, bin_id_y, B_y)

    make_two_figures_from_groups(
        outdir_path=outdir,
        y_bins=y_bins,
        L_bins=L_bins,
        U_bins=U_bins,
        W_bins=W_bins,
        t_bins=t_bins,
        mids=mids_y,
        counts=counts_y,
        fig_tag="binned_by_y",
        xaxis_label="y-quantile bin (midpoint)",
        title_prefix="Sensitivity to perturbations in y(z)",
        dpi=args.dpi,
        show_points=args.show_points,
        max_points=args.max_points,  # kept as a hard cap for safety
        point_alpha=args.point_alpha,
        point_size=args.point_size,
        lw=args.lw,
        font_size=args.font,
        t_ylim=(args.tmin, args.tmax),
        points_per_violin=args.points_per_violin,  # PATCH: new knob
    )

    """ Section For Further Analysis
    # ============================================================
    # B) omega-quantile bins (more balanced than raw k-bins)
    # ============================================================
    bin_id_k, mids_k = quantile_bins_on_integers(omega, args.bins_k)
    B_k = mids_k.size
    counts_k = bin_counts(bin_id_k, B_k)

    y_bins_k = groups_from_bins(y, bin_id_k, B_k)
    L_bins_k = groups_from_bins(L, bin_id_k, B_k)
    U_bins_k = groups_from_bins(U, bin_id_k, B_k)
    W_bins_k = groups_from_bins(W, bin_id_k, B_k)
    t_bins_k = groups_from_bins(t, bin_id_k, B_k)

    make_two_figures_from_groups(
        outdir_path=outdir,
        y_bins=y_bins_k,
        L_bins=L_bins_k,
        U_bins=U_bins_k,
        W_bins=W_bins_k,
        t_bins=t_bins_k,
        mids=mids_k,
        counts=counts_k,
        fig_tag="binned_by_omega",
        xaxis_label="k-quantile (on ω(z)) bin (midpoint)",
        title_prefix="Sensitivity to perturbations in y(z)",
        dpi=args.dpi,
        show_points=args.show_points,
        max_points=args.max_points,
        point_alpha=args.point_alpha,
        point_size=args.point_size,
        lw=args.lw,
        font_size=args.font,
        t_ylim=(args.tmin, args.tmax),
        points_per_violin=args.points_per_violin,
    )

    print(f"Wrote figures to: {outdir.resolve()}")
    print(f"y-bins counts: {counts_y}")
    print(f"omega-bins counts: {counts_k}")
    """

    print(f"Wrote figures to: {outdir.resolve()}")
    print(f"y-bins counts: {counts_y}")


if __name__ == "__main__":
    main()
