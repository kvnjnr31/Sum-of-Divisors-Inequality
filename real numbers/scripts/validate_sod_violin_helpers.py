from __future__ import annotations

"""
validate_sod_violin_helpers.py  (PATCH)

Patch goals (REAL case jitter points):
  1) Add per-violin subsampling control (e.g., 30–40 points per violin).
  2) Keep backward compatibility with existing calls that pass max_points.
  3) Use quantile-stratified subsampling (preserves tails + center).
  4) Remove duplicate violin_group definition (keep ONE canonical version).
"""

from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt


def set_vis_params(fontsize: int, lw: float) -> None:
    """Set global matplotlib rcParams for readability (fonts + line widths)."""
    plt.rcParams.update(
        {
            "font.size": fontsize,
            "axes.titlesize": fontsize + 2,
            "axes.labelsize": fontsize + 1,
            "xtick.labelsize": fontsize,
            "ytick.labelsize": fontsize,
            "legend.fontsize": fontsize,
            "axes.linewidth": lw,
            "lines.linewidth": lw,
            "xtick.major.width": lw,
            "ytick.major.width": lw,
            "xtick.minor.width": lw,
            "ytick.minor.width": lw,
        }
    )


def sigma_sieve(N: int) -> np.ndarray:
    """Exact σ(n) for n<=N via divisor-sum sieve."""
    sig = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sig[d::d] += d
    return sig


def spf_sieve(N: int) -> np.ndarray:
    """Smallest-prime-factor sieve for n<=N."""
    spf = np.zeros(N + 1, dtype=np.int32)
    for i in range(2, N + 1):
        if spf[i] == 0:
            spf[i] = i
            if i * i <= N:
                for j in range(i * i, N + 1, i):
                    if spf[j] == 0:
                        spf[j] = i
    return spf


def distinct_primes_from_spf(n: int, spf: np.ndarray) -> Tuple[List[int], int]:
    """Return (distinct prime divisors, ω(n)) for integer n using SPF."""
    primes: List[int] = []
    last = 0
    while n > 1:
        p = int(spf[n]) if spf[n] != 0 else n
        if p != last:
            primes.append(p)
            last = p
        while n % p == 0:
            n //= p
    return primes, len(primes)


def log_envelope_from_prime_support(primes: List[int]) -> Tuple[float, float]:
    """
    Deterministic envelopes (integer case) for y(n)=log(sigma(n)/n):
        L(n) = -Σ_{p|n} 1/(p-1)
        U(n) =  Σ_{p|n} 1/(p-1)
    """
    if not primes:
        return 0.0, 0.0
    arr = np.asarray(primes, dtype=np.float64)
    s = float(np.sum(1.0 / (arr - 1.0)))
    return -s, s


def quantile_bin_indices(x: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return:
      - bin_id for each element of x (0..B-1)
      - bin_midpoints for labeling (size B)
    Uses quantile edges; collapses bins if edges duplicate.
    """
    q = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(x, q)

    edges = np.unique(edges)
    if edges.size < 3:
        bin_id = np.zeros_like(x, dtype=int)
        mids = np.array([float(np.median(x))], dtype=float)
        return bin_id, mids

    bin_id = np.digitize(x, edges[1:-1], right=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return bin_id, mids


def filter_upper_tail(values: np.ndarray, qmin: float) -> np.ndarray:
    """Boolean mask selecting values >= quantile(values, qmin). Set qmin<=0 to disable."""
    if qmin is None or qmin <= 0.0:
        return np.ones(values.size, dtype=bool)
    thr = float(np.quantile(values, qmin))
    return values >= thr


def bin_by_k(k: np.ndarray, kmax: int | None = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Bin by omega k (distinct prime count).
    Values above kmax are clipped into the last bin.
    """
    k = k.astype(int)
    if kmax is None:
        kmax = int(np.max(k))
    ks = np.arange(1, kmax + 1, dtype=int)
    bin_id = np.clip(k - 1, 0, ks.size - 1)
    mids = ks.astype(float)
    return bin_id, mids


def compress_bins(
    bin_id: np.ndarray,
    mids: np.ndarray,
    *arrays: np.ndarray,
    min_count: int = 2,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], np.ndarray]:
    """
    Drop bins with count < min_count and reindex remaining bins to 0..B'-1.

    Returns:
      new_bin_id, new_mids, new_arrays (same count/order as input arrays), new_counts
    """
    B = int(mids.size)
    counts = np.array([(bin_id == b).sum() for b in range(B)], dtype=int)
    keep_bins = np.where(counts >= min_count)[0]

    if keep_bins.size == 0:
        raise ValueError(
            "All bins are empty after filtering. "
            "Try lowering --qmin or increasing --nsample, or set --qmin 0."
        )

    remap = -np.ones(B, dtype=int)
    remap[keep_bins] = np.arange(keep_bins.size, dtype=int)
    new_bin_id = remap[bin_id]

    mask = new_bin_id >= 0
    new_bin_id = new_bin_id[mask]
    new_mids = mids[keep_bins]
    new_arrays = [a[mask] for a in arrays]
    new_counts = counts[keep_bins]
    return new_bin_id, new_mids, new_arrays, new_counts


# ============================================================
# Jittered scatter: quantile-stratified subsampling (REAL case)
# ============================================================

def _jittered_scatter(
    ax,
    y: np.ndarray,
    x: float,
    width: float,
    *,
    seed: int = 7,
    color: str = "k",
    alpha: float = 1.0,
    size: float = 30.0,
    zorder: int = 1,                 # behind violins, above grid
    max_points: int | None = None,   # legacy cap (per violin)
    n_samples: int | None = None,    # preferred control (per violin)
    **_ignored,                      # swallow unused kwargs safely
) -> None:
    """
    Draw jittered background points behind violins with distribution-aware
    subsampling to reduce clutter while preserving shape.

    Priority:
        n_samples (if provided) > max_points (legacy)

    Sampling strategy:
        Quantile-stratified selection to preserve tails and central mass.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return

    rng = np.random.default_rng(seed)

    # Resolve target count
    if n_samples is not None:
        target = int(n_samples)
    elif max_points is not None:
        target = int(max_points)
    else:
        target = int(y.size)

    if target <= 0:
        return

    # Quantile-stratified subsampling (preserve tails)
    if y.size > target:
        y_sorted = np.sort(y)

        qs = np.linspace(0.0, 1.0, target, endpoint=False)
        idx = (qs * y_sorted.size).astype(int)

        bin_width = max(1, y_sorted.size // target)
        idx += rng.integers(0, bin_width, size=target)
        idx = np.clip(idx, 0, y_sorted.size - 1)

        y_plot = y_sorted[idx]
    else:
        y_plot = y

    # Horizontal jitter
    jitter = (rng.random(y_plot.size) - 0.5) * (width * 0.55)

    ax.scatter(
        np.full(y_plot.size, x) + jitter,
        y_plot,
        s=30,
        c="k",
        alpha=1.0,
        linewidths=0.0,
        zorder=int(zorder),
    )


# ============================================================
# ONE canonical violin_group (removes duplicate definitions)
# ============================================================

def violin_group(
    ax,
    groups: List[np.ndarray],
    positions: np.ndarray,
    *,
    width: float,
    show_points: bool,
    point_seed: int,
    lw: float,
    max_points: int,
    point_size: float,
    point_alpha: float = 0.12,
    points_per_violin: int = 36,   # NEW: per-violin control (30–40 sweet spot)
) -> None:
    """
    Draw violins plus (median + IQR) overlay, per group.
    Optionally overlay jittered sample points (subsampled) behind violins.
    """
    # ---- background points FIRST (behind violins) ----
    if show_points:
        for i, (x0, g) in enumerate(zip(positions, groups)):
            _jittered_scatter(
                ax,
                np.asarray(g, dtype=float),
                float(x0),
                float(width),
                seed=int(point_seed + 97 * i),
                n_samples=int(points_per_violin),     # preferred
                max_points=int(max_points) if max_points is not None else None,  # cap
                alpha=float(point_alpha),
                size=float(point_size),
                zorder=1,
            )

    # ---- violins SECOND ----
    parts = ax.violinplot(
        groups,
        positions=positions,
        widths=width,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )

    for body in parts["bodies"]:
        body.set_alpha(0.55)
        body.set_linewidth(1.0)
        body.set_edgecolor("black")
        body.set_zorder(2)

    # ---- median + IQR LAST (top layer) ----
    cap = 0.12 * width
    for x0, g in zip(positions, groups):
        y = np.asarray(g, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        q1, med, q3 = np.percentile(y, [25, 50, 75]) if y.size >= 3 else (y.min(), float(np.median(y)), y.max())
        ax.plot([x0, x0], [q1, q3], lw=lw, color="k", zorder=3)
        ax.plot([x0 - cap, x0 + cap], [med, med], lw=lw, color="k", zorder=3)


def make_two_figures(
    *,
    outdir_path,
    tag: str,
    bin_id: np.ndarray,
    mids: np.ndarray,
    counts: np.ndarray,
    y2: np.ndarray,
    L2: np.ndarray,
    U2: np.ndarray,
    W: np.ndarray,
    t: np.ndarray,
    dpi: int,
    show_points: bool,
    lw: float,
    width: float,
    max_points: int,
    point_size: float,
    xlabel_prefix: str,
    label_mode: str,  # "float" or "int"
    points_per_violin: int = 36,   # NEW: global knob for both figures
) -> None:
    """
    Create the two output figures:
      (1) y2 / L2 / U2 violins
      (2) W / t violins

    Tick labels include per-bin sample counts: 'label\\n(n=COUNT)'.
    """
    B = int(mids.size)
    pos = np.arange(1, B + 1, dtype=float)

    if label_mode == "int":
        base = [str(int(round(v))) for v in mids]
    else:
        base = [f"{v:.3f}" for v in mids]

    xlabels = [f"{b}\n(n={c})" for b, c in zip(base, counts)]

    y2_bins = [y2[bin_id == b] for b in range(B)]
    L2_bins = [L2[bin_id == b] for b in range(B)]
    U2_bins = [U2[bin_id == b] for b in range(B)]
    W_bins = [W[bin_id == b] for b in range(B)]
    t_bins = [t[bin_id == b] for b in range(B)]

    # Figure 1
    fig = plt.figure(figsize=(10, 10), dpi=dpi)
    ax_y2, ax_L2, ax_U2 = fig.subplots(3, 1, sharex=True)

    violin_group(
        ax_y2, y2_bins, pos,
        width=width,
        show_points=show_points,
        point_seed=7,
        lw=lw,
        max_points=max_points,
        point_size=point_size,
        point_alpha=0.12,
        points_per_violin=points_per_violin,
    )
    ax_y2.set_ylabel(r"$y_2(n)$")
    ax_y2.set_title(rf"Sensitivity to perturbations in y(z): bins vs (y, L, U)")

    violin_group(
        ax_L2, L2_bins, pos,
        width=width,
        show_points=show_points,
        point_seed=11,
        lw=lw,
        max_points=max_points,
        point_size=point_size,
        point_alpha=0.12,
        points_per_violin=points_per_violin,
    )
    ax_L2.set_ylabel(r"$L_2(n)$")

    violin_group(
        ax_U2, U2_bins, pos,
        width=width,
        show_points=show_points,
        point_seed=13,
        lw=lw,
        max_points=max_points,
        point_size=point_size,
        point_alpha=0.12,
        points_per_violin=points_per_violin,
    )
    ax_U2.set_ylabel(r"$U_2(n)$")
    ax_U2.set_xlabel(f"{xlabel_prefix} bin (midpoint)\n(labels include per-bin counts)")
    ax_U2.set_xticks(pos)
    ax_U2.set_xticklabels(xlabels, rotation=25, ha="right")

    for ax in (ax_y2, ax_L2, ax_U2):
        ax.grid(True, axis="y", alpha=0.12)

    fig.tight_layout()
    fig.savefig(outdir_path / f"y2_L2_U2_violins{tag}.png")
    plt.close(fig)

    # Figure 2
    fig = plt.figure(figsize=(10, 8), dpi=dpi)
    ax_W, ax_t = fig.subplots(2, 1, sharex=True)

    violin_group(
        ax_W, W_bins, pos,
        width=width,
        show_points=show_points,
        point_seed=17,
        lw=lw,
        max_points=max_points,
        point_size=point_size,
        point_alpha=0.12,
        points_per_violin=points_per_violin,
    )
    ax_W.set_ylabel(r"$W(n)$")
    ax_W.set_title(rf"Envelope width and normalized position")

    violin_group(
        ax_t, t_bins, pos,
        width=width,
        show_points=show_points,
        point_seed=19,
        lw=lw,
        max_points=max_points,
        point_size=point_size,
        point_alpha=0.12,
        points_per_violin=points_per_violin,
    )
    ax_t.set_ylabel(r"$t(n)$")
    ax_t.set_xlabel(f"{xlabel_prefix} bin (midpoint)\n(labels include per-bin counts)")
    ax_t.set_xticks(pos)
    ax_t.set_xticklabels(xlabels, rotation=25, ha="right")
    ax_t.set_ylim(0.6, 1.0)

    for ax in (ax_W, ax_t):
        ax.grid(True, axis="y", alpha=0.12)

    fig.tight_layout()
    fig.savefig(outdir_path / f"W_t_violins{tag}.png")
    plt.close(fig)
