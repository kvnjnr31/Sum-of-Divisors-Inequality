# gaussian_validate_violin_helpers.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Core Gaussian helper (your existing math, kept intact)
#   - primes up to norm
#   - factorization in Z[i]
#   - y(z), L(z), U(z), W(z), t(z), omega(z)
# ============================================================

def sieve_is_prime(limit: int) -> np.ndarray:
    """Boolean sieve is_prime[0..limit]."""
    if limit < 1:
        return np.zeros(limit + 1, dtype=bool)
    is_prime = np.ones(limit + 1, dtype=bool)
    is_prime[:2] = False
    r = int(math.isqrt(limit))
    for p in range(2, r + 1):
        if is_prime[p]:
            is_prime[p * p : limit + 1 : p] = False
    return is_prime


def norm(a: int, b: int) -> int:
    return a * a + b * b


def abs_val(a: int, b: int) -> float:
    return math.sqrt(a * a + b * b)


def associates(a: int, b: int) -> List[Tuple[int, int]]:
    # {z, -z, iz, -iz}
    return [(a, b), (-a, -b), (-b, a), (b, -a)]


def div_exact(z: Tuple[int, int], p: Tuple[int, int]) -> Tuple[bool, Tuple[int, int]]:
    """Return (True, q) if q=z/p is in Z[i], else (False,(0,0))."""
    a, b = z
    c, d = p
    n = c * c + d * d
    x = a * c + b * d
    y = b * c - a * d
    if x % n != 0 or y % n != 0:
        return False, (0, 0)
    return True, (x // n, y // n)


def is_gaussian_prime(a: int, b: int, is_prime: np.ndarray) -> bool:
    """Gaussian prime test using standard characterization."""
    if a == 0 and b == 0:
        return False

    # units are not prime
    if (abs(a) == 1 and b == 0) or (abs(b) == 1 and a == 0):
        return False

    aa, bb = abs(a), abs(b)
    if aa == 0 or bb == 0:
        p = aa if bb == 0 else bb
        return p < len(is_prime) and bool(is_prime[p]) and (p % 4 == 3)

    n = aa * aa + bb * bb
    return n < len(is_prime) and bool(is_prime[n])


def gaussian_primes_upto_norm(max_norm: int) -> List[Tuple[int, int]]:
    """Return Gaussian primes with a>=0, b>=0 and norm <= max_norm."""
    is_prime = sieve_is_prime(max_norm)
    primes: List[Tuple[int, int]] = []

    r = int(math.isqrt(max_norm))

    # axis primes
    for p in range(2, r + 1):
        if is_prime[p] and (p % 4 == 3):
            primes.append((p, 0))
            primes.append((0, p))

    # interior primes
    for a in range(1, r + 1):
        for b in range(1, r + 1):
            if a * a + b * b > max_norm:
                break
            if is_gaussian_prime(a, b, is_prime):
                primes.append((a, b))

    primes.sort(key=lambda t: (t[0] * t[0] + t[1] * t[1], t[0], t[1]))
    return primes


@dataclass(frozen=True)
class Factorization:
    exponents: Dict[Tuple[int, int], int]


def canonical_rep(a: int, b: int) -> Tuple[int, int]:
    """Pick a canonical associate: a>0, b>=0; if a==0 then b>0."""
    cands = []
    for x, y in associates(a, b):
        if x > 0 and y >= 0:
            cands.append((x, y))
        elif x == 0 and y > 0:
            cands.append((x, y))
    if not cands:
        return (abs(a), abs(b))
    return min(cands, key=lambda t: (t[0] * t[0] + t[1] * t[1], t[0], t[1]))


def factor_gaussian(z: Tuple[int, int], primes: List[Tuple[int, int]]) -> Factorization:
    """Factor z in Z[i] using trial division by a precomputed prime list."""
    a0, b0 = z
    if a0 == 0 and b0 == 0:
        raise ValueError("cannot factor 0")

    a, b = a0, b0
    exps: Dict[Tuple[int, int], int] = {}

    cur_norm = a * a + b * b

    for base in primes:
        pn = base[0] * base[0] + base[1] * base[1]
        if pn == 0:
            continue
        if pn > cur_norm:
            break

        for p in associates(base[0], base[1]):
            cnt = 0
            while True:
                ok, q = div_exact((a, b), p)
                if not ok:
                    break
                a, b = q
                cnt += 1
                cur_norm = a * a + b * b
                if cur_norm == 1:
                    break

            if cnt > 0:
                rep = canonical_rep(p[0], p[1])
                exps[rep] = exps.get(rep, 0) + cnt

            if cur_norm == 1:
                break

        if cur_norm == 1:
            break

    if not (a * a + b * b == 1):
        rep = canonical_rep(a, b)
        exps[rep] = exps.get(rep, 0) + 1

    return Factorization(exponents=exps)


def stats_for_z(z: Tuple[int, int], primes: List[Tuple[int, int]]) -> Tuple[float, float, float, float, float, int]:
    """Return (y, L, U, W, t, omega) for a Gaussian integer z.

    Convention:
        y = log|σ(z)/(2z)|   (matches the real-case |σ(n)/(2n)|)
    """
    fac = factor_gaussian(z, primes)
    if len(fac.exponents) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0

    y = 0.0
    s = 0.0
    corr = 0.0

    for (a, b), alpha in fac.exponents.items():
        pi_abs = abs_val(a, b)
        pi = complex(a, b)

        # y(z)=sum log|pi/(pi-1)| + sum log|1 - pi^{-(alpha+1)}|
        # This equals log|σ(z)/z|, so we subtract log 2 below to get log|σ(z)/(2z)|.
        y += math.log(abs(pi / (pi - 1.0)))
        y += math.log(abs(1.0 - pi ** (-(alpha + 1))))

        s += 1.0 / (pi_abs - 1.0)
        corr += pi_abs ** (-(alpha + 1))

    # Envelope bounds for log|σ(z)/z|
    L = -s
    U = s + corr

    # -----------------------------------------
    # Apply the SAME -log(2) correction to y, L, U
    # so the envelope matches σ(z)/(2z).
    # -----------------------------------------
    LOG2 = math.log(2.0)
    y -= LOG2
    L -= LOG2
    U -= LOG2

    W = max(U - L, 1e-16)
    t = min(max((y - L) / W, 0.0), 1.0)
    omega = len(fac.exponents)
    return y, L, U, W, t, omega

# ============================================================
# Sampling + binning
# ============================================================

def sample_gaussian_integers(box: int, nsamples: int, seed: int) -> np.ndarray:
    """Uniformly sample nsamples points (a,b) with a,b in [-box,box], excluding 0."""
    rng = np.random.default_rng(seed)
    out = np.empty((nsamples, 2), dtype=np.int32)
    k = 0
    while k < nsamples:
        a = int(rng.integers(-box, box + 1))
        b = int(rng.integers(-box, box + 1))
        if a == 0 and b == 0:
            continue
        out[k, 0] = a
        out[k, 1] = b
        k += 1
    return out


def quantile_bin_indices(values: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """Quantile bins for a continuous array; returns (bin_id, midpoints)."""
    q = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(values, q)
    edges = np.unique(edges)
    if edges.size < 3:
        bin_id = np.zeros(values.size, dtype=int)
        mids = np.array([float(np.median(values))], dtype=float)
        return bin_id, mids
    bin_id = np.digitize(values, edges[1:-1], right=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return bin_id, mids


def quantile_bins_on_integers(k: np.ndarray, nbins: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantile bins for integer-valued k (like omega).
    We quantile on k but label bins by midpoint of the k-edges.
    """
    kf = k.astype(float)
    q = np.linspace(0.0, 1.0, nbins + 1)
    edges = np.quantile(kf, q)
    edges = np.unique(edges)
    if edges.size < 3:
        bin_id = np.zeros(k.size, dtype=int)
        mids = np.array([float(np.median(kf))], dtype=float)
        return bin_id, mids
    bin_id = np.digitize(kf, edges[1:-1], right=True)
    mids = 0.5 * (edges[:-1] + edges[1:])
    return bin_id, mids


def bin_counts(bin_id: np.ndarray, nbins: int) -> List[int]:
    return [int(np.sum(bin_id == b)) for b in range(nbins)]


def groups_from_bins(x: np.ndarray, bin_id: np.ndarray, nbins: int) -> List[np.ndarray]:
    return [x[bin_id == b] for b in range(nbins)]


# ============================================================
# Violin plotting utilities (points BEHIND violins)
# ============================================================

def _jittered_scatter(
    ax: plt.Axes,
    y: np.ndarray,
    x0: float,
    width: float,
    *,
    seed: int,
    max_points: int | None = None,   # backward-compatible
    n_samples: int | None = None,    # preferred
    alpha: float,
    size: float,
    zorder: int,
) -> None:
    """
    Jittered scatter overlay for violin plots with distribution-aware
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

    # Resolve sample count
    if n_samples is not None:
        target = int(n_samples)
    elif max_points is not None:
        target = int(max_points)
    else:
        target = int(y.size)

    if target <= 0:
        return

    # Quantile-stratified subsampling
    if y.size > target:
        y_sorted = np.sort(y)

        qs = np.linspace(0.0, 1.0, target, endpoint=False)
        idx = (qs * y_sorted.size).astype(int)

        # Small random jitter inside each quantile bin
        bin_width = max(1, y_sorted.size // target)
        idx += rng.integers(0, bin_width, size=target)
        idx = np.clip(idx, 0, y_sorted.size - 1)

        y_plot = y_sorted[idx]
    else:
        y_plot = y

    # Horizontal jitter
    jitter_x = (rng.random(y_plot.size) - 0.5) * (width * 0.70)
    x = np.full(y_plot.size, x0, dtype=float) + jitter_x

    ax.scatter(
        x,
        y_plot,
        s=20,
        alpha=1.0,
        facecolors="k",
        linewidths=0.0,
        color="0.35",   # neutral grey
        zorder=zorder,
    )


def violin_group(
    ax: plt.Axes,
    groups: List[np.ndarray],
    positions: np.ndarray,
    *,
    width: float,
    show_points: bool,
    point_seed: int,
    max_points: int,
    point_alpha: float,
    point_size: float,
    lw: float,
    points_per_violin: int = 36,
) -> None:
    """
    Draw:
      - optional jitter points (behind)
      - violins
      - median + IQR overlay
    """
    # points first, behind
    if show_points:
        for x0, g in zip(positions, groups):
            _jittered_scatter(
                ax,
                g,
                float(x0),
                width,
                seed=point_seed + int(97 * x0),
                # PATCH: expose per-violin control, while keeping max_points as a cap
                n_samples=int(points_per_violin),
                max_points=int(max_points) if max_points is not None else None,
                alpha=point_alpha,
                size=point_size,
                zorder=1,
            )

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
        body.set_linewidth(lw)
        body.set_zorder(2)

    cap = 0.14 * width
    for x0, g in zip(positions, groups):
        y = np.asarray(g, dtype=float)
        y = y[np.isfinite(y)]
        if y.size == 0:
            continue

        if y.size >= 3:
            q1, med, q3 = np.percentile(y, [25, 50, 75])
        else:
            q1, med, q3 = float(np.min(y)), float(np.median(y)), float(np.max(y))

        ax.plot([x0, x0], [q1, q3], lw=lw, color="k", zorder=3)
        ax.plot([x0 - cap, x0 + cap], [med, med], lw=lw, color="k", zorder=3)


# ============================================================
# One-stop figure maker (mirrors the real-case deliverable)
# ============================================================

def make_two_figures(
    * ,
    outdir_path,
    y: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    W: np.ndarray,
    t: np.ndarray,
    mids: np.ndarray,
    counts: List[int],
    fig_tag: str,
    xaxis_label: str,
    title_prefix: str,
    dpi: int,
    show_points: bool,
    max_points: int,
    point_alpha: float,
    point_size: float,
    lw: float,
    font_size: int,
    t_ylim: Tuple[float, float],
) -> None:
    outdir_path.mkdir(parents=True, exist_ok=True)

    B = mids.size
    pos = np.arange(1, B + 1, dtype=float)

    y_bins = groups_from_bins(y, np.digitize(y, [-np.inf]), 1)  # dummy to satisfy type checkers
    raise RuntimeError("make_two_figures() is not meant to be called directly; use make_two_figures_from_groups().")


def make_two_figures_from_groups(
    *,
    outdir_path,
    y_bins: List[np.ndarray],
    L_bins: List[np.ndarray],
    U_bins: List[np.ndarray],
    W_bins: List[np.ndarray],
    t_bins: List[np.ndarray],
    mids: np.ndarray,
    counts: List[int],
    fig_tag: str,
    xaxis_label: str,
    title_prefix: str,
    dpi: int,
    show_points: bool,
    max_points: int,
    point_alpha: float,
    point_size: float,
    lw: float,
    font_size: int,
    t_ylim: Tuple[float, float],
    points_per_violin: int = 36,   # PATCH: global knob
) -> None:
    outdir_path.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update({
        "font.size": font_size,
        "axes.titlesize": font_size + 2,
        "axes.labelsize": font_size + 1,
        "xtick.labelsize": font_size,
        "ytick.labelsize": font_size,
        "axes.linewidth": lw,
    })

    B = mids.size
    pos = np.arange(1, B + 1, dtype=float)

    xlabels = [f"{m:.3f}\n(n={c})" for m, c in zip(mids, counts)]

    # -------------------------
    # FIG 1: y, L, U (3 panels)
    # -------------------------
    fig = plt.figure(figsize=(9, 9), dpi=dpi)
    ax_y, ax_L, ax_U = fig.subplots(3, 1, sharex=True)

    violin_group(
        ax_y, y_bins, pos,
        width=0.85,
        show_points=show_points,
        point_seed=7,
        max_points=max_points,
        point_alpha=point_alpha,
        point_size=point_size,
        lw=lw,
        points_per_violin=points_per_violin,  # PATCH
    )
    ax_y.set_title(f"{title_prefix}: bins vs (y, L, U)")
    ax_y.set_ylabel(r"$y(z)$")

    violin_group(
        ax_L, L_bins, pos,
        width=0.85,
        show_points=show_points,
        point_seed=11,
        max_points=max_points,
        point_alpha=point_alpha,
        point_size=point_size,
        lw=lw,
        points_per_violin=points_per_violin,  # PATCH
    )
    ax_L.set_ylabel(r"$L(z)$")

    violin_group(
        ax_U, U_bins, pos,
        width=0.85,
        show_points=show_points,
        point_seed=13,
        max_points=max_points,
        point_alpha=point_alpha,
        point_size=point_size,
        lw=lw,
        points_per_violin=points_per_violin,  # PATCH
    )
    ax_U.set_ylabel(r"$U(z)$")
    ax_U.set_xlabel(f"{xaxis_label}\n(labels include per-bin counts)")
    ax_U.set_xticks(pos)
    ax_U.set_xticklabels(xlabels, rotation=25, ha="right")

    fig.tight_layout()
    fig.savefig(outdir_path / f"gaussian_{fig_tag}_y_L_U_violins.png")
    plt.close(fig)

    # -------------------------
    # FIG 2: W, t (2 panels)
    # -------------------------
    fig = plt.figure(figsize=(9, 7.2), dpi=dpi)
    ax_W, ax_t = fig.subplots(2, 1, sharex=True)

    violin_group(
        ax_W, W_bins, pos,
        width=0.85,
        show_points=show_points,
        point_seed=17,
        max_points=max_points,
        point_alpha=point_alpha,
        point_size=point_size,
        lw=lw,
        points_per_violin=points_per_violin,  # PATCH
    )
    ax_W.set_title("Envelope width and normalized position")
    ax_W.set_ylabel(r"$W(z)$")

    violin_group(
        ax_t, t_bins, pos,
        width=0.85,
        show_points=show_points,
        point_seed=19,
        max_points=max_points,
        point_alpha=point_alpha,
        point_size=point_size,
        lw=lw,
        points_per_violin=points_per_violin,  # PATCH
    )
    ax_t.set_ylabel(r"$t(z)$")
    ax_t.set_xlabel(f"{xaxis_label}\n(labels include per-bin counts)")
    ax_t.set_xticks(pos)
    ax_t.set_xticklabels(xlabels, rotation=25, ha="right")
    ax_t.set_ylim(float(t_ylim[0]), float(t_ylim[1]))

    fig.tight_layout()
    fig.savefig(outdir_path / f"gaussian_{fig_tag}_W_t_violins.png")
    plt.close(fig)
