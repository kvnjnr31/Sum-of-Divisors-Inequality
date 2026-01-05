# plot_sigma_over_2n_real_and_gaussian_polished.py
#
# Purpose:
#   Simple visual comparison:
#
#     Left:  n  vs  |σ(n)/(2n)|
#     Right: (Re z, Im z) heatmap of |σ(z)/(2z)|
#
# Final refinements in this patch:
#   1) Overlay WHITE points where |σ(n)/(2n)| ≈ 1 in the LEFT panel
#   2) Ensure overlays do NOT obscure axis lines (z-order control)
#   3) Keep the Gaussian plane exactly as before (linear colorbar 0→2)
#
# Notes:
#   - White markers are drawn *below* reference lines and spines.
#   - ONE_TOL controls how tightly “= 1” is interpreted.

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, FuncFormatter
from matplotlib.colors import Normalize
from pathlib import Path

def k_formatter(x, pos):
    if x >= 1000:
        return f"{int(x/1000)}k"
    return f"{int(x)}"


# ----------------------------
# Real integers: exact sigma
# ----------------------------
def sigma_sieve(N: int) -> np.ndarray:
    sig = np.zeros(N + 1, dtype=np.int64)
    for d in range(1, N + 1):
        sig[d::d] += d
    return sig


# ----------------------------
# Gaussian helpers (your code)
# ----------------------------
from gaussian_validate_helpers import (
    gaussian_primes_upto_norm,
    stats_for_z,
)


def main() -> None:
    # ----------------------------
    # Parameters
    # ----------------------------
    N_real = 10_000
    box = 120
    ONE_TOL = 0.00001 #0.0001

    outdir = Path("outputs")
    outdir.mkdir(exist_ok=True)

    # ----------------------------
    # Real case
    # ----------------------------
    sig = sigma_sieve(N_real)
    n = np.arange(1, N_real + 1, dtype=np.float64)
    y_real = np.abs(sig[1:] / (2.0 * n))

    # Mask where |σ(n)/(2n)| ≈ 1
    one_mask_real = np.isclose(y_real, 1.0, rtol=ONE_TOL)

    # ----------------------------
    # Gaussian case
    # ----------------------------
    max_norm = 2 * box * box
    gprimes = gaussian_primes_upto_norm(max_norm)

    xs, ys, vals = [], [], []

    for a in range(-box, box + 1):
        for b in range(-box, box + 1):
            if a == 0 and b == 0:
                continue
            y, *_ = stats_for_z((a, b), gprimes)
            vals.append(np.exp(y) / 2.0)
            xs.append(a)
            ys.append(b)

    xs = np.asarray(xs)
    ys = np.asarray(ys)
    vals = np.asarray(vals)

    # ----------------------------
    # Plot styling
    # ----------------------------
    plt.rcParams.update({
        "font.size": 14,
        "axes.linewidth": 2.0,
    })

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(14, 6), dpi=250
    )

    # ============================================================
    # Left panel: Real integers
    # ============================================================
    ax1.scatter(
        n,
        y_real,
        s=12,
        facecolors="#2FA4A9",
        edgecolors="#2FA4A9",
        alpha=1,
        zorder=1,
    )

    # Stat markers where |σ(n)/(2n)| ≈ 1 (below axis line)
    ax1.scatter(
        n[one_mask_real],
        y_real[one_mask_real],
        s=40,
        facecolors="#C1121F",
        edgecolors="k",
        linewidths=1.5,
        zorder=2,
    )

    # Reference line drawn ABOVE markers
    # ax1.axhline(1.0, color="k", lw=2, zorder=4)

    ax1.set_xlabel(r"$n$")
    ax1.set_ylabel(r"$|\sigma(n)/(2n)|$")
    ax1.set_title("Real integers")

    ax1.xaxis.set_major_locator(MaxNLocator(6))
    ax1.xaxis.set_major_formatter(FuncFormatter(k_formatter))
    ax1.yaxis.set_major_locator(MaxNLocator(6))

    ax1.set_ylim(0.45, np.percentile(y_real, 99.5))

    # ============================================================
    # Right panel: Gaussian integers (linear color scale)
    # ============================================================
    vals_clipped = np.clip(vals, 0.0, 2.0)

    sc = ax2.scatter(
        xs,
        ys,
        c=vals_clipped,
        s=6,
        cmap="viridis",
        norm=Normalize(vmin=0.0, vmax=2.0),
        linewidths=0.0,
        zorder=1,
    )

    # Stat overlay where |σ(z)/(2z)| ≈ 1 (below axes)
    one_mask_gauss = np.isclose(vals, 1.0, rtol=ONE_TOL)

    ax2.scatter(
        xs[one_mask_gauss],
        ys[one_mask_gauss],
        s=40,
        facecolors="#C1121F",
        edgecolors="k",
        linewidths=1.5,
        zorder=2,
    )

    ax2.set_aspect("equal", adjustable="box")
    ax2.set_xlabel(r"$\mathrm{Re}(z)$")
    ax2.set_ylabel(r"$\mathrm{Im}(z)$")
    ax2.set_title("Gaussian integers")

    ax2.set_xlim(xs.min(), xs.max())
    ax2.set_ylim(ys.min(), ys.max())
    ax2.margins(0)

    # Colorbar (linear 0 → 2)
    cbar = fig.colorbar(sc, ax=ax2, pad=0.03)
    cbar.set_label(r"$|\sigma(z)/(2z)|$")
    ticks = np.arange(0.0, 2.01, 0.25)
    cbar.set_ticks(ticks)
    cbar.set_ticklabels([f"{t:.2g}" for t in ticks])

    from matplotlib.lines import Line2D

    legend_elements = [
        Line2D(
            [0], [0],
            marker='o',
            color='none',
            label=r'$|\sigma(\cdot)/(2\cdot)| = 1$',
            markerfacecolor="#C1121F",
            markeredgecolor='k',
            markersize=9,
        )
    ]

    ax2.legend(
        handles=legend_elements,
        loc="upper right",
        frameon=True,
        framealpha=0.95,
    )


    # ----------------------------
    # Save
    # ----------------------------
    fig.tight_layout()
    outpath = outdir / "sigma_over_2n_real_vs_gaussian_polished.png"
    fig.savefig(outpath, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)

    print(f"Wrote: {outpath.resolve()}")


if __name__ == "__main__":
    main()
