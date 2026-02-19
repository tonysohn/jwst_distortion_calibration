"""
JWST Distortion Plotting Module (v13: Shorter Colorbars)
Updates:
- Reduced colorbar height (shrink=0.6) to avoid overlap with titles.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.stats import sigma_clip
from matplotlib.gridspec import GridSpec


def bin_vectors(x, y, dx, dy, grid_n=20):
    """Bins residuals to mean star position."""
    x_edges = np.linspace(0, 2048, grid_n + 1)
    y_edges = np.linspace(0, 2048, grid_n + 1)
    bx_pos, by_pos, bdx, bdy = [], [], [], []
    x_bin_idx = np.digitize(x, x_edges) - 1
    y_bin_idx = np.digitize(y, y_edges) - 1

    for i in range(grid_n):
        for j in range(grid_n):
            in_bin = (x_bin_idx == j) & (y_bin_idx == i)
            if np.sum(in_bin) > 3:
                d_x = x[in_bin]
                d_y = y[in_bin]
                d_dx = dx[in_bin]
                d_dy = dy[in_bin]
                clean_dx = sigma_clip(d_dx, sigma=2.5, maxiters=1)
                clean_dy = sigma_clip(d_dy, sigma=2.5, maxiters=1)
                if np.ma.count(clean_dx) > 0:
                    mask = ~clean_dx.mask & ~clean_dy.mask
                    if np.sum(mask) > 0:
                        bx_pos.append(np.mean(d_x[mask]))
                        by_pos.append(np.mean(d_y[mask]))
                        bdx.append(np.mean(clean_dx[mask]))
                        bdy.append(np.mean(clean_dy[mask]))
    return np.array(bx_pos), np.array(by_pos), np.array(bdx), np.array(bdy)


def plot_residuals(results, output_dir, label):
    rx = results["residuals_x_mas"]
    ry = results["residuals_y_mas"]
    max_val = max(np.max(np.abs(rx)), np.max(np.abs(ry)))
    limit = max(max_val * 1.5, 5.0)
    fig = plt.figure(figsize=(10, 10))
    gs = GridSpec(
        2, 2, width_ratios=[4, 1], height_ratios=[1, 4], wspace=0.05, hspace=0.05
    )
    ax_main = fig.add_subplot(gs[1, 0])
    ax_histx = fig.add_subplot(gs[0, 0], sharex=ax_main)
    ax_histy = fig.add_subplot(gs[1, 1], sharey=ax_main)
    ax_main.scatter(rx, ry, s=5, alpha=0.3, c="k", edgecolors="none")
    ax_main.axhline(0, color="r", linestyle="-", alpha=0.5, linewidth=1)
    ax_main.axvline(0, color="r", linestyle="-", alpha=0.5, linewidth=1)
    ax_main.set_xlim(-limit, limit)
    ax_main.set_ylim(-limit, limit)
    ax_main.set_aspect("equal")
    ax_main.set_xlabel(r"$\Delta x$ (mas)", fontsize=12)
    ax_main.set_ylabel(r"$\Delta y$ (mas)", fontsize=12)
    ax_main.grid(True, alpha=0.2, linestyle=":")
    ax_histx.hist(
        rx, bins=100, range=(-limit, limit), color="gray", alpha=0.7, density=True
    )
    ax_histx.tick_params(axis="x", which="both", bottom=True, labelbottom=False)
    ax_histx.tick_params(axis="y", which="both", left=False, labelleft=False)
    stats = f"RMS X: {results['rms_x']:.2f} mas\nRMS Y: {results['rms_y']:.2f} mas\nN Stars: {results['n_stars']}"
    ax_histx.text(
        0.02, 0.9, stats, transform=ax_histx.transAxes, va="top", ha="left", fontsize=10
    )
    ax_histy.hist(
        ry,
        bins=100,
        range=(-limit, limit),
        orientation="horizontal",
        color="gray",
        alpha=0.7,
        density=True,
    )
    ax_histy.tick_params(axis="y", which="both", left=True, labelleft=False)
    ax_histy.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
    plt.savefig(os.path.join(output_dir, f"{label}_residuals.pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_trends(results, output_dir, label):
    rx = results["residuals_x_mas"]
    ry = results["residuals_y_mas"]
    x = results["x_sci_used"]
    y = results["y_sci_used"]
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(15, 6))
    axs[0, 0].plot(x, rx, "k.", ms=4)
    axs[0, 0].set(ylabel=r"$\Delta x$ (mas)")
    axs[0, 0].axhline(0, c="r", lw=2)
    axs[1, 0].plot(x, ry, "k.", ms=4)
    axs[1, 0].set(ylabel=r"$\Delta y$ (mas)")
    axs[1, 0].set(xlabel=r"$x_{idl}$ (arcsec)")
    axs[1, 0].axhline(0, c="r", lw=2)
    axs[0, 1].plot(y, rx, "k.", ms=4)
    axs[0, 1].axhline(0, c="r", lw=2)
    axs[1, 1].plot(y, ry, "k.", ms=4)
    axs[1, 1].set(xlabel="$y_{idl}$ (arcsec)")
    axs[1, 1].axhline(0, c="r", lw=2)
    fig.tight_layout()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{label}_trends.pdf"), bbox_inches="tight")
    plt.close(fig)


def get_dynamic_scale(gdx, gdy, target_visual_len=200.0):
    """Calculates a clean scale multiplier so the longest vector takes up ~10% of the plot."""
    max_disp = np.max(np.sqrt(gdx**2 + gdy**2))
    if max_disp < 1e-6:
        return 5000.0, 0.05, "0.05 pix (5000x)"

    # Calculate ideal multiplier
    ideal_mult = target_visual_len / max_disp

    # Snap to a clean multiplier (1, 2, 5, 10, 50, etc.)
    mults = np.array(
        [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
    )
    mult = mults[np.argmin(np.abs(np.log10(mults) - np.log10(ideal_mult)))]

    # Snap the reference legend arrow length to a clean decimal
    ref_vals = np.array([0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0])
    ideal_ref = 100.0 / mult
    ref_val = ref_vals[np.argmin(np.abs(np.log10(ref_vals) - np.log10(ideal_ref)))]

    return mult, ref_val, f"{ref_val:g} pix ({int(mult)}x)"


def plot_comparison_models(gx, gy, dx1, dy1, dx2, dy2, output_dir, label):
    """
    Plots smooth polynomial models on a grid.
    Left: Before | Right: After (with dynamic scaling)
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    def plot_panel(ax, gdx, gdy, title):
        mult, ref_val, label_txt = get_dynamic_scale(gdx, gdy)

        grid_lines = np.linspace(0, 2048, 21)
        for g in grid_lines:
            ax.axhline(g, color="gray", linestyle="-", alpha=0.1)
            ax.axvline(g, color="gray", linestyle="-", alpha=0.1)

        mag_mas = np.sqrt(gdx**2 + gdy**2) * 1000.0

        q = ax.quiver(
            gx,
            gy,
            gdx,
            gdy,
            mag_mas,
            angles="xy",
            scale_units="xy",
            scale=1.0 / mult,
            cmap="viridis",
            width=0.004,
            headlength=0,
            headaxislength=0,
            pivot="tail",
        )

        ax.quiverkey(q, 0.9, 1.02, ref_val, label_txt, labelpos="E")
        ax.set_xlim(-50, 2098)
        ax.set_ylim(-50, 2098)
        ax.set_aspect("equal")
        ax.set_title(f"{title}\n(Scale: {int(mult)}x)", fontsize=16)
        ax.set_xlabel("X (SCI pixels)", fontsize=12)
        ax.set_ylabel("Y (SCI pixels)", fontsize=12)
        return q

    q1 = plot_panel(axes[0], dx1, dy1, "Distortion Model (Before)")
    q2 = plot_panel(axes[1], dx2, dy2, "Residual Model (After)")

    cbar1 = fig.colorbar(q1, ax=axes[0], pad=0.02, shrink=0.76)
    cbar1.set_label("Magnitude (mas)", fontsize=12)

    cbar2 = fig.colorbar(q2, ax=axes[1], pad=0.02, shrink=0.76)
    cbar2.set_label("Magnitude (mas)", fontsize=12)

    out_file = os.path.join(output_dir, f"{label}_model_comparison.png")
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved model comparison: {out_file}")
