"""
JWST Distortion Comparison Tool
Usage:
    python tools/distortion_compare.py path/to/reference.txt path/to/comparison1.txt [path/to/comparison2.txt ...]

Description:
    Compares one or more distortion solutions against a master reference solution.
    Calculates exact spatial errors (mas), local pixel scales, skew changes,
    and generates a 3-panel diagnostic plot (Quiver, Heatmap, Coeff Diff).
"""

import argparse
import os

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import pysiaf
from astropy.io import ascii

# --- Optional Publication-Quality Plotting ---
try:
    import scienceplots

    plt.style.use(["science", "no-latex"])
except ImportError:
    pass


def read_distortion(file_path):
    """Safely reads the master distortion file and extracts metadata."""
    try:
        tab = ascii.read(file_path, format="csv", comment="#")
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

    # Parse header for metadata
    obs_date = "Unknown"
    aper = "Unknown"
    with open(file_path, "r") as f:
        for line in f:
            if "Observation Date:" in line:
                obs_date = line.split(":")[-1].strip()
            elif "Aperture:" in line:
                aper = line.split(":")[-1].strip()

    num_coeffs = len(tab)
    order = pysiaf.utils.polynomial.polynomial_degree(num_coeffs)

    return {
        "file": os.path.basename(file_path),
        "date": obs_date,
        "aper": aper,
        "order": order,
        "table": tab,
        "ex": tab["exponent_x"].data,
        "ey": tab["exponent_y"].data,
        "cx": tab["Sci2IdlX"].data,
        "cy": tab["Sci2IdlY"].data,
    }


def get_linear_terms(dist_dict):
    """Extracts the linear transformation terms (Sci2Idl) to compute scale and skew."""
    tab = dist_dict["table"]
    idx_x = np.where((tab["exponent_x"] == 1) & (tab["exponent_y"] == 0))[0][0]
    idx_y = np.where((tab["exponent_x"] == 0) & (tab["exponent_y"] == 1))[0][0]

    b = tab["Sci2IdlX"][idx_x]
    c = tab["Sci2IdlX"][idx_y]
    e = tab["Sci2IdlY"][idx_x]
    f = tab["Sci2IdlY"][idx_y]

    scale_x = np.sqrt(b**2 + e**2) * 1000  # Convert to mas
    scale_y = np.sqrt(c**2 + f**2) * 1000  # Convert to mas

    angle_x = np.arctan2(e, b)
    angle_y = np.arctan2(f, c)
    skew_arcsec = (
        np.abs(angle_y - angle_x) - (np.pi / 2)
    ) * 206265  # radians to arcsec

    return scale_x, scale_y, skew_arcsec


def compare_solutions(ref, comp, output_dir):
    """Calculates metrics and generates diagnostic plots for a single comparison."""
    # 1. Linear Metrics
    ref_sx, ref_sy, ref_skew = get_linear_terms(ref)
    comp_sx, comp_sy, comp_skew = get_linear_terms(comp)

    delta_sx = comp_sx - ref_sx
    delta_sy = comp_sy - ref_sy
    delta_skew = comp_skew - ref_skew

    # 2. Spatial Grid Evaluation (Sci -> Idl mapping difference)
    nx, ny = (50, 50)
    x = np.linspace(1, 2048, nx)
    y = np.linspace(1, 2048, ny)
    xg, yg = np.meshgrid(x - 1024.5, y - 1024.5)

    xg_ref = pysiaf.utils.polynomial.poly(ref["cx"], xg, yg, order=ref["order"])
    yg_ref = pysiaf.utils.polynomial.poly(ref["cy"], xg, yg, order=ref["order"])

    xg_comp = pysiaf.utils.polynomial.poly(comp["cx"], xg, yg, order=comp["order"])
    yg_comp = pysiaf.utils.polynomial.poly(comp["cy"], xg, yg, order=comp["order"])

    dx_mas = (xg_comp - xg_ref) * 1000.0
    dy_mas = (yg_comp - yg_ref) * 1000.0
    spatial_offset_mas = np.sqrt(dx_mas**2 + dy_mas**2)

    # 3. Scientific Impact Assessment
    max_offset_mas = np.max(spatial_offset_mas)
    max_offset_arcsec = max_offset_mas / 1000.0
    rms_offset_mas = np.sqrt(np.mean(spatial_offset_mas**2))

    # Find the specific pixel where the maximum error occurs
    max_y_idx, max_x_idx = np.unravel_index(
        np.argmax(spatial_offset_mas), spatial_offset_mas.shape
    )
    worst_x = x[max_x_idx]
    worst_y = y[max_y_idx]

    # --- TERMINAL REPORT ---
    print(f"\n{'=' * 70}")
    print(f"COMPARING: {comp['date']} vs REFERENCE: {ref['date']}")
    print(f"{'=' * 70}")
    print(f"[Core Physical Changes]")
    print(f"  Scale X Change (mas) : {delta_sx:+.6f}  (Ref: {ref_sx:.4f})")
    print(f"  Scale Y Change (mas) : {delta_sy:+.6f}  (Ref: {ref_sy:.4f})")
    print(f"  Skew Change (arcsec) : {delta_skew:+.6f}  (Ref: {ref_skew:.4f})")

    print(f"\n[Scientific & Operational Impact]")
    print(f"  Spatial RMS Error    : {rms_offset_mas:.4f} mas")
    print(f"  Worst-Case Location  : Pixel X={worst_x:.0f}, Y={worst_y:.0f}")
    print(f'\n  -> "At most, using the updated solution vs. the old one will result')
    print(
        f'      in an astrometric difference of {max_offset_arcsec:.5f} arcsec ({max_offset_mas:.2f} mas)."'
    )

    # --- PLOTTING ---
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    fig.suptitle(f"Distortion Divergence: {comp['date']} vs {ref['date']}", fontsize=16)

    # Panel 1: Enhanced Quiver Plot
    axes[0].quiver(
        x,
        y,
        dx_mas,
        dy_mas,
        spatial_offset_mas,
        cmap="coolwarm",
        scale_units="xy",
        angles="xy",
    )
    axes[0].set_title("Vector Field of Change", fontsize=14)
    axes[0].set_xlabel("Detector X (pixels)")
    axes[0].set_ylabel("Detector Y (pixels)")
    axes[0].set_xlim(0, 2048)
    axes[0].set_ylim(0, 2048)
    axes[0].set_aspect("equal")

    # Panel 2: Spatial Offset Heatmap
    im = axes[1].imshow(
        spatial_offset_mas, origin="lower", extent=[0, 2048, 0, 2048], cmap="magma"
    )
    axes[1].set_title("Absolute Astrometric Error Impact", fontsize=14)
    axes[1].set_xlabel("Detector X (pixels)")
    axes[1].set_xlim(0, 2048)
    axes[1].set_ylim(0, 2048)
    cbar = fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label("Deviation from Reference (mas)", fontsize=12)

    # Highlight the worst-case pixel on the heatmap
    axes[1].plot(
        worst_x, worst_y, "w*", markersize=12, markeredgecolor="k", label="Max Error"
    )
    axes[1].annotate(
        f"{max_offset_mas:.1f} mas",
        (worst_x, worst_y),
        textcoords="offset points",
        xytext=(10, -10),
        ha="left",
        color="white",
        weight="bold",
        path_effects=[path_effects.withStroke(linewidth=2, foreground="k")],
    )

    # Panel 3: Coefficient Difference
    siaf_indices = ref["table"]["siaf_index"]
    diff_cx = np.abs(comp["cx"] - ref["cx"])
    diff_cy = np.abs(comp["cy"] - ref["cy"])
    diff_cx[diff_cx == 0] = 1e-15
    diff_cy[diff_cy == 0] = 1e-15

    axes[2].plot(
        siaf_indices,
        diff_cx,
        "o-",
        color="steelblue",
        label=r"$|\Delta C_X|$",
        markersize=5,
    )
    axes[2].plot(
        siaf_indices,
        diff_cy,
        "s-",
        color="darkorange",
        label=r"$|\Delta C_Y|$",
        markersize=5,
    )
    axes[2].set_yscale("log")
    axes[2].set_title("Absolute Coefficient Differences", fontsize=14)
    axes[2].set_xlabel("SIAF Polynomial Index")
    axes[2].set_ylabel("Absolute Difference (Log Scale)")
    axes[2].grid(True, alpha=0.3, which="both", linestyle=":")
    axes[2].legend()

    plt.tight_layout()
    out_name = os.path.join(output_dir, f"compare_{comp['date']}_vs_{ref['date']}.png")
    plt.savefig(out_name, dpi=150)
    plt.close(fig)
    print(f"\n-> Saved diagnostic plot to: {out_name}")


def main():
    parser = argparse.ArgumentParser(description="Compare JWST Distortion Solutions")
    parser.add_argument("ref_file", help="Path to the reference distortion .txt file")
    parser.add_argument(
        "comp_files", nargs="+", help="Path(s) to the comparison .txt file(s)"
    )
    args = parser.parse_args()

    ref_data = read_distortion(args.ref_file)
    if not ref_data:
        return

    for comp_path in args.comp_files:
        comp_data = read_distortion(comp_path)
        if comp_data:
            # Automatically determine the exact directory of this specific comparison file
            comp_dir = os.path.dirname(os.path.abspath(comp_path))

            # Pass that directory directly into the plotting function
            compare_solutions(ref_data, comp_data, comp_dir)


if __name__ == "__main__":
    main()
