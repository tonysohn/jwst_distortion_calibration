"""
JWST Distortion Combination Module
Usage:
    python -m calibration.distortion_combine

Description:
    1. Scans for *_distortion_coeffs.txt files across batch subdirectories.
    2. Calculates a sigma-clipped mean (robust average).
    3. Generates a physical spatial stability plot.
    4. Writes a master solution file with standardized naming.
"""

import argparse
import glob
import os

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import ascii, fits
from astropy.stats import sigma_clip

# --- IMPORT CONFIGURATION FROM RUN_CALIBRATION ---
try:
    from . import run_calibration

    DEFAULT_DATA_DIR = run_calibration.DATA_DIR
    BATCH_SUBDIRS = getattr(run_calibration, "BATCH_SUBDIRS", [])
except ImportError:
    DEFAULT_DATA_DIR = "./data"
    BATCH_SUBDIRS = []

FILE_PATTERN = "*_distortion_coeffs.txt"


def get_metadata_from_fits(data_dir):
    """Scans the data directory for a FITS file to extract Instrument, Aperture, and Filter."""
    search_pattern = os.path.join(data_dir, "*.fits")
    files = sorted(glob.glob(search_pattern))

    if not files:
        return "unknown", "unknown", "unknown"

    try:
        with fits.open(files[0]) as hdul:
            header = hdul[0].header
            instr = header.get("INSTRUME", "unknown").strip().lower()
            aper = (
                header.get("APERNAME", header.get("PPS_APER", "unknown"))
                .strip()
                .lower()
            )

            filt_key = header.get("FILTER", "unknown").strip().upper()
            pupil_key = header.get("PUPIL", "unknown").strip().upper()

            filt = pupil_key.lower() if filt_key == "CLEAR" else filt_key.lower()
            return instr, aper, filt
    except Exception:
        return "unknown", "unknown", "unknown"


def read_coefficients(file_list):
    """Reads all coefficient files into a 3D array: [N_Files, N_Coeffs, 4_Columns]"""
    data_cube = []
    meta_data = None
    header = None

    valid_files = []
    for f in file_list:
        try:
            tab = ascii.read(f, format="csv", comment="#")
            row_data = [[row[4], row[5], row[6], row[7]] for row in tab]
            data_cube.append(row_data)
            valid_files.append(f)

            if meta_data is None:
                meta_data = [[row[0], row[1], row[2], row[3]] for row in tab]
                header = tab.colnames
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    return np.array(data_cube), meta_data, header


def compute_robust_mean(data_cube, sigma=2.5):
    """Performs sigma-clipping along the file axis."""
    filtered_data = sigma_clip(
        data_cube, sigma=sigma, axis=0, maxiters=3, cenfunc="median", stdfunc="std"
    )
    robust_mean = np.ma.mean(filtered_data, axis=0)
    robust_std = np.ma.std(filtered_data, axis=0)
    n_surviving = np.ma.count(filtered_data, axis=0)
    std_error = robust_std / np.sqrt(n_surviving)
    return robust_mean, std_error


def plot_stability(data_cube, robust_mean, meta_data, output_dir, label):
    """Generates an improved stability plot using Log-RMS and Spatial Heatmaps."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Panel 1: Log-Scale Coefficient RMS ---
    deviations = data_cube - robust_mean
    rms_per_coeff = np.sqrt(np.mean(deviations**2, axis=0))

    axes[0].semilogy(rms_per_coeff[:, 0], "o-", color="steelblue", label="Sci2IdlX")
    axes[0].semilogy(rms_per_coeff[:, 1], "s-", color="darkorange", label="Sci2IdlY")
    axes[0].set_title("Coefficient Stability (Log Scale)", fontsize=14)
    axes[0].set_xlabel("Coefficient Index (0 to N)", fontsize=12)
    axes[0].set_ylabel("Absolute RMS Scatter", fontsize=12)
    axes[0].grid(True, alpha=0.3, which="both", linestyle=":")
    axes[0].legend()

    # --- Panel 2: 2D Spatial Variance Heatmap ---
    # Evaluate the polynomial for each file to find spatial stability in mas
    grid_1d = np.linspace(0, 2048, 20)
    xg, yg = np.meshgrid(grid_1d, grid_1d)

    N_files = data_cube.shape[0]
    N_coeffs = data_cube.shape[1]

    dx_files, dy_files = [], []

    for f_idx in range(N_files):
        dx, dy = np.zeros_like(xg), np.zeros_like(yg)
        for c_idx in range(N_coeffs):
            ex = int(meta_data[c_idx][2])
            ey = int(meta_data[c_idx][3])
            term = (xg**ex) * (yg**ey)

            dx += data_cube[f_idx, c_idx, 0] * term
            dy += data_cube[f_idx, c_idx, 1] * term

        dx_files.append(dx)
        dy_files.append(dy)

    # Standard deviation across files at each pixel * 1000 (arcsec -> mas)
    spatial_rms_x = np.std(dx_files, axis=0) * 1000.0
    spatial_rms_y = np.std(dy_files, axis=0) * 1000.0
    spatial_rms_total = np.sqrt(spatial_rms_x**2 + spatial_rms_y**2)

    im = axes[1].imshow(
        spatial_rms_total, origin="lower", extent=[0, 2048, 0, 2048], cmap="magma"
    )
    axes[1].set_title("Spatial Stability (Sci2Idl Variation)", fontsize=14)
    axes[1].set_xlabel("X (SCI pixels)", fontsize=12)
    axes[1].set_ylabel("Y (SCI pixels)", fontsize=12)
    cbar = fig.colorbar(im, ax=axes[1])
    cbar.set_label("RMS Variation Across Exposures (mas)", fontsize=12)

    out_file = os.path.join(output_dir, f"{label}_stability.png")
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    plt.close(fig)


def write_master_file(mean_data, meta_data, output_path):
    """Writes the master file with strict column alignment."""
    with open(output_path, "w") as f:
        f.write("# MASTER DISTORTION SOLUTION\n")
        f.write("# Generated by distortion_combine.py\n")
        f.write(f"# Sigma-clipped mean of {len(meta_data)} coefficients\n#\n")

        w_aper, w_idx, w_exp, w_val = 10, 10, 10, 23
        h = [
            "AperName",
            "siaf_index",
            "exponent_x",
            "exponent_y",
            "Sci2IdlX",
            "Sci2IdlY",
            "Idl2SciX",
            "Idl2SciY",
        ]

        f.write(
            f"{h[0]:>{w_aper}} , {h[1]:>{w_idx}} , {h[2]:>{w_exp}} , {h[3]:>{w_exp}} , "
            f"{h[4]:>{w_val}} , {h[5]:>{w_val}} , {h[6]:>{w_val}} , {h[7]:>{w_val}}\n"
        )

        for i, row in enumerate(meta_data):
            aper, ex, ey = str(row[0]), str(row[2]), str(row[3])
            siaf_idx = f"{int(row[1]):02d}" if str(row[1]).isdigit() else str(row[1])

            f.write(
                f"{aper:>{w_aper}} , {siaf_idx:>{w_idx}} , {ex:>{w_exp}} , {ey:>{w_exp}} , "
                f"{mean_data[i, 0]:{w_val}.12e} , {mean_data[i, 1]:{w_val}.12e} , "
                f"{mean_data[i, 2]:{w_val}.12e} , {mean_data[i, 3]:{w_val}.12e}\n"
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--sigma", type=float, default=2.5)
    args = parser.parse_args()

    subdirs = BATCH_SUBDIRS if BATCH_SUBDIRS else [""]

    for subdir in subdirs:
        current_data_dir = (
            os.path.join(args.data_dir, subdir) if subdir else args.data_dir
        )
        current_results_dir = os.path.join(current_data_dir, "calibration", "results")

        search_path = os.path.join(current_results_dir, FILE_PATTERN)
        files = [f for f in sorted(glob.glob(search_path)) if "MASTER" not in f]

        if not files:
            continue

        print(f"\nProcessing {len(files)} files in: {current_results_dir}")

        instr, aper, filt = get_metadata_from_fits(current_data_dir)
        master_name = (
            f"{instr}_siaf_distortion_{aper}_{filt}.txt"
            if "fgs" not in instr
            else f"{instr}_siaf_distortion_{aper}.txt"
        )

        data_cube, meta_data, _ = read_coefficients(files)
        if data_cube.size == 0:
            continue

        robust_mean, std_error = compute_robust_mean(data_cube, sigma=args.sigma)

        output_path = os.path.join(current_results_dir, master_name)
        write_master_file(robust_mean, meta_data, output_path)

        plot_label = master_name.replace(".txt", "")
        plot_stability(
            data_cube, robust_mean, meta_data, current_results_dir, plot_label
        )

    print("\nCombination Complete.")


if __name__ == "__main__":
    main()
