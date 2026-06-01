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
import yaml
from astropy.io import ascii, fits
from astropy.stats import sigma_clip

CONFIG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)
try:
    with open(CONFIG_FILE, "r") as f:
        cfg = yaml.safe_load(f)
    DEFAULT_DATA_DIR = cfg["paths"]["data_dir"]
    BATCH_SUBDIRS = (
        cfg["manual_batch"]["subdirs"] if cfg["manual_batch"]["subdirs"] else [""]
    )
except Exception as e:
    print(f"Warning: Could not load config.yml. Using defaults. ({e})")
    DEFAULT_DATA_DIR = "./data"
    BATCH_SUBDIRS = []

FILE_PATTERN = "*_distortion_coeffs.txt"


def read_coefficients(file_list):
    """Reads coefficients and extracts metadata from headers."""
    data_cube = []
    meta_data = None
    header = None
    obs_dates = []
    aper = "unknown"
    filt = "unknown"

    for f in file_list:
        try:
            # Read table data
            tab = ascii.read(f, format="csv", comment="#")
            data_cube.append([[row[4], row[5], row[6], row[7]] for row in tab])

            # Extract Observation Date, Aperture, and Filter from comments
            with open(f, "r") as fh:
                for line in fh:
                    if "Observation Date:" in line:
                        obs_dates.append(line.split(":")[-1].strip())
                    elif "Aperture:" in line:
                        aper = line.split(":")[-1].strip()
                    elif "Filter/Pupil:" in line:
                        filt = line.split(":")[-1].strip()

            if meta_data is None:
                meta_data = [[row[0], row[1], row[2], row[3]] for row in tab]
                header = tab.colnames
        except Exception as e:
            print(f"Warning: Could not read {f}: {e}")

    return np.array(data_cube), meta_data, header, obs_dates, aper, filt


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


def write_master_file(mean_data, meta_data, output_path, obs_date, aper, filt):
    """Writes the master file with strict column alignment."""
    with open(output_path, "w") as f:
        f.write("# MASTER DISTORTION SOLUTION\n")
        f.write(f"# Observation Date: {obs_date}\n")
        f.write(f"# Aperture: {aper.upper()}\n")
        f.write(f"# Filter/Pupil: {filt.upper()}\n")
        f.write("# Generated by distortion_combine.py\n#\n")
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
            # Using the dynamically passed aperture name instead of hardcoded row[0] just to be safe
            aperture_name = aper.upper()
            ex, ey = str(row[2]), str(row[3])
            siaf_idx = f"{int(row[1]):02d}" if str(row[1]).isdigit() else str(row[1])

            f.write(
                f"{aperture_name:>{w_aper}} , {siaf_idx:>{w_idx}} , {ex:>{w_exp}} , {ey:>{w_exp}} , "
                f"{mean_data[i, 0]:{w_val}.12e} , {mean_data[i, 1]:{w_val}.12e} , "
                f"{mean_data[i, 2]:{w_val}.12e} , {mean_data[i, 3]:{w_val}.12e}\n"
            )


def main(override_data_dir=None, override_subdirs=None):
    parser = argparse.ArgumentParser(
        description="Combine JWST distortion coefficients."
    )
    # Changed to a positional argument with nargs="?" so it's optional but captures raw inputs
    parser.add_argument(
        "data_dir", nargs="?", default=DEFAULT_DATA_DIR, help="Root data directory"
    )
    parser.add_argument("--sigma", type=float, default=2.5, help="Sigma clip threshold")

    args, _ = parser.parse_known_args()

    # Determine the active directory
    active_data_dir = override_data_dir if override_data_dir else args.data_dir

    # Determine subdirectories to process
    if override_subdirs is not None:
        subdirs = override_subdirs
    else:
        # SMART AUTO-DISCOVERY: Find any folders that contain a 'results' or 'calibration/results' folder
        found_subdirs = [
            d
            for d in os.listdir(active_data_dir)
            if os.path.isdir(os.path.join(active_data_dir, d))
            and (
                os.path.isdir(os.path.join(active_data_dir, d, "results"))
                or os.path.isdir(
                    os.path.join(active_data_dir, d, "calibration", "results")
                )
            )
        ]

        if found_subdirs:
            subdirs = found_subdirs
            print(
                f"Auto-detected {len(subdirs)} subdirectories with results: {subdirs}"
            )
        else:
            subdirs = BATCH_SUBDIRS if BATCH_SUBDIRS else [""]

    for subdir in subdirs:
        current_data_dir = (
            os.path.join(active_data_dir, subdir) if subdir else active_data_dir
        )

        # Safely search for the results folder
        current_results_dir = os.path.join(current_data_dir, "results")
        if not os.path.exists(current_results_dir):
            current_results_dir = os.path.join(
                current_data_dir, "calibration", "results"
            )

        search_path = os.path.join(current_results_dir, FILE_PATTERN)
        files = [
            f
            for f in sorted(glob.glob(search_path))
            if "MASTER" not in f and "siaf_distortion" not in f
        ]

        if not files:
            continue

        print(f"\nProcessing {len(files)} files in: {current_results_dir}")

        # 1. Read coefficients and extract metadata from headers
        data_cube, meta_data, _, obs_dates, aper, filt = read_coefficients(files)
        if data_cube.size == 0:
            continue

        # 2. Determine median observation date for naming
        if obs_dates:
            median_date_str = sorted(obs_dates)[len(obs_dates) // 2]
            date_stamp = median_date_str.replace("-", "")
        else:
            median_date_str = "unknown"
            date_stamp = "00000000"

        # 3. Determine Master Filename from the extracted metadata
        # Infer instrument based on the aperture name
        instr = "fgs" if "fgs" in aper.lower() else "niriss"

        if instr == "fgs":
            master_name = f"{instr}_siaf_distortion_{aper.lower()}_{date_stamp}.txt"
        else:
            master_name = f"{instr}_siaf_distortion_{aper.lower()}_{filt.lower()}_{date_stamp}.txt"

        # 4. Calculate robust mean
        robust_mean, std_error = compute_robust_mean(data_cube, sigma=args.sigma)

        # Output the master files to the root of the active directory
        output_path = os.path.join(active_data_dir, master_name)
        write_master_file(
            robust_mean, meta_data, output_path, median_date_str, aper, filt
        )

        # 5. Generate physical stability heatmap and RMS plot in the root directory
        plot_label = master_name.replace(".txt", "")
        plot_stability(data_cube, robust_mean, meta_data, active_data_dir, plot_label)

    print(f"\nCombination Complete. Master files saved to: {active_data_dir}")


if __name__ == "__main__":
    main()
