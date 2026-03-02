"""
JWST Distortion Calibration Script
==================================
Runs the distortion pipeline on a directory of FITS images.

Usage:
    python run_calibration.py

Dependencies:
    Requires corresponding .xymq catalog files for each .fits image.
"""

import glob
import os
import sys
from pathlib import Path

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table
from astropy.time import Time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from calibration.distortion_pipeline import DistortionPipeline, PipelineConfig

# =============================================================================
# LOAD CONFIGURATION
# =============================================================================
CONFIG_FILE = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "config.yml")
)

with open(CONFIG_FILE, "r") as f:
    cfg = yaml.safe_load(f)

# Standard script uses the organized data_dir and manual_batch subdirs
DATA_DIR = cfg["paths"]["data_dir"]
REF_FILE = cfg["paths"]["ref_file"]
BATCH_SUBDIRS = (
    cfg["manual_batch"]["subdirs"] if cfg["manual_batch"]["subdirs"] else [""]
)

# Processing Parameters
proc = cfg["processing"]
OBS_Q_MIN = proc["obs_q_min"]
OBS_Q_MAX = proc["obs_q_max"]
OBS_SNR_MIN = proc["obs_snr_min"]
N_BRIGHT_OBS = proc["n_bright_obs"]
POS_TOLERANCE = proc["pos_tolerance"]
INITIAL_TOLERANCE = proc["initial_tolerance"]
REF_APPLY_PM = proc["ref_apply_pm"]
USE_GRID_FITTING = proc["use_grid_fitting"]
GRID_SIZE = proc["grid_size"]
# =============================================================================


def get_config_from_header(fits_path):
    """Extracts Instrument and Aperture from FITS header."""
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            instr = header.get("INSTRUME", "").strip().upper()
            aper = header.get("APERNAME", header.get("PPS_APER", "")).strip()
            date = header.get("DATE-OBS")
            return instr, aper, date
    except Exception as e:
        print(f"Error reading FITS header: {e}")
        return None, None, None


def process_single_file(fits_file, output_dir, master_ref_catalog):
    """Runs calibration for a single image."""
    print(f"\nProcessing: {fits_file}")

    if not os.path.exists(REF_FILE):
        print(f"Error: Ref file not found: {REF_FILE}")
        return

    detected_instr, detected_aper, date = get_config_from_header(fits_file)
    if not detected_instr:
        print(f"Skipping {fits_file}: Could not read header.")
        return

    # Dynamic Polynomial Degree Selection
    if "NIRISS" in detected_instr:
        poly_degree = 5
    elif "FGS" in detected_instr:
        poly_degree = 4
    else:
        poly_degree = 5
        print(f"Warning: Unknown instrument {detected_instr}, defaulting to degree 5.")

    # EPOCH OF OBSERVATION (for proper motion correction)
    REF_EPOCH = Time(date, format="iso").decimalyear if date != "Unknown" else 2025.0

    file_root = Path(fits_file).stem
    xymq_file = str(Path(fits_file).with_suffix(".xymq"))

    if not os.path.exists(xymq_file):
        print(f"Skipping {fits_file}: Catalog file not found ({xymq_file})")
        return

    config = PipelineConfig(
        working_dir=output_dir,  # <-- Pass the dynamic output directory here
        file_root=file_root,
        instrument=detected_instr,
        aperture_name=detected_aper,
        poly_degree=poly_degree,
        obs_q_min=OBS_Q_MIN,
        obs_q_max=OBS_Q_MAX,
        obs_snr_min=OBS_SNR_MIN,
        n_bright_obs=N_BRIGHT_OBS,
        pos_tolerance_arcsec=POS_TOLERANCE,
        initial_tolerance_arcsec=INITIAL_TOLERANCE,
        use_grid_fitting=USE_GRID_FITTING,
        grid_size=GRID_SIZE,
        max_iters=20,
        damping_factor=0.75,
        ref_apply_pm=REF_APPLY_PM,
        ref_epoch=REF_EPOCH,
        ref_mag_buffer=5.0,
    )

    try:
        pipeline = DistortionPipeline(config)
        pipeline.prepare_and_load_catalogs(xymq_file, fits_file, master_ref_catalog)
        pipeline.run()
    except Exception as e:
        print(f"FAILED to process {fits_file}: {e}")
        import traceback

        traceback.print_exc()


def main():
    # If BATCH_SUBDIRS is populated, process those subdirectories.
    # If it is empty, process DATA_DIR directly.
    subdirs = BATCH_SUBDIRS if BATCH_SUBDIRS else [""]

    # --- NEW: Peek at the first FITS file to determine the global REF_EPOCH ---
    first_fits = None
    for subdir in subdirs:
        search_dir = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
        fits_list = sorted(glob.glob(os.path.join(search_dir, "*.fits")))
        if fits_list:
            first_fits = fits_list[0]
            break

    if not first_fits:
        print(f"No FITS files found in {DATA_DIR} or its specified subdirectories.")
        return

    _, _, first_date = get_config_from_header(first_fits)
    global_ref_epoch = (
        Time(first_date, format="iso").decimalyear
        if first_date and first_date != "Unknown"
        else 2026.0
    )
    # -------------------------------------------------------------------------

    print(f"\nLoading master reference catalog into memory from: {REF_FILE}")
    master_ref_catalog = Table.read(REF_FILE)
    print(
        f"Loaded {len(master_ref_catalog)} reference stars. Starting batch processing...\n"
    )

    global REF_APPLY_PM

    if REF_APPLY_PM:
        CATALOG_EPOCH = 2017.38
        dt = global_ref_epoch - CATALOG_EPOCH

        # Fixed the \D escape sequence warning by using a double backslash
        print(
            f"Applying proper motions: $\\Delta$t = {dt:.3f} years (to Epoch {global_ref_epoch:.2f})..."
        )

        # Check if the catalog contains the necessary PM columns
        if (
            "pmra" in master_ref_catalog.colnames
            and "pmdec" in master_ref_catalog.colnames
        ):
            # Convert mas/yr to degrees/yr
            pmra_deg_yr = master_ref_catalog["pmra"] / 3.6e6
            pmdec_deg_yr = master_ref_catalog["pmdec"] / 3.6e6

            # Apply to coordinates.
            # Note: Standard Gaia/HST catalogs store 'pmra' as (pm_ra * cos(dec)).
            # Therefore, we must divide by cos(dec) to get the true RA shift in degrees.
            cos_dec = np.cos(np.radians(master_ref_catalog["dec_deg"]))

            master_ref_catalog["ra_deg"] += (pmra_deg_yr * dt) / cos_dec
            master_ref_catalog["dec_deg"] += pmdec_deg_yr * dt

            print("  Proper motion applied successfully.")
        else:
            print("  Warning: 'pmra' or 'pmdec' not found. Skipping PM correction.")

        # VERY IMPORTANT: Turn off the flag so distortion_pipeline.py doesn't
        # try to apply it a second time inside the loop!
        REF_APPLY_PM = False

    found_any_files = False

    for subdir in subdirs:
        # Construct the path for the current batch
        current_data_dir = os.path.join(DATA_DIR, subdir) if subdir else DATA_DIR
        current_output_dir = os.path.join(current_data_dir, "calibration")

        search_pattern = os.path.join(current_data_dir, "*.fits")
        fits_files = sorted(glob.glob(search_pattern))

        if not fits_files:
            continue

        found_any_files = True

        print(f"\n{'=' * 60}")
        print(f"Processing Directory: {current_data_dir}")
        print(f"Found {len(fits_files)} files.")
        print(f"{'=' * 60}")

        # Ensure the output directory exists for this specific subfolder
        os.makedirs(current_output_dir, exist_ok=True)

        for f in fits_files:
            process_single_file(f, current_output_dir, master_ref_catalog)

    if not found_any_files:
        print(f"No FITS files found in {DATA_DIR} or its specified subdirectories.")


if __name__ == "__main__":
    main()
