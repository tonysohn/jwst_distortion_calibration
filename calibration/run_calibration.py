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
from pathlib import Path

from astropy.io import fits

from .distortion_pipeline import DistortionPipeline, PipelineConfig

# =============================================================================
# CONFIGURATION
# =============================================================================
DATA_DIR = "/Users/tsohn/JWST/FGS/6608"  # Input directory for FITS/XYMQ files

# List of subdirectories (e.g., filters) to batch process.
# Leave as an empty list [] to process DATA_DIR directly.

# Use below for NIRISS
# BATCH_SUBDIRS = [
#    "F090W",
#    "F115W",
#    "F140M",
#    "F150W",
#    "F158M",
#    "F200W",
#    "F277W",
#    "F356W",
#    "F380M",
#    "F430M",
#    "F444W",
#    "F480M",
# ]

# Use below for FGS
BATCH_SUBDIRS = ["FGS1", "FGS2"]

REF_FILE = "/Users/tsohn/JWST/NIRISS/JWST-Distortion-Calibration/calibration/lmc_calibration_field_hst_2017p38_jwstmags.fits"  # Reference catalog (GAIA/HST)
OUTPUT_DIR = os.path.join(DATA_DIR, "calibration")  # Output directory

# Processing Parameters
SOURCE_METHOD = "xymq"
OBS_Q_MIN = 0.001
OBS_Q_MAX = 0.3
OBS_SNR_MIN = 60.0
N_BRIGHT_OBS = 400
POS_TOLERANCE = 0.1
INITIAL_TOLERANCE = 0.5
REF_APPLY_PM = True
REF_EPOCH = 2026.0
USE_GRID_FITTING = True
GRID_SIZE = 20
# =============================================================================


def get_config_from_header(fits_path):
    """Extracts Instrument and Aperture from FITS header."""
    try:
        with fits.open(fits_path) as hdul:
            header = hdul[0].header
            instr = header.get("INSTRUME", "").strip().upper()
            aper = header.get("APERNAME", header.get("PPS_APER", "")).strip()
            return instr, aper
    except Exception as e:
        print(f"Error reading FITS header: {e}")
        return None, None


def process_single_file(fits_file, output_dir):
    """Runs calibration for a single image."""
    print(f"\nProcessing: {fits_file}")

    if not os.path.exists(REF_FILE):
        print(f"Error: Ref file not found: {REF_FILE}")
        return

    detected_instr, detected_aper = get_config_from_header(fits_file)
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
        source_extraction_method=SOURCE_METHOD,
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
        pipeline.prepare_and_load_catalogs(xymq_file, fits_file, REF_FILE)
        pipeline.run()
    except Exception as e:
        print(f"FAILED to process {fits_file}: {e}")
        import traceback

        traceback.print_exc()


def main():
    # If BATCH_SUBDIRS is populated, process those subdirectories.
    # If it is empty, process DATA_DIR directly.
    subdirs = BATCH_SUBDIRS if BATCH_SUBDIRS else [""]

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
            process_single_file(f, current_output_dir)

    if not found_any_files:
        print(f"No FITS files found in {DATA_DIR} or its specified subdirectories.")


if __name__ == "__main__":
    main()
