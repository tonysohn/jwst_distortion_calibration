import glob
import os
import sys

import yaml
from astropy.io import fits

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from calibration.distortion_pipeline import DistortionPipeline, PipelineConfig
from tools import distortion_combine


def main():
    # 1. Load configuration
    config_file = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "config.yml")
    )
    try:
        with open(config_file, "r") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        print(f"Error loading config.yml: {e}")
        return

    # Use the raw folder for inputs, and the organized folder for outputs
    raw_dir = cfg["paths"]["raw_data_dir"]
    out_base_dir = raw_dir
    ref_file = cfg["paths"]["ref_file"]
    proc_cfg = cfg["processing"]

    print(f"\nScanning {raw_dir} for unorganized FITS images...")

    # 2. Inventory FITS files
    fits_files = glob.glob(os.path.join(raw_dir, "*_cal.fits"))
    if not fits_files:
        fits_files = glob.glob(os.path.join(raw_dir, "*.fits"))

    if not fits_files:
        print("No FITS files found in the raw data directory.")
        return

    # 3. Categorize files by Filter or Detector
    groups = {}
    for f_path in fits_files:
        try:
            with fits.open(f_path) as hdul:
                hdr = hdul[0].header
                inst = hdr.get("INSTRUME", "UNKNOWN").strip().upper()
                aper = (
                    hdr.get("APERNAME", hdr.get("PPS_APER", "UNKNOWN")).strip().upper()
                )

                if inst == "NIRISS":
                    filt_key = hdr.get("FILTER", "UNKNOWN").strip().upper()
                    pup_key = hdr.get("PUPIL", "UNKNOWN").strip().upper()
                    group_name = pup_key if "CLEAR" in filt_key else filt_key
                elif inst == "FGS":
                    group_name = aper
                else:
                    group_name = "UNKNOWN"

                if group_name not in groups:
                    groups[group_name] = []
                groups[group_name].append((f_path, inst, aper))
        except Exception as e:
            print(
                f"Warning: Could not read header of {os.path.basename(f_path)}. ({e})"
            )

    # 4. Process each group
    processed_subdirs = []

    for group_name, files in groups.items():
        if group_name == "UNKNOWN":
            continue

        print(f"\n{'=' * 60}")
        print(f"PROCESSING GROUP: {group_name} ({len(files)} files)")
        print(f"{'=' * 60}")

        # Create dedicated output directory in the organized data_dir
        group_out_dir = os.path.join(out_base_dir, group_name)
        os.makedirs(group_out_dir, exist_ok=True)
        processed_subdirs.append(group_name)

        for f_path, inst, aper in files:
            file_root = (
                os.path.basename(f_path).replace(".fits", "").replace("_cal", "")
            )
            xymq_path = f_path.replace(".fits", ".xymq")

            if not os.path.exists(xymq_path):
                print(f"  -> Skipping {file_root}: Missing .xymq catalog.")
                continue

            config = PipelineConfig(
                working_dir=group_out_dir,  # Outputs go to the organized folder
                file_root=file_root,
                aperture_name=aper,
                instrument=inst,
                poly_degree=5 if inst == "NIRISS" else 4,
                obs_q_min=proc_cfg["obs_q_min"],
                obs_q_max=proc_cfg["obs_q_max"],
                obs_snr_min=proc_cfg["obs_snr_min"],
                ref_apply_pm=proc_cfg["ref_apply_pm"],
                n_bright_obs=proc_cfg["n_bright_obs"],
                pos_tolerance_arcsec=proc_cfg["pos_tolerance"],
                initial_tolerance_arcsec=proc_cfg["initial_tolerance"],
                use_grid_fitting=proc_cfg["use_grid_fitting"],
                grid_size=proc_cfg["grid_size"],
            )

            pipeline = DistortionPipeline(config)
            try:
                pipeline.prepare_and_load_catalogs(xymq_path, f_path, ref_file)
                pipeline.run()
            except Exception as e:
                print(f"  -> Error processing {file_root}: {e}")

    # 5. Programmatically trigger the combination script
    print(f"\n{'=' * 60}")
    print("CALIBRATION COMPLETE. INITIATING MASTER AVERAGING...")
    print(f"{'=' * 60}")

    distortion_combine.main(
        override_data_dir=out_base_dir, override_subdirs=processed_subdirs
    )


if __name__ == "__main__":
    main()
