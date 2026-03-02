# JWST Distortion Calibration Pipeline

This package provides a robust, iterative polynomial distortion calibration tool for JWST instruments, specifically **NIRISS** and **FGS**. It fits Science-to-Ideal (`Sci2Idl`) and Ideal-to-Science (`Idl2Sci`) transformations using reference catalogs (e.g., Gaia, HST).

## Features
* **Iterative Matching:** Uses a "bootstrap" approach to align catalogs with minimal prior knowledge (no WCS required).
* **Robust Fitting:** Implements sigma-clipping, damping factors, 2D radial distance rejection, and robust polynomial fitting to ensure convergence.
* **Dynamic Support:** Automatically handles different image sizes (NAXIS) and instruments (polynomial degrees are dynamically assigned).
* **Visualization:** Generates diagnostic plots including residual maps, spatial trend plots, and "Before/After" vector fields with **dynamic vector scaling**.
* **Batch Processing:** Processes multiple subdirectories (e.g., different filters) sequentially and combines results into high-fidelity master solutions with spatial stability heatmaps.

## Installation

1.  Clone this repository:
    ```bash
    git clone [https://github.com/tonysohn/jwst-distortion-calibration.git](https://github.com/tonysohn/jwst-distortion-calibration.git)
    cd jwst-distortion-calibration
    ```

2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

### 1. Data Preparation
* Place your FITS images (e.g., `*_cal.fits`) in a base data directory.
* Ensure you have corresponding source catalogs (`.xymq` files) in the same directory. There are two ways to generate these:
    1. **`jwst1pass`**: Run the `jwst1pass` routine (found [here](https://www.stsci.edu/~jayander/JWST1PASS/CODE/)) on your FITS files to generate the catalogs.
    2. **Standalone Photometry Script**: Use the included script to extract sources and generate `.xymq` files via `photutils`. The script will automatically detect the instrument from the FITS headers:
       ```bash
       python tools/distortion_photometry /path/to/fits/dir
       ```
* Place your reference catalog (FITS format with RA/Dec) in a known path. The package currently supports the HST LMC Calibration Field catalog. You can install the catalog by doing
	```bash
    pip install jwst-calibration-field
    ```
  See the following page for details: https://github.com/spacetelescope/jwst-calibration-field
* This package assumes input `*_cal.fits` images have WCS accurate to within ~1 arcsec, otherwise the cross-matching of observed and reference catalogs are likely to fail leading to incorrect distortion solutions. In crowded fields like the LMC Calibration Field, JWST images can be offset by a few arcseconds due to guiding on the wrong guide star. If you find such cases, the WCS of corresponding images can be *adjusted* before running the distortion calibration codes by applying an offset using the `jwst` pipeline command `adjust_wcs` as follows:

	```bash
	adjust_wcs jw01501002001_02101_00001_nis_cal.fits -u --overwrite -r -1.042e-3 -d 1.194e-4
    # -u —overwrite updates the WCS of the original image.
    # (Alternatively, use --suffix wcsadj_cal to create a new image.)
    # -r applies the RA offset
    # -d applies the Dec offset
    ```

### 2. Run Calibration (Batch Processing)
You can run the calibration pipeline in either **Automated** or **Manual** mode depending on how your data is structured.


#### Option A: Automated Pipeline (Recommended)
If you have a chaotic folder containing a mix of filters, detectors, and exposures, use the automated batch script. It reads directly from `raw_data_dir` in your `config.yml`.

```bash
python tools/run_calibration_batch.py
```


Run the calibration script. This script automatically detects the instrument (NIRISS/FGS), selects the appropriate polynomial degree, and can loop through designated subdirectories.

```bash
python tools/run_calibration
```
**What this does:**
* Scans the raw directory, dynamically identifies the instrument and filter/detector for each image, and organizes them into subfolders (e.g., `/F090W/` or `/FGS1_FULL/`).

* Runs the iterative distortion fitting for every image.

* Automatically triggers the combination script to average the results per filter.

* Deposits the final averaged `..._distortion_coeffs.txt` master files directly into the root `raw_data_dir` for easy access.


#### Option B: Manual Pipeline
If you prefer to process specific subdirectories (e.g., just one filter), you can manually execute the steps using the `data_dir` and `manual_batch` settings in your `config.yml`.

```bash
python tools/run_calibration.py
python tools/distortion_combine.py /path/to/data_dir
```
### 3 Trend Analysis (Multi-Epoch)

Analyze the long-term physical stability of the detector optics across multiple observing epochs. Gather all your generated master coefficient files (from Step 3) across various years/epochs and place them into a single centralized directory.

```bash
python tools/distortion_trends.py /path/to/centralized/master_files_dir
```
**What this does:**
* Automatically parses filenames to group data by filter (e.g., `F090W`) or detector (e.g., `FGS1_FULL`).
* Extracts precise physical metrics: independent X/Y Pixel Scales (mas), Pixel Skew (arcsec), and Higher-Order Distortion RMS ($\mu$as).
* Generates a comprehensive time-series 4-panel plot and a detailed ASCII summary table for each group.

## Outputs

**Results(`/results`)**
* `*_distortion_coeffs.txt`: SIAF-compatible polynomial coefficients.
   * Columns: `Apername`, `siaf_index`, `exponent_x`, `exponent_y`, `Sci2IdlX`, `Sci2IdlY`, `Idl2SciX`, `Idl2SciY`.

**Plots(`results` and `/plots`)**
* `*_residuals.pdf`: Scatter plot of final residuals $(\Delta x, \Delta y)$ vs zero.
* `*_trends.pdf`: Spatial trends of residuals across the detector X/Y axes.
* `*_model_comparison.png`: Vector field showing the distortion model **Before** correction (50x scale) vs **After** correction(5000x scale).
* `*_stability.png`: 2D heatmap showing spatial RMS variation (in mas) across multiple exposures within a single epoch.

## Configuration

All pipeline parameters are managed via a central `config.yml` file located in the root of the repository. This file looks like below:

```yml
# =============================================================================
# JWST Distortion Pipeline Configuration
# =============================================================================

paths:
  # -> FOR AUTOMATED BATCH: The chaotic folder where all raw FITS/XYMQ files are dumped
  raw_data_dir: "/path/to/raw/directory"

  # -> FOR MANUAL BATCH: The organized base directory
  data_dir: "/path/to/organized/directory"

  # Reference catalog (GAIA/HST)
  ref_file: "/path/to/reference_catalog.fits"

manual_batch:
  # Used ONLY by run_calibration.py. (run_calibration_batch.py ignores this)
  subdirs:
    - "F090W"
    - "F115W"

processing:
  # Shared by ALL scripts to ensure mathematically consistent results
  obs_q_min: 0.001
  obs_q_max: 0.3
  obs_snr_min: 60.0
  n_bright_obs: 400
  pos_tolerance: 0.1
  initial_tolerance: 0.5
  ref_apply_pm: true
  use_grid_fitting: true
  grid_size: 20
```

## Dependencies
* `numpy`
* `matplotlib`
* `scipy`
* `astropy`
* `pysiaf`
* `pyyaml`
* `photutils` (optional for rsource extraction)
