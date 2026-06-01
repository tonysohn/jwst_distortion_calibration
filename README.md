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
* Ensure you have corresponding source catalogs (`.xymq` files) in the same directory.
* You can generate these catalogs by running the `jwst1pass` routine (found [here](https://www.stsci.edu/~jayander/JWST1PASS/CODE/)) on your FITS files.
* Alternatively, use the included standalone photometry script to extract sources and generate `.xymq` files via `photutils`. The script will automatically detect the instrument from the FITS headers (`python tools/distortion_photometry.py /path/to/fits/dir`).
* Place your reference catalog (FITS format with RA/Dec) in a known path so it can be referenced in your `config.yml`.
* The package currently supports the HST LMC Calibration Field catalog, which you can install via `pip`. Use the following terminal commands:
  ```bash
  # Note: Requires git lfs to be installed for the large catalog files
  git lfs install
  pip install jwst-calibration-field
    ```
  See the following page for details: https://github.com/spacetelescope/jwst-calibration-field

* Once installed, the reference FITS file is bundled securely inside the package. To find the exact file path to copy into your `config.yml`'s `ref_file` parameter, run this quick Python command in your terminal:

  ```bash
  python -c "import jwcf, os, glob; print(glob.glob(os.path.join(os.path.dirname(jwcf.__file__), 'data', '*.fits*'))[0])"
  ```
* Alternatively, you can export the catalog to a clean FITS file anywhere on your system by doing:
  ```python
  from jwcf import hst_catalog
    # Load the catalog and save it locally
    catalog = hst_catalog()
    catalog.write("/path/to/save/hst_reference_catalog.fits", format="fits", overwrite=True)
    ```
* For full documentation, visit the [jwcf GitHub](https://github.com/spacetelescope/jwst-calibration-field) repository.
* This package assumes input `*_cal.fits` images have WCS accurate to within ~1 arcsec, otherwise the cross-matching of observed and reference catalogs are likely to fail leading to incorrect distortion solutions.
* In crowded fields like the LMC Calibration Field, JWST images can be offset by a few arcseconds due to guiding on the wrong guide star.
* If you find such cases, the WCS of corresponding images can be *adjusted* before running the distortion calibration codes by applying an offset using the `jwst` pipeline command `adjust_wcs` as follows:

  ```bash
  adjust_wcs jw01501002001_02101_00001_nis_cal.fits -u --overwrite -r -1.042e-3 -d 1.194e-4
    # -u —overwrite updates the WCS of the original image.
    # (Alternatively, use --suffix wcsadj_cal to create a new image.)
    # -r applies the RA offset in degrees
    # -d applies the Dec offset in degrees
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
python tools/run_calibration.py
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
**Important Directory Targeting:** When running distortion_combine.py manually, you must point it to the root data directory that contains your filter/detector subdirectories (e.g., /path/to/1018/). Do not point it directly at an inner results/ folder. The script uses smart auto-discovery to scan the root folder, find the subdirectories, and extract the results automatically.


### 3 Trend Analysis (Multi-Epoch)

Analyze the long-term physical stability of the detector optics across multiple observing epochs. Gather all your generated master coefficient files (from Step 3) across various years/epochs and place them into a single centralized directory.

```bash
python tools/distortion_trends.py /path/to/centralized/master_files_dir
```
**What this does:**
* Automatically parses filenames to group data by filter (e.g., `F090W`) or detector (e.g., `FGS1_FULL`).
* Extracts precise physical metrics: independent X/Y Pixel Scales (mas), Pixel Skew (arcsec), and Higher-Order Distortion RMS ($\mu$as).
* Generates a comprehensive time-series 4-panel plot and a detailed ASCII summary table for each group.

### 4. Solution Comparison & Operational Impact
Evaluate the exact astrometric differences between a reference distortion solution and one or more new solutions. This tool directly calculates the operational impact of updating reference files.

```bash
# Compare one or multiple files against a master reference
python tools/distortion_compare.py path/to/reference.txt path/to/comparison1.txt [path/to/comparison2.txt ...]
```
**What this does:**
* Calculates the exact spatial RMS error (in mas) across the entire detector array
* Identifies and visually highlights the specific "Worst-Case" $(X, Y)$ pixel where the maximum astrometric deviation occurs.
* Automatically saves a 3-panel diagnostic plot directly adjacent to every comparison file processed.

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
* `SciencePlots` (optional for publication-quality figures)
