# JWST Distortion Calibration Pipeline

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
    * *Note: If `.xymq` files are missing, the pipeline can optionally fall back to `photutils` (if enabled) and will automatically cache the results into `.xymq` format.*
* Place your reference catalog (FITS format with RA/Dec) in a known path.

### 2. Run Calibration (Batch Processing)
Run the calibration script. This script automatically detects the instrument (NIRISS/FGS), selects the appropriate polynomial degree, and can loop through designated subdirectories.

```bash
python -m calibration.run_calibration
```

**What this does:**
* Scans `DATA_DIR` (and optionally iterates through `BATCH_SUBDIRS` like filter folders) for FITS files.
* Performs iterative disotrtion fitting for each file.
* Saves individual coefficient files (`*_distortion_coeffs.txt`) and plots to `[DATA_DIR]/[SUBDIR]/calibration/results` and `/plots`.

### 3. Combine Solutions

Generate a master distortion solution by robustly averaging the individual results across your processed directories.

```bash
python -m calibration.distortion_combine
```

**What this does:**
* Reads all coefficient files generated in Step 2.
* Performs a **sigma-clipped average** to remove outlier fits.
* Generate a stability plot showing the Log-RMS coefficient stability and a 2D spatial stability heatmap (in mas).
* Write the final averaged `..._distortion_coeffs.txt` file.

## Outputs

**Results(`/results`)**
* `*_distortion_coeffs.txt`: SIAF-compatible polynomial coefficients.
   * Columns: `Apername`, `siaf_index`, `exponent_x`, `exponent_y`, `Sci2IdlX`, `Sci2IdlY`, `Idl2SciX`, `Idl2SciY`.

**Plots(`/plots`)**
* `*_residuals.pdf`: Scatter plot of final residuals $(\Delta x, \Delta y)$ vs zero.
* `*_trends.pdf`: Spatial trends of residuals across the detector X/Y axes.
* `*_model_comparison.png`: Vector field showing the distortion model **Before** correction (50x scale) vs **After** correction(5000x scale).

## Configuration

Key parameters can be modified in `calibration/run_calibration.py`
```python
DATA_DIR      = "/path/to/base/directory"  # Base directory containing FITS and catalogs
# List of subdirectories (e.g., filters) to batch process. 
# Leave as an empty list [] to process DATA_DIR directly.
BATCH_SUBDIRS = ["F277W", "F380M"]         # Sub

OBS_Q_MAX     = 0.3    # Maximum quality flag for observed stars
OBS_SNR_MIN   = 60.0   # Minimum SNR for observed stars
N_BRIGHT_OBS  = 400    # Number of stars to use for initial alignment
POS_TOLERANCE = 0.1    # Final matching tolerance in arcseconds
REF_APPLY_PM  = True   # Apply Proper Motions to Reference Catalog
REF_EPOCH     = 2026.0 # Epoch of observation
```

## Dependencies
* `numpy`
* `matplotlib`
* `scipy`
* `astropy`
* `pysiaf`
* `photutils` (optional for rsource extraction)
