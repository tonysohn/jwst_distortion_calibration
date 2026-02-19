"""
JWST Distortion Pipeline Controller
-----------------------------------
This module orchestrates the entire calibration process:
1. Loads observed and reference catalogs.
2. Performs initial alignment (pointing correction).
3. Iteratively fits polynomial distortion solutions.
4. Generates diagnostic plots and final SIAF coefficient files.

Author: S. T. Sohn
"""

import datetime
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pysiaf
from astropy.io import fits

from .distortion_core import DistortionFitter, PolynomialDistortion
from .distortion_data import prepare_obs_catalog, prepare_ref_catalog
from .distortion_matching_smart import match_in_ideal_frame, match_with_pointing_prior
from .distortion_plotting import plot_comparison_models, plot_residuals, plot_trends


@dataclass
class PipelineConfig:
    """Configuration parameters for the distortion pipeline."""

    working_dir: str
    file_root: str  # Unique ID for output filenames (e.g., "jw0928...")
    aperture_name: str  # SIAF aperture name (e.g., "NIS_CEN")
    instrument: str = "NIRISS"
    poly_degree: int = 5  # Degree of polynomial (5 for NIRISS, 4 for FGS)
    obs_q_min: float = 0.001
    obs_q_max: float = 0.3
    obs_snr_min: float = 40.0
    source_extraction_method: str = "xymq"
    ref_apply_pm: bool = True
    ref_epoch: float = 2026.0
    ref_mag_buffer: float = 5.0
    ref_buffer_arcsec: float = 30.0
    n_bright_obs: int = 200
    ref_mag_bins: Optional[list] = None
    pos_tolerance_arcsec: float = 0.1
    initial_tolerance_arcsec: float = 0.5
    min_matches: int = 50
    sigma_fit: float = 2.5
    convergence_tol_mas: float = 0.05
    max_iters: int = 20
    damping_factor: float = 0.75
    use_grid_fitting: bool = True
    grid_size: int = 20

    def __post_init__(self):
        self.plot_dir = os.path.join(self.working_dir, "plots")
        self.res_dir = os.path.join(self.working_dir, "results")
        self.data_dir = os.path.join(self.working_dir, "prepared_catalogs")
        os.makedirs(self.plot_dir, exist_ok=True)
        os.makedirs(self.res_dir, exist_ok=True)
        os.makedirs(self.data_dir, exist_ok=True)


class DistortionPipeline:
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.siaf = pysiaf.Siaf(config.instrument)
        self.aperture = self.siaf[config.aperture_name]
        self.fitter = DistortionFitter(config.poly_degree, config.sigma_fit)
        self.obs_catalog = None
        self.ref_catalog = None
        self.history = {"rms_x": [], "rms_y": [], "n_stars": []}
        self.prior_coeffs = None
        self.current_results = None
        self.successful_mag_bin = None
        self.naxis1 = 2048  # Default
        self.naxis2 = 2048

    def prepare_and_load_catalogs(self, xymq_file, fits_file, ref_file):
        """Loads observed and reference catalogs, extracting header info."""
        print(f"\n[1] Loading Obs Catalog from {xymq_file}...")
        self.obs_catalog = prepare_obs_catalog(
            xymq_file,
            fits_file,
            q_max=self.config.obs_q_max,
            snr_min=self.config.obs_snr_min,
            source_method=self.config.source_extraction_method,
        )

        # Read image dimensions
        with fits.open(fits_file) as hdul:
            self.naxis1 = hdul[1].header.get("NAXIS1", 2048)
            self.naxis2 = hdul[1].header.get("NAXIS2", 2048)
            print(f"    Detected Image Dimensions: {self.naxis1} x {self.naxis2}")

        print(f"\n[2] Loading Ref Catalog from {ref_file}...")
        self.ref_catalog = prepare_ref_catalog(
            ref_file,
            self.obs_catalog,
            mag_buffer=self.config.ref_mag_buffer,
            apply_pm=self.config.ref_apply_pm,
            target_epoch=self.config.ref_epoch,
        )

    def _calculate_weights(self, catalog):
        """Calculates flux-based weights for fitting (optional)."""
        if "mag_ab" in catalog.colnames:
            mags = catalog["mag_ab"]
            weights = 10.0 ** (-0.4 * mags)
            weights = weights / np.median(weights)
            return weights
        return None

    def run(self):
        """Main execution loop for iterative calibration."""
        if self.obs_catalog is None:
            raise ValueError("Load catalogs first.")
        print(
            f"\n{'=' * 60}\nSTARTING CALIBRATION: {self.config.file_root} ({self.config.aperture_name})\n{'=' * 60}"
        )

        for i in range(self.config.max_iters):
            print(f"\n--- Iteration {i + 1}/{self.config.max_iters} ---")

            # 1. Update Reference Stars to Ideal Frame (using current SIAF)
            self.update_reference_ideal_coords()

            current_tolerance = (
                self.config.initial_tolerance_arcsec
                if i == 0
                else self.config.pos_tolerance_arcsec
            )
            mag_bins = (
                self.config.ref_mag_bins
                if i == 0
                else ([self.successful_mag_bin] if self.successful_mag_bin else None)
            )

            # 2. Matching
            if i == 0:
                # First pass: Robust pointing determination
                s1_obs, s1_ref, info1 = match_with_pointing_prior(
                    self.obs_catalog,
                    self.ref_catalog,
                    n_bright_obs=self.config.n_bright_obs,
                    ref_mag_bins=mag_bins,
                    pos_tolerance_arcsec=0.5,
                    verbose=False,
                )
                if "mag_bin" in info1:
                    self.successful_mag_bin = info1["mag_bin"]

                # Apply initial offset
                x_idl_all, y_idl_all = self.project_to_ideal(self.obs_catalog)
                x_idl_s1, y_idl_s1 = self.project_to_ideal(s1_obs)
                dx = np.median(s1_ref["x_idl"] - x_idl_s1)
                dy = np.median(s1_ref["y_idl"] - y_idl_s1)
                x_idl_aligned = x_idl_all + dx
                y_idl_aligned = y_idl_all + dy
            else:
                # Subsequent passes: Use current polynomial projection
                x_idl_aligned, y_idl_aligned = self.project_to_ideal(self.obs_catalog)

            obs_matched, ref_matched, info2 = match_in_ideal_frame(
                self.obs_catalog,
                self.ref_catalog,
                x_idl_obs=x_idl_aligned,
                y_idl_obs=y_idl_aligned,
                pos_tolerance_arcsec=current_tolerance,
            )

            if len(obs_matched) < 50:
                print("Too few matches. Aborting.")
                break

            # 3. Fit Distortion Polynomials
            results = self.fit_distortion(obs_matched, ref_matched)

            # Store results
            self.prior_coeffs = results
            self.current_results = results
            self.history["rms_x"].append(results["rms_x"])
            self.history["rms_y"].append(results["rms_y"])
            self.history["n_stars"].append(results["n_stars"])

            print(
                f"  Fit Results: RMS X={results['rms_x']:.3f}, Y={results['rms_y']:.3f} mas, N={results['n_stars']}"
            )

            if self.check_convergence():
                print("\nCONVERGENCE ACHIEVED")
                break

            # Update SIAF object in memory for next iteration
            self.apply_results_to_aperture(results)

        self.finalize()

    def project_to_ideal(self, catalog):
        """Projects science pixels to Ideal frame using current best solution."""
        x_sci = catalog["x_SCI"] - self.aperture.XSciRef
        y_sci = catalog["y_SCI"] - self.aperture.YSciRef
        if self.prior_coeffs is None:
            # Use static SIAF initially
            x_raw = catalog["x_SCI"]
            y_raw = catalog["y_SCI"]
            return self.aperture.sci_to_idl(x_raw, y_raw)
        else:
            # Use evolved coefficients
            cx = self.prior_coeffs.get("Sci2IdlX_Raw", self.prior_coeffs["Sci2IdlX"])
            cy = self.prior_coeffs.get("Sci2IdlY_Raw", self.prior_coeffs["Sci2IdlY"])
            x_idl = self.fitter.poly.evaluate(cx, x_sci, y_sci)
            y_idl = self.fitter.poly.evaluate(cy, x_sci, y_sci)
            return x_idl, y_idl

    def fit_distortion(self, obs, ref):
        """Fits the polynomial transformation."""
        x_sci = obs["x_SCI"] - self.aperture.XSciRef
        y_sci = obs["y_SCI"] - self.aperture.YSciRef
        weights = self._calculate_weights(obs)

        # Define detector bounds for normalization
        # We keep using self.naxis1 here as it helps fitting stability even if plotting is fixed
        x_min = 0 - self.aperture.XSciRef
        x_max = self.naxis1 - self.aperture.XSciRef
        y_min = 0 - self.aperture.YSciRef
        y_max = self.naxis2 - self.aperture.YSciRef

        ap_params = {"xlim": [x_min, x_max], "ylim": [y_min, y_max]}

        return self.fitter.fit_distortion_step(
            x_sci,
            y_sci,
            ref["x_idl"],
            ref["y_idl"],
            aperture_params=ap_params,
            prior_coeffs=self.prior_coeffs,
            weights=weights,
            damping_factor=self.config.damping_factor,
            use_grid=self.config.use_grid_fitting,
            grid_size=self.config.grid_size,
        )

    def update_reference_ideal_coords(self):
        """updates Ref catalog Ideal coords based on current V2/V3 pointing."""
        fits_file = self.obs_catalog.meta["fits_file"]
        with fits.open(fits_file) as hdul:
            header = hdul[1].header
            ra_ref = header.get("CRVAL1", np.mean(self.obs_catalog["ra"]))
            dec_ref = header.get("CRVAL2", np.mean(self.obs_catalog["dec"]))
            pa_v3 = header.get("PA_V3", 0.0)

        att = pysiaf.utils.rotations.attitude(
            self.aperture.V2Ref, self.aperture.V3Ref, ra_ref, dec_ref, pa_v3
        )
        v2, v3 = pysiaf.utils.rotations.getv2v3(
            att, self.ref_catalog["ra"], self.ref_catalog["dec"]
        )
        x, y = self.aperture.tel_to_idl(v2, v3)
        self.ref_catalog["x_idl"] = x
        self.ref_catalog["y_idl"] = y

    def apply_results_to_aperture(self, results):
        """Updates the in-memory SIAF aperture with new coefficients."""
        deg = self.config.poly_degree
        self.aperture._polynomial_degree = deg
        if hasattr(self.aperture, "Sci2IdlDeg"):
            self.aperture.Sci2IdlDeg = deg

        def set_coeffs(prefix, coeffs_arr):
            idx = 0
            for i in range(deg + 1):
                for j in range(i + 1):
                    key = f"{prefix}{i - j}{j}"
                    try:
                        setattr(self.aperture, key, float(coeffs_arr[idx]))
                    except:
                        pass
                    idx += 1

        set_coeffs("Sci2IdlX", results["Sci2IdlX"])
        set_coeffs("Sci2IdlY", results["Sci2IdlY"])
        set_coeffs("Idl2SciX", results["Idl2SciX"])
        set_coeffs("Idl2SciY", results["Idl2SciY"])

    def check_convergence(self):
        """Checks if RMS is stable between iterations."""
        if len(self.history["rms_x"]) < 2:
            return False
        dx = abs(self.history["rms_x"][-1] - self.history["rms_x"][-2])
        dy = abs(self.history["rms_y"][-1] - self.history["rms_y"][-2])
        return (
            dx < self.config.convergence_tol_mas
            and dy < self.config.convergence_tol_mas
        )

    def finalize(self):
        """Saves final coefficients and generates diagnostic plots."""
        print(f"\nSaving results to {self.config.working_dir}")
        coeff_filename = (
            f"{self.config.file_root}_{self.config.aperture_name}_distortion_coeffs.txt"
        )
        self.write_siaf_table(os.path.join(self.config.res_dir, coeff_filename))

        print("Generating Distortion Models for Visualization...")

        # 1. Setup Grid for Plotting
        grid_n = 20
        # Use simple 2048 grid to match plotting code expectations
        x_edges = np.linspace(0, 2048, grid_n + 1)
        y_edges = np.linspace(0, 2048, grid_n + 1)
        xc = (x_edges[:-1] + x_edges[1:]) / 2.0
        yc = (y_edges[:-1] + y_edges[1:]) / 2.0
        gx, gy = np.meshgrid(xc, yc)
        gx_flat, gy_flat = gx.flatten(), gy.flatten()

        xref = self.aperture.XSciRef
        yref = self.aperture.YSciRef
        gx_cen = gx_flat - xref
        gy_cen = gy_flat - yref

        max_dim = 2048.0

        # 2. Fit & Evaluate BEFORE Model (Standard - Raw)
        debug_data = self.current_results.get("grid_debug", {})
        if debug_data and len(debug_data["x"]) > 0:
            x_raw = debug_data["x"]
            y_raw = debug_data["y"]
            dx_raw = debug_data["dx"]
            dy_raw = debug_data["dy"]

            temp_poly = PolynomialDistortion(degree=self.config.poly_degree)
            cx_before, _ = temp_poly.fit_robust(x_raw, y_raw, dx_raw, scale=max_dim)
            cy_before, _ = temp_poly.fit_robust(x_raw, y_raw, dy_raw, scale=max_dim)

            dx_model_before = temp_poly.evaluate(cx_before, gx_cen, gy_cen)
            dy_model_before = temp_poly.evaluate(cy_before, gx_cen, gy_cen)
        else:
            dx_model_before = np.zeros_like(gx_flat)
            dy_model_before = np.zeros_like(gx_flat)

        # 3. Fit & Evaluate AFTER Model (Final Residuals)
        res_x_mas = self.current_results["residuals_x_mas"]
        res_y_mas = self.current_results["residuals_y_mas"]
        res_x_pix = res_x_mas / 1000.0
        res_y_pix = res_y_mas / 1000.0

        x_clean = self.current_results["x_sci_used"] - xref
        y_clean = self.current_results["y_sci_used"] - yref

        temp_poly = PolynomialDistortion(degree=5)
        cx_after, _ = temp_poly.fit_robust(x_clean, y_clean, res_x_pix, scale=max_dim)
        cy_after, _ = temp_poly.fit_robust(x_clean, y_clean, res_y_pix, scale=max_dim)

        dx_model_after = temp_poly.evaluate(cx_after, gx_cen, gy_cen)
        dy_model_after = temp_poly.evaluate(cy_after, gx_cen, gy_cen)

        # 4. Generate Plots
        plot_label = f"{self.config.file_root}_{self.config.aperture_name}"

        # Removed 'dims=' argument here to fix the error
        plot_comparison_models(
            gx_flat,
            gy_flat,
            dx_model_before,
            dy_model_before,
            dx_model_after,
            dy_model_after,
            self.config.plot_dir,
            plot_label,
        )

        plot_results = self.current_results.copy()
        plot_results["x_sci_used"] = self.current_results["x_sci_used"] + xref
        plot_results["y_sci_used"] = self.current_results["y_sci_used"] + yref

        plot_residuals(plot_results, self.config.plot_dir, plot_label)

        # Removed 'dims=' argument here as well
        plot_trends(plot_results, self.config.plot_dir, plot_label)

    def write_siaf_table(self, filename):
        """Writes the coefficients to a standard text format."""
        res = self.current_results
        poly_deg = self.config.poly_degree

        with open(filename, "w") as f:
            f.write("# NIRISS distortion coefficient file\n")
            f.write("#\n")
            f.write(f"# Aperture: {self.config.aperture_name}\n")
            f.write(f"# Generated {datetime.datetime.utcnow().isoformat()} UTC\n")
            f.write("#\n")
            f.write(
                "AperName , siaf_index , exponent_x , exponent_y ,          Sci2IdlX ,          Sci2IdlY ,          Idl2SciX ,          Idl2SciY\n"
            )

            idx = 0
            for i in range(poly_deg + 1):
                for j in range(i + 1):
                    siaf_idx = f"{i - j}{j}"
                    exp_x = i - j
                    exp_y = j
                    s2i_x = res["Sci2IdlX"][idx]
                    s2i_y = res["Sci2IdlY"][idx]
                    i2s_x = res["Idl2SciX"][idx]
                    i2s_y = res["Idl2SciY"][idx]
                    line = (
                        f" {self.config.aperture_name:7s} ,         {siaf_idx:<2s} ,          {exp_x:1d} ,          {exp_y:1d} , "
                        f"{s2i_x:23.12e} , {s2i_y:23.12e} , {i2s_x:23.12e} , {i2s_y:23.12e}\n"
                    )
                    f.write(line)
                    idx += 1
