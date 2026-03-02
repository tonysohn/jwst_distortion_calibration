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
        self.converged = False
        self.total_iters = 0
        self.n_matches_initial = 0

    def prepare_and_load_catalogs(self, xymq_file, fits_file, ref_file):
        """Loads observed and reference catalogs, extracting header info."""
        # print(f"\n[1] Loading Obs Catalog from {xymq_file}...")
        self.obs_catalog = prepare_obs_catalog(
            xymq_file,
            fits_file,
            q_max=self.config.obs_q_max,
            snr_min=self.config.obs_snr_min,
        )

        # Read image dimensions
        with fits.open(fits_file) as hdul:
            self.naxis1 = hdul[1].header.get("NAXIS1", 2048)
            self.naxis2 = hdul[1].header.get("NAXIS2", 2048)
            # print(f"    Detected Image Dimensions: {self.naxis1} x {self.naxis2}")

        # print(f"\n[2] Loading Ref Catalog from {ref_file}...")
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

        assert self.ref_catalog is not None, "Reference catalog failed to load."

        self.converged = False
        self.total_iters = 0

        print(f"\n{'=' * 60}\nCALIBRATING: {self.config.file_root}\n{'=' * 60}")
        print("Iterations: ", end="", flush=True)

        for i in range(self.config.max_iters):
            self.total_iters = i + 1
            print(f"{self.total_iters}.. ", end="", flush=True)

            self.update_reference_ideal_coords()
            current_tolerance = (
                self.config.initial_tolerance_arcsec
                if i == 0
                else self.config.pos_tolerance_arcsec
            )

            if i == 0:
                s1_obs, s1_ref, info1 = match_with_pointing_prior(
                    self.obs_catalog,
                    self.ref_catalog,
                    n_bright_obs=self.config.n_bright_obs,
                    pos_tolerance_arcsec=0.5,
                    verbose=False,
                )
                self.n_matches_initial = info1.get("n_matches", 0)

                x_idl_all, y_idl_all = self.project_to_ideal(self.obs_catalog)
                x_idl_s1, y_idl_s1 = self.project_to_ideal(s1_obs)
                dx, dy = (
                    np.median(s1_ref["x_idl"] - x_idl_s1),
                    np.median(s1_ref["y_idl"] - y_idl_s1),
                )
                x_idl_aligned, y_idl_aligned = x_idl_all + dx, y_idl_all + dy
            else:
                x_idl_aligned, y_idl_aligned = self.project_to_ideal(self.obs_catalog)

            obs_matched, ref_matched, _ = match_in_ideal_frame(
                self.obs_catalog,
                self.ref_catalog,
                x_idl_obs=x_idl_aligned,
                y_idl_obs=y_idl_aligned,
                pos_tolerance_arcsec=current_tolerance,
                verbose=False,
            )

            if len(obs_matched) < 50:
                print("Failed (Too few matches).")
                break

            results = self.fit_distortion(obs_matched, ref_matched)
            self.prior_coeffs, self.current_results = results, results

            self.history["rms_x"].append(results["rms_x"])
            self.history["rms_y"].append(results["rms_y"])
            self.history["n_stars"].append(results["n_stars"])

            if self.check_convergence():
                self.converged = True
                break

            self.apply_results_to_aperture(results)

        # =====================================================================
        # POST-CONVERGENCE EXACT MATHEMATICAL ROTATION ALIGNMENT
        # =====================================================================
        res = self.current_results

        # Find exact residual rotation
        c01_x, c01_y = res["Sci2IdlX"][2], res["Sci2IdlY"][2]
        theta = np.arctan2(c01_x, c01_y)

        # 1. Rotate Forward Model Coefficients
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        new_s2i_x = res["Sci2IdlX"] * cos_t - res["Sci2IdlY"] * sin_t
        new_s2i_y = res["Sci2IdlX"] * sin_t + res["Sci2IdlY"] * cos_t

        # Hard-force machine zero on SIAF index 01
        new_s2i_x[2] = 0.0

        res["Sci2IdlX"], res["Sci2IdlY"] = new_s2i_x, new_s2i_y

        # 2. Perfectly Regenerate Inverse Model (Idl2Sci)
        grid_n = self.config.grid_size
        x_edges = np.linspace(0, self.naxis1, grid_n + 1)
        y_edges = np.linspace(0, self.naxis2, grid_n + 1)
        xc, yc = (x_edges[:-1] + x_edges[1:]) / 2.0, (y_edges[:-1] + y_edges[1:]) / 2.0
        xg, yg = np.meshgrid(xc, yc)

        x_sci_flat = xg.flatten() - self.aperture.XSciRef
        y_sci_flat = yg.flatten() - self.aperture.YSciRef

        x_idl_grid = self.fitter.poly.evaluate(new_s2i_x, x_sci_flat, y_sci_flat)
        y_idl_grid = self.fitter.poly.evaluate(new_s2i_y, x_sci_flat, y_sci_flat)

        i2s_x, _ = self.fitter.poly.fit_robust(
            x_idl_grid, y_idl_grid, x_sci_flat, scale=2048.0
        )
        i2s_y, _ = self.fitter.poly.fit_robust(
            x_idl_grid, y_idl_grid, y_sci_flat, scale=2048.0
        )

        res["Idl2SciX"], res["Idl2SciY"] = i2s_x, i2s_y

        self.current_results = res
        self.apply_results_to_aperture(res)
        # =====================================================================

        print("Done.")
        if self.converged:
            print(f"  >>> Convergence achieved in {self.total_iters} iterations.")
        else:
            print(
                f"  >>> Finished: Maximum iterations ({self.config.max_iters}) reached."
            )

        print(
            f"  >>> Final RMS: X={res['rms_x']:.3f}, Y={res['rms_y']:.3f} mas (N={res['n_stars']})"
        )

        self.finalize()

    def project_to_ideal(self, catalog):
        """Projects science pixels to Ideal frame using current best solution."""
        x_sci = catalog["x_SCI"] - self.aperture.XSciRef
        y_sci = catalog["y_SCI"] - self.aperture.YSciRef
        if self.prior_coeffs is None:
            x_raw = catalog["x_SCI"]
            y_raw = catalog["y_SCI"]
            return self.aperture.sci_to_idl(x_raw, y_raw)
        else:
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
            roll_ref = header.get("ROLL_REF", header.get("PA_V3", 0.0))

        att = pysiaf.utils.rotations.attitude(
            self.aperture.V2Ref, self.aperture.V3Ref, ra_ref, dec_ref, roll_ref
        )
        v2, v3 = pysiaf.utils.rotations.getv2v3(
            att, self.ref_catalog["ra"], self.ref_catalog["dec"]
        )
        x, y = self.aperture.tel_to_idl(v2, v3)

        va_scale = self.obs_catalog.meta.get("va_scale", 1.0)
        self.ref_catalog["x_idl"] = x / va_scale
        self.ref_catalog["y_idl"] = y / va_scale

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
        """Checks if RMS is stable between iterations and prints debug info."""
        if len(self.history["rms_x"]) < 2:
            return False

        dx = abs(self.history["rms_x"][-1] - self.history["rms_x"][-2])
        dy = abs(self.history["rms_y"][-1] - self.history["rms_y"][-2])

        return (
            dx < self.config.convergence_tol_mas
            and dy < self.config.convergence_tol_mas
        )

    def finalize(self):
        """Saves final coefficients, summary, and generates diagnostic plots."""
        print(f"\nSaving results to {self.config.working_dir}")
        coeff_filename = (
            f"{self.config.file_root}_{self.config.aperture_name}_distortion_coeffs.txt"
        )
        self.write_siaf_table(os.path.join(self.config.res_dir, coeff_filename))

        # New Summary File
        summary_filename = f"{self.config.file_root}_summary.txt"
        self.write_summary_file(os.path.join(self.config.res_dir, summary_filename))

        print("Generating Distortion Models for Visualization...")

        grid_n = 20
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

            # POINT 2: Use dynamic degree from config [cite: 193]
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

        # POINT 2: Use dynamic degree from config [cite: 193]
        temp_poly = PolynomialDistortion(degree=self.config.poly_degree)
        cx_after, _ = temp_poly.fit_robust(x_clean, y_clean, res_x_pix, scale=max_dim)
        cy_after, _ = temp_poly.fit_robust(x_clean, y_clean, res_y_pix, scale=max_dim)

        dx_model_after = temp_poly.evaluate(cx_after, gx_cen, gy_cen)
        dy_model_after = temp_poly.evaluate(cy_after, gx_cen, gy_cen)

        plot_label = f"{self.config.file_root}_{self.config.aperture_name}"

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
        plot_trends(plot_results, self.config.plot_dir, plot_label)

    def write_siaf_table(self, filename):
        """Writes the coefficients to a standard text format."""
        res = self.current_results
        poly_deg = self.config.poly_degree

        date_obs = self.obs_catalog.meta.get("date_obs", "unknown")
        filt = self.obs_catalog.meta.get("filter", "unknown")
        fits_name = os.path.basename(self.obs_catalog.meta.get("fits_file", "unknown"))

        with open(filename, "w") as f:
            f.write(f"# {self.config.instrument} distortion coefficient file\n")
            f.write(f"# Source file: {fits_name}\n")
            f.write(f"# Aperture: {self.config.aperture_name}\n")
            f.write(f"# Filter/Pupil: {filt}\n")
            f.write(f"# Observation Date: {date_obs}\n")
            f.write(f"# Generated {datetime.datetime.utcnow().isoformat()} UTC\n")
            f.write("# by tsohn\n")
            f.write("#\n")

            w_aper, w_idx, w_exp, w_val = 10, 10, 10, 23
            headers = [
                "AperName",
                "siaf_index",
                "exponent_x",
                "exponent_y",
                "Sci2IdlX",
                "Sci2IdlY",
                "Idl2SciX",
                "Idl2SciY",
            ]

            header_line = (
                f"{headers[0]:>{w_aper}} , {headers[1]:>{w_idx}} , {headers[2]:>{w_exp}} , {headers[3]:>{w_exp}} , "
                f"{headers[4]:>{w_val}} , {headers[5]:>{w_val}} , {headers[6]:>{w_val}} , {headers[7]:>{w_val}}\n"
            )
            f.write(header_line)

            idx = 0
            for i in range(poly_deg + 1):
                for j in range(i + 1):
                    siaf_idx = f"{i}{j}"  # Correct Degree-Index format
                    exp_x = i - j
                    exp_y = j

                    s2i_x = res["Sci2IdlX"][idx]
                    s2i_y = res["Sci2IdlY"][idx]
                    i2s_x = res["Idl2SciX"][idx]
                    i2s_y = res["Idl2SciY"][idx]

                    line = (
                        f" {self.config.aperture_name:>{w_aper - 1}} ,         {siaf_idx:<2s} ,          {exp_x:1d} ,          {exp_y:1d} , "
                        f"{s2i_x:23.12e} , {s2i_y:23.12e} , {i2s_x:23.12e} , {i2s_y:23.12e}\n"
                    )
                    f.write(line)
                    idx += 1

    def write_summary_file(self, filename):
        """Writes a concise diagnostic summary to an ASCII file."""
        meta = self.obs_catalog.meta
        res = self.current_results

        rms_total = np.sqrt(res["rms_x"] ** 2 + res["rms_y"] ** 2)

        with open(filename, "w") as f:
            f.write("JWST DISTORTION CALIBRATION SUMMARY\n")
            f.write("=" * 40 + "\n")
            f.write(f"File:           {os.path.basename(meta['fits_file'])}\n")
            f.write(f"Instrument:     {meta['instrument']}\n")
            f.write(f"Aperture:       {meta['apername']}\n")
            f.write(f"Filter/Pupil:   {meta['filter']}\n")
            f.write(f"Obs Date:       {meta.get('date_obs', 'N/A')}\n")
            f.write("-" * 40 + "\n")
            f.write(f"VA_SCALE Used:  {meta.get('va_scale', 1.0):.10f}\n")
            f.write(f"Poly Degree:    {self.config.poly_degree}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Iterations:     {self.total_iters}\n")
            f.write(f"Converged:      {self.converged}\n")
            f.write("-" * 40 + "\n")
            f.write(f"Stars in Cat:   {len(self.obs_catalog)}\n")
            f.write(f"Stars Matched:  {self.n_matches_initial} (initial)\n")
            f.write(f"Stars in Fit:   {res['n_stars']} (after 2D rejection)\n")
            f.write("-" * 40 + "\n")
            f.write(f"RMS X (mas):    {res['rms_x']:.3f}\n")
            f.write(f"RMS Y (mas):    {res['rms_y']:.3f}\n")
            f.write(f"RMS Total:      {rms_total:.3f}\n")
            f.write("=" * 40 + "\n")
        print(f"Summary saved to: {filename}")
