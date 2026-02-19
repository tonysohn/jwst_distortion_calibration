"""
JWST Distortion Data Preparation Module
Updates:
- Caches photutils results to .xymq files.
- Auto-loads .xymq if available to save time.
"""

import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS
from scipy.spatial import cKDTree

# Import new photometry module
try:
    from distortion_photometry import measure_sources_photutils
except ImportError:
    measure_sources_photutils = None


def load_xymq_file(xymq_file: str) -> Table:
    """Load XYMQ catalog from file."""
    data = []
    if not os.path.exists(xymq_file):
        raise FileNotFoundError(f"XYMQ file not found: {xymq_file}")

    with open(xymq_file, "r") as f:
        for line in f:
            if line.startswith("#") or line.strip() == "":
                continue
            parts = line.split()
            if len(parts) >= 4:
                try:
                    x, y, m, q = (
                        float(parts[0]),
                        float(parts[1]),
                        float(parts[2]),
                        float(parts[3]),
                    )
                    data.append([x, y, m, q])
                except ValueError:
                    continue
    return Table(rows=data, names=["x", "y", "m", "q"])


def save_xymq_file(catalog: Table, filename: str):
    """Save catalog to XYMQ format (x y m q)."""
    with open(filename, "w") as f:
        f.write("# x y m q\n")
        for row in catalog:
            f.write(f"{row['x']:.3f}  {row['y']:.3f}  {row['m']:.4f}  {row['q']:.3f}\n")
    print(f"    Cached XYMQ saved to: {filename}")


def compute_flux_and_snr(instrumental_mag, exptime, photmjsr=1.0):
    flux = 10 ** (-0.4 * instrumental_mag) / photmjsr * exptime
    snr = np.sqrt(flux)
    return flux, snr


def instrumental_to_ab_magnitude(m_inst, pixar_sr):
    zp_ab = -6.10 - 2.5 * np.log10(pixar_sr)
    return m_inst + zp_ab


def compute_isolation(catalog):
    coords = np.column_stack([catalog["ra"], catalog["dec"]])
    tree = cKDTree(coords)
    distances, _ = tree.query(coords, k=2)
    return distances[:, 1] * 3600.0


def prepare_obs_catalog(
    xymq_file: Optional[str],
    fits_file: str,
    q_min: float = 0.001,
    q_max: float = 0.3,
    snr_min: float = 40.0,
    photmjsr: Optional[float] = None,
    sort_by_magnitude: bool = True,
    source_method: str = "xymq",
) -> Table:
    """
    Prepare observed catalog with caching support for photutils.
    """

    # 1. Load Catalog
    if source_method == "photutils":
        # Define cache filename based on FITS file
        cache_file = fits_file.replace(".fits", ".xymq")

        if os.path.exists(cache_file):
            print(f"Loading cached catalog: {cache_file}")
            catalog = load_xymq_file(cache_file)
        else:
            print(f"Running source extraction (Method: PHOTUTILS)...")
            if not measure_sources_photutils:
                raise ImportError("distortion_photometry module not found.")

            # Determine instrument
            with fits.open(fits_file) as hdul:
                inst = hdul[0].header.get("INSTRUME", "FGS")

            # Run extraction
            # This will also save the diagnostic plot if save_plot=True
            catalog = measure_sources_photutils(
                fits_file, instrument=inst, save_plot=True
            )

            # Save cache
            if len(catalog) > 0:
                save_xymq_file(catalog, cache_file)

    else:
        print(f"Loading observed catalog: {xymq_file}")
        catalog = load_xymq_file(xymq_file)

    # 2. Check FITS file
    if not os.path.exists(fits_file):
        raise FileNotFoundError(f"FITS file not found: {fits_file}")

    # 3. Read FITS header
    with fits.open(fits_file) as hdul:
        header0 = hdul[0].header
        header1 = hdul[1].header
        instname = header0["INSTRUME"]
        apername = header0["APERNAME"]

        if instname.upper() == "NIRISS":
            filtname = (
                header0["PUPIL"] if "CLEAR" in header0["FILTER"] else header0["FILTER"]
            )
        elif instname.upper() == "NIRCAM":
            filtname = header0["FILTER"]
        elif instname.upper() == "FGS":
            filtname = apername
        else:
            filtname = "UNKNOWN"

        exptime = float(header0["TMEASURE"])
        wcs = WCS(header1)
        if photmjsr is None:
            photmjsr = header1.get("PHOTMJSR", 1.0)
        pixar_sr = header1.get("PIXAR_SR", 1.1e-13)

    # 4. Compute Flux/SNR/AB
    flux, snr = compute_flux_and_snr(catalog["m"], exptime, photmjsr)
    mag_ab = instrumental_to_ab_magnitude(catalog["m"], pixar_sr)

    # 5. Filter
    mask = (catalog["q"] > q_min) & (catalog["q"] < q_max) & (snr > snr_min)

    print(f"  Total sources: {len(catalog)}")
    print(f"  After filters (q, snr): {np.sum(mask)}")

    catalog = catalog[mask]
    flux = flux[mask]
    snr = snr[mask]
    mag_ab = mag_ab[mask]

    # 6. WCS: Pixel -> Sky
    sky_coords = wcs.pixel_to_world(catalog["x"] - 1, catalog["y"] - 1)

    # 7. Create Output
    obs_catalog = Table(
        {
            "x_SCI": catalog["x"],
            "y_SCI": catalog["y"],
            "ra": sky_coords.ra.deg,
            "dec": sky_coords.dec.deg,
            "mag_inst": catalog["m"],
            "mag_ab": mag_ab,
            "flux": flux,
            "snr": snr,
            "q": catalog["q"],
        }
    )

    # 8. Metadata
    obs_catalog.meta["xymq_file"] = xymq_file if xymq_file else "N/A"
    obs_catalog.meta["fits_file"] = fits_file
    obs_catalog.meta["instrument"] = instname
    obs_catalog.meta["apername"] = apername
    obs_catalog.meta["filter"] = filtname
    obs_catalog.meta["source_method"] = source_method

    # 9. Isolation
    obs_catalog["isolation"] = compute_isolation(obs_catalog)

    # 10. Sort
    if sort_by_magnitude:
        obs_catalog = obs_catalog[np.argsort(obs_catalog["mag_ab"])]

    print(f"  Final catalog size: {len(obs_catalog)}")
    return obs_catalog


def prepare_ref_catalog(
    ref_catalog_file: str,
    obs_catalog: Table,
    mag_column: str = None,
    mag_range: Tuple[float, float] = None,
    mag_buffer: float = 2.0,
    apply_pm: bool = True,
    target_epoch: float = 2026.0,
    buffer_arcsec: float = 30.0,
) -> Table:
    """Prepare reference catalog."""
    print("\nPreparing reference catalog...")
    ref_cat = Table.read(ref_catalog_file)

    # 1. Column standardization (RA/Dec)
    for col in ref_cat.colnames:
        if col.lower() in ["ra_deg", "ra"]:
            ref_cat.rename_column(col, "ra")
        if col.lower() in ["dec_deg", "dec"]:
            ref_cat.rename_column(col, "dec")

    # 2. Magnitude column
    if mag_column is None:
        inst = obs_catalog.meta.get("instrument", "FGS").upper()
        filt = obs_catalog.meta.get("filter", "").lower()
        if inst == "FGS":
            mag_column = (
                "fgs1_magnitude"
                if "FGS1" in obs_catalog.meta.get("apername", "")
                else "fgs2_magnitude"
            )
        elif inst == "NIRISS":
            mag_column = f"niriss_{filt}_magnitude"
        else:
            mag_column = "j_magnitude"

    print(f"  Using reference magnitude: {mag_column}")
    ref_cat["mag_ref"] = ref_cat[mag_column]

    # 3. Spatial Filter
    buffer_deg = buffer_arcsec / 3600.0
    ra_min, ra_max = (
        obs_catalog["ra"].min() - buffer_deg,
        obs_catalog["ra"].max() + buffer_deg,
    )
    dec_min, dec_max = (
        obs_catalog["dec"].min() - buffer_deg,
        obs_catalog["dec"].max() + buffer_deg,
    )

    spatial_mask = (
        (ref_cat["ra"] > ra_min)
        & (ref_cat["ra"] < ra_max)
        & (ref_cat["dec"] > dec_min)
        & (ref_cat["dec"] < dec_max)
    )
    ref_cat = ref_cat[spatial_mask]

    # 4. Proper Motion
    if apply_pm and "pmra" in ref_cat.colnames:
        dt = target_epoch - 2015.5
        ref_cat["ra"] += (
            (ref_cat["pmra"] / 3.6e6) * dt / np.cos(np.radians(ref_cat["dec"]))
        )
        ref_cat["dec"] += (ref_cat["pmdec"] / 3.6e6) * dt

    print(f"  Loaded {len(ref_cat)} reference stars.")
    return ref_cat
