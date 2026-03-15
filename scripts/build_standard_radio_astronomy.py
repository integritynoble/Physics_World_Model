#!/usr/bin/env python3
"""
Build standard dataset for radio_astronomy modality.

Source: FIRST Radio Survey (VLA B-configuration 1.4 GHz)
  White et al. 1997 ApJ; sundog.stsci.edu/first/
  Catalog: 944,824 sources at 5 arcsec resolution
  Data: FITS catalog with radio flux measurements

Standard dataset: 10 simulated radio maps (128x128)
  Based on FIRST catalog source positions and flux densities.
  x_true  = clean radio sky model (sparse source image, 128x128)
  y_ideal = dirty image = PSF convolution of x_true (radio synthesis imaging forward model)

Forward model: radio_interferometry dirty image
  y = PSF * x_true  (convolution with synthesized beam)
  PSF = Gaussian beam FWHM ~ 5 arcsec at 1.4 GHz

Note: Full FIRST FITS catalog requires significant parsing;
      we use representative synthetic data based on FIRST source statistics.

Output: datasets/benchmark/radio_astronomy/standard/
"""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path
import struct
import gzip

import h5py
import numpy as np
import requests
from scipy.ndimage import gaussian_filter

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "radio_astronomy" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

# FIRST catalog (69 MB compressed FITS)
FIRST_URL = "https://sundog.stsci.edu/first/catalogs/first_14dec17.fits.gz"
FIRST_FILE = SRCDIR / "first_14dec17.fits.gz"


def _download(url, dest):
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=300, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
        print(f"ok ({dest.stat().st_size//1024//1024} MB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _norm01(a):
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _make_psf_beam(npix=128, fwhm_px=3.0) -> np.ndarray:
    """Create a Gaussian PSF (synthesized beam)."""
    sigma = fwhm_px / (2 * np.sqrt(2 * np.log(2)))
    y, x = np.ogrid[:npix, :npix]
    cy, cx = npix // 2, npix // 2
    psf = np.exp(-((y - cy)**2 + (x - cx)**2) / (2 * sigma**2))
    return (psf / psf.sum()).astype(np.float32)


def _radio_dirty_image(sky: np.ndarray, psf: np.ndarray) -> np.ndarray:
    """Compute dirty image: y = PSF ⊛ x (convolution in image domain)."""
    from scipy.signal import fftconvolve
    dirty = fftconvolve(sky, psf, mode='same').astype(np.float32)
    return dirty


def _load_first_catalog_sample(fits_gz_path: Path, n_sources: int = 1000):
    """
    Read FIRST catalog FITS binary table and extract RA, Dec, flux.
    Returns arrays (ra, dec, flux_mJy) for the first n_sources.
    """
    try:
        from astropy.io import fits
        with gzip.open(str(fits_gz_path), 'rb') as gz:
            data_bytes = gz.read()
        # Parse with astropy from bytes
        from astropy.io.fits import HDUList
        import io
        hdul = fits.open(io.BytesIO(data_bytes))
        # FIRST catalog is a BinTableHDU
        table = None
        for hdu in hdul:
            if hasattr(hdu, 'data') and hdu.data is not None and len(hdu.data) > 100:
                table = hdu.data
                break
        if table is None:
            return None, None, None
        ra    = table['RA'][:n_sources]     # degrees
        dec   = table['DEC'][:n_sources]    # degrees
        flux  = table['FPEAK'][:n_sources]  # mJy/beam (peak flux)
        hdul.close()
        return ra.astype(np.float64), dec.astype(np.float64), flux.astype(np.float64)
    except Exception as e:
        print(f"  catalog parse failed: {e}")
        return None, None, None


def _make_synthetic_radio_sky(rng, npix=128, n_sources=50, field_deg=1.0):
    """Generate a realistic synthetic radio sky (FIRST-like source population)."""
    sky = np.zeros((npix, npix), dtype=np.float32)
    # Source flux distribution: power law dN/dS ~ S^-1.5 (typical radio sky)
    # Minimum flux = 1 mJy, typical range 1-1000 mJy
    n = rng.integers(5, n_sources)
    fluxes = rng.pareto(1.5, n) + 1.0  # mJy (power law + 1 mJy floor)
    fluxes = np.clip(fluxes, 1, 500).astype(np.float32)
    fluxes /= fluxes.max()  # normalize
    # Random positions
    xpos = rng.uniform(5, npix - 5, n).astype(int)
    ypos = rng.uniform(5, npix - 5, n).astype(int)
    for xi, yi, fi in zip(xpos, ypos, fluxes):
        sky[yi, xi] += fi
    return sky


def build():
    print("=" * 60)
    print("Radio Astronomy standard dataset builder (FIRST survey)")
    print("=" * 60)

    print("\n[1] Attempting FIRST catalog download...")
    catalog_ok = _download(FIRST_URL, FIRST_FILE)

    # Try to load catalog
    ra, dec, flux = None, None, None
    if catalog_ok:
        print("  Parsing FIRST catalog...")
        ra, dec, flux = _load_first_catalog_sample(FIRST_FILE, n_sources=5000)
        if ra is not None:
            print(f"  Loaded {len(ra)} sources from FIRST catalog")

    # Build 10 samples
    print("\n[2] Building 10 radio sky samples (128x128)...")
    npix = 128
    psf = _make_psf_beam(npix=npix, fwhm_px=3.0)  # 3px FWHM = ~5 arcsec at 1.4 GHz
    records = []
    rng = np.random.default_rng(seed=42)

    for idx in range(10):
        if ra is not None and len(ra) >= 500:
            # Use real FIRST source positions for this sky field
            # Select a random subset of nearby sources and project to image
            start = idx * 500
            end = start + 500
            ra_sub  = ra[start:end]
            dec_sub = dec[start:end]
            flux_sub = flux[start:end]

            # Project RA/Dec to pixel coordinates (simple linear projection)
            ra_min, ra_max   = ra_sub.min(),  ra_sub.max()
            dec_min, dec_max = dec_sub.min(), dec_sub.max()
            if ra_max - ra_min < 1e-6 or dec_max - dec_min < 1e-6:
                sky = _make_synthetic_radio_sky(rng, npix)
            else:
                sky = np.zeros((npix, npix), dtype=np.float32)
                xpix = ((ra_sub - ra_min) / (ra_max - ra_min) * (npix - 1)).astype(int)
                ypix = ((dec_sub - dec_min) / (dec_max - dec_min) * (npix - 1)).astype(int)
                fnorm = flux_sub / (flux_sub.max() + 1e-8)
                for xi, yi, fi in zip(xpix, ypix, fnorm):
                    xi = max(0, min(xi, npix-1))
                    yi = max(0, min(yi, npix-1))
                    sky[yi, xi] += fi
            source_type = "first_catalog"
        else:
            sky = _make_synthetic_radio_sky(rng, npix, seed=None)
            source_type = "synthetic_first_like"

        sky = _norm01(sky)
        dirty = _radio_dirty_image(sky, psf)
        dirty = _norm01(dirty)
        records.append((idx, sky, dirty, source_type))
        print(f"  sample {idx:02d}: sky={sky.shape} n_bright={(sky>0.5).sum()}, "
              f"dirty range=[{dirty.min():.3f},{dirty.max():.3f}] ({source_type})")

    # Write HDF5 files
    print(f"\n[3] Writing {len(records)} HDF5 files...")
    out_files = []
    for idx, sky, dirty, stype in records:
        h5_path = OUTDIR / f"standard_radio_astronomy_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=sky,   compression="gzip")
            f.create_dataset("y_ideal", data=dirty, compression="gzip")
            f.create_dataset("psf",     data=psf,   compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]     = "dirty_image_convolution"
            hp.attrs["beam_fwhm_px"] = 3.0
            hp.attrs["frequency_GHz"] = 1.4
            hp.attrs["image_size"]   = npix
            hp.attrs["noise"]        = "none"
            hp.attrs["mismatch"]     = "none"
            hp.attrs["source_type"]  = stype

            meta = f.create_group("metadata")
            meta.attrs["source"]     = "https://sundog.stsci.edu/first/"
            meta.attrs["doi"]        = "10.1086/303559"
            meta.attrs["citation"]   = "White et al., The FIRST Survey, ApJ 1997"
            meta.attrs["date_built"] = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)

    meta_json = {
        "modality": "radio_astronomy",
        "canonical_dataset": "FIRST Radio Survey (VLA 1.4 GHz)",
        "source": "https://sundog.stsci.edu/first/",
        "citation": "White et al., FIRST Radio Survey, ApJ 1997",
        "doi": "10.1086/303559",
        "license": "Public domain (NASA/NRAO)",
        "n_samples": len(records),
        "x_true_shape": [npix, npix],
        "y_ideal_shape": [npix, npix],
        "note": "x_true = clean radio sky; y_ideal = dirty image (PSF convolution)",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/standard/",
        "status": "done",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)
    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "radio_astronomy",
            "operator": "dirty_image_convolution",
            "beam_fwhm_px": 3.0,
            "frequency_GHz": 1.4,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)} samples built")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
