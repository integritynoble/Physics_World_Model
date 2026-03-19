#!/usr/bin/env python3
"""
Build standard dataset for gravitational_wave modality.

Source: LIGO/Virgo GWOSC (Gravitational Wave Open Science Center)
  10 famous GW events from O1, O2, O3 observing runs
  Data: H1 detector 4KHz HDF5 strain files, 32-second segments

Standard dataset: 10 samples (one per event)
  x_true  = whitened strain (4096 samples = 1 second at 4096 Hz)
             centered on merger time
  y_ideal = x_true (identity: ideal noiseless measurement is the signal itself)

Forward model: GW detector response H maps (h+, hx) to strain d(t) = h(t) + n(t)
  In standard dataset: x_true = whitened d(t) from calibrated O-run data

Events (famous detections):
  GW150914, GW151226, GW170104, GW170814, GW170817,
  GW190412, GW190521, GW190814, GW200105_162426, GW200115_042309

Output: datasets/benchmark/gravitational_wave/standard/standard_gravitational_wave_{N}.h5
"""
from __future__ import annotations

import json
import sys
import urllib.request
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import requests

ROOT   = Path(__file__).resolve().parent.parent
OUTDIR = ROOT / "datasets" / "benchmark" / "gravitational_wave" / "standard"
SRCDIR = OUTDIR / "_raw_src"

OUTDIR.mkdir(parents=True, exist_ok=True)
SRCDIR.mkdir(parents=True, exist_ok=True)

GWOSC_API = "https://gwosc.org/eventapi/json/allevents/"

# 10 famous events: (name, GPS merger time, catalog, version)
EVENTS = [
    ("GW150914",          1126259462.4, "GWTC-1",  "GW150914-v3"),
    ("GW151226",          1135136350.6, "GWTC-1",  "GW151226-v2"),
    ("GW170104",          1167559936.6, "GWTC-1",  "GW170104-v2"),
    ("GW170814",          1186741861.5, "GWTC-1",  "GW170814-v3"),
    ("GW170817",          1187008882.4, "GWTC-1",  "GW170817-v3"),
    ("GW190412",          1239082262.2, "GWTC-2",  "GW190412-v4"),
    ("GW190521",          1242442967.4, "GWTC-2",  "GW190521_074359-v2"),
    ("GW190814",          1249852257.0, "GWTC-2",  "GW190814-v2"),
    ("GW200105_162426",   1262276684.0, "GWTC-3",  "GW200105_162426-v2"),
    ("GW200115_042309",   1263107128.0, "GWTC-3",  "GW200115_042309-v2"),
]


def _get_strain_url(event_key: str, detector: str = "H1", rate: int = 4096) -> str | None:
    """Use GWOSC event API to find the HDF5 strain file URL."""
    # Build event API URL from the key
    # Keys like "GW150914-v3" or "GW190521_074359-v2"
    # Try known catalog paths
    catalogs = ["GWTC-1", "GWTC-2", "GWTC-3", "GWTC-2.1-confident",
                "GWTC-3-confident", "O1_O2-Preliminary"]

    # First, search in the allevents API
    r = requests.get(GWOSC_API, timeout=15)
    if r.status_code != 200:
        return None
    events_data = r.json().get("events", {})

    # Find the event
    if event_key not in events_data:
        # Try fuzzy search
        name = event_key.split("-")[0]  # e.g. "GW150914" from "GW150914-v3"
        candidates = [k for k in events_data if name in k]
        if not candidates:
            return None
        event_key = sorted(candidates)[-1]  # take latest version

    json_url = events_data[event_key].get("jsonurl", "")
    if not json_url:
        return None

    # Fetch event detail
    r2 = requests.get(json_url, timeout=15)
    if r2.status_code != 200:
        return None

    detail = r2.json().get("events", {})
    for k, e in detail.items():
        strain_files = e.get("strain", [])
        for sf in strain_files:
            if (sf.get("detector", "") == detector and
                    sf.get("sampling_rate", 0) == rate and
                    sf.get("url", "").endswith(".hdf5")):
                return sf["url"]
    return None


def _download_strain(url: str, dest: Path) -> bool:
    if dest.exists():
        print(f"  [cache] {dest.name}")
        return True
    print(f"  downloading {dest.name} ({url.split('/')[-1]}) ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=180, stream=True)
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=65536):
                f.write(chunk)
        print(f"ok ({dest.stat().st_size // 1024} KB)")
        return True
    except Exception as e:
        print(f"FAILED: {e}")
        if dest.exists():
            dest.unlink()
        return False


def _norm01(a: np.ndarray) -> np.ndarray:
    lo, hi = float(a.min()), float(a.max())
    return ((a - lo) / (hi - lo + 1e-12)).astype(np.float32)


def _whiten_strain(strain: np.ndarray, fs: int = 4096,
                   fmin: float = 20.0, fmax: float = 1000.0) -> np.ndarray:
    """Simple whitening: divide by sqrt(PSD) in frequency domain, bandpass [fmin,fmax]."""
    N = len(strain)
    S = np.fft.rfft(strain)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)

    # Estimate PSD from this segment using Welch-like estimate
    # Simple approach: smooth the magnitude spectrum
    from scipy.ndimage import uniform_filter1d
    psd_est = np.abs(S) ** 2
    psd_smooth = uniform_filter1d(psd_est, size=max(1, N // 256))
    psd_smooth = np.maximum(psd_smooth, 1e-30)

    # Whiten
    S_white = S / np.sqrt(psd_smooth)

    # Bandpass
    bp_mask = (freqs >= fmin) & (freqs <= fmax)
    S_white[~bp_mask] = 0.0

    whitened = np.fft.irfft(S_white, n=N).astype(np.float64)
    return whitened


def _extract_segment(hdf5_path: Path, gps_merger: float,
                     fs: int = 4096, n_samples: int = 4096) -> np.ndarray | None:
    """Extract n_samples around merger time from GWOSC HDF5 file."""
    try:
        with h5py.File(str(hdf5_path), "r") as f:
            # GWOSC HDF5 structure: /strain/Strain, /meta/GPSstart, /meta/Duration
            strain_ds = None
            for path in ["/strain/Strain", "/Strain", "strain"]:
                if path in f:
                    strain_ds = f[path]
                    break
            if strain_ds is None:
                print(f"    no strain dataset found, keys={list(f.keys())}")
                return None

            strain = strain_ds[:]
            try:
                gps_start = float(f["/meta/GPSstart"][()])
            except Exception:
                try:
                    gps_start = float(f["meta"]["GPSstart"][()])
                except Exception:
                    gps_start = float(f.attrs.get("GPSstart", 0))

            # Find sample index for merger time
            t_merger_offset = gps_merger - gps_start
            i_merger = int(t_merger_offset * fs)
            half = n_samples // 2
            i_start = max(0, i_merger - half)
            i_end   = i_start + n_samples

            if i_end > len(strain):
                i_end = len(strain)
                i_start = max(0, i_end - n_samples)

            segment = strain[i_start:i_end].astype(np.float64)
            if len(segment) < n_samples:
                # Pad with zeros
                segment = np.pad(segment, (0, n_samples - len(segment)))

            return segment

    except Exception as e:
        print(f"    extract failed: {e}")
        return None


def build():
    print("=" * 60)
    print("Gravitational Wave standard dataset builder (GWOSC)")
    print("=" * 60)

    records = []
    for idx, (name, gps, catalog, key) in enumerate(EVENTS):
        print(f"\n[{idx+1}/10] {name} (GPS={gps:.1f})")

        # Get strain file URL
        print("  Finding strain URL...")
        url = _get_strain_url(key, detector="H1", rate=4096)
        if url is None:
            print(f"  Could not find strain URL for {key}")
            # Try L1 detector as fallback
            url = _get_strain_url(key, detector="L1", rate=4096)
            if url is None:
                print(f"  Skipping {name}: no strain URL found")
                continue
            det = "L1"
        else:
            det = "H1"
        print(f"  URL: ...{url[-60:]}")

        # Download
        fname = url.split("/")[-1]
        dest = SRCDIR / fname
        if not _download_strain(url, dest):
            continue

        # Extract segment
        print(f"  Extracting 1s segment around merger...")
        segment = _extract_segment(dest, gps, fs=4096, n_samples=4096)
        if segment is None:
            continue

        # Whiten
        whitened = _whiten_strain(segment, fs=4096, fmin=20.0, fmax=1024.0)
        x_true = _norm01(whitened).astype(np.float32)  # (4096,)
        y_ideal = x_true.copy()  # identity forward model

        records.append((name, gps, det, x_true, y_ideal))
        print(f"  OK: x_true={x_true.shape}, range=[{x_true.min():.3f},{x_true.max():.3f}]")

    if not records:
        print("\nERROR: no samples built")
        return False

    # Write HDF5 files
    print(f"\n[Writing] {len(records)} HDF5 files...")
    out_files = []
    for idx, (name, gps, det, x_true, y_ideal) in enumerate(records):
        h5_path = OUTDIR / f"standard_gravitational_wave_{idx:02d}.h5"
        with h5py.File(str(h5_path), "w") as f:
            f.create_dataset("x_true",  data=x_true,  compression="gzip")
            f.create_dataset("y_ideal", data=y_ideal, compression="gzip")

            hp = f.create_group("H_params")
            hp.attrs["operator"]      = "gw_detector"
            hp.attrs["detector"]      = det
            hp.attrs["sampling_rate"] = 4096
            hp.attrs["n_samples"]     = 4096
            hp.attrs["fmin_hz"]       = 20.0
            hp.attrs["fmax_hz"]       = 1024.0
            hp.attrs["whitened"]      = True
            hp.attrs["noise"]         = "none (whitened real data)"
            hp.attrs["mismatch"]      = "none"

            meta = f.create_group("metadata")
            meta.attrs["event_name"]  = name
            meta.attrs["gps_merger"]  = gps
            meta.attrs["source"]      = "https://gwosc.org"
            meta.attrs["doi"]         = "10.3847/2041-8213/ab0ec7"
            meta.attrs["citation"]    = "LIGO/Virgo, GWTC, PRX 2019"
            meta.attrs["date_built"]  = datetime.now().strftime("%Y-%m-%d")

        out_files.append(h5_path.name)
        print(f"  wrote {h5_path.name}  ({name})")

    meta_json = {
        "modality": "gravitational_wave",
        "canonical_dataset": "GWOSC - 10 famous GW events (O1-O3)",
        "source": "https://gwosc.org",
        "citation": "LIGO/Virgo, GWTC-1/2/3, PRX 2019-2023",
        "doi": "10.3847/2041-8213/ab0ec7",
        "license": "CC-BY-4.0",
        "n_samples": len(records),
        "x_true_shape": [4096],
        "y_ideal_shape": [4096],
        "events": [r[0] for r in records],
        "note": "Whitened strain 1s segments around merger, 4096 Hz, H1 detector",
        "date_built": datetime.now().strftime("%Y-%m-%d"),
        "gcs_path": "gs://pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/standard/",
        "status": "done" if len(records) >= 8 else "partial",
        "files": out_files,
    }
    with open(OUTDIR / "metadata.json", "w") as f:
        json.dump(meta_json, f, indent=2)

    with open(OUTDIR / "spec.json", "w") as f:
        json.dump({
            "modality": "gravitational_wave",
            "operator": "gw_detector",
            "sampling_rate_hz": 4096,
            "n_samples": 4096,
            "whitened": True,
            "fmin_hz": 20.0,
            "fmax_hz": 1024.0,
            "noise": "none",
            "mismatch": "none",
            "split": "standard",
        }, f, indent=2)

    print(f"\nDone: {len(records)}/10 samples built")
    return True


if __name__ == "__main__":
    sys.exit(0 if build() else 1)
