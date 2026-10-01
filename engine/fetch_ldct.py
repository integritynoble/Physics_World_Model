"""Fetch more real paired low-dose CT patients from TCIA for the knowledge engine (public NBIA API, no login).

    python3 -m engine.fetch_ldct --out /path/to/ldct-v2 --chest 12 --abdomen 12 [--slices 3]

The runtime's fetcher (ai4science ... runners/fetch/runners.py) takes the first patients in the list, which are all
chest. This one picks chest (C) and abdomen (L) patients separately, in a fixed order, reconstructed images only (not
the projection data), pairs full and low dose BY SLICE POSITION (as the runtime learned to), and keeps the middle
slices. It writes the same layout the runtime uses (volumes.npz, metadata.json, PROVENANCE.json) into its own folder,
so the runtime's 4-patient corpus is never touched. Data: TCIA LDCT-and-Projection-data, under its usage policy.
"""
from __future__ import annotations

import argparse
import io
import json
import time
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

API = "https://services.cancerimagingarchive.net/nbia-api/services/v1/"
COLLECTION = "LDCT-and-Projection-data"


def _get(endpoint, timeout=180, **params):
    url = API + endpoint + "?" + urllib.parse.urlencode(params)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt == 3:
                raise
            time.sleep(5 * (attempt + 1))


def series_of(pid):
    raw = _get("getSeries", Collection=COLLECTION, PatientID=pid)
    by = {}
    for s in json.loads(raw) if raw.strip() else []:
        d = (s.get("SeriesDescription") or "").lower()
        if "image" not in d:
            continue
        by["full" if "full" in d else "low" if "low" in d else "?"] = s
    return by if "full" in by and "low" in by else None


def read_series(uid):
    import pydicom
    zf = zipfile.ZipFile(io.BytesIO(_get("getImage", timeout=900, SeriesInstanceUID=uid)))
    out = {}
    for name in zf.namelist():
        try:
            ds = pydicom.dcmread(io.BytesIO(zf.read(name)))
            z = round(float(ds.ImagePositionPatient[2]), 2)
        except Exception:
            continue
        arr = ds.pixel_array.astype(np.float32)
        out[z] = arr * float(getattr(ds, "RescaleSlope", 1) or 1) + float(getattr(ds, "RescaleIntercept", 0) or 0)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--chest", type=int, default=12)
    ap.add_argument("--abdomen", type=int, default=12)
    ap.add_argument("--slices", type=int, default=3)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    pats = json.loads(_get("getPatient", Collection=COLLECTION))
    ids = sorted(p.get("PatientId") or p.get("PatientID") for p in pats)
    want = {"C": a.chest, "L": a.abdomen}
    got, meta, arrays = {"C": [], "L": []}, [], {}
    for pid in ids:
        kind = pid[:1]
        if kind not in want or len(got[kind]) >= want[kind]:
            continue
        try:
            by = series_of(pid)
            if by is None:
                meta.append({"patient": pid, "skipped": "no paired full/low image series"})
                continue
            full, low = read_series(by["full"]["SeriesInstanceUID"]), read_series(by["low"]["SeriesInstanceUID"])
            shared = sorted(set(full) & set(low))
            if len(shared) < a.slices:
                meta.append({"patient": pid, "skipped": f"{len(shared)} shared slice positions"})
                continue
            mid = len(shared) // 2
            take = shared[mid - a.slices // 2: mid - a.slices // 2 + a.slices]
            f, l = np.stack([full[z] for z in take]), np.stack([low[z] for z in take])
            if f.shape != l.shape or f.shape[1:] != (512, 512):
                meta.append({"patient": pid, "skipped": f"shape {f.shape} / {l.shape}"})
                continue
        except Exception as exc:
            meta.append({"patient": pid, "skipped": str(exc)[:160]})
            continue
        arrays[f"full_{pid}"], arrays[f"low_{pid}"] = f, l
        got[kind].append(pid)
        meta.append({"patient": pid, "anatomy": {"C": "chest", "L": "abdomen"}[kind], "slices": int(f.shape[0]),
                     "z_positions": [float(z) for z in take],
                     "pair_rmse_hu": float(np.sqrt(((f - l) ** 2).mean()))})
        print(f"{pid}: ok ({len(got['C'])} chest, {len(got['L'])} abdomen)", flush=True)
        if all(len(got[k]) >= want[k] for k in want):
            break
    patients = got["C"] + got["L"]
    np.savez_compressed(out / "volumes.npz", **arrays)
    (out / "metadata.json").write_text(json.dumps({"patients": patients, "units": "HU", "detail": meta}, indent=1))
    (out / "PROVENANCE.json").write_text(json.dumps({
        "corpus": "ldct-v2", "source": "TCIA LDCT-and-Projection-data (Mayo/AAPM), public NBIA API",
        "licence": "TCIA data usage policy", "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "patients": patients, "note": "paired full- and low-dose reconstructions; middle slices; no simulation"},
        indent=1))
    print(json.dumps({"chest": len(got["C"]), "abdomen": len(got["L"]), "skipped": sum("skipped" in m for m in meta)}))


if __name__ == "__main__":
    main()
