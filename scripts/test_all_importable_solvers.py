#!/usr/bin/env python3
"""Run all importable solvers across all modalities and record PSNR/SSIM.

Tests every YAML-defined solver that is importable in pwm_core.
Updates comprehensive_algorithm_test.json with new results.
"""
import sys
import json
import time
import traceback
import importlib
import numpy as np
import h5py
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

import yaml
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

CONFIG_DIR = ROOT / "benchmarks" / "configs"
BENCH_DIR = ROOT / "datasets" / "benchmark"
RESULTS_PATH = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"

# Alias mapping: YAML modality_id → actual dataset directory
MOD_ALIAS = {
    "cassi": "sd_cassi",
    "spc": "spc_kronecker",
}

# Load existing results
with open(RESULTS_PATH) as f:
    results = json.load(f)


def get_psnr_ssim(recon, gt):
    """Compute PSNR and SSIM, handling shape mismatches."""
    recon = np.array(recon, dtype=np.float32)
    gt = np.array(gt, dtype=np.float32)

    if np.iscomplexobj(recon):
        recon = np.abs(recon)
    if np.iscomplexobj(gt):
        gt = np.abs(gt)

    # Align dims: compare first 2D slice if higher-dimensional
    while recon.ndim > 2 and gt.ndim == 2:
        recon = recon[0]
    while gt.ndim > 2 and recon.ndim == 2:
        gt = gt[0]
    if recon.ndim > 2 and gt.ndim > 2:
        if recon.shape[0] != gt.shape[0]:
            recon = recon[0]; gt = gt[0]

    # Resize recon to match gt
    if recon.shape != gt.shape:
        from scipy.ndimage import zoom
        factors = tuple(gt.shape[i] / recon.shape[i] for i in range(min(recon.ndim, gt.ndim)))
        if len(factors) == recon.ndim:
            recon = zoom(recon.astype(np.float64), factors, order=1).astype(np.float32)

    if recon.shape != gt.shape:
        # Last resort: flatten comparison
        n = min(recon.size, gt.size)
        recon = recon.flat[:n]
        gt = gt.flat[:n]

    gt_min, gt_max = gt.min(), gt.max()
    if gt_max > gt_min:
        gt_n = (gt - gt_min) / (gt_max - gt_min)
        recon_n = (recon - gt_min) / (gt_max - gt_min)
        recon_n = np.clip(recon_n, 0, 1)
    else:
        gt_n = np.zeros_like(gt); recon_n = np.zeros_like(recon)

    data_range = 1.0

    try:
        p = float(psnr_fn(gt_n, recon_n, data_range=data_range))
    except Exception:
        mse = float(np.mean((gt_n - recon_n)**2))
        p = float(-10 * np.log10(mse + 1e-10))

    try:
        if gt_n.ndim == 2:
            win = min(7, gt_n.shape[0] // 4 * 2 + 1, gt_n.shape[1] // 4 * 2 + 1)
            win = max(3, win if win % 2 == 1 else win - 1)
            s = float(ssim_fn(gt_n, recon_n, data_range=data_range, win_size=win))
        else:
            s = float(ssim_fn(gt_n.ravel()[:1024], recon_n.ravel()[:1024], data_range=data_range, win_size=3))
    except Exception:
        s = 0.0

    return p, s


def load_sample_ct(sample_idx=0):
    """Load CT sample from npy directory format."""
    sample_dir = BENCH_DIR / "ct" / "public" / f"sample_{sample_idx:02d}"
    if not sample_dir.exists():
        return None
    try:
        return {
            "x_true": np.load(sample_dir / "groundtruth.npy"),
            "y": np.load(sample_dir / "measurement.npy"),
            "H_ideal": np.load(sample_dir / "angles.npy"),
            "sinogram_ideal": np.load(sample_dir / "sinogram_ideal.npy"),
        }
    except Exception:
        return None


def load_sample(mod_id, sample_idx=0):
    """Load sample from public tier dataset."""
    # CT uses npy format
    if mod_id == "ct":
        return load_sample_ct(sample_idx)

    # Apply alias mapping
    ds_id = MOD_ALIAS.get(mod_id, mod_id)

    tier_dir = BENCH_DIR / ds_id / "public"
    if not tier_dir.exists():
        tier_dir = BENCH_DIR / ds_id
    if not tier_dir.exists():
        return None

    h5_files = list(tier_dir.glob("*.h5"))
    if not h5_files:
        h5_files = list(tier_dir.glob("**/*.h5"))
    if not h5_files:
        return None

    h5_path = h5_files[0]
    try:
        with h5py.File(h5_path, "r") as hf:
            key = f"sample_{sample_idx:02d}"
            if key not in hf:
                keys = [k for k in hf.keys() if not k.startswith("_")]
                if not keys:
                    return None
                key = keys[0]
            grp = hf[key]
            sample = {}
            for k in grp.keys():
                sample[k] = np.array(grp[k])

        # Normalize special key names for x_true
        if "x_true" not in sample:
            if "x_true_amplitude" in sample:
                sample["x_true"] = sample["x_true_amplitude"]
            elif "groundtruth" in sample:
                sample["x_true"] = sample["groundtruth"]
            elif "reconstruction_fbp" in sample:
                sample["x_true"] = sample["reconstruction_fbp"]

        # Normalize special key names for y (measurement)
        if "y" not in sample:
            for key in ("sinogram_measured", "bscan_measured", "kspace_undersampled",
                        "projection_measured", "measurement", "interferogram",
                        "kspace", "sinogram", "projection"):
                if key in sample:
                    sample["y"] = sample[key]
                    break

        return sample
    except Exception as e:
        print(f"    [load error] {mod_id}: {e}")
        return None


def run_standard_solver(fn, y, H_ideal, params):
    """Run solver with (y, physics, cfg) -> (recon, info) API."""
    return fn(y, H_ideal, params)


def run_care_solver(fn, y, H_ideal, params, fn_name=""):
    """Run CARE-style solvers."""
    img = y.astype(np.float32)
    is_3d = "3d" in fn_name.lower()

    def _try_care(img_in):
        try:
            return fn(img_in), {}
        except Exception:
            try:
                return fn(img_in, psf=None), {}
            except Exception as e:
                return None, {"error": str(e)}

    if is_3d:
        # Keep full volume (D, H, W)
        if img.ndim == 2:
            img = img[np.newaxis]  # add depth dim
        result, info = _try_care(img)
    else:
        # 2D CARE: take first slice if 3D
        if img.ndim == 3:
            img = img[0]
        result, info = _try_care(img)

    return result, info


def run_dl_image_solver(fn, y, H_ideal, params):
    """Run DL solver that takes just the image."""
    img = y.astype(np.float32)
    if img.ndim == 3:
        img = img[0]
    try:
        result = fn(img)
        return result, {}
    except Exception:
        try:
            result = fn(img, weights_path=None)
            return result, {}
        except Exception as e:
            return None, {"error": str(e)}


def run_mri_solver(fn, y, H_ideal, params):
    """Run MRI solver with (kspace, mask) API."""
    # y may be real-valued undersampled kspace
    kspace = y.astype(np.complex64) if not np.iscomplexobj(y) else y.astype(np.complex64)
    shape = kspace.shape[-2:] if kspace.ndim >= 2 else kspace.shape
    # Use H_ideal as mask if it's the right shape, otherwise create one
    if (H_ideal is not None and isinstance(H_ideal, np.ndarray)
            and H_ideal.shape == shape and H_ideal.dtype in (np.float32, np.float64, bool)):
        mask = (H_ideal != 0).astype(np.float32)
    else:
        mask = (np.abs(kspace) > 0).astype(np.float32)
        if mask.sum() == 0:  # no signal → random mask
            mask = (np.random.RandomState(42).rand(*shape) < 0.25).astype(np.float32)
    try:
        result = fn(kspace, mask, weights_path=None)
        return result, {}
    except Exception:
        try:
            result = fn(kspace, mask)
            return result, {}
        except Exception as e:
            return None, {"error": str(e)}


def run_sci_solver(fn, y, H_ideal, params):
    """Run SCI solver (CASSI/CACTI) with (meas, mask) API."""
    if H_ideal is not None and isinstance(H_ideal, np.ndarray) and H_ideal.ndim >= 2:
        mask = H_ideal.astype(np.float32)
    else:
        mask = np.ones(y.shape[-2:], dtype=np.float32)
    try:
        result = fn(y, mask, weights_path=None)
        return result, {}
    except Exception:
        try:
            result = fn(y, mask)
            return result, {}
        except Exception as e:
            return None, {"error": str(e)}


# Map module short names to calling conventions
CALLING_CONVENTION = {
    "care_unet": "care",
    "redcnn": "dl_image",
    "flatnet": "dl_image",
    "phasenet": "dl_image",
    "ptychonn": "ptychonn",
    "varnet": "mri",
    "modl": "mri",
    "efficientsci": "sci",
    "elp_unfolding": "sci",
    "hdnet": "sci",
    "mst": "sci",
    "hatnet": "sci",
    "ista_net": "sci",
    "lista": "lista",
    "ifcnn": "ifcnn",
    "sim_solver": "sim",
    "dl_sim": "sim",
    "diffusion": "dl_image",
    "diffusion_posterior": "dl_image",
    "destripe_net": "dl_image",
    "nerf_solver": "standard",
    "noise2void": "dl_image",
    "gaussian_splatting_solver": "standard",
    "panorama_solver": "standard",
    "pnp": "pnp_ct",
}


def run_sim_call(fn, y, H_ideal, params, sample):
    """Run SIM solver using raw_frames (9, H, W) when available."""
    raw_frames = sample.get("raw_frames") if sample else None
    if raw_frames is None:
        raw_frames = y
    if raw_frames.ndim != 3:
        return None, {"error": f"raw_frames must be 3D, got {raw_frames.shape}"}
    try:
        result = fn(raw_frames)
        return result, {}
    except Exception:
        try:
            result = fn(raw_frames, H_ideal)
            return result, {}
        except Exception as e:
            return None, {"error": str(e)}


def run_ptychonn_call(fn, y, H_ideal, params, sample):
    """Run PtychoNN using (diffraction_patterns, scan_positions, object_shape)."""
    positions = sample.get("scan_positions") if sample else None
    x_true = sample.get("x_true") if sample else None
    if positions is None:
        return None, {"error": "scan_positions not in dataset"}
    object_shape = tuple(x_true.shape[:2]) if x_true is not None else (256, 256)
    try:
        result = fn(y, positions, object_shape)
        return result, {}
    except Exception as e:
        return None, {"error": str(e)}


def run_ifcnn_call(fn, y, H_ideal, params, sample):
    """Run IFCNN by converting stacked (N, H, W) array to list of 2D images."""
    if y.ndim == 3:
        images = [y[i].astype(np.float32) for i in range(y.shape[0])]
    elif y.ndim == 2:
        images = [y.astype(np.float32), y.astype(np.float32)]  # need at least 2
    else:
        images = [y.astype(np.float32), y.astype(np.float32)]
    # Normalize each image to [0, 1]
    normalized = []
    for img in images:
        mn, mx = img.min(), img.max()
        normalized.append((img - mn) / (mx - mn + 1e-8))
    try:
        result = fn(normalized)
        return result, {}
    except Exception as e:
        return None, {"error": str(e)}


def run_lista_call(fn, y, H_ideal, params, sample):
    """Run LISTA with (y_vector, measurement_matrix) API."""
    if H_ideal is None or H_ideal.ndim < 2:
        return None, {"error": "H_ideal (measurement matrix) not available"}
    # Take one measurement vector if y is 2D
    y_vec = y[0] if y.ndim > 1 else y
    try:
        result = fn(y_vec, H_ideal)
        return result, {}
    except Exception as e:
        return None, {"error": str(e)}


def run_pnp_ct_call(fn, y, H_ideal, params, sample):
    """Run PnP for CT: build Radon forward/adjoint from angles, then call run_pnp."""
    try:
        from skimage.transform import radon, iradon
        angles_deg = H_ideal
        if angles_deg.max() > 2 * np.pi:
            # degrees
            angles_rad = np.deg2rad(angles_deg)
        else:
            angles_rad = angles_deg
            angles_deg = np.rad2deg(angles_rad)

        out_size = sample.get("x_true").shape[0] if sample and sample.get("x_true") is not None else 362

        class CTPhysics:
            def __init__(self):
                self.x_shape = (out_size, out_size)
            def forward(self, x):
                return radon(x.reshape(out_size, out_size), theta=angles_deg, circle=False).astype(np.float32)
            def adjoint(self, sino):
                return iradon(sino, theta=angles_deg, filter_name=None, circle=False,
                              output_size=out_size).astype(np.float32)

        physics = CTPhysics()
        cfg = params if isinstance(params, dict) else {}
        if "denoiser" not in cfg:
            cfg["denoiser"] = "nlm"
        if "iters" not in cfg:
            cfg["iters"] = 15
        return fn(y, physics, cfg)
    except Exception as e:
        return None, {"error": str(e)}


def call_solver(module_name, fn, fn_name, y, H_ideal, params, sample=None):
    """Dispatch to the right calling convention."""
    mod_short = module_name.split(".")[-1]
    conv = CALLING_CONVENTION.get(mod_short, "standard")

    if conv == "standard":
        return run_standard_solver(fn, y, H_ideal, params)
    elif conv == "care":
        return run_care_solver(fn, y, H_ideal, params, fn_name)
    elif conv == "dl_image":
        return run_dl_image_solver(fn, y, H_ideal, params)
    elif conv == "mri":
        return run_mri_solver(fn, y, H_ideal, params)
    elif conv == "sci":
        return run_sci_solver(fn, y, H_ideal, params)
    elif conv == "sim":
        return run_sim_call(fn, y, H_ideal, params, sample)
    elif conv == "ptychonn":
        return run_ptychonn_call(fn, y, H_ideal, params, sample)
    elif conv == "ifcnn":
        return run_ifcnn_call(fn, y, H_ideal, params, sample)
    elif conv == "lista":
        return run_lista_call(fn, y, H_ideal, params, sample)
    elif conv == "pnp_ct":
        return run_pnp_ct_call(fn, y, H_ideal, params, sample)
    else:
        return run_standard_solver(fn, y, H_ideal, params)


def test_solver(mod_id, solver_key, solver_cfg):
    """Test a single solver on sample_00. Returns (psnr, ssim, time, status)."""
    module_path = solver_cfg.get("module", "")
    fn_name = solver_cfg.get("function", "")
    params = solver_cfg.get("params", {})

    # params in YAML is often a string like "2M" — use {} as cfg
    if not isinstance(params, dict):
        params = {}

    sample = load_sample(mod_id)
    if sample is None:
        return None, None, None, "no_dataset"

    y = sample.get("y")
    x_true = sample.get("x_true")
    H_ideal = sample.get("H_ideal")

    if y is None or x_true is None:
        return None, None, None, f"missing_keys: {list(sample.keys())}"

    # Import solver
    try:
        m = importlib.import_module(module_path)
        fn = getattr(m, fn_name)
    except Exception as e:
        return None, None, None, f"import_error: {e}"

    # Run solver
    t0 = time.time()
    try:
        result = call_solver(module_path, fn, fn_name, y, H_ideal, params, sample)
        elapsed = time.time() - t0

        if isinstance(result, tuple):
            recon = result[0]
        else:
            recon = result

        if recon is None:
            info = result[1] if isinstance(result, tuple) else {}
            return None, None, elapsed, f"returned_none: {info.get('error', '')[:80]}"

        p, s = get_psnr_ssim(recon, x_true)
        return p, s, elapsed, "completed"

    except Exception as e:
        elapsed = time.time() - t0
        return None, None, elapsed, f"error: {str(e)[:120]}"


# Load all YAML configs
all_mods = {}
for f in sorted(CONFIG_DIR.glob("*.yaml")):
    if f.name.startswith("_"):
        continue
    with open(f, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    mod_id = data.get("modality_id", f.stem)
    all_mods[mod_id] = data

# Find all importable solvers
print("Finding importable solvers...")
to_test = []
for mod_id, data in sorted(all_mods.items()):
    solvers = data.get("solvers", {}) or {}
    for sk, sv in solvers.items():
        if not sv:
            continue
        module = sv.get("module", "")
        fn_name = sv.get("function", "")
        if not module:
            continue
        try:
            m = importlib.import_module(module)
            if getattr(m, fn_name, None) is not None:
                to_test.append((mod_id, sk, sv))
        except Exception:
            pass

print(f"Found {len(to_test)} importable solvers")
print()

n_done = 0
n_skip = 0
n_error = 0

for mod_id, solver_key, solver_cfg in to_test:
    name = solver_cfg.get("name", solver_key)

    # Skip if already tested with a good result
    existing = (results.get("modalities", {})
                       .get(mod_id, {})
                       .get("solvers", {})
                       .get(solver_key, {}))
    if existing.get("psnr_db") is not None:
        print(f"  SKIP {mod_id}/{solver_key} ({existing['psnr_db']:.2f} dB)")
        n_skip += 1
        continue

    # Skip known slow solvers that do iterative 3D optimization
    slow_modules = {"nerf_solver", "gaussian_splatting_solver"}
    mod_short = solver_cfg.get("module", "").split(".")[-1]
    if mod_short in slow_modules and solver_key != "traditional_cpu":
        # Mark as pending rather than run for hours
        results.setdefault("modalities", {}).setdefault(mod_id, {}).setdefault("solvers", {})[solver_key] = {
            "status": "skipped_slow_optimizer",
            "psnr_db": None, "ssim": None, "exec_time_sec": None,
            "algorithm_name": name,
        }
        print(f"  SLOW {mod_id}/{solver_key}: {name} — skipped (iterative 3D optimizer)")
        n_skip += 1
        with open(RESULTS_PATH, "w") as f_out:
            json.dump(results, f_out, indent=2)
        continue

    print(f"  TEST {mod_id}/{solver_key}: {name} ...", end=" ", flush=True)
    p, s, t, status = test_solver(mod_id, solver_key, solver_cfg)

    # Store result
    results.setdefault("modalities", {}).setdefault(mod_id, {}).setdefault("solvers", {})[solver_key] = {
        "status": status,
        "psnr_db": p,
        "ssim": s,
        "exec_time_sec": t,
        "algorithm_name": name,
    }

    if p is not None:
        print(f"DONE — {p:.2f} dB / SSIM {s:.4f} / {t:.1f}s")
        n_done += 1
    else:
        print(f"FAIL — {status}")
        n_error += 1

    # Save progress after each test
    with open(RESULTS_PATH, "w") as f_out:
        json.dump(results, f_out, indent=2)

print()
print(f"=== SUMMARY ===")
print(f"Completed: {n_done}")
print(f"Skipped (cached): {n_skip}")
print(f"Errors: {n_error}")
print(f"Results: {RESULTS_PATH}")
