#!/usr/bin/env python3
"""Test the fixed solvers for the 27 previously-failing solver tests.

Runs test_all_importable_solvers logic but only on the affected modalities.
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

CONFIG_DIR = ROOT / "benchmarks" / "configs"
BENCH_DIR = ROOT / "datasets" / "benchmark"
RESULTS_PATH = ROOT / "benchmark_results" / "comprehensive_algorithm_test.json"

# Load existing results
with open(RESULTS_PATH) as f:
    results = json.load(f)

# Modalities to re-test (previously failing)
TARGET_MODS = {
    "ct",         # best_quality: pnp_admm → run_pnp
    "flim",       # best_quality: mle_fit_recon dict-irf bug fixed
    "panorama",   # best_quality: multifocus → run_panorama_fusion; famous_dl/small_gpu: ifcnn
    "sim",        # best_quality/famous_dl/small_gpu: need raw_frames
    "ptychography",  # best_quality/famous_dl/small_gpu: need scan_positions
    "matrix",     # famous_dl/small_gpu: lista needs measurement_matrix
    "spc",        # best_quality/famous_dl/small_gpu: SPC shape issues
    "mri",        # best_quality/famous_dl/small_gpu: architecture mismatch
    "cassi",      # best_quality/famous_dl/small_gpu: HDNet/MST weight issues → gap_tv_cassi
    "cacti",      # famous_dl: ELP-Unfolding SpectralPrior rfft→fft fix
    "edx_mapping",  # best_quality: CARE padding issue
}

# Alias mapping
MOD_ALIAS = {"cassi": "sd_cassi", "spc": "spc_kronecker"}


def get_psnr_ssim(recon, gt):
    from skimage.metrics import peak_signal_noise_ratio as psnr_fn
    from skimage.metrics import structural_similarity as ssim_fn
    recon = np.array(recon, dtype=np.float32)
    gt = np.array(gt, dtype=np.float32)
    if np.iscomplexobj(recon): recon = np.abs(recon)
    if np.iscomplexobj(gt): gt = np.abs(gt)
    while recon.ndim > 2 and gt.ndim == 2: recon = recon[0]
    while gt.ndim > 2 and recon.ndim == 2: gt = gt[0]
    if recon.ndim > 2 and gt.ndim > 2:
        if recon.shape[0] != gt.shape[0]: recon = recon[0]; gt = gt[0]
    if recon.shape != gt.shape:
        from scipy.ndimage import zoom
        factors = tuple(gt.shape[i] / recon.shape[i] for i in range(min(recon.ndim, gt.ndim)))
        if len(factors) == recon.ndim:
            recon = zoom(recon.astype(np.float64), factors, order=1).astype(np.float32)
    if recon.shape != gt.shape:
        n = min(recon.size, gt.size)
        recon = recon.flat[:n]; gt = gt.flat[:n]
    gt_min, gt_max = gt.min(), gt.max()
    if gt_max > gt_min:
        gt_n = (gt - gt_min) / (gt_max - gt_min)
        recon_n = np.clip((recon - gt_min) / (gt_max - gt_min), 0, 1)
    else:
        gt_n = np.zeros_like(gt); recon_n = np.zeros_like(recon)
    try:
        p = float(psnr_fn(gt_n, recon_n, data_range=1.0))
    except Exception:
        mse = float(np.mean((gt_n - recon_n)**2))
        p = float(-10 * np.log10(mse + 1e-10))
    try:
        if gt_n.ndim == 2:
            win = min(7, gt_n.shape[0]//4*2+1, gt_n.shape[1]//4*2+1)
            win = max(3, win if win % 2 == 1 else win-1)
            s = float(ssim_fn(gt_n, recon_n, data_range=1.0, win_size=win))
        else:
            s = float(ssim_fn(gt_n.ravel()[:1024], recon_n.ravel()[:1024], data_range=1.0, win_size=3))
    except Exception:
        s = 0.0
    return p, s


def load_sample(mod_id):
    if mod_id == "ct":
        sample_dir = BENCH_DIR / "ct" / "public" / "sample_00"
        if not sample_dir.exists():
            return None
        try:
            return {
                "x_true": np.load(sample_dir / "groundtruth.npy"),
                "y": np.load(sample_dir / "measurement.npy"),
                "H_ideal": np.load(sample_dir / "angles.npy"),
            }
        except Exception:
            return None

    ds_id = MOD_ALIAS.get(mod_id, mod_id)
    tier_dir = BENCH_DIR / ds_id / "public"
    if not tier_dir.exists():
        tier_dir = BENCH_DIR / ds_id
    if not tier_dir.exists():
        return None
    h5_files = list(tier_dir.glob("*.h5"))
    if not h5_files:
        return None
    try:
        with h5py.File(h5_files[0], "r") as hf:
            key = [k for k in hf.keys() if not k.startswith("_")][0]
            grp = hf[key]
            sample = {k: np.array(grp[k]) for k in grp.keys()}
        if "x_true" not in sample:
            for alt in ("x_true_amplitude", "groundtruth", "reconstruction_fbp"):
                if alt in sample: sample["x_true"] = sample[alt]; break
        if "y" not in sample:
            for alt in ("sinogram_measured", "bscan_measured", "kspace_undersampled",
                        "measurement", "interferogram", "kspace", "sinogram"):
                if alt in sample: sample["y"] = sample[alt]; break
        return sample
    except Exception as e:
        print(f"  [load error] {mod_id}: {e}")
        return None


# ── Calling conventions ──────────────────────────────────────────────────────

def run_standard(fn, y, H_ideal, params):
    return fn(y, H_ideal, params)

def run_dl_image(fn, y, H_ideal, params):
    img = y.astype(np.float32)
    if img.ndim == 3: img = img[0]
    try:
        return fn(img), {}
    except Exception:
        try:
            return fn(img, weights_path=None), {}
        except Exception as e:
            return None, {"error": str(e)}

def run_mri(fn, y, H_ideal, params, sample=None):
    kspace = y.astype(np.complex64) if not np.iscomplexobj(y) else y.astype(np.complex64)
    # Combine multi-coil k-space to single coil (RSS)
    if kspace.ndim == 3:
        kspace = np.sqrt(np.sum(np.abs(kspace)**2, axis=0)).astype(np.float32) + 0j
        kspace = kspace.astype(np.complex64)
    shape = kspace.shape[-2:] if kspace.ndim >= 2 else kspace.shape
    # Use mask from sample if available, else derive from data
    if sample is not None and "mask" in sample:
        mask = sample["mask"].astype(np.float32)
        if mask.shape != shape:
            mask = (np.abs(kspace) > 0).astype(np.float32)
    elif H_ideal is not None and isinstance(H_ideal, np.ndarray) and H_ideal.shape == shape:
        mask = (H_ideal != 0).astype(np.float32)
    else:
        mask = (np.abs(kspace) > 0).astype(np.float32)
    try:
        return fn(kspace, mask, weights_path=None), {}
    except Exception:
        try:
            return fn(kspace, mask), {}
        except Exception as e:
            return None, {"error": str(e)}

def run_sci(fn, y, H_ideal, params):
    if H_ideal is not None and isinstance(H_ideal, np.ndarray) and H_ideal.ndim >= 2:
        mask = H_ideal.astype(np.float32)
    else:
        mask = np.ones(y.shape[-2:], dtype=np.float32)
    try:
        return fn(y, mask, weights_path=None), {}
    except Exception:
        try:
            return fn(y, mask), {}
        except Exception as e:
            return None, {"error": str(e)}

def run_sim_call(fn, y, H_ideal, params, sample):
    raw_frames = sample.get("raw_frames") if sample else None
    if raw_frames is None: raw_frames = y
    if raw_frames.ndim != 3:
        return None, {"error": f"raw_frames must be 3D, got {raw_frames.shape}"}
    try:
        return fn(raw_frames), {}
    except Exception:
        try:
            return fn(raw_frames, H_ideal), {}
        except Exception as e:
            return None, {"error": str(e)}

def run_ptychonn_call(fn, y, H_ideal, params, sample):
    positions = sample.get("scan_positions") if sample else None
    x_true = sample.get("x_true") if sample else None
    if positions is None:
        return None, {"error": "scan_positions not in dataset"}
    object_shape = tuple(x_true.shape[:2]) if x_true is not None else (256, 256)
    try:
        return fn(y, positions, object_shape), {}
    except Exception as e:
        return None, {"error": str(e)}

def run_ifcnn_call(fn, y, H_ideal, params, sample):
    if y.ndim == 3:
        images = [y[i].astype(np.float32) for i in range(y.shape[0])]
    else:
        images = [y.astype(np.float32), y.astype(np.float32)]
    normalized = []
    for img in images:
        mn, mx = img.min(), img.max()
        normalized.append((img - mn) / (mx - mn + 1e-8))
    try:
        return fn(normalized), {}
    except Exception as e:
        return None, {"error": str(e)}

def run_lista_call(fn, y, H_ideal, params, sample):
    if H_ideal is None or H_ideal.ndim < 2:
        return None, {"error": "H_ideal (measurement matrix) not available"}
    y_vec = y[0] if y.ndim > 1 else y
    kw = params if isinstance(params, dict) else {}
    try:
        return fn(y_vec, H_ideal, **kw), {}
    except TypeError:
        try:
            return fn(y_vec, H_ideal), {}
        except Exception as e:
            return None, {"error": str(e)}
    except Exception as e:
        return None, {"error": str(e)}

def run_pnp_ct_call(fn, y, H_ideal, params, sample):
    try:
        from skimage.transform import radon, iradon
        angles_deg = H_ideal
        if angles_deg.max() <= 2 * np.pi:
            angles_deg = np.rad2deg(angles_deg)
        out_size = sample["x_true"].shape[0] if sample and "x_true" in sample else 362

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
        if "denoiser" not in cfg: cfg["denoiser"] = "nlm"
        if "iters" not in cfg: cfg["iters"] = 15
        return fn(y, physics, cfg)
    except Exception as e:
        return None, {"error": str(e)}

def run_spc_call(fn, y, H_ideal, params, sample):
    """SPC: pass first measurement vector + sensing matrix."""
    if H_ideal is None or H_ideal.ndim < 2:
        return None, {"error": "H_ideal (sensing matrix) not available"}
    y_vec = y[0].astype(np.float32) if y.ndim > 1 else y.astype(np.float32)
    A = H_ideal.astype(np.float32)
    kw = params if isinstance(params, dict) else {}
    try:
        result = fn(y_vec, A, **kw)
        return result, {}
    except Exception as e:
        return None, {"error": str(e)}

def run_cassi_call(fn, y, H_ideal, params, sample):
    """CASSI: pass measurement + 2D mask + explicit keyword params."""
    mask = H_ideal.astype(np.float32) if H_ideal is not None else np.ones(y.shape, dtype=np.float32)
    kw = params if isinstance(params, dict) else {}
    try:
        result = fn(y.astype(np.float32), mask, **kw)
        return result, {}
    except Exception as e:
        return None, {"error": str(e)}


CALLING_CONVENTION = {
    "care_unet": "dl_image",
    "redcnn": "dl_image",
    "flatnet": "dl_image",
    "phasenet": "dl_image",
    "ptychonn": "ptychonn",
    "varnet": "mri",
    "modl": "mri",
    "mri_solvers": "mri",
    "efficientsci": "sci",
    "elp_unfolding": "sci",
    "hdnet": "sci",
    "mst": "sci",
    "hatnet": "sci",
    "ista_net": "sci",
    "lista": "lista",
    "classical": "lista",
    "ifcnn": "ifcnn",
    "sim_solver": "sim",
    "dl_sim": "sim",
    "spc_solvers": "spc",
    "gap_tv": "cassi",
    "diffusion": "dl_image",
    "diffusion_posterior": "dl_image",
    "destripe_net": "dl_image",
    "nerf_solver": "standard",
    "noise2void": "dl_image",
    "gaussian_splatting_solver": "standard",
    "panorama_solver": "standard",
    "pnp": "pnp_ct",
}


def call_solver(module_name, fn, fn_name, y, H_ideal, params, sample=None):
    mod_short = module_name.split(".")[-1]
    conv = CALLING_CONVENTION.get(mod_short, "standard")
    if conv == "standard": return run_standard(fn, y, H_ideal, params)
    elif conv == "dl_image": return run_dl_image(fn, y, H_ideal, params)
    elif conv == "mri": return run_mri(fn, y, H_ideal, params, sample)
    elif conv == "sci": return run_sci(fn, y, H_ideal, params)
    elif conv == "sim": return run_sim_call(fn, y, H_ideal, params, sample)
    elif conv == "ptychonn": return run_ptychonn_call(fn, y, H_ideal, params, sample)
    elif conv == "ifcnn": return run_ifcnn_call(fn, y, H_ideal, params, sample)
    elif conv == "lista": return run_lista_call(fn, y, H_ideal, params, sample)
    elif conv == "pnp_ct": return run_pnp_ct_call(fn, y, H_ideal, params, sample)
    elif conv == "spc": return run_spc_call(fn, y, H_ideal, params, sample)
    elif conv == "cassi": return run_cassi_call(fn, y, H_ideal, params, sample)
    else: return run_standard(fn, y, H_ideal, params)


def test_solver(mod_id, solver_key, solver_cfg):
    module_path = solver_cfg.get("module", "")
    fn_name = solver_cfg.get("function", "")
    params = solver_cfg.get("params", {})
    if not isinstance(params, dict): params = {}

    sample = load_sample(mod_id)
    if sample is None:
        return None, None, None, "no_dataset"

    y = sample.get("y")
    x_true = sample.get("x_true")
    H_ideal = sample.get("H_ideal")
    if y is None or x_true is None:
        return None, None, None, f"missing_keys: {list(sample.keys())}"

    try:
        m = importlib.import_module(module_path)
        fn = getattr(m, fn_name)
    except Exception as e:
        return None, None, None, f"import_error: {e}"

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
    if f.name.startswith("_"): continue
    with open(f, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    mod_id = data.get("modality_id", f.stem)
    all_mods[mod_id] = data

print("=" * 70)
print("TARGETED SOLVER FIX TESTING — previously-failing modalities")
print("=" * 70)

n_done = 0
n_fail = 0
n_skip = 0

for mod_id in sorted(TARGET_MODS):
    data = all_mods.get(mod_id)
    if data is None:
        print(f"\n{mod_id}: NOT IN YAML CONFIGS")
        continue

    solvers = data.get("solvers", {}) or {}
    print(f"\n{mod_id}:")

    for sk, sv in solvers.items():
        if not sv: continue
        module = sv.get("module", "")
        fn_name = sv.get("function", "")
        name = sv.get("name", sk)

        # Check if already passing
        existing = (results.get("modalities", {})
                           .get(mod_id, {})
                           .get("solvers", {})
                           .get(sk, {}))
        if existing.get("psnr_db") is not None:
            print(f"  SKIP {sk}: {name} ({existing['psnr_db']:.2f} dB)")
            n_skip += 1
            continue

        print(f"  TEST {sk}: {name} ...", end=" ", flush=True)
        p, s, t, status = test_solver(mod_id, sk, sv)

        # Update results
        results.setdefault("modalities", {}).setdefault(mod_id, {}).setdefault("solvers", {})[sk] = {
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
            n_fail += 1

        # Save after each test
        with open(RESULTS_PATH, "w") as f_out:
            json.dump(results, f_out, indent=2)

print(f"\n{'='*70}")
print(f"Done: {n_done} | Failed: {n_fail} | Skipped: {n_skip}")
print(f"Results saved: {RESULTS_PATH}")
