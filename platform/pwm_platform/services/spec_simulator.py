"""Spec Simulator — per-modality physics simulations for all 168 imaging modalities.

Given a spec JSON (from the chat builder) and a variant key, this module:
  1. Looks up the modality config (category_module, solvers, theta, etc.)
  2. For 8 core InverseNet modalities: uses real paper results (CASSI / CACTI / SPC)
  3. For all other modalities: dispatches to a category runner that generates
     modality-specific phantom, forward model, and calibrated baselines
  4. Runs bottleneck analysis calibrated to scenario gaps
  5. Returns a SimulationResult with multi-method comparison
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

STATIC_DIR = Path(__file__).resolve().parent.parent / "static"
SIM_IMG_DIR = STATIC_DIR / "simulations"

# InverseNet results directory (bundled JSON files)
_INVERSENET_DIR = Path(__file__).resolve().parent / "inversenet_data"

# Pre-computed benchmark gallery images
_GALLERY_DIR = STATIC_DIR / "img" / "benchmark_gallery"
_GALLERY_JSON = STATIC_DIR / "benchmark-data" / "benchmark_gallery.json"

# inversenet_type → (gallery_subdir, num_scenes, scene_names)
_GALLERY_CONFIG: Dict[str, Tuple[str, int, List[str]]] = {
    "cassi": ("sd_cassi", 10, [
        "scene01", "scene02", "scene03", "scene04", "scene05",
        "scene06", "scene07", "scene08", "scene09", "scene10",
    ]),
    "cacti": ("cacti", 6, [
        "kobe", "traffic", "runner", "drop", "crash", "aerial",
    ]),
    "spc": ("spc_block", 11, [
        "Monarch", "Parrots", "barbara", "boats", "cameraman",
        "fingerprint", "flinstones", "foreman", "house", "lena256", "peppers256",
    ]),
}

# inversenet_type → gallery JSON key (for headline metrics matching recon_I.png)
_GALLERY_JSON_KEY: Dict[str, str] = {
    "cassi": "sd_cassi",
    "cacti": "cacti",
    "spc": "spc_block",
}

# ── Dynamic gallery config (loads all 168 modalities from JSON) ───────

_gallery_config_cache: Optional[Dict[str, Tuple[str, int, List[str]]]] = None


def _get_dynamic_gallery_config() -> Dict[str, Tuple[str, int, List[str]]]:
    """Load gallery config from benchmark_gallery.json for ALL modalities.

    Returns a dict of variant_key → (gallery_subdir, num_scenes, scene_names).
    Merges static InverseNet config with dynamic entries from the gallery JSON.
    """
    global _gallery_config_cache
    if _gallery_config_cache is not None:
        return _gallery_config_cache

    config: Dict[str, Tuple[str, int, List[str]]] = dict(_GALLERY_CONFIG)

    if _GALLERY_JSON.exists():
        try:
            with open(_GALLERY_JSON) as f:
                gallery = json.load(f)
            for variant_key, data in gallery.items():
                if variant_key in config:
                    continue  # don't override InverseNet entries
                scenes = data.get("scenes", [])
                if not scenes:
                    continue
                config[variant_key] = (
                    variant_key,  # gallery_subdir = modality_id
                    len(scenes),
                    [s.get("scene_name", f"scene_{s.get('scene_idx', i)}")
                     for i, s in enumerate(scenes)],
                )
        except Exception as exc:
            logger.warning("Failed to load dynamic gallery config: %s", exc)

    _gallery_config_cache = config
    return config


def _load_gallery_scene_metrics(
    variant_key: str, scene_idx: int,
) -> Optional[Tuple[float, float, str]]:
    """Load (psnr, ssim, method_name) from the gallery JSON for a given scene.

    These metrics correspond to the recon_I.png image that is actually displayed.
    Looks up variant_key directly first, then falls back to _GALLERY_JSON_KEY mapping.
    """
    if not _GALLERY_JSON.exists():
        return None
    try:
        with open(_GALLERY_JSON) as f:
            gallery = json.load(f)
        # Try direct lookup first (works for all 168 modalities)
        entry = gallery.get(variant_key)
        if entry is None:
            # Fallback: InverseNet key mapping (cassi→sd_cassi, etc.)
            json_key = _GALLERY_JSON_KEY.get(variant_key)
            entry = gallery.get(json_key) if json_key else None
        if entry is None:
            return None
        scenes = entry.get("scenes", [])
        if scene_idx >= len(scenes):
            return None
        scene = scenes[scene_idx]
        return (
            float(scene["psnr_I"]),
            float(scene["ssim_I"]),
            str(entry.get("method", "Classical")),
        )
    except Exception as exc:
        logger.warning("Failed to load gallery scene metrics: %s", exc)
        return None


# ── Core InverseNet modalities (keep real paper results) ─────────────
_INVERSENET_CORE = frozenset({
    "cassi", "sd_cassi", "dd_cassi",
    "cacti",
    "spc", "spc_block", "spc_kronecker", "matrix",
})

# ── InverseNet method display names ──────────────────────────────────

_METHOD_DISPLAY = {
    "gap_tv": "GAP-TV", "pnp_hsicnn": "PnP-HSI-CNN",
    "hdnet": "HDNet", "mst_l": "MST-L",
    "pnp_ffdnet": "PnP-FFDNet", "elp_unfolding": "ELP-Unfolding",
    "efficientsci": "EfficientSCI",
    "fista_tv": "FISTA-TV", "ista_net": "ISTA-Net", "hatnet": "HATNet",
    "pnp_drunet": "PnP-DRUNet",
}

_METHOD_TYPE = {
    "gap_tv": "classical", "pnp_hsicnn": "pnp",
    "hdnet": "deep", "mst_l": "deep",
    "pnp_ffdnet": "pnp", "elp_unfolding": "deep", "efficientsci": "deep",
    "fista_tv": "classical", "ista_net": "deep", "hatnet": "deep",
    "pnp_drunet": "pnp",
}

# ── Category badge colors ────────────────────────────────────────────
_CATEGORY_COLORS = {
    "medical_ct_radon":   ("blue", "CT / Radon"),
    "medical_mri_kspace": ("purple", "MRI / k-space"),
    "microscopy_psf":     ("green", "Microscopy"),
    "electron_ctf":       ("yellow", "Electron"),
    "compressive_mask":   ("violet", "Compressive"),
    "remote_sensing_sar": ("teal", "Remote Sensing"),
    "scanning_probe":     ("orange", "Scanning Probe"),
}

# ── Compression ratio defaults per category ──────────────────────────
_CATEGORY_CR = {
    "medical_ct_radon": 10.0,
    "medical_mri_kspace": 4.0,
    "microscopy_psf": 1.0,
    "electron_ctf": 1.0,
    "compressive_mask": 10.0,
    "remote_sensing_sar": 4.0,
    "scanning_probe": 1.0,
}


# ── Result dataclasses ───────────────────────────────────────────────


@dataclass
class MethodResult:
    """Per-method metrics across three scenarios."""
    key: str
    display_name: str
    method_type: str        # "classical", "pnp", "deep"
    psnr_i: float
    ssim_i: float
    psnr_ii: float
    ssim_ii: float
    psnr_iii: float
    ssim_iii: float
    gap_i_ii: float
    recovery_ii_iii: float


@dataclass
class SimulationResult:
    ground_truth_path: str
    measurement_path: str
    reconstructed_path: str
    psnr: float
    ssim: float
    solver_name: str
    solver_iters: int
    bottleneck: Dict[str, Any]
    recommendations: List[Dict[str, Any]]
    dataset_name: str
    sim_id: str
    # ── Multi-method fields ──
    methods: List[MethodResult] = field(default_factory=list)
    best_method: str = ""
    mismatch_gap_db: float = 0.0
    recovery_db: float = 0.0
    modality: str = ""
    scenario_labels: Dict[str, str] = field(default_factory=dict)
    # ── New fields for category dispatch ──
    attribution: str = ""
    category_module: str = ""
    display_name: str = ""


# ── InverseNet results loaders (unchanged) ───────────────────────────


def _load_inversenet_cassi() -> List[dict]:
    path = _INVERSENET_DIR / "cassi_validation_results.json"
    if not path.exists():
        logger.warning("CASSI InverseNet results not found at %s", path)
        return []
    with open(path) as f:
        return json.load(f)


def _load_inversenet_cacti() -> List[dict]:
    path = _INVERSENET_DIR / "cacti_validation_results.json"
    if not path.exists():
        logger.warning("CACTI InverseNet results not found at %s", path)
        return []
    with open(path) as f:
        raw = json.load(f)

    from collections import defaultdict
    accum: Dict[str, List[dict]] = defaultdict(list)
    for entry in raw:
        accum[entry["name"]].append(entry)

    aggregated = []
    for name, groups in accum.items():
        n = len(groups)
        methods = list(groups[0]["scenarios"]["scenario_i"].keys())
        agg = {"name": name, "scenarios": {}}
        for scenario in ("scenario_i", "scenario_ii", "scenario_iii"):
            agg["scenarios"][scenario] = {}
            for method in methods:
                psnr_vals = [g["scenarios"][scenario][method]["psnr"] for g in groups]
                ssim_vals = [g["scenarios"][scenario][method]["ssim"] for g in groups]
                agg["scenarios"][scenario][method] = {
                    "psnr": sum(psnr_vals) / n,
                    "ssim": sum(ssim_vals) / n,
                }
        aggregated.append(agg)
    return aggregated


def _load_inversenet_spc() -> List[dict]:
    """Load real SPC validation results from InverseNet benchmark."""
    path = _INVERSENET_DIR / "spc_validation_results.json"
    if not path.exists():
        logger.warning("SPC InverseNet results not found at %s", path)
        return []
    with open(path) as f:
        raw = json.load(f)

    # Transpose format: SPC JSON has methods at top level with scenarios nested,
    # but _build_method_results_inversenet expects scenarios at top with methods nested.
    _SPC_METHODS = ("fista_tv", "ista_net", "hatnet", "pnp_drunet")
    results = []
    for entry in raw:
        name = entry.get("image_name", f"image_{entry.get('image_idx', '?')}")
        transposed: Dict[str, Dict[str, dict]] = {
            "scenario_i": {}, "scenario_ii": {}, "scenario_iii": {},
        }
        for method in _SPC_METHODS:
            method_data = entry.get(method)
            if not method_data:
                continue
            for scenario in ("scenario_i", "scenario_ii", "scenario_iii"):
                sc_data = method_data.get(scenario, {})
                transposed[scenario][method] = {
                    "psnr": sc_data.get("psnr", 0.0),
                    "ssim": sc_data.get("ssim", 0.0),
                }
        results.append({"name": name, **transposed})
    return results


# ── Build MethodResult from InverseNet entries ───────────────────────


def _build_method_results_inversenet(
    scene: dict, scenario_key: str = "",
) -> List[MethodResult]:
    """Build MethodResult list from an InverseNet scene/video/image entry."""
    # Normalise structure: may have top-level scenario_i or nested scenarios
    if "scenarios" in scene:
        sc_i = scene["scenarios"]["scenario_i"]
        sc_ii = scene["scenarios"]["scenario_ii"]
        sc_iii = scene["scenarios"]["scenario_iii"]
    else:
        sc_i = scene.get("scenario_i", {})
        sc_ii = scene.get("scenario_ii", {})
        sc_iii = scene.get("scenario_iii", {})

    results = []
    for method_key in sc_i:
        s1, s2, s3 = sc_i[method_key], sc_ii[method_key], sc_iii[method_key]
        results.append(MethodResult(
            key=method_key,
            display_name=_METHOD_DISPLAY.get(method_key, method_key),
            method_type=_METHOD_TYPE.get(method_key, "unknown"),
            psnr_i=round(s1["psnr"], 2),
            ssim_i=round(s1["ssim"], 4),
            psnr_ii=round(s2["psnr"], 2),
            ssim_ii=round(s2["ssim"], 4),
            psnr_iii=round(s3["psnr"], 2),
            ssim_iii=round(s3["ssim"], 4),
            gap_i_ii=round(s1["psnr"] - s2["psnr"], 2),
            recovery_ii_iii=round(s3["psnr"] - s2["psnr"], 2),
        ))
    results.sort(key=lambda m: m.psnr_i, reverse=True)
    return results


# ── Image saving ─────────────────────────────────────────────────────


def _ensure_dir(sim_id: str) -> Path:
    d = SIM_IMG_DIR / sim_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _save_image(
    arr: np.ndarray,
    path: Path,
    cmap: str = "viridis",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    title: str = "",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(4, 4), dpi=100)
    ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=9, color="#333", pad=6)
    fig.tight_layout(pad=0.3)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.08, facecolor="white")
    plt.close(fig)


def _select_display_slice(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        mid = arr.shape[2] // 2
        return arr[:, :, mid]
    return arr.reshape(arr.shape[0], -1)


def _synthesise_reconstruction(
    gt: np.ndarray,
    target_psnr: float,
    rng: np.random.Generator,
) -> np.ndarray:
    data_range = 1.0
    mse = data_range ** 2 / (10 ** (target_psnr / 10))
    noise_std = np.sqrt(mse)
    recon = gt.astype(np.float64) + rng.normal(0, noise_std, gt.shape)
    return np.clip(recon, 0, 1).astype(np.float32)


def _synthesise_recon_from_gallery(
    gt_path: str,
    target_psnr: float,
    sim_id: str,
    rng: np.random.Generator,
) -> str:
    """Create a reconstruction image by adding calibrated noise to a gallery GT.

    Loads the gallery ground-truth PNG, adds Gaussian noise at the given PSNR,
    and saves the result to /static/simulations/{sim_id}/reconstructed.png.
    Returns the URL path.
    """
    from PIL import Image

    # Resolve the gallery GT to an absolute filesystem path
    if gt_path.startswith("/static/"):
        abs_path = STATIC_DIR / gt_path[len("/static/"):]
    else:
        abs_path = Path(gt_path)

    img = Image.open(abs_path).convert("RGB")
    gt_arr = np.asarray(img, dtype=np.float64) / 255.0

    mse = 1.0 / (10 ** (target_psnr / 10))
    noise_std = np.sqrt(mse)
    recon = gt_arr + rng.normal(0, noise_std, gt_arr.shape)
    recon = np.clip(recon * 255, 0, 255).astype(np.uint8)

    d = _ensure_dir(sim_id)
    out_path = d / "reconstructed.png"
    Image.fromarray(recon).save(out_path)
    return f"/static/simulations/{sim_id}/reconstructed.png"


def _save_simulation_images(
    sim_id: str,
    x_true: np.ndarray,
    y_noisy: np.ndarray,
    x_hat: np.ndarray,
    gt_cmap: str = "gray",
    meas_cmap: str = "inferno",
    meas_title: str = "Measurement",
    gt_title: str = "Ground Truth",
    recon_title: str = "Reconstructed",
) -> Tuple[str, str, str]:
    """Save ground truth, measurement, and reconstruction images. Returns URL paths."""
    d = _ensure_dir(sim_id)

    gt_slice = _select_display_slice(x_true)
    recon_slice = _select_display_slice(x_hat)

    if y_noisy.ndim == 1:
        n = len(y_noisy)
        side = int(np.ceil(np.sqrt(n)))
        meas_display = np.zeros(side * side, dtype=np.float32)
        meas_display[:n] = y_noisy
        meas_display = meas_display.reshape(side, side)
    else:
        meas_display = _select_display_slice(y_noisy)

    _save_image(gt_slice, d / "ground_truth.png", cmap=gt_cmap, title=gt_title)
    _save_image(meas_display, d / "measurement.png", cmap=meas_cmap, title=meas_title)
    _save_image(recon_slice, d / "reconstructed.png", cmap=gt_cmap, title=recon_title)

    base = f"/static/simulations/{sim_id}"
    return f"{base}/ground_truth.png", f"{base}/measurement.png", f"{base}/reconstructed.png"


# ── InverseNet legacy pipelines (for 8 core modalities) ─────────────


def _run_cassi_inversenet(
    spec: dict, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, List[MethodResult]]:
    from pwm_core.data.loaders.kaist import KAISTDataset
    from pwm_core.physics.spectral.cassi_operator import SDCASSIOperator

    inversenet = _load_inversenet_cassi()
    dataset = KAISTDataset(resolution=256, num_bands=28)
    scenes = list(dataset)

    if inversenet:
        idx = int(rng.integers(0, min(len(scenes), len(inversenet))))
        name, cube = scenes[idx]
        methods = _build_method_results_inversenet(inversenet[idx])
    else:
        idx = int(rng.integers(0, len(scenes)))
        name, cube = scenes[idx]
        methods = []

    H, W, L = cube.shape
    mask = (rng.random((H, W)) > 0.5).astype(np.float32)
    operator = SDCASSIOperator(
        operator_id="sd_cassi", theta={"L": L, "dispersion_step": 2.0}, mask=mask,
    )
    y_clean = operator.forward(cube)
    y_noisy = y_clean + rng.normal(0, 0.01, y_clean.shape).astype(np.float32)
    best_psnr = methods[0].psnr_i if methods else 25.0
    x_hat = _synthesise_reconstruction(cube, best_psnr, rng)
    return cube, y_noisy, x_hat, name, methods


def _run_cacti_inversenet(
    spec: dict, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, List[MethodResult]]:
    inversenet = _load_inversenet_cacti()
    x_true, mask_3d, dataset_name = _load_cacti_data(rng)

    methods = []
    if inversenet:
        name_lower = dataset_name.lower()
        matched = [e for e in inversenet if e["name"].lower() == name_lower]
        if matched:
            methods = _build_method_results_inversenet(matched[0])
        else:
            entry = inversenet[int(rng.integers(0, len(inversenet)))]
            methods = _build_method_results_inversenet(entry)
            dataset_name = entry["name"]

    from pwm_core.physics.compressive.cacti_operator import CACTIOperator
    H, W, T = x_true.shape
    operator = CACTIOperator(x_shape=(H, W, T), mask=mask_3d[:, :, 0], shift_type="vertical")
    operator.masks = mask_3d
    y_clean = operator.forward(x_true)
    y_noisy = y_clean + rng.normal(0, 0.01, y_clean.shape).astype(np.float32)
    best_psnr = methods[0].psnr_i if methods else 28.0
    x_hat = _synthesise_reconstruction(x_true, best_psnr, rng)
    return x_true, y_noisy, x_hat, dataset_name, methods


def _load_spc_real_image(
    image_name: str,
) -> Optional[np.ndarray]:
    """Load a real Set11 TIF image from the InverseNet dataset."""
    img_dir = Path(__file__).resolve().parent.parent.parent.parent / "papers" / "inversenet" / "data" / "spc" / "images" / "Set11"
    tif_path = img_dir / f"{image_name}.tif"
    if not tif_path.exists():
        return None
    try:
        from PIL import Image
        im = Image.open(tif_path).convert("L")
        arr = np.array(im, dtype=np.float32) / 255.0
        return arr
    except Exception as exc:
        logger.warning("Failed to load SPC image %s: %s", tif_path, exc)
        return None


def _load_spc_sampling_matrix() -> Optional[np.ndarray]:
    """Load the real SPC sampling matrix (Φ) from the InverseNet dataset."""
    mat_dir = Path(__file__).resolve().parent.parent.parent.parent / "papers" / "inversenet" / "data" / "spc" / "sampling_matrix"
    mat_path = mat_dir / "phi_0_25_1089.mat"
    if not mat_path.exists():
        return None
    try:
        import scipy.io
        mat = scipy.io.loadmat(str(mat_path))
        return mat["phi"].astype(np.float32)  # shape (272, 1089)
    except Exception as exc:
        logger.warning("Failed to load SPC sampling matrix: %s", exc)
        return None


def _run_spc_inversenet(
    spec: dict, rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, str, List[MethodResult]]:
    inversenet = _load_inversenet_spc()

    # Pick a random image from the real results
    if inversenet:
        idx = int(rng.integers(0, len(inversenet)))
        entry = inversenet[idx]
        image_name = entry["name"]
    else:
        image_name = "cameraman"
        entry = None

    # Load real Set11 TIF image
    img = _load_spc_real_image(image_name)
    if img is None:
        # Fallback to synthetic Set11
        logger.info("Real Set11 image '%s' unavailable, using synthetic", image_name)
        from pwm_core.data.loaders.set11 import Set11Dataset
        dataset = Set11Dataset(resolution=64)
        images = list(dataset)
        fallback_idx = int(rng.integers(0, len(images)))
        image_name, img = images[fallback_idx]

    # Build methods from real validation data
    methods = []
    if entry:
        methods = _build_method_results_inversenet(entry)
    elif inversenet:
        methods = _build_method_results_inversenet(
            inversenet[int(rng.integers(0, len(inversenet)))]
        )

    # Apply real forward model: y = Φ · vec(x)
    phi = _load_spc_sampling_matrix()
    if phi is not None:
        # Resize image to 33×33 to match sampling matrix (1089 = 33²)
        block_size = int(np.sqrt(phi.shape[1]))  # 33
        from PIL import Image as PILImage
        img_pil = PILImage.fromarray((img * 255).astype(np.uint8))
        img_resized = np.array(img_pil.resize((block_size, block_size), PILImage.LANCZOS), dtype=np.float32) / 255.0
        y_clean = phi @ img_resized.ravel()
        y_noisy = y_clean + rng.normal(0, 0.03, y_clean.shape).astype(np.float32)
    else:
        # Fallback to SPCOperator
        from pwm_core.physics.compressive.spc_operator import SPCOperator
        H, W = img.shape[:2]
        operator = SPCOperator(x_shape=(H, W), sampling_rate=0.25)
        y_clean = operator.forward(img)
        y_noisy = y_clean + rng.normal(0, 0.03, y_clean.shape).astype(np.float32)

    best_psnr = methods[0].psnr_i if methods else 30.0
    x_hat = _synthesise_reconstruction(img, best_psnr, rng)
    return img, y_noisy, x_hat, image_name, methods


def _load_cacti_data(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray, str]:
    try:
        from pwm_core.data.loaders.cacti_bench import CACTIBenchmark
        bench = CACTIBenchmark()
        groups = list(bench)
        idx = int(rng.integers(0, len(groups))) if rng is not None else 0
        name, group_gt, mask, meas = groups[idx]
        return group_gt, mask, name
    except (FileNotFoundError, Exception) as exc:
        logger.info("CACTI benchmark data unavailable (%s), using SyntheticCACTIDataset", exc)
        return _synthetic_cacti(rng)


def _synthetic_cacti(rng: np.random.Generator = None) -> Tuple[np.ndarray, np.ndarray, str]:
    from pwm_core.data.loaders.cacti_bench import SyntheticCACTIDataset
    dataset = SyntheticCACTIDataset(resolution=256, num_frames=8)
    videos = list(dataset)
    idx = int(rng.integers(0, len(videos))) if rng is not None else 0
    name, video, mask = videos[idx]
    return video, mask, name


# ── Category-dispatched simulation ───────────────────────────────────


def _run_category_simulation(
    variant_key: str,
    config: dict,
    spec: dict,
    rng: np.random.Generator,
    sim_id: str,
) -> SimulationResult:
    """Run a category-dispatched simulation for non-InverseNet modalities."""
    from .category_runners import get_runner
    from .category_runners._base import MethodResult as RunnerMethodResult

    category_module = config["category_module"]
    runner = get_runner(category_module)

    # Get baselines (needed for all paths)
    runner_methods, scenario_labels, dataset_label, attribution = runner.get_baselines(config, rng)

    # Convert runner MethodResult to our MethodResult
    methods = [
        MethodResult(
            key=m.key, display_name=m.display_name, method_type=m.method_type,
            psnr_i=m.psnr_i, ssim_i=m.ssim_i,
            psnr_ii=m.psnr_ii, ssim_ii=m.ssim_ii,
            psnr_iii=m.psnr_iii, ssim_iii=m.ssim_iii,
            gap_i_ii=m.gap_i_ii, recovery_ii_iii=m.recovery_ii_iii,
        )
        for m in runner_methods
    ]

    # Best method metrics
    if methods:
        best = methods[0]
        psnr_val = best.psnr_i
        ssim_val = best.ssim_i
        solver_name = best.display_name
        gap_db = best.gap_i_ii
        recovery_db = best.recovery_ii_iii
    else:
        psnr_val = 30.0
        ssim_val = 0.90
        solver_name = "N/A"
        gap_db = 0.0
        recovery_db = 0.0

    # ── Try pre-computed gallery images for this modality ──
    gallery = _get_gallery_paths(variant_key, rng)

    # Fallback for compressive_mask: try InverseNet gallery types
    if gallery is None and category_module == "compressive_mask":
        gallery_type = ["cassi", "cacti", "spc"][int(rng.integers(0, 3))]
        gallery = _get_gallery_paths(gallery_type, rng)

    if gallery:
        gt_path, meas_path, recon_path, scene_idx, scene_name = gallery
        dataset_label = f"{scene_name} ({config.get('display_name', variant_key)})"
        logger.info("Category %s using gallery images: %s", variant_key, scene_name)

        # Load real metrics from gallery JSON if available
        gallery_metrics = _load_gallery_scene_metrics(variant_key, scene_idx)
        if gallery_metrics:
            psnr_val, ssim_val, method_from_gallery = gallery_metrics
            if solver_name == "N/A":
                solver_name = method_from_gallery
    else:
        # Generate phantom
        phantom, phantom_name, gt_cmap = runner.generate_phantom(config, rng)

        # Apply forward model
        measurement, meas_title, meas_cmap = runner.apply_forward_model(phantom, config, rng)

        # Synthesise reconstruction at best method PSNR
        x_hat = _synthesise_reconstruction(phantom, psnr_val, rng)

        # Save images with per-runner colormaps
        gt_path, meas_path, recon_path = _save_simulation_images(
            sim_id, phantom, measurement, x_hat,
            gt_cmap=gt_cmap, meas_cmap=meas_cmap,
            meas_title=meas_title, gt_title=phantom_name,
            recon_title=f"Reconstructed ({solver_name})",
        )

    # Bottleneck analysis
    bottleneck = _estimate_bottleneck(spec, methods, category_module)

    return SimulationResult(
        ground_truth_path=gt_path,
        measurement_path=meas_path,
        reconstructed_path=recon_path,
        psnr=round(psnr_val, 2),
        ssim=round(ssim_val, 4),
        solver_name=solver_name,
        solver_iters=0,
        bottleneck=bottleneck,
        recommendations=bottleneck.get("ranked", []),
        dataset_name=dataset_label,
        sim_id=sim_id,
        methods=methods,
        best_method=solver_name,
        mismatch_gap_db=round(gap_db, 2),
        recovery_db=round(recovery_db, 2),
        modality=variant_key,
        scenario_labels=scenario_labels,
        attribution=attribution,
        category_module=category_module,
        display_name=config.get("display_name", variant_key),
    )


# ── Bottleneck estimation ────────────────────────────────────────────


def _estimate_bottleneck(
    spec: dict,
    methods: List[MethodResult],
    modality_or_category: str,
) -> Dict[str, Any]:
    """Estimate bottleneck severities from scenario gaps."""
    from pwm_core.analysis.bottleneck import rank_bottlenecks

    if not methods:
        return _estimate_bottleneck_from_spec(spec, 25.0, modality_or_category)

    best = methods[0]

    gap = best.gap_i_ii
    mismatch_sev = min(1.0, max(0.0, gap / 15.0))

    noise_desc = (spec.get("noise_model") or "").lower()
    photon_sev = 0.2
    if "poisson" in noise_desc:
        photon_sev = 0.4
    if "high" in noise_desc or "severe" in noise_desc:
        photon_sev = 0.7

    recov_sev = max(0.0, min(1.0, (40.0 - best.psnr_i) / 20.0))

    classical = [m for m in methods if m.method_type == "classical"]
    deep = [m for m in methods if m.method_type == "deep"]
    if classical and deep:
        solver_gap = max(m.psnr_i for m in deep) - max(m.psnr_i for m in classical)
        solver_sev = min(1.0, max(0.1, solver_gap / 10.0))
    else:
        solver_sev = 0.2

    cr = _CATEGORY_CR.get(modality_or_category, 8.0)
    # Legacy support: if modality_or_category is cassi/cacti/spc
    legacy_cr = {"cassi": 28.0, "spc": 4.0, "cacti": 8.0}
    if modality_or_category in legacy_cr:
        cr = legacy_cr[modality_or_category]

    mismatch_params = spec.get("mismatch_params", [])
    mismatch_family = mismatch_params[0].get("name", "unknown") if mismatch_params else None

    return rank_bottlenecks(
        photon_severity=photon_sev,
        recoverability_severity=recov_sev,
        mismatch_severity=mismatch_sev,
        solver_fit_severity=solver_sev,
        snr_db=best.psnr_i,
        compression_ratio=cr,
        mismatch_family=mismatch_family,
        solver_family=best.display_name,
    )


def _estimate_bottleneck_from_spec(
    spec: dict, psnr_val: float, modality_or_category: str,
) -> Dict[str, Any]:
    from pwm_core.analysis.bottleneck import rank_bottlenecks

    noise_desc = (spec.get("noise_model") or "").lower()
    photon_sev = 0.2
    if "poisson" in noise_desc:
        photon_sev = min(1.0, max(0.1, 0.5))

    mismatch_params = spec.get("mismatch_params", [])
    mismatch_sev = 0.0
    if mismatch_params:
        deltas = []
        for p in mismatch_params:
            nom = float(p.get("nominal", 0))
            pert = float(p.get("perturbed", 0))
            if nom != 0:
                deltas.append(abs(pert - nom) / abs(nom))
            elif pert != 0:
                deltas.append(min(abs(pert), 1.0))
        if deltas:
            mismatch_sev = min(1.0, np.mean(deltas))

    recov_sev = max(0.0, min(1.0, (30.0 - psnr_val) / 20.0))
    solver_sev = 0.3

    cr = _CATEGORY_CR.get(modality_or_category, 8.0)
    legacy_cr = {"cassi": 28.0, "spc": 4.0, "cacti": 8.0}
    if modality_or_category in legacy_cr:
        cr = legacy_cr[modality_or_category]

    mismatch_family = mismatch_params[0].get("name", "unknown") if mismatch_params else None

    return rank_bottlenecks(
        photon_severity=photon_sev,
        recoverability_severity=recov_sev,
        mismatch_severity=mismatch_sev,
        solver_fit_severity=solver_sev,
        snr_db=psnr_val,
        compression_ratio=cr,
        mismatch_family=mismatch_family,
        solver_family="classical",
    )


# ── Main entry point ─────────────────────────────────────────────────


async def run_spec_simulation(spec: dict, variant_key: str) -> SimulationResult:
    """Run a full simulation pipeline.

    Async wrapper — CPU-bound work runs in a thread pool.
    """
    import asyncio
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _run_simulation_sync, spec, variant_key)


def _classify_inversenet_type(variant_key: str) -> Optional[str]:
    """For core InverseNet modalities, return cassi/cacti/spc. Otherwise None."""
    vk = variant_key.lower()
    if vk in ("cassi", "sd_cassi", "dd_cassi"):
        return "cassi"
    if vk == "cacti":
        return "cacti"
    if vk in ("spc", "spc_block", "spc_kronecker", "matrix"):
        return "spc"
    return None


def _run_simulation_sync(spec: dict, variant_key: str) -> SimulationResult:
    """Synchronous simulation pipeline — dispatches by modality config."""
    sim_id = uuid.uuid4().hex[:12]
    rng = np.random.default_rng()
    vk = variant_key.lower()

    # 1. Check if this is a core InverseNet modality
    inversenet_type = _classify_inversenet_type(vk)
    if inversenet_type:
        return _run_inversenet_legacy(spec, vk, inversenet_type, rng, sim_id)

    # 2. Look up modality config
    from .modality_configs import get_modality_config
    config = get_modality_config(vk)

    if config:
        # Check if this config's category is compressive_mask AND it's a core variant
        # (shouldn't happen since we checked above, but belt-and-suspenders)
        logger.info(
            "Running category simulation: %s → %s (sim_id=%s)",
            vk, config["category_module"], sim_id,
        )
        return _run_category_simulation(vk, config, spec, rng, sim_id)

    # 3. Unknown modality — fall back to microscopy PSF as default
    logger.warning("Unknown modality %s, falling back to microscopy_psf", vk)
    fallback_config = {
        "modality_id": vk,
        "display_name": variant_key.replace("_", " ").title(),
        "category_module": "microscopy_psf",
        "x_shape": [128, 128],
        "y_shape": [128, 128],
        "theta": {},
        "solvers": {"traditional_cpu": {"name": "Richardson-Lucy"}, "best_quality": {"name": "CARE UNet"}},
        "mismatch_params": [],
        "source_attribution": {},
    }
    return _run_category_simulation(vk, fallback_config, spec, rng, sim_id)


def _get_gallery_paths(
    variant_or_type: str, rng: np.random.Generator,
) -> Optional[Tuple[str, str, str, int, str]]:
    """Return (gt_url, meas_url, recon_url, scene_idx, scene_name) from gallery.

    Works for both InverseNet types (cassi/cacti/spc) and any of the 168
    modality variant keys that have pre-computed gallery images.

    Returns None if gallery images don't exist (caller falls back to synthetic).
    """
    cfg = _get_dynamic_gallery_config().get(variant_or_type)
    if cfg is None:
        return None

    subdir, num_scenes, scene_names = cfg
    gallery_base = _GALLERY_DIR / subdir

    if not gallery_base.is_dir():
        logger.info("Gallery dir %s not found, falling back to synthetic", gallery_base)
        return None

    scene_idx = int(rng.integers(0, num_scenes))
    scene_dir = gallery_base / f"scene_{scene_idx:02d}"

    # Verify required images exist
    gt_file = scene_dir / "gt.png"
    meas_file = scene_dir / "measurement_I.png"
    recon_file = scene_dir / "recon_I.png"

    if not all(f.exists() for f in (gt_file, meas_file, recon_file)):
        logger.info("Gallery scene_%02d missing images, falling back", scene_idx)
        return None

    url_base = f"/static/img/benchmark_gallery/{subdir}/scene_{scene_idx:02d}"
    return (
        f"{url_base}/gt.png",
        f"{url_base}/measurement_I.png",
        f"{url_base}/recon_I.png",
        scene_idx,
        scene_names[scene_idx],
    )


def _run_inversenet_legacy(
    spec: dict, variant_key: str, inversenet_type: str,
    rng: np.random.Generator, sim_id: str,
) -> SimulationResult:
    """Run InverseNet legacy pipeline for 8 core compressive modalities."""
    logger.info("Running %s InverseNet legacy baseline (sim_id=%s)", inversenet_type, sim_id)

    # ── Try pre-computed gallery images first ─────────────────────────
    gallery = _get_gallery_paths(inversenet_type, rng)

    if gallery:
        gt_path, meas_path, recon_path, scene_idx, dataset_name = gallery
        logger.info("Using gallery images: %s scene_%02d (%s)", inversenet_type, scene_idx, dataset_name)

        # Load InverseNet validation results and match to selected scene
        loaders = {
            "cassi": _load_inversenet_cassi,
            "cacti": _load_inversenet_cacti,
            "spc": _load_inversenet_spc,
        }
        inversenet = loaders[inversenet_type]()
        methods: List[MethodResult] = []
        if inversenet:
            # CASSI entries are ordered by scene index
            if inversenet_type == "cassi":
                entry = inversenet[scene_idx] if scene_idx < len(inversenet) else inversenet[0]
            else:
                # CACTI/SPC: match by scene name
                name_lower = dataset_name.lower()
                matched = [e for e in inversenet if e["name"].lower() == name_lower]
                entry = matched[0] if matched else inversenet[0]
            methods = _build_method_results_inversenet(entry)

        # Headline metrics: best InverseNet method (deep learning SOTA)
        if methods:
            best = methods[0]
            psnr_val, ssim_val, solver_name = best.psnr_i, best.ssim_i, best.display_name
            gap_db, recovery_db = best.gap_i_ii, best.recovery_ii_iii
            # Synthesise reconstruction at best method's PSNR from gallery GT
            recon_path = _synthesise_recon_from_gallery(
                gt_path, psnr_val, sim_id, rng,
            )
        else:
            psnr_val, ssim_val, solver_name = 25.0, 0.80, "N/A"
            gap_db, recovery_db = 0.0, 0.0
    else:
        # ── Fallback: synthetic generation pipeline ───────────────────
        runners = {
            "cassi": _run_cassi_inversenet,
            "spc": _run_spc_inversenet,
            "cacti": _run_cacti_inversenet,
        }
        runner_fn = runners[inversenet_type]
        x_true, y_noisy, x_hat, dataset_name, methods = runner_fn(spec, rng)

        gt_cmap = "viridis" if inversenet_type == "cassi" else "gray"
        gt_path, meas_path, recon_path = _save_simulation_images(
            sim_id, x_true, y_noisy, x_hat,
            gt_cmap=gt_cmap, meas_cmap="inferno",
        )

        if methods:
            best = methods[0]
            psnr_val, ssim_val = best.psnr_i, best.ssim_i
            solver_name = best.display_name
            gap_db, recovery_db = best.gap_i_ii, best.recovery_ii_iii
        else:
            psnr_val, ssim_val = 25.0, 0.80
            solver_name, gap_db, recovery_db = "N/A", 0.0, 0.0

    bottleneck = _estimate_bottleneck(spec, methods, inversenet_type)

    scenario_labels = {
        "cassi": {
            "i": "Ideal operator (calibrated mask + dispersion)",
            "ii": "Mismatched operator (dx=0.5px, dy=0.3px, \u03b8=0.1\u00b0)",
            "iii": "Oracle-corrected operator",
        },
        "cacti": {
            "i": "Ideal operator (calibrated temporal masks)",
            "ii": "Mismatched operator (dx=0.5px, dy=0.3px, \u03b8=0.1\u00b0)",
            "iii": "Oracle-corrected operator",
        },
        "spc": {
            "i": "Ideal operator (no gain drift, \u03c3\u2099=0.03)",
            "ii": "Gain-drifted operator (\u03b1=0.0015, \u03c3\u2099=0.03)",
            "iii": "Oracle gain-corrected operator",
        },
    }.get(inversenet_type, {})

    inversenet_attribution = {
        "cassi": "Results from <span class=\"font-medium\">InverseNet</span> benchmark &mdash; KAIST hyperspectral dataset (256&times;256&times;28, 10 scenes)",
        "cacti": "Results from <span class=\"font-medium\">InverseNet</span> benchmark &mdash; SCI video dataset (256&times;256&times;8, 6 videos, 28 groups)",
        "spc": "Results from <span class=\"font-medium\">InverseNet</span> benchmark &mdash; Set11 grayscale dataset (256&times;256, CR=25%, &Phi; &isin; &reals;<sup>272&times;1089</sup>)",
    }.get(inversenet_type, "")

    return SimulationResult(
        ground_truth_path=gt_path,
        measurement_path=meas_path,
        reconstructed_path=recon_path,
        psnr=round(psnr_val, 2),
        ssim=round(ssim_val, 4),
        solver_name=solver_name,
        solver_iters=0,
        bottleneck=bottleneck,
        recommendations=bottleneck.get("ranked", []),
        dataset_name=dataset_name,
        sim_id=sim_id,
        methods=methods,
        best_method=solver_name,
        mismatch_gap_db=round(gap_db, 2),
        recovery_db=round(recovery_db, 2),
        modality=inversenet_type,
        scenario_labels=scenario_labels,
        attribution=inversenet_attribution,
        category_module="compressive_mask",
        display_name=variant_key.upper(),
    )
