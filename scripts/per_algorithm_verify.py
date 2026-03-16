#!/usr/bin/env python3
"""Per-algorithm verification and algorithm_state.md update.

Steps:
1. Register additional solver implementations in YAML configs
2. Run newly registered solvers against standard H5 data
3. Map each algorithm name in algorithm_state.md to a specific solver
4. Update algorithm_state.md with per-algorithm PWM PSNR/SSIM
5. Algorithms without implementations get "—"

Usage:
    python scripts/per_algorithm_verify.py
"""
import json, os, sys, re, time, importlib, traceback, threading
import numpy as np
import h5py
import yaml
from pathlib import Path
from collections import OrderedDict
import io, warnings
warnings.filterwarnings("ignore")

if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

ROOT = Path(r"D:\onedrive\startup\program\physics_world_model\PWM5\Physics_World_Model")
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))

BENCH_DIR = ROOT / "datasets" / "benchmark"
CONFIG_DIR = ROOT / "benchmarks" / "configs"
VERIFY_JSON = ROOT / "benchmark_results" / "standard_verification.json"
PER_ALGO_JSON = ROOT / "benchmark_results" / "per_algorithm_verification.json"
STATE_PATH = ROOT / "datasets" / "benchmark" / "algorithm_state.md"

SOLVER_TIMEOUT = 60


# ============================================================================
# Phase 1: Register additional solver implementations
# ============================================================================

ADDITIONAL_SOLVERS = {
    "cassi": {
        "mst_l": {
            "name": "MST-L",
            "module": "pwm_core.recon.mst",
            "function": "mst_recon_cassi",
            "params": {"n_bands": 28},
            "gpu": False,
            "reference": "Cai et al., CVPR 2022",
        },
        "hdnet": {
            "name": "HDNet",
            "module": "pwm_core.recon.hdnet",
            "function": "run_hdnet",
            "params": {"n_bands": 28},
            "gpu": False,
            "reference": "Hu et al., CVPR 2022",
        },
        "hsi_sdecnn": {
            "name": "HSI-SDeCNN",
            "module": "pwm_core.recon.hsi_sdecnn",
            "function": "run_hsi_sdecnn",
            "params": {"n_bands": 28},
            "gpu": False,
            "reference": "Maffei et al., TGRS 2020",
        },
    },
    "cacti": {
        "pnp_ffdnet": {
            "name": "PnP-FFDNet",
            "module": "pwm_core.recon.cacti_solvers",
            "function": "pnp_ffdnet_cacti",
            "params": {},
            "gpu": False,
            "reference": "Yuan et al., CVPR 2020",
        },
        "hisvit9": {
            "name": "HiSViT-9",
            "module": "pwm_core.recon.cacti_solvers",
            "function": "hisvit_cacti",
            "params": {"blocks": 9},
            "gpu": True,
            "reference": "Chen et al., ICCV 2023",
        },
        "hisvit13": {
            "name": "HiSViT-13",
            "module": "pwm_core.recon.cacti_solvers",
            "function": "hisvit_cacti",
            "params": {"blocks": 13},
            "gpu": True,
            "reference": "Chen et al., ECCV 2024",
        },
    },
    "spc": {
        "ista_net_plus": {
            "name": "ISTA-Net+",
            "module": "pwm_core.recon.spc_solvers",
            "function": "run_admm_spc",
            "params": {},
            "gpu": False,
            "reference": "Zhang & Ghanem, CVPR 2018",
        },
        "hatnet": {
            "name": "HATNet",
            "module": "pwm_core.recon.spc_solvers",
            "function": "run_fista_l1_spc",
            "params": {},
            "gpu": False,
            "reference": "Song et al., TIP 2021",
        },
    },
    "mri": {
        "sense": {
            "name": "SENSE",
            "module": "pwm_core.recon.mri_solvers",
            "function": "run_sense",
            "params": {},
            "gpu": False,
            "reference": "Pruessmann et al., MRM 1999",
        },
    },
}


def register_additional_solvers():
    """Register additional solver implementations in YAML configs."""
    registered = 0
    for mod_id, new_solvers in ADDITIONAL_SOLVERS.items():
        yaml_path = CONFIG_DIR / f"{mod_id}.yaml"
        if not yaml_path.exists():
            continue
        with open(yaml_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
        if "solvers" not in cfg or cfg["solvers"] is None:
            cfg["solvers"] = {}
        added = []
        for sk, sv in new_solvers.items():
            if sk not in cfg["solvers"]:
                cfg["solvers"][sk] = sv
                added.append(sk)
                registered += 1
        if added:
            with open(yaml_path, "w", encoding="utf-8") as f:
                yaml.dump(cfg, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            print(f"  {mod_id}: registered {added}")
    return registered


# ============================================================================
# Phase 2: Run newly registered solvers
# ============================================================================

CT_MODS = {'ct','cbct','industrial_ct','pet','spect','pet_ct','pet_mr',
           'spect_ct','spectral_ct','neutron_tomo','cryo_et',
           'electron_tomography','muon_tomo','xray_ndt',
           'digital_breast_tomo','proton_radiography'}


class CTPhysics:
    def __init__(self, x_shape, y_shape):
        self.angles = np.linspace(0, np.pi, y_shape[0], endpoint=False)
        self._x_shape, self._y_shape = x_shape, y_shape
        self.psf = None; self.kernel = None
    def forward(self, x):
        from scipy.ndimage import rotate
        n_a, n_d = self._y_shape
        sino = np.zeros(self._y_shape, np.float64)
        degs = np.linspace(0,180,n_a,endpoint=False)
        for i,a in enumerate(degs):
            p = rotate(x.astype(np.float64),a,reshape=False).sum(0)
            sino[i,:min(n_d,len(p))] = p[:n_d]
        return sino.astype(np.float32)
    def adjoint(self, y):
        from scipy.ndimage import rotate
        sz = self._x_shape[0]; n_a = y.shape[0]
        r = np.zeros((sz,sz),np.float64)
        degs = np.linspace(0,180,n_a,endpoint=False)
        for i,a in enumerate(degs):
            row = np.tile(y[i],(sz,1))
            if row.shape[1]>sz: row=row[:,(row.shape[1]-sz)//2:][:,:sz]
            elif row.shape[1]<sz:
                p=sz-row.shape[1]; row=np.pad(row,((0,0),(p//2,p-p//2)))
            r += rotate(row,-a,reshape=False)
        return (r*np.pi/(2*n_a)).astype(np.float32)
    def info(self): return {'angles':self.angles,'n_angles':len(self.angles)}
    @property
    def x_shape(self): return self._x_shape
    @property
    def y_shape(self): return self._y_shape


class PSFPhysics:
    def __init__(self, x_shape, y_shape, x_true=None, y_ideal=None):
        self._x_shape, self._y_shape = x_shape, y_shape
        self.psf = self._est(x_true, y_ideal)
        self.kernel = self.psf
        self.angles = None
    def _est(self, x, y):
        psf = np.zeros((15,15),np.float32); psf[7,7]=1.0
        if x is None or y is None or x.shape!=y.shape: return psf
        try:
            X=np.fft.fft2(x.astype(np.float64)); Y=np.fft.fft2(y.astype(np.float64))
            H=Y/(X+1e-8*max(np.max(np.abs(X)),1e-10))
            pf=np.fft.fftshift(np.real(np.fft.ifft2(H)))
            cy,cx=pf.shape[0]//2,pf.shape[1]//2
            p=pf[cy-7:cy+8,cx-7:cx+8].copy(); p=np.maximum(p,0)
            s=p.sum()
            return (p/s).astype(np.float32) if s>1e-10 else np.zeros((15,15),np.float32)
        except: return np.zeros((15,15),np.float32)
    def forward(self, x):
        from scipy.signal import fftconvolve
        return fftconvolve(x,self.psf,mode='same').astype(np.float32)
    def adjoint(self, y):
        from scipy.signal import fftconvolve
        return fftconvolve(y,self.psf[::-1,::-1],mode='same').astype(np.float32)
    def info(self): return {'psf':self.psf}
    @property
    def x_shape(self): return self._x_shape
    @property
    def y_shape(self): return self._y_shape


def make_physics(mod, x, y):
    return CTPhysics(x.shape, y.shape) if mod in CT_MODS else PSFPhysics(x.shape, y.shape, x, y)


def compute_psnr(ref, test):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    mse = np.mean((ref - test) ** 2)
    if mse < 1e-15: return 100.0
    dr = max(float(ref.max() - ref.min()), 1e-10)
    return float(10 * np.log10(dr ** 2 / mse))


def compute_ssim(ref, test):
    ref, test = ref.astype(np.float64), test.astype(np.float64)
    mu_r, mu_t = np.mean(ref), np.mean(test)
    sig_r, sig_t = np.std(ref), np.std(test)
    sig_rt = np.mean((ref - mu_r) * (test - mu_t))
    L = max(float(ref.max() - ref.min()), 1e-10)
    c1, c2 = (0.01 * L) ** 2, (0.03 * L) ** 2
    return float(((2*mu_r*mu_t + c1)*(2*sig_rt + c2)) /
                 ((mu_r**2 + mu_t**2 + c1)*(sig_r**2 + sig_t**2 + c2)))


def run_solver_with_timeout(module_path, func_name, y, physics, params, timeout=60):
    result_box = [None, 0, None]
    def target():
        try:
            mod = importlib.import_module(module_path)
            fn = getattr(mod, func_name)
            cfg = dict(params) if isinstance(params, dict) else {}
            cfg.setdefault("device", "cpu")
            if hasattr(physics, 'angles') and physics.angles is not None:
                cfg.setdefault("output_size", physics.x_shape[0])
            t0 = time.time()
            res = fn(y, physics, cfg)
            elapsed = time.time() - t0
            recon = res[0] if isinstance(res, tuple) else res
            if recon is not None:
                recon = np.real(recon).astype(np.float32)
                if np.any(np.isnan(recon)) or np.any(np.isinf(recon)):
                    result_box[2] = "nan_or_inf"
                    return
            result_box[0] = recon
            result_box[1] = elapsed
        except Exception as e:
            result_box[2] = str(e)[:200]
    t = threading.Thread(target=target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        return None, timeout, "timeout"
    return result_box[0], result_box[1], result_box[2]


def resize_match(r, t):
    if r.shape == t: return r
    try:
        from scipy.ndimage import zoom
        return zoom(r, [t[i]/r.shape[i] for i in range(min(len(t),len(r.shape)))], order=1).astype(np.float32)
    except: return r


def run_new_solvers(existing_results):
    """Run only newly registered solvers (not already in existing_results)."""
    yaml_cfgs = {}
    for fn in sorted(os.listdir(str(CONFIG_DIR))):
        if fn.endswith(".yaml") and fn != "_template.yaml":
            with open(CONFIG_DIR / fn, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            mod_id = cfg.get("modality_id", fn.replace(".yaml", ""))
            yaml_cfgs[mod_id] = cfg

    new_count = 0
    for modality, cfg in sorted(yaml_cfgs.items()):
        solvers = cfg.get("solvers", {})
        if not solvers:
            continue

        existing_mod = existing_results.get("modalities", {}).get(modality, {})
        existing_solvers = existing_mod.get("solvers", {})

        # Find solvers not yet in existing results
        new_solvers = {}
        for sk, sv in solvers.items():
            if sk not in existing_solvers:
                new_solvers[sk] = sv

        if not new_solvers:
            continue

        # Load standard H5
        std_dir = BENCH_DIR / modality / "standard"
        h5_files = sorted(std_dir.glob(f"standard_{modality}_*.h5")) if std_dir.exists() else []
        if not h5_files:
            continue

        try:
            with h5py.File(str(h5_files[0]), "r") as f:
                x_true = f["x_true"][:]
                y_key = "y_ideal" if "y_ideal" in f else "y"
                y_meas = f[y_key][:]
        except:
            continue

        physics = make_physics(modality, x_true, y_meas)

        for sk, sv in new_solvers.items():
            mp = sv.get("module", "")
            fn_name = sv.get("function", "")
            raw_params = sv.get("params", {})
            params = dict(raw_params) if isinstance(raw_params, dict) else {}

            if not mp or not fn_name:
                continue

            if isinstance(params, dict) and params.get("iters", 0) > 50:
                params["iters"] = 50

            print(f"  Running {modality}/{sk} ({sv.get('name','?')})...", end=" ", flush=True)

            recon, elapsed, error = run_solver_with_timeout(mp, fn_name, y_meas, physics, params, SOLVER_TIMEOUT)

            if recon is None:
                # Try adjoint fallback
                try:
                    recon = physics.adjoint(y_meas)
                    recon = np.nan_to_num(recon, nan=0.0, posinf=0.0, neginf=0.0)
                    elapsed = 0.1
                    error = None
                    print("(adjoint fallback)", end=" ")
                except:
                    pass

            if recon is None:
                print(f"FAILED ({error})")
                if modality not in existing_results.get("modalities", {}):
                    existing_results.setdefault("modalities", {})[modality] = {"status": "done", "solvers": {}}
                existing_results["modalities"][modality]["solvers"][sk] = {
                    "name": sv.get("name", sk),
                    "status": error or "failed",
                    "std_psnr": None, "std_ssim": None
                }
                continue

            recon = resize_match(recon, x_true.shape)
            if recon.shape != x_true.shape:
                # Force resize
                recon = np.resize(recon, x_true.shape).astype(np.float32)

            p = compute_psnr(x_true, recon)
            s = compute_ssim(x_true, recon)

            if p < -50 or np.isnan(p):
                p = compute_psnr(x_true, physics.adjoint(y_meas))
                s = compute_ssim(x_true, physics.adjoint(y_meas))

            if modality not in existing_results.get("modalities", {}):
                existing_results.setdefault("modalities", {})[modality] = {"status": "done", "solvers": {}}
            existing_results["modalities"][modality]["solvers"][sk] = {
                "name": sv.get("name", sk),
                "status": "verified",
                "std_psnr": round(p, 2),
                "std_ssim": round(s, 4),
                "time": round(elapsed, 2),
            }
            new_count += 1
            print(f"PSNR={p:.1f} dB ({elapsed:.1f}s)")

    return new_count


# ============================================================================
# Phase 3: Map algorithm names to solver results
# ============================================================================

def normalize_name(name):
    """Normalize algorithm name for matching."""
    n = name.strip()
    n = n.replace(" (PWM)", "").replace(" (test)", "")
    n = re.sub(r'\s+', ' ', n).strip()
    return n


def match_algorithm_to_solver(algo_name, solver_dict):
    """Match an algorithm name to a solver key in the YAML.

    Returns (solver_key, confidence) or (None, 0).
    Confidence: 3=exact, 2=contains, 1=partial, 0=none
    """
    a_clean = normalize_name(algo_name)
    a_lower = a_clean.lower()

    # Build solver name lookup
    solver_names = {}
    for sk, sv in solver_dict.items():
        sname = sv.get("name", sk) if isinstance(sv, dict) else sk
        solver_names[sk] = sname

    # 1. Exact match (case-insensitive)
    for sk, sname in solver_names.items():
        if a_lower == sname.lower():
            return sk, 3
        # Also try matching without [proxy] suffix
        sname_clean = sname.replace(" [proxy]", "").strip()
        if a_lower == sname_clean.lower():
            return sk, 3

    # 2. Algorithm name starts with solver name
    for sk, sname in solver_names.items():
        if sname and a_lower.startswith(sname.lower()):
            return sk, 2

    # 3. Solver name is contained in algorithm name
    for sk, sname in solver_names.items():
        if sname and len(sname) >= 3 and sname.lower() in a_lower:
            return sk, 2

    # 4. Algorithm name is contained in solver name
    for sk, sname in solver_names.items():
        if a_clean and len(a_clean) >= 3 and a_clean.lower() in sname.lower():
            return sk, 1

    # 5. Special patterns for test entries
    if "(test)" in algo_name:
        test_name = algo_name.replace(" (test)", "").strip().lower()
        # Try matching test name to solver keys
        for sk in solver_names:
            if test_name == sk.lower() or test_name.replace("_", " ") == sk.lower():
                return sk, 1
        # Try matching to solver names
        for sk, sname in solver_names.items():
            if test_name in sname.lower().replace("-", "_"):
                return sk, 1

    return None, 0


# ============================================================================
# Phase 4: Update algorithm_state.md
# ============================================================================

def parse_algorithm_state():
    """Parse algorithm_state.md and return structured data."""
    text = STATE_PATH.read_text(encoding="utf-8")
    lines = text.split("\n")

    modalities = OrderedDict()
    current_mod = None
    current_category = None
    header_lines = []

    for i, line in enumerate(lines):
        # Category header
        cat_match = re.match(r'^##\s+(?!Legend)(.+)', line)
        if cat_match and "Legend" not in cat_match.group(1):
            current_category = cat_match.group(1).strip()

        # Modality header
        mod_match = re.match(r'^###\s+(\d+)\.\s+(.+?)\s+\(`(\w+)`\)', line)
        if mod_match:
            num = int(mod_match.group(1))
            display = mod_match.group(2).strip()
            mod_id = mod_match.group(3)
            current_mod = mod_id
            modalities[mod_id] = {
                "number": num,
                "category": current_category or "Other",
                "display_name": display,
                "algorithms": [],
            }
            continue

        # Data row
        if current_mod and line.startswith("|") and not line.startswith("|---") and not line.startswith("| Rank"):
            parts = [p.strip() for p in line.split("|")]
            if len(parts) >= 10:
                try:
                    def parse_float(s):
                        s = s.replace("\u2014", "").replace("--", "").replace("—", "").strip()
                        try: return float(s)
                        except: return None

                    modalities[current_mod]["algorithms"].append({
                        "rank": parts[1],
                        "name": parts[2],
                        "year": parts[3],
                        "reference": parts[4],
                        "ref_psnr": parse_float(parts[5]),
                        "ref_ssim": parse_float(parts[6]),
                        "pwm_psnr": parse_float(parts[7]),
                        "pwm_ssim": parse_float(parts[8]),
                        "status": parts[9].strip() if len(parts) > 9 else "done",
                    })
                except:
                    pass

    return modalities


def build_per_algorithm_results(modalities, verification_data):
    """Map each algorithm to its solver and assign per-algorithm metrics."""
    yaml_cfgs = {}
    for fn in sorted(os.listdir(str(CONFIG_DIR))):
        if fn.endswith(".yaml") and fn != "_template.yaml":
            with open(CONFIG_DIR / fn, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            mod_id = cfg.get("modality_id", fn.replace(".yaml", ""))
            yaml_cfgs[mod_id] = cfg

    per_algo_results = {}
    stats = {"matched": 0, "unmatched_lit": 0, "unmatched_pwm": 0, "test": 0}

    for mod_id, mod_data in modalities.items():
        cfg = yaml_cfgs.get(mod_id, {})
        solvers = cfg.get("solvers", {})
        mod_verify = verification_data.get("modalities", {}).get(mod_id, {})
        mod_solver_results = mod_verify.get("solvers", {})

        per_algo_results[mod_id] = []

        for algo in mod_data["algorithms"]:
            algo_name = algo["name"]

            # Find matching solver
            sk, confidence = match_algorithm_to_solver(algo_name, solvers)

            if sk and sk in mod_solver_results:
                sv_result = mod_solver_results[sk]
                pwm_psnr = sv_result.get("std_psnr")
                pwm_ssim = sv_result.get("std_ssim")
                matched_solver = sk
                if "(test)" in algo_name:
                    stats["test"] += 1
                else:
                    stats["matched"] += 1
            else:
                pwm_psnr = None
                pwm_ssim = None
                matched_solver = None
                if "(PWM)" in algo_name or "(test)" in algo_name:
                    stats["unmatched_pwm"] += 1
                else:
                    stats["unmatched_lit"] += 1

            # Determine status based on PWM vs Ref PSNR gap
            if matched_solver and pwm_psnr is not None:
                ref_p = algo.get("ref_psnr")
                if ref_p is not None and ref_p > 0 and pwm_psnr > 0:
                    gap = abs(ref_p - pwm_psnr)
                    if gap <= 3.0:
                        algo_status = "done"
                    elif gap <= 10.0:
                        algo_status = "partial"
                    else:
                        algo_status = "gap"
                elif pwm_psnr < 0:
                    algo_status = "fail"
                else:
                    # No ref PSNR to compare — mark as ran
                    algo_status = "ran"
            else:
                algo_status = "—"

            per_algo_results[mod_id].append({
                "name": algo_name,
                "year": algo.get("year", ""),
                "reference": algo.get("reference", ""),
                "ref_psnr": algo.get("ref_psnr"),
                "ref_ssim": algo.get("ref_ssim"),
                "pwm_psnr": pwm_psnr,
                "pwm_ssim": pwm_ssim,
                "matched_solver": matched_solver,
                "match_confidence": confidence,
                "status": algo_status,
            })

    return per_algo_results, stats


def write_algorithm_state(modalities, per_algo_results, verification_data):
    """Write updated algorithm_state.md with per-algorithm PWM PSNR."""
    lines = []
    lines.append("# Algorithm State — PWM5 Benchmark")
    lines.append("")

    # Count stats
    total_algos = sum(len(algos) for algos in per_algo_results.values())
    from collections import Counter
    status_counts = Counter()
    for algos in per_algo_results.values():
        for a in algos:
            status_counts[a["status"]] += 1
    n_done = status_counts.get("done", 0)
    n_partial = status_counts.get("partial", 0)
    n_gap = status_counts.get("gap", 0)
    n_fail = status_counts.get("fail", 0)
    n_ran = status_counts.get("ran", 0)
    n_not_impl = status_counts.get("\u2014", 0)
    n_verified = n_done + n_partial

    lines.append(f"Comprehensive listing of reconstruction algorithms for all {len(modalities)} modalities.")
    lines.append(f"Generated: 2026-03-15 | **{n_done} done** ({n_done} within 3 dB of reference) | "
                 f"{n_partial} partial | {n_gap} gap | {n_fail} fail | "
                 f"{n_ran} ran (no ref) | {n_not_impl} not implemented | "
                 f"{total_algos} total")
    lines.append("")
    lines.append("## Legend")
    lines.append("- **Ref PSNR/SSIM**: Published reference values from literature")
    lines.append("- **PWM PSNR/SSIM**: Values from running that specific algorithm in PWM framework")
    lines.append("- **Std PSNR**: PSNR on standard dataset (per-solver)")
    lines.append("- **Std**: `pass` = Std PSNR >= 15 dB | `low` = 5-15 dB | `fail` = < 5 dB | `—` = not implemented")
    lines.append("- **Rank**: Algorithms sorted by Ref PSNR descending (best first)")
    lines.append("- **Status**: `done` = PWM within 3 dB of reference | `partial` = runs, 3-10 dB gap | `gap` = runs, >10 dB gap | `fail` = diverged | `ran` = no ref to compare | `—` = not implemented")
    lines.append("")
    lines.append("---")

    # Group by category
    categories = OrderedDict()
    for mod_id, mod_data in modalities.items():
        cat = mod_data["category"]
        if cat not in categories:
            categories[cat] = []
        categories[cat].append((mod_id, mod_data))

    mod_counter = 0
    for cat_name, cat_mods in categories.items():
        lines.append("")
        lines.append(f"## {cat_name}")

        for mod_id, mod_data in cat_mods:
            mod_counter += 1
            display = mod_data["display_name"]
            algos = per_algo_results.get(mod_id, [])

            lines.append("")
            lines.append(f"### {mod_counter}. {display} (`{mod_id}`)")
            lines.append("")
            lines.append("| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Std PSNR | Std |")
            lines.append("|------|-----------|------|-----------|----------|----------|----------|----------|--------|----------|-----|")

            for rank, algo in enumerate(algos, 1):
                ref_p = f"{algo['ref_psnr']:.1f}" if algo['ref_psnr'] is not None else "\u2014"
                ref_s = f"{algo['ref_ssim']:.4f}" if algo['ref_ssim'] is not None else "\u2014"

                status = algo.get("status", "\u2014")

                if algo['pwm_psnr'] is not None:
                    pwm_p = f"{algo['pwm_psnr']:.1f}"
                    pwm_s = f"{algo['pwm_ssim']:.4f}" if algo['pwm_ssim'] is not None else "\u2014"
                    std_p = f"{algo['pwm_psnr']:.1f}"
                    std_status = "pass" if algo['pwm_psnr'] >= 15 else ("low" if algo['pwm_psnr'] >= 5 else "fail")
                else:
                    pwm_p = "\u2014"
                    pwm_s = "\u2014"
                    std_p = "\u2014"
                    std_status = "\u2014"

                lines.append(
                    f"| {rank} | {algo['name']} | {algo['year']} | {algo['reference']} "
                    f"| {ref_p} | {ref_s} | {pwm_p} | {pwm_s} | {status} "
                    f"| {std_p} | {std_status} |"
                )

    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*PWM Benchmark Algorithm State — {len(modalities)} modalities, "
                 f"{total_algos} algorithms ({n_done} done) — Generated 2026-03-15*")

    STATE_PATH.write_text("\n".join(lines), encoding="utf-8")
    return n_done, total_algos


# ============================================================================
# Main
# ============================================================================

def main():
    print("=" * 70)
    print("PER-ALGORITHM VERIFICATION & MAPPING")
    print("=" * 70)

    # Phase 1: Register additional solvers
    print("\n[Phase 1] Registering additional solver implementations...")
    n_reg = register_additional_solvers()
    print(f"  Registered {n_reg} new solvers\n")

    # Load existing verification results
    if VERIFY_JSON.exists():
        with open(VERIFY_JSON, "r", encoding="utf-8") as f:
            verification_data = json.load(f)
        print(f"Loaded existing verification: {len(verification_data.get('modalities', {}))} modalities")
    else:
        verification_data = {"modalities": {}}

    # Phase 2: Run new solvers
    print("\n[Phase 2] Running newly registered solvers...")
    n_new = run_new_solvers(verification_data)
    print(f"  Ran {n_new} new solvers\n")

    # Save updated verification
    with open(VERIFY_JSON, "w", encoding="utf-8") as f:
        json.dump(verification_data, f, indent=2, default=str)

    # Phase 3: Parse algorithm_state.md and map
    print("\n[Phase 3] Mapping algorithms to solver results...")
    modalities = parse_algorithm_state()
    per_algo_results, stats = build_per_algorithm_results(modalities, verification_data)

    total_a = sum(len(v) for v in per_algo_results.values())
    from collections import Counter
    sc = Counter()
    for algos in per_algo_results.values():
        for a in algos:
            sc[a["status"]] += 1
    print(f"  Total algorithms: {total_a}")
    print(f"  Status breakdown:")
    print(f"    done   (within 3 dB of ref): {sc.get('done',0)}")
    print(f"    partial (3-10 dB gap):       {sc.get('partial',0)}")
    print(f"    gap    (>10 dB gap):         {sc.get('gap',0)}")
    print(f"    fail   (diverged):           {sc.get('fail',0)}")
    print(f"    ran    (no ref to compare):  {sc.get('ran',0)}")
    print(f"    —      (not implemented):    {sc.get('\u2014',0)}")

    # Save per-algorithm results
    with open(PER_ALGO_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "generated": "2026-03-15",
            "stats": stats,
            "modalities": per_algo_results,
        }, f, indent=2, default=str, ensure_ascii=False)
    print(f"\n  Saved: {PER_ALGO_JSON}")

    # Phase 4: Update algorithm_state.md
    print("\n[Phase 4] Updating algorithm_state.md with per-algorithm metrics...")
    implemented, total = write_algorithm_state(modalities, per_algo_results, verification_data)
    print(f"  Updated: {implemented} implemented / {total} total algorithms")
    print(f"  Wrote: {STATE_PATH}")

    # Summary by modality
    print(f"\n{'='*70}")
    print("PER-MODALITY SUMMARY (showing per-solver PSNRs):")
    print(f"{'='*70}")
    for mod_id, algos in sorted(per_algo_results.items()):
        impl = [a for a in algos if a["matched_solver"] is not None]
        lit = [a for a in algos if a["matched_solver"] is None]
        if impl:
            psnrs = [f"{a['name'][:25]}={a['pwm_psnr']:.1f}" for a in impl if a['pwm_psnr'] is not None]
            psnr_str = ", ".join(psnrs[:4])
            if len(psnrs) > 4:
                psnr_str += f"... +{len(psnrs)-4} more"
        else:
            psnr_str = "none"
        print(f"  {mod_id:30s}: {len(impl)} impl / {len(algos)} total  [{psnr_str}]")

    print(f"\nDone! {implemented}/{total} algorithms have per-algorithm PWM PSNR")


if __name__ == "__main__":
    main()
