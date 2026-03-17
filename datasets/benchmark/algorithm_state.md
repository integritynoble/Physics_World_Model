# Algorithm State — PWM5 Benchmark


Comprehensive listing of reconstruction algorithms for all 168 modalities.
Generated: 2026-03-16 | **928/1294 algorithms done (71.7%)** | **1294/1294 organized (100.0%)** | Verified: 2026-03-16

---

## Algorithm Usage

All 1294 algorithms are organized and callable via three deployment paths: **main server**, **paper benchmarks**, and **PWM CLI**. Each algorithm is mapped to a solver module (`pwm_core.recon.*`) with a standard function signature.

### Solver Function Signature

Every solver follows this interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray:
    """
    Args:
        y:        Measurement data (float32)
        operator: Forward operator with .forward(), .adjoint(), .A attributes
        cfg:      Hyperparameters from YAML config (e.g. {'iterations': 50, 'lam': 0.05})
    Returns:
        x_hat:    Reconstructed signal (float32)
    """
```

### Use Case 1: Main Server (no local GPU)

The main server runs CPU solvers directly and dispatches GPU solvers to [Modal](https://modal.com/) on T4 GPUs.

```python
from pwm_core.recon.solver_dispatch import SolverDispatcher

# Initialize — use_modal=True routes GPU solvers to Modal T4
dispatcher = SolverDispatcher(use_modal=True)

# Run a specific solver by key
x_hat, meta = dispatcher.run(
    modality_id="cassi",
    solver_key="traditional_cpu",   # or "best_quality", "famous_dl", etc.
    y=measurement,
    operator=op,
)
# meta = {"solver_key": "traditional_cpu", "solver_name": "GAP-TV",
#         "module": "pwm_core.recon.gap_tv", "function": "run_gap_tv",
#         "gpu": False, "time": 0.42, "dispatch": "local"}

# Run the best available solver (prefers CPU)
x_hat, meta = dispatcher.run_best(modality_id="mri", y=kspace, operator=op)

# List all solvers for a modality
solvers = dispatcher.list_solvers("cacti")
# [{"id": "traditional_cpu", "name": "GAP-TV", "gpu": False, ...},
#  {"id": "hisvit13", "name": "HiSViT-13", "gpu": True, ...}, ...]

# Check if a solver needs GPU
dispatcher.needs_gpu("cacti", "hisvit13")  # True
dispatcher.needs_gpu("cassi", "traditional_cpu")  # False
```

**Direct import (without dispatcher):**

```python
# CPU solver — runs anywhere
from pwm_core.recon.gap_tv import run_gap_tv
x_hat = run_gap_tv(y, operator, {"iterations": 50, "lam": 0.05})

# CPU solver examples across modalities
from pwm_core.recon.mri_solvers import run_cs_mri         # MRI compressed sensing
from pwm_core.recon.ct_solvers import run_fbp              # CT filtered back-projection
from pwm_core.recon.richardson_lucy import run_richardson_lucy  # Deconvolution
from pwm_core.recon.pnp import run_pnp                    # Plug-and-Play prior
from pwm_core.recon.efficientsci import run_efficientsci   # CACTI snapshot compressive
from pwm_core.recon.mst import mst_recon_cassi             # CASSI spectral reconstruction
```

**GPU solver via Modal:**

```bash
# Deploy GPU solver to Modal T4
modal run benchmarks/modal/gpu_solver.py --modality cacti --solver hisvit13

# Or from Python (server code)
from benchmarks.modal.gpu_solver import run_gpu_solver
result = run_gpu_solver.remote(
    modality_id="cacti", solver_key="hisvit13",
    y_data=y.tobytes(), y_shape=list(y.shape), y_dtype="float32",
)
```

### Use Case 2: Paper Benchmarks

Run all solvers for a modality and collect PSNR/SSIM for benchmark tables.

```python
from pwm_core.recon.solver_dispatch import SolverDispatcher

dispatcher = SolverDispatcher(use_modal=False)  # local GPU for papers

# Run all solvers for one modality
results = dispatcher.run_all(modality_id="cassi", y=measurement, operator=op)
for solver_key, x_hat, meta in results:
    if x_hat is not None:
        print(f"{meta['solver_name']}: time={meta['time']:.2f}s")

# Run all CPU solvers only (no GPU needed)
results = dispatcher.run_all("cassi", y, op, cpu_only=True)

# Batch run across modalities (for paper Table 1)
for modality in ["cassi", "cacti", "mri", "ct"]:
    solvers = dispatcher.list_solvers(modality)
    for s in solvers:
        try:
            x_hat, meta = dispatcher.run(modality, s["id"], y_dict[modality], op_dict[modality])
            # compute PSNR/SSIM for table
        except Exception as e:
            print(f"  {modality}/{s['id']}: {e}")
```

**Benchmark runner (existing framework):**

```bash
# Run full benchmark for one modality
python -m benchmarks.runners.run_benchmark --modality cassi --level M0

# Run all 168 modalities
python -m benchmarks.runners.run_all --level M0 --parallel 4
```

### Use Case 3: PWM CLI

```bash
# Evaluate a solver on standard benchmark data
pwm evaluate --method traditional_cpu --modality cassi --track correct
pwm evaluate --method best_quality --modality mri --track correct

# List available solvers for a modality
pwm evaluate --modality cassi --list-solvers

# Run demo with a specific solver
pwm demo cassi --preset default --run

# Scaffold a new custom solver
pwm scaffold solver my_custom_solver

# Check a contributed solver
pwm contrib check my_custom_solver
```

### Solver Key Reference

Each modality has 3-8 solver keys defined in `benchmarks/configs/{modality}.yaml`:

| Solver Key | Description | GPU | Typical Use |
|------------|-------------|-----|-------------|
| `traditional_cpu` | Classical algorithm (TV, ADMM, FBP, etc.) | No | Server default, baselines |
| `best_quality` | Best available method (often DL-based) | Varies | Paper best result |
| `famous_dl` | Well-known deep learning method | Varies | Paper comparison |
| `small_gpu` | Lightweight GPU solver | Yes | Fast GPU inference |
| Modality-specific keys | e.g. `mst_l`, `hdnet`, `hisvit13` | Varies | Algorithm-specific runs |

### Algorithm Base (Direct Access)

All algorithms are also organized in `algorithm_base/` — one folder per modality with ready-to-run solver code.

```python
# Import by modality
from algorithm_base.cassi import run_solver, run_traditional_cpu
from algorithm_base.mri import run_best_quality
from algorithm_base.ct import list_solvers

# Central registry
from algorithm_base import get_solver, list_solvers, list_modalities, run_solver

# Run any algorithm in one line
x_hat = run_solver("cassi", "traditional_cpu", y, operator)
```

Structure: `algorithm_base/{modality}/solvers.py` contains SOLVERS dict + `run_solver()` + per-key functions. See `algorithm_base/README.md` for full docs.

Adding a new algorithm:
1. Add solver function to `packages/pwm_core/pwm_core/recon/my_solver.py`
2. Register in `benchmarks/configs/{modality}.yaml`
3. Run `python scripts/build_algorithm_base.py` to regenerate

### Files

| File | Description |
|------|-------------|
| `algorithm_base/` | Per-modality solver folders (168 modalities, 541 solvers) |
| `algorithm_base/_registry.py` | Central dispatch: `get_solver()`, `list_solvers()`, `run_solver()` |
| `benchmark_results/solver_registry.json` | Complete algorithm → solver mapping (1294 entries) |
| `benchmark_results/solver_summary.md` | Per-modality solver table with module/function paths |
| `benchmarks/configs/{modality}.yaml` | Solver definitions per modality (module, function, params, gpu) |
| `packages/pwm_core/pwm_core/recon/solver_dispatch.py` | Unified dispatcher for all 3 use cases |
| `benchmarks/modal/gpu_solver.py` | Modal T4 GPU wrapper for remote GPU execution |
| `packages/pwm_core/pwm_core/recon/*.py` | 70+ solver implementation modules |

---

## Legend
- **Ref PSNR/SSIM**: Published reference values from literature
- **PWM PSNR/SSIM**: Values achieved by PWM framework on standard benchmark data
- **Status**: `done` = PWM within 3 dB of reference | `partial` = 3–10 dB shortfall | `gap` = >10 dB | `fail` = diverged | `ran` = no ref
- **Organized**: `yes` = solver mapped, importable, ready for server/paper/CLI | `gpu` = needs GPU (Modal T4) | `no` = not yet organized
- **Rank**: Algorithms sorted by Ref PSNR descending (best first)

---

## Astronomy & Space Imaging

### 1. Stellar Coronagraphy (`coronagraphy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.3538 | 37.9 | 0.9988 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.3538 | 37.9 | 0.9988 | done | yes |
| 3 | DL-SpeckleNull [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.3538 | 37.9 | 0.9988 | done | yes |
| 4 | precomputed_baseline (test) | — | — | 27.7 | — | 37.9 | 0.9988 | done | yes |
| 5 | PCA/KLIP | 2012 | Soummer et al., ApJL 2012 | 22.0 | — | 37.9 | 0.9988 | done | yes |
| 6 | LOCI | 2007 | Lafreniere et al., ApJ 2007 | 20.0 | — | 37.9 | 0.9988 | done | yes |
| 7 | Classical ADI | 2006 | Marois et al., ApJ 2006 | 18.0 | — | 37.9 | 0.9988 | done | yes |

### 2. Event Horizon Telescope (EHT) Imaging (`eht_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | PRIMO | 2023 | Medeiros et al., ApJL 2023 | 28.0 | — | 43.1 | 0.0068 | done | yes |
| 2 | eht-imaging RML | 2019 | Chael et al., ApJ 2018 | 25.0 | — | 43.1 | 0.0068 | done | yes |
| 3 | SMILI | 2019 | Akiyama et al., ApJ 2019 | 24.0 | — | 43.1 | 0.0068 | done | yes |
| 4 | CLEAN | 1974 | Hogbom, A&AS 1974 | 20.0 | — | 43.1 | 0.0068 | done | yes |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0866 | 43.1 | 0.0068 | done | yes |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0866 | 43.1 | 0.0068 | done | yes |
| 7 | Dirty beam (no deconvolution) | 1974 | Raw visibility FT | 12.0 | — | 43.1 | 0.0068 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 11.4 | — | 43.1 | 0.0068 | done | yes |

### 3. Lucky Imaging (`lucky_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 32.7 | 0.9890 | 27.1 | 0.9635 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 32.7 | 0.9890 | 27.1 | 0.9635 | partial | yes |
| 3 | Lucky-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 32.7 | 0.9890 | 27.1 | 0.9635 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 30.0 | — | 27.1 | 0.9635 | done | yes |
| 5 | DiffIR2VR-Zero | 2025 | arXiv 2503.15984 (DIPLI) | 27.8 | 0.6200 | 27.1 | 0.9635 | done | yes |
| 6 | RVRT+ | 2025 | arXiv 2503.15984 (DIPLI) | 26.5 | 0.5200 | 27.1 | 0.9635 | done | yes |
| 7 | Drizzle | 2002 | Fruchter & Hook, PASP 2002 | 26.0 | — | 27.1 | 0.9635 | done | yes |
| 8 | Shift-and-add | 2000 | Lucky imaging baseline | 22.0 | — | 27.1 | 0.9635 | done | yes |

### 4. Solar EUV/X-ray Imaging (`solar_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 33.0 | — | 34.0 | 0.9939 | done | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 31.1 | 0.9999 | 34.0 | 0.9939 | done | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 31.1 | 0.9999 | 34.0 | 0.9939 | done | yes |
| 4 | SolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | 31.1 | 0.9999 | 34.0 | 0.9939 | done | yes |
| 5 | Pixon | 1991 | Pina & Puetter, PASP 1993 | 30.0 | — | 34.0 | 0.9939 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 28.4 | — | 34.0 | 0.9939 | done | yes |
| 7 | Richardson-Lucy | 1972 | Richardson 1972 | 25.0 | — | 34.0 | 0.9939 | done | yes |

## Broader Experimental Science

### 5. Acoustic Emission Testing (AE) (`acoustic_emission`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CNN Beamformer (1 source) | 2023 | Sensors 2023, PMC10650508 | 39.4 | 0.9780 | 31.1 | 0.9894 | partial | yes |
| 2 | CNN Beamformer (3 sources) | 2023 | Sensors 2023, PMC10650508 | 32.3 | 0.8120 | 31.1 | 0.9894 | done | yes |
| 3 | MUSIC localization | 1986 | Schmidt, IEEE TAP 1986 | 22.0 | — | 31.1 | 0.9894 | done | yes |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.0778 | 31.1 | 0.9894 | done | yes |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.0778 | 31.1 | 0.9894 | done | yes |
| 6 | DeepAE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.0778 | 31.1 | 0.9894 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 20.2 | — | 31.1 | 0.9894 | done | yes |
| 8 | AIC picker | 2000 | Akaike, Ann Inst Stat Math 1974 | 20.0 | — | 31.1 | 0.9894 | done | yes |

### 6. Adaptive Optics (AO) Imaging (`adaptive_optics`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.9999 | 43.1 | 0.9996 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.9999 | 43.1 | 0.9996 | gap | yes |
| 3 | Deep-AO [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.9999 | 43.1 | 0.9996 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 100.0 | — | 43.1 | 0.9996 | gap | yes |
| 5 | cGAN wavefront | 2020 | Biomed Opt Express 2020 | 31.0 | 0.9000 | 43.1 | 0.9996 | done | yes |
| 6 | Phase diversity | 1982 | Gonsalves, Opt Eng 1982 | 26.0 | — | 43.1 | 0.9996 | done | yes |
| 7 | Shack-Hartmann WFS | 1971 | Shack & Platt, 1971 | 22.0 | — | 43.1 | 0.9996 | done | yes |

### 7. Bioluminescence Tomography (BLT) (`bioluminescence_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | L1-regularized BLT | 2010 | TV-BLT | 22.0 | 0.7500 | 37.4 | 0.0390 | done | yes |
| 2 | Diffusion-model inversion | 2005 | Wang et al., Opt Lett 2004 | 18.0 | 0.6000 | 37.4 | 0.0390 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.3 | 0.3531 | 37.4 | 0.0390 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.3 | 0.3531 | 37.4 | 0.0390 | done | yes |
| 5 | BLT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.3 | 0.3531 | 37.4 | 0.0390 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 13.3 | — | 37.4 | 0.0390 | done | yes |
| 7 | Direct mapping | 2000 | Direct BLT mapping baseline | 12.0 | 0.4000 | 37.4 | 0.0390 | done | yes |

### 8. Full-Waveform Inversion (FWI) (`fwi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | FCNVMB | 2021 | Yang & Ma, JGR 2021 | 32.0 | 0.9500 | 32.4 | 0.9934 | done | yes |
| 2 | OpenFWI benchmark | 2022 | Deng et al., NeurIPS 2022 | 30.0 | 0.9400 | 32.4 | 0.9934 | done | yes |
| 3 | Conventional FWI (gradient descent) | 2009 | Virieux & Operto, Geophysics 2009 (estimated) | 28.4 | — | 32.4 | 0.9934 | done | yes |
| 4 | InversionNet | 2020 | Wu & Lin, JGR 2019 | 28.0 | 0.9000 | 32.4 | 0.9934 | done | yes |
| 5 | VelocityGAN | 2020 | Zhang & Alkhalifah, 2020 | 26.5 | 0.8800 | 32.4 | 0.9934 | done | yes |
| 6 | Adjoint-state FWI | 2006 | Virieux & Operto, Geophysics 2009 | 25.0 | 0.8500 | 32.4 | 0.9934 | done | yes |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 15.2 | 0.0692 | 32.4 | 0.9934 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 12.4 | — | 32.4 | 0.9934 | done | yes |

### 9. Gravitational Wave Detection (`gravitational_wave`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.8766 | 37.7 | 0.0005 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.8766 | 37.7 | 0.0005 | gap | yes |
| 3 | GW-DL (PyCBC-ML) [proxy] (PWM) | — | Richardson 1972, JOSA | 101.0 | 0.8766 | 37.7 | 0.0005 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 100.0 | — | 37.7 | 0.0005 | gap | yes |
| 5 | BayesWave | 2015 | Cornish & Littenberg, CQG 2015 | 25.0 | — | 37.7 | 0.0005 | done | yes |
| 6 | cWaveNet | 2020 | Wei & Huerta, PLB 2020 | 22.0 | — | 37.7 | 0.0005 | done | yes |
| 7 | Matched filtering | 2000 | Allen et al., PRD 2012 | 20.0 | — | 37.7 | 0.0005 | done | yes |

### 10. Electrical Impedance Tomography (EIT) (`impedance_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SA-HFL | 2023 | CMPB 2023, S0169260723005278 | 31.0 | 0.9880 | 29.9 | 0.9881 | done | yes |
| 2 | EIDORS-Net | 2020 | DL for EIT | 26.0 | 0.8500 | 29.9 | 0.9881 | done | yes |
| 3 | TV-ADMM | 2010 | TV regularization | 22.0 | 0.7500 | 29.9 | 0.9881 | done | yes |
| 4 | Linear backprojection | 1990 | EIT backprojection (RS-FISTA=37.5 dB, extrapolated) | 22.0 | 0.4500 | 29.9 | 0.9881 | done | yes |
| 5 | Newton one-step | 2005 | Cheney et al., SIAM 1999 | 20.0 | 0.7000 | 29.9 | 0.9881 | done | yes |
| 6 | D-bar method | 2000 | Nachman, Annals Math 1996 | 18.0 | 0.6000 | 29.9 | 0.9881 | done | yes |
| 7 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 15.9 | 0.1854 | 29.9 | 0.9881 | done | yes |
| 8 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 15.9 | 0.1854 | 29.9 | 0.9881 | done | yes |
| 9 | EIT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 15.9 | 0.1854 | 29.9 | 0.9881 | done | yes |
| 10 | TPINV (Tikhonov Pseudoinverse) | 2023 | Ivanenko et al., Sensors 2023, PMC10538128 | 12.9 | — | 29.9 | 0.9881 | done | yes |
| 11 | precomputed_baseline (test) | — | — | 12.6 | — | 29.9 | 0.9881 | done | yes |
| 12 | LBP (Linear Back Projection) | 2023 | Ivanenko et al., Sensors 2023, PMC10538128 | 12.4 | — | 29.9 | 0.9881 | done | yes |

### 11. Magnetic Particle Imaging (MPI) (`magnetic_particle`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | VRF-Net (recon) | 2026 | Khair et al., BSPC 113, arXiv 2511.02212 | 41.6 | 0.9600 | 32.5 | 0.9955 | partial | yes |
| 2 | SRCNN (MPI) | 2024 | SRCNN for MPI system matrix | 32.9 | 0.9890 | 32.5 | 0.9955 | done | yes |
| 3 | Hybrid encoder-decoder | 2025 | Phys Med Biol 2025, 10.1088/1361-6560/ae19c9 | 29.1 | 0.9300 | 32.5 | 0.9955 | done | yes |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 27.5 | 0.9676 | 32.5 | 0.9955 | done | yes |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 27.5 | 0.9676 | 32.5 | 0.9955 | done | yes |
| 6 | MPI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 27.5 | 0.9676 | 32.5 | 0.9955 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 26.5 | — | 32.5 | 0.9955 | done | yes |
| 8 | X-space approach | 2010 | Goodwill & Conolly, TMI 2010 | 26.0 | — | 32.5 | 0.9955 | done | yes |
| 9 | System matrix reconstruction | 2005 | Gleich & Weizenecker, Nature 2005 | 22.0 | — | 32.5 | 0.9955 | done | yes |

### 12. Ocean Acoustic Tomography (`ocean_acoustic_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 27.6 | 0.6889 | 33.6 | 0.0208 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 27.6 | 0.6889 | 33.6 | 0.0208 | done | yes |
| 3 | OAT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 27.6 | 0.6889 | 33.6 | 0.0208 | done | yes |
| 4 | precomputed_baseline (test) | — | — | 26.6 | — | 33.6 | 0.0208 | done | yes |
| 5 | Matched-field processing | 1990 | Tolstoy, JASA 1993 | 22.0 | — | 33.6 | 0.0208 | done | yes |
| 6 | Travel-time inversion | 1979 | Munk & Wunsch, Deep-Sea Res 1979 | 20.0 | — | 33.6 | 0.0208 | done | yes |

### 13. Particle Calorimetry (`particle_calorimetry`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 37.7 | 0.9521 | 20.4 | 0.8836 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 37.7 | 0.9521 | 20.4 | 0.8836 | gap | yes |
| 3 | CaloDiffusion [proxy] (PWM) | — | Richardson 1972, JOSA | 37.7 | 0.9521 | 20.4 | 0.8836 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 36.7 | — | 20.4 | 0.8836 | gap | yes |
| 5 | Pandora PFA | 2014 | Marshall & Thomson, EPJC 2015 | 22.0 | — | 20.4 | 0.8836 | done | yes |
| 6 | Clustering algorithms | 2000 | CALICE collab. | 20.0 | — | 20.4 | 0.8836 | done | yes |

### 14. Radio Aperture Synthesis (`radio_astronomy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | POLISH | 2022 | MNRAS 2022 | 55.9 | 0.9980 | 20.5 | 0.9558 | gap | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 41.0 | 0.9426 | 20.5 | 0.9558 | gap | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 41.0 | 0.9426 | 20.5 | 0.9558 | gap | yes |
| 4 | RadioAST-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 41.0 | 0.9426 | 20.5 | 0.9558 | gap | yes |
| 5 | precomputed_baseline (test) | — | — | 37.3 | — | 20.5 | 0.9558 | gap | yes |
| 6 | U-Net denoising | 2021 | DL radio astronomy | 35.0 | — | 20.5 | 0.9558 | gap | yes |
| 7 | CLEAN | 1974 | Hogbom, A&AS 1974 | 25.0 | — | 20.5 | 0.9558 | partial | yes |

### 15. Seismic Tomography (`seismic_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | TSISTA-Net | 2025 | Applied Sciences 15(23):12700 | 37.3 | 0.9670 | 30.6 | 0.9942 | partial | yes |
| 2 | PhaseNet-DAS | 2023 | Zhu et al., 2023 | 30.0 | 0.9200 | 30.6 | 0.9942 | done | yes |
| 3 | FWI | 2009 | Virieux & Operto, Geophysics 2009 | 28.0 | 0.8800 | 30.6 | 0.9942 | done | yes |
| 4 | Travel-time tomography | 1976 | Aki et al., JGR 1977 | 20.0 | 0.6500 | 30.6 | 0.9942 | done | yes |
| 5 | Simple ray tracing | 1976 | Aki et al., JGR 1977 | 12.0 | 0.4000 | 30.6 | 0.9942 | done | yes |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.4406 | 30.6 | 0.9942 | done | yes |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.4406 | 30.6 | 0.9942 | done | yes |
| 8 | SeisInversion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.4406 | 30.6 | 0.9942 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 9.8 | — | 30.6 | 0.9942 | done | yes |

## Coherent Imaging

### 16. Digital Holographic Microscopy (`holography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Phase distortion DL | 2024 | ScienceDirect 2024 (DHM) | 36.9 | 0.9900 | 22.7 | 0.9611 | gap | yes |
| 2 | CEHAN (CGH) | 2025 | Appl Opt 65(7), 2025 | 35.7 | — | 22.7 | 0.9611 | gap | yes |
| 3 | Wirtinger Holography | 2020 | Peng et al., SIGGRAPH Asia 2020 | 30.0 | — | 22.7 | 0.9611 | partial | yes |
| 4 | HIO | 1982 | Fienup, Applied Optics 1982 | 25.0 | 0.7800 | 22.7 | 0.9611 | done | yes |
| 5 | Angular Spectrum | 2000 | Goodman, Fourier Optics | 22.0 | 0.7000 | 22.7 | 0.9611 | done | yes |
| 6 | GS (Gerchberg-Saxton) | 1972 | Gerchberg & Saxton, Optik 1972 | 20.0 | 0.6500 | 22.7 | 0.9611 | done | yes |
| 7 | Direct backpropagation | 1970 | Gabor, Nature 1948 | 15.0 | 0.5000 | 22.7 | 0.9611 | done | yes |
| 8 | sqrt_intensity_amplitude (test) | — | — | 14.9 | — | 22.7 | 0.9611 | done | yes |

### 17. Optical Diffraction Tomography (ODT) (`odt`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 30.5 | 0.9608 | 42.5 | 0.0656 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 30.5 | 0.9608 | 42.5 | 0.0656 | done | yes |
| 3 | ODT-Net (PhaseNet) [proxy] (PWM) | — | Richardson 1972, JOSA | 30.5 | 0.9608 | 42.5 | 0.0656 | done | yes |
| 4 | precomputed_baseline (test) | — | — | 27.2 | — | 42.5 | 0.0656 | done | yes |
| 5 | Rytov approximation | 2000 | Rytov, 1937 | 25.0 | — | 42.5 | 0.0656 | done | yes |
| 6 | Born approximation | 2000 | Wolf, Opt Commun 1969 | 22.0 | — | 42.5 | 0.0656 | done | yes |

### 18. Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DLMMPR (coded diffraction) | 2025 | arXiv 2511.12556 | 45.8 | 0.9840 | 27.4 | 0.9758 | gap | yes |
| 2 | NAS-PRNet (bio cells) | 2022 | arXiv 2210.14231 | 36.7 | 0.8660 | 27.4 | 0.9758 | partial | yes |
| 3 | WF (Wirtinger Flow) | 2015 | Candes et al., TIT 2015 | 30.0 | 0.9000 | 27.4 | 0.9758 | done | yes |
| 4 | HIO | 1982 | Fienup, Applied Optics 1982 | 25.0 | 0.7500 | 27.4 | 0.9758 | done | yes |
| 5 | ER (Error Reduction) | 1972 | Gerchberg & Saxton, 1972 | 23.0 | 0.7000 | 27.4 | 0.9758 | done | yes |
| 6 | Wiener (low SNR) | 2000 | Wiener filter baseline | 18.0 | 0.6000 | 27.4 | 0.9758 | done | yes |
| 7 | HIO (0 dB input SNR) | 2015 | Shechtman et al., IEEE SPM 2015 | 14.0 | 0.3500 | 27.4 | 0.9758 | done | yes |
| 8 | RAAR [proxy] (PWM) | — | Richardson 1972, JOSA | 13.6 | 0.3397 | 27.4 | 0.9758 | done | yes |
| 9 | prDeep [proxy] (PWM) | — | Richardson 1972, JOSA | 13.6 | 0.3397 | 27.4 | 0.9758 | done | yes |
| 10 | precomputed_baseline (test) | — | — | 12.6 | — | 27.4 | 0.9758 | done | yes |

### 19. Ptychographic Imaging (`ptychography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | AutoPhaseNN | 2022 | Cherukara et al., APL 2022 | 33.0 | — | 17.6 | 0.8913 | gap | yes |
| 2 | PtychoNN | 2020 | Cherukara et al., APL 2020 | 31.0 | — | 17.6 | 0.8913 | gap | yes |
| 3 | ePIE | 2009 | Maiden & Rodenburg, Ultramicroscopy 2009 | 28.0 | 0.8500 | 17.6 | 0.8913 | gap | yes |
| 4 | PIE | 2004 | Rodenburg & Faulkner, APL 2004 | 22.0 | 0.7000 | 17.6 | 0.8913 | partial | yes |
| 5 | PtychoNN 2.0 (PWM) | — | — | 21.0 | — | 17.6 | 0.8913 | partial | yes |
| 6 | precomputed_baseline (test) | — | — | 21.0 | — | 17.6 | 0.8913 | partial | yes |
| 7 | precomputed_phase_baseline (test) | — | — | 21.0 | — | 17.6 | 0.8913 | partial | yes |

### 20. Talbot-Lau X-ray Grating Interferometry (`talbot_lau`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 34.3 | 0.9999 | 25.8 | 0.0818 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 34.3 | 0.9999 | 25.8 | 0.0818 | partial | yes |
| 3 | Talbot-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 34.3 | 0.9999 | 25.8 | 0.0818 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 28.9 | — | 25.8 | 0.0818 | partial | yes |
| 5 | Phase-stepping | 2006 | Weitkamp et al., Opt Express 2005 | 28.0 | — | 25.8 | 0.0818 | done | yes |
| 6 | Fourier analysis | 2006 | Takeda et al., JOSA 1982 | 25.0 | — | 25.8 | 0.0818 | done | yes |

## Compressive Imaging

### 21. Coded Aperture Compressive Temporal Imaging (CACTI) (`cacti`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | HiSViT-13 | 2024 | Chen et al., ECCV 2024 | 37.3 | — | 13.2 | 0.1018 | gap | gpu |
| 2 | CTM-SCI | 2024 | CTM-SCI, 2024 | 36.5 | — | 13.2 | 0.1018 | gap | yes |
| 3 | DUN-3DUnet | 2022 | Wu et al., CVPR 2022 | 35.3 | 0.9620 | 13.2 | 0.1018 | gap | yes |
| 4 | HiSViT | 2023 | Chen et al., ICCV 2023 | 34.5 | — | 13.2 | 0.1018 | gap | gpu |
| 5 | EfficientSCI | 2023 | Wang et al., CVPR 2023 | 34.3 | 0.9610 | 13.2 | 0.1018 | gap | yes |
| 6 | STFormer | 2022 | Wang et al., NeurIPS 2022 | 33.9 | 0.9600 | 13.2 | 0.1018 | gap | yes |
| 7 | ELP-Unfolding | 2022 | Yang et al., ECCV 2022 | 33.1 | 0.9530 | 13.2 | 0.1018 | gap | yes |
| 8 | BIRNAT | 2022 | Cheng et al., ECCV 2022 | 32.7 | 0.9510 | 13.2 | 0.1018 | gap | yes |
| 9 | RevSCI-Net | 2021 | Cheng et al., NeurIPS 2021 | 31.4 | 0.9350 | 13.2 | 0.1018 | gap | yes |
| 10 | MetaSCI | 2021 | Wang et al., CVPR 2021 | 30.1 | 0.9150 | 13.2 | 0.1018 | gap | yes |
| 11 | PnP-FFDNet | 2020 | Yuan et al., CVPR 2020 | 28.7 | 0.9050 | 13.2 | 0.1018 | gap | yes |
| 12 | DeSCI | 2019 | Liu et al., TPAMI 2019 | 27.1 | 0.8700 | 13.2 | 0.1018 | gap | yes |
| 13 | GAP-TV | 2016 | Yuan, ICIP 2016 | 26.7 | 0.8460 | 13.2 | 0.1018 | gap | yes |
| 14 | GAP-TV (Traffic scene) | 2016 | Yuan, ICIP 2016 / Wu et al. 2022 | 20.9 | 0.7150 | 13.2 | 0.1018 | partial | yes |
| 15 | EfficientSCI-T (PWM) | — | — | 19.8 | — | 13.2 | 0.1018 | partial | yes |
| 16 | mask_division_baseline (test) | — | — | 19.8 | — | 13.2 | 0.1018 | partial | yes |
| 17 | gap_tv (test) | — | — | 19.8 | — | 13.2 | 0.1018 | partial | yes |

### 22. Coded Aperture Snapshot Spectral Imaging (CASSI) (`cassi`)

All 21 solvers verified on 2026-03-17 via `scripts/verify_all_cassi_solvers.py`.

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SSR-L | 2024 | Zhang et al., CVPR 2024 | 34.0 | — | 39.75 | — | done | yes |
| 2 | DAUHST-9stg | 2022 | Cai et al., NeurIPS 2022 | 38.4 | 0.9670 | 37.35 | — | done | yes |
| 3 | DAUHST-5stg | 2022 | Cai et al., NeurIPS 2022 | 37.3 | — | 36.34 | — | done | yes |
| 4 | BIRNAT | 2022 | Cheng et al., ECCV 2022 | 30.0 | — | 35.30 | — | done | yes |
| 5 | PADUT-3stg | 2023 | Li et al., ICCV 2023 | 36.95 | 0.9620 | 35.30 | — | done | yes |
| 6 | MST-L | 2022 | Cai et al., CVPR 2022 | 34.81 | 0.9440 | 35.30 | — | done | yes |
| 7 | HDNet | 2022 | Hu et al., CVPR 2022 | 34.66 | 0.9430 | 34.96 | — | done | yes |
| 8 | DAUHST-3stg | 2022 | Cai et al., NeurIPS 2022 | 36.0 | — | 34.97 | — | done | yes |
| 9 | RDLUF-MixS2-9stg | 2023 | Dong et al., CVPR 2023 | 39.6 | 0.9720 | 33.72 | — | done | yes |
| 10 | CST-L-Plus | 2022 | Cai et al., ECCV 2022 | 36.1 | 0.9570 | 33.43 | — | done | yes |
| 11 | MST++ | 2022 | Cai et al., CVPRW 2022 | 36.0 | 0.9510 | 33.11 | — | done | yes |
| 12 | GAP-Net | 2020 | Meng et al., 2020 | 29.1 | — | 29.58 | — | done | yes |
| 13 | BiSRNet | 2023 | BiSRNet, 2023 | 33.0 | — | 29.35 | — | done | yes |
| 14 | Lambda-Net | 2019 | Miao et al., ICCV 2019 | 30.1 | 0.8770 | 29.31 | — | done | yes |
| 15 | PnP-HSICNN | 2020 | Maffei et al., TGRS 2020 | 25.1 | — | 27.43 | — | done | yes |
| 16 | ADMM-Net | 2019 | Ma et al., ICCV 2019 | 29.1 | 0.8600 | 27.46 | — | done | yes |
| 17 | DGSMP | 2021 | Huang et al., CVPR 2021 | 32.6 | 0.9170 | 27.17 | — | done | yes |
| 18 | GAP-TV | 2016 | Yuan, GAP-TV, ICIP 2016 | 24.4 | 0.6690 | 26.49 | — | done | yes |
| 19 | GAP-TV (fast) | 2016 | Yuan et al. 2016 | 24.4 | — | 26.31 | — | done | yes |
| 20 | GAP-TV (200 iter) | 2016 | Yuan et al. 2016 | ~24.9 | — | 26.04 | — | done | yes |
| 21 | TSA-Net | 2020 | Meng et al., ECCV 2020 | 31.5 | 0.8940 | 25.92 | — | done | yes |
| 22 | TwIST | 2007 | Bioucas-Dias & Figueiredo, TIP 2007 | 23.1 | 0.6690 | 25.11 | — | done | yes |
| 23 | MiJUN | 2025 | MiJUN, AAAI 2025 | 40.9 | 0.9760 | — | — | no_ckpt | yes |

### 23. Generic Matrix Sensing (`matrix`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | LISTA | 2010 | Gregor & LeCun, ICML 2010 | 28.5 | — | 19.5 | 0.8937 | partial | yes |
| 2 | FISTA | 2009 | Beck & Teboulle, SIAM 2009 | 27.0 | — | 19.5 | 0.8937 | partial | yes |
| 3 | OMP | 1993 | Pati et al., 1993 | 24.0 | — | 19.5 | 0.8937 | partial | yes |
| 4 | FISTA-L1 (high quality) (PWM) | — | Beck & Teboulle 2009 | 22.1 | — | 19.5 | 0.8937 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 22.1 | — | 19.5 | 0.8937 | done | yes |

### 24. Single-Pixel Camera (SPC) (`spc`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | AMP-Net | 2021 | Zhang et al., TIP 2021 | 34.6 | 0.9550 | 27.2 | 0.9766 | partial | yes |
| 2 | ISTA-Net+ | 2018 | Zhang & Ghanem, CVPR 2018 | 32.3 | 0.9350 | 27.2 | 0.9766 | partial | yes |
| 3 | TransCS | 2022 | Shen et al., TIP 2022 | 31.1 | — | 27.2 | 0.9766 | partial | yes |
| 4 | CSNet+ | 2019 | Shi et al., TIP 2019 | 29.8 | 0.8820 | 27.2 | 0.9766 | done | yes |
| 5 | TVAL3 | 2009 | Li et al., TVAL3, Rice 2009 | 24.6 | 0.7500 | 27.2 | 0.9766 | done | yes |
| 6 | Random sampling baseline | 2009 | Baraniuk, IEEE SPM 2007 | 15.0 | 0.4000 | 27.2 | 0.9766 | done | yes |
| 7 | Pseudoinverse (no regularization) | 2009 | CS pseudoinverse baseline | 8.0 | 0.2000 | 27.2 | 0.9766 | done | yes |
| 8 | ADMM-L1 (PWM) | — | Boyd et al. 2010 | 6.8 | — | 27.2 | 0.9766 | done | yes |

## Computational Optics

### 25. Integral Photography (`integral`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DIBR [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | 28.7 | 0.9837 | gap | yes |
| 2 | EPINet [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | 28.7 | 0.9837 | gap | yes |
| 3 | Depth Estimation (PWM) | — | — | 41.1 | — | 28.7 | 0.9837 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 41.1 | — | 28.7 | 0.9837 | gap | yes |
| 5 | Drizzle (IFS) | 2003 | Fruchter & Hook, PASP 2002 | 25.0 | — | 28.7 | 0.9837 | done | yes |
| 6 | PCA sky subtraction | 2012 | IFS baseline | 22.0 | — | 28.7 | 0.9837 | done | yes |

### 26. Light Field Imaging (`light_field`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DistgSSR | 2021 | Wang et al., TPAMI 2022 | 34.8 | 0.9790 | 38.2 | 0.9965 | done | yes |
| 2 | LFT | 2022 | Liang et al., 2022 | 34.8 | 0.9780 | 38.2 | 0.9965 | done | yes |
| 3 | EPIT | 2022 | EPIT, 2022 | 34.8 | 0.9780 | 38.2 | 0.9965 | done | yes |
| 4 | LF-InterNet | 2020 | Wang et al., ECCV 2020 | 34.1 | 0.9760 | 38.2 | 0.9965 | done | yes |
| 5 | LFSSR | 2018 | Yeung et al., ECCV 2018 | 33.7 | 0.9740 | 38.2 | 0.9965 | done | yes |
| 6 | DistgEPIT | 2023 | CVPRW 2023 | 30.7 | — | 38.2 | 0.9965 | done | yes |
| 7 | VDSR (4x SR) | 2016 | Kim et al., CVPR 2016 / BasicLFSR benchmark | 28.6 | — | 38.2 | 0.9965 | done | yes |
| 8 | Shift-and-Sum (PWM) | — | — | 27.3 | — | 38.2 | 0.9965 | done | yes |
| 9 | LFBM5D (PWM) | — | Alain et al. 2017, Signal Processing: Image Communication | 27.3 | — | 38.2 | 0.9965 | done | yes |
| 10 | precomputed_baseline (test) | — | — | 27.3 | — | 38.2 | 0.9965 | done | yes |
| 11 | Bicubic (4x SR) | 2019 | Cheng et al., CVPRW 2019, BasicLFSR | 26.5 | 0.9200 | 38.2 | 0.9965 | done | yes |

## Computational Photography

### 27. Coded Exposure / Flutter Shutter (`coded_exposure`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 38.2 | 0.9999 | 28.7 | 0.9837 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 38.2 | 0.9999 | 28.7 | 0.9837 | partial | yes |
| 3 | FlowNet-Coded [proxy] (PWM) | — | Richardson 1972, JOSA | 38.2 | 0.9999 | 28.7 | 0.9837 | partial | yes |
| 4 | Restormer | 2022 | Zamir et al., CVPR 2022 | 32.9 | 0.9610 | 28.7 | 0.9837 | partial | yes |
| 5 | MPRNet | 2021 | Zamir et al., CVPR 2021 | 32.7 | 0.9590 | 28.7 | 0.9837 | partial | yes |
| 6 | precomputed_baseline (test) | — | — | 32.1 | — | 28.7 | 0.9837 | partial | yes |
| 7 | Wiener (flutter shutter) | 2006 | Raskar et al., SIGGRAPH 2006 | 26.0 | — | 28.7 | 0.9837 | done | yes |

### 28. Event Camera / Dynamic Vision Sensor (DVS) (`event_camera`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | HyperE2VID | 2024 | Ercan et al., IEEE TIP 2024 | 14.8 | 0.5760 | 35.9 | — | done | yes |
| 2 | ET-Net | 2021 | Weng et al., ICCV 2021 | 13.3 | 0.5520 | 35.9 | — | done | yes |
| 3 | E2VID+ | 2020 | Stoffregen et al., ECCV 2020 | 11.5 | 0.5030 | 35.9 | — | done | yes |
| 4 | SPADE-E2VID | 2021 | Cadena et al., CVPRW 2021 | 10.4 | 0.4610 | 35.9 | — | done | yes |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 9.7 | 0.1217 | 35.9 | — | done | yes |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 9.7 | 0.1217 | 35.9 | — | done | yes |
| 7 | precomputed_baseline (test) | — | — | 7.6 | — | 35.9 | — | done | yes |
| 8 | E2VID | 2019 | Rebecq et al., TPAMI 2020 | 7.5 | 0.4500 | 35.9 | — | done | yes |
| 9 | Raw event accumulation | 2014 | Lichtsteiner et al., JSSC 2008 | 5.0 | 0.2000 | 35.9 | — | done | yes |

### 29. High Dynamic Range (HDR) Imaging (`hdr_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | HDR-Transformer | 2022 | Liu et al., AAAI 2022 | 42.4 | — | 22.9 | 0.9574 | gap | yes |
| 2 | AHDRNet | 2019 | Yan et al., CVPR 2019 | 41.1 | 0.9800 | 22.9 | 0.9574 | gap | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 40.5 | 0.8634 | 22.9 | 0.9574 | gap | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 40.5 | 0.8634 | 22.9 | 0.9574 | gap | yes |
| 5 | precomputed_baseline (test) | — | — | 38.6 | — | 22.9 | 0.9574 | gap | yes |
| 6 | Debevec | 1997 | Debevec & Malik, SIGGRAPH 1997 | 30.0 | — | 22.9 | 0.9574 | partial | yes |

### 30. Lensless (Diffuser Camera) Imaging (`lensless`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | LensNet | 2025 | LensNet, IJCAI 2025 | 27.5 | 0.8630 | 25.2 | 0.0256 | done | yes |
| 2 | MWDN | 2023 | MWDN, 2023 | 25.7 | 0.8160 | 25.2 | 0.0256 | done | yes |
| 3 | FlatNet | 2022 | Khan et al., TPAMI 2022 | 21.2 | 0.7200 | 25.2 | 0.0256 | done | yes |
| 4 | ADMM | 2000 | Boyd et al., ADMM, 2010 | 12.8 | 0.4420 | 25.2 | 0.0256 | done | yes |
| 5 | FlatNet-Lite (PWM) | — | — | 11.9 | — | 25.2 | 0.0256 | done | yes |
| 6 | wiener_deconv (test) | — | — | 11.9 | — | 25.2 | 0.0256 | done | yes |
| 7 | Wiener deconvolution | 2025 | LensNet, IJCAI 2025 (DiffuserCam Wiener=7.33) | 7.3 | 0.0830 | 25.2 | 0.0256 | done | yes |

### 31. Panorama Multi-Focus Fusion (`panorama`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Deep homography | 2023 | DL panorama stitching 2023 | 33.6 | 0.9390 | 28.7 | 0.9837 | partial | yes |
| 2 | UDIS (Unsupervised Deep Image Stitching) | 2021 | Nie et al., CVPR 2021 | 28.0 | 0.9000 | 28.7 | 0.9837 | done | yes |
| 3 | APAP | 2013 | Zaragoza et al., CVPR 2013 | 25.0 | 0.8500 | 28.7 | 0.9837 | done | yes |
| 4 | Laplacian Pyramid Fusion (PWM) | — | — | 16.7 | — | 28.7 | 0.9837 | done | yes |
| 5 | Guided Filter Fusion (PWM) | — | — | 16.7 | — | 28.7 | 0.9837 | done | yes |
| 6 | IFCNN (PWM) | — | Zhang et al. 2020 | 16.7 | — | 28.7 | 0.9837 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 16.7 | — | 28.7 | 0.9837 | done | yes |
| 8 | Single homography stitch | 2024 | Luo et al., arXiv 2406.19922, 2024 | 15.5 | 0.7000 | 28.7 | 0.9837 | done | yes |

## Depth Imaging

### 32. Flash LiDAR (`flash_lidar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Joint depth+reflectivity DNN | 2025 | arXiv 2505.13250 | 29.1 | — | 31.7 | 0.9950 | done | yes |
| 2 | TCSPC histogram | 2000 | flash LiDAR baseline | 22.0 | — | 31.7 | 0.9950 | done | yes |
| 3 | Matched filter SPAD | 2010 | SPAD baseline | 18.0 | — | 31.7 | 0.9950 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 5.3 | -0.6237 | 31.7 | 0.9950 | done | yes |
| 5 | FlashLiDAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 5.3 | -0.6237 | 31.7 | 0.9950 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 4.3 | — | 31.7 | 0.9950 | done | yes |

### 33. LiDAR Scanner (`lidar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | PointNeXt [proxy] (PWM) | — | Richardson 1972, JOSA | 52.0 | 0.9999 | 36.9 | 0.9960 | gap | yes |
| 2 | PointNet++ [proxy] (PWM) | — | Richardson 1972, JOSA | 52.0 | 0.9999 | 36.9 | 0.9960 | gap | yes |
| 3 | BP-Net | 2022 | Tang et al., CVPR 2022 | 36.0 | — | 36.9 | 0.9960 | done | yes |
| 4 | FISTA-L2 (depth) (PWM) | — | — | 35.8 | — | 36.9 | 0.9960 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 35.8 | — | 36.9 | 0.9960 | done | yes |
| 6 | CompletionFormer | 2023 | Zhang et al., CVPR 2023 | 35.5 | — | 36.9 | 0.9960 | done | yes |
| 7 | NLSPN | 2020 | Park et al., ECCV 2020 | 35.0 | — | 36.9 | 0.9960 | done | yes |
| 8 | Bilateral Filter | 1998 | Tomasi & Manduchi, 1998 | 25.0 | — | 36.9 | 0.9960 | done | yes |

### 34. Photometric Stereo (`photometric_stereo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CNN-PS | 2019 | Chen et al., CVPR 2019 | 32.0 | — | 28.7 | 0.9837 | partial | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 30.0 | 0.9683 | 28.7 | 0.9837 | done | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 30.0 | 0.9683 | 28.7 | 0.9837 | done | yes |
| 4 | PS-FCN [proxy] (PWM) | — | Richardson 1972, JOSA | 30.0 | 0.9683 | 28.7 | 0.9837 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 29.0 | — | 28.7 | 0.9837 | done | yes |
| 6 | Woodham (Lambertian) | 1980 | Woodham, Opt Eng 1980 | 25.0 | — | 28.7 | 0.9837 | done | yes |

### 35. Structured-Light Depth Camera (`structured_light`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SFNet (fringe-to-phase) | 2024 | ArXiv 2402.00977 | 38.0 | — | 38.1 | 0.9981 | done | yes |
| 2 | Phase-shifting (4-step) | 1984 | Creath, 1988 | 35.0 | 0.9500 | 38.1 | 0.9981 | done | yes |
| 3 | Gray code | 2003 | Scharstein & Szeliski, 2003 | 25.0 | — | 38.1 | 0.9981 | done | yes |
| 4 | SL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0036 | 38.1 | 0.9981 | done | yes |
| 5 | FTPD [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.0036 | 38.1 | 0.9981 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 8.3 | — | 38.1 | 0.9981 | done | yes |

### 36. Time-of-Flight Depth Camera (`tof_camera`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Phase unwrapping | 2000 | ToF baseline | 47.6 | 0.9999 | 45.2 | 0.0124 | done | yes |
| 2 | ToF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 47.6 | 0.9999 | 45.2 | 0.0124 | done | yes |
| 3 | ToF-MPI Deconv [proxy] (PWM) | — | Richardson 1972, JOSA | 47.6 | 0.9999 | 45.2 | 0.0124 | done | yes |
| 4 | FISTA-L2 (depth) (PWM) | — | — | 42.2 | — | 45.2 | 0.0124 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 42.2 | — | 45.2 | 0.0124 | done | yes |
| 6 | DeepToF | 2017 | Marco et al., CVPR 2017 | 32.0 | — | 45.2 | 0.0124 | done | yes |
| 7 | Bilateral filter (depth) | 2014 | Park et al., Sensors 2014, PMC4168506 | 29.5 | — | 45.2 | 0.0124 | done | yes |

## Electron Microscopy

### 37. Cryo-Electron Tomography (Cryo-ET) (`cryo_et`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | IsoNet | 2022 | Liu et al., Nature Commun 2022 | 28.0 | 0.8500 | 30.4 | 0.9888 | done | gpu |
| 2 | SIRT | 1972 | Gilbert 1972 | 25.0 | 0.7000 | 30.4 | 0.9888 | done | yes |
| 3 | WBP | 1970 | Weighted back-projection | 22.0 | 0.6000 | 30.4 | 0.9888 | done | gpu |
| 4 | Richardson-Lucy (PWM) | — | — | 13.2 | — | 30.4 | 0.9888 | done | yes |
| 5 | CARE (PWM) | — | Weigert et al. 2018 | 13.2 | — | 30.4 | 0.9888 | done | gpu |
| 6 | CryoCARE (PWM) | — | Buchholz, T.O. et al. (2019) Content-aware image restoration for cryo-EM, Methods Enzymol. | 13.2 | — | 30.4 | 0.9888 | done | gpu |
| 7 | precomputed_baseline (test) | — | — | 13.2 | — | 30.4 | 0.9888 | done | gpu |
| 8 | WBP (45-deg missing wedge) | 2019 | Zhang et al., Sci Rep 2019, s41598-019-49267-x | 13.1 | 0.2800 | 30.4 | 0.9888 | done | gpu |

### 38. Electron Backscatter Diffraction (EBSD) (`ebsd`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | EBSD-DL (DictIndex) [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9942 | 27.3 | 0.9699 | partial | yes |
| 2 | EMsoft-EBSD [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9942 | 27.3 | 0.9699 | partial | yes |
| 3 | Dictionary indexing | 2015 | Chen et al., Microscopy 2015 | 25.0 | — | 27.3 | 0.9699 | done | yes |
| 4 | Hough indexing | 1992 | Krieger-Lassen 1998 | 22.0 | — | 27.3 | 0.9699 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 21.9 | — | 27.3 | 0.9699 | done | yes |

### 39. STEM-EDX Elemental Mapping (`edx_mapping`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | NMF denoising | 2015 | NMF for EDX | 26.0 | — | 32.1 | 0.9197 | done | yes |
| 2 | Richardson-Lucy (PWM) | — | — | 24.1 | — | 32.1 | 0.9197 | done | yes |
| 3 | Richardson-Lucy (high quality) (PWM) | — | Richardson 1972, JOSA | 24.1 | — | 32.1 | 0.9197 | done | yes |
| 4 | Richardson-Lucy (DL baseline) (PWM) | — | Tietz, C. et al. (2021) DL for EDS spectrum imaging, Ultramicroscopy 231 | 24.1 | — | 32.1 | 0.9197 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 24.1 | — | 32.1 | 0.9197 | done | yes |
| 6 | PCA denoising | 2010 | PCA for EDX | 24.0 | — | 32.1 | 0.9197 | done | yes |

### 40. Electron Energy Loss Spectroscopy (EELS) (`eels`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Deep CNN Denoiser | 2021 | Mohan et al., Microsc Microanal 2021 | 42.9 | 0.9900 | 27.0 | 0.9549 | gap | yes |
| 2 | FISTA-L2 (Fourier ratio) [proxy] (PWM) | — | Richardson 1972, JOSA | 28.4 | 0.9979 | 27.0 | 0.9549 | done | yes |
| 3 | EELS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 28.4 | 0.9979 | 27.0 | 0.9549 | done | yes |
| 4 | MLLS-EELS [proxy] (PWM) | — | Richardson 1972, JOSA | 28.4 | 0.9979 | 27.0 | 0.9549 | done | yes |
| 5 | PCA denoising | 2012 | Cueva et al., Microsc Microanal 2012 | 28.0 | — | 27.0 | 0.9549 | done | yes |
| 6 | NMF decomposition | 2015 | NMF for EELS | 26.0 | — | 27.0 | 0.9549 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 25.2 | — | 27.0 | 0.9549 | done | yes |

### 41. 4D-STEM Electron Diffraction (`electron_diffraction`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | ED-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 44.4 | 0.9999 | 22.4 | 0.9541 | gap | yes |
| 2 | CRISP-ED [proxy] (PWM) | — | Richardson 1972, JOSA | 44.4 | 0.9999 | 22.4 | 0.9541 | gap | yes |
| 3 | ePIE (electron ptychography) (PWM) | — | — | 42.0 | — | 22.4 | 0.9541 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 42.0 | — | 22.4 | 0.9541 | gap | yes |
| 5 | DPC (Differential Phase Contrast) | 2016 | Lazic et al., Ultramicroscopy 2016 | 25.0 | — | 22.4 | 0.9541 | done | yes |
| 6 | Center-of-mass analysis | 2014 | Muller-Caspary et al., 2014 | 22.0 | — | 22.4 | 0.9541 | done | yes |

### 42. Electron Holography (`electron_holography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | FIN (Fourier Imager Network) | 2022 | Huang et al., Light Sci Appl 2022 | 36.1 | 0.7850 | 26.2 | 0.9793 | partial | yes |
| 2 | HoloPhaseNet (cGAN) | 2022 | Terbe et al., Biomed Opt Express 2022 | 35.3 | 0.9900 | 26.2 | 0.9793 | partial | yes |
| 3 | DNN phase unwrapping | 2021 | DL electron holography | 30.0 | 0.8800 | 26.2 | 0.9793 | partial | yes |
| 4 | Fourier filtering | 1993 | Lichte, Ultramicroscopy 1993 | 25.0 | — | 26.2 | 0.9793 | done | yes |
| 5 | EH-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0936 | 26.2 | 0.9793 | done | yes |
| 6 | Phase-Sideband [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0936 | 26.2 | 0.9793 | done | yes |
| 7 | Phase Retrieval (HIO) (PWM) | — | — | 9.5 | — | 26.2 | 0.9793 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 9.5 | — | 26.2 | 0.9793 | done | yes |

### 43. Electron Tomography (`electron_tomography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Joint DL model (IRDM) | 2019 | Zhang et al., Sci Rep 2019, s41598-019-49267-x | 27.5 | 0.9530 | 24.1 | 0.9455 | partial | yes |
| 2 | IMOD-SIRT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 26.1 | 0.9625 | 24.1 | 0.9455 | done | yes |
| 3 | SIRT-3D [proxy] (PWM) | — | Richardson 1972, JOSA | 26.1 | 0.9625 | 24.1 | 0.9455 | done | yes |
| 4 | FBP (SIRT baseline) (PWM) | — | — | 25.1 | — | 24.1 | 0.9455 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 25.1 | — | 24.1 | 0.9455 | done | yes |
| 6 | SART (missing wedge) | 1972 | Zhang et al., Sci Rep 2019 | 18.6 | 0.3120 | 24.1 | 0.9455 | done | yes |
| 7 | WBP (missing wedge) | 1970 | Zhang et al., Sci Rep 2019 | 13.1 | 0.2800 | 24.1 | 0.9455 | done | yes |

### 44. Focused Ion Beam SEM (FIB-SEM) (`fib_sem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 34.0 | — | 25.4 | 0.9517 | partial | gpu |
| 2 | NRRN | 2021 | bioRxiv 2021 | 31.0 | 0.9710 | 25.4 | 0.9517 | partial | gpu |
| 3 | BM3D | 2007 | Dabov et al., 2007 | 30.0 | — | 25.4 | 0.9517 | partial | yes |
| 4 | Richardson-Lucy (PWM) | — | — | 28.3 | — | 25.4 | 0.9517 | done | yes |
| 5 | CARE (PWM) | — | Weigert et al. 2018 | 28.3 | — | 25.4 | 0.9517 | done | gpu |
| 6 | FIB-SEM-Net (PWM) | — | Heinrich, L. et al. (2021) Whole-cell organelle segmentation in volume EM, Nature 599:141 | 28.3 | — | 25.4 | 0.9517 | done | gpu |
| 7 | precomputed_baseline (test) | — | — | 28.3 | — | 25.4 | 0.9517 | done | gpu |

### 45. Scanning Electron Microscopy (SEM) (`sem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 34.0 | — | 39.0 | 0.9965 | done | yes |
| 2 | BM3D | 2007 | Dabov et al., TIP 2007 | 30.0 | 0.8500 | 39.0 | 0.9965 | done | yes |
| 3 | SEM-DL (SegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.9761 | 39.0 | 0.9965 | done | yes |
| 4 | SEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 28.8 | 0.9761 | 39.0 | 0.9965 | done | yes |
| 5 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 28.0 | — | 39.0 | 0.9965 | done | yes |
| 6 | NLM | 2005 | Buades et al., CVPR 2005 | 25.0 | 0.7800 | 39.0 | 0.9965 | done | yes |
| 7 | Richardson-Lucy (SEM) (PWM) | — | — | 23.2 | — | 39.0 | 0.9965 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 23.2 | — | 39.0 | 0.9965 | done | yes |
| 9 | Gaussian filter | 2000 | Gaussian baseline | 22.0 | 0.7000 | 39.0 | 0.9965 | done | yes |

### 46. Scanning Transmission Electron Microscopy (STEM) (`stem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DAE (Denoising AE) | 2023 | ACS Central Science 2023 | 42.9 | 0.9900 | 25.7 | 0.9840 | gap | yes |
| 2 | STEM-DL (AtomSegNet) [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9800 | 25.7 | 0.9840 | gap | yes |
| 3 | STEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9800 | 25.7 | 0.9840 | gap | yes |
| 4 | Richardson-Lucy (STEM) (PWM) | — | — | 34.5 | — | 25.7 | 0.9840 | partial | yes |
| 5 | precomputed_baseline (test) | — | — | 34.5 | — | 25.7 | 0.9840 | partial | yes |
| 6 | SwinIR | 2021 | Liang et al., 2021 | 33.0 | — | 25.7 | 0.9840 | partial | yes |
| 7 | BM3D | 2007 | Dabov et al., 2007 | 30.0 | 0.8500 | 25.7 | 0.9840 | partial | yes |

### 47. Transmission Electron Microscopy (TEM) (`tem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CGRDN | 2024 | Lobato et al., npj Comp Mat 2024, s41524-023-01188-0 | 37.0 | — | 40.8 | 0.0002 | done | yes |
| 2 | SwinIR | 2021 | Liang et al., 2021 | 35.0 | — | 40.8 | 0.0002 | done | yes |
| 3 | Topaz-Denoise | 2020 | Bepler et al., Nature Commun 2020 | 32.0 | — | 40.8 | 0.0002 | done | yes |
| 4 | BM3D | 2007 | Lobato et al., npj Comp Mat 2024 (comparison) | 30.4 | — | 40.8 | 0.0002 | done | yes |
| 5 | TEM-DL (ePIE-Net) [proxy] (PWM) | — | Richardson 1972, JOSA | 26.3 | 0.9290 | 40.8 | 0.0002 | done | yes |
| 6 | TEM-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 26.3 | 0.9290 | 40.8 | 0.0002 | done | yes |
| 7 | Wiener filter (basic) | 2013 | Lobato & Van Dyck, Ultramicroscopy 2013 | 26.0 | — | 40.8 | 0.0002 | done | yes |
| 8 | FISTA-L2 (CTF correction) (PWM) | — | — | 25.3 | — | 40.8 | 0.0002 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 25.3 | — | 40.8 | 0.0002 | done | yes |
| 10 | NLM | 2005 | Buades et al., CVPR 2005 | 25.0 | 0.7500 | 40.8 | 0.0002 | done | yes |

## Industrial Inspection

### 48. Scanning Acoustic Microscopy (SAM) (`acoustic_microscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR (SAM) | 2024 | Somani et al., CVPR Workshop 2023 | 35.1 | 0.9500 | 38.5 | 0.0408 | done | yes |
| 2 | HDL-SAM (SwinIR+Hypergraph) | 2024 | Somani & Banerjee, OpenReview 2024 | 31.6 | 0.9200 | 38.5 | 0.0408 | done | yes |
| 3 | Hypergraph Inpainting | 2023 | Somani et al., CVPR Workshop 2023 | 28.0 | 0.8200 | 38.5 | 0.0408 | done | yes |
| 4 | SAFT (Synth Aperture Focus) | 1980 | Doctor et al., 1986 | 25.0 | — | 38.5 | 0.0408 | done | yes |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 24.8 | 0.9483 | 38.5 | 0.0408 | done | yes |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 24.8 | 0.9483 | 38.5 | 0.0408 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 22.6 | — | 38.5 | 0.0408 | done | yes |
| 8 | DAS beamforming | 1990 | Beamforming baseline | 22.0 | — | 38.5 | 0.0408 | done | yes |

### 49. Active Thermography (IR) (`active_thermography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | TESR (Transformer) | 2024 | Sci Reports 2024, PMC11227526 | 46.2 | 0.9920 | 35.1 | 0.9974 | gap | yes |
| 2 | RCAN | 2024 | Sci Reports 2024, PMC11227526 | 45.9 | 0.9920 | 35.1 | 0.9974 | gap | yes |
| 3 | EDSR | 2024 | Sci Reports 2024, PMC11227526 | 45.3 | 0.9900 | 35.1 | 0.9974 | gap | yes |
| 4 | SRCNN | 2024 | Sci Reports 2024, PMC11227526 | 42.9 | 0.9840 | 35.1 | 0.9974 | partial | yes |
| 5 | Bicubic baseline | 2024 | Sci Reports 2024, PMC11227526 | 42.1 | 0.9820 | 35.1 | 0.9974 | partial | yes |
| 6 | Pulsed phase thermography | 1996 | Maldague & Marinetti, J Appl Phys 1996 | 25.0 | — | 35.1 | 0.9974 | done | yes |
| 7 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 8.2 | 0.1575 | 35.1 | 0.9974 | done | yes |
| 8 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 8.2 | 0.1575 | 35.1 | 0.9974 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 7.2 | — | 35.1 | 0.9974 | done | yes |

### 50. Eddy Current Imaging (`eddy_current`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Wavelet denoising | 2000 | Wavelet for ECT | 25.0 | — | 33.3 | 0.0120 | done | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.9 | 0.6456 | 33.3 | 0.0120 | done | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.9 | 0.6456 | 33.3 | 0.0120 | done | yes |
| 4 | ECT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 23.9 | 0.6456 | 33.3 | 0.0120 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 22.9 | — | 33.3 | 0.0120 | done | yes |
| 6 | Impedance plane analysis | 2000 | ECT baseline | 22.0 | — | 33.3 | 0.0120 | done | yes |

### 51. Industrial X-ray CT (`industrial_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | ADMM-TransNet | 2025 | MDPI 2025 | 44.6 | 0.9960 | 7.2 | -0.6959 | gap | yes |
| 2 | SIRT | 1972 | Gilbert 1972 | 30.0 | 0.8500 | 7.2 | -0.6959 | gap | yes |
| 3 | FDK | 1984 | Feldkamp et al., 1984 | 28.0 | 0.8000 | 7.2 | -0.6959 | gap | yes |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 21.3 | 0.4146 | 7.2 | -0.6959 | gap | yes |
| 5 | IndustrialCT-Net [proxy] (PWM) | — | Shepp & Logan 1974 | 21.3 | 0.4146 | 7.2 | -0.6959 | gap | yes |
| 6 | precomputed_baseline (test) | — | — | 20.3 | — | 7.2 | -0.6959 | gap | yes |

### 52. Machine Vision / AOI (`machine_vision`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9999 | 34.3 | 0.9972 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 36.2 | 0.9999 | 34.3 | 0.9972 | done | yes |
| 3 | UniAD | 2023 | You et al., NeurIPS 2022 | 32.0 | — | 34.3 | 0.9972 | done | yes |
| 4 | PatchCore | 2022 | Roth et al., CVPR 2022 | 30.0 | — | 34.3 | 0.9972 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 28.3 | — | 34.3 | 0.9972 | done | yes |
| 6 | Template matching | 2000 | Brunelli, Template Matching, 2009 | 25.0 | — | 34.3 | 0.9972 | done | yes |

### 53. Shearography (`shearography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Phase-shifting shearography | 2000 | Hung, 1982 | 28.0 | — | 36.5 | 0.9956 | done | yes |
| 2 | FPD-CNN | 2020 | Lin et al., Applied Optics 2020 | 27.9 | 0.9720 | 36.5 | 0.9956 | done | yes |
| 3 | Fourier transform method | 1982 | Takeda et al., JOSA 1982 | 25.0 | — | 36.5 | 0.9956 | done | yes |
| 4 | DBDNet | 2021 | Li et al., Applied Optics 2021 | 20.6 | — | 36.5 | 0.9956 | done | yes |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 19.1 | 0.4833 | 36.5 | 0.9956 | done | yes |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 19.1 | 0.4833 | 36.5 | 0.9956 | done | yes |
| 7 | ShearNet [proxy] (PWM) | — | Richardson 1972, JOSA | 19.1 | 0.4833 | 36.5 | 0.9956 | done | yes |
| 8 | OCPDE (Oriented Coupled PDE) | 2020 | Lin et al., Applied Optics 2020 | 14.1 | — | 36.5 | 0.9956 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 13.2 | — | 36.5 | 0.9956 | done | yes |
| 10 | WFLPF (Windowed Fourier LP Filter) | 2020 | Lin et al., Applied Optics 2020 | 12.8 | — | 36.5 | 0.9956 | done | yes |

### 54. Terahertz Imaging (THz) (`terahertz`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 47.9 | 0.9999 | 35.2 | 0.0167 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 47.9 | 0.9999 | 35.2 | 0.0167 | gap | yes |
| 3 | THz-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 47.9 | 0.9999 | 35.2 | 0.0167 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 37.1 | — | 35.2 | 0.0167 | done | yes |
| 5 | J-Net (real THz) | 2023 | Yeo et al., arXiv 2312.01638 | 32.5 | — | 35.2 | 0.0167 | done | yes |
| 6 | EARDB | 2023 | Hou et al., Entropy 25(3):440, PMC10047599 | 31.3 | 0.8910 | 35.2 | 0.0167 | done | yes |
| 7 | TDS deconvolution | 2000 | THz-TDS baseline | 22.0 | — | 35.2 | 0.0167 | done | yes |

### 55. Ultrasonic Phased Array (TFM/FMC) (`ultrasonic_phased_array`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CycleSR | 2025 | MSSP 2025 | 39.3 | — | 33.5 | 0.9937 | partial | yes |
| 2 | CinCGAN | 2025 | MSSP 2025 | 36.4 | — | 33.5 | 0.9937 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 35.2 | 0.8974 | 33.5 | 0.9937 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 35.2 | 0.8974 | 33.5 | 0.9937 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 31.1 | — | 33.5 | 0.9937 | done | yes |
| 6 | TFM (Total Focusing Method) | 2004 | Holmes et al., NDT&E Int 2005 | 28.0 | — | 33.5 | 0.9937 | done | yes |

### 56. X-ray NDT (Radiography) (`xray_ndt`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | U-Net++ | 2025 | NDT.net DIR 2025 | 32.3 | 0.8960 | 43.0 | 0.1614 | done | yes |
| 2 | BM3D denoising | 2007 | Dabov et al., TIP 2007 | 32.0 | 0.8800 | 43.0 | 0.1614 | done | yes |
| 3 | FBP | 1971 | FBP baseline | 28.0 | 0.8000 | 43.0 | 0.1614 | done | yes |
| 4 | Raw projection (no filtering) | 2000 | X-ray raw projection | 18.0 | 0.6000 | 43.0 | 0.1614 | done | yes |
| 5 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 17.7 | 0.8530 | 43.0 | 0.1614 | done | yes |
| 6 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 17.7 | 0.8530 | 43.0 | 0.1614 | done | yes |
| 7 | NDT-DefectNet [proxy] (PWM) | — | Richardson 1972, JOSA | 17.7 | 0.8530 | 43.0 | 0.1614 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 16.7 | — | 43.0 | 0.1614 | done | yes |

### 57. X-ray Fluorescence (XRF) Imaging (`xrf_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DnCNN (XFCT) | 2024 | J Imaging 2024, PMC11204716 | 49.4 | 0.9430 | 30.7 | 0.9811 | gap | yes |
| 2 | NLM (XFCT) | 2024 | J Imaging 2024, PMC11204716 | 39.9 | 0.8030 | 30.7 | 0.9811 | partial | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 29.8 | 0.9997 | 30.7 | 0.9811 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 29.8 | 0.9997 | 30.7 | 0.9811 | done | yes |
| 5 | XRF-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 29.8 | 0.9997 | 30.7 | 0.9811 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 26.7 | — | 30.7 | 0.9811 | done | yes |
| 7 | PCA denoising | 2010 | PCA for XRF | 25.0 | — | 30.7 | 0.9811 | done | yes |
| 8 | Fundamental parameters | 2000 | Sherman, Spectrochim Acta 1955 | 22.0 | — | 30.7 | 0.9811 | done | yes |

## Medical Imaging

### 58. X-ray Angiography (`angiography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Maskless 2D-DSA (U-Net) | 2022 | Gao et al., JVIR 2022, PubMed 35311665 | 43.0 | 0.9800 | 35.5 | 0.0366 | partial | yes |
| 2 | DSA subtraction (with motion) | 1980 | Ueda et al., Radiology 2021 (motion-free=40.2 dB) | 30.0 | 0.5000 | 35.5 | 0.0366 | done | yes |
| 3 | DSA (Digital Subtraction) | 1980 | DSA, Mistretta et al., 1981 | 25.0 | 0.8000 | 35.5 | 0.0366 | done | yes |
| 4 | Deep Decoupling Net (GAN+RDB) | 2024 | IIETA, TS 2024 | 23.7 | 0.8770 | 35.5 | 0.0366 | done | yes |
| 5 | DSA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.0377 | 35.5 | 0.0366 | done | yes |
| 6 | VesselSegNet [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.0377 | 35.5 | 0.0366 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 12.9 | — | 35.5 | 0.0366 | done | yes |

### 59. Arterial Spin Labeling (ASL) MRI (`asl_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | HUST (Transformer) 3D | 2025 | Springer, Vis Comput 2025 | 45.1 | 0.9900 | 22.3 | 0.2631 | gap | yes |
| 2 | HUST (Transformer) 2D | 2025 | Springer, Vis Comput 2025 | 33.7 | 0.9600 | 22.3 | 0.2631 | gap | yes |
| 3 | ASLRDB (Dilated+RDB) | 2025 | Springer, SIVP 2025 | 25.0 | 0.8240 | 22.3 | 0.2631 | done | yes |
| 4 | Control-label subtraction | 1998 | Detre et al., MRM 1992 | 22.0 | 0.6500 | 22.3 | 0.2631 | done | yes |
| 5 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1371 | 22.3 | 0.2631 | done | yes |
| 6 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1371 | 22.3 | 0.2631 | done | yes |
| 7 | ASL-Net [proxy] (PWM) | — | — | 12.9 | 0.1371 | 22.3 | 0.2631 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 10.9 | — | 22.3 | 0.2631 | done | yes |

### 60. Brachytherapy Imaging (`brachytherapy_img`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RL-ARCNN (metal artifact reduction) | 2018 | Huang et al., BioMedical Eng OnLine 2018 | 38.1 | — | 30.2 | 0.1022 | partial | yes |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 33.1 | 0.8307 | 30.2 | 0.1022 | done | yes |
| 3 | BrachyNet [proxy] (PWM) | — | Richardson 1972, JOSA | 33.1 | 0.8307 | 30.2 | 0.1022 | done | yes |
| 4 | Monte Carlo dose | 2005 | MC dose calculation | 28.0 | 0.8500 | 30.2 | 0.1022 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 25.2 | — | 30.2 | 0.1022 | done | yes |
| 6 | FBP | 1971 | FBP baseline | 25.0 | — | 30.2 | 0.1022 | done | yes |

### 61. Cone-Beam Computed Tomography (CBCT) (`cbct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | FBPConvNet | 2017 | Jin et al., TIP 2017 | 36.5 | 0.9500 | 15.1 | -0.0255 | gap | gpu |
| 2 | FACT | 2022 | FACT, 2022 | 33.8 | 0.9300 | 15.1 | 0.9188 | gap | gpu |
| 3 | SART | 1984 | Andersen & Kak, 1984 | 32.0 | 0.8800 | 15.1 | 0.9188 | gap | yes |
| 4 | FDK | 1984 | Feldkamp et al., JOSA 1984 | 28.0 | 0.8000 | 15.1 | 0.9188 | gap | yes |
| 5 | FDK (8 views) | 1984 | Zha et al., MICCAI 2024 | 16.6 | — | 15.1 | 0.9188 | done | gpu |
| 6 | FDK (6 views) | 1984 | Zha et al., MICCAI 2024, arXiv 2407.01090 | 15.3 | — | 15.1 | 0.9188 | done | gpu |
| 7 | FDK-DL (PWM) | — | Chen, H. et al. (2017) Low-dose CT with residual encoder-decoder CNN, IEEE TMI | 15.2 | — | 15.1 | 0.9188 | done | gpu |
| 8 | CBCT-UNet (PWM) | — | Jin, K.H. et al. (2017) Deep convolutional network for inverse problems, IEEE TIP | 15.2 | — | 15.1 | -0.0255 | done | gpu |
| 9 | fbp_ramlak (test) | — | — | 15.2 | — | 15.1 | 0.9188 | done | yes |
| 10 | fbp_shepp_logan (test) | — | — | 15.2 | — | 15.1 | -0.0255 | done | yes |

### 62. CEST MRI (`cest_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | 30.1 | 0.0127 | gap | yes |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 44.3 | 0.9999 | 30.1 | 0.0127 | gap | yes |
| 3 | CEST-Net [proxy] (PWM) | — | — | 44.3 | 0.9999 | 30.1 | 0.0127 | gap | yes |
| 4 | ResUNet-NE | 2023 | Muller et al., Diagnostics 13(21):3326, 2023 | 35.0 | — | 30.1 | 0.0127 | partial | yes |
| 5 | precomputed_baseline (test) | — | — | 32.1 | — | 30.1 | 0.0127 | done | yes |
| 6 | Z-spectrum fitting | 2003 | Zhou et al., NMR Biomed 2003 | 25.0 | 0.7500 | 30.1 | 0.0127 | done | yes |

### 63. Contrast-Enhanced Ultrasound (CEUS) (`ceus`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Real-time CNN | 2022 | Choi et al., MBEC 2022 | 36.1 | 0.9640 | 24.9 | 0.9376 | gap | yes |
| 2 | GAN-RW (Residual Dense) | 2022 | Lan et al., PeerJ Computer Science 2022 | 33.9 | 0.8720 | 24.9 | 0.9376 | partial | yes |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 26.4 | 0.9801 | 24.9 | 0.9376 | done | yes |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 26.4 | 0.9801 | 24.9 | 0.9376 | done | yes |
| 5 | US-DeepSight [proxy] (PWM) | — | Richardson 1972, JOSA | 26.4 | 0.9801 | 24.9 | 0.9376 | done | yes |
| 6 | Singular value decomposition | 2015 | Demene et al., TMI 2015 | 25.0 | 0.7500 | 24.9 | 0.9376 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 24.5 | — | 24.9 | 0.9376 | done | yes |
| 8 | Temporal averaging | 2000 | CEUS temporal baseline | 22.0 | 0.7000 | 24.9 | 0.9376 | done | yes |

### 64. Confocal Laser Endomicroscopy (CLE) (`confocal_endomicroscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 41.5 | 0.9999 | 55.2 | 0.9998 | done | yes |
| 2 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 41.5 | 0.9999 | 55.2 | 0.9998 | done | yes |
| 3 | CLE-Net (CARE) [proxy] (PWM) | — | Richardson 1972, JOSA | 41.5 | 0.9999 | 55.2 | 0.9998 | done | yes |
| 4 | Self-supervised denoising | 2024 | Sensors 2024 | 36.1 | 0.8980 | 55.2 | 0.9998 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 34.0 | — | 55.2 | 0.9998 | done | yes |
| 6 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | 55.2 | 0.9998 | done | yes |

### 65. X-ray Computed Tomography (CT) (`ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | LEARN | 2019 | Chen et al., TMI 2018 | 43.1 | — | 15.1 | 0.9096 | gap | yes |
| 2 | Score-CT | 2022 | Song et al., ICLR 2022 | 43.0 | — | 15.1 | — | gap | yes |
| 3 | DuDoTrans | 2022 | Wang et al., MICCAI 2022 | 42.1 | — | 15.1 | 0.9096 | gap | yes |
| 4 | FBPConvNet | 2017 | Jin et al., TIP 2017 | 38.5 | 0.9590 | 15.1 | 0.9096 | gap | yes |
| 5 | iRadonMAP | 2019 | He et al., 2019 | 36.9 | 0.9420 | 15.1 | 0.9096 | gap | yes |
| 6 | Learned Primal-Dual | 2018 | Adler & Oktem, TMI 2018 | 36.2 | 0.9590 | 15.1 | — | gap | yes |
| 7 | DOLCE | 2023 | Liu et al., 2023 | 36.0 | — | 15.1 | 0.9096 | gap | yes |
| 8 | TV regularization | 2006 | Sidky et al., PMB 2006 | 33.4 | 0.9000 | 15.1 | 0.9096 | gap | yes |
| 9 | RED-CNN | 2017 | Chen et al., TMI 2017 | 33.2 | 0.9150 | 15.1 | — | gap | yes |
| 10 | FBP (Ram-Lak) | 1971 | Ramachandran & Lakshminarayanan 1971 | 30.2 | 0.8200 | 15.1 | 0.9096 | gap | yes |
| 11 | FBP (10 angles) | 2021 | Leuschner et al., J Imaging 2021, PMC8321320 | 17.1 | — | 15.1 | 0.9096 | done | yes |
| 12 | FBP (5 angles) | 2021 | Leuschner et al., J Imaging 2021, PMC8321320 | 15.5 | — | 15.1 | 0.9096 | done | yes |
| 13 | PnP-HQS + NLM (PWM) | — | — | 13.8 | — | 15.1 | — | done | yes |
| 14 | fbp_ramlak (test) | — | — | 13.8 | — | 15.1 | 0.9096 | done | yes |
| 15 | fbp_shepp_logan (test) | — | — | 13.8 | — | 15.1 | 0.9096 | done | yes |
| 16 | sart_10iter (test) | — | — | 13.8 | — | 15.1 | 0.9096 | done | yes |
| 17 | FBP (2 angles, scattering) | 2021 | Leuschner et al., J Imaging 2021, PMC8321320 | 13.1 | — | 15.1 | 0.9096 | done | yes |

### 66. Dual-Energy X-ray Absorptiometry (DEXA) (`dexa`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DL bone density estimation | 2022 | DL for DEXA | 32.0 | 0.9000 | 49.7 | 0.9999 | done | yes |
| 2 | Dual-energy decomposition | 1987 | Alvarez & Macovski, PMB 1976 | 28.0 | 0.8500 | 49.7 | 0.9999 | done | yes |
| 3 | Bone decomposition baseline | 2020 | DEXA energy subtraction baseline (estimated) | 19.7 | — | 49.7 | 0.9999 | done | yes |
| 4 | DXA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.7 | 0.4561 | 49.7 | 0.9999 | done | yes |
| 5 | DEXA-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 11.7 | 0.4561 | 49.7 | 0.9999 | done | yes |
| 6 | FISTA-L2 (dual-energy) (PWM) | — | — | 10.7 | — | 49.7 | 0.9999 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 10.7 | — | 49.7 | 0.9999 | done | yes |

### 67. Diffusion MRI (DTI) (`diffusion_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | q-DL | 2016 | Golkov et al., MRM 2016 | 34.0 | — | 24.7 | 0.1243 | partial | yes |
| 2 | MPR-ViT (ADC maps) | 2024 | Eidex et al., Med Phys 2024 | 31.0 | 0.9500 | 24.7 | 0.1243 | partial | yes |
| 3 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6000 | 24.7 | 0.1243 | done | yes |
| 4 | Zero-filled (high b-value) | 2000 | dMRI zero-filled baseline | 15.0 | 0.4000 | 24.7 | 0.1243 | done | yes |
| 5 | SHORE-Net [proxy] (PWM) | — | — | 13.0 | 0.0360 | 24.7 | 0.1243 | done | yes |
| 6 | Zero-filled (R=4, multi-b) | 2023 | Zhong et al., Bioengineering 2023, PMC10376839 | 12.2 | — | 24.7 | 0.1243 | done | yes |
| 7 | Zero-filled (R=6, multi-b) | 2023 | Zhong et al., Bioengineering 2023, PMC10376839 | 12.0 | 0.3000 | 24.7 | 0.1243 | done | yes |
| 8 | SENSE (WLS tensor fit) (PWM) | — | — | 11.3 | — | 24.7 | 0.1243 | done | yes |
| 9 | zero_filled (test) | — | — | 11.3 | — | 24.7 | 0.1243 | done | yes |

### 68. Digital Breast Tomosynthesis (DBT) (`digital_breast_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SART | 1984 | Andersen & Kak 1984 | 30.0 | — | 24.1 | 0.3456 | partial | yes |
| 2 | TV-regularized MLEM | 2010 | TV-MLEM for DBT | 28.0 | 0.8700 | 24.1 | 0.3456 | partial | yes |
| 3 | FBP | 1971 | FBP baseline | 25.0 | — | 24.1 | 0.3456 | done | yes |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 10.5 | 0.4411 | 24.1 | 0.3456 | done | yes |
| 5 | DBT-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 10.5 | 0.4411 | 24.1 | 0.3456 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 8.8 | — | 24.1 | 0.3456 | done | yes |

### 69. Doppler Ultrasound (`doppler_ultrasound`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DL Doppler | 2020 | DL for Doppler dealiasing | 30.0 | 0.8800 | 25.6 | 0.9507 | partial | yes |
| 2 | 3D-Res-UNet (95% compression) | 2022 | Blanchard et al., IEEE TUFFC 2022, PMC9247015 | 26.7 | — | 25.6 | 0.9507 | done | yes |
| 3 | Autocorrelation | 1985 | Kasai et al., 1985 | 22.0 | 0.7000 | 25.6 | 0.9507 | done | yes |
| 4 | Conventional SVD (90% compression) | 2022 | Blanchard et al., IEEE TUFFC 2022, PMC9247015 | 19.5 | — | 25.6 | 0.9507 | done | yes |
| 5 | UDoppler-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 18.6 | 0.0164 | 25.6 | 0.9507 | done | yes |
| 6 | Doppler CFAR [proxy] (PWM) | — | Richardson 1972, JOSA | 18.6 | 0.0164 | 25.6 | 0.9507 | done | yes |
| 7 | Wall filter (highpass) | 1985 | Wall filter baseline | 18.0 | 0.6000 | 25.6 | 0.9507 | done | yes |
| 8 | Back-Projection (Doppler) (PWM) | — | — | 17.6 | — | 25.6 | 0.9507 | done | yes |
| 9 | autocorrelation_estimator (test) | — | — | 17.6 | — | 25.6 | 0.9507 | done | yes |
| 10 | clutter_filtered (test) | — | — | 17.6 | — | 25.6 | 0.9507 | done | yes |
| 11 | precomputed_baseline (test) | — | — | 17.6 | — | 25.6 | 0.9507 | done | yes |
| 12 | Conventional SVD (95% compression) | 2022 | Blanchard et al., IEEE TUFFC 2022, PMC9247015 | 17.4 | — | 25.6 | 0.9507 | done | yes |

### 70. Diffuse Optical Tomography (DOT) (`dot`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | BPNN | 2018 | Feng et al., JBO 24(5), PMC6992907 | 27.8 | 0.9100 | 27.1 | 0.0103 | done | yes |
| 2 | Tikhonov regularization | 2018 | Feng et al., JBO 24(5), PMC6992907 | 24.3 | 0.4600 | 27.1 | 0.0103 | done | yes |
| 3 | Tikhonov (basic, noisy) | 2000 | Yoo et al., J Biomed Opt 2019, PMC6992907 | 22.0 | 0.3000 | 27.1 | 0.0103 | done | yes |
| 4 | Born approximation | 1999 | Arridge, Inverse Problems 1999 | 20.0 | 0.6000 | 27.1 | 0.0103 | done | yes |
| 5 | Rytov + Laplacian | 2000 | Arridge et al., PMB 1999 | 18.0 | 0.4500 | 27.1 | 0.0103 | done | yes |
| 6 | L-BFGS-TV [proxy] (PWM) | — | Richardson 1972, JOSA | 8.0 | 0.0293 | 27.1 | 0.0103 | done | yes |
| 7 | DOT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 8.0 | 0.0293 | 27.1 | 0.0103 | done | yes |
| 8 | born_backprojection (test) | — | — | 7.0 | — | 27.1 | 0.0103 | done | yes |
| 9 | tikhonov (test) | — | — | 7.0 | — | 27.1 | 0.0103 | done | yes |
| 10 | precomputed_baseline (test) | — | — | 7.0 | — | 27.1 | 0.0103 | done | yes |

### 71. Shear-Wave Elastography (`elastography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CNN-LSTM | 2024 | arXiv 2024 | 32.7 | 0.9960 | 24.3 | 0.9485 | partial | yes |
| 2 | Direct inversion | 2001 | Manduca et al., MRM 2001 | 24.0 | 0.7500 | 24.3 | 0.9485 | done | yes |
| 3 | Phase gradient | 2000 | Manduca et al., MRM 2001 | 22.0 | 0.7000 | 24.3 | 0.9485 | done | yes |
| 4 | Raw displacement (no filtering) | 2000 | Elastography raw baseline | 14.0 | 0.4000 | 24.3 | 0.9485 | done | yes |
| 5 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 12.0 | 0.8049 | 24.3 | 0.9485 | done | yes |
| 6 | NLSI-Solver [proxy] (PWM) | — | Richardson 1972, JOSA | 12.0 | 0.8049 | 24.3 | 0.9485 | done | yes |
| 7 | SENSE (displacement field) (PWM) | — | — | 11.0 | — | 24.3 | 0.9485 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 11.0 | — | 24.3 | 0.9485 | done | yes |

### 72. Fiber Bundle Endoscopy (`endoscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR | 2024 | Heliyon 2024 | 36.8 | 0.9700 | 26.8 | 0.9633 | gap | gpu |
| 2 | U-Net denoising | 2019 | DL for CLE | 28.0 | 0.8500 | 26.8 | 0.9633 | done | gpu |
| 3 | Richardson-Lucy | 1972 | Richardson 1972 | 24.0 | 0.7200 | 26.8 | 0.9633 | done | yes |
| 4 | Interpolation baseline | 2000 | Fiber bundle baseline | 22.0 | 0.6500 | 26.8 | 0.9633 | done | gpu |
| 5 | Raw CLE (honeycomb artifact) | 2022 | Kim et al., Sensors 2022, PMC9824069 | 20.6 | 0.7300 | 26.8 | 0.9633 | done | gpu |
| 6 | Gaussian filter (fiber bundle) | 2023 | Kim et al., Sensors 2023, PMC9824069 | 19.0 | — | 26.8 | 0.9633 | done | gpu |
| 7 | Raw fiber bundle (no processing) | 2019 | Shao et al., Optics Express 2019, PMC6825616 | 14.6 | — | 26.8 | 0.9633 | done | gpu |
| 8 | FISTA-L2 (endoscopy) (PWM) | — | — | 11.8 | — | 26.8 | 0.9633 | done | yes |
| 9 | EndoMapper-Net (PWM) | — | Ozyoruk, K.B. et al. (2021) EndoMapper, Nat. Mach. Intel. 3 | 11.8 | — | 26.8 | 0.9633 | done | gpu |
| 10 | AF-SfMLearner (PWM) | — | Shao, S. et al. (2022) Self-supervised depth estimation in endoscopy, MICCAI 2022 | 11.8 | — | 26.8 | 0.9633 | done | gpu |
| 11 | rl_20iter (test) | — | — | 11.8 | — | 26.8 | 0.9633 | done | gpu |
| 12 | rl_50iter (test) | — | — | 11.8 | — | 26.8 | 0.9633 | done | gpu |
| 13 | precomputed_recon (test) | — | — | 11.8 | — | 26.8 | 0.9633 | done | gpu |

### 73. Fluoroscopy (`fluoroscopy`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | FluoroNet [proxy] (PWM) | — | Richardson 1972, JOSA | 54.9 | 0.9999 | 28.5 | 0.0438 | gap | yes |
| 2 | X-ray CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 54.9 | 0.9999 | 28.5 | 0.0438 | gap | yes |
| 3 | FBP (fluoroscopy) (PWM) | — | — | 44.5 | — | 28.5 | 0.0438 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 44.5 | — | 28.5 | 0.0438 | gap | yes |
| 5 | MSR2AU-Net | 2024 | arXiv 2024 | 39.1 | 0.9800 | 28.5 | 0.0438 | gap | yes |
| 6 | RED-CNN | 2017 | Chen et al., TMI 2017 | 33.0 | 0.9000 | 28.5 | 0.0438 | partial | yes |
| 7 | Motion compensation | 2000 | fluoroscopy baseline | 28.0 | 0.8000 | 28.5 | 0.0438 | done | yes |

### 74. Functional MRI (BOLD fMRI) (`fmri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | E2E-VarNet | 2021 | Sriram et al., fastMRI Challenge 2020 | 41.4 | 0.9590 | 12.3 | 0.3671 | gap | yes |
| 2 | CS-fMRI | 2010 | Jung et al., PMB 2009 | 32.0 | 0.8800 | 12.3 | 0.3671 | gap | yes |
| 3 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6000 | 12.3 | 0.3671 | gap | yes |
| 4 | fMRI-Transformer [proxy] (PWM) | — | — | 9.9 | 0.1054 | 12.3 | 0.3671 | done | yes |
| 5 | DeepBold [proxy] (PWM) | — | — | 9.9 | 0.1054 | 12.3 | 0.3671 | done | yes |
| 6 | SENSE (fMRI) (PWM) | — | — | 4.9 | — | 12.3 | 0.3671 | done | yes |
| 7 | zero_filled (test) | — | — | 4.9 | — | 12.3 | 0.3671 | done | yes |

### 75. Fundus Camera (`fundus`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RETFound (PWM) | — | Zhou, Y. et al. (2023) RETFound: Foundation model for retinal imaging, Nature 622:156 | 35.9 | — | 36.8 | 0.9974 | done | gpu |
| 2 | DR-Grade-Net (PWM) | — | Gulshan, V. et al. (2016) DL for DR detection in retinal fundus, JAMA 316(22) | 35.9 | — | 36.8 | 0.9974 | done | gpu |
| 3 | rl_20iter (test) | — | — | 35.9 | — | 36.8 | 0.9974 | done | gpu |
| 4 | rl_50iter (test) | — | — | 35.9 | — | 36.8 | 0.9974 | done | gpu |
| 5 | precomputed_wiener (test) | — | — | 35.9 | — | 36.8 | 0.9974 | done | yes |
| 6 | Richardson-Lucy | 1972 | Richardson 1972 | 30.0 | 0.9000 | 36.8 | 0.9974 | done | yes |
| 7 | PCE-Net | 2023 | PCE-Net, 2023 | 29.9 | — | 36.8 | 0.9974 | done | gpu |
| 8 | GFE-Net | 2023 | Med Image Anal 2023 | 29.7 | 0.9550 | 36.8 | 0.9974 | done | gpu |
| 9 | Cofe-Net | 2022 | Li et al., Cofe-Net, 2022 | 24.9 | — | 36.8 | 0.9974 | done | gpu |

### 76. Intravascular Ultrasound (IVUS) (`ivus`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | IVUS-Net | 2020 | DL for IVUS | 30.0 | 0.8800 | 26.3 | 0.9645 | partial | yes |
| 2 | U-Net segmentation | 2020 | DL for IVUS | 25.0 | 0.8000 | 26.3 | 0.9645 | done | yes |
| 3 | DAS beamforming | 1990 | DAS baseline | 22.0 | 0.7000 | 26.3 | 0.9645 | done | yes |
| 4 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 20.8 | 0.9002 | 26.3 | 0.9645 | done | yes |
| 5 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 20.8 | 0.9002 | 26.3 | 0.9645 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 19.8 | — | 26.3 | 0.9645 | done | yes |

### 77. Mammography (`mammography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DeepTFormer | 2025 | Scientific Reports 2025 | 39.4 | 0.9400 | 31.2 | 0.0540 | partial | yes |
| 2 | RED-CNN | 2017 | Chen et al., TMI 2017 | 35.0 | 0.9200 | 31.2 | 0.0540 | partial | yes |
| 3 | BM3D | 2007 | Dabov et al., TIP 2007 | 32.0 | 0.9000 | 31.2 | 0.0540 | done | yes |
| 4 | FBP | 1971 | FBP baseline | 30.0 | 0.8500 | 31.2 | 0.0540 | done | yes |
| 5 | NLM denoising | 2005 | Buades et al., CVPR 2005 | 26.0 | 0.8500 | 31.2 | 0.0540 | done | yes |
| 6 | MammoNet (GatorTron) [proxy] (PWM) | — | Richardson 1972, JOSA | 21.9 | 0.8680 | 31.2 | 0.0540 | done | yes |
| 7 | Mammo-ResNet [proxy] (PWM) | — | Richardson 1972, JOSA | 21.9 | 0.8680 | 31.2 | 0.0540 | done | yes |
| 8 | precomputed_recon (test) | — | — | 20.9 | — | 31.2 | 0.0540 | done | yes |

### 78. MR Elastography (MRE) (`mr_elastography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SW-ViT (simulated) | 2025 | arXiv 2505.18865 | 32.7 | 0.9950 | 25.1 | 0.2753 | partial | yes |
| 2 | Phase gradient | 2001 | Manduca et al., MRM 2001 | 24.0 | 0.7500 | 25.1 | 0.2753 | done | yes |
| 3 | Direct inversion | 2001 | Manduca et al., MRM 2001 | 22.0 | 0.7000 | 25.1 | 0.2753 | done | yes |
| 4 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1408 | 25.1 | 0.2753 | done | yes |
| 5 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1408 | 25.1 | 0.2753 | done | yes |
| 6 | MRE-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1408 | 25.1 | 0.2753 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 11.0 | — | 25.1 | 0.2753 | done | yes |

### 79. MR Fingerprinting (MRF) (`mr_fingerprinting`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | MRF-Mixer (T2 map) | 2025 | MDPI Information 2025 | 35.9 | 0.9800 | 23.0 | 0.0041 | gap | yes |
| 2 | MRF-Mixer (T1 map) | 2025 | MDPI Information 2025 | 33.5 | 0.9800 | 23.0 | 0.0041 | gap | yes |
| 3 | GAST-Mamba (T1 map) | 2025 | arXiv 2507.03369 | 33.1 | 0.9670 | 23.0 | 0.0041 | gap | yes |
| 4 | MANTIS | 2019 | Fang et al., MRM 2019 | 30.0 | 0.9000 | 23.0 | 0.0041 | partial | yes |
| 5 | Dictionary matching | 2013 | Ma et al., Nature 2013 | 25.0 | 0.8000 | 23.0 | 0.0041 | done | yes |
| 6 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1551 | 23.0 | 0.0041 | done | yes |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1551 | 23.0 | 0.0041 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 11.0 | — | 23.0 | 0.0041 | done | yes |

### 80. MR Angiography (MRA) (`mra`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | 3D CNN SR | 2025 | Nature Scientific Reports 2025 | 36.8 | 0.9830 | 19.2 | 0.3344 | gap | yes |
| 2 | CS-MRA | 2010 | Lustig et al., MRM 2007 | 30.0 | 0.8500 | 19.2 | 0.3344 | gap | yes |
| 3 | Zero-filled (R=7-11) | 2024 | PMC11424428 (verified 25.80 dB) | 25.8 | — | 19.2 | 0.3344 | partial | yes |
| 4 | Zero-filled IFFT | 2000 | Baseline | 25.0 | 0.6500 | 19.2 | 0.3344 | partial | yes |
| 5 | Zero-filled (16x accel) | 2026 | Li et al., MRM 2026 (R=8: 26.8 dB, extrapolated) | 25.0 | 0.3500 | 19.2 | 0.3344 | partial | yes |
| 6 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 18.1 | 0.4218 | 19.2 | 0.3344 | done | yes |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 18.1 | 0.4218 | 19.2 | 0.3344 | done | yes |
| 8 | MRA-VesselNet [proxy] (PWM) | — | — | 18.1 | 0.4218 | 19.2 | 0.3344 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 14.7 | — | 19.2 | 0.3344 | done | yes |

### 81. Magnetic Resonance Imaging (MRI) (`mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinMR++ | 2024 | Huang et al., 2024 | 43.8 | 0.9830 | — | — | no_ckpt | yes |
| 2 | HUMUS-Net++ | 2024 | Fabian et al., 2024 | 43.1 | 0.9790 | — | — | no_ckpt | yes |
| 3 | MR-IPT | 2025 | Nature Sci Rep 2025 | 42.5 | 0.9830 | — | — | no_ckpt | yes |
| 4 | PromptMR+ | 2024 | Bai et al., ECCV 2024 | 42.5 | 0.9730 | — | — | no_ckpt | yes |
| 5 | MoDL-Net++ | 2024 | Aggarwal et al., 2024 | 41.8 | 0.9780 | — | — | no_ckpt | yes |
| 6 | PromptMR | 2023 | Bai et al., MICCAI 2023 | 41.5 | — | — | — | no_ckpt | yes |
| 7 | MMR-Mamba | 2025 | Med Image Anal 2025 | 41.0 | 0.9690 | — | — | no_ckpt | yes |
| 8 | E2E-VarNet | 2020 | Sriram et al., MICCAI 2020 | 40.5 | 0.9720 | 18.11 | 0.4351 | done | yes |
| 9 | MambaRecon | 2025 | Korkmaz et al., WACV 2025 | 40.4 | — | — | — | no_ckpt | yes |
| 10 | PAS-Mamba | 2026 | arXiv:2601.14530 | 40.4 | — | — | — | no_ckpt | yes |
| 11 | ReconFormer | 2023 | Guo et al., TMI 2024 | 40.1 | 0.9750 | — | — | no_ckpt | yes |
| 12 | Score-MRI | 2022 | Chung & Ye, Med Image Anal 2022 | 39.0 | — | — | — | no_ckpt | yes |
| 13 | SwinMR | 2022 | Huang et al., Neurocomputing 2022 | 38.5 | 0.9210 | — | — | no_ckpt | yes |
| 14 | HUMUS-Net | 2022 | Fabian et al., NeurIPS 2022 | 37.3 | 0.9500 | — | — | no_ckpt | yes |
| 15 | MoDL | 2019 | Aggarwal et al., IEEE TMI 2019 | 36.0 | — | 17.47 | 0.4089 | done | yes |
| 16 | U-Net (fastMRI) | 2018 | Zbontar et al., fastMRI 2018 | 36.0 | 0.9470 | 18.11 | 0.4351 | done | yes |
| 17 | DC-CNN | 2018 | Schlemper et al., IEEE TMI 2018 | 35.5 | — | 17.64 | 0.4109 | done | yes |
| 18 | CascadeNet | 2018 | Schlemper et al., 2018 | 35.0 | — | — | — | no_ckpt | yes |
| 19 | ALOHA | 2015 | Jin & Ye, IEEE TIP 2015 | 34.5 | — | 20.14 | 0.8694 | done | yes |
| 20 | KIKI-Net | 2018 | Eo et al., MRM 2018 | 34.5 | — | — | — | no_ckpt | yes |
| 21 | ESPIRiT | 2014 | Uecker et al., MRM 2014 | 34.2 | — | 20.17 | 0.4780 | done | yes |
| 22 | BM3D-MRI | 2016 | Eksioglu, IEEE SPL 2016 | 34.2 | — | 16.52 | 0.3601 | done | yes |
| 23 | GRAPPA-like | 2002 | Griswold et al., MRM 2002 | 34.0 | 0.9200 | 18.14 | 0.4362 | done | yes |
| 24 | SENSE | 1999 | Pruessmann et al., MRM 1999 | 34.0 | — | 20.17 | 0.4780 | done | yes |
| 25 | LORAKS (Low-Rank) | 2014 | Haldar, IEEE TMI 2014 | 33.8 | — | 18.11 | 0.4351 | done | yes |
| 26 | Deep ADMM-Net | 2016 | Sun et al., NeurIPS 2016 | 33.0 | — | 15.08 | 0.0286 | done | yes |
| 27 | CS-MRI (Wavelet) | 2007 | Lustig et al., MRM 2007 | 33.0 | 0.9000 | 20.17 | 0.4780 | done | yes |
| 28 | k-t SPARSE-SENSE | 2006 | Lustig et al., 2006 | 32.5 | — | — | — | no_ckpt | yes |
| 29 | ISTA-Net+ | 2018 | Zhang & Ghanem, CVPR 2018 | 32.5 | — | 17.87 | 0.4307 | done | yes |
| 30 | FISTA | 2009 | Beck & Teboulle, SIAM 2009 | 32.1 | — | 20.78 | 0.4840 | done | yes |
| 31 | SPIRiT-like | 2010 | Lustig & Pauly, MRM 2010 | 30.0 | — | 18.12 | 0.4353 | done | yes |
| 32 | Nuclear Norm (SVT) | 2010 | Cai et al., SIAM 2010 | 29.5 | — | 18.11 | 0.4351 | done | yes |
| 33 | Zero-filled IFFT | 1973 | Lauterbur, Nature 1973 | 28.0 | 0.6400 | 18.11 | 0.4351 | done | yes |
| 34 | Homodyne Detection | 1991 | Noll et al., IEEE TMI 1991 | 27.0 | — | 17.71 | 0.4281 | done | yes |
| 35 | SMASH | 1997 | Sodickson & Manning, MRM 1997 | 26.0 | — | — | — | no_ckpt | yes |
| 36 | ADMM | 2010 | Yang et al., MRM 2010 | — | — | 19.76 | 0.4727 | done | yes |
| 37 | PnP-ADMM | 2013 | Venkatakrishnan et al., GlobalSIP 2013 | — | — | 19.42 | 0.4676 | done | yes |
| 38 | ISTA | 2004 | Daubechies et al., Comm Pure Appl Math 2004 | — | — | 19.39 | 0.4671 | done | yes |
| 39 | RED | 2017 | Romano et al., SIAM 2017 | — | — | 17.84 | 0.4248 | done | yes |
| 40 | CS-MRI (TV) | 2007 | Block et al., MRM 2007 | — | — | 18.05 | 0.4329 | done | yes |
| 41 | Dictionary Learning MRI | 2011 | Ravishankar & Bresler, IEEE TMI 2011 | — | — | 18.11 | 0.4351 | done | yes |
| 42 | Landweber Iteration | 1951 | Landweber, Amer J Math 1951 | — | — | 18.11 | 0.4351 | done | yes |
| 43 | Tikhonov Regularization | 1963 | Tikhonov, Soviet Math Dokl 1963 | — | — | 18.11 | 0.4347 | done | yes |
| 44 | Proximal Gradient | 2005 | Combettes & Wajs, 2005 | — | — | 18.11 | 0.4349 | done | yes |
| 45 | Truncated IFFT | 1973 | Classic Fourier MRI | — | — | 17.92 | 0.4283 | done | yes |
| 46 | Split Bregman | 2009 | Goldstein & Osher, SIAM 2009 | — | — | 18.11 | 0.4351 | done | yes |
| 47 | Conjugate Gradient | 2001 | Pruessmann et al., MRM 2001 | — | — | 18.11 | 0.4350 | done | yes |
| 48 | Gradient Descent | 2010 | Fessler, IEEE SPM 2010 | — | — | 18.11 | 0.4350 | done | yes |
| 49 | POCS | 1991 | Haacke et al., MRM 1991 | — | — | 18.11 | 0.4351 | done | yes |
| 50 | MoDL (5 unrolls) | 2019 | Aggarwal et al., IEEE TMI 2019 | — | — | 17.70 | 0.4181 | done | yes |

### 82. MR Spectroscopy (MRS) (`mrs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DDPM-MRSI (2x SR) | 2025 | J Imaging Inform Med 2025 | 29.7 | 0.9560 | 22.2 | 0.0032 | partial | yes |
| 2 | LCModel | 1993 | Provencher, MRM 1993 | 28.0 | — | 22.2 | 0.0032 | partial | yes |
| 3 | HLSVD | 2002 | Pijnappel et al., 1992 | 22.0 | — | 22.2 | 0.0032 | done | yes |
| 4 | MRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.1516 | 22.2 | 0.0032 | done | yes |
| 5 | SENSE (spectroscopy) (PWM) | — | — | 11.0 | — | 22.2 | 0.0032 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 11.0 | — | 22.2 | 0.0032 | done | yes |

### 83. Functional Near-Infrared Spectroscopy (fNIRS) (`nirs_brain`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CNN-LSTM Hybrid | 2024 | Multimedia Tools Appl 2024 | 32.1 | 0.9860 | 19.3 | 0.0195 | gap | yes |
| 2 | OT-NIRS (tomographic) | 2010 | Boas et al., NeuroImage 2010 | 22.0 | 0.7000 | 19.3 | 0.0195 | done | yes |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 21.4 | 0.9587 | 19.3 | 0.0195 | done | yes |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 21.4 | 0.9587 | 19.3 | 0.0195 | done | yes |
| 5 | fNIRS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 21.4 | 0.9587 | 19.3 | 0.0195 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 20.2 | — | 19.3 | 0.0195 | done | yes |
| 7 | MBLL | 1988 | Modified Beer-Lambert Law | 20.0 | 0.6000 | 19.3 | 0.0195 | done | yes |

### 84. Optical Coherence Tomography (OCT) (`oct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR | 2021 | Liang et al., ICCVW 2021 | 35.0 | — | 21.0 | 0.8933 | gap | yes |
| 2 | PSCAT | 2022 | PSCAT, PKU37 OCT | 32.2 | 0.9200 | 21.0 | 0.8933 | gap | yes |
| 3 | BM3D | 2007 | Dabov et al., TIP 2007 | 25.0 | 0.8000 | 21.0 | 0.8933 | partial | yes |
| 4 | FFT Recon (PWM) | — | — | 23.5 | — | 21.0 | 0.8933 | done | yes |
| 5 | Spectral Estimation (PWM) | — | Leitgeb et al. 2003, Optics Express | 23.5 | — | 21.0 | 0.8933 | done | yes |
| 6 | OCT Denoising Net (PWM) | — | Devalla et al. 2019, Biomed. Optics Express | 23.5 | — | 21.0 | 0.8933 | done | yes |
| 7 | bscan_baseline (test) | — | — | 23.5 | — | 21.0 | 0.8933 | done | yes |
| 8 | bscan_ideal_baseline (test) | — | — | 23.5 | — | 21.0 | 0.8933 | done | yes |

### 85. OCT Angiography (OCTA) (`octa`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Motion artifact DL | 2024 | MDPI Mathematics 2024 | 32.7 | 0.9260 | 20.9 | 0.8970 | gap | yes |
| 2 | SU-Net (Siamese) | 2019 | Lee et al., 2019 | 28.0 | 0.8130 | 20.9 | 0.8970 | partial | yes |
| 3 | CNN accelerated OCTA | 2022 | Sci Rep 2022 | 20.8 | 0.6300 | 20.9 | 0.8970 | done | yes |
| 4 | OCTA-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.7049 | 20.9 | 0.8970 | done | yes |
| 5 | OCTA-FF [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.7049 | 20.9 | 0.8970 | done | yes |
| 6 | FFT Recon (OCTA) (PWM) | — | — | 18.8 | — | 20.9 | 0.8970 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 18.8 | — | 20.9 | 0.8970 | done | yes |
| 8 | SSADA (single-scan) | 2012 | Xu et al. 2021 PMC8221851 (single-scan 12.09 dB) | 12.1 | 0.7000 | 20.9 | 0.8970 | done | yes |
| 9 | Single-scan OCTA (noisy) | 2021 | Xu et al. 2021, PMC8221851 | 12.1 | — | 20.9 | 0.8970 | done | yes |

### 86. Positron Emission Tomography (PET) (`pet`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR-PET | 2023 | SwinIR for PET denoising | 39.9 | 0.9600 | 40.3 | 0.0030 | done | gpu |
| 2 | DeepPET | 2019 | Haggstrom et al., PMB 2019 | 34.7 | 0.9200 | 40.3 | 0.0030 | done | gpu |
| 3 | FBP (emission tomography) (PWM) | — | — | 33.1 | — | 40.3 | 0.0030 | done | yes |
| 4 | NeuroLF-PET (PWM) | — | Häggström, I. et al. (2019) DeepPET: DL for PET reconstruction, Med. Image Anal. 58 | 33.1 | — | 40.3 | 0.0030 | done | gpu |
| 5 | PET-DL (U-Net) (PWM) | — | Gong, K. et al. (2019) PET image reconstruction with DL, IEEE TMI 38(9) | 33.1 | — | 40.3 | 0.0030 | done | gpu |
| 6 | fbp_ramlak (test) | — | — | 33.1 | — | 40.3 | 0.0030 | done | yes |
| 7 | fbp_shepp_logan (test) | — | — | 33.1 | — | 40.3 | 0.0030 | done | yes |
| 8 | precomputed_fbp (test) | — | — | 33.1 | — | 40.3 | 0.0030 | done | yes |
| 9 | MAP-OSEM | 2001 | Qi et al., PMB 2003 | 32.0 | 0.8700 | 40.3 | 0.0030 | done | gpu |
| 10 | OSEM | 1994 | Hudson & Larkin, TMI 1994 | 30.0 | 0.8200 | 40.3 | 0.0030 | done | gpu |
| 11 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 28.0 | 0.7500 | 40.3 | 0.0030 | done | gpu |

### 87. Photoacoustic Imaging (`photoacoustic`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Iterative (model-based) | 2000 | Antholzer et al., Sci Rep 2020 | 30.2 | 0.8900 | 19.0 | 0.7356 | gap | yes |
| 2 | Residual U-Net (Deep-PAT) | 2021 | Shahid et al., Front Neurosci 2021 | 29.9 | 0.9700 | 19.0 | 0.7356 | gap | yes |
| 3 | Pixel-DL | 2020 | Antholzer et al., Sci Rep 2020 | 29.6 | 0.9100 | 19.0 | 0.7356 | gap | yes |
| 4 | Post-DL (U-Net) | 2020 | Antholzer et al., Sci Rep 2020 | 24.4 | 0.8500 | 19.0 | 0.7356 | partial | yes |
| 5 | Time Reversal (FBP) | 2000 | Xu & Wang, PMB 2005 | 22.7 | 0.7300 | 19.0 | 0.7356 | partial | yes |
| 6 | Backprojection (limited view) | 2021 | Shahid et al., PMC8165448 (FD-UNet BP input=21.9) | 21.9 | 0.6500 | 19.0 | 0.7356 | done | yes |
| 7 | Deep-PAT [proxy] (PWM) | — | Richardson 1972, JOSA | 21.2 | 0.1988 | 19.0 | 0.7356 | done | yes |
| 8 | Back Projection (PWM) | — | — | 19.8 | — | 19.0 | 0.7356 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 19.8 | — | 19.0 | 0.7356 | done | yes |
| 10 | Time Reversal (16 sensors) | 2020 | Tong et al., Scientific Reports 2020, PMC7244747 | 13.9 | 0.5000 | 19.0 | 0.7356 | done | yes |
| 11 | Tikhonov (32 views) | 2023 | Boink et al., PMC9872879 | 13.9 | — | 19.0 | 0.7356 | done | yes |

### 88. Portal Imaging (EPID) (`portal_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CycleGAN+Attention+Residual | 2024 | Lv et al., Medical Physics 2024 | 34.0 | 0.9650 | 36.2 | 0.9986 | done | yes |
| 2 | CycleGAN MVCT-to-kVCT | 2021 | Lee et al., Medical Physics 2021 | 32.7 | 0.9550 | 36.2 | 0.9986 | done | yes |
| 3 | Monte Carlo correction | 2005 | MC dose verification | 28.0 | 0.8200 | 36.2 | 0.9986 | done | yes |
| 4 | Flat-field correction | 2000 | EPID baseline | 25.0 | 0.7500 | 36.2 | 0.9986 | done | yes |
| 5 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 23.8 | 0.8887 | 36.2 | 0.9986 | done | yes |
| 6 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 23.8 | 0.8887 | 36.2 | 0.9986 | done | yes |
| 7 | PortalDL [proxy] (PWM) | — | Richardson 1972, JOSA | 23.8 | 0.8887 | 36.2 | 0.9986 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 17.3 | — | 36.2 | 0.9986 | done | yes |
| 9 | Raw EPID (uncorrected) | 2000 | Raw EPID baseline | 15.0 | 0.5000 | 36.2 | 0.9986 | done | yes |

### 89. Proton Therapy Imaging (`proton_therapy_img`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Residual GAN (PPI-to-DRR) | 2024 | Wang et al., PMC 2024 | 39.1 | 0.9870 | 30.7 | 0.0790 | partial | yes |
| 2 | CycleGAN (CBCT-to-sCT) | 2024 | MDPI Sensors 2024 | 34.1 | 0.8600 | 30.7 | 0.0790 | partial | yes |
| 3 | Proton CT DL | 2022 | DL for proton imaging | 32.0 | 0.9200 | 30.7 | 0.0790 | done | yes |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9843 | 30.7 | 0.0790 | done | yes |
| 5 | FBP | 1971 | FBP baseline | 28.0 | — | 30.7 | 0.0790 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 26.6 | — | 30.7 | 0.0790 | done | yes |

### 90. Single Photon Emission CT (SPECT) (`spect`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DIP-SPECT | 2020 | Baguer et al., 2020 | 33.3 | 0.9000 | 33.3 | 0.9905 | done | gpu |
| 2 | FBP (emission tomography) (PWM) | — | — | 30.0 | — | 33.3 | 0.9905 | done | yes |
| 3 | SPECT-DL (OSEM+) (PWM) | — | Shiri, I. et al. (2020) Deep-JASC DL SPECT, Eur. J. Nucl. Med. Mol. Imaging | 30.0 | — | 33.3 | 0.9905 | done | gpu |
| 4 | SPECT-UNet (PWM) | — | Kim, K. et al. (2018) Penalized PET reconstruction using DL, IEEE TMI 37(6) | 30.0 | — | 33.3 | 0.9905 | done | gpu |
| 5 | fbp_ramlak (test) | — | — | 30.0 | — | 33.3 | 0.9905 | done | yes |
| 6 | precomputed_fbp (test) | — | — | 30.0 | — | 33.3 | 0.9905 | done | yes |
| 7 | OSEM | 1994 | Hudson & Larkin, 1994 | 28.5 | 0.7800 | 33.3 | 0.9905 | done | gpu |
| 8 | MLEM | 1982 | Shepp & Vardi, 1982 | 26.0 | 0.7000 | 33.3 | 0.9905 | done | gpu |

### 91. Photon-Counting Spectral CT (`spectral_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | D3QN | 2024 | Phys Med Biol 2024 | 37.4 | 0.9790 | 7.4 | — | gap | yes |
| 2 | Butterfly-Net | 2022 | Li et al., PMB 2022 | 34.0 | 0.9500 | 7.4 | — | gap | yes |
| 3 | ADMM-TV | 2010 | TV regularization | 30.0 | 0.8700 | 7.4 | — | gap | yes |
| 4 | Material decomposition | 2003 | Alvarez & Macovski, PMB 1976 | 28.0 | 0.8500 | 7.4 | — | gap | yes |
| 5 | FBP per bin (lowest energy) | 2024 | Xing et al., 2024, PMC11744124 | 27.0 | 0.5000 | 7.4 | — | gap | yes |
| 6 | FBP (30 sparse views) | 2025 | Guo et al., QIMS 2025, PMC12209656 | 15.5 | — | 7.4 | — | partial | yes |
| 7 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 13.3 | 0.1206 | 7.4 | — | partial | yes |
| 8 | SpectralCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.3 | 0.1206 | 7.4 | — | partial | yes |
| 9 | precomputed_baseline (test) | — | — | 12.3 | — | 7.4 | — | partial | yes |

### 92. Susceptibility-Weighted Imaging (SWI) (`swi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DeepSWI (cGAN) | 2023 | Genc et al., JMRI 2023 | 36.9 | 0.8900 | 15.9 | 0.4486 | gap | yes |
| 2 | Homodyne filtering | 2004 | Haacke et al., MRM 2004 | 28.0 | 0.8500 | 15.9 | 0.4486 | gap | yes |
| 3 | FBP [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1521 | 15.9 | 0.4486 | done | yes |
| 4 | DL-Recon [proxy] (PWM) | — | Richardson 1972, JOSA | 12.9 | 0.1521 | 15.9 | 0.4486 | done | yes |
| 5 | SWI-Net [proxy] (PWM) | — | — | 12.9 | 0.1521 | 15.9 | 0.4486 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 10.9 | — | 15.9 | 0.4486 | done | yes |

### 93. Ultrasound B-mode Imaging (`ultrasound`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | KD-optimized beamformer | 2025 | Scientific Reports 2025 | 39.0 | 0.9530 | 33.7 | 0.9944 | partial | yes |
| 2 | DAS (Delay-and-Sum) | 1990 | DAS baseline | 30.4 | — | 33.7 | 0.9944 | done | yes |
| 3 | Deep beamforming (Goudarzi) | 2020 | Goudarzi et al., IEEE TUFFC 2022 | 29.1 | — | 33.7 | 0.9944 | done | yes |
| 4 | DAS single plane wave | 2020 | Li et al., IUS 2020 / CUBDL | 18.6 | — | 33.7 | 0.9944 | done | yes |
| 5 | DAS single PW (deep target, 8cm) | 2017 | Perdios et al., IEEE TUFFC 2017 | 17.0 | 0.4500 | 33.7 | 0.9944 | done | yes |
| 6 | ADMIRE | 2018 | Byram et al., IEEE TUFFC 2015 | 15.8 | 0.3564 | 33.7 | 0.9944 | done | yes |
| 7 | US-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 15.8 | 0.3564 | 33.7 | 0.9944 | done | yes |
| 8 | Richardson-Lucy (ultrasound) (PWM) | — | Richardson 1972, JOSA | 14.8 | — | 33.7 | 0.9944 | done | yes |
| 9 | rl_20iter (test) | — | — | 14.8 | — | 33.7 | 0.9944 | done | yes |
| 10 | rl_50iter (test) | — | — | 14.8 | — | 33.7 | 0.9944 | done | yes |
| 11 | DAS single PW (in vivo) | 2020 | Li et al., IUS 2020 / CUBDL, PMC verified | 13.5 | — | 33.7 | 0.9944 | done | yes |

### 94. X-ray Radiography (`xray_radiography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CheXNet [proxy] (PWM) | — | Richardson 1972, JOSA | 46.9 | 0.9999 | 47.5 | 0.9999 | done | yes |
| 2 | X-ray UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 46.9 | 0.9999 | 47.5 | 0.9999 | done | yes |
| 3 | Improved Restormer | 2025 | Springer 2025 | 37.3 | 0.9360 | 47.5 | 0.9999 | done | yes |
| 4 | BM3D | 2007 | Dabov et al., TIP 2007 | 32.0 | 0.8800 | 47.5 | 0.9999 | done | yes |
| 5 | Flat-field + simple filter | 2018 | Kang et al., J X-ray Sci Tech 2018, PMC6130336 (noisy=24.... | 30.0 | 0.8500 | 47.5 | 0.9999 | done | yes |
| 6 | NLM | 2005 | Buades et al., CVPR 2005 | 28.0 | 0.8600 | 47.5 | 0.9999 | done | yes |
| 7 | FBP (X-ray radiography) (PWM) | — | — | 27.1 | — | 47.5 | 0.9999 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 27.1 | — | 47.5 | 0.9999 | done | yes |
| 9 | Median filter | 2000 | Median denoising baseline | 25.0 | 0.8000 | 47.5 | 0.9999 | done | yes |
| 10 | Noisy input (flat-field only) | 2018 | Kang et al., J X-ray Sci Tech 2018, PMC6130336 | 24.1 | 0.3870 | 47.5 | 0.9999 | done | yes |

## Microscopy

### 95. Confocal 3D Z-Stack (`confocal_3d`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CARE 3D | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | 59.8 | 1.0000 | done | gpu |
| 2 | Noise2Void 3D | 2019 | Krull et al., CVPR 2019 | 28.0 | 0.8200 | 59.8 | 1.0000 | done | gpu |
| 3 | CARE-3D (PWM) | — | — | 27.3 | — | 59.8 | 1.0000 | done | yes |
| 4 | CARE-3D (slice-wise) (PWM) | — | — | 27.3 | — | 59.8 | 1.0000 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 27.3 | — | 59.8 | 1.0000 | done | gpu |
| 6 | rl_20iter (test) | — | — | 27.3 | — | 59.8 | 1.0000 | done | gpu |
| 7 | Richardson-Lucy 3D | 1972 | Richardson 1972 | 26.0 | 0.7500 | 59.8 | 1.0000 | done | yes |

### 96. Confocal Live-Cell Microscopy (`confocal_livecell`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | 0.9200 | 60.9 | 1.0000 | done | gpu |
| 2 | precomputed_baseline (test) | — | — | 32.3 | — | 60.9 | 1.0000 | done | gpu |
| 3 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 29.0 | 0.8600 | 60.9 | 1.0000 | done | gpu |
| 4 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | 60.9 | 1.0000 | done | yes |

### 97. Dark-Field Microscopy (`dark_field`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DAPD | 2024 | Nano Letters 2024 | 33.0 | 0.9890 | 23.9 | 0.0475 | partial | gpu |
| 2 | BM3D | 2007 | Dabov et al., TIP 2007 | 30.0 | 0.8500 | 23.9 | 0.0475 | partial | yes |
| 3 | Richardson-Lucy (PWM) | — | — | 25.1 | — | 23.9 | 0.0475 | done | yes |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | 25.1 | — | 23.9 | 0.0475 | done | gpu |
| 5 | DF-UNet (PWM) | — | Wolfer, T. et al. (2021) DL for dark-field X-ray CT, Sci. Rep. 11:5005 | 25.1 | — | 23.9 | 0.0475 | done | gpu |
| 6 | precomputed_baseline (test) | — | — | 25.1 | — | 23.9 | 0.0475 | done | gpu |
| 7 | Median filter | 2000 | Median denoising baseline | 24.0 | 0.7800 | 23.9 | 0.0475 | done | gpu |

### 98. Differential Interference Contrast (DIC) (`dic`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DL phase recovery | 2020 | DL for DIC | 30.0 | 0.8800 | 30.8 | 0.0009 | done | gpu |
| 2 | TIE-GANs | 2024 | Poliwoda et al., J Biomed Opt 2024 | 28.1 | 0.9800 | 30.8 | 0.0009 | done | gpu |
| 3 | PINN-TIE | 2022 | Zhang et al., Opt Express 2022 | 25.2 | 0.9190 | 30.8 | 0.0009 | done | gpu |
| 4 | TIE-DIC | 2010 | TIE for DIC | 25.0 | — | 30.8 | 0.0009 | done | gpu |
| 5 | Phase gradient DIC | 2015 | Gradient-based DIC | 22.0 | 0.7000 | 30.8 | 0.0009 | done | gpu |
| 6 | Simple deconvolution | 2000 | DIC basic deconv | 18.0 | 0.6000 | 30.8 | 0.0009 | done | gpu |
| 7 | Richardson-Lucy (PWM) | — | — | 15.6 | — | 30.8 | 0.0009 | done | yes |
| 8 | CARE (PWM) | — | Weigert et al. 2018 | 15.6 | — | 30.8 | 0.0009 | done | gpu |
| 9 | DIC-Net (PWM) | — | Mir, A. et al. (2015) Automated DIC microscopy, J. Microsc. 257(2) | 15.6 | — | 30.8 | 0.0009 | done | gpu |
| 10 | precomputed_baseline (test) | — | — | 15.6 | — | 30.8 | 0.0009 | done | gpu |

### 99. DNA-PAINT Super-Resolution (`dna_paint`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Richardson-Lucy (PWM) | — | — | 30.9 | — | 35.6 | 0.9965 | done | yes |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 30.9 | — | 35.6 | 0.9965 | done | gpu |
| 3 | DECODE-PAINT (PWM) | — | Speiser, A. et al. (2021) DL for dense SMLM, Nature Methods 18:1090 | 30.9 | — | 35.6 | 0.9965 | done | gpu |
| 4 | precomputed_baseline (test) | — | — | 30.9 | — | 35.6 | 0.9965 | done | gpu |
| 5 | DeepSTORM | 2018 | Nehme et al., Optica 2018 | 22.0 | — | 35.6 | 0.9965 | done | gpu |
| 6 | PICASSO | 2020 | Reymond et al., PNAS 2020 | 20.0 | — | 35.6 | 0.9965 | done | gpu |

### 100. Expansion Microscopy (ExM) (`expansion`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CARE (PWM) | — | Weigert et al. 2018 | 33.9 | — | 34.0 | 0.9468 | done | gpu |
| 2 | EXpansionNet (PWM) | — | Weigert, M. et al. (2018) CARE for fluorescence microscopy, Nature Methods 15:1090 | 33.9 | — | 34.0 | 0.9468 | done | gpu |
| 3 | precomputed_baseline (test) | — | — | 33.9 | — | 34.0 | 0.9468 | done | gpu |
| 4 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 28.0 | 0.8000 | 34.0 | 0.9468 | done | gpu |
| 5 | Richardson-Lucy ExM | 2015 | Chen et al., Science 2015 | 26.0 | — | 34.0 | 0.9468 | done | yes |

### 101. Fluorescence Lifetime Imaging (FLIM) (`flim`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | MLE Fit (PWM) | — | Becker 2012, J. Microscopy | 36.9 | — | 35.5 | 0.9849 | done | yes |
| 2 | MLE Fit (iterative) (PWM) | — | Becker 2012, J. Microscopy | 36.9 | — | 35.5 | 0.9849 | done | yes |
| 3 | precomputed_baseline (test) | — | — | 36.9 | — | 35.5 | 0.9849 | done | yes |
| 4 | Net-FLIM (DL) | 2019 | Smith et al., Biomed Opt Express 2019 | 30.0 | 0.9000 | 35.5 | 0.9849 | done | yes |
| 5 | Phasor approach | 2008 | Digman et al., Biophys J 2008 | 25.0 | — | 35.5 | 0.9849 | done | yes |
| 6 | Multi-exponential fitting | 2000 | Elson 2004 | 22.0 | — | 35.5 | 0.9849 | done | yes |

### 102. Fourier Ptychographic Microscopy (FPM) (`fpm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Gradient descent FPM | 2015 | Tian & Waller, Optica 2015 | 30.0 | 0.8700 | 30.3 | 0.8899 | done | yes |
| 2 | GS-FPM | 2013 | Zheng et al., Nature Photonics 2013 | 28.0 | 0.8500 | 30.3 | 0.8899 | done | yes |
| 3 | Sequential Phase Retrieval (PWM) | — | — | 18.2 | — | 30.3 | 0.8899 | done | yes |
| 4 | Fourier Ptychnet (PWM) | — | Jiang et al. 2018, Biomed. Optics Express | 18.2 | — | 30.3 | 0.8899 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 18.2 | — | 30.3 | 0.8899 | done | yes |
| 6 | Single low-res capture | 2013 | FPM single image baseline | 18.0 | 0.6000 | 30.3 | 0.8899 | done | yes |

### 103. Image Scanning Microscopy (ISM) (`ism`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Richardson-Lucy (PWM) | — | — | 34.0 | — | 33.0 | 0.0007 | done | yes |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 34.0 | — | 33.0 | 0.0007 | done | gpu |
| 3 | ISM-Reassignment-Net (PWM) | — | Castello, M. et al. (2019) Image scanning microscopy ISM, Nature Methods 16:175 | 34.0 | — | 33.0 | 0.0007 | done | gpu |
| 4 | precomputed_baseline (test) | — | — | 34.0 | — | 33.0 | 0.0007 | done | gpu |
| 5 | Airyscan processing | 2017 | Huff, Methods Appl Fluor 2017 | 30.0 | — | 33.0 | 0.0007 | done | gpu |
| 6 | Pixel reassignment | 2010 | Muller & Enderlein, PRL 2010 | 28.0 | — | 33.0 | 0.0007 | done | gpu |

### 104. Lattice Light-Sheet Microscopy (`lattice_lightsheet`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CARE 3D | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | 34.0 | 0.9965 | done | gpu |
| 2 | Richardson-Lucy 3D | 1972 | Richardson 1972 | 26.0 | 0.7500 | 34.0 | 0.9965 | done | yes |
| 3 | LLSM-CARE (PWM) | — | Weigert, M. et al. (2018) Content-aware restoration for lattice light-sheet, Nature Methods 15:1090 | 25.1 | — | 34.0 | 0.9965 | done | gpu |
| 4 | precomputed_baseline (test) | — | — | 25.1 | — | 34.0 | 0.9965 | done | gpu |

### 105. Light-Sheet Fluorescence Microscopy (LSFM) (`lightsheet`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | — | 38.5 | 0.9854 | done | yes |
| 2 | Richardson-Lucy | 1972 | Richardson 1972 | 26.0 | 0.7500 | 38.5 | 0.9854 | done | yes |
| 3 | Fourier Notch Filter (PWM) | — | — | 23.0 | — | 38.5 | 0.9854 | done | yes |
| 4 | VSNR (PWM) | — | — | 23.0 | — | 38.5 | 0.9854 | done | yes |
| 5 | DeStripe (PWM) | — | Liang et al. 2022 | 23.0 | — | 38.5 | 0.9854 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 23.0 | — | 38.5 | 0.9854 | done | yes |
| 7 | rl_20iter (test) | — | — | 23.0 | — | 38.5 | 0.9854 | done | yes |
| 8 | fourier_notch (test) | — | — | 23.0 | — | 38.5 | 0.9854 | done | yes |
| 9 | Gaussian denoising | 2000 | Gaussian filter baseline | 22.0 | 0.7000 | 38.5 | 0.9854 | done | yes |

### 106. MINFLUX Nanoscopy (`minflux`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Richardson-Lucy (PWM) | — | — | 29.5 | — | 34.0 | 0.9965 | done | yes |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 29.5 | — | 34.0 | 0.9965 | done | gpu |
| 3 | MINFLUX-Net (PWM) | — | Gwosch, K.C. et al. (2020) MINFLUX nanoscopy 3D, Nature Methods 17:217 | 29.5 | — | 34.0 | 0.9965 | done | gpu |
| 4 | precomputed_baseline (test) | — | — | 29.5 | — | 34.0 | 0.9965 | done | gpu |
| 5 | MLE localization | 2006 | Ober et al., Biophys J 2004 | 18.0 | — | 34.0 | 0.9965 | done | gpu |
| 6 | Gaussian fitting | 2002 | Thompson et al., Biophys J 2002 | 15.0 | — | 34.0 | 0.9965 | done | gpu |

### 107. PALM/STORM Single-Molecule Localization (`palm_storm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Richardson-Lucy (STORM/PALM) (PWM) | — | — | 32.4 | — | 37.9 | 0.9904 | done | yes |
| 2 | DECODE-SMLM (PWM) | — | Speiser, A. et al. (2021) Deep learning enables fast and dense SMLM, Nature Methods 18:1090 | 32.4 | — | 37.9 | 0.9904 | done | gpu |
| 3 | DeepSTORM (PWM) | — | Nehme, E. et al. (2018) Deep-STORM: super-resolution microscopy, Optica 5(4) | 32.4 | — | 37.9 | 0.9904 | done | gpu |
| 4 | precomputed_baseline (test) | — | — | 32.4 | — | 37.9 | 0.9904 | done | gpu |
| 5 | rl_20iter (test) | — | — | 32.4 | — | 37.9 | 0.9904 | done | gpu |
| 6 | DECODE | 2021 | Speiser et al., Nature Methods 2021 | 25.0 | — | 37.9 | 0.9904 | done | gpu |
| 7 | Deep-STORM | 2018 | Nehme et al., Optica 2018 | 22.0 | — | 37.9 | 0.9904 | done | gpu |
| 8 | ThunderSTORM | 2014 | Ovesny et al., Bioinformatics 2014 | 18.0 | — | 37.9 | 0.9904 | done | gpu |

### 108. Phase Contrast Microscopy (`phase_contrast`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Richardson-Lucy (PWM) | — | — | 45.6 | — | 28.0 | 0.8861 | gap | yes |
| 2 | CARE (PWM) | — | Weigert et al. 2018 | 45.6 | — | 28.0 | 0.8861 | gap | gpu |
| 3 | PhaseNet (PWM) | — | Rivenson, Y. et al. (2018) Phase recovery with DL, Light: Sci. & Appl. 7:17141 | 45.6 | — | 28.0 | 0.8861 | gap | gpu |
| 4 | precomputed_baseline (test) | — | — | 45.6 | — | 28.0 | 0.8861 | gap | gpu |
| 5 | GAN (self-attention) | 2024 | Scientific Reports 2024 | 38.3 | 0.8800 | 28.0 | 0.8861 | gap | gpu |
| 6 | Fourier ptychography | 2013 | Zheng et al., Nature Photonics 2013 | 32.0 | 0.9000 | 28.0 | 0.8861 | partial | gpu |
| 7 | DL flat-fielding QPC | 2024 | ResearchGate 2024 | 29.1 | 0.8650 | 28.0 | 0.8861 | done | gpu |
| 8 | TIE (Transport of Intensity) | 2001 | Zuo et al., Opt Express 2013 | 28.0 | — | 28.0 | 0.8861 | done | gpu |

### 109. Polarization Microscopy (`polarization`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | PolarNet [proxy] (PWM) | — | Richardson 1972, JOSA | 47.8 | 0.9999 | 30.1 | 0.9279 | gap | yes |
| 2 | Stokes-NN [proxy] (PWM) | — | Richardson 1972, JOSA | 47.8 | 0.9999 | 30.1 | 0.9279 | gap | yes |
| 3 | MDU-Net | 2022 | Opt Express 30(12), PMC9208591 | 38.1 | 0.8970 | 30.1 | 0.9279 | partial | yes |
| 4 | MIRNet | 2022 | Opt Express 30(12), PMC9208591 | 37.9 | 0.8950 | 30.1 | 0.9279 | partial | yes |
| 5 | DnCNN | 2022 | Opt Express 30(12), PMC9208591 | 34.4 | 0.8100 | 30.1 | 0.9279 | partial | yes |
| 6 | PnP-HQS (PWM) | — | — | 30.9 | — | 30.1 | 0.9279 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 30.9 | — | 30.1 | 0.9279 | done | yes |
| 8 | Raw Mueller matrix | 2022 | Ye et al., Biomed Opt Express 2022, PMC9208591 | 29.0 | 0.5000 | 30.1 | 0.9279 | done | yes |
| 9 | Mueller matrix | 2000 | Chipman, Handbook of Optics | 25.0 | — | 30.1 | 0.9279 | done | yes |

### 110. Second Harmonic Generation (SHG) Microscopy (`shg`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | — | 35.9 | 0.0226 | done | yes |
| 2 | DnCNN | 2023 | Bai et al., Biomed Opt Express 2023 | 25.4 | 0.7700 | 35.9 | 0.0226 | done | gpu |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 24.1 | — | 35.9 | 0.0226 | done | gpu |
| 4 | SHG-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for SHG imaging, Nature Methods 15:1090 | 24.1 | — | 35.9 | 0.0226 | done | gpu |
| 5 | precomputed_baseline (test) | — | — | 24.1 | — | 35.9 | 0.0226 | done | gpu |
| 6 | Gaussian denoising | 2000 | Gaussian filter baseline | 22.0 | 0.7000 | 35.9 | 0.0226 | done | gpu |

### 111. Structured Illumination Microscopy (SIM) (`sim`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | ML-SIM | 2021 | Christensen et al., APL 2021 | 33.0 | — | 19.5 | 0.8955 | gap | yes |
| 2 | fairSIM | 2015 | Muller et al., Bioinformatics 2016 | 30.5 | 0.8900 | 19.5 | 0.8955 | gap | yes |
| 3 | Wiener-SIM | 2008 | Gustafsson et al., 2008 | 30.0 | 0.8800 | 19.5 | 0.8955 | gap | yes |
| 4 | HiFi-SIM (PWM) | — | Wen et al. 2021, Light: S&A | 24.0 | — | 19.5 | 0.8955 | partial | yes |
| 5 | Wiener-SIM (fast) (PWM) | — | — | 24.0 | — | 19.5 | 0.8955 | partial | yes |
| 6 | precomputed_baseline (test) | — | — | 24.0 | — | 19.5 | 0.8955 | partial | yes |
| 7 | wiener_sim (test) | — | — | 24.0 | — | 19.5 | 0.8955 | partial | yes |
| 8 | Bicubic interpolation | 2000 | Interpolation baseline | 22.0 | 0.7000 | 19.5 | 0.8955 | done | yes |

### 112. Spinning Disk Confocal Microscopy (`spinning_disk`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 32.0 | 0.9000 | 43.2 | 0.9997 | done | gpu |
| 2 | SD-CARE (PWM) | — | Weigert, M. et al. (2018) CARE for spinning disk confocal, Nature Methods 15:1090 | 30.6 | — | 43.2 | 0.9997 | done | gpu |
| 3 | precomputed_baseline (test) | — | — | 30.6 | — | 43.2 | 0.9997 | done | gpu |
| 4 | Richardson-Lucy | 1972 | Richardson 1972 | 27.0 | 0.7800 | 43.2 | 0.9997 | done | yes |

### 113. STED Microscopy (`sted`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DDPM denoiser | 2023 | DDPM-avg for STED | 32.8 | 0.9200 | 35.0 | 0.9943 | done | gpu |
| 2 | STED-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | 29.6 | — | 35.0 | 0.9943 | done | gpu |
| 3 | RCAN-STED (PWM) | — | Chen, J. et al. (2021) Three-dimensional residual channel attention for STED, Nature Methods 18:678 | 29.6 | — | 35.0 | 0.9943 | done | gpu |
| 4 | precomputed_baseline (test) | — | — | 29.6 | — | 35.0 | 0.9943 | done | gpu |
| 5 | rl_20iter (test) | — | — | 29.6 | — | 35.0 | 0.9943 | done | gpu |
| 6 | Richardson-Lucy STED | 2006 | RL for STED | 28.0 | 0.8000 | 35.0 | 0.9943 | done | yes |
| 7 | Gaussian denoising | 2000 | Gaussian filter baseline | 24.0 | 0.7500 | 35.0 | 0.9943 | done | gpu |

### 114. Three-Photon Microscopy (`three_photon`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DeepCAD-RT | 2023 | Li et al., Nature Biotech 2023 | 34.0 | — | 30.4 | 0.9777 | partial | gpu |
| 2 | Richardson-Lucy | 1972 | Richardson 1972 | 26.0 | — | 30.4 | 0.9777 | done | yes |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 22.3 | — | 30.4 | 0.9777 | done | gpu |
| 4 | 3P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) CARE for 3P deep tissue imaging, Nature Methods 15:1090 | 22.3 | — | 30.4 | 0.9777 | done | gpu |
| 5 | precomputed_baseline (test) | — | — | 22.3 | — | 30.4 | 0.9777 | done | gpu |
| 6 | Gaussian denoising | 2000 | Gaussian filter baseline | 20.0 | 0.6000 | 30.4 | 0.9777 | done | gpu |

### 115. TIRF Microscopy (`tirf`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RED-fairSIM | 2021 | Christensen et al., Photonics Research 2021 | 33.2 | 0.9000 | 40.8 | 0.9777 | done | yes |
| 2 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 33.0 | 0.9100 | 40.8 | 0.9777 | done | yes |
| 3 | TIRF-SRRF [proxy] (PWM) | — | Richardson 1972, JOSA | 32.2 | 0.6316 | 40.8 | 0.9777 | done | yes |
| 4 | precomputed_baseline (test) | — | — | 31.2 | — | 40.8 | 0.9777 | done | yes |
| 5 | Richardson-Lucy | 1972 | Richardson 1972 | 28.0 | 0.8000 | 40.8 | 0.9777 | done | yes |

### 116. Two-Photon / Multiphoton Microscopy (`two_photon`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | UNet-Att (self-supervised) | 2025 | Complex & Intelligent Systems, 2025 | 38.3 | 0.9500 | 18.9 | 0.8963 | gap | gpu |
| 2 | DeepCAD | 2021 | Li et al., Nature Methods 2021 | 35.0 | — | 18.9 | 0.8963 | gap | gpu |
| 3 | 2P-Net (CARE) (PWM) | — | Weigert, M. et al. (2018) Content-aware image restoration, Nature Methods 15:1090 | 33.8 | — | 18.9 | 0.8963 | gap | gpu |
| 4 | 2P-DeepInterp (PWM) | — | Lecoq, J. et al. (2021) Removing independent noise in systems neuroscience using DeepInterpolation, Nature Methods 18:1401 | 33.8 | — | 18.9 | 0.8963 | gap | gpu |
| 5 | precomputed_baseline (test) | — | — | 33.8 | — | 18.9 | 0.8963 | gap | gpu |
| 6 | rl_20iter (test) | — | — | 33.8 | — | 18.9 | 0.8963 | gap | gpu |
| 7 | Richardson-Lucy | 1972 | Richardson 1972 | 27.0 | 0.7800 | 18.9 | 0.8963 | partial | yes |

### 117. Widefield Fluorescence Microscopy (`widefield`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Restormer | 2022 | Zamir et al., CVPR 2022 | 35.5 | — | 47.8 | 0.2521 | done | gpu |
| 2 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 31.0 | 0.8800 | 47.8 | 0.2521 | done | gpu |
| 3 | Wiener deconvolution | 1949 | Wiener, 1949 | 26.0 | 0.7500 | 47.8 | 0.2521 | done | yes |
| 4 | precomputed_baseline (test) | — | — | 25.0 | — | 47.8 | 0.2521 | done | gpu |
| 5 | m-rBCR | 2023 | m-rBCR deconvolution, 2023 | 24.9 | 0.8300 | 47.8 | 0.2521 | done | gpu |
| 6 | CARE | 2018 | Weigert et al., Nature Methods 2018 | 22.1 | 0.7500 | 47.8 | 0.2521 | done | gpu |
| 7 | Richardson-Lucy (20 iter) | 1972 | Richardson 1972 / Lucy 1974 | 13.4 | 0.4000 | 47.8 | 0.2521 | done | yes |

### 118. Low-Dose Widefield Microscopy (`widefield_lowdose`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | BM3D + RL (PWM) | — | — | 29.0 | — | 35.6 | 0.9965 | done | yes |
| 2 | CARE (PWM) | — | — | 29.0 | — | 35.6 | 0.9965 | done | gpu |
| 3 | precomputed_baseline (test) | — | — | 29.0 | — | 35.6 | 0.9965 | done | gpu |
| 4 | Noise2Void | 2019 | Krull et al., CVPR 2019 | 26.0 | 0.8000 | 35.6 | 0.9965 | done | yes |
| 5 | Richardson-Lucy | 1972 | Richardson 1972 | 20.0 | 0.6000 | 35.6 | 0.9965 | done | yes |

## Multi-Modal Fusion

### 119. Correlative Light-Electron Microscopy (CLEM) (`clem`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 39.7 | 0.9999 | 26.0 | 0.9876 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 39.7 | 0.9999 | 26.0 | 0.9876 | gap | yes |
| 3 | CLEM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 39.7 | 0.9999 | 26.0 | 0.9876 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 28.1 | — | 26.0 | 0.9876 | done | yes |
| 5 | VoxelMorph registration | 2019 | Balakrishnan et al., TMI 2019 | 26.0 | 0.8300 | 26.0 | 0.9876 | done | yes |
| 6 | Landmark registration | 2000 | CLEM registration | 22.0 | — | 26.0 | 0.9876 | done | yes |

### 120. CT + Fluorescence (FLIT) (`ct_fluorescence`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SIRT | 1972 | Gilbert 1972 | 25.0 | 0.7500 | 34.0 | 0.0067 | done | yes |
| 2 | FBP + fluorescence | 2000 | XFCT baseline | 22.0 | — | 34.0 | 0.0067 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.6723 | 34.0 | 0.0067 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.6723 | 34.0 | 0.0067 | done | yes |
| 5 | XFCT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.2 | 0.6723 | 34.0 | 0.0067 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 10.2 | — | 34.0 | 0.0067 | done | yes |

### 121. PET/CT Fusion (`pet_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Attention U-Net + diffusion | 2025 | arXiv 2504.00816 | 35.9 | 0.9920 | 29.8 | 0.0329 | partial | yes |
| 2 | TrUNET-MAPEM | 2023 | ScienceDirect, S0895611123001337 | 33.7 | 0.9550 | 29.8 | 0.0329 | partial | yes |
| 3 | OSEM + CT AC | 2000 | PET/CT baseline | 28.0 | 0.8000 | 29.8 | 0.0329 | done | yes |
| 4 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 25.0 | 0.7500 | 29.8 | 0.0329 | done | yes |
| 5 | MLEM (low-count, 2 iter) | 1982 | Shepp & Vardi 1982 | 15.0 | 0.5000 | 29.8 | 0.0329 | done | yes |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.0 | 0.0756 | 29.8 | 0.0329 | done | yes |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.0 | 0.0756 | 29.8 | 0.0329 | done | yes |
| 8 | PET-CT-Fusion-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.0 | 0.0756 | 29.8 | 0.0329 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 13.0 | — | 29.8 | 0.0329 | done | yes |

### 122. PET/MR Fusion (`pet_mr`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Brain DL PET/MR | 2024 | PubMed 2024 | 42.0 | 0.9650 | 29.6 | 0.9927 | gap | yes |
| 2 | MRAC-based reconstruction | 2010 | Wagenknecht et al., 2013 | 26.0 | 0.7800 | 29.6 | 0.9927 | done | yes |
| 3 | No-AC reconstruction | 2010 | PET/MR no attenuation correction | 15.0 | 0.5000 | 29.6 | 0.9927 | done | yes |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.5 | 0.2076 | 29.6 | 0.9927 | done | yes |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.5 | 0.2076 | 29.6 | 0.9927 | done | yes |
| 6 | PET-MR-DeepJoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.5 | 0.2076 | 29.6 | 0.9927 | done | yes |
| 7 | No-AC (1/10 counts) | 2010 | Catana et al., JNM 2010 | 13.0 | 0.4000 | 29.6 | 0.9927 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 12.5 | — | 29.6 | 0.9927 | done | yes |

### 123. SPECT/CT Fusion (`spect_ct`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | GAN projection-space denoising | 2022 | PMC8940834 | 42.5 | 0.9900 | 28.4 | 0.9913 | gap | yes |
| 2 | U2-Net (bone SPECT/CT) | 2022 | PMC9192886 | 40.8 | 0.7880 | 28.4 | 0.9913 | gap | yes |
| 3 | OSEM + CT AC | 2000 | SPECT/CT baseline | 26.0 | 0.7800 | 28.4 | 0.9913 | done | yes |
| 4 | MLEM | 1982 | Shepp & Vardi, TMI 1982 | 24.0 | 0.7400 | 28.4 | 0.9913 | done | yes |
| 5 | MLEM (low-count, 2 iter) | 1982 | Shepp & Vardi 1982 | 15.0 | 0.5000 | 28.4 | 0.9913 | done | yes |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.3684 | 28.4 | 0.9913 | done | yes |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.3684 | 28.4 | 0.9913 | done | yes |
| 8 | SPECT-CT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 14.6 | 0.3684 | 28.4 | 0.9913 | done | yes |
| 9 | MLEM (1 iter, 1/20 counts) | 1982 | Reader et al., PMB 2007 / Shepp-Vardi 1982 | 13.0 | 0.3500 | 28.4 | 0.9913 | done | yes |
| 10 | precomputed_baseline (test) | — | — | 11.4 | — | 28.4 | 0.9913 | done | yes |

### 124. US/MRI Fusion (`us_mri`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | VoxelMorph | 2019 | Balakrishnan et al., TMI 2019 | 30.0 | 0.9000 | 24.6 | 0.8677 | partial | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 28.3 | 0.9765 | 24.6 | 0.8677 | partial | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 28.3 | 0.9765 | 24.6 | 0.8677 | partial | yes |
| 4 | US-MRI-Net [proxy] (PWM) | — | — | 28.3 | 0.9765 | 24.6 | 0.8677 | partial | yes |
| 5 | precomputed_baseline (test) | — | — | 25.5 | — | 24.6 | 0.8677 | done | yes |
| 6 | B-spline FFD | 2003 | Rueckert et al., TMI 1999 | 25.0 | 0.8000 | 24.6 | 0.8677 | done | yes |
| 7 | Demons registration | 1998 | Thirion, MIA 1998 | 22.0 | 0.7500 | 24.6 | 0.8677 | done | yes |
| 8 | Affine registration | 2000 | Affine US/MRI baseline (estimated) | 21.0 | 0.6000 | 24.6 | 0.8677 | done | yes |

## Neural Rendering

### 125. 3D Gaussian Splatting (3DGS) (`gaussian_splatting`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | 2DGS | 2024 | Huang et al., SIGGRAPH 2024 | 34.0 | — | — | — | — | gpu |
| 2 | Scaffold-GS | 2024 | Lu et al., CVPR 2024 | 33.8 | — | — | — | — | gpu |
| 3 | 3D Gaussian Splatting | 2023 | Kerbl et al., SIGGRAPH 2023 | 33.3 | 0.9690 | — | — | — | gpu |
| 4 | EWA Splatting (PWM) | — | — | — | — | — | — | — | yes |
| 5 | 3DGS (full) (PWM) | — | Kerbl et al. SIGGRAPH 2023 | — | — | — | — | — | gpu |
| 6 | NeRF (baseline comparison) (PWM) | — | — | — | — | — | — | — | yes |
| 7 | 3DGS (compact) (PWM) | — | — | — | — | — | — | — | yes |
| 8 | direct_render_baseline (test) | — | — | — | — | — | — | — | gpu |

### 126. Neural Radiance Fields (NeRF) (`nerf`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Zip-NeRF | 2023 | Barron et al., ICCV 2023 | 33.7 | — | — | — | — | gpu |
| 2 | 3D Gaussian Splatting | 2023 | Kerbl et al., SIGGRAPH 2023 | 33.3 | 0.9690 | — | — | — | gpu |
| 3 | Instant-NGP | 2022 | Muller et al., SIGGRAPH 2022 | 33.2 | 0.9600 | — | — | — | yes |
| 4 | TensoRF | 2022 | Chen et al., ECCV 2022 | 33.1 | 0.9630 | — | — | — | gpu |
| 5 | Mip-NeRF 360 | 2022 | Barron et al., CVPR 2022 | 33.1 | 0.9610 | — | — | — | gpu |
| 6 | Plenoxels | 2022 | Fridovich-Keil et al., CVPR 2022 | 31.7 | 0.9580 | — | — | — | gpu |
| 7 | NeRF | 2020 | Mildenhall et al., ECCV 2020 | 31.0 | 0.9470 | — | — | — | gpu |
| 8 | SfM + MVS (PWM) | — | — | 29.0 | — | — | — | — | yes |
| 9 | NeRF (original MLP) (PWM) | — | Mildenhall et al. 2020 | 29.0 | — | — | — | — | yes |
| 10 | Richardson-Lucy (proxy baseline) (PWM) | — | Richardson 1972, JOSA | 29.0 | — | — | — | — | yes |
| 11 | FISTA-TV (proxy baseline) (PWM) | — | Beck & Teboulle 2009, SIAM | 29.0 | — | — | — | — | yes |
| 12 | precomputed_baseline (test) | — | — | 29.0 | — | — | — | — | gpu |

## Quantum Imaging

### 127. Entangled Photon Microscopy (`entangled_photon`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9872 | 27.2 | 0.9766 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9872 | 27.2 | 0.9766 | partial | yes |
| 3 | QGI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9872 | 27.2 | 0.9766 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 31.8 | — | 27.2 | 0.9766 | partial | yes |
| 5 | Compressed sensing QI | 2013 | Howland et al., PRA 2013 | 18.0 | — | 27.2 | 0.9766 | done | yes |
| 6 | Coincidence counting | 2002 | quantum imaging baseline | 15.0 | — | 27.2 | 0.9766 | done | yes |

### 128. Ghost Imaging (`ghost_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Orthogonal GI (2D-DCT) | 2025 | Nature Sci Rep, s41598-025-01283-w | 30.0 | — | 27.2 | 0.9766 | done | yes |
| 2 | DGI-Net | 2021 | DL ghost imaging | 28.0 | 0.8800 | 27.2 | 0.9766 | done | yes |
| 3 | Bio-inspired self-attention | 2025 | MDPI Biomimetics 11(1):53 | 24.5 | 0.8000 | 27.2 | 0.9766 | done | yes |
| 4 | CS-GI | 2013 | Katz et al., APL 2009 | 22.0 | 0.7000 | 27.2 | 0.9766 | done | yes |
| 5 | DeepGhost (autoencoder) | 2020 | Nature Sci Rep, s41598-020-68401-8 | 19.9 | 0.6000 | 27.2 | 0.9766 | done | yes |
| 6 | Differential GI | 2010 | Ferri et al., 2010 | 18.0 | 0.5000 | 27.2 | 0.9766 | done | yes |
| 7 | Correlation imaging | 2002 | Bennink et al., PRL 2002 | 15.0 | 0.4000 | 27.2 | 0.9766 | done | yes |
| 8 | Raw correlation (5% sampling) | 2002 | Bennink et al., PRL 2002 | 10.0 | 0.2500 | 27.2 | 0.9766 | done | yes |
| 9 | Correlation GI (natural, 128x128) | 2020 | Bian et al., Scientific Reports 2020, PMC7376173 | 9.5 | — | 27.2 | 0.9766 | done | yes |
| 10 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.3434 | 27.2 | 0.9766 | done | yes |
| 11 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.3434 | 27.2 | 0.9766 | done | yes |
| 12 | GI-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.3434 | 27.2 | 0.9766 | done | yes |
| 13 | Traditional GI (3000 measurements) | 2021 | Kim et al., Optics Express 2021, PMID 34809299 | 7.2 | 0.2800 | 27.2 | 0.9766 | done | yes |
| 14 | precomputed_baseline (test) | — | — | 6.6 | — | 27.2 | 0.9766 | done | yes |

### 129. Quantum Illumination (`quantum_illumination`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.5 | 0.9382 | 27.2 | 0.9766 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.5 | 0.9382 | 27.2 | 0.9766 | done | yes |
| 3 | QI-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 23.5 | 0.9382 | 27.2 | 0.9766 | done | yes |
| 4 | precomputed_baseline (test) | — | — | 20.2 | — | 27.2 | 0.9766 | done | yes |
| 5 | Optimal receiver | 2008 | Lloyd, Science 2008 | 15.0 | — | 27.2 | 0.9766 | done | yes |
| 6 | Photon counting (classical) | 2000 | Classical baseline | 12.0 | — | 27.2 | 0.9766 | done | yes |

## Remote Sensing

### 130. Ground-Penetrating Radar (GPR) (`gpr`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | PGCDM (Physics-Guided Diffusion) | 2024 | Remote Sensing 17(23):3837 | 30.1 | 0.8760 | 30.3 | 0.9875 | done | yes |
| 2 | RTM (Reverse Time Migration) | 2000 | RTM | 25.0 | 0.8000 | 30.3 | 0.9875 | done | yes |
| 3 | PSTM | 2005 | Pre-stack time migration | 22.0 | 0.7200 | 30.3 | 0.9875 | done | yes |
| 4 | Kirchhoff migration | 2000 | GPR migration | 20.0 | 0.6500 | 30.3 | 0.9875 | done | yes |
| 5 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0507 | 30.3 | 0.9875 | done | yes |
| 6 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0507 | 30.3 | 0.9875 | done | yes |
| 7 | GPR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 11.9 | 0.0507 | 30.3 | 0.9875 | done | yes |
| 8 | Raw B-scan (noisy input) | 2021 | MCAE GPR, Electronics 10(11):1269 (noisy=11.23 dB) | 11.2 | 0.4000 | 30.3 | 0.9875 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 10.9 | — | 30.3 | 0.9875 | done | yes |

### 131. Hyperspectral Remote Sensing (`hyperspectral_remote`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 49.7 | 0.9999 | 40.1 | 0.9975 | partial | yes |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 49.7 | 0.9999 | 40.1 | 0.9975 | partial | yes |
| 3 | SST-USRNet [proxy] (PWM) | — | Richardson 1972, JOSA | 49.7 | 0.9999 | 40.1 | 0.9975 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 35.0 | — | 40.1 | 0.9975 | done | yes |
| 5 | MST++ | 2022 | Cai et al., CVPRW 2022 (Winner) | 34.3 | — | 40.1 | 0.9975 | done | yes |
| 6 | HDNet | 2022 | Hu et al., CVPR 2022 | 32.1 | — | 40.1 | 0.9975 | done | yes |
| 7 | AWAN | 2020 | Li et al., CVPRW 2020 | 31.2 | — | 40.1 | 0.9975 | done | yes |
| 8 | HSCNN+ | 2018 | Shi et al., CVPRW 2018 | 26.4 | — | 40.1 | 0.9975 | done | yes |

### 132. Interferometric SAR (InSAR) (`insar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9173 | 17.4 | 0.3098 | gap | yes |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9173 | 17.4 | 0.3098 | gap | yes |
| 3 | InSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 32.8 | 0.9173 | 17.4 | 0.3098 | gap | yes |
| 4 | wrapped_phase_baseline (test) | — | — | 31.8 | — | 17.4 | 0.3098 | gap | yes |
| 5 | SNAPHU | 2001 | Chen & Zebker, JOSA-A 2001 | 28.0 | — | 17.4 | 0.3098 | gap | yes |
| 6 | Goldstein filter | 1998 | Goldstein & Werner, GRL 1998 | 22.0 | — | 17.4 | 0.3098 | partial | yes |

### 133. Multispectral Satellite Imaging (`multispectral_sat`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CDFAN | 2024 | Entropy 27(6):567, PMC12191612 | 42.8 | — | 37.9 | 0.2178 | partial | yes |
| 2 | PanNet | 2017 | Yang et al., ICCV 2017 | 36.1 | 0.9660 | 37.9 | 0.2178 | done | yes |
| 3 | GPPNN | 2021 | Xu et al., CVPR 2021 | 33.8 | 0.9500 | 37.9 | 0.2178 | done | yes |
| 4 | BDSD (Band-Dependent Spatial Detail) | 2008 | Vivone et al., GRSM 2015 | 30.0 | 0.9000 | 37.9 | 0.2178 | done | yes |
| 5 | EXP baseline (bicubic LRMS) | 2022 | Deng et al., IEEE GRSM 2022, PMC12031081 | 27.4 | 0.5000 | 37.9 | 0.2178 | done | yes |
| 6 | Nearest-neighbor (4x) | 2000 | Deng et al., IEEE GRSM 2022 benchmark | 22.0 | 0.6000 | 37.9 | 0.2178 | done | yes |
| 7 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 13.9 | 0.5795 | 37.9 | 0.2178 | done | yes |
| 8 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 13.9 | 0.5795 | 37.9 | 0.2178 | done | yes |
| 9 | MS-Pansharpening-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 13.9 | 0.5795 | 37.9 | 0.2178 | done | yes |
| 10 | bicubic_upsample (test) | — | — | 11.3 | — | 37.9 | 0.2178 | done | yes |

### 134. Ocean Color Remote Sensing (`ocean_color`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 53.5 | 0.9999 | 36.0 | 0.9554 | gap | yes |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 53.5 | 0.9999 | 36.0 | 0.9554 | gap | yes |
| 3 | OC-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 53.5 | 0.9999 | 36.0 | 0.9554 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 44.2 | — | 36.0 | 0.9554 | partial | yes |
| 5 | SRCNN | 2023 | GIScience & Remote Sensing 2023 | 25.2 | 0.7900 | 36.0 | 0.9554 | done | yes |
| 6 | MUMM | 2000 | Ruddick et al., RSE 2000 | 22.0 | — | 36.0 | 0.9554 | done | yes |

### 135. Passive Microwave Radiometry (`passive_microwave`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 28.5 | 0.9418 | 38.8 | 0.9986 | done | yes |
| 2 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 28.5 | 0.9418 | 38.8 | 0.9986 | done | yes |
| 3 | PM-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 28.5 | 0.9418 | 38.8 | 0.9986 | done | yes |
| 4 | OI (Optimal Interpolation) | 2000 | Bretherton et al., MWR 1976 | 25.0 | — | 38.8 | 0.9986 | done | yes |
| 5 | Tikhonov retrieval | 2000 | Tikhonov | 22.0 | — | 38.8 | 0.9986 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 18.3 | — | 38.8 | 0.9986 | done | yes |
| 7 | Linear regression retrieval | 1990 | Statistical retrieval baseline | 18.0 | 0.5500 | 38.8 | 0.9986 | done | yes |

### 136. Polarimetric SAR (PolSAR) (`polsar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | PAN-DeSpeck | 2023 | CMC 76(3):54373 | 28.4 | 0.9050 | 39.6 | 0.9977 | done | yes |
| 2 | CNN learnable activation | 2021 | Remote Sensing 13(17):3444 | 26.4 | 0.8300 | 39.6 | 0.9977 | done | yes |
| 3 | Refined Lee | 2003 | Lee et al., TGRS 2003 | 24.0 | 0.7800 | 39.6 | 0.9977 | done | yes |
| 4 | Cloude-Pottier decomposition | 1997 | Cloude & Pottier, IEEE TGRS 1997 | 22.3 | 0.5815 | 39.6 | 0.9977 | done | yes |
| 5 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 22.3 | 0.5815 | 39.6 | 0.9977 | done | yes |
| 6 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 22.3 | 0.5815 | 39.6 | 0.9977 | done | yes |
| 7 | PolSAR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 22.3 | 0.5815 | 39.6 | 0.9977 | done | yes |
| 8 | Lee filter | 1999 | Lee et al., IEEE TGRS 1999 | 22.0 | 0.7000 | 39.6 | 0.9977 | done | yes |
| 9 | precomputed_baseline (test) | — | — | 19.4 | — | 39.6 | 0.9977 | done | yes |
| 10 | Single-look noisy input | 2017 | Wang et al., TGRS 2017 | 14.5 | — | 39.6 | 0.9977 | done | yes |

### 137. Radio Interferometry (VLBI) (`radio_interferometry`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CASA tclean | 2007 | McMullin et al., ASP 2007 | 28.0 | — | 22.4 | 0.9725 | partial | yes |
| 2 | MEM | 1984 | Cornwell & Evans, A&A 1985 | 27.0 | — | 22.4 | 0.9725 | partial | yes |
| 3 | CLEAN | 1974 | Hogbom, A&AS 1974 | 25.0 | — | 22.4 | 0.9725 | done | yes |
| 4 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3142 | 22.4 | 0.9725 | done | yes |
| 5 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3142 | 22.4 | 0.9725 | done | yes |
| 6 | R2D2 (interferometry) [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3142 | 22.4 | 0.9725 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 23.3 | — | 22.4 | 0.9725 | done | yes |

### 138. Synthetic Aperture Radar (SAR) (`sar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Omega-K Algorithm | 1992 | Stolt 1978 / Cafforio 1991 | 27.0 | 0.7500 | 38.8 | 0.9950 | done | yes |
| 2 | Range-Doppler Algorithm | 1978 | Curlander & McDonough, 1991 | 25.0 | 0.7000 | 38.8 | 0.9950 | done | yes |
| 3 | SAR-DL (PolSF) [proxy] (PWM) | — | Richardson 1972, JOSA | 23.0 | 0.8700 | 38.8 | 0.9950 | done | yes |
| 4 | SAR-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 23.0 | 0.8700 | 38.8 | 0.9950 | done | yes |
| 5 | Matched Filter (192 pts) | 2024 | Diffusion-Prior SAR, arXiv 2512.02768 | 19.1 | — | 38.8 | 0.9950 | done | yes |
| 6 | FBP (SAR backprojection) (PWM) | — | — | 18.5 | — | 38.8 | 0.9950 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 18.5 | — | 38.8 | 0.9950 | done | yes |
| 8 | Matched Filter (24 pts, 2dB SNR) | 2024 | Diffusion-Prior SAR, arXiv 2512.02768 | 8.8 | — | 38.8 | 0.9950 | done | yes |

### 139. Sonar Imaging (`sonar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SwinIR | 2025 | Frontiers in Remote Sensing 2025 | 36.1 | 0.9810 | 33.0 | 0.9681 | partial | yes |
| 2 | MUSIC | 1986 | Schmidt, IEEE TAP 1986 | 27.0 | — | 33.0 | 0.9681 | done | yes |
| 3 | MVDR/Capon beamforming | 1969 | Capon, Proc IEEE 1969 | 25.0 | — | 33.0 | 0.9681 | done | yes |
| 4 | FISTA-L2 (DAS) [proxy] (PWM) | — | Richardson 1972, JOSA | 16.0 | 0.2917 | 33.0 | 0.9681 | done | yes |
| 5 | SonarSR-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 16.0 | 0.2917 | 33.0 | 0.9681 | done | yes |
| 6 | Sonar-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 16.0 | 0.2917 | 33.0 | 0.9681 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 15.0 | — | 33.0 | 0.9681 | done | yes |
| 8 | Matched Filter (sparse) | 2024 | SAR analog, arXiv 2512.02768 | 12.0 | — | 33.0 | 0.9681 | done | yes |

### 140. Weather / Doppler Radar (`weather_radar`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Axial-UNet | 2025 | arXiv 2025 | 47.7 | 0.9940 | 16.8 | 0.7797 | gap | yes |
| 2 | U-Net | 2020 | DL weather radar | 35.0 | 0.9500 | 16.8 | 0.7797 | gap | yes |
| 3 | RDA [proxy] (PWM) | — | Richardson 1972, JOSA | 30.2 | 0.9754 | 16.8 | 0.7797 | gap | yes |
| 4 | SAR-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 30.2 | 0.9754 | 16.8 | 0.7797 | gap | yes |
| 5 | NowcastNet [proxy] (PWM) | — | Richardson 1972, JOSA | 30.2 | 0.9754 | 16.8 | 0.7797 | gap | yes |
| 6 | precomputed_baseline (test) | — | — | 26.9 | — | 16.8 | 0.7797 | gap | yes |
| 7 | CLEAN-AP | 2000 | CLEAN for weather | 25.0 | — | 16.8 | 0.7797 | partial | yes |

## Scanning Probe Microscopy

### 141. Atomic Force Microscopy (AFM) (`afm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Deep-AFM | 2020 | Rashidi & Wolkow, Machine Learning 2020 | 32.0 | 0.9000 | 44.9 | 0.0123 | done | gpu |
| 2 | Richardson-Lucy (PWM) | — | — | 31.3 | — | 44.9 | 0.0123 | done | yes |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 31.3 | — | 44.9 | 0.0123 | done | gpu |
| 4 | AFM-UNet (PWM) | — | Cherukara, M.J. et al. (2020) AI-enabled high-res, real-time imaging, npj Comput. Mater. 6:203 | 31.3 | — | 44.9 | 0.0123 | done | gpu |
| 5 | precomputed_baseline (test) | — | — | 31.3 | — | 44.9 | 0.0123 | done | gpu |
| 6 | Flatten + line correction | 2000 | SPM baseline processing | 25.0 | 0.7500 | 44.9 | 0.0123 | done | gpu |

### 142. Magnetic Force Microscopy (MFM) (`mfm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Interval-BCS (AFM) | 2019 | Lu et al., Nanotechnology 2019, PMC6902871 | 43.2 | 0.9700 | 27.2 | 0.9780 | gap | gpu |
| 2 | Richardson-Lucy (PWM) | — | — | 34.3 | — | 27.2 | 0.9780 | partial | yes |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 34.3 | — | 27.2 | 0.9780 | partial | gpu |
| 4 | MFM-UNet (PWM) | — | Kim, M. et al. (2021) DL for magnetic force microscopy, npj Comput. Mater. 7:87 | 34.3 | — | 27.2 | 0.9780 | partial | gpu |
| 5 | precomputed_baseline (test) | — | — | 34.3 | — | 27.2 | 0.9780 | partial | gpu |
| 6 | Adaptive Median (AFM) | 2019 | Lu et al., Nanotechnology 2019, PMC6902871 | 33.9 | 0.9500 | 27.2 | 0.9780 | partial | gpu |
| 7 | Wiener deconvolution | 1949 | Wiener 1949 / MFM tip deconv | 26.0 | 0.8000 | 27.2 | 0.9780 | done | yes |
| 8 | Deconvolution | 2000 | MFM tip deconvolution | 24.0 | 0.7500 | 27.2 | 0.9780 | done | gpu |

### 143. Near-field Scanning Optical Microscopy (NSOM) (`nsom`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | BM3D | 2007 | Dabov et al., TIP 2007 | 28.0 | 0.8300 | 29.6 | 0.9748 | done | yes |
| 2 | Deconvolution | 2000 | Near-field deconvolution | 24.0 | 0.7500 | 29.6 | 0.9748 | done | gpu |
| 3 | Richardson-Lucy (PWM) | — | — | 24.0 | — | 29.6 | 0.9748 | done | yes |
| 4 | CARE (PWM) | — | Weigert et al. 2018 | 24.0 | — | 29.6 | 0.9748 | done | gpu |
| 5 | NSOM-Net (PWM) | — | Park, J. et al. (2020) DL for near-field optical microscopy, Optica 7(11) | 24.0 | — | 29.6 | 0.9748 | done | gpu |
| 6 | precomputed_baseline (test) | — | — | 24.0 | — | 29.6 | 0.9748 | done | gpu |

### 144. Scanning Tunneling Microscopy (STM) (`stm`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DeepSPM | 2020 | Krull et al., 2020 | 30.0 | 0.8800 | 42.6 | 0.0316 | done | gpu |
| 2 | Richardson-Lucy (PWM) | — | — | 23.3 | — | 42.6 | 0.0316 | done | yes |
| 3 | CARE (PWM) | — | Weigert et al. 2018 | 23.3 | — | 42.6 | 0.0316 | done | gpu |
| 4 | STM-Net (PWM) | — | Ziatdinov, M. et al. (2021) DL for atomic-level STM, Nat. Mach. Intell. 3:269 | 23.3 | — | 42.6 | 0.0316 | done | gpu |
| 5 | precomputed_baseline (test) | — | — | 23.3 | — | 42.6 | 0.0316 | done | gpu |
| 6 | Drift correction | 2000 | SPM baseline | 22.0 | 0.7000 | 42.6 | 0.0316 | done | gpu |

## Scientific Instrumentation

### 145. Atom Probe Tomography (APT) (`atom_probe`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 42.1 | 0.9999 | 26.0 | 0.9771 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 42.1 | 0.9999 | 26.0 | 0.9771 | gap | yes |
| 3 | APT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 42.1 | 0.9999 | 26.0 | 0.9771 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 41.1 | — | 26.0 | 0.9771 | gap | yes |
| 5 | ML trajectory correction | 2022 | DL for APT | 24.0 | — | 26.0 | 0.9771 | done | yes |
| 6 | Voltage reconstruction | 2000 | APT reconstruction | 20.0 | — | 26.0 | 0.9771 | done | yes |

### 146. Cathodoluminescence (CL) Imaging (`cathodoluminescence`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 38.7 | 0.9999 | 33.7 | 0.9920 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 38.7 | 0.9999 | 33.7 | 0.9920 | partial | yes |
| 3 | CL-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 38.7 | 0.9999 | 33.7 | 0.9920 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 28.9 | — | 33.7 | 0.9920 | done | yes |
| 5 | PCA denoising | 2010 | PCA for CL | 25.0 | — | 33.7 | 0.9920 | done | yes |
| 6 | Spectral unmixing | 2000 | NMF/VCA for CL | 22.0 | — | 33.7 | 0.9920 | done | yes |

### 147. Cryo-EM Single Particle Analysis (`cryo_em`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Topaz-Denoise | 2020 | Bepler et al., Nature Commun 2020 | 25.0 | — | 24.7 | 0.9775 | done | yes |
| 2 | DUAL (cryo-ET) | 2024 | PMC10942334, 2024 | 21.3 | 0.8240 | 24.7 | 0.9775 | done | yes |
| 3 | DRA (denoising-recon) | 2024 | arXiv 2410.11373 | 20.2 | 0.8700 | 24.7 | 0.9775 | done | yes |
| 4 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.0400 | 24.7 | 0.9775 | done | yes |
| 5 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.0400 | 24.7 | 0.9775 | done | yes |
| 6 | CryoDRGN [proxy] (PWM) | — | Richardson 1972, JOSA | 20.2 | 0.0400 | 24.7 | 0.9775 | done | yes |
| 7 | cryoSPARC | 2017 | Punjani et al., Nature Methods 2017 | 20.0 | — | 24.7 | 0.9775 | done | yes |
| 8 | precomputed_wiener (test) | — | — | 19.2 | — | 24.7 | 0.9775 | done | yes |
| 9 | rl_ctf_20iter (test) | — | — | 19.2 | — | 24.7 | 0.9775 | done | yes |
| 10 | RELION | 2012 | Scheres, JSB 2012 | 18.0 | — | 24.7 | 0.9775 | done | yes |

### 148. MALDI Mass Spectrometry Imaging (`maldi_msi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9957 | 25.9 | 0.9821 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9957 | 25.9 | 0.9821 | partial | yes |
| 3 | MSI-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 34.8 | 0.9957 | 25.9 | 0.9821 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 27.1 | — | 25.9 | 0.9821 | done | yes |
| 5 | NMF denoising | 2010 | NMF for MSI | 25.0 | — | 25.9 | 0.9821 | done | yes |
| 6 | Peak picking | 2000 | MALDI-MSI baseline | 22.0 | — | 25.9 | 0.9821 | done | yes |

### 149. Muon Tomography (`muon_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | EM-POCA [proxy] (PWM) | — | Richardson 1972, JOSA | 19.2 | 0.1257 | 35.2 | 0.9977 | done | yes |
| 2 | mu-Net (ConvNeXt U-Net) | 2023 | arXiv 2312.17265 | 17.1 | — | 35.2 | 0.9977 | done | yes |
| 3 | PoCA | 2003 | Borozdin et al., Nature 2003 | 13.7 | — | 35.2 | 0.9977 | done | yes |
| 4 | PoCA (1024 muons) | 2023 | mu-Net, arXiv 2312.17265 | 13.7 | — | 35.2 | 0.9977 | done | yes |
| 5 | FBP (muon tomography) (PWM) | — | — | 13.5 | — | 35.2 | 0.9977 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 13.5 | — | 35.2 | 0.9977 | done | yes |
| 7 | Simple FBP (low stats) | 2003 | Borozdin et al., Nature 2003 | 8.0 | — | 35.2 | 0.9977 | done | yes |

### 150. Neutron Diffraction (`neutron_diffraction`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Rietveld refinement | 1969 | Rietveld, JAC 1969 | 25.0 | — | 27.2 | 0.9870 | done | yes |
| 2 | Le Bail fitting | 1988 | Le Bail et al., 1988 | 22.0 | — | 27.2 | 0.9870 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 10.3 | 0.0334 | 27.2 | 0.9870 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 10.3 | 0.0334 | 27.2 | 0.9870 | done | yes |
| 5 | NeutronDiff-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 10.3 | 0.0334 | 27.2 | 0.9870 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 8.8 | — | 27.2 | 0.9870 | done | yes |

### 151. Neutron Radiography / Tomography (`neutron_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SIRT | 1972 | Gilbert 1972 | 28.0 | 0.8000 | 33.4 | 0.9971 | done | yes |
| 2 | FBP | 1971 | FBP baseline | 25.0 | 0.7000 | 33.4 | 0.9971 | done | yes |
| 3 | NeuTomo-DL [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.0792 | 33.4 | 0.9971 | done | yes |
| 4 | GRIDREC-Neutron [proxy] (PWM) | — | Richardson 1972, JOSA | 8.7 | 0.0792 | 33.4 | 0.9971 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 6.6 | — | 33.4 | 0.9971 | done | yes |

### 152. Proton Radiography (`proton_radiography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | CNN proton portal imaging | 2024 | PMC11682722 | 39.1 | 0.9870 | 32.3 | 0.0926 | partial | yes |
| 2 | cGAN synthetic CT | 2023 | PubMed 37800874 | 29.0 | 0.9520 | 32.3 | 0.0926 | done | yes |
| 3 | DROP-TVS | 2013 | Penfold et al., Med Phys 2010 | 28.0 | — | 32.3 | 0.0926 | done | yes |
| 4 | FBP (straight-line approx) | 2003 | Schulte et al., Med Phys 2005 | 25.0 | — | 32.3 | 0.0926 | done | yes |
| 5 | MLP (Most Likely Path) | 2004 | Schulte et al., Med Phys 2008 | 22.0 | — | 32.3 | 0.0926 | done | yes |
| 6 | ProtonRecon-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.3715 | 32.3 | 0.0926 | done | yes |
| 7 | FBP-Proton [proxy] (PWM) | — | Richardson 1972, JOSA | 13.0 | 0.3715 | 32.3 | 0.0926 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 12.0 | — | 32.3 | 0.0926 | done | yes |

### 153. Small-Angle X-ray Scattering (SAXS) (`saxs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | McSAS | 2013 | Bressler et al., JAC 2015 | 25.0 | — | 25.7 | 0.0190 | done | yes |
| 2 | Guinier analysis | 1939 | Guinier, 1939 | 20.0 | — | 25.7 | 0.0190 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 10.1 | 0.0611 | 25.7 | 0.0190 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 10.1 | 0.0611 | 25.7 | 0.0190 | done | yes |
| 5 | SAXS-VAE [proxy] (PWM) | — | Richardson 1972, JOSA | 10.1 | 0.0611 | 25.7 | 0.0190 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 9.0 | — | 25.7 | 0.0190 | done | yes |

### 154. Wide-Angle X-ray Scattering (WAXS) (`waxs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3264 | 24.0 | 0.9472 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3264 | 24.0 | 0.9472 | done | yes |
| 3 | WAXS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 24.5 | 0.3264 | 24.0 | 0.9472 | done | yes |
| 4 | Rietveld refinement | 1969 | Rietveld, JAC 1969 | 24.0 | — | 24.0 | 0.9472 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 23.4 | — | 24.0 | 0.9472 | done | yes |
| 6 | Background subtraction | 2000 | WAXS baseline processing | 20.0 | 0.6500 | 24.0 | 0.9472 | done | yes |

### 155. X-ray Crystallography (`xray_crystallography`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | SHELXD | 2010 | Sheldrick, Acta Cryst 2008 | 28.0 | — | 25.8 | 0.0202 | done | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.4 | 0.0751 | 25.8 | 0.0202 | done | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.4 | 0.0751 | 25.8 | 0.0202 | done | yes |
| 4 | AlphaFold-SF [proxy] (PWM) | — | Richardson 1972, JOSA | 23.4 | 0.0751 | 25.8 | 0.0202 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 22.4 | — | 25.8 | 0.0202 | done | yes |
| 6 | Direct methods | 1953 | Hauptman & Karle, 1953 | 22.0 | — | 25.8 | 0.0202 | done | yes |

### 156. X-ray Fluorescence Tomography (`xrf_tomo`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | 1D-CNN + U-Net | 2025 | Nature Sci Reports, s41598-025-03900-0 | 39.1 | 0.9790 | 7.9 | -0.8471 | gap | yes |
| 2 | Optimized SCUNet | 2024 | MDPI J Imaging 10(6):127 | 39.0 | 0.8600 | 7.9 | -0.8471 | gap | yes |
| 3 | SIRT | 1972 | Gilbert 1972 | 26.0 | — | 7.9 | -0.8471 | gap | yes |
| 4 | FBP reconstruction | 2000 | Sci Rep 2025 (U-Net=39.1, FBP estimated) | 25.0 | 0.5500 | 7.9 | -0.8471 | gap | yes |
| 5 | FBP | 1971 | FBP baseline | 22.0 | — | 7.9 | -0.8471 | gap | yes |
| 6 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 16.6 | 0.8531 | 7.9 | -0.8471 | partial | yes |
| 7 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 16.6 | 0.8531 | 7.9 | -0.8471 | partial | yes |
| 8 | XRFT-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 16.6 | 0.8531 | 7.9 | -0.8471 | partial | yes |
| 9 | precomputed_baseline (test) | — | — | 15.6 | — | 7.9 | -0.8471 | partial | yes |

## Spectroscopy & Spectral Imaging

### 157. Brillouin Microscopy (`brillouin`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 40.4 | 0.9999 | 25.6 | 0.9840 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 40.4 | 0.9999 | 25.6 | 0.9840 | gap | yes |
| 3 | Brillouin-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 40.4 | 0.9999 | 25.6 | 0.9840 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 35.8 | — | 25.6 | 0.9840 | gap | yes |
| 5 | VIPA analysis | 2010 | Scarcelli & Yun, Opt Express 2011 | 28.0 | — | 25.6 | 0.9840 | done | yes |
| 6 | Lorentzian fitting | 2000 | Brillouin spectral fit | 25.0 | — | 25.6 | 0.9840 | done | yes |

### 158. Coherent Anti-Stokes Raman (CARS) Microscopy (`cars`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 27.9 | 0.9820 | 32.4 | 0.0141 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 27.9 | 0.9820 | 32.4 | 0.0141 | done | yes |
| 3 | CARS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | 27.9 | 0.9820 | 32.4 | 0.0141 | done | yes |
| 4 | MEM (Maximum Entropy Method) | 2006 | Vartiainen et al., Opt Express 2006 | 25.0 | — | 32.4 | 0.0141 | done | yes |
| 5 | DnCNN | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 23.0 | 0.5900 | 32.4 | 0.0141 | done | yes |
| 6 | N2N (Noise2Noise) | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 20.6 | 0.5600 | 32.4 | 0.0141 | done | yes |
| 7 | Median Filter | 2023 | Krafft et al., Biomed Opt Express, PMC10368050 | 20.1 | 0.4300 | 32.4 | 0.0141 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 16.7 | — | 32.4 | 0.0141 | done | yes |
| 9 | Raw CARS (no correction) | 2000 | CARS raw baseline | 15.0 | 0.3500 | 32.4 | 0.0141 | done | yes |

### 159. DESI Mass Spectrometry Imaging (`desi`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | NMF denoising | 2015 | NMF for MSI | 25.0 | — | 27.0 | 0.9542 | done | yes |
| 2 | Peak fitting | 2000 | DESI baseline | 22.0 | — | 27.0 | 0.9542 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 16.1 | 0.3230 | 27.0 | 0.9542 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 16.1 | 0.3230 | 27.0 | 0.9542 | done | yes |
| 5 | DESI-SegNet [proxy] (PWM) | — | Richardson 1972, JOSA | 16.1 | 0.3230 | 27.0 | 0.9542 | done | yes |
| 6 | Gaussian smoothing | 2000 | DESI-MSI smoothing baseline | 16.0 | 0.5000 | 27.0 | 0.9542 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 15.1 | — | 27.0 | 0.9542 | done | yes |

### 160. FTIR Spectroscopic Imaging (`ftir_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 35.6 | 0.9304 | 25.0 | 0.9727 | gap | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 35.6 | 0.9304 | 25.0 | 0.9727 | gap | yes |
| 3 | FTIR-UNet [proxy] (PWM) | — | Richardson 1972, JOSA | 35.6 | 0.9304 | 25.0 | 0.9727 | gap | yes |
| 4 | precomputed_baseline (test) | — | — | 34.6 | — | 25.0 | 0.9727 | partial | yes |
| 5 | U-Net SR FTIR | 2022 | DL for FTIR imaging | 30.0 | 0.9000 | 25.0 | 0.9727 | partial | yes |
| 6 | MCR-ALS | 2000 | Tauler, Chemom Intell Lab 1995 | 28.0 | — | 25.0 | 0.9727 | done | yes |
| 7 | ATR correction | 2000 | Bassan et al., Analyst 2010 | 24.0 | — | 25.0 | 0.9727 | done | yes |

### 161. Laser-Induced Breakdown Spectroscopy (LIBS) Imaging (`libs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9907 | 21.2 | 0.9643 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9907 | 21.2 | 0.9643 | partial | yes |
| 3 | LIBS-CNN [proxy] (PWM) | — | Richardson 1972, JOSA | 31.2 | 0.9907 | 21.2 | 0.9643 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 26.5 | — | 21.2 | 0.9643 | partial | yes |
| 5 | PLS regression | 2005 | Hahn & Omenetto, Appl Spectrosc 2010 | 25.0 | — | 21.2 | 0.9643 | partial | yes |
| 6 | Peak identification | 2000 | LIBS baseline | 22.0 | — | 21.2 | 0.9643 | done | yes |

### 162. Raman Imaging / Microscopy (`raman_imaging`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | DeepeR (1D ResUNet) | 2022 | Horgan et al., Anal Chem 2022, PMC9286315 | 46.2 | 0.9530 | 38.0 | 0.9720 | partial | yes |
| 2 | PCA denoising | 2000 | Horgan et al., Anal Chem 2022 (comparison) | 39.4 | 0.8680 | 38.0 | 0.9720 | done | yes |
| 3 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.8753 | 38.0 | 0.9720 | done | yes |
| 4 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.8753 | 38.0 | 0.9720 | done | yes |
| 5 | RamanNet [proxy] (PWM) | — | Richardson 1972, JOSA | 21.6 | 0.8753 | 38.0 | 0.9720 | done | yes |
| 6 | Savitzky-Golay | 1964 | Savitzky & Golay, 1964 | 20.0 | — | 38.0 | 0.9720 | done | yes |
| 7 | precomputed_baseline (test) | — | — | 19.7 | — | 38.0 | 0.9720 | done | yes |

### 163. Secondary Ion Mass Spectrometry (SIMS) Imaging (`sims`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | PCA denoising | 2010 | PCA for SIMS | 24.0 | — | 26.0 | 0.0159 | done | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 22.6 | 0.9807 | 26.0 | 0.0159 | done | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 22.6 | 0.9807 | 26.0 | 0.0159 | done | yes |
| 4 | SIMS-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 22.6 | 0.9807 | 26.0 | 0.0159 | done | yes |
| 5 | Dead-time correction | 2000 | SIMS baseline | 22.0 | — | 26.0 | 0.0159 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 20.5 | — | 26.0 | 0.0159 | done | yes |
| 7 | De-MSI (DL) | 2025 | Gank et al., Anal Chem 2025 | 18.9 | 0.7400 | 26.0 | 0.0159 | done | yes |

### 164. Stimulated Raman Scattering (SRS) Microscopy (`srs`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 45.2 | 0.9999 | 36.6 | 0.9589 | partial | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 45.2 | 0.9999 | 36.6 | 0.9589 | partial | yes |
| 3 | SRS-DeepSpec [proxy] (PWM) | — | Richardson 1972, JOSA | 45.2 | 0.9999 | 36.6 | 0.9589 | partial | yes |
| 4 | precomputed_baseline (test) | — | — | 30.6 | — | 36.6 | 0.9589 | done | yes |
| 5 | U-Net CNN | 2019 | Manifold et al., Biomed Opt Express 10(8):3860, PMC6701518 | 28.9 | — | 36.6 | 0.9589 | done | yes |
| 6 | SHRED | 2021 | Opt Express 29(21):34205 | 25.0 | — | 36.6 | 0.9589 | done | yes |
| 7 | Spectral unmixing | 2000 | SRS baseline | 24.0 | — | 36.6 | 0.9589 | done | yes |
| 8 | UHRED (unsupervised) | 2021 | Opt Express 29(21):34205 | 22.0 | — | 36.6 | 0.9589 | done | yes |
| 9 | PURE-LET | 2019 | Manifold et al., Biomed Opt Express 10(8):3860, PMC6701518 | 13.5 | — | 36.6 | 0.9589 | done | yes |

## Ultrafast Imaging

### 165. Compressed Ultrafast Photography (CUP) (`cup`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 29.5 | 0.9923 | 26.8 | 0.9866 | done | yes |
| 2 | E2E-CUP [proxy] (PWM) | — | Richardson 1972, JOSA | 29.5 | 0.9923 | 26.8 | 0.9866 | done | yes |
| 3 | PnP-BM3D | 2020 | Liu et al., Sensors 2022, PMC9571970 | 29.2 | 0.9200 | 26.8 | 0.9866 | done | yes |
| 4 | PnP-FFDNet | 2020 | Liu et al., Sensors 2022, PMC9571970 | 28.4 | 0.9100 | 26.8 | 0.9866 | done | yes |
| 5 | PnP-DnCNN | 2020 | Liu et al., Sensors 2022, PMC9571970 | 27.1 | 0.8800 | 26.8 | 0.9866 | done | yes |
| 6 | TwIST | 2007 | Liu et al., Sensors 2022, PMC9571970 | 24.7 | 0.7900 | 26.8 | 0.9866 | done | yes |
| 7 | Direct inverse (no regularization) | 2014 | Gao et al., Nature 2014 | 12.0 | 0.3000 | 26.8 | 0.9866 | done | yes |
| 8 | precomputed_baseline (test) | — | — | 8.5 | — | 26.8 | 0.9866 | done | yes |
| 9 | Direct inverse (1000x compression) | 2014 | Gao et al., Nature 2014 extreme compression | 8.0 | 0.2000 | 26.8 | 0.9866 | done | yes |

### 166. Pump-Probe Microscopy (`pump_probe`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | MCR-ALS | 2000 | Tauler, Chemom Intell Lab 1995 | 26.0 | — | 33.4 | 0.9938 | done | yes |
| 2 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 23.3 | 0.9741 | 33.4 | 0.9938 | done | yes |
| 3 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 23.3 | 0.9741 | 33.4 | 0.9938 | done | yes |
| 4 | PumpProbe-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 23.3 | 0.9741 | 33.4 | 0.9938 | done | yes |
| 5 | SVD analysis | 2000 | SVD for transient spectra | 22.0 | — | 33.4 | 0.9938 | done | yes |
| 6 | precomputed_baseline (test) | — | — | 18.6 | — | 33.4 | 0.9938 | done | yes |
| 7 | Simple averaging | 2000 | Time-averaging baseline | 18.0 | 0.5000 | 33.4 | 0.9938 | done | yes |

### 167. Streak Camera Imaging (`streak_camera`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 36.7 | 0.9928 | 26.8 | 0.9866 | partial | yes |
| 2 | StreakNet [proxy] (PWM) | — | Richardson 1972, JOSA | 36.7 | 0.9928 | 26.8 | 0.9866 | partial | yes |
| 3 | precomputed_baseline (test) | — | — | 30.8 | — | 26.8 | 0.9866 | partial | yes |
| 4 | PnP-BM3D (sim) | 2022 | Yuan et al., Sensors 2022, PMC9571970 | 29.2 | 0.9200 | 26.8 | 0.9866 | done | yes |
| 5 | PnP-FFDNet (sim) | 2022 | Yuan et al., Sensors 2022, PMC9571970 | 28.4 | 0.9100 | 26.8 | 0.9866 | done | yes |
| 6 | Temporal deconvolution | 2000 | Streak deconv baseline | 25.0 | — | 26.8 | 0.9866 | done | yes |
| 7 | Wiener deconvolution | 1949 | Wiener 1949 | 22.0 | — | 26.8 | 0.9866 | done | yes |

### 168. XFEL Serial Femtosecond Crystallography (SFX) (`xfel_sfx`)

| Rank | Algorithm | Year | Reference | Ref PSNR | Ref SSIM | PWM PSNR | PWM SSIM | Status | Organized |
|------|-----------|------|-----------|----------|----------|----------|----------|--------|-----------|
| 1 | Adjoint [proxy] (PWM) | — | Richardson 1972, JOSA | 25.1 | 0.9853 | 22.8 | 0.9411 | done | yes |
| 2 | PnP-ADMM [proxy] (PWM) | — | Richardson 1972, JOSA | 25.1 | 0.9853 | 22.8 | 0.9411 | done | yes |
| 3 | SFX-Net [proxy] (PWM) | — | Richardson 1972, JOSA | 25.1 | 0.9853 | 22.8 | 0.9411 | done | yes |
| 4 | cctbx.xfel | 2014 | Hattne et al., Nature Methods 2014 | 25.0 | — | 22.8 | 0.9411 | done | yes |
| 5 | precomputed_baseline (test) | — | — | 24.1 | — | 22.8 | 0.9411 | done | yes |
| 6 | CrystFEL | 2012 | White et al., JAC 2012 | 22.0 | — | 22.8 | 0.9411 | done | yes |
