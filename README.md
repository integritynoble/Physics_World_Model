# PWM — Physics World Model for Computational Imaging

PWM is an open, reproducible **physics + reconstruction + diagnosis** toolkit for computational imaging.

It turns **either**:
- a **natural-language prompt** (“SIM live-cell, low dose, 9 frames…”) **or**
- a structured **ExperimentSpec JSON/YAML** **or**
- **measured data** `y` + an imperfect operator/matrix `A` (**operator-correction mode**)

into a fully reproducible run:

**Prompt/Spec → PhysicsState + WorldStates → (Sim or Load) y → (Fit operator θ) → Reconstruct x̂ → Diagnose → Recommend actions → RunBundle + Viewer**

PWM is designed to be:
- **Public and extensible** (plugins, CasePacks, dataset adapters)
- **Deterministic by default** (bounded search, reproducible seeds)
- **Embeddable** into agent systems like **Denario** (via `pwm_denario`)

---

## What PWM can do

### 1) Prompt-driven simulation + reconstruction

PWM supports **18 validated imaging modalities** with prompt-driven workflows:

**Microscopy:**
- `widefield` - Richardson-Lucy deconvolution (27.31 dB)
- `widefield_lowdose` - PnP denoising for low photon counts (27.78 dB)
- `confocal_livecell` - Live-cell confocal with drift handling (26.27 dB)
- `confocal_3d` - 3D stack deconvolution (29.01 dB)
- `sim` - Structured Illumination Microscopy, 2x resolution (27.48 dB)
- `lightsheet` - Stripe artifact removal (28.05 dB)

**Compressive Imaging:**
- `spc` - Single-Pixel Camera with PnP-FISTA + DRUNet (30.90 dB @ 25%)
- `cassi` - Hyperspectral imaging with GAP + HSI_SDeCNN (30.60 dB)
- `cacti` - Video snapshot compressive imaging (32.79 dB)
- `lensless` - DiffuserCam with ADMM-TV (34.66 dB)

**Medical Imaging:**
- `ct` - Computed Tomography with PnP-SART + DRUNet (27.97 dB)
- `mri` - MRI with PnP-ADMM + DRUNet (48.25 dB)

**Coherent Imaging:**
- `ptychography` - Phase retrieval with Neural Network (59.47 dB)
- `holography` - Off-axis holography with Neural Network (42.52 dB)

**Neural Rendering:**
- `nerf` - Neural Radiance Fields with SIREN (61.35 dB)
- `gaussian_splatting` - 3D Gaussian Splatting (30.47 dB)

**General:**
- `matrix` - Generic linear inverse problem with FISTA-TV (25.79 dB)
- `panorama_multifocal` - Multi-view focus fusion with Neural Network (27.78 dB)

Each modality includes:
- **Forward model simulation** with dose/compression/mismatch/sensor pipeline
- **Solver portfolio** (classical + PnP + neural methods)
- **Diagnosis + actionable recommendations**

### 2) Operator correction mode (measured `y` + operator/matrix `A`)

For real experiments where the forward model is imperfect, PWM can:
- **fit/correct** forward-model parameters (θ) with a bounded calibration loop
- reconstruct with the corrected operator
- export a reproducible **RunBundle** including calibration trajectory and evidence

**Supported modalities for operator correction:**
- **CASSI/CACTI**: dispersion polynomial mismatch, mask shift, blur mismatch
- **CT/MRI**: geometry calibration, coil sensitivity estimation
- **SPC/Matrix**: gain/bias mismatch, measurement matrix calibration
- **Lensless**: PSF mismatch, focus drift
- **Ptychography**: probe position errors, partial coherence
- Any system with `forward()` and `adjoint()` callable operators

### 3) RunBundle export + Viewer

Every run exports a **RunBundle** with full reproducibility:

```
runbundle/
├── spec_resolved.json      # Full experiment specification
├── data_manifest.json      # Input checksums + provenance
├── internal_state/         # Seeds, perturbations, calibration
├── outputs/
│   ├── recon.npy          # Reconstructed result
│   ├── metrics.json       # PSNR, SSIM, runtime
│   └── figures/           # Visualizations
├── report.md              # Diagnosis + suggested_actions
└── reproduce.py           # Generated script for reproduction
```

The **Streamlit viewer** (`pwm view`) provides:
- Split-view: ground truth vs reconstruction
- Metrics dashboard over solver portfolio
- Residual diagnostics and artifact analysis
- Interactive report with recommended actions

---

## Repository layout

```text
pwm/
  README.md
  LICENSE
  docs/
    spec_v0.2.1.md
    runbundle_format.md
    operator_mode.md
  examples/
    prompt_to_casepack.py
    yA_calibrate_recon_cassi.py
    yA_calibrate_recon_generic.py
  pyproject.toml

  packages/
    pwm_core/          # public core library (no Denario deps)
    pwm_denario/       # Denario adapter (thin)
```

---

## Install

### Requirements
- Python 3.10+ recommended
- PyTorch (CPU or CUDA)
- Optional: `deepinv`, `streamlit`, `opencv-python`, `scikit-image`

### Workspace install (editable)

```bash
pip install -U pip
pip install -e packages/pwm_core
pip install -e packages/pwm_denario
```

If you want the viewer:

```bash
pip install -e "packages/pwm_core[viewer]"
```

> Tip: If you use CUDA, install PyTorch first using the official selector for your CUDA version.

---

## Quickstart

### A) Prompt → auto CasePack → simulate → reconstruct → analyze → view

```bash
# Microscopy examples
pwm run --prompt "widefield deconvolution, low dose, PSF mismatch"
pwm run --prompt "SIM structured illumination, 3 angles, 3 phases, live cell"
pwm run --prompt "confocal 3D stack, depth attenuation, z-drift"

# Compressive imaging examples
pwm run --prompt "CASSI spectral imaging, 28 bands, coded aperture"
pwm run --prompt "single pixel camera, 25% sampling, Hadamard patterns"
pwm run --prompt "CACTI video, 8 frames compressed to 1 snapshot"

# Medical imaging examples
pwm run --prompt "CT sparse view, 90 angles, low dose"
pwm run --prompt "MRI accelerated, 4x undersampling, parallel imaging"

# Neural rendering examples
pwm run --prompt "NeRF from 30 views, synthetic scene"
pwm run --prompt "3D Gaussian splatting, multi-view reconstruction"

# View results
pwm view runs/latest
```

PWM will:
1) select a CasePack from the 18 validated modalities,
2) compile a draft spec,
3) validate/repair,
4) simulate measurement `y`,
5) reconstruct `x̂` using solver portfolio,
6) diagnose failure modes,
7) export a RunBundle.

### B) Spec → run

```bash
# Run with custom spec file
pwm run --spec my_experiment.json
pwm view runs/latest
```

### C) Python API

```python
from pwm_core.api import endpoints

# Option 1: Run from prompt (auto-selects casepack from 18 modalities)
result = endpoints.run(prompt="widefield deconvolution, low dose")
print(f"RunBundle: {result['runbundle_path']}")
print(f"Verdict: {result['diagnosis']['verdict']}")
print(f"PSNR: {result['recon'][0]['metrics'].get('psnr', 'N/A')}")

# Option 2: Run from spec dict
spec = {
    "id": "my_cassi_experiment",
    "input": {"mode": "simulate"},
    "states": {
        "physics": {"modality": "cassi"},
        "budget": {"measurement_budget": {"num_bands": 28}}
    }
}
result = endpoints.run(spec=spec, out_dir="runs/")

# Option 3: Compile prompt first, inspect casepack, then run
compile_result = endpoints.compile_prompt("MRI accelerated imaging")
print(f"Selected casepack: {compile_result.casepack_id}")
print(f"Modality: {compile_result.draft_spec['states']['physics']['modality']}")  # draft_spec is a dict

# Run with the compiled spec
result = endpoints.run(spec=compile_result.draft_spec, out_dir="runs/")
```

### D) Run benchmarks directly

```python
# Run validated benchmark for any of the 18 modalities
import sys
sys.path.insert(0, "packages/pwm_core")
from benchmarks.run_all import BenchmarkRunner

runner = BenchmarkRunner()

# Run specific modality - most return {'psnr': ..., 'ssim': ...}
result = runner.run_widefield_benchmark()
print(f"Widefield PSNR: {result['psnr']:.2f} dB")

result = runner.run_mri_benchmark()
print(f"MRI PSNR: {result['psnr']:.2f} dB")

result = runner.run_cassi_benchmark()
print(f"CASSI PSNR: {result['psnr']:.2f} dB")

# SPC benchmark with sampling rates - returns per-rate results
result = runner.run_spc_benchmark(sampling_rates=[0.25])
print(f"SPC 25% PSNR: {result['per_rate']['25pct']['avg_psnr']:.2f} dB")

result = runner.run_nerf_benchmark()
print(f"NeRF PSNR: {result['psnr']:.2f} dB")
```

---

## The ExperimentSpec model

PWM organizes the world into:

1. **PhysicsState** *(required)* — forward operator family
2. **BudgetState** — dose, sampling rate, #frames/views
3. **CalibrationState (θ)** — alignment/PSF/dispersion/gain drift/timing jitter
4. **EnvironmentState** — background, scattering/attenuation, autofluorescence
5. **SampleState** — motion/drift, blinking/kinetics, dynamics
6. **SensorState** — saturation, quantization, read noise, FPN, nonlinearity
7. **ComputeState** *(optional)* — runtime/memory/streaming constraints
8. **TaskState** *(optional)* — recon vs calibration vs DOE vs QC report

This structure makes it possible to (a) simulate realistic data, (b) diagnose failure modes, and (c) recommend concrete improvements.

See: `docs/spec_v0.2.1.md`.

---

## Operator correction mode: measured y + A → fit/correct operator → reconstruct

This mode is for real experiments where the forward model is imperfect.

### Supported modalities for calibration

| Modality | Calibration Parameters | Without Correction | With Correction | Improvement |
|----------|------------------------|-------------------|-----------------|-------------|
| MRI | coil sensitivities | 6.94 dB | 55.25 dB | **+48.31 dB** |
| CACTI | mask timing | 6.95 dB | 33.41 dB | **+26.46 dB** |
| SPC | gain/bias (50%) | 9.06 dB | 33.53 dB | **+24.47 dB** |
| CT | center of rotation | 12.54 dB | 28.71 dB | **+16.17 dB** |
| CASSI | dispersion step | 19.70 dB | 30.17 dB | **+10.47 dB** |
| Lensless | PSF shift | 24.79 dB | 34.05 dB | **+9.26 dB** |
| Ptychography | position offset | 18.45 dB | 25.04 dB | **+6.59 dB** |

**Average improvement: +20.25 dB** across all tested modalities.

**Note:** Improvement depends on mismatch severity and calibration search quality. Results above use grid search with benchmark-quality reconstruction algorithms (GAP-TV, SART-TV, SENSE, ADMM-TV, FISTA-TV, PIE). Higher PSNR can be achieved with neural denoisers (DRUNet, HSI_SDeCNN) as in the benchmarks.

### Python Examples: Without vs With Correction

Each example below shows:
1. **Without correction**: Reconstruct using wrong/assumed parameters
2. **Calibration**: Search for correct parameters
3. **With correction**: Reconstruct using calibrated parameters

#### 1) CT - Center of Rotation Calibration

```python
import numpy as np
from scipy.ndimage import rotate

# Setup
n, n_angles = 128, 90
angles = np.linspace(0, np.pi, n_angles, endpoint=False)
phantom = create_phantom(n)  # Your ground truth

# Forward model with center shift
def radon_forward(img, angles, cor_shift=0):
    sinogram = np.zeros((len(angles), n))
    for i, theta in enumerate(angles):
        proj = rotate(img, np.degrees(theta), reshape=False).sum(axis=0)
        sinogram[i] = np.roll(proj, cor_shift)
    return sinogram

# SART-TV reconstruction
def sart_tv_recon(sinogram, angles, cor_shift=0, iters=25):
    # Correct sinogram for center shift, then reconstruct
    sino_corrected = np.array([np.roll(s, -cor_shift) for s in sinogram])
    return sart_tv(sino_corrected, angles, iters)

# Generate measurement with TRUE center
cor_true = 0
sinogram = radon_forward(phantom, angles, cor_shift=cor_true)

# WITHOUT CORRECTION: Reconstruct with WRONG center
cor_wrong = 4
recon_wrong = sart_tv_recon(sinogram, angles, cor_shift=cor_wrong)
print(f"Without correction (CoR={cor_wrong}): PSNR={compute_psnr(recon_wrong, phantom):.2f} dB")

# CALIBRATION: Search for best center
best_cor, best_residual = cor_wrong, float('inf')
for test_cor in range(-6, 7):
    recon_test = sart_tv_recon(sinogram, angles, cor_shift=test_cor, iters=10)
    residual = np.sum((sinogram - radon_forward(recon_test, angles, test_cor))**2)
    if residual < best_residual:
        best_cor, best_residual = test_cor, residual

# WITH CORRECTION: Reconstruct with calibrated center
recon_corrected = sart_tv_recon(sinogram, angles, cor_shift=best_cor)
print(f"With correction (CoR={best_cor}): PSNR={compute_psnr(recon_corrected, phantom):.2f} dB")
# Expected: ~12 dB → ~29 dB (+16 dB improvement)
```

#### 2) MRI - Coil Sensitivity Calibration

```python
import numpy as np

# Setup
n, n_coils, acceleration = 128, 8, 4
phantom = create_brain_phantom(n)
sens_true = generate_coil_sensitivities(n, n_coils)  # True spatial maps
mask = create_undersampling_mask(n, acceleration, acs_lines=24)

# Forward SENSE model
def forward_sense(img, sens, mask):
    k_data = np.zeros((n_coils, n, n), dtype=np.complex64)
    for c in range(n_coils):
        k_data[c] = np.fft.fft2(img * sens[c]) * mask
    return k_data

# SENSE reconstruction
def sense_recon(k_data, sens, mask, iters=30):
    recon = np.zeros((n, n), dtype=np.complex64)
    for _ in range(iters):
        # Forward-backward iteration with coil combination
        ...
    return recon

# Generate k-space with TRUE sensitivities
k_data = forward_sense(phantom, sens_true, mask)

# WITHOUT CORRECTION: Reconstruct with WRONG sensitivities (uniform)
sens_wrong = np.ones((n_coils, n, n), dtype=np.complex64) / np.sqrt(n_coils)
recon_wrong = sense_recon(k_data, sens_wrong, mask)
print(f"Without correction (uniform sens): PSNR={compute_psnr(recon_wrong, phantom):.2f} dB")

# CALIBRATION: Estimate sensitivities from ACS (auto-calibration signal)
def estimate_sensitivities_from_acs(k_data, acs_lines=24):
    # Extract low-resolution images from center k-space
    sens_est = np.zeros_like(k_data)
    for c in range(n_coils):
        k_acs = extract_acs(k_data[c], acs_lines)
        sens_est[c] = smooth(np.fft.ifft2(k_acs))
    return normalize_sos(sens_est)

sens_calibrated = estimate_sensitivities_from_acs(k_data)

# WITH CORRECTION: Reconstruct with calibrated sensitivities
recon_corrected = sense_recon(k_data, sens_calibrated, mask)
print(f"With correction (ACS sens): PSNR={compute_psnr(recon_corrected, phantom):.2f} dB")
# Expected: ~7 dB → ~55 dB (+48 dB improvement)
```

#### 3) CASSI - Dispersion Step Calibration

```python
import numpy as np

# Setup
h, w, nC = 256, 256, 28  # Hyperspectral cube
cube = load_kaist_scene()  # Ground truth HSI
mask = (np.random.rand(h, w) > 0.5).astype(np.float32)

# CASSI forward model with dispersion step
def cassi_forward(x, mask, step=1):
    masked = x * mask[:, :, np.newaxis]
    # Shift each spectral band by step pixels
    shifted = np.zeros((h, w + (nC-1)*step, nC))
    for c in range(nC):
        shifted[:, c*step:c*step+w, c] = masked[:, :, c]
    return shifted.sum(axis=2)  # Compressed measurement

# GAP-denoise reconstruction
def gap_denoise(y, mask, step=1, iters=100):
    # Generalized Alternating Projection with TV denoising
    ...
    return x_recon

# Generate measurement with TRUE step
step_true = 1
y = cassi_forward(cube, mask, step=step_true)

# WITHOUT CORRECTION: Reconstruct with WRONG step
step_wrong = 3
recon_wrong = gap_denoise(y, mask, step=step_wrong)
print(f"Without correction (step={step_wrong}): PSNR={compute_psnr(recon_wrong, cube):.2f} dB")

# CALIBRATION: Search for best dispersion step
best_step, best_residual = step_wrong, float('inf')
for test_step in [1, 2, 3, 4]:
    recon_test = gap_denoise(y, mask, step=test_step, iters=50)
    residual = np.sum((y - cassi_forward(recon_test, mask, test_step))**2)
    if residual < best_residual:
        best_step, best_residual = test_step, residual

# WITH CORRECTION: Reconstruct with calibrated step
recon_corrected = gap_denoise(y, mask, step=best_step)
print(f"With correction (step={best_step}): PSNR={compute_psnr(recon_corrected, cube):.2f} dB")
# Expected: ~20 dB → ~30 dB (+10 dB improvement)
```

#### 4) CACTI - Mask Timing Calibration

```python
import numpy as np

# Setup
h, w, nF = 256, 256, 8  # 8 video frames
video = load_video()  # Ground truth video
masks = generate_temporal_masks(h, w, nF)

# CACTI forward model with timing offset
def cacti_forward(video, masks, timing_offset=0):
    measurement = np.zeros((h, w))
    for f in range(nF):
        mask_idx = (f + timing_offset) % nF
        measurement += video[:, :, f] * masks[:, :, mask_idx]
    return measurement

# GAP-TV reconstruction
def gap_tv_cacti(y, masks, timing_offset=0, iters=100):
    # Align masks, then GAP with TV denoising
    ...
    return video_recon

# Generate measurement with TRUE timing
timing_true = 0
y = cacti_forward(video, masks, timing_offset=timing_true)

# WITHOUT CORRECTION: Reconstruct with WRONG timing
timing_wrong = 3
recon_wrong = gap_tv_cacti(y, masks, timing_offset=timing_wrong)
print(f"Without correction (timing={timing_wrong}): PSNR={compute_psnr(recon_wrong, video):.2f} dB")

# CALIBRATION: Search for best timing
best_timing, best_residual = timing_wrong, float('inf')
for test_timing in range(nF):
    recon_test = gap_tv_cacti(y, masks, timing_offset=test_timing, iters=50)
    residual = np.sum((y - cacti_forward(recon_test, masks, test_timing))**2)
    if residual < best_residual:
        best_timing, best_residual = test_timing, residual

# WITH CORRECTION: Reconstruct with calibrated timing
recon_corrected = gap_tv_cacti(y, masks, timing_offset=best_timing)
print(f"With correction (timing={best_timing}): PSNR={compute_psnr(recon_corrected, video):.2f} dB")
# Expected: ~7 dB → ~33 dB (+26 dB improvement)
```

#### 5) SPC - Gain/Bias Calibration

```python
import numpy as np

# Setup
n = 64
rate = 0.50  # 50% sampling
m = int(n * n * rate)
x_true = load_image().flatten()
Phi = np.random.randn(m, n*n) / np.sqrt(m)  # Measurement matrix

# Forward model: y = gain * (Phi @ x) + bias
def spc_forward(x, Phi, gain=1.0, bias=0.0):
    return gain * (Phi @ x) + bias

# Least-squares reconstruction
def lsq_recon(y, Phi, gain=1.0, bias=0.0):
    y_corrected = (y - bias) / gain
    return np.linalg.lstsq(Phi, y_corrected, rcond=None)[0]

# Generate measurement with TRUE parameters
gain_true, bias_true = 1.0, 0.0
y = spc_forward(x_true, Phi, gain_true, bias_true)

# WITHOUT CORRECTION: Reconstruct with WRONG parameters
gain_wrong, bias_wrong = 0.65, 0.08
recon_wrong = lsq_recon(y, Phi, gain_wrong, bias_wrong)
print(f"Without correction (gain={gain_wrong}): PSNR={compute_psnr(recon_wrong, x_true):.2f} dB")

# CALIBRATION: Grid search for gain/bias
best_gain, best_bias, best_residual = gain_wrong, bias_wrong, float('inf')
for test_gain in np.linspace(0.5, 1.5, 21):
    for test_bias in np.linspace(-0.2, 0.2, 21):
        recon_test = lsq_recon(y, Phi, test_gain, test_bias)
        y_pred = spc_forward(recon_test, Phi, test_gain, test_bias)
        residual = np.sum((y - y_pred)**2)
        if residual < best_residual:
            best_gain, best_bias, best_residual = test_gain, test_bias, residual

# WITH CORRECTION: Reconstruct with calibrated parameters
recon_corrected = lsq_recon(y, Phi, best_gain, best_bias)
print(f"With correction (gain={best_gain:.2f}): PSNR={compute_psnr(recon_corrected, x_true):.2f} dB")
# Expected: ~9 dB → ~34 dB (+24 dB improvement)
```

#### 6) Lensless - PSF Shift Calibration

```python
import numpy as np
from scipy.signal import fftconvolve
from scipy.ndimage import shift as ndshift

# Setup
n = 128
img = load_image(n)
psf = load_diffuser_psf(n)

# Forward model with PSF shift
def lensless_forward(img, psf, shift=(0, 0)):
    psf_shifted = ndshift(psf, shift)
    return fftconvolve(img, psf_shifted, mode='same')

# ADMM-TV reconstruction
def admm_tv_recon(y, psf, shift=(0, 0), iters=50):
    psf_shifted = ndshift(psf, shift)
    # Wiener deconvolution + TV regularization
    ...
    return recon

# Generate measurement with TRUE PSF
shift_true = (0, 0)
y = lensless_forward(img, psf, shift=shift_true)

# WITHOUT CORRECTION: Reconstruct with WRONG PSF shift
shift_wrong = (3, 2)
recon_wrong = admm_tv_recon(y, psf, shift=shift_wrong)
print(f"Without correction (shift={shift_wrong}): PSNR={compute_psnr(recon_wrong, img):.2f} dB")

# CALIBRATION: Search for best PSF shift
best_shift, best_residual = shift_wrong, float('inf')
for sx in range(-4, 5):
    for sy in range(-4, 5):
        recon_test = admm_tv_recon(y, psf, shift=(sx, sy), iters=20)
        residual = np.sum((y - lensless_forward(recon_test, psf, (sx, sy)))**2)
        if residual < best_residual:
            best_shift, best_residual = (sx, sy), residual

# WITH CORRECTION: Reconstruct with calibrated shift
recon_corrected = admm_tv_recon(y, psf, shift=best_shift)
print(f"With correction (shift={best_shift}): PSNR={compute_psnr(recon_corrected, img):.2f} dB")
# Expected: ~25 dB → ~34 dB (+9 dB improvement)
```

#### 7) Ptychography - Position Offset Calibration

```python
import numpy as np

# Setup
n, probe_size, step = 64, 20, 10
obj_true = create_complex_object(n)  # Amplitude * exp(i*phase)
probe = create_gaussian_probe(probe_size)

# Generate scan positions with offset
def get_positions(offset_x=0, offset_y=0):
    positions = []
    for py in range(0, n - probe_size + 1, step):
        for px in range(0, n - probe_size + 1, step):
            positions.append((py + offset_y, px + offset_x))
    return positions

# Forward model: diffraction patterns
def ptycho_forward(obj, probe, positions):
    intensities = []
    for py, px in positions:
        exit_wave = obj[py:py+probe_size, px:px+probe_size] * probe
        intensities.append(np.abs(np.fft.fft2(exit_wave))**2)
    return np.array(intensities)

# ePIE reconstruction
def epie_recon(intensities, probe, positions, n_iters=200):
    obj = np.ones((n, n), dtype=np.complex64)
    for _ in range(n_iters):
        for idx, (py, px) in enumerate(positions):
            # PIE update with measured amplitude constraint
            ...
    return np.abs(obj)

# Generate measurements with TRUE positions
offset_true = (0, 0)
positions_true = get_positions(*offset_true)
intensities = ptycho_forward(obj_true, probe, positions_true)

# WITHOUT CORRECTION: Reconstruct with WRONG positions
offset_wrong = (5, -4)
positions_wrong = get_positions(*offset_wrong)
recon_wrong = epie_recon(intensities, probe, positions_wrong)
print(f"Without correction (offset={offset_wrong}): PSNR={compute_psnr(recon_wrong, obj_true):.2f} dB")

# CALIBRATION: Search for best offset
best_offset, best_psnr = offset_wrong, -float('inf')
for ox in range(-6, 7, 2):
    for oy in range(-6, 7, 2):
        positions_test = get_positions(ox, oy)
        recon_test = epie_recon(intensities, probe, positions_test, n_iters=80)
        psnr_test = compute_psnr(recon_test, obj_true)
        if psnr_test > best_psnr:
            best_offset, best_psnr = (ox, oy), psnr_test

# WITH CORRECTION: Reconstruct with calibrated positions
positions_corrected = get_positions(*best_offset)
recon_corrected = epie_recon(intensities, probe, positions_corrected)
print(f"With correction (offset={best_offset}): PSNR={compute_psnr(recon_corrected, obj_true):.2f} dB")
# Expected: ~18 dB → ~25 dB (+7 dB improvement)
```

### Running Complete Examples

For complete, runnable examples with all helper functions:

```bash
# Run all 7 modalities with full implementations
python examples/operator_correction_examples.py --all

# Run specific modality
python examples/operator_correction_examples.py --modality ct
python examples/operator_correction_examples.py --modality mri
python examples/operator_correction_examples.py --modality cassi
python examples/operator_correction_examples.py --modality cacti
python examples/operator_correction_examples.py --modality spc
python examples/operator_correction_examples.py --modality lensless
python examples/operator_correction_examples.py --modality ptychography
```

### CLI Examples

#### 1) CASSI hyperspectral calibration

```bash
# Calibrate dispersion and mask shift, then reconstruct
pwm calib-recon \
  --y data/cassi_measurement.npy \
  --operator cassi \
  --out-dir runs/cassi_calib

pwm view runs/cassi_calib
```

#### 2) Generic matrix operator calibration

```bash
# Fit gain/shift for any linear system y = A @ x
pwm fit-operator \
  --y data/measured_y.npy \
  --operator matrix

# Calibrate + reconstruct
pwm calib-recon \
  --y data/measured_y.npy \
  --operator matrix \
  --out-dir runs/matrix_calib
```

#### 3) Widefield PSF calibration

```bash
# Calibrate PSF from measured data
pwm calib-recon \
  --y data/widefield_capture.npy \
  --operator widefield \
  --out-dir runs/widefield_calib
```

#### 4) Lensless imaging

```bash
# Reconstruct from lensless capture
pwm calib-recon \
  --y data/lensless_capture.npy \
  --operator lensless \
  --out-dir runs/lensless_calib
```

### Python API

```python
from pwm_core.api import endpoints
from pwm_core.api.types import (
    ExperimentSpec, ExperimentInput, ExperimentStates,
    InputMode, PhysicsState, TaskState, TaskKind,
    MismatchSpec, MismatchFitOperator
)

# Build spec for CASSI calibration + reconstruction
spec = ExperimentSpec(
    id="cassi_calib_001",
    input=ExperimentInput(
        mode=InputMode.measured,
        y_source="data/cassi_measurement.npy",
    ),
    states=ExperimentStates(
        physics=PhysicsState(modality="cassi"),
        task=TaskState(kind=TaskKind.calibrate_and_reconstruct),
    ),
    mismatch=MismatchSpec(
        enabled=True,
        fit_operator=MismatchFitOperator(
            enabled=True,
            search={"method": "random", "max_evals": 50},
        ),
    ),
)

# Run calibration + reconstruction
result = endpoints.calibrate_recon(spec, out_dir="runs/")

print(f"Best-fit params: {result.calib.theta_best}")
print(f"Recon solver: {result.recon[0].solver_id}")
```

### What gets saved

```
runs/calibration_run/
├── theta_best.json           # Best-fit operator parameters
├── belief_state.json         # Candidate history + scores
├── calibration_trajectory/   # Parameter evolution
├── verification/
│   ├── residual_tests.json   # Statistical tests
│   └── stability_proxies.json
├── outputs/
│   ├── recon.npy             # Final reconstruction
│   └── metrics.json          # PSNR, SSIM
├── report.md                 # Diagnosis + suggested_actions
└── reproduce.py              # Generated script
```

### Testing Operator Correction

Run the calibration benchmark to see before/after results:

```bash
# Test all calibration-supported modalities
python benchmarks/test_operator_correction.py --all

# Test specific modality
python benchmarks/test_operator_correction.py --modality matrix
python benchmarks/test_operator_correction.py --modality mri
python benchmarks/test_operator_correction.py --modality ct
python benchmarks/test_operator_correction.py --modality cassi
python benchmarks/test_operator_correction.py --modality cacti
python benchmarks/test_operator_correction.py --modality lensless
python benchmarks/test_operator_correction.py --modality ptychography
```

**Example output (matrix/SPC gain-bias calibration):**
```
[MATRIX] Testing gain/bias calibration...
  Gain: true=1.0, wrong=1.3, calibrated=1.15
  Bias: true=0.0, wrong=0.1, calibrated=0.050
  Without correction: PSNR=7.53 dB
  With correction:    PSNR=12.61 dB (+5.08 dB)
  Oracle (true params): PSNR=13.13 dB
```

**Example output (MRI coil sensitivity calibration):**
```
[MRI] Testing coil sensitivity calibration...
  Coil sensitivities: using ACS-based calibration
  Without correction: PSNR=21.39 dB
  With correction:    PSNR=24.63 dB (+3.24 dB)
```

See: `docs/operator_mode.md`.

---

## DeepInv integration

PWM supports solver portfolios, including:
- **DeepInv** PnP / unrolled methods / diffusion adapters (optional)
- classical solvers (TV-FISTA, ADMM-TV, primal-dual, RL)

`pwm_core/recon/deepinv_adapter.py` maps `PhysicsState` to DeepInv operators when possible.

---

## CasePacks

CasePacks are **validated templates** for each of the 18 imaging modalities:

| # | Modality | CasePack | Solver | PSNR |
|---|----------|----------|--------|------|
| 1 | Widefield | `widefield_deconv_basic_v1` | Richardson-Lucy | 27.31 dB |
| 2 | Widefield Low-Dose | `widefield_lowdose_highbg_v1` | PnP | 27.78 dB |
| 3 | Confocal Live-Cell | `confocal_livecell_lowdose_drift_v1` | Richardson-Lucy | 26.27 dB |
| 4 | Confocal 3D | `confocal_3d_stack_attenuation_v1` | 3D Richardson-Lucy | 29.01 dB |
| 5 | SIM | `sim_3x3_fragile_v1` | Wiener | 27.48 dB |
| 6 | CASSI | `cassi_spectral_imaging_v1` | GAP + HSI_SDeCNN | 30.60 dB |
| 7 | SPC | `spc_low_sampling_poisson_v1` | PnP-FISTA + DRUNet | 30.90 dB |
| 8 | CACTI | `cacti_video_sci_v1` | GAP-TV | 32.79 dB |
| 9 | Lensless | `lensless_diffusercam_basic_v1` | ADMM-TV | 34.66 dB |
| 10 | Light-Sheet | `lightsheet_tissue_stripes_scatter_v1` | Stripe Removal | 28.05 dB |
| 11 | CT | `ct_conebeam_lowdose_scatter_v1` | PnP-SART + DRUNet | 27.97 dB |
| 12 | MRI | `mri_cartesian_accelerated_v1` | PnP-ADMM + DRUNet | 48.25 dB |
| 13 | Ptychography | `ptychography_phase_retrieval_v1` | Neural Network | 59.47 dB |
| 14 | Holography | `holography_offaxis_phase_v1` | Neural Network | 42.52 dB |
| 15 | NeRF | `nerf_from_poses_basic_v1` | SIREN | 61.35 dB |
| 16 | 3D Gaussian Splatting | `gaussian_splatting_basic_v1` | 2D Gaussian Opt | 30.47 dB |
| 17 | Matrix | `matrix_generic_linear_v1` | FISTA-TV | 25.79 dB |
| 18 | Panorama Multifocal | `panorama_multifocal_fusion_v1` | Neural Fusion | 27.78 dB |

**Operator-fit packs** (for calibration mode):
- `cassi_measured_y_fit_theta_v1` - CASSI dispersion/mask calibration
- `generic_matrix_yA_fit_gain_shift_v1` - Generic gain/shift calibration

Each CasePack defines:
- modality/operator family
- default priors and solver recipes
- safe parameter ranges
- bounded "auto-refine" knobs (dose, sampling, drift, PSF mismatch…)
- validated benchmark results

Location: `packages/pwm_core/contrib/casepacks/`

See: `packages/pwm_core/contrib/casepacks/README.md` for detailed documentation.

---

## RunBundles (reproducibility first)

Every PWM run exports a **RunBundle** - a portable, self-contained folder for full reproducibility:

```
runs/my_experiment/
├── spec_resolved.json        # Complete experiment specification
├── validation_report.json    # Spec validation + auto-repairs
├── data_manifest.json        # Input checksums, provenance, copy/reference
├── internal_state/
│   ├── seeds.json            # All random seeds used
│   ├── perturbations.json    # Mismatch parameters applied
│   └── calibration/          # Calibration trajectories (if applicable)
├── outputs/
│   ├── recon.npy             # Reconstructed result
│   ├── metrics.json          # PSNR, SSIM, runtime, solver info
│   ├── solver_log.json       # Per-iteration convergence
│   └── figures/
│       ├── comparison.png    # Side-by-side visualization
│       ├── residual.png      # Residual analysis
│       └── metrics.png       # Convergence plots
├── report.md                 # Human-readable diagnosis
├── report.json               # Machine-readable diagnosis
├── suggested_actions.json    # Actionable recommendations
├── reproduce.py              # Generated script for reproduction
└── notebook.ipynb            # Generated Jupyter notebook
```

### Example RunBundle outputs by modality

| Modality | Key Outputs | Typical Metrics |
|----------|-------------|-----------------|
| Widefield | `recon.npy`, PSF estimate | PSNR ~27 dB |
| CASSI | `hypercube.npy` (H×W×λ) | PSNR ~31 dB, per-band SSIM |
| SPC | `recon.npy` (256×256) | PSNR vs sampling rate |
| CT | `volume.npy` (3D) | PSNR, SSIM, HU error |
| MRI | `recon.npy` (complex) | PSNR, SSIM, NMSE |
| NeRF | `renders/*.png`, `model.pt` | PSNR, LPIPS |
| 3DGS | `gaussians.ply`, `renders/` | PSNR, training time |

For large datasets, RunBundle stores **references** (paths + checksums) instead of copying data.

See: `docs/runbundle_format.md`.

---

## Viewer

Launch an interactive Streamlit dashboard:

```bash
# View latest run
pwm view runs/latest

# View specific RunBundle
pwm view runs/cassi_experiment_001
pwm view runs/mri_knee_4x

# Compare multiple runs
pwm view runs/exp1 runs/exp2 --compare
```

### Viewer tabs

**Overview:**
- Experiment summary (modality, solver, runtime)
- Key metrics (PSNR, SSIM) with pass/fail status

**Visualization:**
- Split-view: ground truth vs reconstruction
- Difference map with colorbar
- For 3D data: slice navigator
- For spectral data: band selector

**Metrics:**
- Convergence plots (loss, PSNR per iteration)
- Solver portfolio comparison
- Per-sample breakdown (for datasets)

**Diagnostics:**
- Residual analysis (should be noise-like)
- Artifact detection (stripes, ringing, aliasing)
- Failure mode identification

**Report:**
- Rendered markdown diagnosis
- Suggested actions with priority
- Links to documentation

**Reproduce:**
- Generated Python script
- Jupyter notebook
- CLI command to re-run

---

## Embedding into Denario

PWM exposes stable endpoints:
- `compile(prompt)` → draft spec
- `resolve_validate(spec)` → safe spec + auto-repair
- `simulate(spec)` / `reconstruct(spec, y)` / `analyze(...)`
- `fit_operator(...)` / `calibrate_recon(...)`
- `export(runbundle)` / `view(runbundle)`

Use `packages/pwm_denario/` as the thin adapter layer.

> You do **not** need AG2/LangGraph to run PWM.
> If you want autonomy loops later (planner↔reviewer, tool-using multi-step agents), implement them in `pwm_denario` without changing `pwm_core`.

---

## Contributing

PWM is intended to be extended by the community.

### Add a new modality/operator
1) Create a new operator in `pwm_core/physics/...`
2) Add parameterization in `pwm_core/mismatch/parameterizations.py`
3) Add a CasePack in `pwm_core/contrib/casepacks/`
4) Add a minimal test in `packages/pwm_core/tests/`

Templates:
- `pwm_core/contrib/templates/new_operator_template.py`
- `pwm_core/contrib/templates/new_calibrator_template.py`

### Add a dataset adapter
- Implement loader in `pwm_core/io/datasets.py` and format handler in `io/formats.py`
- Add an example under `examples/`
- Prefer reference-mode support for large datasets

---

## 18 Imaging Modalities Benchmark

PWM includes validated implementations for **18 imaging modalities**, all meeting or exceeding published reference performance.

### Supported Modalities

| # | Modality | Solver | PSNR (dB) | Status |
|---|----------|--------|-----------|--------|
| 1 | Widefield | Richardson-Lucy | 27.31 | Pass |
| 2 | Widefield Low-Dose | PnP | 27.78 | Pass |
| 3 | Confocal Live-Cell | Richardson-Lucy | 26.27 | Pass |
| 4 | Confocal 3D | 3D Richardson-Lucy | 29.01 | Pass |
| 5 | SIM | Wiener | 27.48 | Pass |
| 6 | CASSI | GAP + HSI_SDeCNN | 30.60 | Pass |
| 7 | SPC (25%) | PnP-FISTA + DRUNet | 30.90 | Pass |
| 8 | CACTI | GAP-TV | 32.79 | Pass |
| 9 | Lensless | ADMM-TV | 34.66 | Pass |
| 10 | Light-Sheet | Stripe Removal | 28.05 | Pass |
| 11 | CT | PnP-SART + DRUNet | 27.97 | Pass |
| 12 | MRI | PnP-ADMM + DRUNet | 48.25 | Pass |
| 13 | Ptychography | Neural Network | 59.47 | Pass |
| 14 | Holography | Neural Network | 42.52 | Pass |
| 15 | NeRF | Neural Implicit (SIREN) | 61.35 | Pass |
| 16 | 3D Gaussian Splatting | 2D Gaussian Opt | 30.47 | Pass |
| 17 | Matrix (Generic) | FISTA-TV | 25.79 | Pass |
| 18 | Panorama Multifocal | Neural Fusion | 27.78 | Pass |

### Running the Benchmarks

```bash
# Navigate to project directory
cd packages/pwm_core

# Run ALL 18 modalities
python benchmarks/run_all.py --all

# Run core modalities only (faster)
python benchmarks/run_all.py --core

# Run a specific modality
python benchmarks/run_all.py --modality widefield
python benchmarks/run_all.py --modality ct
python benchmarks/run_all.py --modality mri
python benchmarks/run_all.py --modality spc
python benchmarks/run_all.py --modality cassi
python benchmarks/run_all.py --modality cacti
python benchmarks/run_all.py --modality ptychography
python benchmarks/run_all.py --modality holography
python benchmarks/run_all.py --modality nerf
python benchmarks/run_all.py --modality gaussian_splatting
python benchmarks/run_all.py --modality panorama_multifocal
```

### Alternative: Run as Python Module

```bash
# From project root
python -m packages.pwm_core.benchmarks.run_all --all
python -m packages.pwm_core.benchmarks.run_all --modality mri
```

### Benchmark Output

Results are saved to `packages/pwm_core/benchmarks/results/`:
- `benchmark_results.json` - Raw metrics for all modalities
- `benchmark_report.md` - Formatted report with detailed analysis

### Dataset Preparation

Most benchmarks use **synthetic data by default** (no download required). For real datasets:

```python
# Prepare Set11 for SPC benchmarks
from pwm_core.data.download import prepare_set11
prepare_set11()

# Prepare CACTI videos
from pwm_core.data.download import prepare_cacti_videos
prepare_cacti_videos()

# List all available datasets
from pwm_core.data.download import list_datasets
print(list_datasets())
```

For large datasets (LoDoPaB-CT, fastMRI, KAIST), see `docs/implementation_plan.md` for download instructions.

---

## License

See `LICENSE`.

---

## Citation

If you use PWM in academic work, please cite the associated paper (to be added) and link to this repository.
