# Comprehensive 6-Point Check — Electrical Impedance Tomography (EIT)

**URL:** https://pwm.platformai.org/benchmark/impedance_tomo
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Electrical Impedance Tomography (EIT)

**Physical principle:** EIT reconstructs the internal conductivity (and permittivity) distribution σ(x) of a body by injecting electrical currents through surface electrodes and measuring the resulting boundary voltages. The governing physics is the Calderón problem: given the Dirichlet-to-Neumann (voltage-to-current) map on the boundary, recover σ(x) in the interior. The problem is severely ill-posed (small conductivity changes cause tiny voltage perturbations at the boundary), limiting spatial resolution to roughly 10–15% of the body diameter. EIT is used in lung ventilation monitoring, breast cancer screening, and industrial process tomography.

**Forward model:**
```
∇ · (σ(x) ∇ u(x)) = 0      in Ω
u|_∂Ω = V_pattern             (applied voltage pattern)
I_measured = ∫_electrode σ ∂u/∂n dS  (measured current)

Or in matrix form:
  V_meas = M · σ + η    (linearized, ΔV = J · Δσ)

where:
  σ(x)        — unknown conductivity distribution [S/m]
  u(x)        — electric potential field inside domain Ω
  J           — Jacobian (sensitivity) matrix (∂V_meas/∂σ)
  V_meas      — boundary voltage measurements (N_electrodes × N_patterns)
  η           — measurement noise (~0.1–1% of signal)
```

**Inverse problem:** Recover the conductivity map σ(x) from boundary voltage measurements V_meas; the inverse is notoriously ill-conditioned (condition number ~10⁶), requiring strong regularization.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(sinusoidal current injection) → F(conducting body) → D(electrode array)

**Key mismatch parameters:**
- `electrode_contact_impedance`: skin-electrode contact resistance; nominal 200 Ω, perturbed 1000 Ω (poor contact)
- `measurement_noise`: voltage noise floor; nominal 0.1% SNR, perturbed 1.0% SNR
- `conductivity_contrast`: ratio of anomaly to background conductivity; nominal 2:1, perturbed 10:1 (high contrast, nonlinearity)
- `electrode_count`: number of electrodes; nominal 32, perturbed 16 (under-determined system)

**Dataset format:**
- `x_true: (H, W)` — ground-truth 2D conductivity map σ(x,y) [S/m]
- `y: (N_meas,)` — boundary voltage difference measurements (vectorized)

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| NOSER (Newton one-step error reconstructor) | Classical | Cheney et al., Int. J. Imaging Syst. Technol. 2:66 (1990) | Tikhonov-regularized one-step Newton inversion; still widely used clinical EIT |
| TV-Gauss-Newton | Classical | Borsic et al., Physiol. Meas. 30:S1 (2009) | Total variation regularization via iterative Gauss-Newton with edge-preserving properties |
| D-bar method | Classical (exact) | Knudsen et al., Physiol. Meas. 28:S101 (2007) | Mathematically exact reconstruction via ∂-bar equations for Calderón's problem |
| EIT-CNN | Deep Learning | Hamilton & Hauptmann, IEEE Trans. Med. Imaging 37:2367 (2018) | First deep learning EIT reconstruction achieving real-time performance |
| EIT-Transformer | Transformer | Chen et al., IEEE Trans. Instrum. Meas. 72:1 (2023) | Transformer-based end-to-end EIT inversion with attention over measurement patterns |

---

## 4. Literature & State of the Art (2024–2025)

1. **Agnelli et al. (2024)** "Neural networks for the approximation of Euler's elastica in EIT reconstructions," *Inverse Probl.* — physics-constrained network learning Euler elastica regularizer for topologically faithful conductivity maps.
2. **Liu et al. (2024)** "Res-EIT: Residual learning for electrical impedance tomography with limited measurements," *IEEE Sensors J.* — residual network achieving SSIM >0.85 on 16-electrode EIT from simulated and real data.
3. **Harrach (2023)** "Uniqueness and stable determination in EIT with finitely many electrodes," *Inverse Probl.* — theoretical analysis establishing reconstruction stability bounds for the complete electrode model.
4. **Wei et al. (2024)** "Deep unrolling for EIT with physics-constrained iterations," *Med. Phys.* — unrolled iterative network enforcing finite-element forward model consistency at each layer.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/impedance_tomo_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/impedance_tomo_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/impedance_tomo_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/impedance_tomo/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

EIT is correctly modeled as the Calderón problem with a linearized Jacobian-based forward model and severely ill-conditioned inversion, and the algorithm routing spans classical NOSER/TV-Gauss-Newton/D-bar methods through deep learning and transformer-based approaches. The mismatch parameters — electrode contact impedance, measurement noise, conductivity contrast, and electrode count — accurately capture the dominant sources of degradation in clinical and industrial EIT deployments. The benchmark structure reflects the current state of EIT research balancing mathematical rigor with practical deep learning reconstruction methods.

---
*Comprehensive 6-point check by deep-check pipeline v3*

---

## GPU Server Algorithm Test Results

**Test Date:** 2026-03-11T05:45:34
**Test Tier:** public (sample_00)
**GPU:** NVIDIA GeForce GTX 1660 Ti, CUDA 12.4, PyTorch 2.6.0

| Solver | PSNR (dB) | SSIM | Time (s) | Status |
|--------|-----------|------|----------|--------|
| precomputed_baseline | 11.20 | 0.3124 | 0.00 | PASS |

*Tested by GPU server algorithm pipeline v1 (test_all_algorithms.py)*
