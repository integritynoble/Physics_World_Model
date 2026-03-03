# Benchmark QA Check — CT (Computed Tomography)

**URL:** https://pwm.platformai.org/benchmark/ct
**Local dataset:** `datasets/benchmark/ct/`
**Check Date:** 2026-03-03 (comprehensive 6-point review)

---

## 1. Benchmark Page Errors

### HIGH

- **H1. PSNR_norm undefined in scoring formula.** `0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × consistency` — normalization bounds not specified. Fix: define PSNR_norm = (PSNR - min) / (max - min) with explicit bounds.
- **H2. Forward operator notation inconsistent.** Uses R (Radon), Π (projection), Ĥ (consistency metric) for the same operator. Fix: unify notation.
- **H3. Beam hardening beta range on website differs from local spec.json.** Website shows negative β values for Public tier; local spec.json shows `beam_hardening_beta: [0.0, 0.1]` (non-negative). Fix: reconcile website with actual data.

### MEDIUM

- **M1. Leaderboard rank reversal unexplained.** DuDoTrans leads Public but drops to #4 on Hidden. No robustness discussion.
- **M2. Gallery algorithm comparison has no per-algorithm PSNR/SSIM numbers** — only images.
- **M3. Oracle scenario (III) shows PSNR *lower* than Mismatch (II)** — contradicts expected improvement.
- **M4. Missing DOIs** on all reference citations.
- **M5. Scenario IV (Blind Calibration) mentioned but never defined.**

---

## 2. Local Dataset Inspection

### Public tier: `ct_challenge_public.h5`
- **11 samples**, shape x_true=(362,362), sinogram=(60,736), float32 ✓
- **Source: LoDoPaB-CT test split** (Leuschner et al. 2021, *Scientific Data* 8:109) ✓
- Spec ranges: center_offset [-2,2]px, angle_error [-3,3]°, beam_hardening [0,0.1], detector_tilt [-1,1]° ✓
- True spec values embedded as HDF5 attributes ✓
- All metadata present (shape, n_views, n_det, D_so, D_sd, source citation) ✓

### Dev tier: `ct_challenge_dev.h5`
- **20 samples**, same shapes and dtypes ✓
- Source: LoDoPaB-CT **validation** split, patients 0-63 ✓
- **Diversity augmentation (rotation/flip/zoom)** applied to prevent matching to public tier ✓
- Wider spec ranges than public ✓

### Hidden tier: `ct_challenge_hidden.h5`
- **20 samples**, same shapes and dtypes ✓
- Source: LoDoPaB-CT validation split, patients 64-127 + adversarial modifications ✓
- **Adversarial mods:** metal inserts, low-contrast lesions, calcifications, high-contrast bone ✓
- Widest spec ranges: center_offset [-5,5]px, angle_error [-8,8]°, beam_hardening [0,0.3], detector_tilt [-3,3]° ✓
- Proper difficulty progression: Public ⊂ Dev ⊂ Hidden spec ranges ✓

### Dataset Quality Verdict: GOOD
- Real clinical data from established source ✓
- Patient separation across tiers ✓
- Augmentation for dev/hidden ✓
- Adversarial mods for hidden ✓

---

## 3. Public Dataset Acceptance Assessment

**Source: LoDoPaB-CT (Leuschner et al. 2021)**
- Published in **Scientific Data** (Nature portfolio) — top-tier data journal ✓
- Based on **LIDC/IDRI** (Lung Imaging Database Consortium) — the most widely-used public CT dataset in the field ✓
- **Zenodo record 3384092**, CC BY 4.0 ✓
- **Cited 200+ times** — widely accepted by CT reconstruction community ✓
- Used in AAPM benchmarks, Helsinki Tomography Challenge ✓

**Verdict: EXCELLENT** — LoDoPaB-CT is the gold standard for 2D sparse-view CT benchmarking.

### Suggestions for improvement:
- Consider adding a few samples from **2DeteCT** (Keulen et al., 2024) — emerging new benchmark with real experimental CT data (not simulated from clinical CTs)
- Consider **Mayo Clinic Low-Dose CT dataset** (AAPM Grand Challenge) for clinical relevance with matched low/normal dose pairs

---

## 4. Algorithm Coverage Assessment

### Currently tested (8 methods on leaderboard):
| # | Method | Year | Type | Status |
|---|--------|------|------|--------|
| 1 | FBP | classical | Analytical baseline | ✓ Good baseline |
| 2 | TV-ADMM | 2008 | Iterative | ✓ Classical CS |
| 3 | FBPConvNet | 2017 | Post-processing DL | ✓ |
| 4 | RED-CNN | 2017 | Post-processing DL | ✓ |
| 5 | PnP-ADMM | 2013 | Plug-and-Play | ✓ |
| 6 | Learned Primal-Dual | 2018 | Unrolled iterative | ✓ |
| 7 | DuDoTrans | 2022 | Dual-domain Transformer | ✓ Recent |
| 8 | DOLCE | 2023 | Diffusion-based | ✓ Recent |

### Missing important algorithms (should add):
| # | Method | Year | Why important |
|---|--------|------|---------------|
| 1 | **DRONE** | 2024 | Dual-domain residual optimization — state-of-art sparse-view |
| 2 | **CLEAR** | 2024 | Adversarial reconstruction for low-dose CT — NeurIPS |
| 3 | **DiffusionMBIR** | 2023 | Diffusion model + model-based iterative reconstruction |
| 4 | **WGAN-VGG** | 2018 | GAN-based CT denoising — highly cited (1000+) |
| 5 | **iCT-Net** | 2023 | Implicit neural representation for CT |
| 6 | **DPMA** (Dual-domain Prior Multi-scale Attention) | 2025 | Latest dual-domain approach from Scientific Reports |
| 7 | **SIREN/INR-based** | 2024 | Implicit neural representation methods |

### Solver Registry Gap:
- `solver_registry.yaml` only lists: FBP, PnP-ADMM, RED-CNN
- Missing from registry: DOLCE, DuDoTrans, Learned Primal-Dual, TV-ADMM, FBPConvNet (all on leaderboard but not in YAML)
**Fix:** Add all leaderboard methods to solver_registry.yaml with proper module paths.

---

## 5. Improvement Suggestions

### Dataset Improvements:
1. **Increase public tier to 20+ samples** — 11 samples is too few for reliable statistics
2. **Add 3D cone-beam challenge variant** — current benchmark is 2D fan-beam only; real clinical CT is 3D
3. **Add dose-level variation** — current benchmark has fixed I₀=10000; add multi-dose levels (I₀=1000, 5000, 50000)
4. **Add time-resolved (4D-CT) challenge** — breathing motion, cardiac gating
5. **Include 2DeteCT experimental data** (real measured, not simulated from clinical) for validation tier

### Algorithm Testing Improvements:
6. **Add 2024-2025 methods**: DRONE, CLEAR, DPMA, DiffusionMBIR
7. **Add implicit neural representation methods**: SIREN, NeRF-inspired CT
8. **Add GAN-based methods**: WGAN-VGG (highly cited baseline)
9. **Add self-supervised methods**: Noise2Inverse, equivariant imaging
10. **Register all leaderboard methods in solver_registry.yaml**

### Benchmark Design Improvements:
11. **Define PSNR_norm explicitly** in scoring formula
12. **Add per-algorithm per-mismatch-knob breakdown** — show which mismatch (center offset vs beam hardening) hurts which algorithm most
13. **Add confidence intervals** — 11-20 samples per tier needs bootstrap CIs
14. **Add clinical task metrics** — detection sensitivity (lesion-level), not just pixel-level PSNR/SSIM
15. **Clarify Scenario IV (Blind Calibration)** methodology — currently undefined on page

---

## 6. Status & Action Items

| Item | Status | Priority |
|------|--------|----------|
| Public dataset source quality | ✅ Excellent (LoDoPaB-CT) | — |
| Dev/hidden augmentation | ✅ Good (rotation/flip/zoom + adversarial) | — |
| Patient separation across tiers | ✅ Good | — |
| HDF5 schema documented | ✅ Good (README has loading code) | — |
| PSNR_norm undefined on website | ❌ Fix needed | HIGH |
| Notation inconsistency (R/Π/Ĥ) | ❌ Fix needed | HIGH |
| Website spec ranges vs local spec.json mismatch | ❌ Verify | HIGH |
| Add 7+ missing algorithms | ⚠️ Suggested | MEDIUM |
| Increase public tier sample count | ⚠️ Suggested | MEDIUM |
| Add solver_registry entries | ❌ Fix needed | MEDIUM |
| Add confidence intervals | ⚠️ Suggested | LOW |
| Add clinical task metrics | ⚠️ Suggested | LOW |

---

*Comprehensive 6-point review on 2026-03-03. Covers: benchmark page errors, local dataset inspection, public dataset acceptance, algorithm coverage, and improvement suggestions.*
