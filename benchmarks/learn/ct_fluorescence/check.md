# Comprehensive Benchmark QA Check -- CT + Fluorescence (FLIT)

**URL:** https://pwm.platformai.org/benchmark/ct_fluorescence
**Check Date:** 2026-03-03 (comprehensive 6-point review)

---

## 1. Benchmark Page Errors

### Summary

| Severity | Count |
|----------|-------|
| HIGH     | 6     |
| MEDIUM   | 4     |
| LOW      | 3     |

### HIGH Severity

**H1. Forward model misclassified as `microscopy_psf` -- fundamentally wrong physics**
CT + Fluorescence (FLIT) fuses two distinct imaging modalities:
- **CT pathway**: X-ray projection and filtered back-projection (Radon transform)
- **Fluorescence pathway**: Diffuse optical tomography governed by the diffusion equation or radiative transfer equation (RTE)

The config sets `category_module: microscopy_psf` and models the forward
operator as a 2D PSF convolution `I(d) = I0 * exp(-integral mu dl) + noise`.
This is an attenuation-only CT model and completely ignores the fluorescence
diffusion physics. The actual FLIT forward model requires:
- Beer-Lambert attenuation for CT
- Coupled diffusion equations for excitation and emission light propagation
- Fluorophore yield mapping
- CT-derived tissue optical property assignment

**Fix:** Implement a dedicated `ct_fluorescence` or `flit_fusion` category
module that couples Radon-based CT reconstruction with diffusion-equation-based
fluorescence molecular tomography (FMT).

**H2. DAG spec notation oversimplifies the coupled physics**
The DAG `Pi --> D (CT) + M --> R,P --> D (FLI) --> Fusion` and the notation
`(Pi -> D) + (M -> R -> P -> D) -> (oplus)` treat CT and fluorescence as
independent parallel channels with a late fusion step. In reality:
- CT provides anatomical structure AND optical property maps (absorption,
  scattering coefficients) that parametrize the fluorescence forward model
- The fluorescence forward model depends on CT output -- this is a serial
  dependency, not parallel
- The fusion is not a simple concatenation/addition; CT-derived tissue maps
  feed into the fluorescence diffusion solver

**Fix:** Revise DAG to show the serial CT->optical-property-mapping->FMT dependency.

**H3. PSNR_norm undefined in scoring formula**
The composite score `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - Hx||/||y||)`
uses PSNR_norm but never defines:
- Normalization bounds (min/max PSNR for mapping to [0,1])
- Whether normalization is per-scene or global
- A worked numerical example

Without this, scores are non-reproducible by external teams.

**Fix:** Define PSNR_norm = (PSNR - PSNR_min) / (PSNR_max - PSNR_min) with
explicit bounds (e.g., PSNR_min = 10 dB, PSNR_max = 50 dB) and provide a
worked calculation for one baseline method.

**H4. Mismatch parameter ranges are inconsistent between config and live page**
- Config YAML: optical_property_assignment_error [0.0, 30.0], autofluorescence
  [0.0, 50.0], registration [0.0, 3.0] -- all non-negative
- Live page: optical_property_assignment_error [-6.0, 12.0], autofluorescence
  [-10.0, 20.0], registration [-0.6, 1.2] -- signed ranges, different magnitudes

The config and the live page present fundamentally different parameter spaces.
Additionally:
- Negative autofluorescence is physically meaningless (autofluorescence is
  always a positive additive signal)
- The config ranges [0, 50] and page ranges [-10, 20] differ by a factor of 2.5x
  in span

**Fix:** Reconcile YAML config with live page. Clamp autofluorescence to [0, max].
Cite tissue autofluorescence literature for realistic ranges (e.g., Monici 2005,
Billinton & Knight 2001).

**H5. Noise model is generic -- missing FLIT-specific noise sources**
The learning materials list "Poisson (photon counting), speckle (coherent),
multiplicative" as noise sources. FLIT-specific noise includes:
- **CT**: Poisson photon statistics (correct) + electronic readout noise + beam
  hardening artifacts (polychromatic spectrum)
- **Fluorescence**: Autofluorescence background (dominant noise in vivo),
  tissue scattering-induced signal loss, excitation light leakage through
  emission filters, detector dark current
- **Cross-modal**: Registration misalignment noise, optical property
  estimation errors propagated from CT to fluorescence forward model

The generic noise model does not capture the dominant autofluorescence
contamination or the coupled CT-to-fluorescence error propagation.

**Fix:** Implement separate noise models for each pathway with explicit
autofluorescence background level and excitation leakage parameters.

**H6. Signal equation is CT-only -- fluorescence equation missing**
The signal equation `I(d) = I0 * exp(-integral mu(l) dl) + noise` is the
Beer-Lambert law for X-ray attenuation. There is no fluorescence signal
equation. The fluorescence forward model should include:
- Excitation light propagation: `-nabla . D_ex nabla Phi_ex + mu_a Phi_ex = S_ex`
- Fluorophore emission: `Q(r) = eta * mu_af * Phi_ex(r)`
- Emission light propagation: `-nabla . D_em nabla Phi_em + mu_a Phi_em = Q`
- Surface measurement: `m(r_d) = -D_em nabla Phi_em . n`

**Fix:** Add coupled diffusion equations for the fluorescence pathway alongside
the CT attenuation equation.

### MEDIUM Severity

**M1. Only 3 scenes per tier -- insufficient for statistical significance**
Three scenes per tier (9 total across public/dev/hidden) cannot produce
meaningful confidence intervals. Imaging benchmarks typically use 50-200+ test
cases. The expanded config defines B1=12, B2/B3/B4=60, grand_total=192 cases,
but the live challenge uses only 3 per tier.

**Fix:** Increase to at least 20 scenes per tier or implement bootstrapped
confidence intervals on the 3-scene scores.

**M2. 2D data format for an inherently multi-dimensional problem**
Config shows x_shape [64, 64] and y_shape [64, 64]. Real FLIT involves:
- CT: 3D volumetric reconstruction from 2D projections
- Fluorescence: 3D source distribution from surface measurements
The 2D-to-2D formulation trivializes the dimensionality mismatch that makes
FLIT reconstruction challenging.

**Fix:** At minimum, use 3D volumes (e.g., 64x64x64 or 128x128x32) for object
space and appropriate measurement geometries for each modality.

**M3. Leaderboard methods vs. config solvers are disjoint**
- **Leaderboard:** Cross-Modal Xformer (0.699), PnP-ADMM Joint (0.656),
  FDot-Net (0.644), Born/Rytov+FBP (0.581)
- **Config solvers:** Adjoint (traditional_cpu), PnP-ADMM (best_quality)

The leaderboard uses modality-specific methods (FDot-Net, Cross-Modal Xformer)
that have no corresponding solver entries in the config. The config's Adjoint
solver is a generic method unrelated to any leaderboard entry.

**Fix:** Add solver config entries for all leaderboard algorithms and ensure
reproducibility from the config.

**M4. `shepp_logan` synthetic generator -- wrong domain**
The fallback synthetic generator is `shepp_logan`, a CT-only phantom that
contains no fluorescence information (fluorophore distributions, tissue optical
properties). FLIT requires dual-contrast phantoms with both X-ray attenuation
maps and fluorophore concentration maps.

**Fix:** Create a FLIT-specific phantom generator with paired CT attenuation +
fluorophore distribution maps (e.g., digital mouse phantom with embedded
fluorescent inclusions).

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | HDF5 schema undocumented -- no key names, array shapes, dtypes, or compression details |
| L2 | References incomplete -- "Cross-Modal Xformer, 2024" has no authors, venue, or DOI; "Gao et al., BOE 2021" needs full citation |
| L3 | Wavelength/energy range listed as "0 -- 0 nm" in physics fundamentals -- should specify X-ray keV range and fluorescence excitation/emission wavelengths |

---

## 2. Local Dataset Inspection

### File Inventory

**NO LOCAL DATASET FILES** -- directory `datasets/benchmark/ct_fluorescence/` does not exist.

```
$ ls datasets/benchmark/ct_fluorescence 2>/dev/null
(directory does not exist -- exit code 2)
```

### Gallery Images (present)

Gallery images exist at:
```
platform/pwm_platform/static/img/benchmark_gallery/ct_fluorescence/scene_0{0..3}/
```

Each scene directory contains:
- `gt.png` -- ground truth
- `measurement_I.png`, `measurement_II.png` -- measurement views
- `recon_I.png`, `recon_II.png`, `recon_III.png` -- reconstruction results

Total: 4 scenes (scene_00 through scene_03), 6 images each = 24 gallery images.

### Config Files (present)

| File | Status | Notes |
|------|--------|-------|
| `benchmarks/configs/ct_fluorescence.yaml` | Present | Base config, maturity M0 |
| `benchmarks/expanded_configs/ct_fluorescence_expanded.yaml` | Present | Expanded with image sizes and noise levels |

### Learning Materials (present, 6 files)

| File | Size | Status |
|------|------|--------|
| `README.md` | 1,451 B | Present |
| `01_physics_fundamentals.md` | 2,163 B | Present -- but physics is CT-only |
| `02_forward_model.md` | 2,663 B | Present -- missing fluorescence equations |
| `03_reconstruction_algorithms.md` | 2,030 B | Present -- only 2 solver tiers |
| `04_pwm_benchmark.md` | 2,355 B | Present |
| `05_hands_on_tutorial.md` | 3,537 B | Present |
| `modify_plan.md` | -- | Present -- documents algorithm mismatch and fix plan |

### Dataset Integrity Assessment: **FAIL**

No benchmark data exists locally. The fallback generator (shepp_logan) produces
CT-only phantoms with no fluorescence component. Any benchmark run without
manual data preparation will produce physically meaningless single-modality
results that do not test CT+fluorescence fusion at all.

---

## 3. Public Dataset Source Assessment

### Current Source: None (generated fallback)

| Property | Value |
|----------|-------|
| Dataset ID | (empty) |
| Dataset URL | (none) |
| Citation | (none) |
| License | (none) |
| Fallback | `generated` via `shepp_logan` |

### Assessment

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Source well-known? | FAIL | No real dataset; Shepp-Logan is CT-only |
| Accepted by professors? | FAIL | No FLIT researcher would accept Shepp-Logan as a fluorescence benchmark |
| Dev tier protected? | UNKNOWN | No data exists |
| Hidden tier protected? | UNKNOWN | No data exists |

### Recommended FLIT Datasets

| Dataset | Source | Type | Suitability |
|---------|--------|------|-------------|
| Digimouse + fluorescence inclusions | Simulation | 3D digital mouse phantom | HIGH -- standard multi-modal imaging phantom; CT attenuation from Digimouse + embedded fluorescent inclusions |
| IVIS SpectrumCT data (PerkinElmer) | Experimental | In vivo mouse FLIT | HIGH -- commercial CT+fluorescence system with paired data |
| Monte Carlo (MCX) + CT simulation | Simulation | Photon transport + Radon | HIGH -- physics-accurate ground truth for both modalities |
| Hybrid uCT-FMT datasets (Ale et al. 2012) | Experimental | Micro-CT + fluorescence | MEDIUM -- published multimodal datasets with some ground truth |
| Synthetic dual-contrast phantoms | Simulation | Custom FLIT generator | HIGH -- paired attenuation + fluorophore maps with known ground truth |

**Fix:** At minimum, implement a synthetic FLIT phantom generator that produces
paired CT attenuation coefficient maps and fluorophore concentration maps with
tissue-realistic optical properties (absorption, scattering, anisotropy).

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard (from live page)

| Rank | Method | Score | Type |
|------|--------|-------|------|
| 1 | Cross-Modal Xformer + gradient | 0.699 | Transformer |
| 2 | PnP-ADMM (Joint) + gradient | 0.656 | Plug-and-Play |
| 3 | FDot-Net + gradient | 0.644 | Deep Learning |
| 4 | Born/Rytov + FBP + gradient | 0.581 | Classical |

### Currently in Algorithm Catalog (`_algorithm_catalog.py`)

| Slot | Algorithm | Type | Source |
|------|-----------|------|--------|
| Classical | Born/Rytov + FBP | Classical | Arridge & Schotland, Inverse Probl. 2009 |
| PnP | PnP-ADMM (Joint) | Plug-and-Play | Venkatakrishnan et al., 2013 |
| Deep Learning | FDot-Net | Deep Learning | Gao et al., BOE 2021 |
| Transformer | Cross-Modal Xformer | Transformer | Multi-modal transformer, 2024 |

### Currently in Solver Config (`ct_fluorescence.yaml`)

| Tier | Name | Module | Notes |
|------|------|--------|-------|
| traditional_cpu | Adjoint | `pwm_core.recon.adjoint` | Generic adjoint -- not FLIT-specific |
| best_quality | PnP-ADMM | `pwm_core.recon.pnp_admm` | Generic PnP -- partially relevant |

### Critical Gap: Solver Config vs. Leaderboard/Catalog

The solver config contains 2 generic methods. The algorithm catalog contains
4 FLIT-specific methods. The leaderboard shows 4 methods matching the catalog.
But the solver config has zero overlap with the leaderboard -- the Adjoint
solver does not appear on the leaderboard, and FDot-Net/Cross-Modal Xformer
have no solver config entries. This means users cannot reproduce leaderboard
results from the config.

### Missing Famous/Recent FLIT and XFCT Algorithms

| Priority | Algorithm | Year | Type | Why Include |
|----------|-----------|------|------|-------------|
| CRITICAL | **FEM-FMT** (Ntziachristos et al.) | 2005-2010 | Classical | Finite-element fluorescence molecular tomography -- the foundational FMT solver; uses diffusion equation on tetrahedral meshes |
| CRITICAL | **Normalized Born** (Ntziachristos & Weissleder) | 2001 | Classical | Standard normalized ratio method for FMT; removes source coupling; most widely used FMT algorithm |
| HIGH | **1D-CNN + U-Net for XFCT** (Sci. Reports 2025) | 2025 | Deep Learning | End-to-end 1D CNN for XRF signal extraction + U-Net reconstruction; R^2 > 0.99, SSIM 0.979, reduces processing from 6 min to 1.25 sec per slice |
| HIGH | **Deep Image Prior for XFCT** (IJCARS 2024) | 2024 | Unsupervised DL | DIP pre-denoising for XFCT projections; CNR improved 3.7-4.6x; detection limit halved (0.069 to 0.035 mg/mL) |
| HIGH | **MLAA-style joint** (Rezaei et al.) | 2012 | Classical | Maximum likelihood joint activity+attenuation; applicable to joint CT+fluorescence when reformulated |
| MEDIUM | **DOT-guided FMT** (Ntziachristos, Applied Optics 2008) | 2008 | Hybrid | Diffuse optical tomography provides optical property maps to improve FMT accuracy |
| MEDIUM | **Deep-XFCT** (Energies 2022) | 2022 | Deep Learning | Deep learning 3D mineral liberation analysis combining XRF and CT |
| MEDIUM | **Unrolled ADMM for FMT** | 2023 | Model-Driven DL | Algorithm-unrolling approach for fluorescence tomography with physics-informed layers |
| LOW | **Monte Carlo FMT** (MCX-based) | Standard | Reference | Gold-standard photon transport simulation for validation of diffusion-based methods |
| LOW | **Hybrid uCT-FMT** (Ale et al.) | 2012 | Hybrid | Published multimodal reconstruction framework combining micro-CT and FMT |

### Performance Observations from Leaderboard

- **Cross-Modal Xformer** leads at 0.699 -- transformer architectures benefit
  from cross-attention between CT and fluorescence feature maps
- **Born/Rytov + FBP** scores only 0.581 -- the classical baseline suffers
  most under mismatch, which is expected since it uses a linearized approximation
- The 0.118 gap between best (transformer) and worst (classical) is moderate,
  suggesting mismatch severity is not extreme on current test data
- All methods append "+ gradient" -- suggesting gradient-based spec estimation
  is used by all approaches to handle mismatch parameters

---

## 5. Improvement Suggestions

### Priority 1 -- Physics Correctness (Weeks 1-3)

1. **Implement coupled CT+fluorescence forward model** -- Replace `microscopy_psf`
   with a dedicated module implementing Beer-Lambert for CT and coupled diffusion
   equations for fluorescence excitation/emission propagation.
2. **Add fluorescence signal equation** -- Include the excitation-emission-detection
   chain: excitation source -> tissue propagation -> fluorophore excitation ->
   emission -> tissue propagation -> surface detection.
3. **Fix noise model** -- Separate CT noise (Poisson + readout) from fluorescence
   noise (autofluorescence background + excitation leakage + dark current). Model
   the cross-modal error propagation from CT optical property estimation to
   fluorescence reconstruction.
4. **Revise DAG** -- Show serial dependency: CT -> optical property estimation ->
   fluorescence forward model, not parallel fusion.
5. **Fix wavelength/energy range** -- Specify X-ray energy (e.g., 20-140 keV for
   CT) and fluorescence wavelengths (e.g., 650-900 nm NIR window for in vivo).

### Priority 2 -- Dataset Quality (Weeks 2-4)

6. **Create FLIT-specific phantom generator** -- Dual-contrast phantoms with
   paired X-ray attenuation maps and fluorophore concentration distributions,
   using tissue-realistic optical properties.
7. **Replace Shepp-Logan fallback** -- Use digital mouse phantom (Digimouse) or
   MCX-based simulation for generating physically realistic FLIT data.
8. **Upgrade to 3D** -- Both CT and fluorescence are inherently 3D problems;
   the 2D [64,64] format trivializes the reconstruction challenge.
9. **Increase scene count** -- From 3 to at least 20 per tier for statistical
   significance, or implement bootstrapped confidence intervals.
10. **Reconcile mismatch parameter ranges** -- Fix YAML config vs. live page
    discrepancy; clamp autofluorescence to non-negative values.

### Priority 3 -- Algorithm Baselines (Weeks 3-5)

11. **Add Normalized Born as classical baseline** -- The standard FMT
    reconstruction method; should replace or supplement Born/Rytov + FBP.
12. **Add 1D-CNN + U-Net XFCT solver** -- State-of-the-art deep learning for
    X-ray fluorescence CT with demonstrated SSIM > 0.97 (Sci. Reports 2025).
13. **Add FEM-FMT solver** -- Finite-element fluorescence molecular tomography
    as the physics-based reference solver.
14. **Reconcile solver config with leaderboard** -- Add solver entries for
    FDot-Net and Cross-Modal Xformer so leaderboard results are reproducible.
15. **Add `famous_dl` and `small_gpu` solver tiers** -- Currently only
    `traditional_cpu` and `best_quality` tiers exist; the algorithm selection
    guide references `famous_dl` and `small_gpu` but they are not defined.

### Priority 4 -- Documentation (Week 5)

16. **Define PSNR_norm** -- Specify normalization bounds with a worked example.
17. **Complete all references** -- Add DOIs and full citations for all algorithms
    (especially "Cross-Modal Xformer, 2024" and "Gao et al., BOE 2021").
18. **Document HDF5 schema** -- Specify key names, array shapes, data types,
    and compression for benchmark data files.
19. **Fix wavelength range** -- Replace "0 -- 0 nm" with actual spectral ranges.

---

## 6. Action Items

| # | Priority | Severity | Action | Owner |
|---|----------|----------|--------|-------|
| A1 | P1 | HIGH | Replace `microscopy_psf` with coupled CT+diffusion forward model | Physics team |
| A2 | P1 | HIGH | Add fluorescence signal equations (excitation/emission diffusion) | Physics team |
| A3 | P1 | HIGH | Fix noise model: separate CT/fluorescence noise + cross-modal error propagation | Physics team |
| A4 | P1 | HIGH | Revise DAG to show serial CT->optical-property->FMT dependency | Physics team |
| A5 | P1 | HIGH | Define PSNR_norm bounds and add worked scoring example | Platform team |
| A6 | P1 | HIGH | Reconcile mismatch parameter ranges between YAML config and live page | Platform team |
| A7 | P2 | HIGH | Create FLIT dual-contrast phantom generator (CT attenuation + fluorophore maps) | Data team |
| A8 | P2 | HIGH | Replace Shepp-Logan fallback with Digimouse or MCX-based FLIT simulation | Data team |
| A9 | P2 | MEDIUM | Upgrade from 2D [64,64] to 3D volumes for both modalities | Data team |
| A10 | P2 | MEDIUM | Increase scene count from 3 to 20+ per tier | Data team |
| A11 | P3 | HIGH | Add Normalized Born and FEM-FMT as classical FLIT baselines | Recon team |
| A12 | P3 | HIGH | Add 1D-CNN + U-Net XFCT solver (SOTA 2025) | Recon team |
| A13 | P3 | MEDIUM | Add solver config entries for FDot-Net and Cross-Modal Xformer | Recon team |
| A14 | P3 | MEDIUM | Define `famous_dl` and `small_gpu` solver tiers | Recon team |
| A15 | P4 | MEDIUM | Complete all reference DOIs (Cross-Modal Xformer, Gao et al.) | Docs team |
| A16 | P4 | LOW | Document HDF5 schema (keys, shapes, dtypes, compression) | Docs team |
| A17 | P4 | LOW | Fix wavelength/energy range (X-ray keV + fluorescence nm) in learning materials | Docs team |
| A18 | P4 | LOW | Fix gallery rendering if images are not displaying on live page | Frontend team |

---

## Appendix: Key References

1. **Ntziachristos, V. & Weissleder, R.** (2001) "Experimental three-dimensional
   fluorescence reconstruction of diffuse media by use of a normalized Born
   approximation," Optics Letters 26(12), 893-895.
   DOI: [10.1364/OL.26.000893](https://doi.org/10.1364/OL.26.000893)

2. **Arridge, S.R. & Schotland, J.C.** (2009) "Optical tomography: forward and
   inverse problems," Inverse Problems 25(12), 123010.
   DOI: [10.1088/0266-5611/25/12/123010](https://doi.org/10.1088/0266-5611/25/12/123010)

3. **Gao, Y. et al.** (2021) "FDot-Net: Deep learning for fluorescence diffuse
   optical tomography," Biomedical Optics Express.

4. **Venkatakrishnan, S.V. et al.** (2013) "Plug-and-Play Priors for Model Based
   Reconstruction," IEEE GlobalSIP.

5. **Scientific Reports (2025)** -- "Deep learning based rapid X-ray fluorescence
   signal extraction and image reconstruction for preclinical benchtop XFCT
   applications." DOI: [10.1038/s41598-025-03900-0](https://doi.org/10.1038/s41598-025-03900-0)

6. **IJCARS (2024)** -- "Fundamental study on improving the quality of X-ray
   fluorescence computed tomography images by applying deep image prior."
   DOI: [10.1007/s11548-024-03307-8](https://doi.org/10.1007/s11548-024-03307-8)

7. **Ale, A. et al.** (2012) "FMT-XCT: in vivo animal studies with hybrid
   fluorescence molecular tomography-X-ray computed tomography," Nature Methods
   9, 615-620. DOI: [10.1038/nmeth.2014](https://doi.org/10.1038/nmeth.2014)

8. **Ntziachristos, V.** (2010) "Going deeper than microscopy: the optical imaging
   frontier in biology," Nature Methods 7(8), 603-614.
   DOI: [10.1038/nmeth.1483](https://doi.org/10.1038/nmeth.1483)

9. **Monici, M.** (2005) "Cell and tissue autofluorescence research and diagnostic
   applications," Biotechnology Annual Review 11, 227-256.

10. **Rezaei, A. et al.** (2012) "Simultaneous Reconstruction of Activity and
    Attenuation in Time-of-Flight PET," IEEE TMI 31(12), 2224-2233.

---

*Comprehensive 6-point review on 2026-03-03 -- covering benchmark page errors, local dataset inspection, public dataset source assessment, algorithm coverage assessment, improvement suggestions, and prioritized action items. M0 maturity. CRITICAL: wrong forward model (microscopy_psf instead of coupled CT+diffusion), no fluorescence signal equation, no local dataset, Shepp-Logan fallback has no fluorescence component. Leaderboard has 4 FLIT-specific methods but solver config contains only 2 generic methods with zero overlap.*