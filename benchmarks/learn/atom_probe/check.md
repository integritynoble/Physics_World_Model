# Comprehensive Benchmark QA Check — Atom Probe Tomography (APT)

**URL:** https://pwm.platformai.org/benchmark/atom_probe
**HTTP Status:** 200
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Manual deep analysis + web research

---

## Table of Contents

1. [Benchmark Page Errors](#1-benchmark-page-errors)
2. [Local Dataset Inspection](#2-local-dataset-inspection)
3. [Public Dataset Source Assessment](#3-public-dataset-source-assessment)
4. [Algorithm Coverage Assessment](#4-algorithm-coverage-assessment)
5. [Improvement Suggestions](#5-improvement-suggestions)
6. [Action Items](#6-action-items)

---

## 1. Benchmark Page Errors

### Summary

| Severity | Count |
|----------|-------|
| HIGH     | 5     |
| MEDIUM   | 5     |
| LOW      | 3     |

### HIGH Severity

**H1. Dataset source listed as Protein Data Bank (PDB) -- incorrect for APT**
- Webpage states dataset source is "Protein Data Bank (PDB) (Berman et al., NAR 2000)"
- PDB is a repository for protein crystal structures, not atom probe tomography data
- Local config (`atom_probe.yaml`) shows `dataset_id: ''`, `dataset_url: ''`, `fallback: generated`, `synthetic_generator: shepp_logan`
- The benchmark actually uses synthetically generated data, not PDB
**Fix:** Remove the PDB reference. State "Synthetically generated (Shepp-Logan phantom)" or acquire a real APT dataset.

**H2. Mismatch parameter ranges on webpage differ from local config**
- Webpage ranges (from web fetch):
  - flight_path_error: -0.1 to 0.2 mm (public), -0.12 to 0.18 mm (dev), -0.07 to 0.23 mm (hidden)
  - voltage_calibration: 0.996-1.008 (public), 0.9952-1.0072 (dev), 0.9972-1.0092 (hidden)
  - detection_efficiency: 0.58-0.64 (public), 0.576-0.636 (dev), 0.586-0.646 (hidden)
  - tip_radius_error: -1.0 to 2.0 nm (public), -1.2 to 1.8 nm (dev), -0.7 to 2.3 nm (hidden)
- Local config (`atom_probe.yaml`) ranges:
  - flight_path_error: -0.5 to 0.5 mm
  - voltage_calibration: 0.98 to 1.02
  - detection_efficiency: 0.5 to 0.7
  - tip_radius_error: -5.0 to 5.0 nm
- The webpage per-tier ranges are much narrower than the config ranges, suggesting the config stores the overall envelope while the webpage shows tier-specific subsets. However, this is inconsistent and undocumented.
**Fix:** Sync webpage per-tier ranges with local config, or document the sub-range strategy explicitly.

**H3. Shepp-Logan phantom is physically inappropriate for APT**
- Local config uses `shepp_logan` as the synthetic generator
- Shepp-Logan is a CT head phantom (ellipses representing skull, brain tissue, etc.)
- APT data represents 3D atomic positions with mass-to-charge ratios, not 2D attenuation maps
- Using a CT phantom as ground truth for an ion-based 3D tomography modality is physically meaningless
**Fix:** Replace with an APT-appropriate synthetic generator (e.g., simulated nano-precipitates in alloy matrix, lattice structure phantoms).

**H4. Data dimensions [64, 64] are unrealistically small for APT**
- Local config: x_shape=[64,64], y_shape=[64,64]
- Real APT datasets contain millions to billions of ions in 3D point clouds
- A 64x64 2D grid does not represent APT data, which is fundamentally 3D point cloud data (x, y, z, m/z)
- The expanded config offers 128x128, 256x256, 512x512 but these are still 2D grids
**Fix:** Redesign the data format to be 3D point cloud or 3D voxel grid (e.g., [64, 64, 64] minimum).

**H5. Forward model category "microscopy_psf" is incorrect for APT**
- Config: `category_module: microscopy_psf`, `forward_model_type: nonlinear_operator`
- APT is not a PSF-convolution imaging system; it is a field-evaporation + time-of-flight mass spectrometry technique
- The signal equation `I(m/z, x, y) = Y(m/z) * c(x,y) * primary_dose + noise` describes SIMS, not APT
- APT forward model should involve: field evaporation probability, ion trajectory (flight path), ToF-to-m/z conversion, detector hit position mapping
**Fix:** Implement a dedicated APT forward model based on point-projection reconstruction physics.

### MEDIUM Severity

| ID | Issue |
|----|-------|
| M1 | Leaderboard method "CalibFormer" has no published reference -- unverifiable |
| M2 | Leaderboard method "ResNet-Calib" has no published reference -- unverifiable |
| M3 | Leaderboard method "CrystalNet" listed in auto-check but "Deconv" listed on webpage -- inconsistency between auto-check and web fetch |
| M4 | Scoring formula uses PSNR_norm (40%) + SSIM (40%) + consistency (20%), but config only lists `metrics: [psnr, ssim]` with no consistency metric defined |
| M5 | DAG is `S --> D` (Sampling --> Detector) which is extremely generic; real APT would need `E --> T --> D` (Evaporation --> Trajectory --> Detector) or similar |

### LOW Severity

| ID | Issue |
|----|-------|
| L1 | Wavelength/Energy range listed as "0 -- 0 nm" in physics fundamentals (meaningless for ion probe) |
| L2 | No alt-text on gallery images |
| L3 | `reference_psnr: null` and `expected_psnr_range: null` in config -- no baseline expectations set |

---

## 2. Local Dataset Inspection

### File Inventory

| Tier | File | Exists |
|------|------|--------|
| Public | `datasets/benchmark/atom_probe/` | **NO** |
| Dev | `datasets/benchmark/atom_probe/` | **NO** |
| Hidden | `datasets/benchmark/atom_probe/` | **NO** |

**No local dataset exists.** The directory `datasets/benchmark/atom_probe/` does not exist.

### Learning Materials Inventory

| File | Size (bytes) | Status |
|------|------|--------|
| README.md | 1,451 | Present |
| 01_physics_fundamentals.md | 1,913 | Present |
| 02_forward_model.md | 2,710 | Present |
| 03_reconstruction_algorithms.md | 2,031 | Present |
| 04_pwm_benchmark.md | 2,368 | Present |
| 05_hands_on_tutorial.md | 3,518 | Present |

### Config Files

| File | Status | Notes |
|------|--------|-------|
| `benchmarks/configs/atom_probe.yaml` | Present | Main config, maturity M0 |
| `benchmarks/expanded_configs/atom_probe_expanded.yaml` | Present | Expanded config, 192 total cases |

### Config Inconsistencies

| Field | atom_probe.yaml | expanded_config | Issue |
|-------|----------------|-----------------|-------|
| x_shape | [64, 64] | [128/256/512] (multi-size) | Config has 64x64 but expanded has 128/256/512 |
| mismatch_params | 4 parameters with ranges | `mismatch_params: []` (empty) | Expanded config has empty mismatch params |
| data_source | `fallback: generated` | `type: generated` | Both confirm synthetic -- no real data |

### Maturity Level: **M0 (lowest)**

The expanded config confirms `maturity: M0`, meaning this modality is at the earliest development stage with no real data, no validated forward model, and generic solvers.

### Dataset Integrity Assessment: **FAIL -- no dataset exists**

---

## 3. Public Dataset Source Assessment

### Current Source: Synthetically Generated (Shepp-Logan) -- **POOR**

| Criterion | Rating | Notes |
|-----------|--------|-------|
| Public: Well-known? | **POOR** | Shepp-Logan is well-known for CT, but physically wrong for APT |
| Public: Accepted by professors? | **FAIL** | No APT researcher would accept a CT phantom as APT data |
| Dev: Protected? | N/A | No dataset exists |
| Hidden: Protected? | N/A | No dataset exists |
| Webpage claims PDB source? | **INCORRECT** | PDB is for protein crystal structures, not APT |

### What Real APT Datasets Should Look Like

Real APT data consists of:
- 3D point clouds with (x, y, z) coordinates for each detected ion
- Mass-to-charge ratio (m/z) for each ion (enabling elemental identification)
- Millions to billions of ions per dataset
- From instruments like CAMECA LEAP (Local Electrode Atom Probe)

### Available Real APT Data Sources

| Source | Description | Accessibility |
|--------|-------------|---------------|
| APT Repository (Univ. Sydney) | Community-shared APT datasets | Open |
| NIST APT datasets | Standard reference materials | Open |
| Max-Planck MPIE APT data | Shared datasets from Gault group | Varies |
| Atom Probe Informatics (API) | Emerging standardization effort | Limited |

### Assessment: **FAIL -- no credible data source for APT benchmark**

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard: 4 algorithms

| # | Algorithm | Type | PSNR (pub/dev/hid) | SSIM (pub/dev/hid) |
|---|-----------|------|--------------------|--------------------|
| 1 | CalibFormer + gradient | Transformer-based calibration | 30.28 / 28.08 / 26.16 | 0.921 / 0.882 / 0.836 |
| 2 | ResNet-Calib + gradient | CNN calibration | 28.48 / 23.09 / 22.89 | 0.891 / 0.735 / 0.727 |
| 3 | PnP-BM3D + gradient | Plug-and-play | 25.43 / 23.35 / 21.78 | 0.815 / 0.745 / 0.680 |
| 4 | Deconv + gradient | Deconvolution | 21.81 / 20.99 / 20.04 | 0.682 / 0.645 / 0.601 |

### Local Solvers: 2 registered

| Tier | Solver | Module |
|------|--------|--------|
| traditional_cpu | Adjoint | `pwm_core.recon.adjoint` |
| best_quality | PnP-ADMM | `pwm_core.recon.pnp_admm` |

### Missing Famous/Recent APT Algorithms

| Priority | Algorithm | Year | Why It Should Be Included |
|----------|-----------|------|---------------------------|
| **HIGH** | Bas protocol (point-projection) | 1995 | The foundational APT reconstruction algorithm -- every APT paper uses it |
| **HIGH** | Geiser-Larson wide-FOV reconstruction | 2007-2011 | Standard in CAMECA IVAS/AP Suite software, used by nearly all APT labs |
| **HIGH** | Dynamic Reconstruction (DR) | 2011+ | Larson et al., accounts for evolving tip shape during evaporation |
| **HIGH** | AtomNet (3D deep learning) | 2024 | First 3D deep learning for APT point cloud processing (Wei et al., Acta Materialia 2024) |
| **MEDIUM** | ML-enhanced crystallographic analysis | 2018 | Stephenson et al., ML for detector pattern recognition |
| **MEDIUM** | ML chemical short-range order | 2023 | Nature Comms, ML to break APT resolution limits |
| **MEDIUM** | Phase segmentation DL | 2019 | Zelenty et al., Scientific Reports, DL-based edge detection for APT |
| **LOW** | Bottom-up volume reconstruction | 2013 | Alternative to standard top-down point-projection |
| **LOW** | Model-driven reconstruction | 2020 | Physics-informed approach beyond analytic back-projection |

### Fundamental Problem with Current Algorithm Coverage

The 4 leaderboard algorithms (CalibFormer, ResNet-Calib, PnP-BM3D, Deconv) are **generic image reconstruction methods**, not APT-specific algorithms. Real APT reconstruction is fundamentally different:
- APT operates on 3D point clouds, not 2D images
- The core task is ion trajectory inversion (detector hit -> specimen position), not image deconvolution
- m/z identification is integral to reconstruction, not a separate step
- Standard APT uses the Bas/Geiser point-projection protocol, not gradient descent on a PSF model

**Total gap: 9 missing algorithms, with a fundamental mismatch in algorithm paradigm**

---

## 5. Improvement Suggestions

### 5.1 Forward Model (CRITICAL -- complete redesign needed)

1. **Replace microscopy_psf with a dedicated APT forward model** implementing:
   - Field evaporation probability (Kingham curves, evaporation field)
   - Ion trajectory from specimen tip to detector (point-projection geometry)
   - Time-of-flight to mass-to-charge conversion
   - Detector efficiency and multi-hit losses
2. **Change DAG from `S --> D` to `E --> T --> D`** (Evaporation --> Trajectory --> Detection)
3. **Implement proper mismatch physics**: tip shape uncertainty, flight path length error, voltage pulse shape, detection efficiency variation

### 5.2 Data Format (CRITICAL -- 3D point cloud needed)

4. **Change from 2D [64, 64] grid to 3D representation**:
   - Option A: 3D voxel grid [Nx, Ny, Nz] with compositional channels
   - Option B: Point cloud format (N ions x 4 features: x, y, z, m/z)
5. **Replace Shepp-Logan phantom** with APT-appropriate synthetic data:
   - Simulated binary/ternary alloy microstructures
   - Precipitate-matrix systems (e.g., Al-Li-Mg, Fe-based alloys)
   - Grain boundary segregation phantoms

### 5.3 Dataset Source

6. **Acquire or generate proper APT datasets**:
   - Contact NIST for standard reference APT datasets
   - Use TAPSim or similar APT simulation software to generate realistic synthetic data
   - Partner with MPIE (Max-Planck Institut fur Eisenforschung) for open APT data
7. **Remove incorrect PDB reference** from webpage immediately

### 5.4 Algorithms

8. **Implement Bas/Geiser point-projection** as the baseline algorithm
9. **Add AtomNet (2024)** as the deep learning benchmark
10. **Add Dynamic Reconstruction** as the advanced classical method

### 5.5 Metrics

11. **Add APT-specific metrics** beyond PSNR/SSIM:
    - Spatial resolution (nearest-neighbor distribution)
    - Compositional accuracy (at% error)
    - Spatial distribution function (SDF) correlation
    - Cluster detection accuracy (for precipitate benchmarks)

### 5.6 Infrastructure

12. **Update maturity from M0 to at least M1** once real data and proper forward model are in place
13. **Fix wavelength/energy range** from "0 -- 0 nm" to appropriate values (or remove for ion probe)

---

## 6. Action Items

| Priority | Action | Owner | Status |
|----------|--------|-------|--------|
| **CRITICAL** | Remove incorrect PDB dataset source from webpage | Web team | TODO |
| **CRITICAL** | Replace Shepp-Logan with APT-appropriate synthetic generator | Forward model team | TODO |
| **CRITICAL** | Redesign forward model from microscopy_psf to dedicated APT point-projection | Forward model team | TODO |
| **CRITICAL** | Change data format from 2D [64,64] to 3D point cloud or voxel grid | Data team | TODO |
| **HIGH** | Sync webpage mismatch ranges with local config or document sub-range strategy | Web team | TODO |
| **HIGH** | Implement Bas/Geiser point-projection as baseline solver | Algorithm team | TODO |
| **HIGH** | Add AtomNet (2024) deep learning solver | Algorithm team | TODO |
| **HIGH** | Acquire real APT data from NIST, MPIE, or APT community repositories | Data team | TODO |
| **MEDIUM** | Add APT-specific metrics (compositional accuracy, spatial resolution) | Metrics team | TODO |
| **MEDIUM** | Fix DAG from S-->D to E-->T-->D | Config team | TODO |
| **MEDIUM** | Add Dynamic Reconstruction solver | Algorithm team | TODO |
| **MEDIUM** | Resolve leaderboard inconsistency (CrystalNet vs Deconv) | Web team | TODO |
| **LOW** | Fix wavelength range "0 -- 0 nm" in physics fundamentals | Docs team | TODO |
| **LOW** | Add missing references with DOIs for all leaderboard methods | Docs team | TODO |
| **LOW** | Set reference_psnr and expected_psnr_range in config | Config team | TODO |

---

## Appendix: Key References

- Bas et al. "A general protocol for the reconstruction of 3D atom probe data." Applied Surface Science 87/88:298-304 (1995). -- Foundational APT reconstruction.
- Geiser et al. "Wide-field-of-view atom probe reconstruction." Microscopy and Microanalysis (2007-2009). -- Standard commercial implementation.
- Larson et al. "Local Electrode Atom Probe Tomography: A User's Guide." Springer (2013). ISBN 978-1-4614-8721-0. -- Comprehensive APT reference.
- Gault et al. "Atom Probe Tomography." Nature Reviews Methods Primers 1:51 (2021). -- Modern review.
- Wei et al. "3D deep learning for enhanced atom probe tomography analysis of nanoscale microstructures." Acta Materialia (2024). arXiv:2404.16524. -- AtomNet, first 3D DL for APT.
- Stephenson et al. "Machine-learning-based atom probe crystallographic analysis." Ultramicroscopy (2018). -- ML for APT detector patterns.
- Zhou et al. "Quantitative 3D imaging of chemical short-range order via ML-enhanced APT." Nature Communications 14:7410 (2023). -- ML breaking APT resolution limits.
- Zelenty et al. "Phase segmentation in atom-probe tomography using deep learning-based edge detection." Scientific Reports 9:20632 (2019). -- DL segmentation for APT.
- Peng et al. "Machine learning enhanced atom probe tomography analysis: a snapshot review." arXiv:2504.14378 (2025). -- Comprehensive recent review.

---

*Comprehensive 6-point review on 2026-03-03. Atom Probe Tomography (APT) is at maturity M0 with CRITICAL issues: the forward model, data format, synthetic generator, and dataset source are all physically inappropriate for this modality. The benchmark currently repurposes generic 2D image reconstruction infrastructure rather than implementing the fundamentally different 3D point-cloud-based ion trajectory reconstruction that defines real APT. A complete redesign of the forward model, data pipeline, and algorithm suite is required before this modality can be considered scientifically credible.*