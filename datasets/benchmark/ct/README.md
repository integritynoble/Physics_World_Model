# CT — 2-D Fan-Beam Sparse-View / Low-Dose

## Public Data Source

All three tiers use **real patient CT images from LoDoPaB-CT**
(Leuschner et al. 2021, *Scientific Data* doi:10.1038/s41597-021-00893-z),
sourced from the LIDC/IDRI lung CT database. Zenodo record 3384092, CC BY 4.0.

| Tier | Source | Patients | Scenes |
|------|--------|----------|--------|
| Public | LoDoPaB-CT **test** split | Test patients | 11 slices |
| Dev | LoDoPaB-CT **validation** split — first half | Val patients 0–63 | 20 slices |
| Hidden | LoDoPaB-CT **validation** split — second half + adversarial | Val patients 64–127 | 20 slices |

Each tier uses entirely different patients — no shared scenes across tiers.

## Spec DAG

```
R(θ) ──► Π(fan-beam) ──► D(noise, mismatch)
```

## Forward Model

Fan-beam (divergent-ray) geometry matching a clinical scanner setup:

| Parameter | Value | Physical |
|-----------|-------|----------|
| IMAGE_SIZE | 362 × 362 px | FOV ≈ 26 cm × 26 cm |
| pixel_size | — | 0.718 mm/px |
| D_so | 800 px | ≈ 575 mm |
| D_sd | 568 px | ≈ 408 mm |
| n_det | 736 | — |
| det_spacing | 1.496 px | ≈ 1.07 mm |
| n_views (public/dev) | 60 | sparse |
| n_views (hidden) | 40–90 | per-sample random |

Noise: Beer-Lambert + Poisson(I₀ = 10 000) + readout N(0, 5²).

## LoDoPaB-CT Normalisation

```
x_true ∈ [0, 1]    x = (HU + 1000) / 4071
  0.00 → air (−1000 HU)
  0.25 → soft tissue / water (0 HU)
  0.42 → cortical bone (700 HU)
  1.00 → maximum density (3071 HU)
```

## Mismatch ThetaSpace

| Knob | Symbol | Description |
|------|--------|-------------|
| `center_offset_px` | Δc | Lateral shift of centre-of-rotation |
| `angle_error_deg` | Δθ | Systematic angular calibration error |
| `beam_hardening_beta` | β | Polychromatic BH: p_eff = p + β·p² |
| `detector_tilt_deg` | φ | Rigid tilt of detector plane |

## Scoring

```
Score = 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency
```

## Dataset Structure

```
ct/
├── lodopab_src/
│   ├── ground_truth_test.zip        (~1.5 GB) — public tier source
│   └── ground_truth_validation.zip  (~1.5 GB) — dev + hidden tier source
├── simulate_scenes.py         Procedural phantom generator (fallback only)
├── generate_dataset.py        Builds all H5 files + PNG images
├── public/    11 real LoDoPaB-CT test slices — GT + ideal sino + true spec (visible)
├── dev/       20 real LoDoPaB-CT validation slices (patients 0–63) — blind eval
└── hidden/    20 real LoDoPaB-CT validation slices (patients 64–127) + adversarial mods
```

## Reading the HDF5

```python
import h5py, json, numpy as np

with h5py.File("ct_challenge_dev.h5", "r") as f:
    grp = f["sample_00"]
    x_true     = grp["x_true"][:]             # (362, 362) float32  — GT attenuation map
    sino_ideal = grp["sinogram_ideal"][:]      # (60, 736)  float32  — nepers, no mismatch
    sino_meas  = grp["sinogram_measured"][:]   # (60, 736)  float32  — nepers, with mismatch
    angles     = grp["angles_nominal"][:]      # (60,)      float32  — radians
    spec       = json.loads(grp.attrs["spec_ranges"])
    true_spec  = json.loads(grp.attrs["true_spec"])
```

## Procedural Scene Types (Dev)

| Scene type | Anatomy |
|------------|---------|
| `chest_upper` | Carina level: trachea, large bilateral lungs, upper mediastinum |
| `chest_mid`   | Heart level: cardiac shadow, full lungs, descending aorta, ribs |
| `chest_lower` | Diaphragm: small lung bases, liver onset, stomach |
| `abdomen_upper` | Liver level: liver, spleen, kidneys, stomach, no lungs |
| `abdomen_mid`   | Kidney/bowel level: bowel loops, psoas, retroperitoneal fat |

## Adversarial Modifications (Hidden)

| Modification | Freq | Challenge |
|---|---|---|
| Metal implants | 35% | High-density streaks, dynamic range |
| Low-contrast lesions | 30% | Subtle nodules, hepatic cysts |
| Calcifications | 20% | Punctate high-density spots |
| High-contrast bone | 15% | Extreme dynamic range |

## References

- Leuschner et al. (2021) LoDoPaB-CT. *Scientific Data* 8:109.
  doi:10.1038/s41597-021-00893-z
- Feldkamp, Davis & Kress (1984) *JOSA A* 1:612-619.
- PWM Benchmark: https://pwm.platformai.org/benchmark/ct
