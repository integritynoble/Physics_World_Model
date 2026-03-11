# Priority 5 — Multimodality Combinations Implementation Plan

**Status:** In Planning
**Target Completion:** 2026-03-15 (estimated 4-6 hours)

---

## Overview

Priority 5 modalities are **multimodality combinations** that integrate two or more complementary imaging physics:

- **PET/CT:** Activity imaging (PET) with anatomical reference (CT) — CT provides attenuation map
- **PET/MR:** Activity imaging (PET) with soft-tissue contrast (MR) — MR for motion/attenuation
- **SPECT/CT:** Emission imaging (SPECT) with CT anatomy — CT for attenuation correction

These are realistic clinical workflows (PET/CT is standard in oncology). The key challenge is the **multimodal inverse problem**: reconstruct one modality using data from both, with cross-physics constraints.

---

## Physics Models

### 1. PET/CT

**Forward model:**
```
y_ct  = R * x_ct + n_ct           (CT sinogram from attenuation map)
y_pet = D(μ_map) * R * x_pet + n_pet  (PET sinogram with attenuation correction)

where:
  x_ct         — CT image (Hounsfield units)
  x_pet        — PET activity (MBq/cc)
  R            — Radon transform (parallel-beam, 256×256)
  D(μ_map)     — attenuation diagonal matrix (depends on CT)
  μ_map        — linear attenuation coefficient map (derived from CT)
  n_ct, n_pet  — Poisson noise
```

**Key mismatch parameters:**
- CT-to-PET registration error (shift in µ_map)
- Attenuation correction uncertainty (HU-to-µ calibration error)
- Scatter correction error (PET-specific)
- Motion between CT and PET acquisitions

**Inverse problem:**
Recover x_pet given y_pet and y_ct. Using y_ct to estimate µ_map → data-consistent PET reconstruction.

---

### 2. PET/MR

**Forward model:**
```
y_mr  = F * S * x_mr + n_mr          (MR k-space from soft tissue)
y_pet = D(μ_mr) * R * x_pet + n_pet  (PET with MR-based attenuation)

where:
  x_mr         — MR image (proton density / T1 contrast)
  x_pet        — PET activity
  F            — 2D FFT (MR k-space)
  S            — coil sensitivities (multi-coil MR)
  R            — Radon (PET)
  D(μ_mr)      — attenuation from MR (complex: soft-tissue lookup table)
  μ_mr         — MR-derived attenuation (usually poorer than CT)
```

**Key mismatch:**
- MR attenuation estimation error (soft tissue ~0.1 cm⁻¹, harder to get from MR)
- Motion artifact coupling (MR + PET acquired separately, misalignment)
- B0 inhomogeneity affecting MR contrast
- PET scatter in low-attenuation regions

---

### 3. SPECT/CT

**Forward model:**
```
y_ct    = R * x_ct + n_ct
y_spect = D(μ_map) * R * x_spect + scatter(y_spect) + n_spect

where:
  x_spect      — SPECT activity (Bq/cc)
  D(μ_map)     — attenuation matrix from CT
  scatter()    — photon scatter in SPECT (energy-dependent)
```

**Simpler than PET/CT** — geometry is same (both Radon), only difference is attenuation + scatter.

---

## Implementation Strategy

### Phase 1: PET/CT (Highest Priority, Highest Clinical Impact)

**Steps:**
1. **Reuse generators:** Create `pet_ct/generate_dataset.py` that:
   - Calls `ct_gen.generate_scene()` to create x_ct phantom
   - Calls `pet_gen.radon_transform(x_ct)` to compute Radon(x_ct)
   - Derives µ_map from CT: `μ = (HU + 1000) / 4071 * 0.081`  [mm⁻¹]
   - Computes attenuation matrix D from µ_map
   - Calls `pet_gen.radon_transform(x_pet)` with D-weighted Radon → y_pet
   - Adds Poisson noise
   - Saves to HDF5: {x_ct, x_pet, y_ct, y_pet, H_ideal}

2. **Mismatch parameters:**
   - `ct_registration_shift`: (-3, 3) pixels (public) → (-5, 5) (hidden)
   - `hu_to_mu_scale_error`: 0% (public) → ±5% (hidden)
   - `scatter_fraction`: 0% (public, pure attenuation) → 15% (hidden)

3. **Baseline algorithm:**
   - FBP(y_ct) → x_ct_estimate → µ_map_estimate
   - OSEM + AC(y_pet, µ_map_estimate) → x_pet_estimate
   - Expected PSNR: ~35 dB (good attenuation correction)

**Estimated effort:** 2-3 hours

### Phase 2: SPECT/CT (Medium Priority, Similar to PET/CT)

**Effort:** ~1 hour (mostly reuse PET/CT structure, swap SPECT forward model)

### Phase 3: PET/MR (Lower Priority, Complex Attenuation)

**Effort:** 2-3 hours (MR attenuation estimation is complex)

---

## Quick Implementation (MVP)

For rapid deployment, create **minimal multimodality datasets** that focus on the attenuation-correction inverse problem:

### PET/CT Minimal Version

```python
# datasets/benchmark/pet_ct/generate_dataset.py (sketch)

def generate_petct_sample(seed=0):
    # 1. Generate CT phantom
    x_ct = generate_ct_gt(seed)  # 362×362

    # 2. Generate PET phantom (different activity distribution)
    x_pet = generate_lesion_activity_map(seed + 1)  # 256×256, resampled

    # 3. Resample CT to 256×256 for Radon consistency
    x_ct_resized = zoom(x_ct, 256 / 362)

    # 4. Compute Radon projections
    theta = np.linspace(0, np.pi, 180, endpoint=False)
    y_ct = radon_transform(x_ct_resized, theta)        # (180, 367)

    # 5. Compute attenuation map from CT
    HU_MIN, HU_MAX = 0, 3071
    mu_map = (x_ct_resized * HU_MAX + 1000) / 4071 * 0.081
    D = np.diag(np.exp(-line_integral(mu_map, theta)))  # attenuation diagonal

    # 6. Compute PET sinogram with attenuation
    y_pet_ideal = radon_transform(x_pet, theta)
    y_pet_attenuated = D @ y_pet_ideal  # attenuation correction

    # 7. Add Poisson noise
    y_pet_measured = poisson_noise(y_pet_attenuated)
    y_ct_measured  = poisson_noise(y_ct)

    # 8. Save
    with h5py.File(f'pet_ct_challenge_public.h5', 'a') as f:
        g = f.create_group(f'sample_{idx:02d}')
        g.create_dataset('x_ct', data=x_ct_resized)
        g.create_dataset('x_pet', data=x_pet)
        g.create_dataset('y_ct', data=y_ct_measured)
        g.create_dataset('y_pet', data=y_pet_measured)
        g.create_dataset('H_ideal', data=np.array([D, theta]))  # operators
```

---

## Deployment Checklist

- [ ] **PET/CT public tier** (≥10 samples, 12/13 realistic)
- [ ] **PET/CT dev tier** (20 samples, higher mismatch)
- [ ] **PET/CT hidden tier** (20 samples, adversarial)
- [ ] **check.md** for PET/CT with 6-point assessment
- [ ] **Baseline algorithm** (FBP + OSEM) achieves ≥35 dB
- [ ] **Upload to GCS** at `challenge-data/v1.0/pet_ct_challenge_{tier}.h5`
- [ ] **Update state.md** → PET/CT: done | benchmark: in-progress
- [ ] **Gallery images** for 4 scenes (via precompute_all_gallery.py)
- [ ] **Repeat for SPECT/CT** (fast track, 1 hour)
- [ ] **Defer PET/MR** (complex, lower priority)

---

## Code Reuse

**Existing functions to leverage:**
- `datasets/benchmark/ct/generate_dataset.py` → `generate_ct_gt()`, `radon_transform()`
- `datasets/benchmark/pet/generate_dataset.py` → Poisson noise, attenuation matrix
- `benchmarks/learn/CT/check.md`, `benchmarks/learn/PET/check.md` → template for PET/CT check.md

**Helper utilities:**
- `scipy.ndimage.zoom` — resample between 362×362 (CT) and 256×256 (PET)
- `scipy.ndimage.rotate` — line integral for attenuation projection
- `scipy.sparse.diags` — efficient diagonal matrix (D)

---

## Success Criteria

| Criterion | Target |
|-----------|--------|
| PET/CT public tier baseline PSNR | ≥35 dB |
| PET/CT dev tier baseline PSNR | ≥32 dB |
| PET/CT hidden tier baseline PSNR | ≥28 dB |
| All 3 tiers complete | 60+ samples total |
| Documentation (check.md) | PASS status |
| GCS upload | Complete |
| Gallery integration | 4 scenes visible on benchmark page |

---

## Timeline

| Phase | Task | Effort | Est. Time |
|-------|------|--------|-----------|
| 1 | PET/CT generator | 1.5h | 15:00 |
| 2 | PET/CT baseline testing | 0.5h | 15:30 |
| 3 | Check.md + gallery | 0.5h | 16:00 |
| 4 | SPECT/CT (fast track) | 1h | 17:00 |
| 5 | GCS upload + state update | 0.5h | 17:30 |
| 6 | PET/MR (deferred or next batch) | 2-3h | *future* |

**Ready to start:** 2026-03-11 15:00 (after tier completions)

---

*Plan reviewed 2026-03-11 — aligned with remaining CPU time and server capacity.*
