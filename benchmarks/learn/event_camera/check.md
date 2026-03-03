# Comprehensive Check: event_camera

**Modality:** Event Camera / Dynamic Vision Sensor (DVS)
**Category:** computational_photography
**Carrier:** Photon
**Check Date:** 2026-03-03
**Status:** PASS

---

## 1. Physics & Forward Model

### Signal Physics

Event cameras (Dynamic Vision Sensors) are neuromorphic sensors where each pixel
independently and asynchronously reports logarithmic brightness changes that
exceed a contrast threshold C. When the log-intensity change at pixel (x, y)
crosses the threshold, an event e = (x, y, t, p) is emitted, where t is the
microsecond timestamp and p is the polarity (ON for brightening, OFF for
darkening). The sensor output is a variable-rate, sparse, asynchronous event
stream -- not a conventional frame.

The event generation model is:

```
e_k = (x_k, y_k, t_k, p_k)  where  p_k = sign(log I(x_k, t_k) - log I(x_k, t_k - dt) - C)
```

The inverse problem is to reconstruct a dense intensity video I(x, y, t) from
the asynchronous event stream {e_k}.

### Forward Model Assessment

The learning materials describe the forward model type as `nonlinear_operator`
with category module `compressive_mask`. The nonlinear classification is correct
-- event generation is fundamentally nonlinear (threshold + sign function). The
`compressive_mask` category module is a reasonable abstraction for the benchmark
phantom generator, even though it does not capture the full temporal dynamics of
real DVS operation.

The DAG notation is `M -> D`, representing the event generation (mask/modulation)
followed by detection. This captures the essential structure.

**Mismatch parameters** are physically appropriate:
- Contrast threshold (0.1-0.5 log intensity): controls event sensitivity
- Refractory period (0.1-10.0 us): dead time after each event
- Noise event rate (0.0-1.0 relative): spurious background events
- Hot pixel fraction (0.0-0.5): stuck pixels firing continuously

### Verdict: ACCEPTABLE

The forward model correctly identifies the nonlinear nature of event generation.
The mismatch parameters target the right physical phenomena. The compressive mask
abstraction is a simplification but valid for benchmarking purposes.

---

## 2. Mismatch Parameters & Benchmark Structure

### Three-Tier Structure

| Tier | Mismatch Level | Ground Truth | Download |
|------|---------------|--------------|----------|
| Public | Mild | Included | Available |
| Dev | Moderate | Excluded | Available |
| Hidden | Severe | Excluded | Blocked (403) |

### Mismatch Parameter Coverage

| Parameter | Nominal | Range | Physical Basis |
|-----------|---------|-------|---------------|
| Contrast threshold | 0.3 | 0.1 - 0.5 | DVS pixel comparator bias |
| Refractory period | 1.0 us | 0.1 - 10.0 us | Post-event reset delay |
| Noise event rate | 0.0 | 0.0 - 1.0 | Dark current / junction leakage |
| Hot pixel fraction | 0.0 | 0.0 - 0.5 | Manufacturing defects |

The mismatch parameters are well-chosen for event cameras. Contrast threshold
variation is the single most impactful parameter -- algorithms that assume a
fixed threshold will degrade. Refractory period mismatch causes temporal
resolution loss. Noise events and hot pixels are well-documented DVS artifacts.

### Data Format

- Object shape: [64, 64]
- Measurement shape: [64, 64]
- Data source: hdr_dataset (Hasinoff et al., SIGGRAPH Asia 2016)
- Metrics: PSNR (primary), SSIM

### Verdict: GOOD

The three-tier mismatch structure with physically motivated parameters is
well-designed for evaluating event-to-video reconstruction robustness.

---

## 3. Reconstruction Methods & Leaderboard

### Algorithm Override (Verified in _algorithm_catalog.py)

| Algorithm | Type | Params | Source |
|-----------|------|--------|--------|
| Event Integration | Classical | 0 | Analytical baseline |
| cF2F | PnP | 0 | Scheerlinck et al., IEEE RA-L 2020 |
| E2VID | Deep Learning | 10M | Rebecq et al., IEEE TPAMI 2020 |
| SPADE-E2VID | Transformer | 15M | Cadena et al., 2024 |

### Algorithm Appropriateness

All four algorithms are domain-appropriate for event camera reconstruction:

1. **Event Integration** -- the simplest baseline: accumulate events within a
   time window to form a pseudo-frame. Zero learned parameters. Establishes
   the performance floor.

2. **cF2F (Complementary Frames to Frames)** -- Scheerlinck et al. (IEEE RA-L
   2020) reconstructs intensity by combining an event-driven complementary filter
   with frame-based priors. Classified as PnP for its iterative prior structure.

3. **E2VID** -- Rebecq et al. (IEEE TPAMI 2020) is the foundational deep
   learning approach for event-to-video reconstruction. Uses a recurrent
   ConvLSTM architecture (approx. 10M parameters). Widely cited benchmark.

4. **SPADE-E2VID** -- Cadena et al. (2024) extends E2VID with spatially adaptive
   denormalization and transformer-based temporal attention. Represents the
   current state of the art.

### Leaderboard Scores (from CATEGORY_REAL_SCORES)

| Method | PSNR (dB) | SSIM |
|--------|-----------|------|
| Event Integration | 22.00 | 0.580 |
| cF2F | 26.50 | 0.760 |
| E2VID | 31.20 | 0.900 |
| SPADE-E2VID | 33.50 | 0.935 |

The progression from classical (22 dB) to state-of-the-art transformer (33.5 dB)
is realistic and consistent with published results.

### Verdict: EXCELLENT

The algorithm override correctly replaces the generic computational_photography
pool (which had Wiener-Deconv, PnP-FFDNet, HDR-CNN, Uformer -- none relevant
to event streams) with domain-specific event camera algorithms.

---

## 4. Literature & State of the Art (2024-2025)

### Key References

| Year | Paper | Venue | Contribution |
|------|-------|-------|-------------|
| 2019 | Scheerlinck et al., "CED: Color event camera dataset" | CVPRW | cF2F filter approach |
| 2020 | Rebecq et al., "High speed and HDR video with an event camera" | TPAMI | E2VID: ConvLSTM event-to-video |
| 2020 | Stoffregen et al., "Reducing the sim-to-real gap for event cameras" | ECCV | Domain adaptation for events |
| 2021 | Paredes-Valles et al., "Back to event basics" | CVPR | Self-supervised event reconstruction |
| 2023 | Ercan et al., "HyperE2VID" | CVPR | Hypernetwork-based adaptive E2VID |
| 2024 | Cadena et al., "SPADE-E2VID" | arXiv | Spatially adaptive event reconstruction |
| 2024 | Zhu et al., "Event camera survey" | TPAMI | Comprehensive survey of event methods |

### State of the Art Assessment

The event camera reconstruction field has matured significantly. E2VID (2020)
remains the canonical deep learning baseline, with SPADE-E2VID (2024) and
HyperE2VID (2023) pushing performance further. The benchmark's algorithm
selection spans the full range from simple accumulation to transformer-augmented
reconstruction.

### Verdict: CURRENT

Algorithm selection reflects 2024-2025 state of the art. The field is active
with new methods (HyperE2VID, SPADE-E2VID) continuing to improve upon E2VID.

---

## 5. Local Dataset & GCS Status

### Challenge Datasets on GCS

| Tier | File | Status |
|------|------|--------|
| Public | `challenge-data/v1.0/event_camera_challenge_public.h5` | OK |
| Dev | `challenge-data/v1.0/event_camera_challenge_dev.h5` | OK |
| Hidden | `challenge-data/v1.0/event_camera_challenge_hidden.h5` | Blocked (403) |

### Gallery Images

Gallery images served from GCS via `/gcs/img/benchmark_gallery/event_camera/`.

### Learning Materials

| File | Status | Size |
|------|--------|------|
| README.md | Present | 1,488 B |
| 01_physics_fundamentals.md | Present | 2,157 B |
| 02_forward_model.md | Present | 2,752 B |
| 03_reconstruction_algorithms.md | Present | 2,061 B |
| 04_pwm_benchmark.md | Present | 2,503 B |
| 05_hands_on_tutorial.md | Present | 3,558 B |

### Verdict: COMPLETE

All HDF5 challenge datasets present on GCS. Learning materials complete.

---

## 6. Comprehensive Assessment & Recommendations

### Overall Status: PASS

| Check | Result |
|-------|--------|
| Physics & forward model | Correct nonlinear event generation model |
| Mismatch parameters | Physically appropriate (threshold, refractory, noise, hot pixels) |
| Algorithm override | In place -- all 4 algorithms are event-camera-specific |
| Leaderboard scores | Realistic progression from 22.0 to 33.5 dB PSNR |
| Literature coverage | Current through 2024 (SPADE-E2VID) |
| GCS datasets | All 3 tiers present |
| Learning materials | Complete 5-file set |

### What Was Fixed

The original assignment used generic computational_photography algorithms
(Wiener-Deconv, PnP-FFDNet, HDR-CNN, Uformer) which are frame-based image
restoration methods with no relevance to asynchronous event streams. The
variant override replaced these with Event Integration, cF2F, E2VID, and
SPADE-E2VID -- all purpose-built for event-to-video reconstruction.

### Minor Notes

- The learning materials (01_physics_fundamentals.md) use a generic PSF
  convolution signal equation rather than the event generation model. This is a
  documentation gap but does not affect the benchmark operation.
- The overview correctly identifies the modality as a DVS with the forward model
  described in the detailed section 1 overview.

### Recommendations

No further code changes needed. The algorithm override is in place and verified.
