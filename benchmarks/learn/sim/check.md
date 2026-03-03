# Benchmark QA Check — sim

**URL:** https://pwm.platformai.org/benchmark/sim
**HTTP Status:** 200
**Page Size:** 112,899 bytes
**Check Date:** 2026-03-03 14:20 UTC

## Summary

| Severity | Count |
|----------|-------|
| WARNING | 6 |
| INFO | 59 |

## Issues by Category

### Leaderboard Consistency

- 🟡 **WARNING**: Very low score value found: 0.0350 — may indicate broken solver
- 🟡 **WARNING**: Very low PSNR value: 2.0 dB — may indicate failed reconstruction
- 🟡 **WARNING**: Very low PSNR value: 4.8 dB — may indicate failed reconstruction

### Forward Model

- 🟡 **WARNING**: No explicit forward model equation (y = ...) found on page
- 🟡 **WARNING**: Notation inconsistency: Forward operator denoted as both H and A
- 🔵 **INFO**: YAML forward model uses symbols ['N_patterns'] not found on page (YAML eq: y_k = PSF ** (I_k * x) + n,  k = 1..N_patterns)

### Spec Ranges

- 🟡 **WARNING**: Negative physical quantity: -500 m — verify sign is intended

### Images & Links

- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/sim_challenge_public.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/sim_challenge_dev.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sim/scene_03/recon_III.png — verify it loads

### Algorithm Names

- 🔵 **INFO**: Solver 'HiFi-SIM' from registry not found on page
- 🔵 **INFO**: Solver 'DL-SIM' from registry not found on page

### Physics Consistency

- 🔵 **INFO**: Signal shape [512, 512] from YAML not clearly shown on page
- 🔵 **INFO**: Wavelength range 400–650 nm from YAML not found on page

---
*Auto-generated by `benchmarks/learn/check_all_modalities.py`*
