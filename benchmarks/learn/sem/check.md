# Benchmark QA Check — sem

**URL:** https://pwm.platformai.org/benchmark/sem
**HTTP Status:** 200
**Page Size:** 110,196 bytes
**Check Date:** 2026-03-03 14:20 UTC

## Summary

| Severity | Count |
|----------|-------|
| WARNING | 12 |
| INFO | 56 |

## Issues by Category

### Leaderboard Consistency

- 🟡 **WARNING**: Very low score value found: 0.0200 — may indicate broken solver
- 🟡 **WARNING**: Very low PSNR value: 0.0 dB — may indicate failed reconstruction
- 🟡 **WARNING**: Very low PSNR value: 3.8 dB — may indicate failed reconstruction

### Forward Model

- 🟡 **WARNING**: No explicit forward model equation (y = ...) found on page
- 🟡 **WARNING**: Notation inconsistency: Forward operator denoted as both H and A
- 🔵 **INFO**: YAML forward model uses symbols ['I_b', 'E_0'] not found on page (YAML eq: y = G * eta(E_0) * I_b * x + n)

### Spec Ranges

- 🟡 **WARNING**: Negative physical quantity: -30 keV — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -10 nm — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -30 keV — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -5 nm — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -10 nm — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -5 kV — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -20 kV — verify sign is intended

### Images & Links

- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/sem_challenge_public.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/sem_challenge_dev.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sem/scene_03/recon_III.png — verify it loads

### Physics Consistency

- 🔵 **INFO**: Signal shape [512, 512] from YAML not clearly shown on page

---
*Auto-generated by `benchmarks/learn/check_all_modalities.py`*
