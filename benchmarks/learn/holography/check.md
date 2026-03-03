# Benchmark QA Check — holography

**URL:** https://pwm.platformai.org/benchmark/holography
**HTTP Status:** 200
**Page Size:** 111,261 bytes
**Check Date:** 2026-03-03 14:13 UTC

## Summary

| Severity | Count |
|----------|-------|
| WARNING | 4 |
| INFO | 57 |

## Issues by Category

### Leaderboard Consistency

- 🟡 **WARNING**: Very low PSNR value: 1.2 dB — may indicate failed reconstruction
- 🟡 **WARNING**: Very low PSNR value: 1.1 dB — may indicate failed reconstruction

### Forward Model

- 🟡 **WARNING**: No explicit forward model equation (y = ...) found on page
- 🟡 **WARNING**: Notation inconsistency: Forward operator denoted as both H and A
- 🔵 **INFO**: YAML forward model uses symbols ['U_obj', 'U_ref'] not found on page (YAML eq: I = |U_obj * exp(i*phi) + U_ref * exp(i*k*sin(theta)*r)|^2)

### Images & Links

- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/holography_challenge_public.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/holography_challenge_dev.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/holography/scene_03/recon_III.png — verify it loads

### Physics Consistency

- 🔵 **INFO**: Signal shape [512, 512] from YAML not clearly shown on page
- 🔵 **INFO**: Wavelength range 400–700 nm from YAML not found on page

---
*Auto-generated by `benchmarks/learn/check_all_modalities.py`*
