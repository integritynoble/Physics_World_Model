# Benchmark QA Check — confocal_3d

**URL:** https://pwm.platformai.org/benchmark/confocal_3d
**HTTP Status:** 200
**Page Size:** 111,255 bytes
**Check Date:** 2026-03-03 14:06 UTC

## Summary

| Severity | Count |
|----------|-------|
| WARNING | 5 |
| INFO | 56 |

## Issues by Category

### Leaderboard Consistency

- 🟡 **WARNING**: Very low PSNR value: 3.7 dB — may indicate failed reconstruction
- 🟡 **WARNING**: Very low PSNR value: 1.1 dB — may indicate failed reconstruction

### Forward Model

- 🟡 **WARNING**: No explicit forward model equation (y = ...) found on page
- 🟡 **WARNING**: Notation inconsistency: Forward operator denoted as both H and A

### Spec Ranges

- 🟡 **WARNING**: Negative physical quantity: -300 nm — verify sign is intended

### Images & Links

- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/confocal_3d_challenge_public.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/confocal_3d_challenge_dev.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/confocal_3d/scene_03/recon_III.png — verify it loads

### Physics Consistency

- 🔵 **INFO**: Signal shape [256, 256, 64] from YAML not clearly shown on page
- 🔵 **INFO**: Wavelength range 400–650 nm from YAML not found on page

---
*Auto-generated by `benchmarks/learn/check_all_modalities.py`*
