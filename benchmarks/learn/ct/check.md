# Benchmark QA Check — ct

**URL:** https://pwm.platformai.org/benchmark/ct
**HTTP Status:** 200
**Page Size:** 134,651 bytes
**Check Date:** 2026-03-03 13:54 UTC

## Summary

| Severity | Count |
|----------|-------|
| WARNING | 7 |
| INFO | 67 |

## Issues by Category

### Leaderboard Consistency

- 🟡 **WARNING**: Very low score value found: 0.0300 — may indicate broken solver
- 🟡 **WARNING**: Very low PSNR value: 1.3 dB — may indicate failed reconstruction
- 🟡 **WARNING**: Very low PSNR value: 1.3 dB — may indicate failed reconstruction

### Forward Model

- 🟡 **WARNING**: Notation inconsistency: Forward operator denoted as both H and A
- 🟡 **WARNING**: Notation inconsistency: Operator denoted as both H and R

### Spec Ranges

- 🟡 **WARNING**: Negative physical quantity: -140 kV — verify sign is intended
- 🟡 **WARNING**: Negative physical quantity: -800 m — verify sign is intended

### Images & Links

- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/ct_challenge_public.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/ct_challenge_dev.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/scene_03/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/recon_dolce.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/recon_fbp.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/recon_fbpconv.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/recon_lpd.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/recon_pnp-admm.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_00/recon_pnp-drunet.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/recon_dolce.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/recon_fbp.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/recon_fbpconv.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/recon_lpd.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/recon_pnp-admm.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_01/recon_pnp-drunet.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/recon_dolce.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/recon_fbp.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/recon_fbpconv.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/recon_lpd.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/recon_pnp-admm.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_02/recon_pnp-drunet.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/recon_dolce.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/recon_fbp.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/recon_fbpconv.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/recon_lpd.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/recon_pnp-admm.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/ct/algorithms/scene_03/recon_pnp-drunet.png — verify it loads

### Physics Consistency

- 🔵 **INFO**: Signal shape [256, 256] from YAML not clearly shown on page

---
*Auto-generated by `benchmarks/learn/check_all_modalities.py`*
