# Benchmark QA Check — cassi

**URL:** https://pwm.platformai.org/benchmark/sd_cassi
**HTTP Status:** 200
**Page Size:** 165,938 bytes
**Check Date:** 2026-03-03 14:05 UTC

## Summary

| Severity | Count |
|----------|-------|
| WARNING | 4 |
| INFO | 101 |

## Issues by Category

### Leaderboard Consistency

- 🟡 **WARNING**: Suspiciously high score: 0.9940 — possible data leak or evaluation bug
- 🟡 **WARNING**: Suspiciously high score: 0.9940 — possible data leak or evaluation bug
- 🟡 **WARNING**: Very low PSNR value: 4.9 dB — may indicate failed reconstruction

### Forward Model

- 🟡 **WARNING**: Notation inconsistency: Forward operator denoted as both H and A

### Images & Links

- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/sd_cassi_challenge_public.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/challenge-data/v1.0/sd_cassi_challenge_dev.h5 — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_00/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_01/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_02/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_03/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_04/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_05/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_06/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_07/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_08/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/measurement_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/recon_I.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/recon_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/measurement_II.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/scene_09/recon_III.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_00/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_00/recon_ssr-l.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_01/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_01/recon_ssr-l.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_02/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_02/recon_ssr-l.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_03/gt.png — verify it loads
- 🔵 **INFO**: GCS image reference: /gcs/img/benchmark_gallery/sd_cassi/algorithms/scene_03/recon_ssr-l.png — verify it loads

### Algorithm Names

- 🔵 **INFO**: Solver 'GAP+HSI-SDeCNN' from registry not found on page

---
*Auto-generated by `benchmarks/learn/check_all_modalities.py`*
