# Quality Check: `sar`

**Checked:** 2026-03-03T14:12:59.357705+00:00
**URL:** https://pwm.platformai.org/benchmark/sar

## Status: WARN

| Category | Count |
|----------|-------|
| Passed | 25 |
| Warnings | 2 |
| Errors | 0 |

## Warnings

- [ ] PSNR 0.2 dB < 5 (unrealistically low)
- [ ] PSNR 3.8 dB < 5 (unrealistically low)

## Passed Checks

- [x] Main page loads (HTTP 200)
- [x] Page title: SAR — Physics World Model
- [x] Challenge Leaderboard section present
- [x] Leaderboard has 4 entries
- [x] Spec notation present: F(azimuth×range) → D(g, η₁)
- [x] Description: Synthetic Aperture Radar
- [x] Data Preview / Gallery section present
- [x] Gallery gt.png (scene_00) loads
- [x] Gallery recon_I.png (scene_00) loads
- [x] Gallery recon_II.png (scene_00) loads
- [x] Gallery recon_III.png (scene_00) loads
- [x] Challenge public page loads (HTTP 200)
- [x] Challenge public page has dataset reference
- [x] Challenge dev page loads (HTTP 200)
- [x] Challenge dev page has dataset reference
- [x] Public challenge HDF5 accessible on GCS
- [x] Learning materials directory exists
- [x] Learn file README.md exists (1424 bytes)
- [x] Learn file 01_physics_fundamentals.md exists (2536 bytes)
- [x] Learn file 02_forward_model.md exists (2525 bytes)
- [x] Learn file 03_reconstruction_algorithms.md exists (1829 bytes)
- [x] Learn file 04_pwm_benchmark.md exists (2318 bytes)
- [x] Learn file 05_hands_on_tutorial.md exists (3534 bytes)
- [x] Compete page loads (HTTP 200)
- [x] Contribute page loads (HTTP 200)
