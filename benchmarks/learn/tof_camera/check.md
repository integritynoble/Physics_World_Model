# Quality Check: `tof_camera`

**Checked:** 2026-03-03T14:11:16.168025+00:00
**URL:** https://pwm.platformai.org/benchmark/tof_camera

## Status: WARN

| Category | Count |
|----------|-------|
| Passed | 25 |
| Warnings | 2 |
| Errors | 0 |

## Warnings

- [ ] PSNR 0.4 dB < 5 (unrealistically low)
- [ ] PSNR 2.3 dB < 5 (unrealistically low)

## Passed Checks

- [x] Main page loads (HTTP 200)
- [x] Page title: ToF Camera — Physics World Model
- [x] Challenge Leaderboard section present
- [x] Leaderboard has 4 entries
- [x] Spec notation present: P(modulated) → Σ(correlation) → D(g, η₁)
- [x] Description: Time-of-Flight Depth Camera
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
- [x] Learn file README.md exists (1442 bytes)
- [x] Learn file 01_physics_fundamentals.md exists (2586 bytes)
- [x] Learn file 02_forward_model.md exists (2452 bytes)
- [x] Learn file 03_reconstruction_algorithms.md exists (1816 bytes)
- [x] Learn file 04_pwm_benchmark.md exists (2308 bytes)
- [x] Learn file 05_hands_on_tutorial.md exists (3547 bytes)
- [x] Compete page loads (HTTP 200)
- [x] Contribute page loads (HTTP 200)
