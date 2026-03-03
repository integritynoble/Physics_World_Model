# Quality Check: `structured_light`

**Checked:** 2026-03-03T14:11:19.760493+00:00
**URL:** https://pwm.platformai.org/benchmark/structured_light

## Status: WARN

| Category | Count |
|----------|-------|
| Passed | 25 |
| Warnings | 2 |
| Errors | 0 |

## Warnings

- [ ] PSNR 0.8 dB < 5 (unrealistically low)
- [ ] PSNR 2.5 dB < 5 (unrealistically low)

## Passed Checks

- [x] Main page loads (HTTP 200)
- [x] Page title: Structured Light — Physics World Model
- [x] Challenge Leaderboard section present
- [x] Leaderboard has 4 entries
- [x] Spec notation present: S(pattern) → Π(triangulation) → D(g, η₁)
- [x] Description: Structured-Light Depth Camera
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
- [x] Learn file README.md exists (1456 bytes)
- [x] Learn file 01_physics_fundamentals.md exists (2692 bytes)
- [x] Learn file 02_forward_model.md exists (2475 bytes)
- [x] Learn file 03_reconstruction_algorithms.md exists (1844 bytes)
- [x] Learn file 04_pwm_benchmark.md exists (2352 bytes)
- [x] Learn file 05_hands_on_tutorial.md exists (3602 bytes)
- [x] Compete page loads (HTTP 200)
- [x] Contribute page loads (HTTP 200)
