# Quality Check: `sim`

**Checked:** 2026-03-03T14:12:55.582339+00:00
**URL:** https://pwm.platformai.org/benchmark/sim

## Status: WARN

| Category | Count |
|----------|-------|
| Passed | 25 |
| Warnings | 2 |
| Errors | 0 |

## Warnings

- [ ] PSNR 2.0 dB < 5 (unrealistically low)
- [ ] PSNR 4.8 dB < 5 (unrealistically low)

## Passed Checks

- [x] Main page loads (HTTP 200)
- [x] Page title: SIM — Physics World Model
- [x] Challenge Leaderboard section present
- [x] Leaderboard has 4 entries
- [x] Spec notation present: S(grating) → C(PSF) → Σ_φ → D(g, η₃)
- [x] Description: Structured Illumination Microscopy
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
- [x] Learn file README.md exists (1444 bytes)
- [x] Learn file 01_physics_fundamentals.md exists (3146 bytes)
- [x] Learn file 02_forward_model.md exists (2738 bytes)
- [x] Learn file 03_reconstruction_algorithms.md exists (2666 bytes)
- [x] Learn file 04_pwm_benchmark.md exists (2562 bytes)
- [x] Learn file 05_hands_on_tutorial.md exists (3545 bytes)
- [x] Compete page loads (HTTP 200)
- [x] Contribute page loads (HTTP 200)
