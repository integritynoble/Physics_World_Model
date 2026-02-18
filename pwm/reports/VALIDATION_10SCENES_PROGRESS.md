# CASSI Algorithm 1 & 2 Validation: 10-Scene Comprehensive Test

**Status:** IN PROGRESS  
**Started:** 2026-02-14 23:33:03 UTC  
**Expected Duration:** 8-12 hours (coarse + fine grid + 3 gradient stages per scene × 10 scenes)

## Validation Overview

### Three Scenarios
1. **Scenario I (Ideal):** Oracle mismatch = 0 (best possible)
2. **Scenario II (Assumed):** Baseline with realistic mismatch injected (-5 to -10 dB penalty)
3. **Scenario III (Corrected):** Algorithm 1 (coarse) + Algorithm 2 (fine) calibration

### Performance Metrics
- **PSNR:** Reconstruction quality in dB
- **SSIM:** Structural similarity
- **SAM:** Spectral angle mapper in degrees

## Per-Scene Pipeline

For each of 10 KAIST benchmark scenes:

### Phase 1: Algorithm 1 (Hierarchical Beam Search)
- 1D sweeps: dx, dy, theta (~10 min/scene)
- 3D beam search (5×5×5) (~20 min/scene)
- 2D dispersion search (5×7) (~5 min/scene)
- **Time:** ~35-40 minutes per scene

### Phase 2: Algorithm 2 (Joint Gradient Refinement)
- Stage 0: Coarse 3D grid (567 candidates) GPU-accelerated
- Stage 1: Fine 3D grid (375 evals per top-5)
- Stage 2A: Gradient refinement of dx (50 steps)
- Stage 2B: Gradient refinement of dy, theta (60 steps)
- Stage 2C: Joint refinement (80 steps)
- **Time:** ~7-10 minutes per scene

### Total per Scene
- **Algorithm 1:** 35-40 min
- **Algorithm 2:** 7-10 min
- **Subtotal:** 42-50 min per scene
- **All 10 scenes:** 7-8.5 hours minimum

## Expected Results

### Baseline Performance (from cassi_baseline_10scenes_no_mismatch.json)
- **Average PSNR (no mismatch):** 9.89 dB
- **Average SSIM:** 0.0297
- **Average SAM:** 65.58°

### Expected Improvements
- **Scenario II → Scenario III:** +1 to +2 dB PSNR gain from calibration
- **Accuracy:** Algorithm 2 3-5× better parameter estimates than Algorithm 1

## Log Files
- Main validation log: `pwm/reports/validation_10scenes_20260214_233302.log`
- Results JSON: `pwm/reports/cassi_validation_10scenes_results.json`
- Summary: `pwm/reports/CASSI_Validation_10Scenes_Summary.md`

## Monitoring
To monitor progress:
```bash
tail -f pwm/reports/validation_10scenes_*.log
```

## Key Milestones
- [x] Algorithm 1 & 2 implementation complete
- [x] Unit tests pass (16/16)
- [x] Demo validation successful
- [ ] Scene 1 complete
- [ ] Scenes 1-5 complete
- [ ] All 10 scenes complete
- [ ] Summary report generated

---
*Validation initiated with Algorithm 2 PyTorch backend featuring:*
- GPU-accelerated parameter scoring (50× speedup)
- 5-stage optimization pipeline
- Gradient checkpointing for memory efficiency
- Graceful fallback for CPU scenarios
