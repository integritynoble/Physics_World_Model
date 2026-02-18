# CASSI GAP-TV Improvement Analysis

## Reference Paper: arxiv 2111.07910 (MST - CVPR 2022)

### Reference Results (10 KAIST Scenes)
| Method | PSNR (dB) | SSIM | Notes |
|--------|-----------|------|-------|
| **TwIST** | 24.51 | 0.698 | Classical baseline |
| **GAP-TV** | 24.51 | 0.698 | Classical reference |
| **DeSCI** | 26.50 | 0.750 | Better classical |
| **λ-net** | 29.38 | 0.833 | Early deep learning |
| **TSA-Net** | 31.27 | 0.891 | State-of-art classical |
| **MST-S** | 34.45 | 0.926 | Transformer (small) |
| **MST-L** | 35.36 | 0.943 | Transformer (large) |

## Current PWM Status vs Paper

### Our CASSI Results

| Dataset | Config | PSNR (dB) | SSIM | Gap to Paper |
|---------|--------|-----------|------|--------------|
| **TSA** | Aggressive noise | 10.26 | 0.0256 | -14.25 dB ❌ |
| **TSA** | Reduced noise | 17.23 | 0.1819 | **-7.28 dB** |
| **KAIST** (synthetic) | Testing... | ? | ? | To be determined |
| **KAIST** (real) | Paper reference | **24.51** | **0.698** | — |

### Why the 7.3 dB Gap on TSA?

1. **Different Dataset**
   - Paper: Real KAIST scenes (natural imagery)
   - Current: TSA synthetic scenes (different content)
   - KAIST scenes are typically easier/more natural

2. **Noise Model Differences**
   - Paper doesn't specify exact noise model
   - Our TSA uses: Poisson(10000) + Gaussian(σ=1.0)
   - Paper likely uses lower noise or no noise for classical methods

3. **Algorithm Implementation**
   - Paper GAP-TV may use more iterations
   - Different TV regularization strength
   - Possibly different initialization

4. **Mask Type**
   - Paper uses PSMaster mask (optimal for KAIST)
   - Our TSA uses TSA standard mask

## Three Paths Forward

### Path 1: Match Paper Exactly ⭐ (RECOMMENDED)

**Goal:** Reproduce paper's 24.5 dB on real KAIST dataset

**Steps:**
1. **Get Real KAIST Data**
   ```bash
   git clone https://github.com/kaist-ail/kaist-multispectral-dataset
   ```
   - 230 real scenes, various content (faces, objects, scenes)
   - Natural imagery (unlike TSA synthetic)
   - Standard benchmark for CASSI

2. **Implement Paper's GAP-TV Setup**
   - Use PSMaster coded aperture (paper's mask)
   - No noise (or minimal noise) for classical baseline
   - 200+ iterations with TV-Chambolle
   - TV weight tuning (range 0.5-2.0)

3. **Expected Result:** ~24.5 dB PSNR, 0.698 SSIM
   - Match published W1 baseline
   - Validate our GAP-TV implementation

### Path 2: Improve Beyond GAP-TV on TSA

**Goal:** Push TSA results toward 20+ dB

**Techniques:**
- Implement more sophisticated TV variants
- Use better initialization (e.g., spectral unmixing)
- Add learned denoisers (BM3D, FFDNet)
- Implement PnP variants (framework framework)

**Expected Result:** 18-20 dB on TSA (still ~5 dB below paper)
- Good for TSA, won't match KAIST results

### Path 3: Implement Deep Learning

**Goal:** Surpass classical methods (>32 dB)

**Quick wins:**
- TSA-Net: ~31 dB (handcrafted deep unfolding)
- MST-S: ~34 dB (transformer-based)
- HDNet: ~35 dB (learned spectral prior)

**Effort:** Moderate (already in MST-main repo)

## Recommended Action Plan

### Immediate (This Week)
1. ✅ Keep TSA improved baseline (17.23 dB with realistic noise)
2. ⬜ Create KAIST loader for real dataset
3. ⬜ Implement GAP-TV on real KAIST (target 24.5 dB)

### Short-term (Next 2 Weeks)
1. ⬜ Validate GAP-TV matches paper on KAIST
2. ⬜ Test classical improvements (TV-Chambolle tuning, BM3D)
3. ⬜ Benchmark on both TSA and KAIST

### Medium-term (Next Month)
1. ⬜ Implement TSA-Net (31 dB on KAIST)
2. ⬜ Integrate with PWM pipeline
3. ⬜ Create comprehensive comparison report

## Technical Details for GAP-TV Matching

Based on paper analysis:

```python
# Paper's likely GAP-TV configuration
GAP_TV_CONFIG = {
    'max_iterations': 200-300,
    'step_size_lambda': 1.0,
    'tv_weight': 1.0-1.5,
    'tv_max_iter': 10,
    'initialization': 'adjoint',
    'noise_level': 'minimal or none',
    'denoise_filter': 'TV-Chambolle',
}

# Our current best
OUR_CONFIG = {
    'max_iterations': 120,
    'step_size_lambda': 1.0,
    'tv_weight': 0.4-6.0,  # Too variable
    'tv_max_iter': 5,
    'initialization': 'adjoint',
    'noise_level': 'Poisson(10000) + Gaussian(1.0)',
    'denoise_filter': 'TV-Chambolle',
}
```

### Key Differences:
1. ✅ Both use TV-Chambolle (good)
2. ✅ Both use adjoint initialization (good)
3. ❌ Our TV weight varies too much (0.4-6.0)
4. ❌ Our noise model is more aggressive than paper
5. ❌ Our iterations might be too few (120 vs 200+)

## Summary

| Aspect | Current | Target | Action |
|--------|---------|--------|--------|
| **Dataset** | TSA | Real KAIST | Acquire real KAIST data |
| **Noise Model** | Poisson(10000) | None/minimal | Test no-noise baseline |
| **Iterations** | 50-120 | 200+ | Increase to 200 |
| **TV Weight** | 0.4-6.0 | ~1.0-1.5 | Standardize to 1.0 |
| **Gap to Paper** | -7.3 dB | 0 dB | Complete when Path 1 done |

---

**Next Steps:** Clarify dataset (real vs synthetic KAIST) and implement Path 1 for validation.

