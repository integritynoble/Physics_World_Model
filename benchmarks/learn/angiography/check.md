# Comprehensive Benchmark QA Check — Angiography

**URL:** https://pwm.platformai.org/benchmark/angiography
**HTTP Status:** 200 OK
**Check Date:** 2026-03-03 (comprehensive 6-point review)
**Reviewer:** Local server (automated + manual deep analysis)

---

## 1. Benchmark Page Errors

### 1.1 Cross-Source Discrepancies

The web page content was cross-referenced against local learning materials
(`benchmarks/learn/angiography/04_pwm_benchmark.md`,
`benchmarks/learn/angiography/03_reconstruction_algorithms.md`) and the
previous automated check (`check.md`). Several significant inconsistencies
were identified.

| # | Severity | Description | Evidence | Suggested Fix |
|---|----------|-------------|----------|---------------|
| 1 | **HIGH** | **Algorithm name mismatch between web page and prior automated check.** Web page shows Learned Primal-Dual, FBPConvNet, PnP-ADMM, FBP (all with "+ gradient"). Prior automated check lists X-Restormer, PnP-DnCNN, Dual-ResNet, TV-Regularized. These are entirely different algorithm sets with no overlap. | WebFetch extraction vs prior `check.md` line 19 | Determine the authoritative source. If the web page was recently updated, update the automated checker script to parse current names. If the web page is wrong, fix the server-side leaderboard. |
| 2 | **HIGH** | **Image dimension mismatch.** Web page states 1024x1024 pixels. Local learning material `04_pwm_benchmark.md` states Object=[512,512], Measurements=[512,512]. | Web page vs `04_pwm_benchmark.md` lines 27-29 | Reconcile: confirm the actual HDF5 data shape and update whichever source is incorrect. |
| 3 | **HIGH** | **Dataset source mismatch.** Web page references AAPM Low-Dose CT Grand Challenge (McCollough et al., Med. Phys. 2017) and IntrA (intracranial aneurysm 3DRA). Local learning material references XCAD (Ma et al., ICCV 2021) with a GitHub link to `Benny0323/XCAD`. These are fundamentally different datasets: AAPM is CT abdomen phantoms, IntrA is 3D brain vessel meshes, and XCAD is 2D coronary X-ray angiography frames. | Web page vs `04_pwm_benchmark.md` lines 33-38 | XCAD appears to be the most domain-appropriate dataset for X-ray angiography. Clarify which dataset is actually used in the HDF5 files and fix all references to be consistent. |
| 4 | **HIGH** | **Mismatch parameter contradiction.** Web page defines 3 mismatch parameters (Contrast Timing, Motion, Scatter) with specific ranges per tier. Local learning material `04_pwm_benchmark.md` line 45 states "No mismatch parameters defined for this modality." | Web page vs `04_pwm_benchmark.md` line 45 | Update `04_pwm_benchmark.md` to include the mismatch parameters, or remove them from the web page if they are not implemented. |
| 5 | **MEDIUM** | **Scoring formula inconsistency.** Web page uses composite: 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x (1 - residual_norm). Local learning material lists only PSNR as primary metric and SSIM as secondary, with no consistency term. | Web page vs `04_pwm_benchmark.md` lines 57-60 | Update `04_pwm_benchmark.md` to reflect the composite scoring formula actually used on the server. |
| 6 | **MEDIUM** | **Forward model notation mismatch.** Web page describes DSA subtraction: y_post - y_pre = delta_mu x t_vessel + n. Learning materials and automated check use a different spec notation: Pi(proj) -> D(g, eta_1). The latter looks like a generic projection operator, not DSA-specific. | Web page vs prior `check.md` line 27 | Align the notation. The DSA subtraction model is the domain-appropriate formulation. |
| 7 | **MEDIUM** | **Tier weighting formula not explicit on web page.** The page shows "Overall" scores but does not publish the tier-weighting formula (e.g., equal weights across Public/Dev/Hidden, or different weights). | Web page observation | Publish the composite weighting formula explicitly on the benchmark page for reproducibility. |
| 8 | **LOW** | **Sample count is small.** Only 3 scenes per tier (9 total). This is statistically limited for robust algorithm ranking; confidence intervals on PSNR differences may overlap. | Web page extraction | Consider expanding to at least 10-20 scenes per tier, as done for other PWM modalities (e.g., `spc_kronecker` uses 20). |
| 9 | **LOW** | **Oracle correction shows minimal recovery.** Gallery degradation analysis shows -0.6 dB loss from mismatch with ~0.0 dB oracle recovery, suggesting the mismatch correction pathway may not be functioning or the mismatch effect is negligible. | Web page gallery analysis | Investigate whether oracle correction is properly implemented for this modality. |

### 1.2 Page Functionality

| Check | Status |
|-------|--------|
| Main page loads (HTTP 200) | PASS |
| Title present: "X-ray Angiography" | PASS |
| Challenge Leaderboard section | PASS |
| Leaderboard entries: 4 methods | PASS |
| Gallery images (24/24 load) | PASS (per prior automated check) |
| Challenge public page loads | PASS |
| Challenge dev page loads | PASS |
| HDF5 references present | PASS |
| Compete / Contribute pages | PASS |
| Forward model reference | PASS |

---

## 2. Local Dataset Inspection

**NO LOCAL DATASET**

The directory `datasets/benchmark/angiography/` exists but is empty (no files).
No HDF5, NumPy, PNG, or other data files were found.

| Property | Value |
|----------|-------|
| Directory path | `datasets/benchmark/angiography/` |
| Files found | 0 |
| Total size | 0 bytes |
| HDF5 files | None |
| README/metadata | None |

**Impact:** Cannot run benchmarks locally without downloading data from the
server. No offline validation of data integrity, dimensions, or mismatch
parameter distributions is possible.

**Recommendation:** Add a download script or `dvc pull` step in the
documentation. At minimum, place a `README.md` in the empty directory
explaining how to obtain the data.

---

## 3. Public Dataset Source Assessment

### 3.1 Datasets Referenced

Three different datasets are referenced across sources, creating confusion:

| Dataset | Source | Domain | Format | Peer-Reviewed | Professor-Accepted |
|---------|--------|--------|--------|:-------------:|:-------------------:|
| **XCAD** (Ma et al., ICCV 2021) | Referenced in `04_pwm_benchmark.md` | Coronary X-ray angiography | 1,621 unlabeled 2D frames + synthetic masks | Yes (ICCV 2021) | Yes -- ICCV is top-tier CV venue |
| **AAPM Low-Dose CT** (McCollough et al., 2017) | Referenced on web page | CT abdomen | Full-dose / quarter-dose CT slices | Yes (Med. Phys. 2017) | Yes -- standard CT benchmark |
| **IntrA** (Yang et al., CVPR 2020) | Referenced on web page | Intracranial aneurysm 3DRA | 103 3D vessel meshes (no raw 2D images) | Yes (CVPR 2020) | Yes -- well-cited CVPR paper |

### 3.2 Assessment

- **XCAD** is the most appropriate dataset for this benchmark. It contains real
  clinical coronary X-ray angiography frames, directly matching the "X-ray
  Angiography" modality. It is MIT-licensed, accessible via GitHub, and the
  associated paper (Ma et al., ICCV 2021) has strong citations. Professors
  working in medical imaging would accept this as a legitimate data source.

- **AAPM** is a gold-standard CT dataset, but it is **CT**, not angiography.
  Using it here would be a domain mismatch (CT reconstruction vs. DSA vessel
  imaging). Its presence on the web page likely indicates a copy-paste error
  from another modality (e.g., the CT benchmark).

- **IntrA** provides 3D vessel mesh annotations but **no raw 2D X-ray
  images**, making it unsuitable as a reconstruction benchmark data source. It
  is useful for 3D segmentation but not for 2D DSA reconstruction.

### 3.3 Verdict

**XCAD is the correct dataset for this benchmark.** The web page references to
AAPM and IntrA appear erroneous and should be corrected. A professor reviewing
this benchmark would immediately flag the AAPM reference as a domain mismatch.

---

## 4. Algorithm Coverage Assessment

### 4.1 Current Leaderboard

The web page shows 4 algorithms (all appended with "+ gradient"):

| Rank | Algorithm | Overall | Public PSNR/SSIM | Dev PSNR/SSIM | Hidden PSNR/SSIM | Category |
|------|-----------|---------|------------------|---------------|-------------------|----------|
| 1 | Learned Primal-Dual + gradient | 0.722 | 34.40 / 0.964 | 27.52 / 0.870 | 26.57 / 0.847 | Unrolled DL |
| 2 | FBPConvNet + gradient | 0.717 | 34.58 / 0.965 | 27.87 / 0.878 | 25.26 / 0.810 | Post-processing DL |
| 3 | PnP-ADMM + gradient | 0.695 | 30.71 / 0.927 | 26.68 / 0.850 | 25.49 / 0.817 | Plug-and-Play |
| 4 | FBP + gradient | 0.630 | 25.64 / 0.822 | 24.98 / 0.802 | 22.71 / 0.720 | Classical |

**Notable:** The prior automated check.md listed entirely different algorithms
(X-Restormer, PnP-DnCNN, Dual-ResNet, TV-Regularized). This either indicates
the web page was updated after the automated check ran, or the automated
checker parsed a different section of the page.

### 4.2 Missing Algorithms

The following well-known algorithms from recent literature (2020-2025) are
absent from the leaderboard. These are commonly used in angiography / DSA /
CT reconstruction and should be considered for inclusion:

| Priority | Algorithm | Year | Type | Why Missing Is a Gap |
|----------|-----------|------|------|----------------------|
| **HIGH** | **U-Net / 3D U-Net** | 2015/2024 | Post-processing DL | Dominant architecture in medical image reconstruction; 3D U-Net outperforms 2D for DSA (Duan et al., Med. Phys. 2024). A baseline U-Net is expected. |
| **HIGH** | **Restormer / X-Restormer** | 2022/2023 | Transformer DL | CVPR 2022 oral; SOTA for image restoration. If previously on leaderboard (per old check.md), unclear why removed. |
| **HIGH** | **ADMM-Net / ADMM-TransNet** | 2022/2025 | Unrolled optimization | ADMM-TransNet achieves 44.6 dB PSNR at 128 views; a natural next step beyond PnP-ADMM. |
| **HIGH** | **Deep Unfolding / ISTA-Net++** | 2018/2021 | Unrolled optimization | Core unrolled architecture family; only Learned Primal-Dual represents this class. |
| **MEDIUM** | **Diffusion Models (Score-based)** | 2023/2025 | Generative DL | Emerging SOTA for inverse problems in imaging; two-stage DSA synthesis pipelines reported in 2025. |
| **MEDIUM** | **AngioNet** | 2021 | Segmentation + Reconstruction | Specifically designed for X-ray angiography vessel segmentation (Nature Sci. Rep. 2021). |
| **MEDIUM** | **FlowVM-Net** | 2024 | Temporal DL | Exploits temporal information in coronary angiography sequences; domain-specific advantage. |
| **MEDIUM** | **Total Variation (TV)** | Classical | Optimization | Standard regularization baseline; the old check listed TV-Regularized but it is absent from current web page. |
| **LOW** | **GAN-based DSA** | 2019/2024 | Generative DL | Self-supervised DSA generation approaches (e.g., adversarial learning for vessel segmentation). |
| **LOW** | **VWI Assistant** (multicenter DL) | 2025 | Clinical DL | Multi-sequence integrated platform with 79.9% clinical utilization rate (Nature npj Dig. Med. 2025). |

### 4.3 Algorithm Diversity Analysis

| Category | Present | Expected | Gap |
|----------|---------|----------|-----|
| Classical (FBP, TV) | 1 | 2 | Missing TV regularization |
| Plug-and-Play | 1 | 2 | Missing PnP-DnCNN or RED |
| Post-processing DL | 1 | 3 | Missing U-Net, Restormer |
| Unrolled DL | 1 | 3 | Missing ADMM-Net, ISTA-Net++ |
| Transformer DL | 0 | 1 | Missing Restormer/SwinIR |
| Diffusion/Generative | 0 | 1 | Missing score-based methods |
| Domain-specific | 0 | 2 | Missing AngioNet, FlowVM-Net |
| **Total** | **4** | **14** | **10 missing** |

---

## 5. Improvement Suggestions

### 5.1 Dataset Improvements

1. **Resolve the dataset identity crisis.** Three different datasets (XCAD,
   AAPM, IntrA) are referenced across sources. Settle on XCAD as the primary
   source and correct all references. Add a canonical `dataset_card.md` to the
   repository documenting provenance, license, and preprocessing steps.

2. **Increase scene count from 3 to at least 10 per tier.** With only 3 scenes
   per tier, leaderboard rankings are statistically fragile. XCAD has 1,621
   frames; a 10/10/10 split is easily achievable.

3. **Add temporal sequences.** DSA is inherently a temporal modality
   (pre-contrast vs. post-contrast frames over time). The current benchmark
   appears to use single-frame pairs. Adding multi-frame sequences would
   enable evaluation of temporal methods (FlowVM-Net, recurrent networks).

4. **Populate the local dataset directory.** The `datasets/benchmark/angiography/`
   directory is empty. Add download scripts, DVC tracking, or at minimum a
   README with download instructions.

5. **Validate image dimensions.** Confirm whether the actual data is 512x512
   or 1024x1024 and correct all documentation to match.

### 5.2 Algorithm Improvements

6. **Add a U-Net baseline.** This is the most widely used architecture in
   medical image reconstruction and its absence is conspicuous. A standard
   2D U-Net post-processing baseline would take minimal effort.

7. **Add a Transformer-based method.** Restormer or SwinIR would represent the
   modern Transformer class. If X-Restormer was previously on the leaderboard,
   restore it.

8. **Add TV regularization.** This is a standard optimization baseline that
   bridges the gap between FBP and learned methods. Its previous presence on
   the old leaderboard suggests it was removed without replacement.

9. **Add at least one domain-specific method.** AngioNet or FlowVM-Net would
   demonstrate that the benchmark is relevant to the angiography community,
   not just a generic inverse-problems testbed.

10. **Add a diffusion model.** Score-based diffusion priors for inverse
    problems are rapidly becoming standard (2024-2025). Including one would
    future-proof the leaderboard.

### 5.3 Infrastructure Improvements

11. **Fix the mismatch parameter documentation.** The web page defines 3
    mismatch parameters but the learning materials say "No mismatch parameters
    defined." This is confusing for users trying to understand the benchmark.

12. **Publish the scoring formula explicitly.** The composite formula
    (0.4 PSNR + 0.4 SSIM + 0.2 consistency) should be documented on the
    benchmark page and in the learning materials with mathematical notation.

13. **Publish the tier-weighting formula.** How the "Overall" score combines
    Public/Dev/Hidden tier scores is not transparent.

14. **Add confidence intervals or statistical significance tests.** With 3
    scenes per tier, rank differences of <0.01 in composite score (e.g.,
    Learned Primal-Dual at 0.722 vs FBPConvNet at 0.717) are likely not
    statistically significant.

15. **Investigate oracle correction.** The gallery shows ~0.0 dB recovery
    from oracle mismatch correction, which suggests either the correction
    is not implemented or the mismatch effect is negligible.

---

## 6. Action Items

| # | Priority | Action | Owner | Status |
|---|----------|--------|-------|--------|
| 1 | **CRITICAL** | Resolve dataset identity: confirm XCAD is the actual source; remove AAPM/IntrA references from web page | Backend / Data | TODO |
| 2 | **CRITICAL** | Fix image dimension mismatch (512x512 vs 1024x1024) across web page and learning materials | Backend / Docs | TODO |
| 3 | **CRITICAL** | Reconcile algorithm names: web page shows Learned Primal-Dual/FBPConvNet/PnP-ADMM/FBP but prior check listed X-Restormer/PnP-DnCNN/Dual-ResNet/TV-Regularized | Backend / QA | TODO |
| 4 | **HIGH** | Fix mismatch parameter contradiction: web page has 3 params, learning materials say "none defined" | Docs | TODO |
| 5 | **HIGH** | Fix scoring formula inconsistency: web page uses composite (0.4/0.4/0.2), docs list PSNR-only | Docs | TODO |
| 6 | **HIGH** | Increase scene count from 3 to >= 10 per tier for statistical robustness | Data | TODO |
| 7 | **HIGH** | Add U-Net baseline to leaderboard | Algorithms | TODO |
| 8 | **HIGH** | Add Restormer/Transformer baseline to leaderboard | Algorithms | TODO |
| 9 | **MEDIUM** | Populate local `datasets/benchmark/angiography/` with data or download script | Data / Infra | TODO |
| 10 | **MEDIUM** | Add TV regularization baseline to leaderboard | Algorithms | TODO |
| 11 | **MEDIUM** | Add at least one domain-specific method (AngioNet or FlowVM-Net) | Algorithms | TODO |
| 12 | **MEDIUM** | Publish composite scoring formula and tier-weighting formula on benchmark page | Frontend / Docs | TODO |
| 13 | **MEDIUM** | Investigate oracle correction (~0.0 dB recovery) for correctness | Backend | TODO |
| 14 | **LOW** | Add diffusion model baseline to leaderboard | Algorithms | TODO |
| 15 | **LOW** | Add temporal multi-frame sequences to dataset for DSA-specific evaluation | Data | TODO |
| 16 | **LOW** | Add confidence intervals to leaderboard rankings | Framework | TODO |

---

## Appendix: Key References

1. **XCAD Dataset:** Ma et al., "Self-Supervised Vessel Segmentation via
   Adversarial Learning," ICCV 2021.
   GitHub: https://github.com/Benny0323/XCAD

2. **IntrA Dataset:** Yang et al., "IntrA: 3D Intracranial Aneurysm Dataset
   for Deep Learning," CVPR 2020.
   GitHub: https://github.com/intra3d2019/IntrA

3. **AAPM Low-Dose CT:** McCollough et al., "Low-Dose CT Grand Challenge,"
   Med. Phys. 2017.

4. **FBPConvNet:** Jin et al., "Deep Convolutional Neural Network for Inverse
   Problems in Imaging," IEEE TIP 2017.

5. **Learned Primal-Dual:** Adler & Oktem, "Learned Primal-Dual
   Reconstruction," IEEE TMI 2018.

6. **PnP-ADMM:** Chan et al., "Plug-and-Play ADMM for Image Restoration,"
   IEEE TIP 2017.

7. **Restormer:** Zamir et al., "Restormer: Efficient Transformer for
   High-Resolution Image Restoration," CVPR 2022 (Oral).

8. **ADMM-TransNet:** Combining Convolution and Transformer for Sparse-View
   CT Reconstruction, 2025.

9. **DL-based DSA:** Duan et al., "Training of a deep learning based digital
   subtraction angiography method using synthetic data," Med. Phys. 2024.

10. **AngioNet:** Iyer et al., "AngioNet: a convolutional neural network for
    vessel segmentation in X-ray angiography," Sci. Rep. 2021.

11. **FlowVM-Net:** "Enhanced Vessel Segmentation in X-Ray Coronary
    Angiography Using Temporal Information Fusion," 2024.
    GitHub: https://github.com/wgyhhhh/FlowVM-Net

12. **Deep generative models in DSA:** Systematic review, Artif. Intell. Rev.,
    Springer Nature, 2026.

13. **VWI Assistant:** "Rapid vessel segmentation and reconstruction of head
    and neck angiograms," npj Digital Medicine, 2025.

---

*Comprehensive 6-point review on 2026-03-03. This review identified 3 CRITICAL
issues (dataset identity crisis, dimension mismatch, algorithm name mismatch),
3 HIGH issues (mismatch parameter contradiction, scoring formula inconsistency,
insufficient scene count), and 10 additional MEDIUM/LOW improvements. The
benchmark is functional but requires significant documentation reconciliation
before it can be considered publication-ready.*