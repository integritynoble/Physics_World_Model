# Benchmark QA Check -- Compressed Ultrafast Photography (CUP)

**URL:** <https://pwm.platformai.org/benchmark/cup>
**Review Date:** 2026-03-03
**Reviewer:** Claude Opus 4.6 (automated)
**Modality ID:** `cup`
**Category:** Ultrafast Imaging | **Carrier:** Photon | **DAG:** M --> Sigma --> D

---

## 1. Benchmark Page Errors

### HIGH Severity

1. **Leaderboard algorithm mismatch between page and local config.**
   The live leaderboard shows 4 algorithms: AL-DL + gradient, PnP-FFDNet + gradient,
   CUP-Net + gradient, TwIST + gradient. The local config (`benchmarks/configs/cup.yaml`)
   lists only 2 solvers: Adjoint (traditional_cpu) and PnP-ADMM (best_quality). Neither
   Adjoint nor PnP-ADMM appears on the leaderboard; conversely, none of the 4 leaderboard
   algorithms (AL-DL, PnP-FFDNet, CUP-Net, TwIST) appear in the local config. The
   `modify_plan.md` documents 4 algorithms (TwIST, PnP-FFDNet, CUP-Net, AL-DL) but the
   YAML was never updated to match.

2. **Mismatch parameter values differ between page and config.**
   The live page shows perturbed values of `dmd_encoding_error=0.4`,
   `streak_sweep_calibration=1.0`, `temporal_spatial_coupling=2.0`. The local YAML config
   defines ranges [0, 2], [-5, 5], [0, 10] respectively. The page's "perturbed" column
   appears to show a single example point, not the full range -- this is confusing and
   could mislead users into thinking those are the only mismatch values used.

3. **Scoring formula inconsistency with local metrics config.**
   The live page shows a 3-component scoring formula:
   `0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * (1 - ||y - Hx||/||y||)` (40/40/20 weighting).
   However, the local config (`cup.yaml`) lists only `psnr` and `ssim` as metrics with
   `psnr` as primary. The consistency term (measurement fidelity, 20% weight) is absent
   from the local config entirely. The local `04_pwm_benchmark.md` learning doc also
   omits the consistency term.

### MEDIUM Severity

4. **Dataset source mismatch: DAVIS 2017 vs. shepp_logan.**
   The live page cites "DAVIS 2017 (Pont-Tuset et al., arXiv 2017)" as the dataset source,
   but the local config shows `synthetic_generator: shepp_logan` with empty `dataset_id`
   and `dataset_url`. DAVIS 2017 is a video object segmentation dataset (90 natural video
   sequences), while Shepp-Logan is a 2D phantom. These are fundamentally different data --
   the page may have been updated to use DAVIS while the config still points to the
   synthetic fallback.

5. **Signal dimensions: 64x64 in config vs. 128/256/512 in expanded config.**
   The base config (`cup.yaml`) specifies `x_shape: [64, 64]` and `y_shape: [64, 64]`,
   but the expanded config defines Small=128x128, Standard=256x256, Large=512x512. The
   learning doc (04) also states [64, 64]. It is unclear which resolution the live
   benchmark actually uses.

6. **Missing "+ gradient" explanation on leaderboard.**
   All 4 leaderboard entries append "+ gradient" to the algorithm name (e.g., "AL-DL +
   gradient", "TwIST + gradient"). The page does not explain what "+ gradient" means --
   likely a gradient-descent-based spec correction step during reconstruction. This
   terminology is not documented in the learning materials.

7. **Tier sample count: page says 5 per tier (15 total) vs. expanded config says 192.**
   The live page indicates 5 scenes per tier (public/dev/hidden = 15 total). The expanded
   config tallies B1=12, B2=60, B3=60, B4=60 = 192 grand total. These are different
   benchmarking axes (B1-B4 vs. tiers) but the relationship is never clarified.

### LOW Severity

8. **No citation or DOI for the benchmark itself.**
   The page credits Yao et al. Photon. Res. 2021 for the top method but provides no
   citable reference for the PWM CUP benchmark itself.

9. **Empty fields in config YAML.**
   `reference_psnr: null`, `expected_psnr_range: null`, solver references are empty
   strings, `data_source.citation: ''`, `data_source.license: ''`. These should be
   populated for a published benchmark.

10. **Learning doc solver table references tiers that do not exist.**
    `03_reconstruction_algorithms.md` mentions `famous_dl` and `small_gpu` solver tiers in
    the "Algorithm Selection Guide" but these tiers are not defined in `cup.yaml`.

---

## 2. Local Dataset Inspection

**Local path:** `datasets/benchmark/cup/` -- **DOES NOT EXIST**

No local dataset directory was found. The benchmark config indicates:
- `data_source.fallback: generated` with `synthetic_generator: shepp_logan`
- `data_source.dataset_id: ''` (empty)
- `data_source.dataset_url: ''` (empty)

The live page references DAVIS 2017 and HDF5 files hosted on GCS (Google Cloud Storage),
and the automated check script confirms both public and dev HDF5 files are accessible
on GCS. However, there is no local mirror or download script for the CUP dataset.

**What exists locally:**
- `benchmarks/configs/cup.yaml` -- base config (2 solvers, shepp_logan fallback)
- `benchmarks/expanded_configs/cup_expanded.yaml` -- expanded config (192 cases)
- `benchmarks/learn/cup/` -- 5 learning docs + README + modify_plan
- `docs/modality_benchmarks/cup.md` -- benchmark spec doc (B1-B4 tiers)

---

## 3. Public Dataset Source Assessment

### DAVIS 2017 (claimed on live page)
- **What it is:** DAVIS (Densely Annotated VIdeo Segmentation) 2017 is a video object
  segmentation benchmark with 90 natural video sequences (Pont-Tuset et al., arXiv:1704.00675).
- **Appropriateness for CUP:** MODERATE. DAVIS provides natural video frames that can
  serve as ground-truth spatiotemporal datacubes for simulated CUP acquisition. However,
  DAVIS videos are at ~24fps with ~480p resolution -- vastly different from real CUP's
  trillion-fps ultrafast regime. The temporal dynamics in DAVIS (people walking, cars
  driving) bear no physical resemblance to ultrafast phenomena (laser ablation, photon
  propagation, plasma dynamics).
- **License:** CC-BY-NC 4.0 (DAVIS 2017). Acceptable for academic benchmarks.

### Shepp-Logan Phantom (configured locally)
- **What it is:** A 2D mathematical phantom commonly used in CT/MRI reconstruction.
- **Appropriateness for CUP:** LOW. Shepp-Logan is a static 2D phantom with no temporal
  structure. CUP reconstructs 3D datacubes (x, y, t). Using a 2D phantom eliminates the
  temporal reconstruction challenge entirely.

### Assessment
Neither dataset is ideal for a CUP benchmark. The gold standard would be:
1. Experimentally captured CUP data from real ultrafast events
2. Physics-based synthetic ultrafast scenes (e.g., simulated photon propagation,
   wavefront dynamics, laser-matter interactions)

---

## 4. Algorithm Coverage Assessment

### Currently on Leaderboard (4 algorithms)

| Rank | Algorithm | Type | PSNR (Public/Dev/Hidden) | SSIM (Public/Dev/Hidden) | Overall | Reference |
|------|-----------|------|--------------------------|--------------------------|---------|-----------|
| 1 | AL-DL + gradient | Hybrid (model + DL) | 32.17 / 26.24 / 23.94 | 0.944 / 0.839 / 0.766 | 0.692 | Yao et al., Photon. Res. 2021 |
| 2 | PnP-FFDNet + gradient | Plug-and-Play | 26.44 / 22.83 / 22.26 | 0.844 / 0.724 / 0.701 | 0.612 | Yuan et al., 2020 |
| 3 | CUP-Net + gradient | Deep Learning (U-Net) | 30.73 / 21.98 / 20.27 | 0.927 / 0.689 / 0.612 | 0.610 | Parker et al., 2021 |
| 4 | TwIST + gradient | Classical CS | 23.38 / 20.40 / 20.50 | 0.746 / 0.618 / 0.622 | 0.555 | Bioucas-Dias & Figueiredo, IEEE TIP 2007 |

### Key Observations
- **Large public-to-hidden PSNR drop for CUP-Net:** 30.73 -> 20.27 dB (10.46 dB drop),
  indicating severe overfitting to the nominal forward model. CUP-Net performs well when
  there is no mismatch but collapses under severe mismatch.
- **AL-DL is most robust:** Only 8.23 dB drop (32.17 -> 23.94), suggesting the
  augmented-Lagrangian framework provides better mismatch tolerance.
- **TwIST is most stable but lowest quality:** Only 2.88 dB drop (23.38 -> 20.50),
  classical methods are inherently more robust to mismatch but limited in peak quality.

### Missing Algorithms (should be added)

| Algorithm | Type | Why Include | Reference |
|-----------|------|-------------|-----------|
| **GAP-TV** | Classical (total variation) | Widely used CUP baseline; already in modify_plan but not on leaderboard | Yuan, 2016 |
| **D-HAN (Deep High-dim Adaptive Net)** | Deep Unfolding | State-of-the-art deep unfolding for CUP; combines physics model with U-Net | -- |
| **ML-ADMM** | Unsupervised (manifold learning) | 2024 method using manifold learning + ADMM; novel unsupervised approach | JOSA A, 2024 |
| **LI-CUP Deep Unfolding** | Deep Unfolding | 2025 method for line-integral CUP; latest deep unfolding approach | Opt. Lett. 2025 |
| **Self-supervised NN** | Self-supervised | Untrained self-supervised approach (no training data needed) | Chinese Sci. Bull. 2024 |
| **STFormer** | Transformer | State-of-the-art for snapshot compressive imaging; applicable to CUP | ECCV 2022 / updated 2024 |
| **RevSCI** | Reversible DL | Memory-efficient reversible SCI network; strong PSNR/SSIM | CVPR 2021 |
| **PnP-ADMM** | Plug-and-Play | Already in local config as best_quality solver but not on leaderboard | Multiple |
| **Streak-Net + gradient** | Deep Learning | Listed in old automated check (line 19) but NOT on live leaderboard | -- |

### Coverage Gaps
- **No transformer-based method** on the leaderboard (STFormer, CST, MST families)
- **No deep unfolding method** (D-HAN, ADMM-Net, ISTA-Net++)
- **No self-supervised / untrained method** (important for real-world CUP where training
  data is scarce)
- **No diffusion-based method** (emerging class for inverse problems)
- **Streak-Net** is mentioned in the old check.md but absent from the current leaderboard

---

## 5. Improvement Suggestions

### Data & Benchmarking

1. **Replace Shepp-Logan with physics-based synthetic ultrafast scenes.**
   Generate temporal datacubes simulating real ultrafast phenomena: photon propagation
   in scattering media, laser ablation plumes, plasma filament formation, shock wave
   propagation. This would test algorithms on temporally structured data that matches
   real CUP use cases.

2. **Add real experimental CUP data tier.**
   Partner with CUP labs (e.g., Liang group at Caltech/INRS, Gao group at UCLA) to
   obtain calibrated experimental data with known ground truth. Even 2-3 real scenes
   would dramatically increase benchmark credibility.

3. **Clarify the relationship between B1-B4 axes and Public/Dev/Hidden tiers.**
   The expanded config defines 192 cases across 4 benchmark axes (B1-B4) while the
   live page shows 15 scenes across 3 tiers. Document how these map to each other.

4. **Increase scene count from 5 per tier to at least 20.**
   With only 5 scenes per tier, statistical significance of PSNR/SSIM differences is
   questionable. Consider bootstrapping or using more scenes.

### Algorithm & Scoring

5. **Add the consistency term to the local config.**
   The live page's scoring formula includes `0.2 * (1 - ||y - Hx||/||y||)` but this
   is not reflected locally. Implement and document this metric.

6. **Add at least 3 more algorithms from different families:**
   - One transformer-based (STFormer or CST)
   - One deep unfolding (D-HAN or ADMM-Net)
   - One self-supervised (untrained NN for CUP)

7. **Document what "+ gradient" means.**
   Add explanation to the benchmark page and learning materials that "+ gradient" refers
   to the gradient-based spec correction step in the PWM mismatch framework.

### Config & Documentation

8. **Synchronize cup.yaml with the live leaderboard.**
   Add TwIST, PnP-FFDNet, CUP-Net, and AL-DL as solver entries. Remove or supplement
   the Adjoint and PnP-ADMM entries that do not appear on the leaderboard.

9. **Populate empty config fields.**
   Fill in `reference_psnr`, `expected_psnr_range`, `data_source.citation`,
   `data_source.license`, and solver `reference` fields.

10. **Fix 03_reconstruction_algorithms.md.**
    Remove references to nonexistent `famous_dl` and `small_gpu` tiers, or define
    them in the YAML.

---

## 6. Action Items

| Priority | Action | Owner | Effort |
|----------|--------|-------|--------|
| P0 (Critical) | Sync `cup.yaml` solvers with live leaderboard (add TwIST, PnP-FFDNet, CUP-Net, AL-DL) | Config | 1h |
| P0 (Critical) | Add consistency metric (`measurement_fidelity`) to local metrics config | Metrics | 2h |
| P0 (Critical) | Resolve dataset source: update config to reference DAVIS 2017 OR update page to reflect shepp_logan | Data | 1h |
| P1 (High) | Replace shepp_logan with physics-based ultrafast synthetic generator | Data | 1-2 weeks |
| P1 (High) | Clarify signal dimensions (64x64 vs 128/256/512) across all docs | Docs | 2h |
| P1 (High) | Document "+ gradient" spec correction in learning materials and page | Docs | 1h |
| P1 (High) | Populate all empty YAML fields (citations, licenses, references, thresholds) | Config | 2h |
| P2 (Medium) | Add transformer-based algorithm (STFormer) to leaderboard | Algorithms | 1 week |
| P2 (Medium) | Add deep unfolding algorithm (D-HAN) to leaderboard | Algorithms | 1 week |
| P2 (Medium) | Add self-supervised method to leaderboard | Algorithms | 1 week |
| P2 (Medium) | Clarify B1-B4 vs. Public/Dev/Hidden tier mapping in documentation | Docs | 2h |
| P2 (Medium) | Increase scene count from 5 to 20+ per tier | Data | 1 week |
| P3 (Low) | Fix `03_reconstruction_algorithms.md` references to nonexistent tiers | Docs | 30min |
| P3 (Low) | Add citable DOI/reference for the PWM CUP benchmark itself | Docs | 1h |
| P3 (Low) | Investigate Streak-Net (in old check but not on leaderboard) | Algorithms | 2h |

---

## Appendix: Key References

1. **CUP Original:** Gao, L., Liang, J., Li, C. & Wang, L. V. "Single-shot compressed
   ultrafast photography at one hundred billion frames per second." *Nature* 516, 74-77 (2014).

2. **AL-DL (Rank 1):** Yao, Y. et al. "High-fidelity image reconstruction for compressed
   ultrafast photography via an augmented-Lagrangian and deep-learning hybrid algorithm."
   *Photonics Research* 9(2), B30-B37 (2021).
   <https://opg.optica.org/prj/fulltext.cfm?uri=prj-9-2-B30&id=446779>

3. **CUP-Net (Rank 3):** Parker, M. "CUP-Net: Compressed Ultrafast Photography Using
   Convolutional Neural Networks." Dartmouth ENGS 88, 2021.
   <https://digitalcommons.dartmouth.edu/engs88/15/>

4. **TwIST (Rank 4):** Bioucas-Dias, J. M. & Figueiredo, M. A. T. "A new TwIST: Two-step
   iterative shrinkage/thresholding algorithms for image restoration." *IEEE Trans. Image
   Process.* 16(12), 2992-3004 (2007).

5. **Deep Learning for CUP:** Bai, Y. et al. "Deep-learning-based image reconstruction for
   compressed ultrafast photography." *Optics Letters* 45(16), 4400-4403 (2020).
   <https://opg.optica.org/ol/abstract.cfm?uri=ol-45-16-4400>

6. **ML-ADMM (2024):** "Image reconstruction for compressed ultrafast photography based on
   manifold learning and the alternating direction method of multipliers." *JOSA A* 41(8),
   1585 (2024). <https://opg.optica.org/josaa/abstract.cfm?uri=josaa-41-8-1585>

7. **LI-CUP (2025):** "Line integral compressed ultrafast photography for large time-scale
   measurements." *Optics Letters* 50(6), 1799 (2025).
   <https://opg.optica.org/ol/abstract.cfm?uri=ol-50-6-1799>

8. **CUP Tutorial:** "Tutorial on compressed ultrafast photography." *PMC* (2024).
   <https://pmc.ncbi.nlm.nih.gov/articles/PMC10826888/>

9. **DI-CUP (2024):** Yao, Y. et al. "Discrete Illumination-Based Compressed Ultrafast
   Photography for High-Fidelity Dynamic Imaging." *Advanced Science* (2024).
   <https://advanced.onlinelibrary.wiley.com/doi/10.1002/advs.202403854>

10. **Self-supervised CUP (2024):** "Realizing high-fidelity image reconstruction for
    compressed ultrafast photography with an untrained self-supervised neural network-based
    algorithm." *Chinese Science Bulletin* (2024).
    <https://www.sciengine.com/CSB/doi/10.1360/TB-2024-0038>

11. **DAVIS 2017:** Pont-Tuset, J. et al. "The 2017 DAVIS Challenge on Video Object
    Segmentation." arXiv:1704.00675 (2017). <https://arxiv.org/abs/1704.00675>

12. **PnP-SCI:** Yuan, X. et al. "Plug-and-play Algorithms for Large-scale Snapshot
    Compressive Imaging." *CVPR 2020 (Oral)*.
    <https://github.com/liuyang12/PnP-SCI>

---

*Comprehensive 6-point review on 2026-03-03. Covers benchmark page accuracy, local dataset
state, public data source validity, algorithm coverage (4 current + 9 missing), 10
improvement suggestions, and 15 prioritized action items across config, data, algorithms,
and documentation.*