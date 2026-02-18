# InverseNet Paper Polish Plan — ECCV 2026

> **Goal:** Polish `inversenet_paper.tex` for ECCV 2026 submission
> **Basis:** Existing paper with SPC, CASSI, CACTI (3 modalities, 11 methods, 27 scenes)
> **Scope:** Paper improvements only. Mention non-public test data and PWM targeting system rail briefly.

---

## 1. Current Paper Assessment

The paper (`inversenet_paper.tex`) is a solid draft with:
- Three-scenario protocol (I: Ideal, II: Baseline, III: Oracle)
- 11 methods across 3 modalities (SPC, CASSI, CACTI), 240+ experiments
- Clear findings: mask-awareness taxonomy, inverse performance-robustness relationship
- Well-structured quantitative tables

### Gaps for ECCV 2026

| # | Gap | Impact | Section |
|---|-----|--------|---------|
| G1 | No visual reconstruction examples | High — ECCV reviewers expect qualitative comparisons | Sec 4 |
| G2 | No mismatch severity ablation | High — only one mismatch level tested | Sec 4 |
| G3 | Missing recent references (2023-2025) | Medium — incomplete related work | Sec 2 |
| G4 | No recovery ratio vs. ideal PSNR scatter plot | Medium — the inverse relationship claim needs visualization | Sec 4.4 |
| G5 | No mention of non-public test data for future evaluation | Low — strengthens contribution | Sec 6 |
| G6 | No mention of PWM targeting system rail context | Low — positions paper in larger framework | Sec 1 or 6 |
| G7 | No supplementary material | Medium — ECCV allows supplementary | New |
| G8 | Abstract could be sharper on novelty | Medium — first sentence buries the lead | Abstract |
| G9 | No per-scene visual analysis (heatmap of rho per scene per method) | Medium — shows scene-dependent behavior | Sec 4 |
| G10 | Residual gap discussion needs a figure | Low — currently text-only | Sec 5 |

---

## 2. Plan: Paper Improvements

### 2.1 Add Visual Reconstruction Examples (G1) — HIGH PRIORITY

**What:** Add a figure showing reconstructed image/spectral/video patches for Scenario I vs II vs III, for one representative scene per modality.

**Figure layout (1 figure, 3 rows):**
```
Row 1 (CASSI): Ground truth | Scenario I (MST-L) | Scenario II (MST-L) | Scenario III (MST-L) | Error map II | Error map III
Row 2 (CACTI): Ground truth | Scenario I (ELP-Unfolding) | Scenario II | Scenario III | Error map II | Error map III
Row 3 (SPC):   Ground truth | Scenario I (HATNet) | Scenario II | Scenario III | Error map II | Error map III
```

- CASSI: Show one spectral band (e.g., band 14) of Scene 1 (KAIST)
- CACTI: Show one temporal frame (e.g., frame 4) of "kobe"
- SPC: Show full reconstructed image of "cameraman"
- Error maps: absolute difference, jet colormap, same scale per row

**Where:** New Figure 2, before the per-modality results tables. Move current Figure 1 up.

**Implementation:** Generate from existing `results/` reconstruction arrays using a new script `scripts/generate_visual_comparison.py`.

### 2.2 Add Mismatch Severity Ablation (G2) — HIGH PRIORITY

**What:** Sweep mismatch severity (mild / moderate / severe) for one representative method per modality, show how Delta_deg and rho change with severity.

**Ablation design:**

| Modality | Method | Mild | Moderate (paper default) | Severe |
|----------|--------|------|--------------------------|--------|
| SPC | HATNet | alpha=0.0005 | alpha=0.0015 | alpha=0.005 |
| CASSI | MST-L | dx=0.2, a1=2.01 | dx=0.5, a1=2.02 | dx=1.5, a1=2.05 |
| CACTI | ELP-Unfolding | 0.5x params | 1x params (default) | 2x params |

**Figure:** Line plot — x-axis: severity level, y-axis: PSNR for each scenario. Three subplots (one per modality). Shows how the gap between I and II widens with severity, and how III tracks.

**Where:** New Section 4.5 "Mismatch Severity Analysis" or fold into each modality subsection.

**Implementation:** Modify existing validation scripts to accept severity multiplier.

### 2.3 Add Recovery Ratio Scatter Plot (G4) — MEDIUM PRIORITY

**What:** Scatter plot of rho (y-axis) vs. Scenario I PSNR (x-axis) for all 11 methods across all 3 modalities. Color-coded by modality, shape-coded by method type (classical/mask-aware/mask-oblivious).

**Purpose:** Visualizes the paper's central finding — the inverse relationship between ideal performance and recovery ratio. One figure replaces several paragraphs of text.

**Where:** New Figure in Section 4.4 (Cross-Modality Analysis).

### 2.4 Update Related Work (G3) — MEDIUM PRIORITY

**Add references to:**
- DAUHST (Cai et al., CVPR 2022) — deep unfolding HSI transformer
- CST (Cai et al., NeurIPS 2022) — cross-stage spectral transformer
- PADUT (Li et al., CVPR 2023) — progressive attention deep unfolding transformer
- BiSRNet (2023) — bidirectional spectral reconstruction
- Mismatch-aware training references (if any exist in 2024-2025)
- Robust reconstruction references (model uncertainty, distributional robustness)
- Diffusion-based video compressive imaging methods (2024) if published
- STFormer (2022) — spatial-temporal transformer for video compressive sensing

**Where:** Section 2 (Related Work). Add a paragraph on "Robustness and mismatch-aware methods" if recent references exist.

### 2.5 Add Per-Scene Heatmap (G9) — MEDIUM PRIORITY

**What:** Heatmap figure showing rho per scene per method for CASSI (10 scenes x 4 methods) and CACTI (6 videos x 4 methods). Reveals which scenes are hardest to recover.

**Where:** Supplementary material, referenced from Section 4.

### 2.6 Polish Abstract (G8) — MEDIUM PRIORITY

**Current first sentence:** "Compressive imaging systems rely on accurate knowledge of the forward measurement operator for high-quality reconstruction."

**Proposed revision:** Lead with the problem and gap more directly. Sharpen the numbers. Example direction:
- Open with the mismatch problem (10-21 dB degradation exists but is ignored by benchmarks)
- State the contribution crisply (first cross-modality operator-mismatch benchmark)
- Report the most striking result (e.g., 20.58 dB loss that is 93% recoverable)

### 2.7 Mention Non-Public Test Data (G5) — LOW PRIORITY

**What:** Add 1-2 sentences in Section 6 (Conclusion / Future Directions) mentioning that InverseNet will include sealed non-public test sets for blind evaluation, preventing overfitting to known test scenes.

**Proposed text (for Conclusion):**
> "To prevent overfitting to known test scenes, future InverseNet rounds will include sealed non-public test sets generated after solver submission deadlines, enabling blind evaluation of calibration and reconstruction methods."

**No infrastructure needed** — just mention it.

### 2.8 Mention PWM Targeting System Rail (G6) — LOW PRIORITY

**What:** Add 1-2 sentences positioning InverseNet within the Physics World Model framework as a targeting system rail for computational imaging evaluation.

**Proposed text (for Introduction or Conclusion):**
> "InverseNet serves as the targeting system rail for the Physics World Model (PWM), providing a durable evaluation infrastructure for computational imaging. While reconstruction solvers are continuously replaced with improved methods, the three-scenario protocol, scoring formulas, and benchmark datasets remain fixed, ensuring fair longitudinal comparison."

**No infrastructure needed** — just mention it.

### 2.9 Add Supplementary Material (G7) — MEDIUM PRIORITY

**Contents:**
- Per-scene PSNR/SSIM tables for all three modalities (full detail)
- Mismatch severity ablation full results
- Additional visual reconstruction examples (all scenes)
- Per-scene recovery ratio heatmaps
- Implementation details (hyperparameters, runtime)
- Dataset generation code description

### 2.10 Add Residual Gap Figure (G10) — LOW PRIORITY

**What:** Bar chart showing Delta_res = PSNR_I - PSNR_III for all methods, grouped by modality. Highlights that CASSI has the largest residual gap (7.48 dB for MST-L) due to dispersion mismatch.

**Where:** Section 5 (Discussion), supports the "residual gap analysis" paragraph.

---

## 3. Implementation Order

| Priority | Task | Effort | Depends On |
|----------|------|--------|------------|
| 1 | G2: Mismatch severity ablation (run experiments) | 2-3 days | Existing scripts |
| 2 | G1: Visual reconstruction figure (generate from existing results) | 1 day | Existing results/ arrays |
| 3 | G4: Recovery ratio scatter plot | 0.5 day | Existing results |
| 4 | G8: Polish abstract | 0.5 day | — |
| 5 | G3: Update related work references | 1 day | Literature search |
| 6 | G9: Per-scene heatmap | 0.5 day | Existing results |
| 7 | G5: Mention non-public test data (2 sentences) | 10 min | — |
| 8 | G6: Mention PWM targeting system rail (2 sentences) | 10 min | — |
| 9 | G7: Compile supplementary material | 1 day | After G1, G2, G9 |
| 10 | G10: Residual gap figure | 0.5 day | Existing results |

**Total estimated effort:** ~7 days

---

## 4. Files to Create/Modify

| # | File | Action | Description |
|---|------|--------|-------------|
| 1 | `inversenet_paper.tex` | **Modify** | All paper text changes (abstract, related work, figures, mentions) |
| 2 | `scripts/generate_visual_comparison.py` | **Create** | Generate Scenario I/II/III visual patches + error maps |
| 3 | `scripts/run_severity_ablation.py` | **Create** | Mismatch severity sweep (mild/moderate/severe) |
| 4 | `scripts/generate_scatter_plot.py` | **Create** | rho vs ideal PSNR scatter plot |
| 5 | `scripts/generate_heatmaps.py` | **Create** | Per-scene rho heatmaps for supplementary |
| 6 | `figures/visual_comparison.png` | **Generate** | Scenario I/II/III reconstruction patches |
| 7 | `figures/severity_ablation.png` | **Generate** | Mismatch severity sweep plot |
| 8 | `figures/rho_scatter.png` | **Generate** | Recovery ratio vs ideal PSNR |
| 9 | `figures/residual_gap.png` | **Generate** | Residual gap bar chart |
| 10 | `inversenet_supplementary.tex` | **Create** | Supplementary material document |

---

## 5. ECCV 2026 Submission Checklist

- [ ] Paper fits within 14 pages + references (ECCV LNCS format)
- [ ] All figures are vector/high-res (300+ DPI)
- [ ] Abstract under 250 words
- [ ] All tables have captions above, figures have captions below
- [ ] References are complete and use ECCV/LNCS format (splncs04.bst)
- [ ] Supplementary material compiled separately
- [ ] Code/data release URL placeholder included
- [ ] Anonymous submission (remove author names for review)
- [ ] Non-public test data mentioned in future work
- [ ] PWM targeting system rail mentioned as context
