InverseNet: Benchmarking Operator Mismatch and Calibration Across Compressive Imaging Modalities
arXiv Submission Package

CONTENTS:
=========

Main Paper:
-----------
inversenet_paper.tex          Main paper (19 pages)
inversenet_supplementary.tex  Supplementary material (15 pages)

Support Files:
--------------
llncs.cls                     LLNCS document class
splncs04.bst                  LLNCS bibliography style
figures/                      All figures (PDF format, ~2.5 MB)

COMPILATION:
============

To compile locally:
  pdflatex inversenet_paper.tex
  pdflatex inversenet_supplementary.tex

(Bibliography is inline in both documents; no external .bib file needed)

CHANGES IN THIS VERSION:
========================

This version incorporates improvements addressing pre-submission review:

1. Scenario IV CASSI/CACTI: Explicit caveat that grid search covers only 2/8 
   mismatch parameters; remaining 6 held at ground-truth values.

2. Table 1: Corrected bold notation to distinguish best absolute recovery 
   (Δrec) from best recovery ratio (ρ).

3. Abstract: Tightened from ~230 to ~150 words; added Spearman correlation 
   (r_s = -0.71, p < 0.01) for inverse performance–robustness relationship.

4. Cross-modality table: Added explicit clarification that metrics use SSIM 
   basis; cross-references to PSNR-based values in main tables.

5. Real hardware validation: Added mechanistic justification for excluding 
   deep learning methods (operator-conditioned networks have nonlinear 
   mask dependence).

6. CACTI real data: Explained 256× residual gap between GAP-TV and PnP-FFDNet 
   (difference in reconstruction mechanism).

7. PnP-HSICNN citation: Corrected to consistently cite zheng2021pnp.

8. Limitations: Explicitly named methods missing from benchmark (DAUHST, CST, 
   DiffSCI); acknowledged N=4 per-modality sample size limitation.

9. Data availability: Removed placeholder URL (solveeverything.org); kept 
   GitHub repository link.

10. Bibliography: Removed 4 uncited entries (elser2003phase, jumper2021alphafold, 
    liu2024udc, wang2022stformer).

All 34 cited references verified to have matching bibliography entries.

SUBMISSION DATE: March 4, 2026
