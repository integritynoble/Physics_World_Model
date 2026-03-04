InverseNet: Benchmarking Operator Mismatch and Calibration Across Compressive Imaging Modalities
arXiv Submission Package

CONTENTS:
=========

Combined Paper & Supplementary:
-------------------------------
inversenet_paper.tex          Main paper + Supplementary Material (33 pages)
inversenet_paper.pdf          Combined PDF (4.0 MB)

Support Files:
--------------
llncs.cls                     LLNCS document class
splncs04.bst                  LLNCS bibliography style
figures/                      All figures (PDF format, ~2.5 MB)

COMPILATION:
============

To compile locally:
  pdflatex inversenet_paper.tex

(Bibliography is inline; no external .bib file needed)
(Supplementary material is now part of the main .tex file as Appendix A-D)

AUTHOR INFORMATION:
===================

Chengshuai Yang (Correspondence: integrityyang@gmail.com)
  NextGen PlatformAI C Corp, USA

Xin Yuan (xyuan@westlake.edu.cn)
  School of Engineering, Westlake University, Hangzhou, China

PAPER ORGANIZATION:
===================

Main Paper (Pages 1-19):
  - Introduction & Related Work
  - Benchmark protocol & datasets
  - Experimental results (CASSI, CACTI, SPC)
  - Real hardware validation
  - Discussion & conclusion

Supplementary Appendix (Pages 20-33):
  - Appendix A: Per-scene detailed results
  - Appendix B: Real hardware validation methodology
  - Appendix C: Scenario IV blind calibration results
  - Appendix D: Implementation details & runtime analysis

CHANGES IN REVISED VERSION (March 4, 2026):
============================================

This version incorporates 10 improvements addressing pre-submission review:

1. Scenario IV CASSI/CACTI: Explicit caveat (2/8 parameters only)
2. Table 1: Corrected bold notation (best ρ marked with †)
3. Abstract: Tightened to ~150 words (added Spearman r_s=-0.71)
4. Cross-modality table: Clarified SSIM basis
5. Real hardware: Mechanistic justification for deep learning exclusion
6. CACTI residual gap: Explained 256× difference
7. PnP-HSICNN citation: Corrected to zheng2021pnp
8. Limitations: Named missing methods (DAUHST, CST, DiffSCI)
9. Data availability: Removed placeholder URL
10. Bibliography: Removed 4 uncited entries

All 34 cited references verified.

ARXIV SUBMISSION:
=================

Upload the entire contents of this folder to https://arxiv.org/submit
or create a tar.gz: tar -czf inversenet_arxiv.tar.gz *
