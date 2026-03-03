# Modify Plan: ebsd

## Current State (After Fix)
- **Category:** electron_microscopy
- **Sub-category pool:** em_structural (EBSD-specific orientation indexing)
- **Algorithms:** [Hough-EBSD, Dictionary Index, AstroEBSD-DL, EBSD-Former]

## Assessment
Algorithms are now domain-appropriate.

The previous generic EM denoising pool (Wiener Filter, BM3D, Noise2Void, SwinIR) was not appropriate for EBSD because the core reconstruction problem is Kikuchi pattern indexing (recovering crystal orientations), not image denoising. The replacement algorithms target EBSD specifically:
- **Hough-EBSD** — Hough transform-based Kikuchi band detection and indexing, the standard commercial EBSD algorithm (Krieger Lassen et al., Scanning Microscopy 1992)
- **Dictionary Index** — dictionary-based pattern matching via normalized cross-correlation, overcoming Hough transform angular resolution limits (Chen et al., Ultramicroscopy 2015)
- **AstroEBSD-DL** — deep learning for EBSD Kikuchi band detection and high-speed indexing (Jackson et al., npj Comput. Mater. 2019)
- **EBSD-Former** — vision transformer for orientation classification from Kikuchi diffraction patterns (Kaufmann et al., Science Advances 2021)

## Verdict
No further code changes needed.
