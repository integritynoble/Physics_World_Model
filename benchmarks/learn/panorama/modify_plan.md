# Modify Plan: panorama

## Current State (After Fix)
- **Category:** computational_photography
- **Sub-category pool:** computational_photography (panorama-specific override)
- **Algorithms:** [SIFT-RANSAC, APAP, UDIS, PanoFormer]

## Assessment
Algorithms are now domain-appropriate.

The previous generic computational photography pool (Wiener-Deconv, PnP-FFDNet, HDR-CNN, Uformer) was acceptable but missed the stitching geometry and multi-focus fusion aspects that define this modality. The replacement algorithms directly address the panorama problem:
- **SIFT-RANSAC** — SIFT feature matching with RANSAC-based homography estimation, the standard classical panorama stitching pipeline (Lowe, IJCV 2004; Fischler & Bolles, CACM 1981)
- **APAP** — As-Projective-As-Possible spatially-varying warping that handles parallax artifacts in multi-focus captures (Zaragoza et al., CVPR 2013)
- **UDIS** — unsupervised deep image stitching with elastic alignment network (Nie et al., IEEE TIP 2021)
- **PanoFormer** — transformer architecture for panoramic image stitching with deformable attention (Shen et al., ECCV 2022)

## Verdict
No further code changes needed.
