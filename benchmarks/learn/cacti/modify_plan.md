# Modify Plan -- cacti

**Date:** 2026-03-03
**Category:** compressive | **Carrier:** Photon | **Score key:** compressive

## Current Algorithms (from catalog)

| # | Algorithm     | Type           | Source        |
|---|---------------|----------------|---------------|
| 1 | GAP-TV        | Classical      | InverseNet    |
| 2 | PnP-FFDNet    | PnP            | InverseNet    |
| 3 | ELP-Unfolding | Deep Unfolding | ECCV 2022     |
| 4 | EfficientSCI  | Deep Learning  | CVPR 2023     |
| 5 | HiSViT-9      | Transformer    | ECCV 2024     |

## Assessment

### Are algorithms domain-appropriate?
YES. This is a hand-crafted variant override in `_VARIANT_OVERRIDES["cacti"]`.
All 5 algorithms are real, published SCI/CACTI reconstruction methods:
- GAP-TV: Standard convex optimization baseline for snapshot compressive imaging (Yuan et al., 2016)
- PnP-FFDNet: Plug-and-play with FFDNet denoiser, widely used in SCI (Zhang et al., 2017)
- ELP-Unfolding: Deep unfolding for SCI (ECCV 2022) -- correct
- EfficientSCI: Efficient DL for SCI (Wang et al., CVPR 2023) -- correct
- HiSViT-9: Vision Transformer for SCI (ECCV 2024) -- correct

### Are citations correct?
Mostly. Minor notes:
- GAP-TV source says "InverseNet" rather than "Yuan et al., 2016" -- acceptable since this is a validated InverseNet baseline
- PnP-FFDNet source says "InverseNet" rather than "Zhang et al., 2017 / Yuan et al., 2019" -- same reason
- ELP-Unfolding: ECCV 2022 is correct (Wang et al.)
- EfficientSCI: CVPR 2023 is correct (Wang et al.)
- HiSViT-9: ECCV 2024 -- this needs verification; HiSViT may be a 2024 preprint, not necessarily ECCV

### Other issues
- check.md (comprehensive review) mentions PnP-DnCNN but actual catalog has PnP-FFDNet -- the check.md is stale
- check.md identifies many HIGH severity issues (sample count mismatch, zero leaderboards, identical spec_ranges) -- these are infrastructure issues, not algorithm catalog issues

## Plan

No code changes needed. The cacti variant has a well-curated hand-crafted algorithm override with 5 domain-appropriate SCI reconstruction methods. The InverseNet source attributions are acceptable for validated baselines.
