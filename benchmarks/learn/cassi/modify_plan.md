# Modify Plan -- cassi

**Date:** 2026-03-03
**Category:** compressive | **Carrier:** Photon | **Score key:** compressive

## Current Algorithms (from catalog)

| # | Algorithm    | Type          | Source              |
|---|--------------|---------------|---------------------|
| 1 | GAP-TV       | Classical     | Yuan et al., 2016   |
| 2 | PnP-FFDNet   | PnP           | Zhang et al., 2017  |
| 3 | EfficientSCI | Deep Learning | Wang et al., 2023   |
| 4 | MST-L        | Transformer   | Cai et al., CVPR 2022 |

## Assessment

### Are algorithms domain-appropriate?
PARTIALLY APPROPRIATE, BUT THERE IS A ROUTING ISSUE.

The `cassi` variant key exists in MODALITY_CATALOG (category=compressive, carrier=Photon) and gets the generic `compressive` pool. However, the actual benchmark variant key used is `sd_cassi`, not `cassi`. The `cassi` page returns HTTP 404 because `cassi` is not in `list_all_variant_keys()` -- only `sd_cassi` is.

The `sd_cassi` variant has a hand-crafted override with HSI-specific algorithms:
- GAP-TV (InverseNet validated)
- PnP-HSICNN (InverseNet validated, HSI-specific)
- HDNet (InverseNet validated, HSI-specific)
- MST-L (InverseNet validated, Mask-aware Spectral Transformer)

The generic compressive pool that `cassi` maps to (GAP-TV, PnP-FFDNet, EfficientSCI, MST-L) is SCI-oriented (video compressive), not HSI-oriented. EfficientSCI in particular is a video SCI method, not a spectral CASSI method.

### Are citations correct?
For the compressive pool: yes, all citations are correct.
For the sd_cassi override: InverseNet source attributions are acceptable for validated baselines.

### Other issues
- CRITICAL: The `cassi` benchmark page returns 404. Only `sd_cassi` has a working page.
- The `cassi` entry in MODALITY_CATALOG exists but there is no corresponding challenge dataset or variant page.
- The check.md correctly reports this as HTTP 404 ERROR.
- Learning materials exist under `benchmarks/learn/cassi/` but the benchmark page itself does not work.

## Plan

No code changes needed to the algorithm catalog. The `cassi` variant is effectively dead (404 page, no challenge data). The working variant is `sd_cassi` which has correct hand-crafted HSI algorithms. The `cassi` MODALITY_CATALOG entry and learning materials exist but are orphaned from the actual benchmark system. If `cassi` should be made live, it would need to be added to `list_all_variant_keys()` and have challenge datasets generated -- but that is an infrastructure issue, not an algorithm catalog issue.
