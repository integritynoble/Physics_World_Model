# Modify Plan -- angiography

## Algorithm Catalog Review

**Category:** medical | **Carrier:** X-ray | **Score key:** medical

| Algorithm | Type | Source |
|-----------|------|--------|
| FBP | Classical | Analytical baseline |
| PnP-ADMM | PnP | Venkatakrishnan et al., 2013 |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 2017 |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 2018 |

### Domain Appropriateness

**Good fit.** X-ray angiography is an X-ray projection imaging modality. The carrier "X-ray" does not trigger carrier routing, so it falls through to the generic `medical` pool which contains CT/X-ray reconstruction algorithms. This is reasonable because:

- **FBP** -- Standard analytical baseline for any X-ray projection/CT problem. Appropriate.
- **PnP-ADMM** -- Venkatakrishnan et al., 2013 is the canonical PnP reference. Appropriate.
- **FBPConvNet** -- Jin et al., IEEE TIP 2017 is a real CT reconstruction paper. Applicable to X-ray projection reconstruction.
- **Learned Primal-Dual** -- Adler & Oktem, IEEE TMI 2018 is a real deep unrolling paper for CT. Applicable to X-ray angiography.

All citations are correct and real publications.

**Minor note:** Angiography-specific methods (e.g., DSA subtraction, vessel-specific segmentation networks) could provide more domain-specific flavor, but the current CT-family algorithms are technically valid since angiography uses X-ray projections through the same physics.

### Learning Materials

`03_reconstruction_algorithms.md` lists only "FBP (DSA baseline)" which partially matches (FBP is present on the leaderboard). The other 3 leaderboard algorithms are not documented in learning materials.

## Proposed Changes

1. **`03_reconstruction_algorithms.md`**: Add PnP-ADMM, FBPConvNet, and Learned Primal-Dual to the solver comparison table.

No code changes needed in `_algorithm_catalog.py`. Algorithms and citations are appropriate.

**Priority:** LOW -- only learning materials need sync.
