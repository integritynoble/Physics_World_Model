# Modify Plan -- angiography

## Algorithm Catalog Review

**Category:** medical | **Carrier:** X-ray | **Score key:** medical

| Algorithm | Type | Source |
|-----------|------|--------|
| FBP | Classical | Feldkamp et al., JOSA A 1, 612 (1984) |
| TV-ADMM | Compressed Sensing | Rudin et al., Physica D 60, 259 (1992) + ADMM |
| FBPConvNet | Deep Learning | Jin et al., IEEE TIP 26, 4509 (2017) |
| Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 37, 1322 (2018) |

### Domain Appropriateness

**Good fit.** X-ray angiography is an X-ray projection/tomographic imaging modality. The carrier "X-ray" does not trigger carrier routing, so it falls through to the generic `medical` pool which contains CT/X-ray reconstruction algorithms. This is appropriate because:

- **FBP** — Standard analytical baseline for cone-beam rotational angiography (3DRA). Feldkamp 1984 citation is correct.
- **TV-ADMM** — Total variation compressed sensing for sparse-view 3DRA. Appropriate.
- **FBPConvNet** — Jin et al., IEEE TIP 2017 is a real, well-cited CT reconstruction paper. Applicable to X-ray angiography reconstruction.
- **Learned Primal-Dual** — Adler & Oktem, IEEE TMI 2018 is a real deep unrolling paper for CT. Applicable to rotational angiography.

All citations are correct and real publications.

**Minor note:** Angiography-specific methods (e.g., DSA subtraction algorithms, vessel-segmentation networks) could provide domain-specific flavor, but the current CT-family algorithms are technically valid since rotational angiography uses X-ray projections through the same physics.

### Learning Materials

`03_reconstruction_algorithms.md` lists only "FBP (DSA baseline)" which partially matches. Should add TV-ADMM, FBPConvNet, and Learned Primal-Dual to the solver comparison table.

## Proposed Changes

1. **`03_reconstruction_algorithms.md`**: Add TV-ADMM, FBPConvNet, and Learned Primal-Dual to the solver comparison table with proper citations.

No code changes needed in `_algorithm_catalog.py`. Algorithms and citations are appropriate.

**Priority:** LOW — only learning materials need sync.
