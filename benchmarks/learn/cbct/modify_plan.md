# Modify Plan -- cbct

**Date:** 2026-03-03
**Category:** medical | **Carrier:** X-ray | **Score key:** medical

## Current Algorithms (from catalog)

| # | Algorithm          | Type           | Source                            |
|---|--------------------|----------------|-----------------------------------|
| 1 | FBP                | Classical      | Analytical baseline               |
| 2 | PnP-ADMM          | PnP            | Venkatakrishnan et al., 2013      |
| 3 | FBPConvNet         | Deep Learning  | Jin et al., IEEE TIP 2017         |
| 4 | Learned Primal-Dual| Deep Unrolling | Adler & Oktem, IEEE TMI 2018      |

## Assessment

### Are algorithms domain-appropriate?
YES. CBCT (Cone-Beam CT) is routed to the `medical` pool via (medical, X-ray) which stays as the default `medical` category (no carrier routing override for X-ray -- it falls through to the base medical pool). All four algorithms are well-known, heavily-cited CT reconstruction methods:

- FBP (Filtered Back Projection): THE standard analytical CT reconstruction method. For cone-beam, FDK (Feldkamp-Davis-Kress) would be more precise, but FBP is the accepted shorthand in the benchmark context.
- PnP-ADMM: Venkatakrishnan et al., IEEE GlobalSIP 2013 -- pioneering PnP paper, widely applied to CT.
- FBPConvNet: Jin et al., IEEE TIP 2017 -- landmark CNN post-processing method for CT, ~2000 citations.
- Learned Primal-Dual: Adler & Oktem, IEEE TMI 2018 -- gold standard deep unrolling method for CT, ~1500 citations.

### Are citations correct?
YES. All four citations are accurate and correspond to real, highly-cited papers:
- FBP: "Analytical baseline" is a standard label
- PnP-ADMM: Venkatakrishnan et al., 2013 -- correct
- FBPConvNet: Jin et al., IEEE TIP 2017 -- correct
- Learned Primal-Dual: Adler & Oktem, IEEE TMI 2018 -- correct

### Other issues
- The check.md (comprehensive review) reports TransCT, FBPConvNet, PnP-DRUNet, FBP -- different from the actual catalog output. The check.md is stale, predating the current algorithm catalog.
- The check.md identifies that no HDF5 dataset files exist for CBCT (CRITICAL). This is an infrastructure issue, not an algorithm issue.
- FBP is technically a 2D fan-beam algorithm; for cone-beam CT, FDK would be more precise. However, "FBP" is commonly used as a catch-all term in the CT community and is acceptable.
- The medical pool is CT-centric, which is perfectly appropriate for CBCT.

## Plan

No code changes needed. The medical pool algorithms are all well-established, correctly-cited CT reconstruction methods that are directly appropriate for CBCT.
