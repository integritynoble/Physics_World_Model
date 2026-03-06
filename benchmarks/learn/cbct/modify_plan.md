# Modify Plan -- cbct

**Date:** 2026-03-06
**Category:** medical | **Carrier:** X-ray | **Score key:** medical

## Current Algorithms (from catalog)

| # | Algorithm           | Type           | Source                            |
|---|---------------------|----------------|-----------------------------------|
| 1 | FBP (FDK)           | Classical      | Feldkamp et al., JOSA A 1, 612 (1984) |
| 2 | TV-ADMM             | Compressed Sensing | Sidky & Pan, Phys. Med. Biol. 53, 4777 (2008) |
| 3 | FBPConvNet          | Deep Learning  | Jin et al., IEEE TIP 26, 4509 (2017) |
| 4 | Learned Primal-Dual | Deep Unrolling | Adler & Oktem, IEEE TMI 37, 1322 (2018) |

## Assessment

### Are algorithms domain-appropriate?

YES. CBCT (Cone-Beam CT) is routed to the `medical` pool via (medical, X-ray) — no carrier routing override for X-ray, falls through to the base medical pool. All four algorithms are well-known, heavily-cited CT reconstruction methods:

- **FBP (FDK)**: THE standard analytical cone-beam CT reconstruction method. Feldkamp-Davis-Kress (FDK, 1984) is the original paper. "FBP" is used as shorthand but refers to FDK in the CBCT context.
- **TV-ADMM**: Sidky & Pan, PMB 2008 is the landmark paper for total variation CT reconstruction from sparse views, directly applicable to CBCT.
- **FBPConvNet**: Jin et al., IEEE TIP 2017 — landmark CNN post-processing method for CT, ~2000 citations. Directly applicable to CBCT.
- **Learned Primal-Dual**: Adler & Oktem, IEEE TMI 2018 — gold standard deep unrolling method for CT, ~1500 citations.

### Are citations correct?

YES. All four citations are accurate and correspond to real, highly-cited papers.

### Other issues

- FBP is technically the fan-beam/parallel-beam algorithm; for cone-beam CT, FDK (Feldkamp-Davis-Kress) would be more precise. However, "FBP" is commonly used as a catch-all term in the CT community and is accepted shorthand for FDK in this context.
- The medical pool is CT-centric, which is perfectly appropriate for CBCT.

## Plan

No code changes needed. The medical pool algorithms are all well-established, correctly-cited CT reconstruction methods that are directly appropriate for CBCT.

**Priority:** NONE — no changes needed.
