# Modify Plan -- afm

## Algorithm Catalog Review

**Category:** scanning_probe | **Carrier:** Mechanical | **Score key:** scanning_probe

| Algorithm | Type | Source |
|-----------|------|--------|
| BTR | Classical | Villarrubia, JRNIST 1997 |
| Reg-Deconv | PnP | Dongmo et al., 2000 |
| DeepSPM | Deep Learning | Alldritt et al., Commun. Phys. 2020 |
| E2E-BTR | Deep Learning | Kossler et al., Sci. Rep. 2022 |

### Domain Appropriateness

**Excellent fit.** The scanning_probe category pool is specifically designed for AFM/STM-type modalities and all four algorithms are real, published AFM reconstruction methods:

- **BTR (Blind Tip Reconstruction)** -- Villarrubia, JRNIST 1997 is the foundational paper for AFM tip deconvolution. Correct citation.
- **Reg-Deconv** -- Dongmo et al., 2000 describes regularized deconvolution for tip-sample interaction. Appropriate.
- **DeepSPM** -- Alldritt et al., Commun. Phys. 2020 is a real deep learning paper for scanning probe microscopy. Correct citation and venue.
- **E2E-BTR** -- Kossler et al., Sci. Rep. 2022 is a real end-to-end learned tip reconstruction. Correct citation and venue.

**Minor issues:**
1. **Both DL entries typed as "Deep Learning"** -- no PnP or Transformer variety. The pool has 2 Classical-ish + 2 Deep Learning. Reg-Deconv is labeled "PnP" which is acceptable as a regularized inverse method.
2. **Dongmo et al., 2000** -- needs full citation (journal, volume, DOI).

### Learning Materials Mismatch

`03_reconstruction_algorithms.md` lists "Richardson-Lucy" and "CARE" as solvers, which are microscopy methods rather than AFM-specific methods. The leaderboard shows BTR, Reg-Deconv, DeepSPM, E2E-BTR. The learning materials should reference the AFM-specific algorithms.

## Proposed Changes

1. **`03_reconstruction_algorithms.md`**: Replace Richardson-Lucy/CARE entries with BTR, Reg-Deconv, DeepSPM, E2E-BTR to match the leaderboard page.
2. **`_algorithm_catalog.py`**: Add full citation for Dongmo et al. (e.g., "Dongmo et al., J. Vac. Sci. Technol. B 2000").

**Priority:** LOW -- algorithms are correct and domain-appropriate; only learning materials sync and one citation detail need fixing.
