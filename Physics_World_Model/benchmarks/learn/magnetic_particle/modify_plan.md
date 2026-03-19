# Modify Plan — magnetic_particle

## Current State

- **Category:** experimental_science
- **Carrier:** Magnetic
- **Score key:** experimental_science
- **Algorithms (from catalog):**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021
- **Leaderboard (live):** Tikhonov, PnP-RED, ResUNet, SwinIR (4 entries)

## Assessment

The algorithms are **acceptable but generic**. Magnetic Particle Imaging (MPI) has a well-developed reconstruction literature that could be better represented.

- **Tikhonov** is appropriate -- Tikhonov-regularized system matrix inversion is the standard classical MPI reconstruction method (Knopp et al., PMB 2010).
- **PnP-RED** is a reasonable generic PnP method, though no MPI-specific PnP paper exists yet. Acceptable as a placeholder.
- **ResUNet** is generic. MPI has specific deep learning methods like **MPI-NET** (Dittmer et al., 2020) and **Deep Image Prior for MPI** (Askin et al., 2022).
- **SwinIR** is a generic image restoration transformer. No MPI-specific transformer exists yet, so this is acceptable as a placeholder.

The "experimental_science" category is a catch-all, and the generic algorithms are defensible. However, MPI reconstruction is a mature enough field that domain-specific algorithms would add credibility.

## Recommended Changes (Optional)

If improving specificity:
1. Add a variant override for `magnetic_particle`:
   - Classical: **Tikhonov (System Matrix)** -- Knopp et al., PMB 2010
   - PnP: **PnP-RED** (keep)
   - Deep Learning: **MPI-NET** -- Dittmer et al., 2020
   - Transformer: **SwinIR** (keep as generic placeholder)

These are minor improvements. The current generic algorithms are not wrong, just not MPI-specific.

## Current Algorithm Count (updated 2026-03-06)

Full pool (11 algorithms): Tikhonov, Wiener Filter, Matched Filter, PnP-RED, PnP-ADMM, ResUNet, Domain-Adapted-CNN, SwinIR, ExpFormer, DiffusionExperimental, ScoreExperimental.

**Status:** PASS — check.md written 2026-03-06

## Verdict

No code changes needed (generic algorithms are acceptable). Optional improvement: replace ResUNet with MPI-NET for better domain specificity.
