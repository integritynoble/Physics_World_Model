# Modify Plan — mammography

## Current State

- **Category:** medical
- **Carrier:** X-ray
- **Score key:** medical
- **Algorithms (from catalog):**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018
- **Leaderboard (live):** FBP, PnP-ADMM, FBPConvNet, Learned Primal-Dual (4 entries)

## Assessment

The algorithms are **appropriate** for mammography.

- **FBP** (Filtered Back Projection) is the standard analytical reconstruction for X-ray projection imaging, including digital breast tomosynthesis (DBT) which is the tomographic form of mammography. For 2D mammography (single projection), FBP is less directly relevant, but mammography does involve denoising and scatter correction that can be framed as inverse problems.
- **PnP-ADMM** is a well-established PnP framework applicable to X-ray imaging reconstruction.
- **FBPConvNet** (Jin et al., IEEE TIP 2017) is a deep learning method designed for CT/X-ray reconstruction -- applicable to mammography.
- **Learned Primal-Dual** (Adler & Oktem, IEEE TMI 2018) is a physics-informed deep unrolling method for tomographic imaging -- well-suited.

The medical/X-ray routing gives mammography the same CT-focused algorithm set. This is a reasonable fit because mammography (especially digital breast tomosynthesis) shares the X-ray projection physics with CT. The algorithms are all real, published, and properly cited.

## Verdict

No code changes needed.
