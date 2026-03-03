# Modify Plan: portal_imaging

## Current State
- **Category:** medical
- **Carrier:** MV
- **Score key:** medical (via carrier routing: ("medical", "MV") -> "medical")
- **Algorithms:**
  1. FBP (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Venkatakrishnan et al., 2013
  3. FBPConvNet (Deep Learning) -- Jin et al., IEEE TIP 2017
  4. Learned Primal-Dual (Deep Unrolling) -- Adler & Oktem, IEEE TMI 2018

## Assessment

Portal imaging uses Electronic Portal Imaging Devices (EPIDs) with megavoltage (MV) X-ray beams during radiation therapy to verify patient positioning and beam delivery. The carrier routing `("medical", "MV") -> "medical"` correctly keeps the CT-like reconstruction pool, which is appropriate because:

- Portal imaging produces 2D radiographic projection images or MVCT reconstructions
- The reconstruction problem (projection to image) is similar to CT reconstruction
- FBP, FBPConvNet, and Learned Primal-Dual are applicable to MV cone-beam CT reconstruction

The algorithms are appropriate:
- **FBP** -- standard for CBCT/MVCT reconstruction. Correct.
- **PnP-ADMM** -- generic iterative reconstruction with prior. Correct.
- **FBPConvNet** -- post-processing on FBP, applicable to MVCT. Correct.
- **Learned Primal-Dual** -- learned iterative CT reconstruction. Correct.

The leaderboard shows these same methods, confirming the assignment.

## Required Changes

No code changes needed. The medical (CT-like) algorithms are appropriate for EPID/MV portal imaging.
