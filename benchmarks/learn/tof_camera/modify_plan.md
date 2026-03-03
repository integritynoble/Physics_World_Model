# Modify Plan: Time-of-Flight Depth Camera

**Created:** 2026-03-03
**Status:** Algorithms are a partial mismatch but acceptable as a generic depth-imaging pool

## Assessment

ToF camera falls under `depth_imaging` category with carrier `Photon/IR`. It receives:

- SGM (Classical) -- Semi-Global Matching (Hirschmuller, TPAMI 2007)
- PnP-ADMM (PnP) -- generic plug-and-play prior
- PSMNet (Deep Learning) -- Pyramid Stereo Matching (Chang & Chen, CVPR 2018)
- RAFT-Stereo (Transformer) -- stereo depth estimation (Lipson et al., 3DV 2021)

### Issue

SGM, PSMNet, and RAFT-Stereo are **stereo matching** algorithms that estimate depth from binocular image pairs. ToF cameras use a fundamentally different principle: they measure depth from the phase shift of modulated light (correlation-based ToF) or direct photon arrival times (direct ToF / SPAD). ToF-specific reconstruction involves:

- Phase unwrapping (for multi-frequency ToF)
- Multi-path interference (MPI) correction
- Amplitude-based denoising

ToF-specific algorithms would include:
- Classical: Weighted least-squares depth filtering or phase unwrapping
- Deep Learning: DeepToF (Marco et al., ECCV 2018), Deep End-to-End ToF (Su et al., CVPR 2018)

### Decision

The `depth_imaging` category is a shared pool across ToF, structured light, and stereo modalities. While stereo-specific names (SGM, PSMNet) are not ideal for ToF, the category serves as a "depth estimation" umbrella. Adding ToF-specific routing would require a new carrier-based split or variant override.

## Deferred Items

1. **Carrier routing**: Could add `("depth_imaging", "Photon/IR")` to `_CARRIER_ROUTING` pointing to a `tof_depth` sub-pool with phase-unwrapping and MPI-correction methods. Low priority since the benchmark measures reconstruction quality generically (PSNR/SSIM on depth maps).

No code changes required at this time.
