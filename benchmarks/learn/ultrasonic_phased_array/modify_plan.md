# Modify Plan: Ultrasonic Phased Array (TFM/FMC)

**Created:** 2026-03-03
**Status:** Algorithms are a partial mismatch but acceptable as a generic NDT pool

## Assessment

Ultrasonic phased array falls under `industrial_inspection` category with carrier `Acoustic`. It receives:

- TSR (Classical) -- Thermographic Signal Reconstruction (Shepard et al., 2003)
- PnP-ADMM (PnP) -- generic plug-and-play prior
- DefectNet (Deep Learning) -- U-Net for NDT
- LSTM-NDT (Recurrent) -- Fang et al., 2022

### Issue

TSR is a **thermography-specific** algorithm (polynomial fitting of IR thermal decay curves). It has no relevance to ultrasonic phased array imaging, which uses:

- Total Focusing Method (TFM) -- the gold-standard classical beamformer for Full Matrix Capture (FMC) data
- Delay-and-Sum (DAS) -- basic beamforming
- SAFT (Synthetic Aperture Focusing Technique)
- Deep learning: DL-TFM, UNet-based defect segmentation on B-scans

PnP-ADMM and DefectNet are reasonable generic NDT methods. LSTM-NDT is also plausible for temporal ultrasonic signal processing.

### Decision

The `industrial_inspection` category is a shared pool across thermography, ultrasonic, eddy current, and THz NDT modalities. TSR is the weakest link for ultrasonic methods, but splitting NDT into sub-pools (thermographic vs. ultrasonic vs. electromagnetic) would require new carrier routing.

## Deferred Items

1. **Carrier routing**: Could add `("industrial_inspection", "Acoustic")` to `_CARRIER_ROUTING` pointing to an `ultrasonic_ndt` pool with TFM/DAS as the classical baseline. Medium priority -- TSR is clearly wrong for ultrasonic data.
2. **Classical baseline**: Replace TSR with DAS or TFM for acoustic NDT variants.

No code changes required at this time -- acceptable as generic NDT pool.
