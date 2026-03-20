# Modify Plan: polarization

## Current State
- **Category:** microscopy
- **Carrier:** Photon
- **Score key:** microscopy
- **Algorithms:**
  1. Richardson-Lucy (Classical) -- Richardson 1972 / Lucy 1974
  2. PnP-FISTA (PnP) -- Bai et al., 2020
  3. CARE (Deep Learning) -- Weigert et al., Nat. Methods 2018
  4. Restormer (Transformer) -- Zamir et al., CVPR 2022

## Assessment

Polarization microscopy images birefringent specimens by analyzing polarization state changes of transmitted/reflected light. The category `microscopy` is reasonable. The reconstruction involves recovering specimen retardance, orientation, or birefringence maps from intensity measurements under different polarization configurations.

The generic microscopy deconvolution algorithms (Richardson-Lucy, CARE, etc.) are partially applicable -- deconvolution and denoising are part of the pipeline. However, the unique aspect of polarization microscopy is Mueller matrix / Jones calculus decomposition, which these algorithms do not address.

That said, for a general reconstruction benchmark where the goal is to recover a 2D image (retardance map or birefringence map) from measurements, generic image restoration algorithms can serve as baselines. The mismatch is mild to moderate.

More specific algorithms would include:
- Mueller matrix decomposition (Lu-Chipman, JOSA A 1996)
- Stokes parameter estimation (Classical)
- PolScope (Shribak & Oldenbourg, Appl. Opt. 2003)

## Required Changes

No code changes needed. The generic microscopy algorithms are acceptable as reconstruction baselines, though not ideal. The mismatch is mild since the benchmark evaluates image quality metrics (PSNR/SSIM) rather than domain-specific polarimetric quantities.
