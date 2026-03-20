# OCT (Optical Coherence Tomography) Benchmark Dataset

## Overview

Optical Coherence Tomography B-scan benchmark with axial PSF convolution,
multiplicative speckle noise, depth-dependent signal roll-off, and motion
artifacts. Uses synthetic retinal and anterior segment phantoms with
realistic layered tissue structure.

## Forward Model

```
bscan_ideal(z, x)    = PSF_axial(z) * reflectivity(z, x)
bscan_atten(z, x)    = bscan_ideal * exp(-rolloff * z)
bscan_speckle(z, x)  = bscan_atten * speckle_noise (Rayleigh)
bscan_measured(z, x)  = shift(bscan_speckle, motion) + shot_noise

where:
  reflectivity(z, x) -- 2D cross-sectional tissue reflectivity (ground truth)
  PSF_axial           -- Gaussian axial PSF (FWHM = coherence length)
  rolloff             -- depth-dependent signal attenuation (dB/mm)
  speckle_noise       -- multiplicative Rayleigh noise from coherent interference
  motion              -- lateral A-scan displacement (eye saccades)
  shot_noise          -- additive Gaussian noise
```

## Imaging Parameters

| Parameter | Value |
|-----------|-------|
| Centre wavelength | 850 nm |
| Pixel size | 3.0 um |
| Image size | 256 x 256 px |
| FOV | 768 um (axial) x 768 um (lateral) |
| Dynamic range | 50 dB |

## Mismatch Parameters (ThetaSpace)

| Knob | Symbol | Description | Public | Dev | Hidden |
|------|--------|-------------|--------|-----|--------|
| `speckle_snr_db` | SNR_s | Speckle noise level | 22-35 dB | 18-35 dB | 15-35 dB |
| `axial_psf_fwhm_um` | FWHM_z | Axial resolution | 3-8 um | 3-12 um | 3-15 um |
| `motion_artifact_px` | d_motion | Lateral motion | 0-3 px | 0-6 px | 0-10 px |
| `signal_rolloff_db` | alpha_z | Depth signal falloff | 2-6 dB/mm | 2-8 dB/mm | 2-10 dB/mm |

## Phantom Types

| Type | Description | Tier |
|------|-------------|------|
| Normal retina | 10 layered structures (ILM to choroid) with natural curvature | Public, Dev |
| Foveal retina | Normal retina with foveal depression (thinned inner layers) | Public |
| Drusen | Sub-RPE deposits elevating RPE layer | Public, Hidden |
| Intraretinal cysts | Fluid-filled dark voids in INL/ONL | Public, Hidden |
| Serous detachment | Sub-RPE fluid separating RPE from Bruch membrane | Public, Hidden |
| Epiretinal membrane | Bright membrane above ILM with traction folds | Public, Hidden |
| Anterior segment | Cornea, anterior chamber, iris, lens, angle structures | Public, Dev, Hidden |
| Multi-pathology | Drusen + cysts on same retina | Hidden |
| Low contrast | Very low contrast retina (adversarial) | Hidden |
| Cataract | Anterior segment with lens opacity | Hidden |

## Retinal Layer Structure

```
Vitreous (dark)
  ILM ----------- Inner Limiting Membrane (bright boundary)
  NFL ----------- Nerve Fiber Layer (BRIGHT)
  GCL ----------- Ganglion Cell Layer (dark)
  IPL ----------- Inner Plexiform Layer (medium)
  INL ----------- Inner Nuclear Layer (medium-dark)
  OPL ----------- Outer Plexiform Layer (medium)
  ONL ----------- Outer Nuclear Layer (DARK)
  ELM ----------- External Limiting Membrane (boundary)
  IS/OS --------- Inner/Outer Segment junction (VERY BRIGHT)
  OS  ----------- Outer Segments (medium-dark)
  RPE ----------- Retinal Pigment Epithelium (VERY BRIGHT)
  Bruch's ------- Bruch Membrane (bright boundary)
  Choroid ------- Choroidal tissue (medium, depth-attenuated)
Sclera (fading)
```

## Dataset Structure

```
oct/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (4 normal + 4 pathological + 4 anterior)
|   +-- oct_challenge_public.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- dev/       20 samples (augmented, medium mismatch)
|   +-- oct_challenge_dev.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- hidden/    20 samples (adversarial, wide mismatch)
    +-- oct_challenge_hidden.h5
    +-- spec.json / true_spec.json
    +-- images/sample_XX_*/
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32         -- Ground-truth tissue reflectivity
+-- bscan_ideal (256, 256) float32    -- Clean B-scan (PSF + rolloff)
+-- bscan_measured (256, 256) float32 -- Degraded B-scan (all effects)
```

## CPU Reconstruction

Median filtering + bilateral denoising (edge-preserving speckle reduction):
  1. Median filter (size=3) to remove impulse-like speckle
  2. Bilateral-like smoothing via guided Gaussian filtering
  3. Edge detection via local intensity difference weighting

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

- Huang et al. (1991) "Optical coherence tomography," Science 254:1178-1181.
- Drexler & Fujimoto (2008) "Optical Coherence Tomography," Springer.
- Maggioni et al. (2012) "BM3D for OCT," IEEE Trans. Image Processing 21:1715-1728.
- Hu et al. (2020) "speckle2void," Biomedical Optics Express 11:817-830.
