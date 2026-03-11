# Ultrasound B-Mode Benchmark Dataset

## Overview

High-quality B-mode ultrasound imaging benchmark with PICMUS/CIRS-style
tissue-mimicking phantoms, depth-dependent PSF convolution, multiplicative
speckle noise, and log-compression forward model.

## Forward Model

```
scatterer_field = x_true * rayleigh_scatterers
envelope        = depth_dependent_PSF_conv(scatterer_field)
attenuated      = envelope * 10^(-2*alpha*f*depth/20)
y_linear        = attenuated * speckle_envelope + electronic_noise
y               = 20*log10(y_linear / max) -> clipped to [-DR, 0] -> [0,1]
```

## Depth-Dependent PSF Model

The PSF varies with depth to model beam divergence:
  - sigma_lateral(d) = (f_number * lambda / 2) * (1 + defocus_rate * d)
  - sigma_axial = n_cycles * lambda / 2 (constant)
  - Image divided into 8 depth bands, PSFs blended smoothly

| Parameter | Value |
|-----------|-------|
| Frequency | 5 MHz |
| Speed of sound | 1540 m/s |
| F-number | 1.5 |
| Pulse cycles | 3.5 |
| Pixel size | 0.15 mm |
| FOV | 38.4 mm |
| Dynamic range | 60 dB |
| Depth defocus rate | 0.003 per pixel |

## Mismatch Parameters (ThetaSpace)

| Knob | Symbol | Description | Public | Dev | Hidden |
|------|--------|-------------|--------|-----|--------|
| `speed_of_sound_error_pct` | SoS err | Focus error from SoS mismatch | 0-3% | 0-5% | 0-8% |
| `attenuation_dB_cm_MHz` | alpha | Tissue attenuation | 0.3-0.7 | 0.3-0.9 | 0.3-1.2 |
| `speckle_density` | N_s | Scatterers per resolution cell | 10-25 | 8-35 | 5-50 |
| `snr_db` | SNR | Electronic SNR | 30-40 dB | 25-38 dB | 20-35 dB |

## Phantom Types (PICMUS/CIRS-Style)

| Type | Description | Tier |
|------|-------------|------|
| Anechoic cysts | Dark fluid-filled cysts at varying depths (PICMUS standard) | Public, Dev |
| Hyperechoic lesions | Bright solid inclusions with internal texture | Public, Dev |
| Mixed cyst+lesion | Both anechoic and hyperechoic targets (clinical scenario) | Public, Dev |
| Layered tissue | Skin/fat/muscle/organ with specular interfaces | Public, Dev |
| Point targets | Grid of point reflectors + wire targets (resolution test) | Public, Dev |
| Complex anatomy | Micro-calcifications, irregular masses, acoustic shadows | Hidden |

## Dataset Structure

```
ultrasound/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (PICMUS-standard test configurations)
|   +-- ultrasound_challenge_public.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- dev/       20 samples (augmented, medium mismatch)
|   +-- ultrasound_challenge_dev.h5
|   +-- spec.json / true_spec.json
|   +-- images/sample_XX_*/
+-- hidden/    20 samples (adversarial, wide mismatch)
    +-- ultrasound_challenge_hidden.h5
    +-- spec.json / true_spec.json
    +-- images/sample_XX_*/
```

## HDF5 Structure (per sample)

```
sample_XX/
+-- x_true (256, 256) float32         -- Ground-truth tissue reflectivity
+-- bmode_ideal (256, 256) float32    -- Clean B-mode (log-compressed [0,1])
+-- bmode_measured (256, 256) float32 -- Noisy B-mode (log-compressed [0,1])
+-- psf (K, K) float32               -- Nominal PSF at mid-depth
```

## CPU Reconstruction

Wiener deconvolution in the frequency domain:
  F_recon = F_signal * conj(F_psf) / (|F_psf|^2 + K)
where K is the noise regularization parameter.

## Scoring

```
Score = 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * Consistency
```

## References

- Liebgott et al. (2016) "Plane-wave imaging challenge in medical ultrasound" IEEE IUS.
- Perrot et al. (2021) "So you think you can DAS?" IEEE TUFFC 68(2):355-381.
- Matrone et al. (2015) "DMAS" IEEE TUFFC 62(3):537-545.
- Jensen (1996) Field II ultrasound simulation program. MPC 4:351-353.
- CIRS Inc. Multi-Purpose Multi-Tissue Ultrasound Phantom (Model 040GSE).
