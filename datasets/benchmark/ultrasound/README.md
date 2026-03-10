# Ultrasound B-Mode Benchmark Dataset

## Overview

B-mode ultrasound imaging benchmark with PSF convolution, speckle noise,
and log-compression forward model. Uses synthetic tissue-mimicking phantoms
with realistic tissue reflectivity values and sub-resolution scatterer texture.

## Forward Model

```
ideal(r)     = |s(r) * PSF(r)|            -- PSF-convolved reflectivity
B(r)         = ideal(r) * speckle + noise  -- speckle + electronic noise
y(r)         = 20 * log10(B(r) / B_max)   -- log-compressed B-mode

where:
  s(r)     -- tissue reflectivity map (ground truth)
  PSF(r)   -- Gaussian point spread function
  speckle  -- Rayleigh-distributed multiplicative noise (N scatterers/cell)
  noise    -- additive Gaussian electronic noise
```

## PSF Model

Gaussian separable PSF:
  - sigma_lateral = f_number * lambda / 2
  - sigma_axial = n_cycles * lambda / 2
  - lambda = c / f (acoustic wavelength)

| Parameter | Value |
|-----------|-------|
| Frequency | 5 MHz |
| Speed of sound | 1540 m/s |
| F-number | 1.5 |
| Pulse cycles | 3.5 |
| Pixel size | 0.15 mm |
| Dynamic range | 60 dB |

## Mismatch Parameters (ThetaSpace)

| Knob | Symbol | Description | Public | Dev | Hidden |
|------|--------|-------------|--------|-----|--------|
| `speed_of_sound_error_pct` | SoS err | Focus error from SoS mismatch | 0-3% | 0-5% | 0-8% |
| `attenuation_dB_cm_MHz` | alpha | Tissue attenuation | 0.3-0.7 | 0.3-0.9 | 0.3-1.2 |
| `speckle_density` | N_s | Scatterers per resolution cell | 10-25 | 8-35 | 5-50 |
| `snr_db` | SNR | Electronic SNR | 30-40 dB | 25-38 dB | 20-35 dB |

## Phantom Types

| Type | Description | Tier |
|------|-------------|------|
| Bright cysts | Hyperechoic cysts with bright walls, tissue background | Public |
| Dark cysts | Anechoic fluid-filled cysts, vessel structures | Public |
| Layered tissue | Skin/fat/muscle/organ layers with interface reflections | Public, Dev |
| Point targets | Grid of point reflectors, wire phantoms | Public |
| Complex anatomy | Micro-calcifications, irregular masses, acoustic shadows | Hidden |

## Dataset Structure

```
ultrasound/
+-- README.md
+-- generate_dataset.py
+-- public/    12 samples (4 cyst + 4 layered + 4 point target)
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
+-- psf (K, K) float32               -- Point spread function used
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

- Perrot et al. (2021) "So you think you can DAS?" IEEE TUFFC 68(2):355-381.
- Matrone et al. (2015) "DMAS" IEEE TUFFC 62(3):537-545.
- Gasse et al. (2017) "IQ-Net" IEEE TUFFC 64(10):1535-1543.
- Jensen (1996) Field II ultrasound simulation program. MPC 4:351-353.
