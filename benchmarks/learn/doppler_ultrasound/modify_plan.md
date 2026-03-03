# Modify Plan: doppler_ultrasound (Doppler Ultrasound)

## Current State

- **Category:** medical
- **Carrier:** Acoustic
- **Score key:** medical_ultrasound (routed via carrier)
- **Algorithms served:**
  1. DAS (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Goudarzi et al., 2020
  3. ABLE (Deep Learning) -- Luijten et al., IEEE TMI 2020
  4. MU-Net (Deep Learning) -- Hyun et al., IEEE TUFFC 2022

## Assessment

Reasonable match. Doppler ultrasound uses the same transducer hardware and
beamforming pipeline as B-mode ultrasound, with the addition of
autocorrelation-based velocity estimation from the Doppler frequency shift.
The carrier-based routing (`("medical", "Acoustic") -> "medical_ultrasound"`)
correctly directs this to the ultrasound pool.

- DAS (Delay-and-Sum) is the standard beamforming baseline applicable to all US.
- PnP-ADMM for ultrasound is valid for image-domain enhancement.
- ABLE and MU-Net are deep learning beamformers trained on US channel data.

The primary Doppler-specific aspect (clutter filtering, velocity estimation)
is not addressed by these algorithms, but the image formation / beamforming
step is well-represented. The mismatch is minor since the benchmark focuses
on the image reconstruction quality rather than velocity accuracy.

## Verdict

No code changes needed. The ultrasound beamforming pool is appropriate for the
image reconstruction aspect of Doppler ultrasound.
