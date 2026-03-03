# Modify Plan: photoacoustic

## Current State
- **Category:** medical
- **Carrier:** Acoustic
- **Score key:** medical_ultrasound (via carrier routing)
- **Algorithms:**
  1. DAS (Classical) -- Analytical baseline
  2. PnP-ADMM (PnP) -- Goudarzi et al., 2020
  3. ABLE (Deep Learning) -- Luijten et al., IEEE TMI 2020
  4. MU-Net (Deep Learning) -- Hyun et al., IEEE TUFFC 2022

## Assessment

Photoacoustic imaging (PAI) uses pulsed laser excitation to generate acoustic waves, which are then detected by ultrasound transducers. The carrier routing `("medical", "Acoustic") -> "medical_ultrasound"` routes photoacoustic to the ultrasound algorithm pool. While PAI shares some reconstruction aspects with ultrasound (both use acoustic wave detection and beamforming), the physics is fundamentally different:

- In ultrasound, the transducer both transmits and receives. DAS beamforming is standard.
- In PAI, laser absorption generates acoustic sources within tissue. The reconstruction is a **thermoacoustic inverse problem** -- reconstructing the initial pressure distribution from time-series acoustic measurements.

The algorithms are partially appropriate:
- **DAS** -- Delay-and-Sum is used in PAI as a simple baseline. Appropriate.
- **PnP-ADMM** (Goudarzi et al., 2020) -- referenced as ultrasound PnP, but PnP-ADMM is applicable to PAI. Acceptable.
- **ABLE** (Luijten et al., IEEE TMI 2020) -- this is specifically an ultrasound beamforming network, not designed for PAI. Somewhat mismatched.
- **MU-Net** (Hyun et al., IEEE TUFFC 2022) -- ultrasound-specific network. Mismatched.

More domain-specific PAI algorithms:
- Universal Back-Projection (Xu & Wang, PRE 2005) -- standard PAI reconstruction
- Model-Based (Rosenthal et al., IEEE TMI 2013) -- model-based PAI reconstruction
- Deep-PAI (Hauptmann et al., IEEE TMI 2018) -- learned PAI reconstruction

The mismatch is moderate. DAS and PnP-ADMM are fine, but ABLE and MU-Net are ultrasound-specific.

## Required Changes

Consider adding a variant override in `_algorithm_catalog.py` for `photoacoustic` with PAI-specific algorithms. However, since the reconstruction pipeline (acoustic wave to image) shares significant overlap with ultrasound, the current assignment is not severely wrong. This is a low-priority change.

### Files to modify (optional)
- `platform/pwm_platform/services/benchmark_database/_algorithm_catalog.py` -- optionally add variant override for `photoacoustic` with PAI-specific algorithms (Universal Back-Projection, Model-Based, Deep-PAI, PAT-Transformer)
