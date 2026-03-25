# Workshop Proposal: Open Benchmarking for Computational Imaging

## Proposed for ISBI 2027

### Workshop Title
**Physics World Model: Standardized, Trust-Verified Evaluation Across 172 Imaging Modalities**

### Summary (200 words)

Computational imaging spans medical, scientific, and industrial domains, yet
evaluation practices remain fragmented: each modality develops its own metrics,
each group writes its own benchmarking code, and cross-modality comparisons are
nearly impossible.

We propose a workshop introducing the Physics World Model (PWM), an open-source
framework that standardizes imaging algorithm evaluation through a universal
six-tuple specification (Omega, E, B, I, O, epsilon), a compiled OperatorGraph
representation, and a trust-verified Certificate issued by an automated Judge.

PWM currently covers 172 imaging modalities with 2,732 cataloged algorithms,
from classical filtered backprojection to state-of-the-art diffusion models.
Its 4-scenario evaluation protocol (Ideal, Assumed, Corrected, Oracle)
decomposes performance into recoverability, noise sensitivity, and model
mismatch — providing diagnostic insight beyond aggregate PSNR/SSIM.

The workshop features:
1. Tutorial on the PWM protocol and trust ratchet
2. Hands-on session: evaluate algorithms across CT, MRI, spectral imaging
3. Cross-modality transfer challenge: can a CT solver help MRI?
4. Discussion: toward a shared evaluation standard for IEEE Signal Processing

### Format
- Half-day (4 hours)
- Tutorial (90 min) + hands-on (60 min) + challenge (30 min) + panel (60 min)

### Relevance to ISBI
- Covers signal processing, medical imaging, and computational photography
- Universal primitive decomposition aligns with ISBI's cross-domain scope
- Open-source and CPU-runnable — accessible to all participants

### Key Innovation
- First framework offering trust-verified evaluation certificates across
  multiple imaging modalities with a single protocol
- Cross-modality transfer engine enables novel research directions
