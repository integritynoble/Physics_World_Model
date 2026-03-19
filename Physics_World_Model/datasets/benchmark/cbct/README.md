# CBCT -- Cone-Beam Computed Tomography

## Overview

Cone-Beam CT benchmark with dental/head phantoms and CBCT-specific mismatch.

## Forward Model

Radon projection (parallel-beam approximation of cone-beam in 2D) + Beer-Lambert
noise model: Poisson(I0=5000) + readout N(0, 3.0^2).

## Mismatch Parameters

| Parameter | Description |
|-----------|-------------|
| `scatter_fraction` | Scatter-to-primary ratio (0.2-0.6) |
| `truncation_fov_factor` | FOV truncation (0.7-1.0) |
| `ring_artifact_amplitude` | Detector non-uniformity (0-0.05) |
| `rotation_offset_deg` | Rotation centre misalignment (0-3 deg) |

## Phantom Types

| Type | Anatomy |
|------|---------|
| dental_panoramic | Mandible, teeth, tongue, airway |
| head_axial | Skull, brain, ventricles, sinuses |
| dental_mixed | Maxilla + mandible, teeth, sinuses |
| head_lower | Skull base, petrous bones, TMJ |

## Tiers

| Tier | Samples | Views | Mismatch |
|------|---------|-------|----------|
| Public | 12 | 180 | Mild |
| Dev | 20 | 180 | Medium |
| Hidden | 20 | 120-240 | Severe + adversarial |

## References

- Feldkamp, Davis & Kress (1984) JOSA A 1:612-619.
- PWM Benchmark: https://pwm.platformai.org/benchmark/cbct
