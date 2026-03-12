# CACTI — Coded Aperture Compressive Temporal Imaging

## Overview

CACTI is a computational imaging technique that captures high-speed video
from a single 2D snapshot measurement via a time-varying coded aperture.

This package provides benchmark data for evaluating reconstruction algorithms
under **forward-model mismatch**.

## Forward Model

```
y(h,w) = gain · Σ_t Φ_mismatch(h,w,t) · x(h,w,t) + offset + noise
```

## Dataset Design

| Tier   | Source               | Spatial  | Samples | T values    | Access           |
|--------|----------------------|----------|---------|-------------|------------------|
| Public | CACTI sim videos     | 256×256  | 20      | 8           | Full (GT+spec)   |
| Dev    | Procedural (mild)    | 512×512  | 20      | 8, 16, 32   | Blind (y+spec)   |
| Hidden | Procedural (hard)    | 512×512  | 20      | 8, 16, 32   | Server-only      |

**Dev/Hidden are generated procedurally** — no external datasets. The generator
code + secret seeds (kept private on PWM servers) fully determine each sample.
Even though the generator code is included, the derived datasets are
unreproducible without the private seed manifest.

## Procedural Scene Types

| Recipe ID | Name        | Difficulty | Description                              |
|-----------|-------------|------------|------------------------------------------|
| 0         | urban       | easy-med   | Rectangle blobs + grid patterns          |
| 1         | nature      | easy       | Smooth textures + large soft objects     |
| 2         | textile     | hard       | Near-periodic textures (stripes/checker) |
| 3         | particles   | hard       | Many tiny moving dots                    |
| 4         | thin_struct | hard       | Lines, wires, strokes                    |
| 5         | occlusion   | medium     | Layered objects crossing paths           |
| 6         | cam_shake   | medium     | Strong global camera motion              |

Dev uses mostly: urban, nature, occlusion (easy-medium)
Hidden adds: textile, particles, thin_struct, cam_shake (hard)

## T-value Distribution

- **Dev**: 40% T=8, 40% T=16, 20% T=32
- **Hidden**: 30% T=8, 30% T=16, 40% T=32 (harder, long integration)

## Mismatch Parameters

| Parameter        | Range           | Dev (mild)  | Hidden (severe) |
|------------------|-----------------|-------------|-----------------|
| `mask_dx`        | [0.2, 0.8] px  | 0.35        | 0.65            |
| `mask_dy`        | [0.1, 0.5] px  | 0.20        | 0.40            |
| `mask_rotation`  | [0.0, 0.3] deg | 0.08        | 0.22            |
| `mask_blur`      | [0.0, 0.5] px  | 0.10        | 0.35            |
| `clock_offset`   | [-0.1, 0.1]    | -0.03       | 0.08            |
| `gain_drift`     | [0.95, 1.05]   | 0.98        | 1.04            |
| `offset_drift`   | [-0.02, 0.02]  | -0.01       | 0.015           |

## Scoring

```
Score = 0.4 × PSNR_norm + 0.4 × SSIM + 0.2 × Consistency
```

## References

- Llull et al. "Coded aperture compressive temporal imaging." Opt. Express 2013.
- Yuan et al. "Snapshot compressive imaging." IEEE SPM 2021.
- PWM Benchmark: https://pwm.platformai.org/benchmark/cacti
