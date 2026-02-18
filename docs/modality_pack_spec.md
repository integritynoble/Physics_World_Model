# Modality Pack Specification

> A self-contained, validated, portable definition of a computational imaging modality.

**Version**: 1.0.0
**Status**: ACTIVE

---

## 1. Purpose

A modality pack is a self-contained directory that fully defines a computational imaging modality for the PWM targeting system. It contains everything needed to:

1. Compile an OperatorGraph for the modality
2. Sample realistic mismatch parameters
3. Generate noisy measurements
4. Evaluate solvers on the modality
5. Score results against the correct metrics

Modality packs are the mechanism by which external consortia, industry partners, and domain experts add new modalities to PWM without navigating the full codebase.

---

## 2. Directory Structure

```
my_modality_pack/
  ├── graph.yaml           # REQUIRED: OperatorGraphSpec
  ├── mismatch.yaml        # REQUIRED: Mismatch parameter definitions
  ├── photon.yaml          # REQUIRED: Noise model specification
  ├── metrics.yaml         # REQUIRED: Metric selection and thresholds
  ├── meta.yaml            # REQUIRED: Pack metadata
  ├── README.md            # REQUIRED: Physics description and references
  └── LICENSE              # REQUIRED: MIT, BSD, or Apache-2.0
```

All 7 files are required. A pack missing any file fails validation.

---

## 3. File Specifications

### 3.1 `graph.yaml` -- OperatorGraphSpec

Defines the forward model as a DAG of primitives from `PRIMITIVE_REGISTRY`.

```yaml
# graph.yaml
name: "my_modality"
version: "1.0.0"

nodes:
  - id: "mask"
    primitive_id: "coded_mask"          # Must exist in PRIMITIVE_REGISTRY
    physics_tier: "tier0_geometry"
    params:
      mask_type: "binary_random"
      resolution: [256, 256]

  - id: "dispersion"
    primitive_id: "prism_dispersion"
    physics_tier: "tier1_approx"
    params:
      num_channels: 28
      dispersion_px: 2.0

  - id: "detector"
    primitive_id: "detector_integration"
    physics_tier: "tier0_geometry"
    params:
      detector_size: [256, 310]

edges:
  - from: "mask"
    to: "dispersion"
  - from: "dispersion"
    to: "detector"

x_shape: [28, 256, 256]     # Input signal shape
y_shape: [256, 310]          # Measurement shape
```

**Rules**:
- Every `primitive_id` must exist in `graph/primitives.py` PRIMITIVE_REGISTRY
- DAG must be acyclic
- `x_shape` and `y_shape` must be consistent with the graph topology
- `physics_tier` must be one of: `tier0_geometry`, `tier1_approx`, `tier2_full`, `tier3_learned`

### 3.2 `mismatch.yaml` -- Mismatch Parameters

Defines what can go wrong between the true operator and the nominal operator.

```yaml
# mismatch.yaml
modality: "my_modality"

parameters:
  - name: "dx"
    node_id: "mask"
    description: "Lateral mask shift in pixels"
    range: [-3.0, 3.0]
    distribution: "uniform"
    unit: "pixels"
    severity:
      mild: [-0.75, 0.75]
      moderate: [-1.5, 1.5]
      severe: [-3.0, 3.0]
      catastrophic: [-6.0, 6.0]

  - name: "dispersion_error"
    node_id: "dispersion"
    description: "Error in dispersion calibration"
    range: [-0.5, 0.5]
    distribution: "normal"
    distribution_params: { mean: 0.0, std: 0.2 }
    unit: "pixels/channel"
    severity:
      mild: { std: 0.05 }
      moderate: { std: 0.1 }
      severe: { std: 0.2 }
      catastrophic: { std: 0.5 }

compound_mismatch:
  - name: "realistic"
    description: "Simultaneous mask shift + dispersion error"
    parameters: ["dx", "dispersion_error"]
```

**Rules**:
- Every `node_id` must reference a node in `graph.yaml`
- Severity levels must include all 4: mild, moderate, severe, catastrophic
- At least 1 parameter required
- Compound mismatch definitions are optional but encouraged

### 3.3 `photon.yaml` -- Noise Model

```yaml
# photon.yaml
modality: "my_modality"

noise_model:
  type: "poisson_gaussian"     # poisson | gaussian | poisson_gaussian
  params:
    photon_count: 1000         # Average photons per pixel
    read_noise_std: 5.0        # Read noise standard deviation (electrons)
    dark_current: 0.1          # Dark current (electrons/pixel/second)

snr_range:
  low: 10                     # dB
  typical: 25                 # dB
  high: 40                    # dB
```

### 3.4 `metrics.yaml` -- Metric Selection

```yaml
# metrics.yaml
modality: "my_modality"

primary_metrics:
  - "psnr"
  - "ssim"

secondary_metrics:
  - "sam"           # Spectral Angle Mapper (for spectral modalities)
  - "lpips"         # Perceptual quality

thresholds:
  rho_minimum: 0.30          # Below this = safety brake (per targeting_system.md S6.4)
  oracle_gap_target: 2.0     # dB -- "good enough" calibration
```

### 3.5 `meta.yaml` -- Pack Metadata

```yaml
# meta.yaml
name: "my_modality"
version: "1.0.0"
description: "Brief description of the modality and its applications"
domain: "spectral"            # spectral | temporal | spatial | medical | microscopy | other
author: "Alice Chen"
affiliation: "MIT"
contact: "alice@mit.edu"
license: "Apache-2.0"

references:
  - title: "Original paper describing this imaging system"
    doi: "10.1234/example"
  - title: "Calibration method paper"
    url: "https://arxiv.org/abs/..."

supported_solvers:            # Known-working solvers (optional but helpful)
  - "gap_tv"
  - "fista_tv"
  - "pnp_admm"

status: "experimental"        # experimental | candidate | stable
```

---

## 4. Validation

### 4.1 Installation Command

```bash
pwm install-modality ./my_modality_pack
```

### 4.2 Validation Steps

The installer runs these checks in order:

| Step | Check | Failure Action |
|------|-------|---------------|
| 1 | All 7 required files present | Reject with list of missing files |
| 2 | All YAML files parse without errors | Reject with parse error details |
| 3 | `meta.yaml` has all required fields | Reject with missing field list |
| 4 | `graph.yaml` references only existing primitives | Reject with unknown primitive IDs |
| 5 | `graph.yaml` DAG is acyclic | Reject with cycle description |
| 6 | `mismatch.yaml` node_ids match `graph.yaml` nodes | Reject with mismatched IDs |
| 7 | Graph compiles successfully via `graph/compiler.py` | Reject with compiler error |
| 8 | Sandbox evaluation completes with at least 1 baseline solver | Reject with evaluation error |
| 9 | `LICENSE` is MIT, BSD, or Apache-2.0 | Reject (required for registry inclusion) |
| 10 | `test_registry_integrity.py` passes after insertion | Reject with integrity error |

### 4.3 Post-Installation

On successful validation:

1. Entries added to: `modalities.yaml`, `mismatch_db.yaml`, `photon_db.yaml`, `metrics_db.yaml`
2. Solver registry updated with supported solvers
3. Modality available via `pwm evaluate --modality my_modality`

### 4.4 Local vs Registry

| Mode | Command | Visible to others? |
|------|---------|-------------------|
| **Local install** | `pwm install-modality ./pack` | No -- local only |
| **Registry PR** | `gh pr create` with pack in `contrib/modalities/` | Yes -- after merge |

---

## 5. Versioning

| Rule | Detail |
|------|--------|
| **Pack version** | Semantic versioning (major.minor.patch) |
| **Breaking changes** | Major version bump required if graph topology, mismatch params, or metrics change |
| **Additive changes** | Minor version bump for new mismatch parameters, new metrics, new supported solvers |
| **Fixes** | Patch version for documentation, typo fixes, parameter corrections |

---

## 6. Version History

| Version | Date | Change | Authority |
|---------|------|--------|-----------|
| 1.0.0 | 2026-02-18 | Initial modality pack specification | PWM core team |

---

## References

- `docs/RAIL_CONSTITUTION.md` Article 2.3 -- New modality process
- `rails/gear01_targeting_system.md` -- Harness architecture
- `graph/compiler.py` -- OperatorGraph compiler
- `graph/primitives.py` -- PRIMITIVE_REGISTRY
- `contrib/modalities.yaml` -- Existing modality registry
