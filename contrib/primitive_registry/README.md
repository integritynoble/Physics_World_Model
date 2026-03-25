# Primitive Registry

This directory is the top-level contribution point for domain primitive registries
used by the Physics World Model operator graph compiler.

Each registry defines a set of named primitives for a specific domain or level of
abstraction.  Registries are **append-only** after publication: existing entries
must never be deleted or renamed, only new entries added.

## Directory Layout

```
contrib/primitive_registry/
├── general/v1/           — 12 universal computational primitives
├── imaging/v1/           — 11 physics-of-imaging primitives
├── acoustics/v1/         — 7 acoustics primitives
├── combustion/v1/        — 6 combustion primitives
├── particle_physics/v1/  — 6 particle physics primitives
├── remote_sensing/v1/    — 7 remote sensing primitives
└── mappings/
    └── imaging_to_general/v1/   — cross-walk mappings
```

## Reconciliation

Name mismatches between registry primitives and the paper
(`papers/universal_simulation/paper.tex`) are documented in:

- `general/v1/CHANGELOG.md` — reconciles 12 registry primitives vs 12 paper
  primitives (Table 5); explains the mathematical-vs-computational split.
- `imaging/v1/CHANGELOG.md` — reconciles 11 registry primitives vs 11 paper
  primitives (Methods); documents 4 deliberate renames/additions.

## Adding a new registry

1. Create `contrib/primitive_registry/<domain>/v1/primitives.yaml`.
2. Add cross-walk entries to `contrib/primitive_registry/mappings/<domain>_to_general/v1/mappings.yaml`.
3. Update this README's Directory Layout table.
4. Register the new primitives in `packages/pwm_core/pwm_core/graph/primitives.py`.
