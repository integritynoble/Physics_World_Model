# PWM contrib/datasets

This folder contains **dataset adapters** and small example datasets (when licensing allows).
Contributors can add adapters so PWM can load measurements and ground truth consistently.

## Philosophy
- Adapters should be thin wrappers that translate external data formats into:
  - `y` (measurements), `x` (optional ground-truth), and metadata (units, axes, acquisition info)
- Adapters must be deterministic and avoid hidden downloads unless explicitly requested.

## Layout (suggested)
- `adapters/`: python modules that implement `DatasetAdapter`
- `manifests/`: JSON/YAML manifests that describe dataset structure & splits
- `examples/`: tiny synthetic datasets for CI tests

## How to contribute
1. Copy `adapters/template_adapter.py` and implement:
   - `can_handle(path)`  
   - `load(path)` -> dict with `y`, optional `x`, and `meta`
2. Add a short manifest under `manifests/`.
3. Add tests under `packages/pwm_core/tests/` (recommended).

Generated on: 2026-02-01
