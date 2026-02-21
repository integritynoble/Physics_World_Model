## Summary

<!-- Brief description of changes (1-3 sentences) -->

## Contribution Level

<!-- Check one -->
- [ ] **Solver** (Level 1) — new `ReconSolver` for an existing modality
- [ ] **Calibrator** (Level 2) — new calibration method
- [ ] **Modality** (Level 3) — new modality (operator + CasePack + solver + tests)
- [ ] **Primitive** (Level 4) — new OperatorGraph node type (requires RFC)
- [ ] **Other** — bug fix, docs, CI, etc.

## Checklist

### All PRs
- [ ] Tests pass locally: `pytest packages/pwm_core/tests/`
- [ ] No large binary files committed (data generated programmatically)
- [ ] No credentials, API keys, or patient data included

### Solver PRs (Level 1)
- [ ] Implements `run_<solver>(y, physics, cfg) -> (x_hat, info)`
- [ ] Entry added to `packages/pwm_core/contrib/solver_registry.yaml`
- [ ] `pwm contrib check <solver>` passes
- [ ] Benchmark PSNR documented

### Calibrator PRs (Level 2)
- [ ] Implements `calibrate_<method>(y, H_nom, budget) -> (H_hat, info)`
- [ ] Operator correction test added to `packages/pwm_core/benchmarks/test_operator_correction.py`
- [ ] Improvement over uncalibrated baseline documented (dB gain)

### Modality PRs (Level 3)
- [ ] Operator implements `forward()` and `adjoint()`
- [ ] `check_adjoint()` test passes (`<Ax,y> == <x,A^T y>`)
- [ ] CasePack JSON follows naming: `<modality>_<description>_v<N>.json`
- [ ] Entries in all 6 YAML registries
- [ ] ID format: `<domain>_<name>_v<N>` (lowercase, underscores)
- [ ] Benchmark entry in `packages/pwm_core/benchmarks/run_all.py`

### Primitive PRs (Level 4)
- [ ] RFC issue opened and linked: #___
- [ ] Physics justification with adjoint proof provided
- [ ] `PrimitiveOp` adjoint correctness test passes
- [ ] 90-day comment period completed

## Test Results

```
# Paste test output here
pytest packages/pwm_core/tests/ -q
```

## Related Issues

<!-- Link any related issues: Fixes #123, Closes #456 -->
