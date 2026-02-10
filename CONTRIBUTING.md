# Contributing to Physics World Model (PWM)

Thank you for your interest in contributing to PWM! This guide covers how to
add new modalities, solvers, and participate in community challenges.

## Getting Started

1. Fork the repository and clone your fork.
2. Install development dependencies:
   ```bash
   pip install -e "packages/pwm_core[dev]"
   ```
3. Run existing tests to verify your setup:
   ```bash
   pytest packages/pwm_core/tests/
   ```

## Adding a New Imaging Modality

Adding a modality requires four components: an operator, a CasePack, a solver
entry, and tests. Follow this template PR workflow.

### Step 1: Physics Operator

Create your operator in `packages/pwm_core/pwm_core/physics/<modality>/`.

Your operator must implement the `PhysicsOperator` protocol defined in
`packages/pwm_core/pwm_core/physics/base.py`:

```python
class MyModalityOperator:
    """Forward model for <modality>."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply forward model: y = A(x)."""
        ...

    def adjoint(self, y: np.ndarray) -> np.ndarray:
        """Apply adjoint: x_approx = A^T(y)."""
        ...
```

Key requirements:
- `forward()` and `adjoint()` must be implemented.
- Include a `check_adjoint()` test verifying `<Ax, y> == <x, A^T y>`.
- Operator parameters should be serializable (no lambda functions).

### Step 2: CasePack

Create a CasePack JSON file in `packages/pwm_core/contrib/casepacks/`:

```
packages/pwm_core/contrib/casepacks/<modality>_<description>_v1.json
```

Follow the naming convention from `docs/contracts/registry_conventions.md`:
- Lowercase only, underscores as separators.
- Domain prefix matching the modality name.
- Version suffix `_v<N>`.

The CasePack must include:
- `casepack_version`: `"0.2.1"`
- `id`: Matching the filename (without `.json`)
- `modality`: The modality identifier
- `base_spec`: Default experiment configuration
- `benchmark_results`: Expected PSNR/SSIM from running the default solver

### Step 3: Solver Registry Entry

Add your solver to `packages/pwm_core/contrib/solver_registry.yaml`:

```yaml
<modality>_<solver_name>_v1:
  modality: <modality>
  name: "<Human-Readable Name>"
  tier: traditional_cpu  # or famous_dl, best_quality
  params:
    iters: 50
    # solver-specific parameters
```

### Step 4: Tests

Add tests in `packages/pwm_core/tests/`:

```python
# test_<modality>_operator.py

def test_forward_adjoint():
    """Verify adjoint consistency: <Ax, y> approx <x, A^T y>."""
    op = MyModalityOperator(...)
    x = np.random.randn(...)
    y = np.random.randn(...)
    lhs = np.sum(op.forward(x) * y)
    rhs = np.sum(x * op.adjoint(y))
    assert abs(lhs - rhs) / max(abs(lhs), 1e-8) < 1e-6

def test_reconstruction_quality():
    """Verify solver achieves minimum PSNR on synthetic data."""
    # Generate synthetic data, run solver, check PSNR >= threshold
    ...
```

### Step 5: Benchmark Validation

Add a benchmark entry in `packages/pwm_core/benchmarks/run_all.py` for your
modality and verify it passes:

```bash
python -m packages.pwm_core.benchmarks.run_all --modality <modality>
```

### PR Checklist

- [ ] Operator implements `forward()` and `adjoint()`
- [ ] `check_adjoint()` test passes
- [ ] CasePack JSON is valid and follows naming conventions
- [ ] Solver entry added to `solver_registry.yaml`
- [ ] Unit tests pass: `pytest packages/pwm_core/tests/test_<modality>*.py`
- [ ] Benchmark achieves reasonable PSNR (documented in CasePack)
- [ ] No large binary files committed (all data generated programmatically)
- [ ] ID format follows `<domain>_<name>_v<N>` convention

## Registry Conventions

All registry IDs must follow the format specified in
`docs/contracts/registry_conventions.md`:

- Format: `<domain>_<name>_v<N>`
- Lowercase only, underscores as separators
- No special characters beyond `[a-z0-9_]`
- Monotonic versioning (never delete old versions)

## Weekly Challenges

PWM runs weekly reconstruction challenges. See
`community/CONTRIBUTING_CHALLENGE.md` for participation details.

## Code Style

- Python 3.10+ with type annotations.
- Use `from __future__ import annotations` for forward references.
- Follow existing patterns in the codebase.
- Run `pytest` before submitting PRs.

## Reporting Issues

- Use GitHub Issues with appropriate labels.
- For bugs, include: steps to reproduce, expected behavior, actual behavior.
- For feature requests, describe the use case and proposed approach.

## License

By contributing, you agree that your contributions will be licensed under the
same license as the project.
