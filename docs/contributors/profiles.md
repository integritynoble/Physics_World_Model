# Contributor Profiles -- Persona-Based Onboarding Guide

> Find your path onto the rail. Each profile maps a background to a first contribution.

**Version**: 1.0.0
**Status**: ACTIVE
**Rationale**: Different contributors bring different skills. A physics PhD should not follow the same onboarding path as an ML student. This guide matches backgrounds to contribution levels, starter tasks, and mentorship resources.

---

## PWM Contribution Levels

| Level | Name | What You Build | Merge Lane | Recognition |
|-------|------|----------------|------------|-------------|
| **Level 1** | Solver | Reconstruction algorithm `run_<name>(y, physics, cfg)` | Fast Lane (48h auto-merge) | Contributors page |
| **Level 2** | Calibrator | Theta-fitting routine `calibrate_<name>(y, H_nom, budget)` | Fast Lane (48h auto-merge) | Contributors page |
| **Level 3** | Modality | Full imaging modality (graph + mismatch + photon + metrics + meta YAML) | Review Lane (7-day review) | Co-author on benchmark paper |
| **Level 4** | Primitive | New atomic operator in `PRIMITIVE_REGISTRY` | Governance Lane (RFC + 90-day comment) | Co-author + named in RAIL_CONSTITUTION |

---

## Profile 1: ML Student

### Background

You have taken courses in machine learning, optimization, and linear algebra. You are comfortable with Python, NumPy, and PyTorch. You have implemented gradient descent, ADMM, or proximal operators in homework or projects. You may not have domain expertise in any specific imaging modality.

### Recommended Entry: Level 1 -- Solver

Solvers are the fastest path to a contribution. You do not need to understand the physics of any specific modality. You only need to implement `run_<name>(y, physics, cfg) -> (x_hat, info)` using the `LinearLikeOperator` protocol (`physics.forward`, `physics.adjoint`, `physics.x_shape`, `physics.y_shape`).

### First Task

1. Run `pwm scaffold solver my_fista` to generate your solver directory.
2. Implement FISTA, PnP-ADMM, or any proximal method in the generated `solver.py`.
3. Self-test: `python -m contrib.solvers.my_fista.solver`
4. Evaluate: `pwm evaluate --sandbox --modality widefield --solver my_fista`
5. Validate: `pwm contrib check my_fista`
6. Submit PR.

### Time to First Result

**2-4 hours** from clone to sandbox evaluation result.

### Starter Issue Label

`good-first-issue:solver`

### 1-Day Starter Path

| Hour | Activity |
|------|----------|
| 0-1 | Clone repo, `pip install -e "packages/pwm_core[dev]"`, run `pytest packages/pwm_core/tests/ -x` to verify setup |
| 1-2 | Read `packages/pwm_core/contrib/templates/contrib_solver_template.py`. Run the self-test. |
| 2-3 | Scaffold your solver with `pwm scaffold solver <name>`. Implement your algorithm. |
| 3-4 | Run `pwm evaluate --sandbox --modality widefield --solver <name>`. Check PSNR. |
| 4-6 | Try 2-3 more modalities. Tune step size and iterations. |
| 6-8 | Run `pwm contrib check <name>`. Fix any isolation or signature issues. Submit PR. |

### Common Pitfalls

- **Importing forbidden modules**: Your solver must NOT import from `graph.compiler`, `graph.primitives`, or `targeting.*`. The CI will reject it.
- **Accessing ground truth**: Do not use `H_true`, `x_gt`, or any oracle data. The harness enforces this.
- **Hardcoded shapes**: Use `physics.x_shape` and `physics.y_shape`, not literal dimensions.
- **Non-deterministic behavior**: Set random seeds. Results must be reproducible.

### Mentorship Pointers

- Read `docs/contracts/runbundle_schema.md` to understand what the harness produces.
- Study `packages/pwm_core/contrib/templates/contrib_solver_template.py` as the canonical example.
- The `examples/level1_solver/solver.py` reference FISTA implementation is a complete working example.
- Weekly challenges in `community/challenges/` provide concrete reconstruction targets.

---

## Profile 2: Imaging PhD

### Background

You are a doctoral student or postdoc working on a specific imaging modality (e.g., CASSI, FPM, holography, OCT, light-field). You understand the forward physics of your modality deeply. You may have MATLAB or Python code for your system's forward model. You want your modality represented on the PWM benchmark.

### Recommended Entry: Level 3 -- Modality

You have the domain knowledge to define a complete modality. This is the highest-impact individual contribution: once your modality is on the rail, every solver in the ecosystem can be evaluated against it.

### First Task

1. Run `pwm scaffold modality my_modality` to generate the modality directory.
2. Fill in the 5 YAML files: `graph.yaml`, `mismatch.yaml`, `photon.yaml`, `metrics.yaml`, `meta.yaml`.
3. Define the operator graph: which primitives compose your forward model.
4. Define the mismatch space: what parameters can be miscalibrated in your system.
5. Evaluate: `pwm evaluate --sandbox --modality my_modality --solver traditional_cpu`
6. Submit PR.

### Time to First Result

**1-2 days** from clone to sandbox evaluation with an existing solver running on your modality.

### Starter Issue Label

`good-first-issue:modality`

### 1-Day Starter Path

| Hour | Activity |
|------|----------|
| 0-2 | Clone repo, install, read `docs/modality_standards.md` and `docs/modality_pack_spec.md`. Study an existing modality (e.g., widefield, CASSI). |
| 2-4 | Scaffold your modality. Map your system's forward model to PWM primitives. |
| 4-6 | Fill in `graph.yaml` (nodes and edges for your operator graph). |
| 6-8 | Fill in `mismatch.yaml` (what parameters drift in your system). Run scaffold validation. |

### Common Pitfalls

- **Inventing new primitives when existing ones suffice**: Check `PRIMITIVE_REGISTRY` first. Most imaging systems compose from existing primitives (mask, FFT, convolution, subsampling).
- **Forgetting the adjoint**: Every node in your graph must have a valid adjoint. Use the adjoint dot-product test.
- **Unrealistic mismatch ranges**: Calibrate your `mismatch.yaml` parameter ranges against real-world calibration drift from your lab measurements.
- **Missing metadata**: Every modality must include `meta.yaml` with citation, dimensionality, and domain tags.

### Mentorship Pointers

- Start with Level 1 (solver) to learn the harness before attempting Level 3.
- Read `docs/RAIL_CONSTITUTION.md` Article 2.3 for the modality addition process.
- Your modality contribution earns co-authorship on the PWM benchmark paper (see `docs/contributors/CREDITS.md`).
- Stewards with domain expertise can review your mismatch model -- reach out via the issue tracker.

---

## Profile 3: Physicist

### Background

You are a physicist or applied mathematician who works on wave propagation, scattering theory, diffraction, or transport equations. You understand the mathematical structure of imaging operators (linearity, shift invariance, unitarity, adjoint properties). You may want to contribute new atomic operators that do not yet exist in the primitive registry.

### Recommended Entry: Level 4 -- Primitive

Primitives are the atoms of the operator graph. Adding a new primitive (e.g., a wavelet transform, a scattering operator, a non-uniform FFT) requires deep mathematical understanding and affects the entire ecosystem. This is the most impactful and most heavily reviewed contribution.

### First Task

1. Write an RFC issue describing: the physics justification, tier classification, adjoint proof, and at least 2 modalities that benefit.
2. Implement a `PrimitiveOp` with `forward()` and `adjoint()`.
3. Write adjoint correctness tests (dot-product test with multiple random trials).
4. Use the `packages/pwm_core/contrib/templates/tier2_wrapper.py` template if your kernel wraps existing code.
5. Submit RFC, then implementation PR after community discussion.

### Time to First Result

**1-2 weeks** from RFC submission to merged primitive (includes 7-day review minimum and community discussion).

### Starter Issue Label

`rfc:primitive`

### 1-Day Starter Path

| Hour | Activity |
|------|----------|
| 0-2 | Read `docs/RAIL_CONSTITUTION.md` Articles 1.1 and 2.1 to understand frozen vs evolvable components. |
| 2-4 | Study `packages/pwm_core/contrib/templates/new_operator_template.py` and `tier2_wrapper.py`. |
| 4-6 | Draft your RFC: physics justification, adjoint derivation, tier placement, which modalities benefit. |
| 6-8 | Implement a prototype `PrimitiveOp` and run the adjoint dot-product test locally. |

### Common Pitfalls

- **Skipping the RFC**: Primitive contributions require community discussion before implementation. Do not submit a PR without an approved RFC.
- **Incorrect adjoint**: The adjoint must satisfy `<Ax, y> == <x, A^T y>` to machine precision for linear operators. Use the automated dot-product test.
- **Scope creep**: A primitive should do one thing. If your operator is a composition of existing primitives, define it as a graph template, not a new primitive.
- **Breaking existing graphs**: New primitives are additive only. You must not modify existing primitive IDs or behavior.

### Mentorship Pointers

- Level 4 contributions earn co-authorship AND are named in the RAIL_CONSTITUTION (see `docs/contributors/CREDITS.md`).
- Study existing primitives in the registry to understand the tier classification system.
- Reach out to stewards for mathematical review of your adjoint proof.
- Start with Level 1 or Level 2 to build familiarity with the harness before attempting Level 4.

---

## Profile 4: Industry Engineer

### Background

You are a software engineer or systems engineer at a company that builds or uses imaging systems. You are comfortable with Python, CI/CD, and production-grade code. You may have calibration data from real hardware. You want to contribute calibrators, improve existing solvers, or integrate PWM into your workflow.

### Recommended Entry: Level 2 -- Calibrator

Calibrators fit the gap between nominal and actual hardware parameters. Your experience with real hardware calibration data makes you uniquely suited to this task. You implement `calibrate_<name>(y, H_nom, budget) -> (H_hat, info)`.

### First Task

1. Run `pwm scaffold solver my_cal --calibrator` to generate a calibrator directory.
2. Implement your calibration strategy: grid search, Bayesian optimization, gradient-based, or beam search.
3. Self-test: `python -m contrib.calibrators.my_cal.calibrator`
4. Evaluate: `pwm evaluate --sandbox --modality widefield --solver traditional_cpu --calibrator my_cal`
5. Validate: `pwm contrib check my_cal`
6. Submit PR.

### Time to First Result

**4-8 hours** from clone to sandbox evaluation with your calibrator.

### Starter Issue Label

`good-first-issue:calibrator`

### 1-Day Starter Path

| Hour | Activity |
|------|----------|
| 0-2 | Clone repo, install, read `packages/pwm_core/contrib/templates/new_calibrator_template.py`. |
| 2-4 | Study the `H_nom` interface: `get_theta()`, `set_theta()`, `forward()`, `adjoint()`. Read the `examples/level2_calibrator/calibrator.py` reference. |
| 4-6 | Scaffold and implement your calibrator. Test against the toy operator. |
| 6-8 | Run `pwm evaluate --sandbox` with your calibrator. Submit PR. |

### Common Pitfalls

- **Unbounded parameter search**: Always respect the bounds defined in the modality's `mismatch.yaml`. Unbounded search wastes compute and may hit safety brakes.
- **Ignoring the budget**: The `budget` parameter limits compute time. Your calibrator must respect it or be disqualified.
- **Not testing with mismatch**: Test your calibrator on Scenario III (corrected), not just Scenario I (ideal). The whole point is recovering performance under miscalibration.
- **Overfitting to one modality**: A good calibrator generalizes. Test across 3+ modalities.

### Mentorship Pointers

- Read `docs/RAIL_CONSTITUTION.md` Article 1.8 for the frozen calibrator signature.
- The `examples/level2_calibrator/calibrator.py` reference implementation shows a complete grid-search calibrator.
- If you have real calibration data from your hardware, consider contributing a modality (Level 3) as well.
- Industry contributions with real-world validation data are highly valued by the community.

---

## Quick Reference: Which Level Is Right for Me?

| I want to... | Level | Merge Speed | Starter Command |
|--------------|-------|-------------|-----------------|
| Try a new reconstruction algorithm | Level 1 (Solver) | 48h auto-merge | `pwm scaffold solver <name>` |
| Improve calibration for real hardware | Level 2 (Calibrator) | 48h auto-merge | `pwm scaffold solver <name> --calibrator` |
| Add my imaging modality to the benchmark | Level 3 (Modality) | 7-day review | `pwm scaffold modality <name>` |
| Add a new physics primitive | Level 4 (Primitive) | RFC + governance | Open RFC issue first |

---

## CLI Quick Reference

```bash
# Scaffold a new contribution
pwm scaffold solver <name>
pwm scaffold solver <name> --calibrator
pwm scaffold modality <name>

# Evaluate in sandbox mode (fast, small data)
pwm evaluate --sandbox --modality widefield --solver <name>

# Run contribution checks (isolation, signature, forbidden imports)
pwm contrib check <name>
```

---

## References

- `CONTRIBUTING.md` -- General contribution guide
- `docs/RAIL_CONSTITUTION.md` -- Frozen signatures and governance
- `docs/GOVERNANCE.md` -- Three-speed merge authority
- `docs/contributors/CREDITS.md` -- Authorship and recognition policy
- `docs/contracts/registry_conventions.md` -- ID naming conventions
- `packages/pwm_core/contrib/templates/contrib_solver_template.py` -- Solver template
- `packages/pwm_core/contrib/templates/new_calibrator_template.py` -- Calibrator template
- `packages/pwm_core/contrib/templates/tier2_wrapper.py` -- Physics kernel wrapper template
