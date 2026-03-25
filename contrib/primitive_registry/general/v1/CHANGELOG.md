# General Primitives v1 — Changelog

## v1.0 (2026-03-25) — Initial publication

### Primitive set
12 primitives: Discretize, Transform, Compose, Adjoint, Invert, Regularize,
Sample, Reduce, Broadcast, Differentiate, Stochastize, Validate.

### Reconciliation note vs paper (universal_simulation/paper.tex Table 5)

The paper's Table 5 lists a different set of 12 primitives:
Differentiate, Integrate, Solve, Evaluate, Evolve, Transform, Project,
Sample, Couple, Constrain, Discretize, Optimize.

**4 names are shared**: Discretize, Transform, Sample, Differentiate.

The two sets represent different levels of abstraction:
- **Paper primitives** are *mathematical operations* (Integrate, Solve, Evolve) —
  they describe what computation means in the domain.
- **Registry primitives** are *computational building blocks* (Compose, Adjoint,
  Invert, Regularize) — they describe how to assemble operator graphs.

These are complementary, not rival. The mapping is:

| Paper primitive | Registry equivalent(s) |
|----------------|----------------------|
| Differentiate  | Differentiate (direct) |
| Integrate      | Invert ∘ Differentiate (approximate) |
| Solve          | Invert |
| Evaluate       | Compose + Reduce |
| Evolve         | Compose (time-stepped) |
| Transform      | Transform (direct) |
| Project        | Reduce |
| Sample         | Sample (direct) |
| Couple         | Compose + Broadcast |
| Constrain      | Regularize |
| Discretize     | Discretize (direct) |
| Optimize       | Adjoint + Regularize + Reduce |

### Decision
Keep the registry primitives as-is (they are already published and referenced
by the cross-domain mapping files). The paper will be updated in the next
revision to add a "Computational Realization" table that maps paper primitives
to registry primitives. Both sets are authoritative at their level.
