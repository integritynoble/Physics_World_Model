# Imaging Primitives v1 — Changelog

## v1.0 (2026-03-25) — Initial publication

### Primitive set
11 primitives: Propagate, Modulate, Project, Encode, Convolve, Accumulate,
Detect, Sample, Disperse, Scatter, Attenuate.

### Reconciliation note vs paper (universal_simulation/paper.tex Methods)

The paper lists: modulate, propagate, disperse, accumulate, project, convolve,
subsample, reflect, detect, noise, encode.

**6 names are shared** (case-insensitive): Propagate, Modulate, Project,
Encode, Convolve, Accumulate, Detect (7), Disperse (8).

**Differences:**
| Paper name | Registry name | Reason for change |
|-----------|--------------|-------------------|
| subsample  | Sample       | Generalised to cover all sampling modes |
| reflect    | Scatter      | Scatter is a superset (includes reflection, refraction, diffraction) |
| noise      | (removed)    | Noise is a parameter of Stochastize in general/v1, not a separate imaging primitive |
| —          | Attenuate    | Added: attenuation is a distinct physics operation missing from paper's list |

### Decision
Registry names are authoritative for implementation. The paper will reference
this changelog in the next revision.
