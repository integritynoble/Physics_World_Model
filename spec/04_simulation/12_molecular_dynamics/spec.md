# Specification: Villin Headpiece HP35 Conformational Sampling

> Source: `papers/universal_simulation/benchmark/12_molecular_dynamics/spec.md`

## Equations

```
# Newton's equations of motion with Langevin thermostat:
#   m_i * d^2 r_i / dt^2 = -nabla_i V(r) - gamma * m_i * v_i + sqrt(2*gamma*k_B*T*m_i) * xi(t)
#
# Force field (AMBER ff14SB):
#   V(r) = V_bond + V_angle + V_dihedral + V_vdW + V_elec
#   V_bond = sum k_b (r - r_0)^2
#   V_angle = sum k_a (theta - theta_0)^2
#   V_dihedral = sum V_n (1 + cos(n*phi - delta))
#   V_vdW = sum 4*epsilon * ((sigma/r)^12 - (sigma/r)^6)  (Lennard-Jones)
#   V_elec = sum q_i*q_j / (4*pi*epsilon_0*r_ij)
#
# Implicit solvent: Generalized Born (GB-Neck2) model
```

## Parameters

*See source spec.*

## Observables & Tolerance

# Equilibrium observables that converge on microsecond timescale:
# 1. RMSD from native: <RMSD> and RMSD(t) trajectory
# 2. Radius of gyration: <R_g> (experimental SAXS: 10.5 +/- 0.5 Angstrom)
# 3. Secondary structure content: fraction of alpha-helix vs time
# 4. Ramachandran distribution: (phi, psi) angles for all residues
# 5. B-factors: <delta_r_i^2> per residue (compare to crystallographic B-factors)

**Tolerance**: # RMSD from native: <= 2.0 Angstrom # R_g: within experimental uncertainty (+/- 0.5 Angstrom)

## Variations

Variations
# Vary: protein (HP35, Trp-cage, BBA5, chignolin), force field (ff14SB, CHARMM36m,
#        OPLS-AA), solvent (implicit GB, explicit TIP3P, TIP4P-Ew), temperature
#        (280-340 K), simulation length, pH
# Public: HP35 with ff14SB/GB-Neck2, experimental reference data

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/12_molecular_dynamics/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/12_molecular_dynamics/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/12_molecular_dynamics/spec.md`
