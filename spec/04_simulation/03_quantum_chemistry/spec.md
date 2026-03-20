# Specification: Helium Atom Ground State Energy (Configuration Interaction)

> Source: `papers/universal_simulation/benchmark/03_quantum_chemistry/spec.md`

## Equations

```
# Time-independent Schrodinger equation:
#   H psi(r1, r2) = E psi(r1, r2)
#
# Hamiltonian (atomic units, Z = 2):
#   H = -1/2 nabla_1^2 - 1/2 nabla_2^2 - Z/r1 - Z/r2 + 1/|r1 - r2|
#
# CI expansion:
#   psi = sum_{ij} c_{ij} phi_i(r1) phi_j(r2)  (antisymmetrized)
#   phi_i = R_{nl}(r) Y_lm(theta, phi)  (hydrogen-like orbitals or STO/GTO basis)
#
# Matrix eigenvalue problem:
#   H_CI c = E c
```

## Parameters

*See source spec.*

## Observables & Tolerance

# 1. Ground state energy E_0 (should be -2.9037 Ha; exact: -2.903724 Ha)
# 2. Ionization energy IE = E(He+) - E(He) (experimental: 24.587 eV)
# 3. Electron density rho(r) = integral |psi|^2 d(other electron)
# 4. <r1> expectation value (mean electron-nucleus distance)
# 5. Correlation energy: E_corr = E_CI - E_HF
observables:

**Tolerance**: # Absolute error in ground state energy: <= 1 mHa (milli-Hartree) # This requires capturing electron correlation beyond Hartree-Fock

## Variations

Variations
# Vary: atom (He, Li+, Be2+), basis set (STO-3G to cc-pVQZ), n_max (2-6),
#        l_max (1-5), method (HF, CISD, Full-CI)
# Public: He with STO-6G, analytical/NIST ground truth
# Dev: Li+ and Be2+ with larger basis sets

## Run

```python
import sys; sys.path.insert(0, '~/Physics_World_Model/pwm/public')
# See papers/universal_simulation/benchmark/03_quantum_chemistry/spec.md for full details

# Load public benchmark data
from pathlib import Path
import numpy as np
public_dir = Path('papers/universal_simulation/benchmark/03_quantum_chemistry/public/')
# Run the simulation task according to the spec above

```

## Full Spec

`papers/universal_simulation/benchmark/03_quantum_chemistry/spec.md`
