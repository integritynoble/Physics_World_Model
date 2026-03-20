# Use Case 4: Scientific Simulation (Examples) — Index

> Physics simulation examples from the PWM papers.
> Each spec defines the physics equations, simulation code, and validation.

## What Is Scientific Simulation?

PWM can generate synthetic physics data for any of its 169 imaging modalities.
These simulations serve as:
1. **Benchmarking**: generate test data with known ground truth
2. **Algorithm development**: test reconstruction algorithms without real hardware
3. **Education**: illustrate the forward model of an imaging modality
4. **Paper examples**: reproduce figures from the PWM papers

## Available Simulation Specs

| Spec File | Domain | Simulation | Validation |
|-----------|--------|-----------|------------|
| [ct_simulation.md](ct_simulation.md) | X-ray CT | Radon transform + Poisson noise | LoDoPaB-CT analytical |
| [mri_simulation.md](mri_simulation.md) | MRI | k-space sampling + noise | fastMRI analytical |
| [optics_simulation.md](optics_simulation.md) | Diffraction Optics | Fresnel propagation | Lommel function |
| [wave_simulation.md](wave_simulation.md) | Acoustics/Seismic | Wave equation (FDTD) | Analytical solutions |

## From the Universal Simulation Paper

The 12-domain benchmark from `papers/universal_simulation/benchmark/`:

| Domain | Spec File Location | Physics |
|--------|--------------------|---------|
| Classical Mechanics | `papers/universal_simulation/benchmark/01_classical_mechanics/spec.md` | DEM granular flow |
| Electromagnetics | `papers/universal_simulation/benchmark/02_electromagnetics/spec.md` | Maxwell's equations |
| Quantum Chemistry | `papers/universal_simulation/benchmark/03_quantum_chemistry/spec.md` | DFT |
| Fluid Dynamics | `papers/universal_simulation/benchmark/04_fluid_dynamics/spec.md` | Navier-Stokes |
| Thermodynamics | `papers/universal_simulation/benchmark/05_thermodynamics/spec.md` | Heat equation |
| Structural Mechanics | `papers/universal_simulation/benchmark/06_structural_mechanics/spec.md` | FEM |
| Chemical Kinetics | `papers/universal_simulation/benchmark/07_chemical_kinetics/spec.md` | ODE systems |
| Epidemiology | `papers/universal_simulation/benchmark/08_epidemiology/spec.md` | SIR/SEIR |
| Optics | `papers/universal_simulation/benchmark/09_optics/spec.md` | Fresnel diffraction |
| Inverse Problems | `papers/universal_simulation/benchmark/10_inverse_problems/spec.md` | CT reconstruction |
| Seismic | `papers/universal_simulation/benchmark/11_seismic/spec.md` | Wave propagation |
| Molecular Dynamics | `papers/universal_simulation/benchmark/12_molecular_dynamics/spec.md` | MD simulation |

## Quick Simulation Example

```python
import sys
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public')
sys.path.insert(0, 'path/to/Physics_World_Model/pwm/public/packages/pwm_core')

# Example: Simulate CT sinogram
from skimage.transform import radon
from skimage.data import shepp_logan_phantom
import numpy as np

phantom = shepp_logan_phantom()
angles  = np.linspace(0, 180, 180, endpoint=False)
sino    = radon(phantom, theta=angles)

print(f"Phantom shape: {phantom.shape}")
print(f"Sinogram shape: {sino.shape}  (angles × detectors)")

# Add Poisson noise (I0=1e5)
I0 = 1e5
sino_noisy = -np.log(np.random.poisson(I0 * np.exp(-sino / sino.max())).astype(float) / I0 + 1e-10)
```
