---
omega:
  description: "Domain -- spatial extent, mesh, material properties"
  required: true
  type: object
equations:
  description: "Equations -- forward model, physics parameters, noise model"
  required: true
  type: object
boundary_conditions:
  description: "Boundary conditions -- environmental constraints"
  required: false
  type: object
  default: {}
initial_conditions:
  description: "Initial conditions -- calibration, prior"
  required: false
  type: object
  default: {}
observables:
  description: "Observables -- detector geometry, measurement space"
  required: true
  type: object
tolerance:
  description: "Tolerance -- acceptance threshold (epsilon)"
  required: true
  type: number
spec_version: "1.0.0"
---

# CoreSpec v1 -- Universal Six-Tuple (Omega, E, B, I, O, epsilon)

The canonical PWM problem specification. Any solver accepting the same
six-tuple can consume it. Realizes S1 (finite specifiability).

See `dyson_swarm_strategy.md` S2 for the ExperimentSpec -> CoreSpec mapping.
