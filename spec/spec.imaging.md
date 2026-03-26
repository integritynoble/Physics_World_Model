---
domain: imaging
domain_version: v1.0
extends: spec.core
primitive_registry: imaging/v1
diagnostic_decomposition: ImagingTriadReport
gates:
  - G1_recoverability
  - G2_carrier_budget
  - G3_operator_mismatch
noise_models:
  - poisson
  - gaussian
  - speckle
  - rician
default_solver: traditional_cpu
default_track: correct
four_scenario_protocol: true
---

# Imaging DomainProfile v1

Extends CoreSpec with computational imaging semantics:
- Triad diagnostic decomposition (G1/G2/G3)
- 4-scenario evaluation protocol (Ideal/Assumed/Corrected/Oracle)
- Physics tier classification (geometric -> approximation -> wave -> learned)
