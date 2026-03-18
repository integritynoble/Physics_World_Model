# Specification: Villin Headpiece HP35 Conformational Sampling

## Domain
domain: 3D molecular configuration space
system: villin headpiece subdomain HP35 (35 residues, ~580 atoms)
dimension: 3N (N ~ 580 atoms, plus solvent)

## Equations
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
#   V_solv = -0.5 * (1/epsilon_in - 1/epsilon_out) * sum q_i*q_j / f_GB(r_ij)
#
# NOTE: This validates conformational sampling near the native state,
# NOT protein folding (which requires millisecond timescales on Anton hardware)

equations: |
  m_i * ddot{r}_i = -grad_i V(r) - gamma * m_i * dot{r}_i + noise_i(t)
  V = V_bond + V_angle + V_dihedral + V_LJ + V_coulomb + V_GB_solvent
  noise: <xi_i(t) xi_j(t')> = 2*gamma*k_B*T*m_i * delta_ij * delta(t-t')

parameters:
  force_field: AMBER_ff14SB
  solvent_model: GB-Neck2  # implicit solvent (Generalized Born)
  temperature: 300         # K
  gamma_friction: 1.0      # ps^-1 (Langevin friction coefficient)
  dt: 2.0e-3              # ps (2 fs timestep, with SHAKE for H-bonds)
  T_simulation: 1.0e6     # ps (1 microsecond)
  T_equilibration: 1.0e4  # ps (10 ns equilibration, discarded)
  cutoff_nonbonded: 999.0  # Angstrom (no cutoff for GB implicit solvent)
  pH: 7.0                 # standard protonation states
  ionic_strength: 0.15    # M (physiological salt concentration for GB)

## Boundary Conditions
# No periodic boundaries (implicit solvent, isolated molecule)
# SHAKE constraints on bonds involving hydrogen
boundary: |
  no_periodic_box: implicit solvent (no PBC needed)
  SHAKE: constrain all bonds to hydrogen
  center_of_mass: remove translational and rotational drift every 1000 steps

## Initial Conditions
# Start from NMR structure (PDB: 1YRF)
# Minimize energy (1000 steps steepest descent)
# Heat from 0 K to 300 K over 100 ps
# Equilibrate at 300 K for 10 ns
initial: |
  coordinates: PDB 1YRF (NMR native structure, first model)
  minimization: 1000 steps steepest descent
  heating: 0 -> 300 K over 100 ps
  equilibration: 300 K, 10 ns (Langevin, discarded from production)

## Observables
# Equilibrium observables that converge on microsecond timescale:
# 1. RMSD from native: <RMSD> and RMSD(t) trajectory
# 2. Radius of gyration: <R_g> (experimental SAXS: 10.5 +/- 0.5 Angstrom)
# 3. Secondary structure content: fraction of alpha-helix vs time
# 4. Ramachandran distribution: (phi, psi) angles for all residues
# 5. B-factors: <delta_r_i^2> per residue (compare to crystallographic B-factors)
# Validated against D.E. Shaw Research trajectory and experimental NMR/SAXS data
observables:
  - RMSD: <RMSD> from native  # Angstrom (target: < 2.0 A)
  - RMSD_trajectory: RMSD(t) for 10000 snapshots  # Angstrom
  - radius_of_gyration: <R_g>  # Angstrom (experimental: 10.5 +/- 0.5)
  - secondary_structure: helical_fraction(t) for 10000 snapshots
  - per_residue_RMSF: RMSF(residue_index) for 35 residues  # Angstrom

## Tolerance
# RMSD from native: <= 2.0 Angstrom
# R_g: within experimental uncertainty (+/- 0.5 Angstrom)
# These are equilibrium observable tolerances, not trajectory-level
tolerance:
  RMSD_absolute: 2.0      # Angstrom
  Rg_absolute: 0.5        # Angstrom
  metric: RMSD_from_native

## Primitives Required
primitives: [evaluate_nonlinear, evolve, sample, couple, constrain]
# N (force evaluation: LJ, Coulomb, bonded), E (Verlet/leapfrog integration),
# S (Langevin thermostat noise), K (bonded/nonbonded force coupling),
# B (SHAKE constraints, COM removal)

## Task Instance Variations
# Vary: protein (HP35, Trp-cage, BBA5, chignolin), force field (ff14SB, CHARMM36m,
#        OPLS-AA), solvent (implicit GB, explicit TIP3P, TIP4P-Ew), temperature
#        (280-340 K), simulation length, pH
# Public: HP35 with ff14SB/GB-Neck2, experimental reference data
# Dev: different temperatures, alternative force fields
# Hidden: explicit solvent (much more expensive), folding from extended state,
#          mutations (point mutants with known experimental DeltaDelta-G)
