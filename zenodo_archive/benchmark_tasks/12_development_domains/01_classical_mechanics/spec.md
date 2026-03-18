# Specification: Granular Flow with Frictional Contact (2D)

## Domain
domain: rectangular_box
geometry: [0, L_x] x [0, L_y]
L_x: 0.5  # meters
L_y: 0.3  # meters
dimension: 2

## Equations
# Discrete Element Method (DEM) with Hertz-Mindlin contact model
# For each particle i (i = 1, ..., N):
#   m_i * d^2 x_i / dt^2 = sum_j F_ij^n + sum_j F_ij^t + m_i * g
#
# Normal contact force (Hertz):
#   F_ij^n = k_n * delta_ij^(3/2) * n_ij - gamma_n * v_ij^n
#   delta_ij = max(0, R_i + R_j - |x_i - x_j|)
#
# Tangential contact force (Mindlin with Coulomb friction):
#   F_ij^t = min(|k_t * s_ij - gamma_t * v_ij^t|, mu * |F_ij^n|) * t_ij
#
# Parameters:
#   N = 5000 particles
#   R_i ~ Uniform(1.5mm, 2.5mm)  (polydisperse)
#   rho_particle = 2500 kg/m^3 (glass beads)
#   k_n = 2e6 N/m^(3/2)  (Hertz stiffness)
#   k_t = 0.8 * k_n
#   gamma_n = 50 s^-1  (normal damping)
#   gamma_t = 25 s^-1  (tangential damping)
#   mu = 0.5  (Coulomb friction coefficient)
#   g = [0, -9.81] m/s^2

equations: |
  m_i * ddot{x}_i = sum_j(F_ij_n + F_ij_t) + m_i * g
  F_ij_n = k_n * delta_ij^1.5 * n_ij - gamma_n * v_ij_n
  F_ij_t = min(k_t * s_ij - gamma_t * v_ij_t, mu * |F_ij_n|) * t_ij
  delta_ij = max(0, R_i + R_j - |x_i - x_j|)

parameters:
  N: 5000
  R_mean: 2.0e-3       # m
  R_std: 0.5e-3         # m (uniform spread)
  rho_particle: 2500    # kg/m^3
  k_n: 2.0e6            # N/m^(3/2)
  k_t: 1.6e6            # N/m^(3/2)
  gamma_n: 50           # s^-1
  gamma_t: 25           # s^-1
  mu_friction: 0.5      # dimensionless
  g: [0, -9.81]         # m/s^2

## Boundary Conditions
# Rigid walls on all 4 sides with same Hertz-Mindlin contact
# Bottom wall: fixed
# Top wall: movable (applied stress protocol varies per task instance)
# Left/right walls: fixed
boundary: |
  wall_bottom: y = 0, fixed, Hertz contact with particles
  wall_top: y = L_y, stress-controlled (sigma_top varies per instance)
  wall_left: x = 0, fixed, Hertz contact
  wall_right: x = L_x, fixed, Hertz contact

## Initial Conditions
# Particles randomly placed in a loose packing (phi_0 ~ 0.60)
# with zero velocity
# Gravity settling phase: t = [0, 0.5s] to reach static equilibrium
# Shear/compression phase: t = [0.5s, T_final]
initial: |
  x_i ~ random packing with volume fraction phi_0 = 0.60
  v_i = [0, 0] for all i
  settling_phase: t in [0, 0.5] s (gravity only)
  loading_phase: t in [0.5, T_final] s (applied stress on top wall)

## Observables
# 1. Bulk stress tensor: sigma_ij = (1/V) * sum_contacts (f_c x l_c)
# 2. Volume fraction profile: phi(y) averaged in horizontal strips
# 3. Coordination number: Z = 2*N_contacts / N_particles
# 4. Shear band location (if applicable): y_shear from strain localization
observables:
  - bulk_stress_tensor: sigma_xx, sigma_yy, sigma_xy  # Pa
  - volume_fraction_profile: phi(y) in 20 horizontal bins
  - coordination_number: Z (scalar)
  - kinetic_energy: E_k(t)  # J, should decay to near-zero during settling

## Tolerance
# Relative error in bulk stress tensor components: <= 1e-3
# Relative error in volume fraction profile: <= 1e-2
# Coordination number: absolute error <= 0.1
tolerance:
  stress_relative: 1.0e-3
  phi_relative: 1.0e-2
  Z_absolute: 0.1
  metric: L2_relative_norm

## Primitives Required
primitives: [differentiate, evaluate_nonlinear, evolve, couple, constrain, discretize]
# partial (force gradient), N (Hertz-Mindlin nonlinearity), E (Verlet time-stepping),
# K (contact detection / neighbor lists), B (wall BCs), G (particle positions)

## Task Instance Variations (100 instances x 3 tiers)
# Vary: N (3000-8000), mu (0.2-0.8), applied stress (1-50 kPa),
#        polydispersity (monodisperse to R_std=1mm), gravity angle (0-30 deg)
# Public tier: standard parameters, ground truth provided
# Dev tier: modified friction and particle count, blind evaluation
# Hidden tier: adversarial (near-jamming, extreme polydispersity, tilted gravity)
