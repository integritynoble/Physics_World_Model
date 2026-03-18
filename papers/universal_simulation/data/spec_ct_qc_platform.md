# Specification: CT Quality Control — Automated Scanner Validation

## Domain
domain: ACR CT phantom image (axial slices)
geometry: 512 x 512 pixels, 9 ROI regions per ACR module
pixel_size: scanner-dependent (typically 0.5-1.0 mm)
dimension: 2 (per-slice analysis across 4 ACR modules)

## Equations
# This is a measurement-validation pipeline, not a forward/inverse problem.
# The "equations" are 9 metric extraction functions applied to phantom images:
#
#   CT_number(ROI_k)    = mean(HU in ROI_k)           for k in {water, bone, air, acrylic, polyethylene}
#   geometric_accuracy  = |d_measured - d_nominal| / d_nominal
#   slice_thickness     = FWHM of wire ramp profile (mm)
#   uniformity          = max(|HU_peripheral - HU_center|) across 4 peripheral ROIs
#   noise               = std(HU in uniform region)
#   LCD                 = count of visible low-contrast objects at each contrast level
#   artifact_eval       = max streak/ring artifact amplitude (HU)
#   spatial_resolution  = highest visible line-pair group (lp/cm)
#   HU_linearity        = max |HU_measured(k) - HU_expected(k)| across materials
#
# Statistical process control:
#   z_metric(t) = (metric(t) - baseline_mean) / baseline_std
#   drift_flag  = Western Electric rules applied to z_metric time series

equations: |
  metrics: [CT_number, geometric_accuracy, slice_thickness, uniformity,
            noise, LCD, artifact_eval, spatial_resolution, HU_linearity]
  SPC: z_score = (metric - baseline_mean) / baseline_std
  drift: Western_Electric_rules(z_score_series)

parameters:
  phantom: ACR CT 464 (or equivalent)
  n_modules: 4                  # ACR phantom modules
  n_metrics: 9                  # ACR-aligned QA metrics
  ROI_diameter: 20 mm           # standard ROI size
  SPC_window: 20                # Shewhart chart lookback
  western_electric_rules: [1_beyond_3sigma, 2_of_3_beyond_2sigma,
                           4_of_5_beyond_1sigma, 8_consecutive_same_side]

## Boundary Conditions
# Four-layer threshold hierarchy (CasePack architecture):
#   Layer 1 (standard):       ACR/AAPM published tolerances
#   Layer 2 (scanner-model):  manufacturer-specific overrides (e.g., GE vs Siemens)
#   Layer 3 (protocol):       protocol-specific (e.g., pediatric head vs adult body)
#   Layer 4 (site-override):  institution-specific policy overrides
#
# Precedence: site > protocol > scanner-model > standard

boundary: |
  threshold_hierarchy: [standard, scanner_model, protocol, site_override]
  CT_number_water: |HU - 0| <= 7 HU (ACR standard)
  geometric_accuracy: |error| <= 1 mm
  slice_thickness: |measured - nominal| <= 1.5 mm
  uniformity: max_deviation <= 7 HU (standard), 5 HU (site override)
  noise: std <= scanner_model_baseline * 1.15
  LCD: >= 4 objects at 6 mm, >= 3 at 4 mm
  spatial_resolution: >= 5 lp/cm (standard), >= 6 lp/cm (site override)

## Initial Conditions
# The CommissioningBundle: immutable baseline snapshot from scanner commissioning
# SHA-256 signed, semantic version chained, full audit trail

initial: |
  baseline: CommissioningBundle v1.0 (SHA-256 signed)
  contents: [baseline_means, baseline_stds, scanner_model, protocol_set]
  version_chain: semantic versioning with cryptographic hash chain
  audit: immutable log of all baseline updates with physicist sign-off

## Observables
# Per-metric: PASS/FAIL status, measured value, threshold, z-score, drift flag
# Per-scanner: overall QC status, root-cause diagnosis (if FAIL), trend summary
# Fleet-level: cross-scanner comparison, fleet-wide drift detection

observables:
  - per_metric: {status: PASS|FAIL, value: float, threshold: float, z_score: float, drift: bool}
  - per_scanner: {overall_status: PASS|FAIL, root_cause: string, trend: SPC_chart}
  - fleet_summary: {n_pass: int, n_fail: int, n_drift: int, flagged_scanners: list}
  - report_outputs: [JSON, PDF, evidence_artifacts]

## Tolerance
# A scanner passes QC if and only if all 9 metrics pass their threshold
# AND no Western Electric drift rule is triggered
# AND the physicist signs off (human-in-the-loop, not automatable)

tolerance:
  per_metric: all 9 metrics within threshold
  drift: no Western Electric rule triggered
  final_decision: qualified medical physicist (QMP) sign-off required
  metric: binary PASS/FAIL per metric, aggregate PASS/FAIL per scanner

## Primitives Required
# Mapping to 12 general computational primitives:
#   evaluate (N):    metric extraction from pixel data (nonlinear: FWHM, LCD counting)
#   integrate (int): ROI averaging (mean HU over region)
#   constrain (B):   threshold evaluation (pass/fail against 4-layer hierarchy)
#   transform (F):   Fourier analysis for spatial resolution (MTF)
#   discretize (G):  ROI placement on phantom geometry
#   optimize (O):    root-cause diagnosis scoring engine
#   evolve (E):      SPC time-series tracking (drift detection over sessions)
primitives: [evaluate, integrate, constrain, transform, discretize, optimize, evolve]
