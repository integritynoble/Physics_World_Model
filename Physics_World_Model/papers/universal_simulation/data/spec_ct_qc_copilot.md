# Specification: CT QC Copilot — Clinical Decision-Support Workflow

## Domain
domain: 9-step clinical QC workflow for CT scanner fleet
geometry: per-scanner session (one phantom scan → one QC report)
fleet_size: 1 to N scanners (validated at N=30)
dimension: temporal (longitudinal QC sessions per scanner)

## Equations
# The Copilot workflow is a 9-step deterministic pipeline:
#
# Step 1: Phantom scan acquisition           (technician, manual)
# Step 2: DICOM ingestion + PHI validation   (automated, 0.5s)
# Step 3: Metric computation                 (automated, 0.8s)
# Step 4: Threshold evaluation               (automated, 0.01s)
# Step 5: Baseline comparison                (automated, 0.1s)
# Step 6: Drift detection (SPC)              (automated, 5ms)
# Step 7: Root-cause diagnosis if FAIL       (automated, 0.3s)
# Step 8: Report generation                  (automated, 1.2s)
# Step 9: Physicist review & sign-off        (human, ~4 min)
#
# Principle: "system computes, physicist decides"
# Steps 2-8 are fully automated; Steps 1, 9 require human action
#
# Root-cause diagnosis:
#   score(cause_k) = sum_i w_i * indicator(metric_i fails pattern_k)
#   diagnosis = argmax_k score(cause_k)

equations: |
  workflow: [acquire, ingest, compute, threshold, baseline, drift, diagnose, report, review]
  automated_steps: [ingest, compute, threshold, baseline, drift, diagnose, report]
  human_steps: [acquire, review]
  root_cause: score(k) = sum(w_i * match(metric_i, pattern_k))
  diagnosis: argmax_k score(k)
  time_total: sum(step_times) = 4.2 +/- 0.8 min (vs 67 +/- 12 min manual)

parameters:
  n_metrics: 9                    # ACR-aligned
  n_root_causes: 12               # scored diagnosis patterns
  confidence_threshold: 0.7       # minimum score for diagnosis
  report_formats: [JSON, PDF, evidence_artifacts]
  reproducibility: bit-exact (SHA-256 verified)

## Boundary Conditions
# Human-AI responsibility boundary:
#   - System MUST NOT make regulatory decisions
#   - System MUST present all evidence to physicist
#   - Physicist MUST review before sign-off
#   - All automated results are advisory, not prescriptive

boundary: |
  responsibility: system computes, physicist decides
  regulatory: physicist sign-off required for all QC reports
  PHI: DICOM anonymization validated before processing
  audit: immutable log with timestamps, user IDs, actions
  versioning: CasePack version + CommissioningBundle version tracked per session

## Initial Conditions
# Per-scanner: CommissioningBundle baseline from acceptance testing
# Per-fleet: scanner registry with model, protocol set, site thresholds
# Per-session: fresh DICOM series from phantom scan

initial: |
  scanner_baseline: CommissioningBundle (SHA-256 signed, versioned)
  fleet_registry: {scanner_id, model, protocols, site_thresholds}
  session_input: DICOM series (ACR phantom, validated metadata)

## Observables
# Clinical performance metrics:
#   - Time savings: 94% reduction (67 min → 4.2 min per scanner)
#   - Sensitivity: 100% (4/4 drifting scanners detected, N=30 fleet)
#   - Specificity: 100% (0/26 false positives, N=30 fleet)
#   - Physicist-hours saved: 377 hours/year for 30-scanner fleet
#   - Metric agreement: within 1.2 HU (CT number), 0.10 mm (geometric)

observables:
  - per_session: {status: PASS|FAIL, metrics: 9x float, drift_flags: 9x bool, diagnosis: string}
  - per_scanner: {trend: time_series, SPC_chart: Shewhart, last_N_sessions: list}
  - fleet_dashboard: {n_pass: int, n_fail: int, n_drift: int, action_items: list}
  - efficiency: {time_manual_min: 67, time_automated_min: 4.2, reduction_pct: 94}
  - accuracy: {sensitivity: 1.0, specificity: 1.0, HU_agreement: 1.2, mm_agreement: 0.10}

## Tolerance
# The Copilot passes validation if:
#   1. All 9 metrics agree with manual measurement within tolerance
#   2. No false negatives on known-drifting scanners
#   3. No false positives on known-stable scanners
#   4. Bit-exact reproducibility across runs
#   5. Physicist can complete review in < 10 minutes

tolerance:
  metric_agreement: CT_number <= 1.5 HU, geometric <= 0.15 mm
  sensitivity: >= 0.95 (target 1.0)
  specificity: >= 0.95 (target 1.0)
  reproducibility: bit-exact (SHA-256)
  review_time: <= 10 min per scanner
  metric: sensitivity + specificity + reproducibility

## Primitives Required
# Mapping to 12 general computational primitives:
#   evaluate (N):    nonlinear metric extraction (FWHM, LCD counting, artifact scoring)
#   integrate (int): ROI averaging, SPC window statistics
#   constrain (B):   4-layer threshold hierarchy, pass/fail logic
#   transform (F):   Fourier analysis for MTF/spatial resolution
#   discretize (G):  ROI placement, phantom geometry parsing
#   optimize (O):    root-cause scoring engine (argmax over diagnosis patterns)
#   evolve (E):      longitudinal SPC tracking (Western Electric rules over time)
#   couple (K):      multi-metric joint evaluation (all 9 must pass for scanner pass)
primitives: [evaluate, integrate, constrain, transform, discretize, optimize, evolve, couple]
