# Plan: Medical Physics QA/QC Copilot for PWM

## Overview

Extend PWM into clinical medical physics by building a **CT QC Copilot** as
the first vertical, then expanding to PET/CT and SPECT via the same harness
with new CasePacks. PWM becomes a **metric-first QA copilot** for imaging
systems — computing QC metrics directly from reconstructed phantom images,
detecting drift, diagnosing root causes, and generating audit-grade reports,
while a qualified medical physicist retains oversight and sign-off authority.

The OperatorGraph forward model is available but **gated behind
`troubleshoot_mode=true`** — used only when operator-correction or
reprojection-based diagnosis is needed, not for routine QC.

This plan follows the SolveEverything.org framework
(https://solveeverything.org/): PWM provides the **targeting system** (what
to measure, when, and what to investigate next) and the **decision logs**
(immutable, auditable QC records), while the medical physicist provides the
**clinical judgment** and **regulatory sign-off**.

### Deliverables

1. **CT QC Copilot module** in `packages/pwm_core/` (metric-first pipeline)
2. **Rail Paper additions** (clinical medical physics section)
3. **Implementation code** (CasePacks, DICOM ingestion, diagnosis, reports)

---

## Part 1: CT QC Copilot Module Specification

### 1.1 Architecture (Metric-First)

The primary path computes QA metrics **directly from reconstructed DICOM
images** — no forward-model simulation required for routine QC. The
OperatorGraph is a secondary path for troubleshooting.

```
DICOM Phantom Scan (ACR / vendor phantom)
        │
        ▼
┌──────────────────────┐
│  DICOM Ingester       │  ← PHI-safe validation, series selection,
│                       │    canonical resampling, selection log
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  CasePack Loader      │  ← Load ROI geometry, metric set, thresholds,
│  (ACR CT CasePack)    │    report template for this phantom type
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  QA Metric Engine     │  ← Compute metrics from image volumes + metadata
│  (metric-first)       │    (no forward model needed)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Threshold Evaluator  │  ← Apply layered thresholds:
│                       │    standard → scanner → protocol → site
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Baseline Compare     │  ← Compare to immutable CommissioningBundle
│  + Drift Detector     │    (versioned, signed, with service events)
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│  Diagnosis Engine     │  ← Scored evidence features (ring index, cupping
│  (if any metric fails)│    index, streak index) + disambiguation planner
└──────────┬───────────┘
           │                    ┌─────────────────────┐
           │  troubleshoot=true │  CT OperatorGraph    │
           │  ─────────────────►│  (forward model for  │
           │                    │   operator-correction)│
           │                    └─────────────────────┘
           ▼
┌──────────────────────┐
│  Report Generator     │  ← PDF + PhysicistReport.json + evidence/
│                       │    artifacts, extends RunBundle
└──────────┬───────────┘
           │
           ▼
    RunBundle (with clinical_qc namespace)
    ├── physicist_report.json     (structured, versioned)
    ├── physicist_report.pdf      (human-readable, with sign-off block)
    └── evidence/                 (ROI images, trend plots, metric derivations)
```

### 1.2 CasePack Concept

A **CasePack** is the packaged, versioned object that makes a QC workflow
reproducible across sites. Each phantom/test combination has its own CasePack.

```yaml
# CasePack schema
casepack:
  id: "acr_ct_v1.0"
  name: "ACR CT Phantom QC"
  version: "1.0.0"
  author: "PWM Clinical Team"
  phantom_type: "ACR CT 464"

  series_selection:
    # Deterministic rules for identifying the correct DICOM series
    rules:
      - name: "module_1_axial"
        match:
          series_description_contains: ["ACR", "Module 1", "QC"]
          slice_thickness_range: [4.0, 6.0]
          image_type_contains: "AXIAL"
        fallback: "select thinnest axial series through phantom center"
    log_selection: true   # always log why a series was chosen

  roi_definitions:
    # Phantom geometry for ROI placement
    water_roi:
      shape: circle
      center_method: phantom_center_auto  # auto-detect from image
      radius_mm: 20.0
      slice_selection: central
    peripheral_rois:
      shape: circle
      count: 4
      radius_mm: 10.0
      offset_from_center_mm: 60.0
      positions: [12_oclock, 3_oclock, 6_oclock, 9_oclock]
    insert_rois:
      bone: {offset_angle_deg: 90, offset_radius_mm: 50}
      air: {offset_angle_deg: 180, offset_radius_mm: 50}
      acrylic: {offset_angle_deg: 270, offset_radius_mm: 50}
      polyethylene: {offset_angle_deg: 0, offset_radius_mm: 50}

  metric_set:
    - ct_number_water
    - ct_number_bone
    - ct_number_air
    - ct_number_acrylic
    - ct_number_polyethylene
    - geometric_accuracy
    - slice_thickness
    - uniformity
    - noise_std
    - low_contrast_detectability
    - artifact_evaluation
    - spatial_resolution

  threshold_set: "acr_ct_thresholds"  # references threshold YAML
  report_template: "ct_qa_report"     # references report template
  evidence_artifacts:
    - roi_overlay_image
    - uniformity_profile_plot
    - hu_trend_plot
    - noise_trend_plot
```

**CasePack lifecycle:**
- CasePacks are versioned and immutable once published
- Sites can extend (add metrics, tighten thresholds) but not weaken published packs
- New phantom types = new CasePack (PET daily QC, SPECT acceptance, etc.)
- PET/CT and SPECT expansion = **same harness, new CasePacks**

### 1.3 Layered Threshold System

Thresholds are **never hardcoded**. They live in YAML with four override layers:

```yaml
# Threshold resolution order (later layers override earlier):
# 1. standard_default  (ACR/AAPM published)
# 2. scanner_model     (vendor/model-specific known baseline)
# 3. protocol          (kernel, slice thickness, dose level)
# 4. site_override     (local physicist customization)

threshold_layers:
  standard_default:
    ct_number_water:
      pass_range: [-5.0, 5.0]
      unit: HU
      source: "ACR CT Accreditation"
    uniformity:
      pass_range: [0.0, 5.0]
      unit: HU
      source: "ACR CT Accreditation"
    noise_std:
      tolerance_from_baseline_sigma: 2.0
      unit: HU
      source: "AAPM TG-233"
    geometric_accuracy:
      pass_range: [0.0, 2.0]
      unit: mm
      source: "ACR CT Accreditation"
    slice_thickness:
      tolerance_from_nominal: 1.5
      unit: mm
      source: "ACR CT Accreditation"

  # Layer 2: scanner model defaults (optional)
  scanner_model:
    "Siemens SOMATOM Force":
      noise_std:
        tolerance_from_baseline_sigma: 1.5  # tighter for this scanner
    "GE Revolution CT":
      uniformity:
        pass_range: [0.0, 4.0]

  # Layer 3: protocol defaults (optional)
  protocol:
    "adult_abdomen_120kVp_B30f":
      noise_std:
        expected_baseline: 8.5

  # Layer 4: site overrides (physicist-managed)
  site_override:
    # Example: site chooses tighter water HU tolerance
    ct_number_water:
      pass_range: [-3.0, 3.0]
      approved_by: "Dr. Smith, DABR"
      approval_date: "2026-01-15"
```

**In reports**: always show both `(standard threshold)` and `(applied threshold)`
so auditors can see exactly which layer determined pass/fail.

### 1.4 Immutable CommissioningBundle (Baseline)

The baseline "golden state" is an **immutable, versioned, signed** artifact:

```yaml
commissioning_bundle:
  version: "1.0.0"
  scanner_id: "CT-Room3"
  scanner_model: "Siemens SOMATOM Force"
  date: "2025-06-15T10:00:00Z"
  approved_by: "Dr. Smith, DABR"
  service_event: "initial_installation"  # or: tube_change, software_upgrade, etc.

  # Immutability guarantee
  sha256_inputs: "abc123..."    # hash of raw DICOM + protocol metadata
  sha256_outputs: "def456..."   # hash of computed metrics + OperatorGraph state

  metrics:
    ct_number_water: {value: 1.2, unit: HU}
    uniformity: {value: 2.1, unit: HU}
    noise_std: {value: 7.8, unit: HU}
    # ... all metrics

  operator_graph_state:     # scanner parameters at commissioning
    center_of_rotation_offset: 0.02
    hu_calibration_slope: 1.001
    hu_calibration_intercept: -0.3
    # ... all learnable params

  provenance:
    pwm_version: "0.4.0"
    git_hash: "abc1234..."
    casepack: "acr_ct_v1.0"
```

**Rules:**
- New baseline = new version (never overwrite)
- Every baseline records the `service_event` that triggered it
- Baselines are chained: `previous_version` field for audit trail
- Drift detection always compares to the **current active baseline**

### 1.5 Mismatch Library (CT-specific)

Each mismatch type maps to observable artifacts, affected metrics, and
diagnostic tests. Used by the diagnosis engine.

| Mismatch Parameter | Observable Artifact | QA Metric Affected | Diagnostic Test |
|---|---|---|---|
| `center_of_rotation` | Ring/arc artifacts, blur | Uniformity, spatial resolution | Wire phantom or edge test |
| `detector_gain_drift` | Ring artifacts (single/few channels) | Uniformity (bands) | Air scan analysis |
| `hu_calibration_slope` | CT number inaccuracy | CT number accuracy (all inserts) | ACR phantom Module 1 |
| `hu_calibration_intercept` | Systematic HU offset | CT number accuracy | ACR phantom Module 1 |
| `scatter_fraction_drift` | Cupping/capping artifacts | Uniformity (center vs edge) | Uniformity phantom |
| `beam_hardening_residual` | Cupping in uniform phantoms | Uniformity | Water phantom |
| `gantry_tilt_error` | Geometric distortion | Geometric accuracy | ACR phantom Module 1 |
| `slice_thickness_drift` | Partial volume errors | Slice thickness (wire ramp) | ACR phantom Module 1 |
| `noise_floor_increase` | Elevated image noise | Noise (std dev in ROI) | Uniform phantom |
| `detector_dead_channel` | Streak/line artifacts | Visual inspection + uniformity | Air scan analysis |

### 1.6 Scored Diagnosis Engine

When a QA metric fails, the diagnosis engine computes **artifact signatures**
from the image data, scores each mismatch hypothesis, and plans the minimal
disambiguation test.

**Artifact signature features** (computed in `diagnosis_features.py`):

| Feature | Computation | Indicates |
|---|---|---|
| `ring_index` | Azimuthal variance in uniform region | Detector gain drift / CoR offset |
| `cupping_index` | Center-minus-periphery HU difference | Scatter/beam-hardening mismatch |
| `streak_index` | Directional gradient energy | Dead channel / metal artifact |
| `hu_drift_index` | Deviation from baseline across inserts | HU calibration drift |
| `noise_ratio` | Current σ / baseline σ | Tube aging / detector degradation |
| `geometric_distortion_index` | Measured vs expected marker distances | Geometry drift |

**Scoring algorithm:**

```python
def diagnose(failed_metrics, image_features, mismatch_library):
    """
    Scored diagnosis with disambiguation.

    1. Compute artifact features from image data
    2. For each mismatch hypothesis:
       - evidence_score = weighted sum of matching features
       - penalty = features that contradict this hypothesis
       - score = evidence_score - penalty
    3. Rank hypotheses by score
    4. If top-2 scores are within 20%:
       - Plan disambiguation test (the one scan that maximally
         separates the two hypotheses)
    5. Return DiagnosisReport with:
       - Ranked hypotheses + scores + evidence features
       - Recommended next test (if ambiguous)
       - Confidence level (high / moderate / low)
    """
```

**Unit-testable**: synthetic artifact injection (ring-like, cupping-like,
streak-like patterns on clean phantom images) validates that the diagnosis
engine correctly identifies each artifact type.

### 1.7 Report Output (PDF + JSON + Evidence)

Every QC run produces three outputs, all stored within the RunBundle:

**1. `physicist_report.json`** (machine-readable, versioned):
```json
{
  "version": "1.0.0",
  "scanner_id": "CT-Room3",
  "date": "2026-02-19T14:30:00Z",
  "casepack": "acr_ct_v1.0",
  "overall_decision": "PASS",
  "metrics": {
    "ct_number_water": {
      "value": 0.8, "unit": "HU",
      "standard_threshold": [-5.0, 5.0],
      "applied_threshold": [-3.0, 3.0],
      "threshold_layer": "site_override",
      "status": "PASS"
    }
  },
  "diagnosis": null,
  "baseline_ref": "CT-Room3_v1.0_2025-06-15",
  "drift_alerts": [],
  "sha256": "..."
}
```

**2. `physicist_report.pdf`** (human-readable):
- Header: Scanner ID, date, protocol, physicist name, technologist
- Summary: PASS/FAIL overall, with per-metric status (green/yellow/red)
- Metrics table: Current vs baseline vs **both** standard and applied thresholds
- Trend plots: Time series with control limits
- Diagnosis (if any fail): Ranked hypotheses + evidence features + next test
- Signature block: For qualified medical physicist sign-off
- Compliance: Standards references (ACR, AAPM TG-233)

**3. `evidence/` folder**:
- ROI overlay images (showing where measurements were taken)
- Uniformity profile plots
- Trend charts (per metric over time)
- Artifact feature maps (ring index heatmap, etc.)
- Metric derivation logs (how each number was computed)

### 1.8 Integration: Extend RunBundle, Don't Reinvent

Clinical QC runs are **native PWM runs** with a `clinical_qc` task namespace,
not a parallel system:

| Concept | Research PWM | Clinical PWM |
|---|---|---|
| Run artifact | `RunBundle` | `RunBundle` with `task: clinical_qc` |
| Task state | `simulate` / `operator_correction` | `qc_report` (new task state) |
| Input spec | `ExperimentSpec` | `QASpec` (extends ExperimentSpec) |
| Output | Reconstruction + metrics | `PhysicistReport` + evidence + PDF |
| Baseline | Scenario I ideal | `CommissioningBundle` (immutable, signed) |
| Diagnosis | `TriadReport` | `TriadReport` (Gate 3 = calibration drift) |
| Registry | `modalities.yaml` | `modalities.yaml` + CasePack YAML |

The runner, caching, and registry infrastructure is shared. Clinical QC is
just another mode, not another system.

---

## Part 2: Input Mode Mapping

PWM's three existing input modes cover the clinical use case with extensions:

### Mode 1: Natural Language Prompt

**Research**: `"SIM live-cell, low dose, 9 frames, correct PSF mismatch"`

**Clinical**:
```
"CT QC, ACR phantom, scanner Room-3, 120kVp adult abdomen"
"PET/CT daily QC, scanner PET-01, check normalization + uniformity"
"Diagnose ring artifacts on CT-02, last scan showed uniformity fail"
"Show QC trends for CT-Room3 over last 6 months"
```

PlanAgent parses intent → maps to `mode: qc_report` + clinical modality key.

### Mode 2: Structured QASpec (extends ExperimentSpec)

```yaml
mode: qc_report
modality: clinical_ct
scanner:
  id: "CT-Room3"
  manufacturer: "Siemens"
  model: "SOMATOM Force"
protocol:
  kVp: 120
  mA: 200
  slice_thickness_mm: 5.0
  kernel: "B30f"
phantom: acr_ct               # triggers CasePack lookup
casepack: "acr_ct_v1.0"
baseline_ref: "CT-Room3_v1.0_2025-06-15"
troubleshoot_mode: false       # true = activate OperatorGraph
compliance: [ACR, AAPM_TG233]
dicom_path: "/data/qc/2026-02-19/CT-Room3/"
```

### Mode 3: Measured Data + Operator (troubleshooting only)

```
DICOM images (= y)  +  Scanner OperatorGraph (= A, with current params)
→ Reprojection consistency check
→ Estimate mismatch parameters (CoR offset, gain drift)
→ Operator correction
→ Re-evaluate QA metrics with corrected model
```

This mode is activated by `troubleshoot_mode: true` and engages the full
OperatorGraph forward model. Not needed for routine QC.

---

## Part 3: Rail Paper Additions

### 3.1 New Subsection in Section 9 (Roadmap): "Clinical Medical Physics Vertical"

Add after the Three-Phase Roadmap, before the closing paragraph. ~1 page.

**Content outline:**

1. **The natural vertical.** PWM's mismatch detection and 4-scenario protocol
   map directly to clinical QA/QC workflows. The Triad Law applies to
   clinical imaging: Gate 1 (protocol design inadequacy), Gate 2 (dose/count
   budget), Gate 3 (scanner calibration drift). Gate 3 dominates in clinical
   practice — most QA failures are calibration drift, not protocol or dose
   issues — precisely mirroring the research finding.

2. **Standards alignment.** ACR CT accreditation, AAPM TG-126 (PET/CT QA),
   AAPM TG-177 (SPECT/CT QA), AAPM TG-233 (CT QC performance metrics).
   PWM computationally implements existing standards; it does not invent new
   QA criteria.

3. **The copilot model** (SolveEverything.org framing). PWM provides
   Gear 1 (targeting: which scanner needs attention), Gear 6 (decision logs:
   immutable QC records), and Gear 2 (outcome contracts: pass/fail against
   thresholds with full evidence). The qualified medical physicist provides
   clinical judgment, sign-off, and regulatory accountability.
   "Autopilot for QC + Digital Twin for troubleshooting."

4. **CasePack extensibility.** Each phantom/test combination is a versioned
   CasePack (ROIs, metrics, thresholds, report template). Adding PET/CT or
   SPECT requires a new CasePack, not new code — same harness, new content.

5. **Economic argument.** A single medical physicist may oversee 10–50
   scanners. PWM reduces per-scanner QC time from hours to minutes while
   improving consistency, traceability, and accreditation readiness. The
   abundance flywheel applies: each scanner model added lowers the marginal
   cost of the next.

### 3.2 New Entry in Section 10 (Call to Action): "Medical Physicists"

> **Medical physicists.** PWM's mismatch library and CasePack architecture
> provide a ready-made computational backbone for clinical QA/QC programs
> aligned with ACR, AAPM TG-126, TG-177, and TG-233. Contributing a
> scanner-specific mismatch signature — the mapping from parameter drift to
> observable artifact — adds your device model to the PWM diagnostic library
> and benefits every site running that scanner. We invite diagnostic and
> nuclear medical physicists to pilot the CT QC Copilot, validate the
> CasePack thresholds against their institutional data, and contribute to the
> clinical validation program. The SolveEverything.org framework
> (https://solveeverything.org/) provides the governance model: PWM targets,
> the physicist decides.

### 3.3 References to Add

- AAPM TG-233 (CT QC performance metrics)
- AAPM TG-126 (PET/CT QA)
- AAPM TG-177 (SPECT/CT QA)
- ACR CT Accreditation Program (phantom testing requirements)
- SolveEverything.org (https://solveeverything.org/)

---

## Part 4: Implementation Plan

### 4.1 Directory Structure

```
packages/pwm_core/
  pwm_core/
    clinical/                              # NEW: Clinical medical physics
      __init__.py
      ct/
        __init__.py
        dicom_ingester.py                  # PHI-safe DICOM parsing + series selection
        qa_metrics.py                      # Metric computation from image volumes
        diagnosis.py                       # Scored root-cause diagnosis engine
        diagnosis_features.py             # Artifact signature extraction
        baseline.py                        # Immutable CommissioningBundle management
        drift_detector.py                  # Time-series drift detection
        report_generator.py                # PDF + JSON + evidence output
        operator_graph.py                  # CT OperatorGraph (troubleshoot_mode only)
      casepacks/
        __init__.py
        casepack_loader.py                 # Load + validate CasePack definitions
        acr_ct.yaml                        # ACR CT phantom CasePack
      common/
        __init__.py
        scanner_registry.py                # Scanner model database
        threshold_resolver.py              # 4-layer threshold resolution
        phi_filter.py                      # PHI de-identification hooks
        report_templates/
          ct_qa_report.html                # HTML report template
      pet_ct/                              # Phase 2 (stub)
        __init__.py
      spect_ct/                            # Phase 3 (stub)
        __init__.py

  contrib/
    clinical_ct_thresholds.yaml            # Layered thresholds (standard/scanner/protocol)
    clinical_ct_mismatch.yaml              # CT mismatch library
    clinical_pet_ct.yaml                   # PET/CT CasePack (Phase 2 stub)
    clinical_spect_ct.yaml                 # SPECT/CT CasePack (Phase 3 stub)
```

### 4.2 Implementation Sprints (Reordered: Ingestion First)

**Sprint 1: End-to-End Skeleton (DICOM → Minimal Metrics → Report)**

| Step | File | What |
|------|------|------|
| 1 | Directory structure + `__init__.py` | Scaffold |
| 2 | `dicom_ingester.py` | PHI validation, series selection with logged reasoning, canonical resampling, vendor-neutral orientation |
| 3 | `phi_filter.py` | Non-patient enforcement + opt-in de-id for future real-world use |
| 4 | `casepacks/acr_ct.yaml` | ACR CT CasePack (ROIs, series rules, metric set) |
| 5 | `casepack_loader.py` | Load + validate CasePack schema |
| 6 | `qa_metrics.py` (minimal) | Water HU + uniformity only (2 metrics) |
| 7 | `report_generator.py` | HTML → PDF report + `physicist_report.json` + `evidence/` folder |
| 8 | End-to-end test | DICOM directory → 2 metrics → report bundle |

**Sprint 2: Full Metric Suite + Baseline**

| Step | File | What |
|------|------|------|
| 9 | `qa_metrics.py` (full) | All 12 ACR-aligned metrics |
| 10 | `clinical_ct_thresholds.yaml` | 4-layer threshold definitions |
| 11 | `threshold_resolver.py` | Resolve standard → scanner → protocol → site |
| 12 | `baseline.py` | Immutable CommissioningBundle (versioned, signed, chained) |
| 13 | `scanner_registry.py` | Scanner model database with default params |

**Sprint 3: Diagnosis + Drift Detection**

| Step | File | What |
|------|------|------|
| 14 | `diagnosis_features.py` | Ring/cupping/streak/HU-drift/noise indices from images |
| 15 | `diagnosis.py` | Scored hypothesis ranking + disambiguation test planner |
| 16 | `clinical_ct_mismatch.yaml` | Full mismatch library with evidence signatures |
| 17 | `drift_detector.py` | Control charts, Western Electric rules, trend alerts |
| 18 | Synthetic artifact tests | Inject ring/cupping/streak patterns → verify diagnosis |

**Sprint 4: OperatorGraph + Troubleshooting Mode**

| Step | File | What |
|------|------|------|
| 19 | `operator_graph.py` | CT OperatorGraph templates (Tier 1–2 only; Tier 3+ deferred) |
| 20 | Reprojection consistency | Estimate CoR offset from projection data |
| 21 | Integration test | `troubleshoot_mode=true` → operator correction → re-evaluate |

**Sprint 5: Rail Paper + PET/CT Stubs**

| Step | File | What |
|------|------|------|
| 22 | `papers/rail/sections/09_roadmap.tex` | Clinical medical physics subsection |
| 23 | `papers/rail/sections/10_call_to_action.tex` | Medical physicist paragraph |
| 24 | `papers/rail/rail_paper.bib` | AAPM TG references |
| 25 | `clinical_pet_ct.yaml` | PET/CT CasePack stub (NEMA NU-2 metrics, TG-126) |
| 26 | `clinical_spect_ct.yaml` | SPECT/CT CasePack stub (TG-177) |

**Phase D (future): PET/CT Extension**

Same pipeline primitives: ingest → metrics → baseline/drift → diagnosis → report.
New CasePacks per modality:
- PET daily QC CasePack
- PET acceptance CasePack (NEMA NU-2)
- SPECT acceptance CasePack (TG-177)
- Shared "coupled failure" logic for PET/CT (CT attenuation artifacts → PET quantification drift)

### 4.3 Python Module Specs

#### `dicom_ingester.py`
- **PHI safety**: Enforce phantom-only studies (check PatientName, InstitutionName patterns); opt-in `phi_filter.py` hook for future real-world use
- **Series selection**: Apply CasePack `series_selection.rules` deterministically; log `series_selection_reason` into the RunBundle (why this series was chosen, what alternatives existed)
- **Canonical resampling**: Resample to canonical orientation/spacing for vendor-neutral ROI placement; record resampling parameters
- **Output**: `CTScanBundle` Pydantic model with metadata + image volumes + selection log

#### `qa_metrics.py`
- **Input**: `CTScanBundle` + CasePack ROI definitions
- **Metric-first**: Compute metrics directly from reconstructed image volumes (no forward model)
- **Per-metric computation**: Each metric is a standalone function with documented formula and ACR reference
- **Output**: `QAMetricsReport` (Pydantic) with per-metric values, units, derivation metadata

#### `baseline.py`
- **Immutability**: `CommissioningBundle` is frozen after creation; new baseline = new version
- **Signing**: SHA-256 of inputs + outputs; `approved_by` + `service_event` metadata
- **Versioning**: Chained via `previous_version` field; full audit trail
- **Comparison**: Current metrics vs active baseline → per-metric PASS/WARNING/FAIL

#### `diagnosis_features.py`
- **Ring index**: Azimuthal variance in uniform ROI → detector gain / CoR
- **Cupping index**: Center-minus-periphery HU → scatter / beam hardening
- **Streak index**: Directional gradient energy → dead channel / metal
- **HU drift index**: Multi-insert deviation from baseline → HU calibration
- **Noise ratio**: Current σ / baseline σ → tube/detector aging
- **Geometric distortion index**: Measured vs expected distances → geometry drift
- All features are unit-testable with synthetic artifact injection

#### `diagnosis.py`
- **Scored ranking**: Each hypothesis gets `evidence_score - penalty`
- **Disambiguation**: If top-2 within 20%, recommend the minimal scan that separates them
- **Output**: `DiagnosisReport` with ranked hypotheses, scores, evidence features, recommended next test, confidence level

#### `drift_detector.py`
- **Control charts**: Mean, σ, UCL/LCL from baseline
- **Western Electric rules**: Point outside limits, 2-of-3 outside 2σ, run of 7, trend of 7
- **Alert levels**: INFO → WARNING → ACTION_REQUIRED → FAIL
- **Output**: `DriftReport` with alerts, trend data, visualization-ready arrays

#### `report_generator.py`
- **Triple output**: PDF + `physicist_report.json` + `evidence/` folder
- **JSON**: Structured, versioned, with both standard and applied thresholds per metric
- **PDF**: HTML template → weasyprint (no LaTeX dependency for clinical deployments)
- **Evidence**: ROI overlays, trend plots, feature maps, derivation logs
- **Extends RunBundle**: All outputs stored within the standard RunBundle structure with `task: clinical_qc`

#### `threshold_resolver.py`
- **4-layer resolution**: standard_default → scanner_model → protocol → site_override
- **Audit trail**: Report shows which layer determined each threshold
- **Validation**: Site overrides cannot weaken standard thresholds (warning if attempted)

---

## Part 5: Key Design Decisions

### 5.1 Metric-First, Not Model-First

Routine QC = compute metrics from reconstructed images. The OperatorGraph
forward model is a power tool for troubleshooting, not a prerequisite for QC.
This reduces MVP scope and dependency risk.

Monte Carlo (Tier 3+) is **future research**, not in v1.

### 5.2 Copilot, Not Replacement

| PWM Does | Physicist Does |
|---|---|
| Automated QC metric computation | Clinical judgment on edge cases |
| Drift detection and alerting | Decision to accept/reject/investigate |
| Scored root-cause hypotheses | Verification and confirmation |
| Report drafting (PDF + JSON) | Review, sign-off, regulatory responsibility |
| "What test next?" recommendations | Final test selection and execution |
| Trend visualization | Interpretation in clinical context |

### 5.3 Standards-First

Every metric, threshold, and workflow maps to a published standard:
- ACR CT Accreditation Program
- AAPM TG-233 (CT performance metrics)
- AAPM TG-126 (PET/CT QA)
- AAPM TG-177 (SPECT/CT QA)

PWM computationally implements existing standards. It does not invent new QA criteria.

### 5.4 The Triad Law in Clinical Context

| Gate | Research Imaging | Clinical Medical Physics |
|---|---|---|
| Gate 1 (Recoverability) | Compression ratio, null space | Protocol design: insufficient projections, coverage gaps |
| Gate 2 (Carrier Budget) | Photon count, SNR | Dose: too low mAs, too short acquisition |
| Gate 3 (Operator Mismatch) | Mask shift, PSF drift | Scanner calibration: HU drift, geometry offset, detector gain |

**Prediction**: Gate 3 dominates in clinical practice, mirroring the research finding.

### 5.5 SolveEverything.org Gear Mapping

Per the SolveEverything.org framework (https://solveeverything.org/):

| Gear | SolveEverything Name | Clinical PWM Implementation |
|---|---|---|
| 1 | Targeting System | Which scanner needs attention? Which test next? |
| 2 | Outcome Contracts | Pass/fail against ACR/AAPM thresholds with evidence |
| 5 | Data Trusts | Immutable CommissioningBundles + QC time series |
| 6 | Decision Logs | Full RunBundle audit trail for every QC session |
| 7 | Two-Source Rule | Scored diagnosis requires multiple evidence features |
| 9 | Fairness Targets | Consistent QC across all scanners regardless of vendor |

### 5.6 PET/CT and SPECT: Same Harness, New CasePacks

Expansion to PET/CT and SPECT means:
- Same pipeline: ingest → metrics → baseline/drift → diagnosis → report
- New CasePacks: PET daily QC, PET acceptance (NEMA NU-2), SPECT acceptance (TG-177)
- Shared coupled-failure logic: CT attenuation artifacts → PET quantification drift
- No new architecture; the CasePack abstraction handles modality differences

---

## Summary: File Changes

| File | Action | Description |
|---|---|---|
| `packages/pwm_core/pwm_core/clinical/` | CREATE | Clinical module directory tree |
| `packages/pwm_core/pwm_core/clinical/ct/dicom_ingester.py` | CREATE | PHI-safe DICOM parsing |
| `packages/pwm_core/pwm_core/clinical/ct/qa_metrics.py` | CREATE | QA metric computation |
| `packages/pwm_core/pwm_core/clinical/ct/diagnosis.py` | CREATE | Scored root-cause diagnosis |
| `packages/pwm_core/pwm_core/clinical/ct/diagnosis_features.py` | CREATE | Artifact signature extraction |
| `packages/pwm_core/pwm_core/clinical/ct/baseline.py` | CREATE | Immutable CommissioningBundle |
| `packages/pwm_core/pwm_core/clinical/ct/drift_detector.py` | CREATE | Time-series drift detection |
| `packages/pwm_core/pwm_core/clinical/ct/report_generator.py` | CREATE | PDF + JSON + evidence output |
| `packages/pwm_core/pwm_core/clinical/ct/operator_graph.py` | CREATE | CT OperatorGraph (troubleshoot mode) |
| `packages/pwm_core/pwm_core/clinical/casepacks/acr_ct.yaml` | CREATE | ACR CT phantom CasePack |
| `packages/pwm_core/pwm_core/clinical/casepacks/casepack_loader.py` | CREATE | CasePack loader/validator |
| `packages/pwm_core/pwm_core/clinical/common/threshold_resolver.py` | CREATE | 4-layer threshold resolution |
| `packages/pwm_core/pwm_core/clinical/common/phi_filter.py` | CREATE | PHI safety hooks |
| `packages/pwm_core/pwm_core/clinical/common/scanner_registry.py` | CREATE | Scanner model database |
| `packages/pwm_core/pwm_core/clinical/common/report_templates/ct_qa_report.html` | CREATE | Report template |
| `packages/pwm_core/contrib/clinical_ct_thresholds.yaml` | CREATE | Layered QA thresholds |
| `packages/pwm_core/contrib/clinical_ct_mismatch.yaml` | CREATE | CT mismatch library |
| `packages/pwm_core/contrib/clinical_pet_ct.yaml` | CREATE | PET/CT CasePack stub |
| `packages/pwm_core/contrib/clinical_spect_ct.yaml` | CREATE | SPECT/CT CasePack stub |
| `papers/rail/sections/09_roadmap.tex` | EDIT | Clinical medical physics subsection |
| `papers/rail/sections/10_call_to_action.tex` | EDIT | Medical physicist call to action |
| `papers/rail/rail_paper.bib` | EDIT | AAPM TG references |
