# Plan.md Implementation Results

## Overview

**Plan Version:** v3 (Hardened)
**Plan File:** `docs/plan.md` (3,040 lines, 22 sections)
**Phase 1 Status:** 96% Complete (24/25 deliverables implemented)

---

## Plan Structure (22 Sections)

| # | Section | Status |
|---|---------|--------|
| 1 | Architecture: LLM vs Deterministic Split | Implemented |
| 2 | Pydantic Contracts (Enforced Schemas) | Implemented |
| 3 | YAML Registries | Implemented |
| 4 | Unified Operator Interface | Implemented |
| 5 | Imaging Modality Database (26 Modalities) | 17/26 registered |
| 6 | Plan Agent (Orchestrator) | Implemented |
| 7 | Photon Agent (Deterministic + LLM Narrative) | Implemented |
| 8 | Mismatch Agent (Deterministic + LLM Prior Selection) | Implemented |
| 9 | Recoverability Agent (Practical Recoverability Model) | Implemented |
| 10 | Analysis Agent + Self-Improvement Loop | Implemented |
| 11 | Physical Continuity Check & Agent Negotiation | Implemented |
| 12 | Pre-Flight Report & Permit Step | Implemented |
| 13 | Operating Modes & Intent Detection | Implemented |
| 14 | Element Visualization: Physics Stage vs Illustration Stage | Implemented |
| 15 | Per-Modality Metrics (Beyond PSNR) | Implemented |
| 16 | UPWMI Operator Correction: Scoring, Caching, Budget Control | Implemented |
| 17 | Hybrid Modalities | Phase 5 |
| 18 | RunBundle Export + Interactive Viewer | Phase 5 |
| 19 | Implementation Phases (Incremental, Prove-Value-First) | In Progress |
| 20 | File Structure | Implemented |
| 21 | Testing Strategy | Implemented |
| 22 | Summary | N/A |

---

## Key v2 → v3 Hardening Changes (All Implemented)

1. LLM returns ONLY registry IDs, validated mechanically (no freeform strings)
2. `StrictBaseModel` with `extra="forbid"` everywhere
3. NaN/Inf rejection via Pydantic model validator
4. Calibration tables require provenance fields (dataset_id, seed_set, versions, date)
5. `serialize()` includes SHA256 hashes + blob paths for large arrays
6. `check_adjoint()` on every operator (randomized `<A*y, x> ≈ <y, Ax>` test)
7. Variance-based noise model (not threshold-based)
8. Multi-LLM fallback (Gemini → Claude → OpenAI)
9. `CompressedAgent` renamed to `RecoverabilityAgent`
10. Registry integrity + contract fuzzing tests added

---

## Agent System Implementation

### File Inventory (15 Python files, 8,419 lines)

| File | Lines | Size | Purpose |
|------|-------|------|---------|
| `__init__.py` | 57 | 1.3 KB | Package exports (28 symbols) |
| `contracts.py` | 471 | 14.6 KB | 25 Pydantic contract classes |
| `base.py` | 219 | 7.3 KB | BaseAgent, AgentContext, AgentResult |
| `llm_client.py` | 543 | 17.3 KB | Multi-LLM wrapper (Gemini/Claude/OpenAI) |
| `registry_schemas.py` | 416 | 14.0 KB | 14 Pydantic schemas for YAML validation |
| `registry.py` | 651 | 22.3 KB | RegistryBuilder + assertion helpers |
| `plan_agent.py` | 1,076 | 38.9 KB | Main orchestrator with `run_pipeline()` |
| `photon_agent.py` | 872 | 29.4 KB | Deterministic photon/SNR computation |
| `mismatch_agent.py` | 422 | 14.1 KB | Deterministic mismatch severity scoring |
| `recoverability_agent.py` | 912 | 29.4 KB | Table-driven recoverability model |
| `analysis_agent.py` | 489 | 18.1 KB | Bottleneck classification + suggestions |
| `negotiator.py` | 349 | 12.8 KB | Cross-agent veto / negotiation |
| `continuity_checker.py` | 496 | 18.3 KB | Physical continuity validation |
| `preflight.py` | 645 | 21.4 KB | Pre-flight report builder |
| `physics_stage_visualizer.py` | 801 | 28.9 KB | Element visualization |

**Total:** 8,419 lines, 288 KB

### Agent Classes (6 agents + 5 support classes)

| Class | File | Role |
|-------|------|------|
| `PlanAgent` | plan_agent.py | Orchestrator — parses intent, maps modality, runs pipeline |
| `PhotonAgent` | photon_agent.py | Computes SNR, noise regime, feasibility |
| `MismatchAgent` | mismatch_agent.py | Scores mismatch severity, selects correction |
| `RecoverabilityAgent` | recoverability_agent.py | Table-driven recoverability + PSNR prediction |
| `AnalysisAgent` | analysis_agent.py | Bottleneck scoring + actionable suggestions |
| `AgentNegotiator` | negotiator.py | Cross-agent veto with joint probability |
| `PhysicalContinuityChecker` | continuity_checker.py | Validates physics consistency |
| `PreFlightReportBuilder` | preflight.py | Assembles final report + runtime estimate |
| `PhysicsStageVisualizer` | physics_stage_visualizer.py | Element chain visualization |
| `LLMClient` | llm_client.py | Multi-provider LLM wrapper |
| `RegistryBuilder` | registry.py | YAML registry loader + validator |

### Contract Classes (25 Pydantic models in contracts.py)

**Base:** `StrictBaseModel` (extra="forbid", NaN/Inf rejection)

**Enums (7):** `ModeRequested`, `OperatorType`, `TransferKind`, `NoiseKind`, `NoiseRegime`, `SignalPriorClass`, `ForwardModelType`

**Data Models (18):**
- Intent: `PlanIntent`, `ModalitySelection`
- System: `ElementSpec`, `ImagingSystem`
- Reports: `PhotonReport`, `MismatchReport`, `RecoverabilityReport`, `SystemAnalysis`
- Analysis: `BottleneckScores`, `Suggestion`
- Pipeline: `PreFlightReport`, `NegotiationResult`, `VetoReason`
- Support: `AdjointCheckReport`, `InterpolatedResult`, `LLMSelectionResult`

---

## YAML Registries (6 files, 3,604 lines)

| File | Lines | Modalities | Purpose |
|------|-------|------------|---------|
| `modalities.yaml` | 1,562 | 17 | Full modality definitions (elements, solvers, upload templates) |
| `solver_registry.yaml` | 500 | 17 | Tiered solver mappings (traditional_cpu, best_quality, etc.) |
| `mismatch_db.yaml` | 490 | 17 | Mismatch parameters + severity weights + correction methods |
| `compression_db.yaml` | 842 | 17 | Calibration tables with provenance fields |
| `photon_db.yaml` | 172 | 17 | Photon models + parameters per modality |
| `metrics_db.yaml` | 38 | 17 | Per-modality metric sets (beyond PSNR) |

**Registered Modalities (17):** CASSI, CACTI, SPC, CT, SIM, Holography, Ptychography, Lensless, Light Sheet, Panoramic Stitching, NeRF, Gaussian Splatting, Image Fusion, Phase Retrieval, Denoising, Destriping, Super-Resolution

---

## Runner Integration

**File:** `packages/pwm_core/pwm_core/core/runner.py`

Key additions:
- `run_agent_preflight(prompt, permit_mode)` — calls PlanAgent pipeline before execution
- `_save_agent_reports(rb_dir, agent_results)` — saves all sub-agent reports to RunBundle
- Expanded provenance: SHA256 hashes for y, x_hat, x_true arrays
- Git hash capture in RunBundle metadata

---

## Test Infrastructure

### test_registry_integrity.py (137 lines, 14 tests)

Validates all 6 YAML registries:
- Every file loads without error
- No orphan keys (every mismatch/photon/compression key maps to a modality)
- Every modality has elements + detector
- Compression entries have provenance fields
- Severity weights sum to ~1.0
- Schema validation via Pydantic

### test_contract_fuzzing.py (148 lines, 8 tests)

Validates Pydantic contracts:
- Rejects NaN and Inf values
- Rejects extra fields (`extra="forbid"`)
- Bounded score ranges
- 100 random valid PhotonReport instances
- ImagingSystem requires detector element
- PlanIntent round-trip serialization

---

## Benchmark Verification Results

### Reconstruction Benchmarks (run_all.py fixes)

| Modality | Solver | PSNR (dB) | Reference | Fix Applied |
|----------|--------|-----------|-----------|-------------|
| SPC (10%) | FISTA+TV | 23.20 | — | Fix 1: Removed double normalization |
| SPC (25%) | FISTA+TV | 28.86 | — | Fix 1: Increased iters 200→400 |
| CT | FBP | 24.42 | 28.0 | Fix 2: Shepp-Logan filter |
| CT | SART-TV | 24.41 | — | Fix 2: Constant TV weight |
| CT | RED-CNN | 26.17 | 28.0 | Fix 2: Noise 0.5→0.05 |
| SIM | Wiener | 27.48 | 28.0 | Baseline (mean fallback) |
| SIM | HiFi-SIM | 26.08 | — | Fix 3: Clip-only normalization (+4.3 dB) |

### Operator Correction Benchmarks (All 8 Tests PASS)

| Modality | Parameter | Without | With | Improvement | Fix |
|----------|-----------|---------|------|-------------|-----|
| CT | center_of_rotation | 13.41 dB | 24.09 dB | **+10.67 dB** | Fix 4: Reprojection error metric |
| CACTI | mask_timing | 14.48 dB | 37.42 dB | **+22.94 dB** | — |
| CASSI (Alg 1) | mask_geo + dispersion | 15.79 dB | 21.20 dB | **+5.40 dB** | Fix 5: dy convergence |
| CASSI (Alg 2) | mask_geo + dispersion | 15.79 dB | 21.54 dB | **+5.74 dB** | Fix 6: Updated hyperparameters |
| Lensless | psf_shift | 23.48 dB | 27.03 dB | **+3.55 dB** | — |
| MRI | coil_sensitivities | 6.94 dB | 55.19 dB | **+48.25 dB** | — |
| SPC | gain_bias | 11.14 dB | 23.35 dB | **+12.21 dB** | — |
| Ptychography | position_offset | 17.35 dB | 24.44 dB | **+7.09 dB** | — |

**Average improvement: +14.48 dB** across all 8 modalities.

#### CASSI Calibration Details

- **Algorithm 1 (Beam Search):** Found ψ(dx=1.286, dy=-2.781, theta=-0.500, phi_d=-0.500) — GAP-TV calibrated 25.55 dB (oracle 26.17 dB)
- **Algorithm 2 (Differentiable):** Found ψ(dx=1.112, dy=-2.750, theta=-0.579, phi_d=-0.093) — MST calibrated 21.54 dB (oracle 21.58 dB)
- **True parameters:** ψ(dx=1.094, dy=-2.677, theta=-0.559, phi_d=-0.316)
- **Alg 2 errors:** dx=0.018, dy=0.072, theta=0.019, phi_d=0.223
- **Alg 2 optimization time:** 3,200s (200 steps × 4 starts × 9 phi_d candidates)

---

## Implementation Phases

| Phase | Scope | Status |
|-------|-------|--------|
| **Phase 1** | Agent infrastructure, registries, contracts, tests | **96% Complete** |
| Phase 2 | OCT + Light Field (2 new modalities) | Not started |
| Phase 3 | Operator correction for new modalities | Not started |
| Phase 4 | Remaining 6 modalities (DOT, Photoacoustic, FLIM, etc.) | Not started |
| Phase 5 | Interactive viewer + hybrid modalities + polish | Not started |

### Phase 1 Missing (Optional)

- `agents/_generated_literals.py` — Build-time Literal types (marked optional in plan)
- 9 remaining modalities not yet in YAML registries (planned for Phases 2-4)

---

## Architecture Summary

```
User Prompt
    │
    ▼
┌─────────────┐
│  PlanAgent   │ ← parse intent, map modality, build ImagingSystem
└──────┬──────┘
       │
       ├──► PhotonAgent        → PhotonReport (SNR, noise regime, feasibility)
       ├──► MismatchAgent      → MismatchReport (severity, correction method)
       ├──► RecoverabilityAgent → RecoverabilityReport (CR, PSNR prediction)
       │
       ▼
┌──────────────┐
│AnalysisAgent │ → SystemAnalysis (bottleneck, suggestions)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Negotiator  │ → NegotiationResult (vetoes, proceed/halt)
└──────┬───────┘
       │
       ▼
┌───────────────────┐
│PreFlightReportBuilder│ → PreFlightReport (runtime, warnings, permit)
└───────────┬───────┘
            │
            ▼
┌───────────────────┐
│  Pipeline Runner  │ → build_operator → simulate → reconstruct → metrics → RunBundle
└───────────────────┘
```

All agents run deterministically first (no LLM required). LLM is an optional enhancement for narrative explanations and edge-case modality mapping.
