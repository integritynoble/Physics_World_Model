# Gear 6: Decision Logs -- The Truth

> Public, unchangeable record of why an AI made a choice.

**Status: BUILT**

---

## The Principle

Decision logs record not just *what* an AI system decided, but *why*. Every calibration adjustment, every parameter change, every reconstruction choice is logged with evidence, confidence, and compute consumed. The log is cryptographically hashed so it cannot be altered after the fact. This makes AI decisions auditable, reproducible, and trustworthy.

---

## PWM Implementation

PWM implements decision logging through two complementary systems: **DR-IS** (Decision Records for Imaging Systems) for individual decisions, and **RunBundle** for complete run audit trails.

### DR-IS: Decision Records for Imaging Systems

Every calibration decision is logged as a DR-IS record:

```json
{
  "timestamp": "2026-Q2-R3-scenario-042",
  "action": "adjust_dx",
  "old_value": 0.0,
  "new_value": 1.47,
  "evidence": "Grid search stage 0: best PSNR at dx=1.5, refined to 1.47",
  "triad_gate": "gate_3_mismatch",
  "confidence": 0.92,
  "compute_consumed_gpu_sec": 85.3,
  "hash": "sha256:a4f2c..."
}
```

**DR-IS fields**:

| Field | Purpose |
|-------|---------|
| `timestamp` | When the decision was made |
| `action` | What parameter was changed |
| `old_value` / `new_value` | Before and after values |
| `evidence` | What data/computation justified the change |
| `triad_gate` | Which Triad gate this decision addresses |
| `confidence` | System's confidence in the decision (0-1) |
| `compute_consumed_gpu_sec` | How much compute was spent reaching this decision |
| `hash` | SHA-256 hash for tamper detection |

### RunBundle: The Complete Audit Trail

Every PWM run produces a RunBundle containing the full chain of evidence:

```
run_{spec_id}_{uuid}/
  artifacts/
    x_hat.npy               # Reconstructed signal
    y.npy                    # Measurements
    x_true.npy               # Ground truth (if available)
    metrics.json             # PSNR, SSIM, runtime
    images/                  # PNG visualizations
  internal_state/
    diagnosis.json           # Diagnosis result
    recon_info.json          # Reconstruction metadata
  agents/                    # Agent report snapshots
  logs/                      # Run logs + DR-IS records
```

**RunBundle manifest** (v0.3.0):
- `version`: Schema version
- `spec_id`: Which ExperimentSpec produced this run
- `provenance`: git_hash, seeds, platform, environment
- `metrics`: All computed metrics
- `artifacts`: File manifest with checksums
- `hashes`: SHA-256 of every artifact

### Integrity Verification

`community/validate.py` verifies RunBundle integrity:
- Recompute SHA-256 hashes and compare to manifest
- Verify DR-IS chain (each record's hash includes the prior record's hash)
- Check that all declared artifacts exist and have correct shapes/dtypes
- Validate that compute consumption matches declarations

---

## Key Files

| File | Description |
|------|-------------|
| `docs/targeting_system.md` S6.3 | DR-IS schema specification |
| `docs/contracts/runbundle_schema.md` | RunBundle v0.3.0 format specification |
| `community/validate.py` | RunBundle integrity verification |
| `docs/purpose.md` Layer 3 | Observability layer (RunBundle, DR-IS, drift monitor) |

---

## What's Built

- **DR-IS schema**: Full specification with timestamp, action, evidence, confidence, compute, hash
- **RunBundle v0.3.0**: Complete audit trail format with manifest, provenance, metrics, artifacts
- **SHA-256 hashing**: Every artifact and decision record cryptographically signed
- **Integrity validation**: `community/validate.py` verifies RunBundle completeness and hash integrity
- **Compute logging**: GPU-seconds consumed logged per decision and per run
- **Agent report snapshots**: All 17 agent modules log their reports into the RunBundle

---

## What's Next

- **DR-IS chaining**: Implement hash chaining where each record includes the prior record's hash (blockchain-style immutability)
- **Drift monitor integration**: Continuous tracking of operator fidelity across runs, with alerts when metrics degrade
- **Cross-run provenance**: Link RunBundles across related experiments (e.g., a calibration run references the diagnostic run that motivated it)
- **External audit tooling**: Standalone verifier that checks RunBundles without requiring PWM installation

---

## Connections

- **Every gear**: Decision logs are the connective tissue -- every gear produces logs, and every gear's outputs are auditable through them
- **Gear 1 (Targeting System)**: Every harness submission must include a complete RunBundle with DR-IS records
- **Gear 2 (Outcome Contracts)**: RunBundles serve as proof of work for contract verification
- **Gear 7 (Two-Source Rule)**: Divergence between solvers is logged as a DR-IS record with escalation evidence
