# Gear 2: Outcome Contracts -- The Fuel

> Pay for verified outcomes, not effort hours.

**Status: PARTIAL**

---

## The Principle

Outcome contracts shift incentives from activity to results. Instead of paying for "100 hours of calibration work," you pay for "recovery ratio >= 0.80 on this modality." The targeting system (Gear 1) provides the verification mechanism; outcome contracts provide the economic incentive to clear the target.

---

## PWM Implementation

PWM defines three quantified outcome metrics that any contract can reference. These metrics are mechanically verifiable by the LIP-Arena harness -- no committee, no subjective judgment.

### Core Outcome Metrics

| Metric | Definition | Target |
|--------|-----------|--------|
| Recovery ratio rho | (PSNR_III - PSNR_II) / (PSNR_I - PSNR_II) | >= 0.80 across 20+ modalities |
| Oracle gap | PSNR_I - PSNR_III | <= 2 dB under bounded compute |
| Return on Imaging Compute (RoIC) | dB recovered per GPU-hour of calibration | Tracked per modality, must improve monotonically |

### Outcome Contract Template

A "pay for calibration result" contract specifies:

1. **Outcome metric**: Which metric(s) must be achieved (e.g., rho >= 0.80 on CASSI)
2. **Verification method**: Which harness track and scenario protocol (e.g., Track 1, 4-scenario, 10 scenes)
3. **Compute budget**: Maximum GPU-hours permitted (e.g., 10 GPU-hours per scene)
4. **Payment trigger**: Conditions under which payment is released (e.g., rho >= 0.80 on >= 8/10 scenes)
5. **Tail-risk clause**: Minimum performance on worst-case scenario (e.g., rho >= 0.50 on every scene)
6. **Audit trail**: RunBundle with DR-IS records delivered as proof of work

### Track-Specific Scoring Weights

| Track | Criterion | Weight |
|-------|-----------|-------:|
| Track 1: Correct | Recovery ratio | 0.30 |
| | Parameter recovery | 0.20 |
| | Uncertainty calibration | 0.15 |
| | Tail-risk score | 0.15 |
| | Cross-modality transfer | 0.10 |
| | Compute efficiency (RoIC) | 0.10 |

---

## Key Files

| File | Description |
|------|-------------|
| `docs/purpose.md` Layer 1 | Quantified targets (rho, oracle gap, RoIC) |
| `docs/purpose.md` Layer 8 | Outcome-based evaluation and governance |
| `docs/targeting_system.md` S5 | Anti-Goodhart scoring (gaming penalties, prospective dominance) |
| `docs/targeting_system.md` S3 | Track scoring weights |

---

## What's Built

- **Three core metrics defined**: Recovery ratio, oracle gap, RoIC -- all mechanically computable from 4-scenario results
- **Harness verification**: `pwm evaluate` produces scores that can serve as contract verification
- **Anti-Goodhart protections**: Gaming penalties ensure outcomes reflect genuine capability, not metric hacking
- **Tail-risk emphasis**: Bottom-10% scenarios weighted heavily; average-case gaming is a losing strategy
- **Compute efficiency tracking**: RoIC (dB/GPU-hour) is a first-class metric alongside quality

---

## What's Next

- **Formal contract template**: Codify the outcome contract format as a YAML/JSON schema
- **Payment trigger protocol**: Define the exact conditions and verification steps for outcome clearance
- **Multi-party contracts**: Template for contracts involving method developer + lab partner + compute provider
- **Escrow integration**: Link contract verification to compute escrow release (Gear 3)
- **Modality-specific targets**: Per-modality rho targets reflecting physical difficulty (e.g., CASSI rho >= 0.80, DOT rho >= 0.60)

---

## Connections

- **Gear 1 (Targeting System)**: The harness verifies the outcome -- no harness, no verifiable contract
- **Gear 3 (Compute Escrow)**: Compute is the cost; outcomes trigger escrow release
- **Gear 6 (Decision Logs)**: RunBundle serves as the audit trail proving the outcome was achieved
- **Gear 9 (Fairness Targets)**: Outcomes include tail-risk and cross-modality metrics, not just average PSNR
