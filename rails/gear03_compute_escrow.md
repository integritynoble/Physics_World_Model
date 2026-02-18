# Gear 3: Compute Escrow -- The Ammo

> Pre-committed compute unlocked by clearing targets.

**Status: PARTIAL**

---

## The Principle

Compute escrow pre-commits GPU resources to specific imaging tasks, then releases them upon milestone clearance. This prevents compute waste (brute-force approaches that burn resources without proportional quality gain) and creates economic pressure toward efficient methods. You don't get more compute until you prove the last allocation was well-spent.

---

## PWM Implementation

PWM tracks compute consumption at every level -- from individual calibration runs to full benchmark suites. The harness enforces budgets mechanically and penalizes overruns.

### BudgetState

Every ExperimentSpec includes a `BudgetState` that declares measurement and compute constraints:
- `measurement_budget`: num_bands, num_views, num_frames, compression_ratio
- Compute limits: wall-clock time, GPU-hours, peak memory

### Budget Enforcement in the Harness

| Enforcement Point | Rule | Consequence |
|-------------------|------|-------------|
| Submission declaration | Declare GPU-hours, peak memory, wall-clock per scenario | Required for scoring |
| Runtime monitoring | Track actual consumption during execution | Logged in RunBundle |
| 2x budget threshold | Actual > 2x declared | Disqualification for that scenario |
| Sandbagging detection | Declared < 0.5x actual | Disqualification (compute dishonesty) |

### RoIC: The Efficiency Metric

Return on Imaging Compute (RoIC) = dB recovered per GPU-hour. This is a first-class metric, not an afterthought:
- Tracked per modality, per solver, per calibration algorithm
- Must improve monotonically (methods that burn more compute for the same gain are penalized)
- Enables rational resource allocation: invest compute where RoIC is highest

### Calibration Budget Sweeps

The calibration budget module runs systematic sweeps to characterize the compute-quality tradeoff:
- Vary GPU budget from 1 min to 1 hour
- Measure quality gain at each budget level
- Identify diminishing returns threshold
- Report the Pareto frontier of (budget, quality)

---

## Key Files

| File | Description |
|------|-------------|
| `packages/pwm_core/pwm_core/world/budget.py` | BudgetState definition and enforcement |
| `experiments/pwmi_cassi/cal_budget.py` | Calibration budget sweep experiments |
| `docs/purpose.md` Layer 8 | Compute escrow and RoIC in governance context |
| `docs/targeting_system.md` S6.1 | Declared compute budget as submission requirement |

---

## What's Built

- **BudgetState**: Pydantic model tracking measurement and compute budgets per experiment
- **2x budget enforcement**: Harness disqualifies submissions exceeding 2x declared budget
- **RoIC metric**: Defined and tracked (dB per GPU-hour)
- **Calibration budget sweeps**: Experimental framework for compute-quality tradeoff analysis (CASSI validated)
- **Compute logging**: Every RunBundle records GPU-seconds consumed per stage

---

## What's Next

- **Escrow release protocol**: Define how compute is allocated, metered, and released upon milestone clearance
- **Staged release**: Partial compute released at intermediate milestones (e.g., 50% at rho >= 0.50, full at rho >= 0.80)
- **Cross-modality allocation**: Route compute to modalities with highest marginal RoIC
- **Compute marketplace**: Allow method developers to bid for compute allocation based on projected RoIC

---

## Connections

- **Gear 1 (Targeting System)**: The harness enforces compute budgets and measures RoIC
- **Gear 2 (Outcome Contracts)**: Outcomes trigger compute release -- no results, no more compute
- **Gear 8 (Compute + Energy)**: RoIC makes compute *measurable*, enabling rational co-location decisions
