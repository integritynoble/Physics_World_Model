# Gear 8: Compute + Energy -- The Base

> Co-locate data centers with clean energy.

**Status: OUT OF SCOPE**

---

## The Principle

The abundance engine requires massive compute, and that compute requires energy. Co-locating data centers with renewable energy sources (solar, wind, hydro, geothermal) makes the engine sustainable. Without clean energy, scaling compute means scaling emissions.

---

## PWM Implementation

Compute + energy infrastructure is an infrastructure-level concern outside PWM's scope. PWM does not dictate where GPU clusters are located or what powers them. However, PWM makes a critical contribution: **it makes compute consumption measurable**, which is a prerequisite for rational co-location decisions.

### What PWM Provides: Measurability

| Metric | What It Measures | Why It Matters for Energy |
|--------|-----------------|--------------------------|
| RoIC (dB/GPU-hour) | Quality gained per unit of compute | Higher RoIC = less total compute needed |
| GPU-seconds per stage | Compute consumed by each calibration step | Identifies which stages are compute-hungry |
| Budget declarations | Pre-committed compute limits per scenario | Prevents unbounded compute consumption |
| Pareto frontier | Compute-quality tradeoff curve | Shows the efficient frontier -- where to stop investing compute |

### The Efficiency Argument

PWM's emphasis on RoIC creates indirect pressure toward energy efficiency:
- Methods with high RoIC achieve good results with less compute
- The harness penalizes brute-force approaches (2x budget disqualification)
- Calibration budget sweeps identify diminishing-returns thresholds
- Outcome contracts (Gear 2) reward efficiency, not just quality

---

## Key Files

| File | Description |
|------|-------------|
| `docs/purpose.md` Layer 9 | Distribution and maintenance, including compute tracking |
| `packages/pwm_core/pwm_core/world/budget.py` | BudgetState with compute tracking |

---

## What's Built

- **RoIC metric**: Defined and tracked per modality, per solver, per calibration algorithm
- **Compute logging**: GPU-seconds recorded in every RunBundle and DR-IS record
- **Budget enforcement**: Harness disqualifies submissions exceeding 2x declared budget
- **Efficiency as first-class metric**: RoIC carries 10% weight in Track 1 scoring

---

## What's Next

- **Carbon tracking**: Estimate CO2 per RunBundle based on GPU model and data center location (informational, not enforced)
- **Efficiency targets**: Set RoIC improvement targets per quarter alongside quality targets
- PWM's contribution remains making compute *measurable and optimizable*, not infrastructure deployment

---

## Connections

- **Gear 3 (Compute Escrow)**: Escrow mechanism meters compute; energy costs determine the price of that compute
- **Gear 2 (Outcome Contracts)**: RoIC in contracts creates economic incentive for compute efficiency
