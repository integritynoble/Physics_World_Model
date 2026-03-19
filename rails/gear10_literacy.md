# Gear 10: Literacy -- The Manual

> Teaching everyone to read a target and check a decision log.

**Status: PARTIAL**

---

## The Principle

The abundance engine only works if people can understand it. Literacy means teaching every stakeholder -- researchers, clinicians, engineers, regulators, patients -- to read a target, check a decision log, and verify a claim. Without literacy, the other nine gears spin in the dark.

---

## PWM Implementation

PWM provides extensive working-process documentation and quickstart guides. The foundation is strong; what's missing is a "civic literacy" guide that teaches non-experts to read PWM's outputs.

### 26 Modality Working-Process Documents

Each validated modality has a dedicated working-process document explaining:
- The physics of the imaging modality
- How PWM models it (OperatorGraph, parameters, solver portfolio)
- Expected performance benchmarks
- Common failure modes and how to diagnose them
- CLI and Python examples

Available modalities with working-process docs:
`widefield` | `widefield_lowdose` | `confocal_livecell` | `confocal_3d` | `sim` | `lightsheet` | `spc` | `cassi` | `cacti` | `lensless` | `ct` | `mri` | `ptychography` | `holography` | `phase_retrieval` | `fpm` | `oct` | `light_field` | `integral` | `flim` | `dot` | `photoacoustic` | `nerf` | `gaussian_splatting` | `matrix` | `panorama`

### Quickstart Guides

`docs/quickstart/` provides entry points for:
- First run (prompt-driven simulation)
- Operator correction mode
- Running benchmarks
- Reading results and RunBundles

### Contribution Guides

| Guide | Audience |
|-------|----------|
| `community/CONTRIBUTING_CHALLENGE.md` | Developers submitting new methods or challenge entries |
| `community/OPEN_CORE_BOUNDARY.md` | Developers understanding what's MIT vs. partner-only |
| `README.md` | Everyone: install, quickstart, modality catalog |

---

## Key Files

| File | Description |
|------|-------------|
| `docs/quickstart/` | Quickstart guide directory |
| `docs/*_working_process.md` (26 files) | Per-modality physics + workflow documentation |
| `community/CONTRIBUTING_CHALLENGE.md` | Challenge submission guide |
| `community/OPEN_CORE_BOUNDARY.md` | Open-core licensing policy |

---

## What's Built

- **26 working-process documents**: One per validated modality, covering physics, modeling, benchmarks, failure modes
- **Quickstart guides**: Getting started with PWM in minutes
- **Contribution guides**: How to submit methods, challenge entries, dataset adapters
- **CLI help**: `pwm run --help`, `pwm evaluate --help`, `pwm view --help`
- **Inline code documentation**: Consistent API across 64 modalities, 43+ solvers

---

## What's Next

### "How to Read a PWM Target" Tutorial

A civic literacy guide teaching non-experts to:

1. **Read a recovery ratio**: "rho = 0.80 means 80% of the mismatch-induced quality loss has been recovered by calibration"
2. **Interpret a TriadReport**: Which gate is binding, what the evidence scores mean, what action to take
3. **Check a RunBundle hash**: Verify that results haven't been tampered with (SHA-256 verification)
4. **Understand a DR-IS record**: What decision was made, why, how much compute it cost, how confident the system was
5. **Read the 4-scenario table**: What Scenarios I-IV mean and how to interpret the PSNR progression

### Additional Literacy Goals

- **Video walkthroughs**: Screen recordings of `pwm evaluate` runs with commentary
- **Glossary**: Centralized definitions of ISA terminology (TriadReport, RunBundle, DR-IS, recovery ratio, oracle gap, RoIC, etc.)
- **Regulatory primer**: How RunBundle + DR-IS maps to audit requirements in medical imaging (FDA, CE marking)
- **Lab onboarding guide**: Step-by-step for a new partner lab to connect to the LIP-Arena evaluation pipeline

---

## Connections

- **Gear 1 (Targeting System)**: Literacy means understanding targets -- what rho >= 0.80 means and how to verify it
- **Gear 6 (Decision Logs)**: Literacy means being able to read and audit DR-IS records and RunBundles
- **Gear 9 (Fairness Targets)**: Equitable participation requires that documentation is accessible to non-experts
