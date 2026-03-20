# Maturity Levels: L0 to L5

PWM follows the Industrial Intelligence Stack maturation curve. Each level has a clear definition, quantified criteria, and the harness difficulty required for promotion.

---

## Level Definitions

### L0: The Muddle (Pre-PWM)

No agreement on what "good imaging" means. Each lab uses different metrics, test images, and noise models. Results are not comparable. Calibration is manual and undocumented.

- **AI role**: Non-existent
- **Characteristic**: "We got 32 dB on our test set" (incomparable to any other result)

### L1: Measurable (Current -- Partial)

Clear metrics exist. Leaderboards show performance per modality. The 4-scenario protocol provides a common evaluation framework. Results are comparable across labs and methods.

- **AI role**: Referee and scorekeeper (automated evaluation)
- **Characteristic**: "MST-L achieves 35.29 dB Scenario I, 20.82 dB Scenario II, recovery ratio 0.26"
- **PWM status**: Achieved for CASSI (5-param), SPC (1-param). Partial for CACTI.

### L2: Repeatable

Best practices documented as standard operating procedures. Any trained engineer can follow the procedure and get comparable results. Calibration workflows are codified.

- **AI role**: Template assistance and auto-completion
- **Characteristic**: "Follow the CASSI calibration SOP; expected gain +5 dB in 4 minutes"
- **PWM status**: Achieved for CASSI spatial mismatch. Not yet for dispersion or other modalities.

### L3: Automated (Target -- 12-18 months)

The critical inflection point. Checklists become code. PWM executes 80% of calibration work autonomously. Humans handle exceptions and out-of-family cases.

- **AI role**: Primary worker, human handles exceptions
- **Characteristic**: "Submit measurement, receive corrected reconstruction + TriadReport + RunBundle. Recovery ratio > 0.80."
- **Requirements**: 20+ modalities, automated Triad diagnosis, compute-bounded operation, uncertainty-calibrated outputs

### L4: Industrialized (Target -- 24-36 months)

Labs buy ISA outcomes as a service. PWM-compatible agents from multiple providers are interchangeable. The market stops hiring humans for routine calibration.

- **AI role**: Primary worker, humans design new operator families only
- **Characteristic**: Calibration purchased as a service, not performed as a research project
- **Requirements**: Multiple competing agents on the open harness, cross-modality transfer, out-of-family detection

### L5: Commoditized / Solved (Target -- 36+ months)

Multiple providers deliver identical calibration quality at competitive prices. Imaging calibration is as ordinary as auto-exposure in a camera.

- **AI role**: Utility (like electricity)
- **Characteristic**: "Any imaging system self-calibrates on first power-up"
- **Requirements**: 100+ modalities, zero-shot generalization, real-time adaptive calibration
- **Primary metric**: RoIC -- dB per dollar of compute

---

## Current Position: L1 -> L2

| Evidence for L1 | Gaps to L2 |
|-----------------|------------|
| Recovery ratio, oracle gap, RoIC defined | Calibration SOPs not documented for all modalities |
| 4-scenario protocol validated on 3 modalities | Procedures require expert parameter tuning |
| Results comparable across methods (leaderboard) | Working-process docs cover physics but not step-by-step SOPs |
| LIP-Arena harness operational (sealed-simulator) | Live-lab prospective sets not yet active |

---

## Quantified Targets

| Metric | Current | L3 Target | L5 Target |
|--------|---------|-----------|-----------|
| Modalities covered | 64 | 100+ | 200+ |
| Mismatch params per modality | 3-5 | 10+ | Any |
| Recovery ratio rho | 30-50% | 80%+ | 95%+ |
| Oracle gap | 5-12 dB | <= 2 dB | <= 0.5 dB |
| Validated calibration modalities | 3 | 20+ | 100+ |
| Zero-shot generalization | 0% | 50%+ | 90%+ |
| Out-of-family detection | 0% | 90%+ | 99%+ |
| Uncertainty calibration | Not measured | 90% @ 90% CI | 95% @ 95% CI |
| Counterfactual packs published | 0 | 10+ | 50+ |
| RoIC tracking | Defined | Tracked + improving | Commoditized |

---

## What Each Gear Contributes to Each Transition

| Gear | L0 -> L1 | L1 -> L2 | L2 -> L3 | L3 -> L4 | L4 -> L5 |
|------|----------|----------|----------|----------|----------|
| 1. Targeting | Define metrics | Validate protocol | Automate evaluation | Open to external | Rolling rounds |
| 2. Contracts | -- | Define outcomes | Codify templates | Enforce commercially | Commodity pricing |
| 3. Escrow | -- | Track budgets | Enforce budgets | Staged release | Market-based |
| 4. Actions | -- | Software actuation | Automated calibration | Hardware API | Self-calibrating |
| 5. Data Trusts | -- | Registry + synthetic | Partner contributions | Federated evaluation | Open data utility |
| 6. Logs | -- | RunBundle format | DR-IS chaining | Audit compliance | Regulatory standard |
| 7. Two-Source | -- | Multi-solver | Agreement protocol | Independent verification | Redundant utility |
| 8. Energy | -- | Track RoIC | Efficiency targets | Carbon tracking | Clean compute |
| 9. Fairness | -- | Tail-risk scoring | Modality equity | Access tiers | Universal access |
| 10. Literacy | -- | Working-process docs | SOPs + tutorials | Lab onboarding | Self-explanatory |

---

## Reference

Full maturity level definitions: `docs/purpose.md` S"Maturation Levels: L0 to L5"
