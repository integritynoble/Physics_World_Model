# PWM Stage 1 Purpose: Imaging System Autonomy

## Discipline Definition

Stage 1 defines a new discipline -- **Imaging System Autonomy (ISA)** -- where imaging performance is determined by operator-level reasoning, diagnosis, and correction rather than solver-only optimization.

> **Physical generalization** = transfer across modalities and conditions by reusing operator-level abstractions (OperatorGraph + parameter priors), not dataset-specific heuristics.

Imaging System Autonomy subsumes "computational imaging" by adding two missing layers: physics-grounded diagnosis (why did reconstruction fail?) and autonomous correction (what minimal intervention fixes it?). Traditional computational imaging optimizes the solver; ISA optimizes the operator.

---

## Canonical Purpose Statement

The purpose of Stage 1 of the Physics World Model is to establish Imaging System Autonomy as a **science of physical generalization** by building autonomous, physics-grounded systems that can represent, diagnose, and correct any imaging modality expressible in the OperatorGraph IR.

Through a universal **OperatorGraph** intermediate representation, PWM enables modality-agnostic reasoning across diverse physical carriers. By autonomously inferring effective forward operators, identifying dominant physical bottlenecks, and computing minimal corrective interventions, PWM transforms reconstruction from an artisanal practice into a **reproducible engineering outcome**.

Stage 1 culminates in **Imaging System Autonomy**: the ability of any instrument within a declared operator family and tolerance envelope to self-specify, self-diagnose, and self-correct, ensuring reliable information recovery whenever permitted by physical law.

### Falsifiable Completion Criterion

> Stage 1 is achieved when, given only measurements and a nominal OperatorGraph family, PWM can infer an effective operator, diagnose the dominant gate, and produce a reproducible RunBundle whose corrected reconstruction is within 2 dB of oracle performance -- under bounded compute (declared GPU-hours and wall-clock budget), without ground truth, and with auditable uncertainty estimates.

### One-Line Version

> Stage 1 of PWM turns computational imaging into a self-explanatory, self-correcting physical system, enabling universal generalization across modalities and conditions through operator-level inference.

---

## Formal Stage-1 Objective

For any physical imaging system $S$ within a declared operator family, PWM must:

1. **Compile a Universal Physics IR** -- Automatically map diverse physical carriers (photons, electrons, spins, acoustic waves, neutrons) into a canonical OperatorGraph that standardizes interfaces for simulation, diagnosis, and correction.

2. **Operationalize the Triad Law** -- Decompose every imaging failure into three measurable gates:
   - **Recoverability** (sampling geometry): Does the measurement encode enough information?
   - **Carrier Budget** (noise/quantum limits): Is the signal-to-noise ratio sufficient?
   - **Operator Mismatch** (system fidelity): Does the assumed model match the true physics?

   This provides a physically attributable *why* behind every success or failure, not just a metric. Triad outputs are mandatory artifacts for every benchmark submission and production run.

3. **Infer and Attribute the Effective Operator** -- Estimate $\hat{H}$ from measurements alone and rank the parameter subspace responsible for degradation. Every inference must output confidence intervals, identifiability flags, and calibration uncertainty -- not just point estimates.

4. **Quantify Recoverability Limits** -- Determine, for each system configuration, what information is theoretically recoverable and what is irreversibly lost, expressed as bounds on achievable reconstruction quality.

5. **Compute Minimal Feasible Interventions** -- Find the smallest parameter corrections that recover the largest quality gains, following a Pareto-efficient calibration strategy with explicit cost-benefit trade-off curves.

6. **Produce Reproducible, Audit-Grade Reconstructions** -- Every imaging run produces a complete RunBundle capturing the full physical state, Triad diagnosis, correction trajectory, and uncertainty estimates, enabling cross-instrument verification and regulatory audit.

7. **Operate Within Declared Compute Budgets** -- All inference, diagnosis, and correction must complete within declared resource bounds (wall-clock time, GPU-hours, memory). Compute-bounded autonomy prevents brute-force solutions and ensures practical deployability.

When these conditions are met, imaging becomes a **deterministic engineering process** rather than expert craftsmanship.

---

## The Triad Law: A Unified Diagnostic Framework

Every imaging failure has exactly three possible root causes. PWM diagnoses all three simultaneously and outputs a **TriadReport** as a mandatory artifact:

```
                    +-----------------------+
                    |   Imaging Failure     |
                    |   (PSNR < target)     |
                    +-----------+-----------+
                                |
              +-----------------+-----------------+
              |                 |                 |
    +---------v-------+ +------v--------+ +------v--------+
    | Recoverability  | | Carrier Budget| |   Operator    |
    |   (Sampling)    | |   (Noise)     | |   Mismatch    |
    +-----------------+ +---------------+ +---------------+
    | Null space of H | | SNR, photon   | | H_true != H   |
    | Field of view   | | budget, dose  | | Calibration   |
    | Resolution limit| | Quantum limit | | drift, wear   |
    +-----------------+ +---------------+ +---------------+
    | Gate: Is info   | | Gate: Is info | | Gate: Is the  |
    |   encoded?      | |   detectable? | |   model right?|
    +-----------------+ +---------------+ +---------------+
              |                 |                 |
              +-----------------+-----------------+
                                |
                    +-----------v-----------+
                    |     TriadReport       |
                    |  (mandatory artifact) |
                    +-----------------------+
                    | - Dominant gate ID    |
                    | - Evidence scores     |
                    | - Confidence interval |
                    | - Recommended action  |
                    +-----------------------+
```

**Key insight**: Most existing work optimizes solvers (the reconstruction algorithm) while ignoring Gate 3 (operator mismatch). PWM's primary contribution is proving that Gate 3 is often the dominant bottleneck: a 1-pixel mask shift degrades MST-L by 14.5 dB, while the best solver upgrade improves it by only 1-2 dB.

---

## The Industrial Intelligence Stack for Imaging

PWM is designed as a complete Industrial Intelligence Stack -- not just a model, but the full infrastructure required to industrialize imaging. Each layer must be built, and the targeting system (Layer 4) only functions when the layers beneath it are solid.

### Layer 1: Purpose and Payoff

**Quantified, falsifiable target -- not a vague aspiration.**

| Metric | Definition | Target |
|--------|-----------|--------|
| Recovery ratio $\rho$ | $(\text{PSNR}_{III} - \text{PSNR}_{II}) / (\text{PSNR}_{I} - \text{PSNR}_{II})$ | $\geq 0.80$ across 20+ modalities |
| Oracle gap | $\text{PSNR}_{I} - \text{PSNR}_{III}$ | $\leq 2$ dB under bounded compute |
| Return on Imaging Compute (RoIC) | dB recovered per GPU-hour of calibration | Tracked per modality, must improve monotonically |

The purpose is mathematical and verifiable before any work begins. If you cannot state success as a number, you do not have a purpose.

### Layer 2: Task Taxonomy

**The map that breaks "fix imaging" into atomic, measurable actions.**

Every ISA task decomposes into a sequence of OperatorGraph operations:

| Task Class | Atomic Actions | Example |
|-----------|---------------|---------|
| **Compile** | Parse modality spec, instantiate OperatorGraph, validate topology | CASSI: Source -> Mask -> Dispersion -> Sensor -> Noise |
| **Diagnose** | Evaluate each Triad gate, rank parameter sensitivities, attribute degradation | Gate 3 binding: dx sensitivity = 4.2 dB/pixel |
| **Correct** | Estimate mismatch parameters, apply minimal intervention, verify improvement | dx: 1.5 -> 0.03 px (RMSE), +5.06 dB gain |
| **Verify** | Re-project through corrected operator, check consistency invariants, issue RunBundle | $\|y - \hat{H}\hat{x}\| < \epsilon$ |

This is the assembly-line instruction manual. Each action is testable in isolation.

### Layer 3: Observability

**You cannot fix what you cannot see.**

PWM's nervous system consists of:

- **RunBundle**: Permanent record of every imaging run -- inputs, operator state, Triad diagnosis, correction trajectory, outputs, uncertainty, compute consumed
- **Decision Record for Imaging Systems (DR-IS)**: Cryptographically signed log of every calibration decision (which parameters were adjusted, by how much, based on what evidence)
- **Drift Monitor**: Continuous tracking of operator fidelity metrics across runs, detecting degradation before it becomes critical
- **Rate Dashboards**: Everything expressed as rates -- dB per GPU-hour, recovery ratio per modality, parameter RMSE per calibration iteration

If a metric is not logged, it does not exist. Every TriadReport, every parameter estimate, every uncertainty bound is persisted and auditable.

### Layer 4: The Targeting System (The Harness)

**The engine that makes truth cheap to verify and channels progress toward measurable outcomes.**

This is the most critical layer. The targeting system is not a competition -- it is the **quality control infrastructure** that continuously stress-tests every claim PWM makes. It must exist before the agent ships.

> **Principle: Publish the harness before shipping the agent.** Build the counterfactual pack (adversarial test cases) first; the agent comes second. This proves domain understanding and makes cheating mechanically impossible.

#### Design Principles

1. **Blinded, rolling submissions on secret data.** Models never see test scenarios before evaluation. No amount of overfitting can help. New secret scenarios are added every quarter; old ones are retired.

2. **Make truth cheap to verify.** Evaluation is fully automated -- submit a RunBundle, get a score. No committee, no subjective review. The harness is the judge.

3. **Economic incentives tied to measurable performance.** Compute budgets are allocated based on demonstrated progress. Resources flow toward methods that move the needle on declared targets, not toward methods that publish well.

4. **Optimize for tail risk, not average case.** A system that scores 95% on easy scenarios but catastrophically fails on hard ones is worse than one that scores 85% uniformly. The harness weights worst-case performance heavily.

5. **Independent, automated safety brakes.** If any reliability metric regresses below a declared threshold (e.g., recovery ratio drops below 0.3 on any modality), the system is automatically flagged. Pre-committed boundaries, not post-hoc judgment.

#### The Counterfactual Pack

The core artifact of the targeting system. A **counterfactual pack** is a curated set of adversarial imaging scenarios designed to expose specific failure modes:

| Pack Type | What It Tests | Example |
|-----------|--------------|---------|
| **Mismatch Escalation** | Graceful degradation under increasing operator drift | dx sweep: 0.1 -> 0.5 -> 1.0 -> 2.0 -> 5.0 pixels |
| **Cross-Modality Transfer** | Whether calibration learned on CASSI generalizes to SPC, CT, MRI | Train on spectral mismatch, test on gain drift |
| **Out-of-Family** | Detection when the nominal operator family is wrong | Declared model: affine shift. True model: nonlinear warping |
| **Carrier Switch** | Transfer across physical carriers | Photon-trained system tested on electron/spin/acoustic data |
| **Tail-Risk Stress** | Performance under worst-case combined mismatch | All 5+ parameters perturbed simultaneously at extreme values |
| **No-Ground-Truth** | Correction quality without reference images | Evaluate via re-projection error, held-out channels, physical invariants |
| **Compute-Bounded** | Quality under strict resource limits | Same scenario at 1 GPU-min, 10 GPU-min, 1 GPU-hour budgets |

New packs are released quarterly. Old packs become public training data. The frontier always moves.

#### Required Submission Artifacts

Every harness submission must produce three outputs:

| Artifact | Description |
|----------|-------------|
| **Reconstruction** $\hat{x}$ | The corrected reconstruction |
| **Operator Estimate** $\hat{\theta} \pm \sigma_\theta$ | Mismatch parameters with calibrated uncertainty and identifiability flags |
| **TriadReport** | Dominant gate attribution, evidence scores, confidence, recommended action |

Submitting only a reconstruction (without diagnosis) is not accepted. The harness tests **understanding**, not just output quality.

#### Evaluation Tracks

**Track A: Design** -- Given requirements, specify an optimal imaging system.

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Constraint satisfaction | 0.30 | Does the design meet all hard requirements? |
| Pareto efficiency | 0.25 | Distance to Pareto frontier across objectives |
| Robustness margin | 0.25 | Tolerance to mismatch/drift/noise before failure |
| Calibration cost | 0.20 | Effort needed to reach target performance |

**Track B: Correct** -- Given a measurement with unknown mismatch, correct and reconstruct.

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Recovery ratio $\rho$ | 0.30 | Fraction of mismatch loss recovered |
| Parameter recovery | 0.20 | RMSE of estimated vs true mismatch |
| Uncertainty calibration | 0.15 | Do 90% CIs actually contain truth 90% of the time? |
| Tail-risk score | 0.15 | Performance on worst-case scenarios (bottom 10%) |
| Cross-modality transfer | 0.10 | Performance on modalities not seen during development |
| Compute efficiency | 0.10 | RoIC: dB recovered per GPU-hour |

**Track C: No-Ground-Truth** -- Correct without reference images.

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Self-consistency | 0.35 | Forward re-projection error $\|y - \hat{H}\hat{x}\|$ |
| Physical invariants | 0.25 | Energy conservation, spectral smoothness, symmetry |
| Held-out channel agreement | 0.20 | Prediction of withheld measurement channels |
| Cross-modality transfer | 0.10 | Generalization to unseen modalities |
| Compute efficiency | 0.10 | Quality per GPU-hour |

#### 4-Scenario Protocol (Track B)

| Scenario | Measurement | Reconstruction Operator | Purpose |
|----------|-------------|------------------------|---------|
| I (Ideal) | True $H$ | True $H$ | Oracle upper bound |
| II (Assumed) | True $H$ | Nominal $H_{\text{nom}}$ | Mismatch impact baseline |
| III (Corrected) | True $H$ | Calibrated $\hat{H}$ | Calibration benefit |
| IV (Oracle Mask) | True $H$ | True $H$ with nominal dispersion | Partial oracle bound |

#### Blinding and Rolling Submission

1. **Quarterly Release**: 10-20 new scenarios per track, including 2-3 out-of-family and 2-3 tail-risk stress tests
2. **Blind Phase** (8 weeks): Systems submit RunBundles against secret test data. No ground truth access. No leaderboard during this phase.
3. **Automated Assessment**: Harness evaluates all submissions mechanically. No human judgment in scoring.
4. **Publication**: Full results, all methodologies, all RunBundles made public. Retired scenarios become training data.
5. **Rolling Baseline**: PWM's own latest system is always a submission. If external methods beat it, that is signal, not threat.

#### Safety Brakes

Pre-committed thresholds that trigger automatic flags:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Recovery ratio regression | $\rho < 0.30$ on any validated modality | Block deployment, root-cause analysis required |
| Uncertainty miscalibration | Coverage deviates > 15% from declared CI | Flag all outputs as "uncalibrated" |
| Out-of-family miss | System confidently diagnoses wrong gate | Mandatory retraining on expanded family |
| Compute budget exceeded | > 2x declared GPU-hours | Submission disqualified for that scenario |
| Consistency violation | Re-projection error > 3x median | Output quarantined pending review |

### Layer 5: The Model Layer

PWM's agent: the OperatorGraph compiler, Triad diagnostics, calibration algorithms (grid search, gradient refinement), and reconstruction solvers (GAP-TV, MST, HDNet, etc.). The model is scaffolded by the management workflow:

- Multiple calibration strategies propose corrections
- Triad diagnosis validates each proposal
- Best correction is selected by measurable evidence, not by model confidence alone
- The model is the least durable part of the stack -- it will be replaced; the harness endures

### Layer 6: Actuation

The mechanisms by which PWM's decisions affect the physical world:

- **Software actuation**: Corrected operator $\hat{H}$ fed back into reconstruction pipeline
- **Hardware actuation** (future): Calibration commands sent to instrument controllers (stage motors, source tuning, detector gain adjustment)
- **Reporting actuation**: RunBundle published to audit trail, TriadReport delivered to operator
- All actions are logged in DR-IS, reversible where possible, and bounded by safety brakes

### Layer 7: Verification and Red Teaming

Continuous, independent stress-testing -- the immune system:

- **Adversarial Red Team**: Dedicated effort to break each new capability before deployment. Paid to find failures, not confirm success.
- **Decision Records (DR-IS)**: Every calibration decision cryptographically signed and permanently logged. Full chain of evidence from measurement to corrected output.
- **Regression Suite**: 2900+ automated tests. Any code change must pass the full suite. No exceptions.
- **Cross-Validation**: Results on one dataset cannot be trusted until replicated on a held-out dataset from a different instrument or lab.

### Layer 8: Governance and Incentives

Aligning effort with outcomes:

- **Outcome-based evaluation**: Methods are judged by recovery ratio and oracle gap, not by publication count or novelty claims
- **Compute escrow**: GPU budgets allocated to calibration tasks are tracked. Efficiency (RoIC) is a first-class metric. Brute-force approaches that burn compute without proportional quality gain are penalized.
- **Prestige shift**: The hero is not the person who solves one modality -- it is the person who builds a harness that makes all modalities testable. Infrastructure builders outrank individual solver authors.
- **Open harness, competitive agents**: The targeting system (counterfactual packs, evaluation code, RunBundle format) is fully open. Agents (specific calibration methods) compete on the open harness.

### Layer 9: Distribution and Maintenance

Reliable operation as a utility:

- **Standardized interfaces**: OperatorGraph IR is the universal protocol. Any modality that can be expressed as an OperatorGraph can be calibrated by any PWM-compatible agent.
- **Multi-solver redundancy**: No single-point-of-failure dependency on one reconstruction method. GAP-TV, MST, HDNet, PnP all available as interchangeable solvers.
- **Continuous monitoring**: Drift detection on deployed systems. Automatic re-calibration triggers when operator fidelity degrades beyond threshold.
- **Rate-based operations**: Everything measured as throughput -- modalities calibrated per week, dB recovered per GPU-hour, scenarios cleared per quarter.

---

## Maturation Levels: L0 to L5

Following the Industrial Intelligence Stack maturation curve, PWM progresses through six levels. Each level has a clear definition, and promotion requires passing the harness at the corresponding difficulty.

### L0: The Muddle (Pre-PWM)

No agreement on what "good imaging" means. Each lab uses different metrics, different test images, different noise models. Results are not comparable across papers. Calibration is manual, unreproducible, and undocumented.

- **AI role**: Non-existent
- **Characteristic**: "We got 32 dB on our test set" (incomparable to any other result)

### L1: Measurable (Current -- Partial)

Clear metrics exist (PSNR, SSIM, SAM, recovery ratio). Leaderboards show performance per modality. The 4-scenario protocol provides a common evaluation framework. Results are comparable.

- **AI role**: Referee and scorekeeper (automated evaluation)
- **Characteristic**: "MST-L achieves 35.29 dB Scenario I, 20.82 dB Scenario II, recovery ratio 0.26" (comparable, reproducible)
- **PWM status**: Achieved for CASSI (5-param), SPC (1-param). Partial for CACTI.

### L2: Repeatable

Best practices documented as standard operating procedures. Calibration workflows are codified: "For CASSI with suspected mask drift, run Algorithm 1 (grid search) then Algorithm 2 (gradient refinement) with these default parameters." Any trained engineer can follow the procedure and get comparable results.

- **AI role**: Template assistance and auto-completion
- **Characteristic**: "Follow the CASSI calibration SOP; expected gain +5 dB in 4 minutes"
- **PWM status**: Achieved for CASSI spatial mismatch. Not yet for dispersion or other modalities.

### L3: Automated (Target -- 12-18 months)

**The critical inflection point.** Checklists become code. PWM executes 80% of calibration work autonomously. Humans handle exceptions and out-of-family cases. The system self-diagnoses via Triad Law and self-corrects via minimal interventions.

- **AI role**: Primary worker, human handles exceptions
- **Characteristic**: "Submit measurement, receive corrected reconstruction + TriadReport + RunBundle in declared compute budget. Recovery ratio > 0.80."
- **Requirements**: 20+ modalities, automated Triad diagnosis, compute-bounded operation, uncertainty-calibrated outputs

### L4: Industrialized (Target -- 24-36 months)

The market stops hiring humans for routine calibration. Labs buy ISA outcomes: "calibrate my CASSI system to within 2 dB of oracle" as a service. PWM-compatible agents from multiple providers are interchangeable.

- **AI role**: Primary worker, humans design new operator families only
- **Characteristic**: Calibration is purchased as a service, not performed as a research project
- **Requirements**: Multiple competing agents on the open harness, cross-modality transfer, out-of-family detection

### L5: Commoditized / Solved (Target -- 36+ months)

Multiple providers deliver identical calibration quality at competitive prices. Imaging calibration is as ordinary as auto-exposure in a camera. The problem is compute-bound: more quality requires only more GPU-hours, not more expertise.

- **AI role**: Utility (like electricity)
- **Characteristic**: "Any imaging system self-calibrates on first power-up"
- **Requirements**: 100+ modalities, zero-shot generalization, real-time adaptive calibration
- **Primary metric**: Return on Imaging Compute (RoIC) -- dB per dollar of compute

---

## Current State Assessment

### Where PWM Sits on the Stack

| Layer | Status | Evidence |
|-------|--------|---------|
| 1. Purpose | Defined | Recovery ratio, oracle gap, RoIC targets declared |
| 2. Task Taxonomy | Built | OperatorGraph IR, 64 modalities, 89 templates, atomic task decomposition |
| 3. Observability | Partial | RunBundle exists, DR-IS not yet implemented, drift monitor planned |
| 4. Targeting System | Early | 4-scenario protocol operational, counterfactual packs not yet curated, no blinded external submissions |
| 5. Model Layer | Active | Alg 1 + Alg 2 calibration, 5 reconstruction solvers, Triad diagnosis prototype |
| 6. Actuation | Software only | Corrected operator feeds reconstruction; no hardware actuation yet |
| 7. Verification | Strong | 2900+ tests, regression suite, cross-validation on 10 KAIST scenes |
| 8. Governance | Not started | No compute escrow, no outcome-based allocation |
| 9. Distribution | Foundation | Standardized OperatorGraph IR, multi-solver redundancy |

### Current Maturation: L1 (Measurable) transitioning to L2 (Repeatable)

**L1 evidence**: Clear metrics, comparable results, 4-scenario protocol validated on 3 modalities.

**L2 gaps**: Calibration SOPs not yet documented for all modalities. Procedures still require expert parameter tuning for new modalities.

### Track-Level Progress

| Track | Status | Validated Modalities |
|-------|--------|---------------------|
| Track A: Design | Foundation built (64 modalities, 89 templates) | Not yet evaluated |
| Track B: Correct | Active development | CASSI (5-param, $\rho$=28-51%), SPC (1-param, $\rho$=68-72%), CACTI (partial) |
| Track C: No-GT | Not started | Foundations exist (re-projection error, invariants) |

### Key Empirical Findings

1. **Gate 3 is binding.** A 1-pixel mask shift degrades MST-L by 14.5 dB; switching from GAP-TV to MST-L improves only 10 dB under ideal conditions. Operator fidelity is the dominant bottleneck.

2. **Neural solvers amplify mismatch sensitivity.** MST-L drops 14.5 dB under CASSI mismatch; GAP-TV drops only 1.9 dB. Learned solvers overfit to the training-time forward model.

3. **Calibration universally helps.** Every validated modality shows positive gain: Scenario III > Scenario II, always.

4. **Recovery ratio is solver-dependent.** The optimal strategy pairs strong solvers with strong calibration.

5. **Dispersion mismatch is a new frontier.** Sub-pixel spectral dispersion drift causes 6+ dB additional degradation beyond spatial mismatch alone.

---

## Quantified Targets

| Metric | Current | L3 Target | L5 Target | Timeline |
|--------|---------|-----------|-----------|----------|
| Modalities covered | 64 | 100+ | 200+ | L3: 18mo, L5: 36mo |
| Mismatch params per modality | 3-5 | 10+ | Any | L3: 18mo |
| Recovery ratio $\rho$ | 30-50% | 80%+ | 95%+ | L3: 18mo, L5: 36mo |
| Oracle gap | 5-12 dB | $\leq$ 2 dB | $\leq$ 0.5 dB | L3: 18mo, L5: 36mo |
| Validated calibration modalities | 3 | 20+ | 100+ | L3: 18mo, L5: 36mo |
| Zero-shot generalization | 0% | 50%+ | 90%+ | L3: 24mo, L5: 36mo |
| Out-of-family detection | 0% | 90%+ | 99%+ | L3: 24mo |
| Uncertainty calibration | Not measured | 90% coverage at 90% CI | 95% at 95% CI | L3: 18mo |
| Counterfactual packs published | 0 | 10+ | 50+ | L3: 18mo, L5: 36mo |
| RoIC (dB per GPU-hour) | Not tracked | Tracked, improving | Commoditized | L3: 12mo |

---

## Roadmap

### Near-Term: L1 -> L2 (0-6 months)

**Goal: Make calibration repeatable -- anyone can follow the SOP and get comparable results.**

1. Complete CACTI 3-scenario validation (10 scenes, 5 methods)
2. Document calibration SOPs for CASSI, SPC, CACTI
3. Publish first 3 counterfactual packs (one per validated modality)
4. Add 5+ modalities to Track B: CT, MRI, OCT, ptychography, light-field
5. Implement DR-IS (Decision Records for Imaging Systems)
6. Add uncertainty quantification to all calibration outputs
7. Begin tracking RoIC per modality
8. Implement Track A prototype: requirements $\to$ modality selection $\to$ parameter optimization

### Medium-Term: L2 -> L3 (6-18 months)

**Goal: Automate calibration -- PWM handles 80% of cases, humans handle exceptions.**

1. Expand to 20+ calibrated modalities with validated recovery ratios
2. Automated Triad diagnosis operational across all modalities
3. Cross-modality transfer: calibration trained on one modality applied to another
4. Launch first external harness submissions (open counterfactual packs + blinded evaluation)
5. Achieve $\rho \geq 0.80$ on 10+ modalities
6. Implement Track C (no-ground-truth) evaluation
7. Out-of-family detection on 5+ modalities
8. Compute escrow: GPU budgets tied to demonstrated RoIC improvements
9. Publish the harness specification paper

### Long-Term: L3 -> L4 -> L5 (18-36 months)

**Goal: Industrialize, then commoditize -- calibration becomes a utility.**

1. Autonomous mismatch detection without explicit mismatch model
2. Real-time adaptive calibration during acquisition
3. Full Track A: natural language $\to$ validated pipeline in minutes
4. 100+ modalities with validated calibration
5. Hardware API integration for closed-loop actuation
6. Multiple competing agents on the open harness
7. Imaging calibration purchased as a service, not performed as research
8. Stage 1 complete: **Imaging System Autonomy achieved at L5**
