# LIP Arena: PWM's Built-in Evaluation Harness

**LIP = Living Imaging Physics**

**The Targeting System for Imaging System Autonomy (ISA)**

> A targeting system must test on future events that did not exist at training time, so nobody can memorize the answers. It must be funded and structured to try to break the system, constantly injecting hard cases and distribution shifts.
>
> -- Design spine derived from the [SolveEverything](https://solveeverything.org/) blueprint (Wissner-Gross & Diamandis) and the Industrial Intelligence Stack

**Design Provenance.** LIP Arena operationalizes three core SolveEverything principles for the imaging domain:

| SolveEverything Principle | LIP Arena Implementation |
|--------------------------|--------------------------|
| **Targeting Authorities** -- publicly funded leaderboards that use "blinded clears" (unseen test questions) to rank systems | Commit-Measure-Score protocol: prospective measurements generated *after* submission deadline; leaderboard ranked by prospective-dominated score ($0.7 \times S_{\text{prospective}}$) |
| **Decision Records for AI Systems (DR-AIS)** -- permanent, auditable logs of every AI decision | Decision Records for Imaging Systems (DR-IS): every calibration action logged with evidence, gate attribution, confidence, compute consumed, and SHA-256 hash (Section 7.3) |
| **Outcome-Based Procurement** -- pay for verified results, not effort or promises | Anti-Goodhart scoring: gaming penalized, recovery ratio $\rho$ and RoIC (dB/GPU-hr) are the currency; submissions must be right *for the right reasons* (Section 6) |

LIP Arena is the first domain-specific instantiation of the SolveEverything targeting authority pattern.

---

## 1. What LIP Arena Is

LIP (Living Imaging Physics) Arena is PWM's built-in evaluation harness (Layer 4 of the Industrial Intelligence Stack). It is **not** a separate system and **not** a static benchmark. It ships with PWM, runs locally via `pwm evaluate`, and continuously stress-tests every claim Imaging System Autonomy makes using a live, prospective, adversarial protocol.

```bash
# Score any method against the built-in harness
pwm evaluate --method my_solver --modality cassi --track correct
pwm evaluate --method gap_tv --modality spc --track no-gt
```

**"Living"** encodes the core SolveEverything principle: measurements are created *after* the submission deadline. The physics is alive -- drifting, shifting, surprising. No memorization. No overfitting. No gaming.

LIP Arena guarantees two properties:

| Property | Mechanism |
|----------|-----------|
| **Prospective + blinded** | Commit-Measure-Score protocol: submissions are frozen before measurements exist |
| **Adversarial + anti-gaming** | Dedicated Red Team module whose only job is to break submissions every round |

Everything else -- tracks, scoring, governance -- exists to enforce these two properties.

---

## 2. Two Parallel Evaluation Modes

LIP Arena operates in two complementary modes. Both enforce the same core guarantee -- submissions are sealed before test data exists -- but they differ in scope, turnaround, and authority. A submission to either mode produces the same artifacts (RunBundle, TriadReport, operator estimate) and uses the same 4-Scenario Protocol.

```
  ┌───────────────────────────────────────────────────────────────────────────┐
  │                         LIP Arena: Two Modes                             │
  ├─────────────────────────────────┬─────────────────────────────────────────┤
  │       INSTANT MODE              │         FULL ROUND MODE                 │
  │       (Available 24/7)          │         (Quarterly)                     │
  ├─────────────────────────────────┼─────────────────────────────────────────┤
  │                                 │                                         │
  │  SUBMIT ──► GENERATE ──► RUN   │  COMMIT ──► MEASURE ──► EXECUTE ──► SCORE│
  │  (~sec)     (~1 min)   (~5-30m)│  (2 wk)    (2 wk)     (1 wk)    (1 wk) │
  │                                 │                                         │
  │  Sealed simulator only          │  Sealed simulator + live-lab data       │
  │  Standard tolerance envelopes   │  Red Team adversarial injection         │
  │  Results on LIP in minutes      │  Authoritative ranking                  │
  │  Rolling, always-on             │  Quarterly schedule with audits         │
  │                                 │                                         │
  │  PWM re-evaluated on every      │  All RunBundles published               │
  │  submission's test set          │  Failure taxonomy updated               │
  │  (no stale baselines)           │  Retired scenarios become training data │
  └─────────────────────────────────┴─────────────────────────────────────────┘
```

| Property | Instant Mode | Full Round Mode |
|----------|-------------|-----------------|
| Availability | 24/7, on-demand | Quarterly schedule |
| Data source | Sealed simulator only | Sealed simulator + live-lab |
| Red Team injection | Standard tolerance envelopes | Adversarial escalation schedule |
| Turnaround | Minutes to ~1 hour | 6 weeks (full protocol) |
| Leaderboard weight | Separate instant leaderboard | Authoritative ranking ($0.3 S_{\text{retro}} + 0.7 S_{\text{prospective}}$) |
| Use case | Development iteration, CI/CD, rolling baseline | Publication-grade evaluation, official ranking |

### Instant Mode

Instant Mode is the always-available, real-time evaluation path. After submission, the container is locked, fresh prospective data is generated, the 4-Scenario Protocol runs, and results appear on the LIP leaderboard -- all within minutes.

**How it works:**

1. **Submit.** Team uploads a container image. The container is SHA-256 sealed on receipt. No further modifications.
2. **Generate.** The sealed simulator immediately draws fresh random seeds, mismatch parameters (from declared tolerance envelopes), scene content (from held-out databases), and noise realizations. All generated *after* the seal -- memorization is impossible.
3. **Execute.** The container runs in a sandboxed environment with no network access. The full 4-Scenario Protocol executes across all declared modalities. Compute budget enforced.
4. **Publish.** Scores, RunBundles, and TriadReports appear on the LIP leaderboard. The result is locked and timestamped. PWM's own rolling baseline is always present for comparison.

```bash
# Submit to Instant Mode (single modality)
pwm submit --mode instant --container ./my_solver.sif --modality cassi

# Submit to Instant Mode (all declared modalities)
pwm submit --mode instant --container ./my_solver.sif --modality all

# Check status (results appear when ready)
pwm status --submission-id <sha256-hash>

# View LIP leaderboard
pwm leaderboard --mode instant --modality cassi
pwm leaderboard --mode full    --round 2026-Q3
```

**Expected turnaround times** (based on flagship paper validated benchmarks, single A100 GPU):

| Modality | Scenarios | Est. Wall-Clock | Bottleneck |
|----------|-----------|----------------|------------|
| CASSI | I-IV, 10 scenes | ~8 min | Grid search over 5 mismatch params |
| CACTI | I-IV, 6 videos | ~5 min | Temporal mask reconstruction |
| SPC | I-IV, 11 images | ~3 min | 25% sampling, FISTA-TV + PnP-DRUNet |
| CT | I-IV, per phantom | ~6 min | Sinogram backprojection |
| Ptychography | I-IV, per scan | ~10 min | Phase retrieval iterations |
| MRI | I-IV, per slice | ~4 min | Multi-coil SENSE + correction |
| Lensless | I-IV, per scene | ~5 min | PSF deconvolution |
| **All 7 validated** | **Full suite** | **~45 min** | **Parallelizable across GPUs** |

### Full Round Mode

Full Round Mode is the quarterly, authoritative evaluation. It adds live-lab data from partner labs and Red Team adversarial injection on top of the sealed-simulator sets.

**Phase 1: Commit (2 weeks)**

Method submissions include:
- **Container image** with the full pipeline (inference, calibration, reconstruction, diagnosis)
- **Declared compute budget** (GPU-hours, peak memory, wall-clock limit per scenario)
- **Operator family declarations** (which OperatorGraph families the method claims to handle)

After the deadline, submissions are cryptographically sealed. No modifications permitted.

**Phase 2: Measure (2 weeks)**

The PWM harness generates new measurement sets from two sources:

**(A) Live-Lab Prospective Sets**

Partner labs capture new physical measurements after the commit deadline:
- Bead scans, resolution targets, tissue phantoms, spectral calibration scenes
- Measurement recipes and hardware IDs hidden until after scoring
- Labs rotate each round to prevent lab-specific overfitting
- Physical hardware drift is real (not simulated) -- the hardest test

**(B) Sealed-Simulator Prospective Sets**

The same public OperatorGraph families, but with new random seeds and parameter draws generated post-deadline:
- Mismatch parameters drawn from declared tolerance envelopes (with adversarial tails -- see Red Team)
- Scene content drawn from held-out image databases
- Noise realizations freshly sampled
- The sealed simulator ships with PWM but runs sandboxed; only the measurement outputs are used for scoring

Both sources are used every round. A submission must perform well on both to rank.

**Phase 3: Execute (1 week)**

All submissions run in a sealed compute environment:
- No network access
- No access to ground truth, measurement recipes, or hardware IDs
- Compute budgets strictly enforced (exceeding 2x budget = disqualification for that scenario)
- All outputs captured: reconstruction, operator estimate, TriadReport, RunBundle

**Phase 4: Score (1 week)**

Fully automated -- no committee, no subjective review:
- Scores computed against held-out ground truth (Track 1, 2) or consistency metrics (Track 3)
- All RunBundles published
- All methodologies published
- Retired scenarios become public training data
- PWM's own latest system is always a submission (rolling baseline)

---

## 3. Evaluation Tracks

Four tracks, each prospective and adversarial by default. Both Instant Mode and Full Round Mode support all four tracks.

### Track 1: Correct (Live Drift Correction)

**Goal**: Given measurement $y$ and nominal model $H_{\text{nom}}$, infer $\hat{H}$, correct mismatch, reconstruct $\hat{x}$.

**Prospective upgrade**: Measurement sets are generated after submission deadline. Both live-lab measurements (with real hardware drift) and sealed-simulator measurements (with post-deadline parameter draws).

**Required outputs** (all mandatory, enforced by harness):

| Artifact | Format |
|----------|--------|
| Reconstruction $\hat{x}$ | NumPy array, declared dtype and shape |
| Operator estimate $\hat{\theta} \pm \sigma_\theta$ | Parameter vector + covariance + identifiability flags per parameter |
| TriadReport | Dominant gate ID, evidence scores per gate, confidence interval, recommended action |
| RunBundle hash | SHA-256 of the complete RunBundle (inputs, operator state, correction trajectory, outputs, compute consumed) |

**4-Scenario Protocol**:

| Scenario | Measurement | Reconstruction Operator | Purpose |
|----------|-------------|------------------------|---------|
| I (Ideal) | True $H$ | True $H$ | Oracle upper bound |
| II (Assumed) | True $H$ | Nominal $H_{\text{nom}}$ | Mismatch impact baseline |
| III (Corrected) | True $H$ | Calibrated $\hat{H}$ | Calibration benefit |
| IV (Oracle Mask) | True $H$ | True $H$ with nominal dispersion | Partial oracle bound |

**Scoring**:

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Recovery ratio $\rho$ | 0.30 | $(\text{PSNR}_{III} - \text{PSNR}_{II}) / (\text{PSNR}_{I} - \text{PSNR}_{II})$ |
| Parameter recovery | 0.20 | RMSE of $\hat{\theta}$ vs $\theta_{\text{true}}$ |
| Uncertainty calibration | 0.15 | Do declared 90% CIs contain truth 90% of the time? |
| Tail-risk score | 0.15 | Performance on bottom-10% hardest scenarios |
| Cross-modality transfer | 0.10 | Performance on modalities not declared in commit |
| Compute efficiency (RoIC) | 0.10 | dB recovered per GPU-hour |

### Track 2: Diagnose (Triad Attribution Under Shift)

**Goal**: Identify whether failure is caused by sampling (Gate 1), noise (Gate 2), or operator mismatch (Gate 3).

**Prospective upgrade**: Inject hard cases where the dominant gate *flips* between rounds. Example: Round N has Gate 3 dominant; Round N+1 the same modality has Gate 2 dominant due to a different noise regime. Systems that memorize "CASSI = Gate 3" fail.

**Required outputs**:

| Artifact | Format |
|----------|--------|
| TriadReport | Gate attribution with evidence scores and confidence |
| Gate ranking | Ordered list of gates by impact magnitude (dB attributable to each) |
| Recommended action | What intervention would address the dominant gate |

**Scoring**:

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Gate attribution accuracy | 0.35 | Correct identification of dominant gate |
| Evidence quality | 0.25 | Are the evidence scores physically meaningful and consistent? |
| Action relevance | 0.20 | Would the recommended action actually address the dominant gate? |
| Shift robustness | 0.20 | Accuracy when the dominant gate flips vs training distribution |

### Track 3: No-GT (Consistency + Invariants)

**Goal**: Correct without ground truth -- the realistic deployment scenario.

**Prospective upgrade**: Include adversarial cases where re-projection error is misleading unless the operator is correct. Example: a wrong operator that happens to produce low re-projection error on the measurement (the "fitting $y$ without understanding $H$" trap). Systems that rely only on $\|y - \hat{H}\hat{x}\|$ without physical invariant checks will fail these cases.

**Required outputs**:

| Artifact | Format |
|----------|--------|
| Reconstruction $\hat{x}$ | NumPy array |
| Operator estimate $\hat{\theta} \pm \sigma_\theta$ | With identifiability flags |
| TriadReport | Gate attribution based on consistency evidence |
| Self-consistency score | Forward re-projection error + declared confidence |

**Scoring**:

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Self-consistency | 0.30 | Re-projection error $\|y - \hat{H}\hat{x}\|$ relative to noise floor |
| Physical invariants | 0.25 | Energy conservation, spectral smoothness, symmetry, positivity |
| Held-out channel agreement | 0.20 | Prediction accuracy on withheld measurement channels |
| Adversarial trap survival | 0.15 | Correct rejection of misleading low-error solutions |
| Compute efficiency | 0.10 | Quality per GPU-hour |

### Track 4: Design (Requirements -> Robust OperatorGraph)

**Goal**: Propose system designs that are inherently robust and cheaply calibratable.

**Prospective upgrade**: The Red Team introduces unexpected drift and noise at evaluation time that was not in the requirements spec. Designs are scored on how gracefully they degrade and how cheaply they can be re-calibrated under surprise conditions.

**Required outputs**:

| Artifact | Format |
|----------|--------|
| OperatorGraph specification | Complete graph with all parameters declared |
| Performance prediction | Expected PSNR, SSIM under nominal + tolerance envelope |
| Robustness margin | Maximum mismatch before performance drops below threshold |
| Calibration cost estimate | GPU-hours required to correct from tolerance-edge mismatch |

**Scoring**:

| Criterion | Weight | Description |
|-----------|-------:|-------------|
| Constraint satisfaction | 0.25 | Does the design meet all hard requirements? |
| Pareto efficiency | 0.20 | Distance to Pareto frontier across objectives |
| Robustness margin | 0.25 | Tolerance to surprise drift/noise at evaluation time |
| Calibration cost | 0.20 | Effort to restore performance after Red Team perturbation |
| Prediction accuracy | 0.10 | How close was the predicted performance to actual? |

---

## 4. Modality Coverage

PWM registers **64 imaging modalities** across 13 categories. The complete registry with IDs, forward model types, default solvers, and validation status is maintained in a dedicated document:

> **[imaging_modalities.md](imaging_modalities.md)** -- Full 64-modality registry

LIP Arena can evaluate any registered modality via `pwm evaluate --modality <id>`. The framework is universal: all 64 modalities share the same OperatorGraph IR, the same 11 physical primitives, the same Triad decomposition, and the same 4-Scenario Protocol.

### Currently Validated (7 modalities)

These modalities have completed the full 4-Scenario Protocol with flagship paper results and form the LIP Arena rolling baseline:

| Modality | Category | Carrier | Example Mismatch Parameters |
|----------|----------|---------|---------------------------|
| **CASSI** | Compressive | Optical photon | Mask shift $\Delta x$, $\Delta y$; rotation $\theta$; dispersion slope $a_1$; axis offset $\alpha$ (5 params) |
| **CACTI** | Compressive | Optical photon | Spatial shifts; rotation; temporal clock; duty cycle; gain; offset; noise (8 params) |
| **SPC** | Compressive | Optical photon | Exponential gain drift $\alpha$; measurement noise $\sigma_y$ (2 params) |
| **Lensless** | Comp. Photography | Optical photon | PSF shift; scale drift; defocus offset (3+ params) |
| **CT** | Medical | X-ray photon | Center-of-rotation offset; angular offset; detector tilt; beam hardening (3+ params) |
| **Ptychography** | Coherent | Electron | Probe position error; defocus; aberration coefficients (3+ params) |
| **MRI** | Medical | Nuclear spin/RF | Coil sensitivity error; k-space trajectory; off-resonance phase (3+ params) |

### Expansion Roadmap

| Phase | Timeline | Target Modalities | Examples |
|-------|----------|-------------------|---------|
| A | 0-6 months | 7 validated | CASSI, CACTI, SPC, CT, Ptychography, MRI, Lensless |
| B | 6-12 months | 10+ | Add OCT, Photoacoustic, SIM, Ultrasound |
| C | 12-24 months | 20+ | Add PET, SPECT, Holography, NeRF, SEM, TEM, SAR |
| D | 24+ months | 64 (all registered) | Full registry, all categories, hardware-in-the-loop |

Any registered modality can be submitted to Instant Mode today via the sealed simulator -- validation status only affects whether flagship-paper baselines exist for comparison. A researcher working on SEM or Doppler Ultrasound can still submit, receive scores, and appear on the LIP leaderboard; the modality simply won't have an established rolling baseline until formal validation is complete.

---

## 5. The Red Team Module

A dedicated, funded adversarial layer whose **only job** is to break submissions every round. This implements SolveEverything's principle: "hire experts / other AIs to try to trick or game the system."

The Red Team operates independently from the evaluation track custodians. It has its own budget and mandate. Red Team injection applies to Full Round Mode; Instant Mode uses standard tolerance envelopes.

### Red Team Injection Categories

Every round, the Red Team must include at least one instance of each:

| Category | Description | Example |
|----------|-------------|---------|
| **Novel mismatch type** | A mismatch mechanism not present in any prior training pack | Chromatic aberration drift in a system previously tested only for spatial shift |
| **Compound mismatch** | Small errors across many parameters simultaneously (not one dominant) | dx=0.3, dy=0.2, $\theta$=0.1, $a_1$=0.01, $\alpha$=0.05, gain=0.98 -- individually mild, jointly degrading |
| **Out-of-family physics** | True physics includes effects not in the declared operator family | Nonlinear detector response in a system declared as linear; scattering in a ballistic model |
| **Distribution shift** | Scene content with textures, materials, or structures never seen in training | Industrial metallurgy samples for a system trained on biomedical tissue; underwater scenes for an aerial system |
| **Compute traps** | Scenarios that tempt brute-force grid search; budget enforcement penalizes it | High-dimensional parameter space where gradient methods succeed in budget but grid search exceeds 2x budget |
| **Gate-flip scenarios** | Cases where the dominant Triad gate is different from the historical prior for that modality | CASSI scenario where Gate 2 (noise) dominates instead of the usual Gate 3 (mismatch) |
| **Misleading consistency** | Cases where a wrong operator produces low re-projection error (Track 3 trap) | Degenerate operator that fits $y$ well but produces physically meaningless $\hat{x}$ |

### Red Team Escalation Schedule

Difficulty increases predictably across rounds:

| Round | Mismatch Severity | Novel Types | Compound Params | Out-of-Family Fraction |
|-------|------------------|-------------|-----------------|----------------------|
| 1-2 | Mild (within published tolerance) | 1 per round | 2-3 simultaneous | 5% of scenarios |
| 3-4 | Moderate (1-2x tolerance) | 2 per round | 3-5 simultaneous | 10% of scenarios |
| 5-6 | Severe (2-5x tolerance) | 2 per round | 5+ simultaneous | 15% of scenarios |
| 7+ | Catastrophic (>5x tolerance) | 3+ per round | All params | 20% of scenarios |

### Red Team Reporting

After each round, the Red Team publishes a **Failure Mode Taxonomy Update**:
- New failure modes discovered
- Which submissions were broken and how
- Recommended additions to operator family definitions
- Updated tolerance envelopes based on real-world evidence

This is how the benchmark evolves. The failure taxonomy is a public good.

---

## 6. Anti-Goodhart Scoring

Standard benchmarks die from Goodhart's Law: "when a measure becomes a target, it ceases to be a good measure." LIP Arena prevents this through two mechanisms.

### 6.1 Prospective Dominance

Every submission receives two scores:

| Score Type | Source | Purpose |
|-----------|--------|---------|
| **Retrospective score** | Static test set (known distributions, public after prior rounds) | Baseline capability check |
| **Prospective score** | Post-deadline measurements (live-lab + sealed-simulator) | True generalization test |

**Leaderboard ranking is dominated by prospective performance:**

$$S_{\text{rank}} = 0.3 \times S_{\text{retro}} + 0.7 \times S_{\text{prospective}}$$

A system that scores 95% retrospective but 60% prospective will rank below a system that scores 80% on both. This makes memorization and overfitting to public data a losing strategy.

### 6.2 Gaming Penalty

If a method improves reconstruction quality (PSNR) but fails integrity checks, it **loses ranking**. This prevents metric hacking.

A submission is penalized when any of the following checks fail:

| Check | Condition for Penalty | Penalty |
|-------|----------------------|---------|
| **Triad attribution sanity** | Dominant gate attribution contradicts physical evidence (e.g., claims Gate 1 when operator was clearly wrong) | $-0.15 \times S_{\text{track}}$ |
| **Uncertainty calibration** | Declared 90% CIs contain truth less than 75% of the time (severely overconfident) | $-0.10 \times S_{\text{track}}$ |
| **Identifiability consistency** | Parameters flagged as "identifiable" have RMSE > 3x the flagged uncertainty | $-0.10 \times S_{\text{track}}$ |
| **Compute honesty** | Declared budget < 0.5x actual consumption (sandbagging) | Disqualification |
| **Reconstruction-only submission** | Missing TriadReport or operator estimate | Not scored (incomplete) |

**Net effect**: You must be right **for the right reasons**. A high-PSNR reconstruction with wrong diagnosis, overconfident uncertainty, or missing artifacts scores worse than a moderate-PSNR reconstruction with correct, calibrated, complete outputs.

### 6.3 Composite Track Scores

After prospective dominance weighting and gaming penalties:

$$S_{\text{total}} = 0.35 \times S_{\text{correct}} + 0.20 \times S_{\text{diagnose}} + 0.25 \times S_{\text{no-gt}} + 0.20 \times S_{\text{design}}$$

Track 1 (Correct) weighted highest: the core ISA capability. Track 3 (No-GT) weighted second: the most practically relevant. Track 2 (Diagnose) and Track 4 (Design) weighted equally: understanding and prevention.

---

## 7. Governance and Reproducibility

LIP Arena is infrastructure, not a one-off competition. It must operate as a utility.

### 7.1 Submission Requirements

| Requirement | Specification |
|-------------|--------------|
| **Containerized** | Docker/Singularity image with all dependencies frozen |
| **Declared compute budget** | GPU-hours, peak memory, wall-clock limit per scenario |
| **Operator family declarations** | Which OperatorGraph families the system claims to support |
| **Immutable RunBundle** | SHA-256 hash of complete output bundle (inputs, state, trajectory, outputs) |
| **Reproducibility guarantee** | Same container + same input must produce bit-identical output |

### 7.2 Round Reports

After every Full Round, LIP Arena publishes:

| Report Component | Purpose |
|-----------------|---------|
| **Full scores** (all tracks, all submissions) | Transparency -- no hidden rankings |
| **All RunBundles** | Any researcher can inspect any submission's reasoning |
| **Failure mode taxonomy update** (from Red Team) | How the benchmark evolves; what new failure modes were discovered |
| **Retired scenario release** | Old prospective data becomes public training data |
| **Counterfactual pack update** | New adversarial scenarios added to public training pool |
| **RoIC leaderboard** | dB per GPU-hour per modality -- efficiency tracking |

Instant Mode publishes scores and RunBundles immediately upon completion. Cumulative Instant Mode statistics are included in quarterly round reports.

### 7.3 Decision Records for Imaging Systems (DR-IS)

Every calibration decision within a submission is logged:

```
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

DR-IS records are part of the RunBundle. They enable post-hoc audit of *how* a system reached its answer, not just *what* it answered.

### 7.4 Safety Brakes

Pre-committed thresholds that trigger automatic flags. These are mechanical, not discretionary:

| Condition | Threshold | Action |
|-----------|-----------|--------|
| Recovery ratio regression | $\rho < 0.30$ on any validated modality | Block deployment, root-cause analysis required |
| Uncertainty miscalibration | Coverage deviates > 15% from declared CI | Flag all outputs as "uncalibrated" |
| Out-of-family miss | System confidently diagnoses wrong gate on known OOF scenario | Mandatory retraining on expanded family |
| Compute budget exceeded | > 2x declared GPU-hours | Submission disqualified for that scenario |
| Consistency violation | Re-projection error > 3x noise-floor median | Output quarantined pending review |

---

## 8. How LIP Arena Maps to the Industrial Intelligence Stack

| Stack Layer | LIP Arena Component |
|-------------|-------------------|
| 1. Purpose & Payoff | Recovery ratio $\geq 0.80$, oracle gap $\leq 2$ dB, RoIC tracked |
| 2. Task Taxonomy | 4 tracks (Correct, Diagnose, No-GT, Design), each with atomic actions |
| 3. Observability | RunBundle, DR-IS, TriadReport, RoIC dashboards |
| **4. Targeting System** | **LIP Arena module within PWM: dual-mode (Instant + Full Round) Commit-Measure-Score protocol + Red Team; runs via `pwm evaluate` and `pwm submit`** |
| 5. Model Layer | PWM's shipped default methods (current best) + submitted methods; the harness outlasts any individual method |
| 6. Actuation | Corrected operators fed to reconstruction; future: hardware-in-the-loop |
| 7. Verification | Red Team, DR-IS audit trail, gaming penalties, safety brakes |
| 8. Governance | Outcome-based ranking, compute escrow via declared budgets, open harness |
| 9. Distribution | OperatorGraph IR as universal protocol, containerized submissions, public round reports |

---

## 9. Validated Baselines from the PWM Flagship Paper

LIP Arena is grounded in empirically validated results from the PWM flagship paper. These serve as the **rolling baseline** -- the scores that every new submission must beat. PWM's own system is always a submission; the numbers below are its current performance.

### 9.1 Correction Gains (PWM Rolling Baseline, Autonomous Grid-Search)

| Modality | Solver | Gain (dB) | 95% CI | Recovery $\rho$ | Cohen's $d$ | RoIC (dB/GPU-hr) |
|----------|--------|-----------|--------|-----------------|-------------|-------------------|
| CASSI | GAP-TV | +0.76 | [0.68, 0.83] | 85% | 2.8 | 0.9 |
| CACTI | GAP-TV | +10.21 | [9.5, 10.9] | 100% | 8.1 | 42 |
| SPC | FISTA-TV | +7.71 | [6.8, 8.7] | 86% | 3.5 | 15 |
| CT | FBP | +10.68 | [9.9, 11.4] | 100% | 9.3 | 120 |
| Ptychography | ePIE | +7.09 | [6.2, 8.0] | 100% | 5.7 | 85 |
| MRI (R=4) | SENSE | +1.75-7.14 | -- | 20% | 3.1 | 5790 |
| Lensless | ADMM | +3.55 | [3.1, 3.9] | 78% | 4.2 | 35 |

**Interpretation:** Gain = $\text{PSNR}_{III} - \text{PSNR}_{II}$ (corrected vs mismatched). Recovery ratio $\rho = (\text{PSNR}_{III} - \text{PSNR}_{II}) / (\text{PSNR}_{I} - \text{PSNR}_{II})$. All CIs are 95% bootstrap over $B = 1{,}000$ resamples.

### 9.2 Central Result: Gate 3 Dominance

The flagship paper's central empirical claim -- validated across all 7 modalities -- is:

> **For 9/9 configurations:** $C_{\text{mismatch}} > \max(C_{\text{noise}}, C_{\text{recover}})$
>
> Single-parameter operator correction recovers **more reconstruction quality** than the gap between a classical solver and a state-of-the-art deep network operating on the same mismatched operator.

This means the LIP Arena leaderboard will primarily differentiate submissions by their ability to **diagnose and correct operator mismatch** (Gate 3), not by the reconstruction algorithm alone. A classical solver with correct calibration beats a deep network with wrong calibration.

### 9.3 Hardware Validation Baselines

Real-data experiments from the flagship paper establish the simulation-to-hardware gap:

| Modality | Metric | Simulation Prediction | Hardware Measurement | Gap |
|----------|--------|----------------------|---------------------|-----|
| CASSI | Mismatch degradation | Large | Small (<0.22 dB) | Hardware mask has pre-existing uncorrected errors that absorb perturbation |
| CACTI | Residual ratio (GAP-TV) | Moderate | 10.4x | Real mask replication errors amplify mismatch |
| CT | CoR offset recovery | 8-9 dB loss | 100% recovery | Matches simulation |
| Ptychography | Phase degradation | 16.1 dB | >99.9% recovery | Matches simulation |

These gaps inform the sealed simulator's parameter ranges and Red Team injection severity.

### 9.4 Acceptance Thresholds for LIP Leaderboard

Based on the flagship paper's validated baselines, a submission must meet these minimum thresholds to appear on the LIP leaderboard:

| Criterion | Threshold | Rationale |
|-----------|-----------|-----------|
| Recovery ratio $\rho$ | $\geq 0.30$ on any single modality | Below this, correction is negligible |
| Minimum gain | $> 0.0$ dB on at least one modality | Must demonstrate positive correction |
| Complete outputs | TriadReport + operator estimate + RunBundle | Reconstruction-only submissions rejected |
| Compute budget | Within 2x declared budget | Over-budget = disqualification |
| Reproducibility | Bit-identical on re-run | Non-deterministic containers rejected |

---

## 10. Maturation of the Harness Itself

LIP Arena does not launch fully formed. It matures alongside PWM:

### Phase A: Internal Harness (0-6 months)

- All evaluation runs locally via `pwm evaluate`
- Sealed-simulator prospective sets only (no partner labs yet)
- 7 validated modalities (CASSI, CACTI, SPC, CT, Ptychography, MRI, Lensless) with datasets from [imaging_modalities.md](imaging_modalities.md)
- **Instant Mode available from day 1** -- `pwm submit --mode instant` for sealed-simulator evaluation; results on LIP within minutes
- Rolling baseline seeded with flagship paper validated numbers (Section 9.1)
- Red Team = PWM development team (adversarial self-testing)
- Publish first 7 counterfactual packs (one per validated modality)
- Establish Commit-Measure-Score tooling and Instant Mode infrastructure

### Phase B: Pilot External Rounds (6-12 months)

- First live-lab partner (1-2 labs); live-lab prospective sets begin; Full Round Mode operational
- 10+ modalities in sealed simulator (add OCT, Photoacoustic, SIM, Ultrasound from expansion roadmap)
- Independent Red Team budget allocated
- PWM harness opened to third-party method submissions
- Round reports published publicly
- **Instant Mode open to external submissions** -- any researcher can submit a container and get scores within the hour
- Instant Mode leaderboard accumulates history; trends visible

### Phase C: Full Operation (12-24 months)

- 5+ partner labs across modalities
- 20+ modalities, all four tracks active
- Red Team operates independently with dedicated budget
- Quarterly Full Rounds on schedule
- **Instant Mode becomes CI/CD integrated** -- teams can trigger evaluation on every commit
- Leaderboard with prospective dominance scoring (Full Round authoritative; Instant Mode for development)

### Phase D: Utility (24+ months)

- **Instant Mode is the primary interface** -- rolling submissions replace quarterly as the default; Full Rounds become quarterly audits with live-lab + Red Team escalation
- All 64 registered modalities active in sealed simulator
- Hardware-in-the-loop scenarios (live instruments feed directly to Instant Mode pipeline)
- LIP Arena protocol extractable as a standalone standard; anyone can run an instance
- Imaging calibration becomes a commodity evaluated by LIP Arena scores
- Instant Mode turnaround target: < 15 minutes for single-modality, < 1 hour for full suite

---

## 11. Summary: What Makes LIP Arena Different

| Traditional Benchmark | LIP (Living Imaging Physics) Arena |
|----------------------|-----------|
| Static dataset released once | Prospective measurements generated after submission deadline |
| Memorization possible | Memorization mechanically impossible |
| Scores reconstruction only | Scores reconstruction + diagnosis + uncertainty + reasoning |
| No adversarial pressure | Dedicated Red Team with escalating difficulty |
| Average-case ranking | Prospective-dominated + tail-risk weighted ranking |
| Gaming rewarded (optimize PSNR, ignore understanding) | Gaming penalized (wrong diagnosis = rank loss) |
| One-shot evaluation | Dual-mode: Instant (minutes, 24/7) + Full Round (quarterly, authoritative) |
| Wait weeks for results | Instant Mode: submit container, get scores on LIP in minutes |
| Closed evaluation | Open round reports, public RunBundles, published failure taxonomies |
| Benchmark separate from methods | Ships with the methods it evaluates |
| Handful of modalities | 64 registered modalities across 13 categories ([full registry](imaging_modalities.md)) |
| Abstract -- no concrete baselines | Grounded in flagship paper: 7 validated modalities with $\rho$, CIs, Cohen's $d$ |
| No design provenance | Built on [SolveEverything](https://solveeverything.org/) targeting authority pattern |

**The harness ships with the agent. The counterfactual pack is built first; the model comes second. Install one repo, get both. This is how you industrialize imaging.**

---

## 12. Submitting a New Method

PWM ships with the current best methods. To replace one:

1. **Implement the `ReconSolver` protocol** -- your method must accept a measurement `y`, an operator `H`, and return a reconstruction `x_hat` with uncertainty estimates.

2. **Register in the solver YAML** -- add an entry to `contrib/solver_registry.yaml` with your solver's parameters, tier classification, and supported modalities.

3. **Run the harness locally**:
   ```bash
   # Score your method on a specific modality
   pwm evaluate --method my_solver --modality cassi --track correct

   # Run the full 4-scenario protocol
   pwm evaluate --method my_solver --modality cassi --scenarios I,II,III,IV

   # Compare against the current default
   pwm evaluate --method my_solver --method mst_l --modality cassi
   ```

4. **Submit to Instant Mode** (optional but recommended) -- get official LIP scores before opening a PR:
   ```bash
   # Build your container
   pwm container build --solver my_solver --output my_solver.sif

   # Submit to Instant Mode -- results on LIP within minutes
   pwm submit --mode instant --container ./my_solver.sif --modality cassi

   # Submit across all declared modalities
   pwm submit --mode instant --container ./my_solver.sif --modality all

   # Check results
   pwm status --submission-id <sha256-hash>
   pwm leaderboard --mode instant --modality cassi
   ```

5. **Beat the current default** -- if your method achieves a higher recovery ratio, lower oracle gap, or better RoIC than the shipped default (Section 9.1 baselines) on the harness, it is a candidate to become the new default.

6. **Open a PR** -- submit your method with RunBundle artifacts and Instant Mode LIP scores demonstrating the improvement. The PR review verifies that the harness results are reproducible.
