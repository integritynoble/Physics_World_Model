# Track 9 — PWM Grand Challenge: Low-Dose CT Reconstruction (= Track B in two-track strategy)

**Date proposed:** 2026-05-16 — **revised 2026-05-19 (Heyang Zhao intern integration); revised 2026-05-22 (two-track sequencing — Track 9 = Track B follows Track A validation)**
**Owner:** Director + **Heyang Zhao (intern, NextGen PlatformAI, weeks 1-12)** + continuing researcher (postdoc or extended intern, months 4+) + new UTSW PI (Track K dependency)
**Status:** ⏳ Proposed; Months 6-24 post-mainnet (D9+180 → D9+730) — sequencing changed 2026-05-22 to follow PWM-CI-1 validation
**Source:** Vision-alignment discussion 2026-05-16; complements `PWM_VISION_ALIGNMENT_AUDIT_2026-05-16.md`; sequencing per `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md`
**Vision link:** `pwm-team/long-term-vision/PWM_LONG_TERM_VISION_2026-05-12.md` §1.3 (five-year goal), §3.4 (4-layer registry), §7 (CS expansion), §8 (AI agents)
**Portfolio context:** `pwm-team/plan/PWM_GRAND_CHALLENGE_LANDSCAPE_2026-05-16.md` — Tier 1 candidate #1 of 6
**Intern integration:** `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` — Heyang's 16-week mission produces Track 9's first 3 months of empirical groundwork
**COI note:** This grand challenge is intentionally chosen to be **separable from Dr. Zaman's lab work**. Pill-camera / capsule endoscopy was considered and rejected pending Track K completion due to COI risk (see § Why Low-Dose CT Specifically).

---

## Two-track sequencing — Track 9 = Track B (added 2026-05-22)

Per `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md`, PWM adopts a **two-track user acquisition strategy**:

- **Track A (fast academic/research wedge):** PWM-CI-1 (CASSI reconstruction benchmark; public data; low friction; 3-6 months). Launches FIRST.
- **Track B (medical imaging flagship):** Track 9 (low-dose CT; clinical credibility; RSNA / ISBI 2028 target). **Launches AFTER Track A validates the mechanism.**

**Sequencing gate:** Track 9 does NOT actively recruit hospital partners, file IRB, or commit Track K mentor capital UNTIL PWM-CI-1 hits **30+ external method submissions** (target milestone: D9+180 / Month 6).

**Why:** Burning UTSW relationship capital + IRB time + Track K mentor commitment BEFORE the mining mechanism is validated risks wasting institutional credibility on infrastructure that isn't proven. Track A's PWM-CI-1 launch (Months 1-6) is the cheaper proving ground for the verified-AI4Science platform mechanism.

**What Track 9 DOES during Months 1-6 (Phase 1a + 1b):**
- ✅ Quiet preparatory work (Heyang weeks 1-9: baseline reproduction on LIDC-IDRI + AAPM)
- ✅ Track K mentor search at LOW intensity (informal conversations; no formal commitments yet)
- ✅ Track 9 paper-drafting groundwork (Heyang weeks 10-12 unchanged unless Director approves re-scope to PWM-CI-1)
- ❌ Hospital partner recruitment — HOLD until Track A validates
- ❌ IRB filings — HOLD until Track A validates
- ❌ Track K mentor formal commitment — HOLD until Track A validates
- ❌ RSNA / ISBI 2028 public announcement — HOLD until Track A validates

**What Track 9 ACTIVATES at Phase 2 (D9+180 onwards) IF Track A succeeds:**
- ✅ Track K mentor formal commitment + IRB filing
- ✅ PWM-MED-1 mini low-dose CT benchmark (public LIDC-IDRI; non-clinical claims) launches
- ✅ Hospital partner recruitment begins
- ✅ RSNA / ISBI 2028 pre-announcement

**Failure mode (if Track A fails):**
- If PWM-CI-1 gets <10 external submissions by D9+180, mining mechanism is not validated
- Track 9 reverts to extended preparatory work; do NOT launch PWM-MED-1
- Re-evaluate strategy per `PWM_USER_ACQUISITION_STRATEGY` §9.2

**Track 9's existing scope (RSNA / ISBI 2028 grand challenge; 5-year canonical benchmark; NIH R21 / R01 trajectory) is UNCHANGED.** Only the sequencing of preparatory work + public commitments changes — Track 9 becomes a Phase 2/3 activation, not a Phase 1 activation.

---

## Goal

Build the **first of PWM's flagship grand challenges**: the canonical, open, vendor-agnostic benchmark + theoretical framework + reference solution for low-dose CT reconstruction, hosted on PWM.

By the end of Track 9, the following sentence should be defensibly true:

> *"If your low-dose CT reconstruction method is not on the PWM Low-Dose CT Challenge leaderboard, it is not state-of-the-art."*

That is the AlphaFold-CASP model applied to a problem you can actually deliver as a solo founder with 2 Reserve-funded collaborators.

---

## Why this track is needed

Three independent reasons converge on Track 9:

**1. PWM needs a flagship killer-app benchmark.** The current registry has 1,597 artifacts but no problem it's *famous for solving*. Trust and adoption follow flagship achievements — Bitcoin had "censorship-resistant digital gold," Ethereum had "Cryptokitties / Uniswap," AlphaFold had "CASP14." PWM needs one. A grand challenge with PWM-canonical benchmark + reference solution + ongoing leaderboard is the most credible way to produce one in 24 months.

**2. Director needs a faculty path.** Track 5 (UTSW PI transition) is necessary but insufficient. To convert from Research Associate to Assistant Professor, Director needs an *independent research program* with publications, grants, and demonstrated leadership of a research direction. Track 9 produces all three: 3-5 publications, R21 (and downstream R01), and named leadership of the "PWM Low-Dose CT Challenge."

**3. Track 8 leaderboards need bootstrapping.** The audit identified empty leaderboards as Gap 2. Reference cert grants (8b) help, but the highest-leverage seed is **one canonical benchmark with PWM team as the originating SP**. Track 9 produces it.

## Why Low-Dose CT specifically

Considered three candidate grand challenges; Low-Dose CT wins on the metrics that matter for your specific situation:

| Factor | Low-Dose CT | Pill-camera (WCE) | Universal CASSI |
|---|---|---|---|
| COI risk vs Dr. Zaman | ✅ Clean separation | ❌ Direct overlap (Zaman lab) | ✅ Clean |
| Clinical urgency | ⭐⭐⭐ Lung screening, pediatric, follow-up | ⭐⭐ Small-bowel disease | ⭐ Specialty optical |
| Annual procedure volume (US) | ~85M scans | ~500K-1M studies | N/A (research instrument) |
| NIH fundability | ⭐⭐⭐ R21 → R01 via NIBIB/NCI/NHLBI | ⭐⭐ R21 via NIDDK | ⭐ Limited |
| UTSW clinical access | ⭐⭐⭐ Radiology dept; routine | ⭐⭐⭐ Requires Zaman lab | ⭐ Limited |
| Path to faculty (radiology / med AI) | ⭐⭐⭐ Standard | ⭐⭐ Specialty | ⭐ Engineering only |
| Industry vacuum to fill | ⭐⭐⭐ No cross-vendor benchmark exists | ⭐⭐ Smaller need | ⭐ Limited adoption |
| Director's existing expertise | ⭐⭐⭐ CT in 172-modality engine | ⭐⭐ Possible from Zaman work | ⭐⭐⭐ Yang & Yuan 2026 |

The COI separation is the single decisive factor. Track 9 cannot depend on Dr. Zaman's lab data, funding, or IRB until Track K completes — otherwise Track 9 itself becomes a COI vulnerability that endangers Track 5 and the faculty path.

---

## Position in the planned grand-challenge portfolio

Track 9 is **the first of a planned portfolio of 3-5 PWM-led grand challenges by 2031.** Low-Dose CT was selected to go first because it maximizes alignment with Director's existing expertise, UTSW clinical access, and the faculty pathway (Track 5). It is **not the endpoint** — it is the template that subsequent grand challenges will replicate.

Candidate future grand challenges (named but not calendar-committed):

| Candidate | Lead recruiting profile | Earliest realistic start |
|---|---|---|
| Real-time accelerated MRI reconstruction | Track 9 postdoc transitioning to independent PI | D9 + 24-36 mo |
| Multimodal cancer characterization (imaging + genomics + pathology → treatment response) | Joint imaging + oncology recruit | D9 + 36-48 mo |
| Virtual cell / single-cell perturbation prediction | Recruited single-cell ML scientist | D9 + 36-60 mo |
| Climate model downscaling / weather nowcasting | Recruited climate-AI scientist | D9 + 48-60 mo |
| Biological-age prediction across diverse cohorts (longevity-adjacent) | Recruited longevity-AI / epidemiology scientist | D9 + 60+ mo *(speculative — verification cycles long, benchmark consensus weak; narrower scope than "reverse aging")* |

The replication model is **founder seeds the pattern, then the pattern replicates**: Director leads Track 9 personally; Tracks 10 onward are led by recruited PIs (Track 9 postdoc → independent PI for adjacent imaging fields; new external recruits for fields outside Director's domain). This is the same orchestration model DeepMind used to spawn AlphaProteo and AlphaMissense after AlphaFold — concentrated execution of one challenge, then organizational replication.

By 2031, the realistic portfolio target is **3-5 flagship grand challenges** with PWM-team reference solutions (drawn from the first four candidates above), **20-50 community-authored benchmarks** with active competition (where speculative-tier areas like biological-age may enter if a recruited PI develops them), and **5,000-50,000 long-tail registered principles** via permissionless contribution. Speculative candidates are tracked for opportunism — if a strong recruit happens to bring expertise in that area, the door is open; PWM does not actively seek leadership in fields with verification timelines longer than the founder's career horizon.

**For grant-funder context (NumFOCUS, Sloan, Schmidt, CZI, NIH):** Track 9 is the demonstrator that the grand-challenge model works on a real clinical problem. Funding Track 9 funds the template that scales to all subsequent challenges. Each future grand challenge will be the subject of a separate funding application — R01 mechanisms for clinical fields; CZI / Sloan / Schmidt program grants for biology, climate, and materials science.

**Strategic caveat:** This portfolio framing is *ambition*, not commitment. PWM commits to delivering Track 9. Future tracks depend on (a) Track 9 success, (b) recruited PIs accepting offers, (c) NIH funding cycles, and (d) DAO governance maturity by Year 5. Do not represent any specific Year 3+ track as a deliverable; the portfolio model is the scaling pattern, not a project plan.

---

## Researcher staffing plan

The 24-month track is staffed in two phases:

| Phase | Months | Personnel | Status |
|---|---|---|---|
| **Phase A: Empirical foundation** | Month 1-3 (weeks 1-12 of intern term) | **Heyang Zhao** (intern, NextGen PlatformAI) full-time @ ~40 hrs/wk | ✅ Already onboarded per 2026-05-17 workplan |
| Phase A transition padding | Month 4 (weeks 13-16 of intern term) | Heyang Zhao continuation; handoff documentation; recruitment outreach for Phase B begins | ✅ Within intern's 16-week term |
| **Phase B: Theoretical + reference solution** | Month 5-18 | Continuing researcher: either (a) Heyang Zhao extended past 16 weeks if mutually agreed, OR (b) recruited postdoc, OR (c) recruited senior PhD student | ⏳ Decide by Month 3 |
| **Phase C: Competition + R21** | Month 18-24 | Phase B researcher continues + Director-led launch event | ⏳ Depends on Phase B choice |

**Key change vs. 2026-05-16 draft:** The intern integration **removes recruitment risk for Phase A** (months 1-3) — the longest-lead operational item is already solved. Recruitment risk now applies only to the Phase B transition at month 4. This dramatically de-risks Track 9's launch.

**Heyang's specific role within Track 9** (per `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md`):
- Reproduce 3-5 published low-dose CT methods as Docker-reproducible baselines
- Build the comparison-evidence table for the `ct_qc_platform/` paper revision (delivers Track 8h directly)
- Establish the data + metrics + evaluation pipeline that 9a (dataset construction) inherits
- Run the dataset-cleaning pipeline on public datasets (LIDC-IDRI, AAPM 2016) before 9a's UTSW IRB-gated data lands
- Co-author the 9a dataset paper (Nature Scientific Data) — credited co-author, not just acknowledgment

---

## Sub-tracks

### 9a — Benchmark dataset construction

**Goal:** Acquire and curate a multi-vendor, multi-protocol, multi-site paired-dose CT dataset and publish it as a canonical L3 benchmark on PWMRegistry.

| Item | Target |
|---|---|
| Total patient scans | 500-1,000 paired (normal-dose + matched low-dose; or simulated low-dose from raw projections) |
| Vendor coverage | Minimum 2 of {Siemens, GE, Canon, Philips}; goal 3 |
| Anatomy coverage | Chest (lung screening focus) + abdomen (oncology follow-up); optional pediatric subset |
| Site coverage | UTSW lead + 1-2 partner institutions (target: large academic medical center with different scanner fleet) |
| Annotations | Lesion bounding boxes / segmentations by ≥2 board-certified radiologists; majority-vote ground truth |
| Format | DICOM raw projection data + reconstructed images + annotations + metadata |
| Distribution | PhysioNet credentialed access (HIPAA-compliant) + L3 spec hash on PWMRegistry |
| Compute infrastructure | S3 / Azure storage; ~5-10 TB |

**Effort:** 9 months from IRB submission to first public release.

**Director actions:**
- IRB application via new UTSW PI's department (Track K dependency)
- Negotiate partner-site MOU
- Recruit annotation radiologists (~$5K honorarium each)
- Define metadata schema and DICOM cleaning pipeline (jointly with **Heyang Zhao** in Phase A)

**Heyang Zhao actions (Phase A, weeks 1-12 of intern term):**
- Set up DICOM cleaning pipeline against LIDC-IDRI + AAPM 2016 Low-Dose CT data (interim datasets while UTSW IRB lands)
- Build the metadata schema implementation + validation tests
- Author the dataset-schema markdown + Python data-loader library
- Begin co-authoring the Nature Scientific Data paper

**Deliverables:**
- `pwm-team/grand-challenges/low-dose-ct/dataset-schema.md`
- L3 spec registered on PWMRegistry
- Dataset paper (target: Nature Scientific Data or Medical Image Analysis Data Track) — **Heyang co-author**
- PhysioNet listing
- Reproducible-baseline tables from Heyang's intern work (also feed Track 8h `ct_qc_platform` paper revision)

### 9b — Dose-equivalence theoretical framework

**Goal:** Define and publish a mathematical framework for "method X at dose D achieves task-Y diagnostic-equivalence to normal-dose with probability ≥ p, confidence interval [a, b]." This is the FPB-theorem-style contribution that converts Track 9 from "another denoising benchmark" into a citable scientific advance.

**Framework structure (proposal — to be refined):**

```
Given:
  - Task T (e.g., 5mm pulmonary nodule detection at AUC ≥ 0.95)
  - Reference dose D_ref (normal-dose acquisition)
  - Reduced dose D_red < D_ref
  - Method M (reconstruction algorithm)

Method M is "dose-equivalent at level (D_red, T, ε, α)" iff:
  P[ |Performance(M, D_red, T) - Performance(reference, D_ref, T)| < ε ] ≥ 1 - α

where performance is measured via task-specific clinical metric
(e.g., per-lesion sensitivity at fixed FPR, segmentation Dice, etc.)
```

The framework gives every reconstruction method a **5-tuple credential**:
- Dose ratio achievable (e.g., 0.25× = 25% of normal dose)
- Task (e.g., lung nodule detection)
- Equivalence margin ε
- Confidence level α
- Patient subpopulation (e.g., adult chest, pediatric abdomen)

This is what the field currently lacks. Every commercial vendor publishes a single number ("50% dose reduction"); none gives task-conditional probabilistic guarantees.

**Effort:** 6 months theoretical work + 3 months empirical validation against 9a dataset.

**Director actions:**
- Lead theoretical development (this is your style of work)
- Co-author with Heyang Zhao (Phase A) and/or continuing researcher (Phase B) on validation experiments
- Submit framework paper to top venue (target: IEEE Trans. Medical Imaging or Medical Image Analysis)

**Heyang Zhao actions (Phase A, weeks 9-16 of intern term):**
- Validation experiments against the 5-tuple credential framework using Heyang's reproduced baselines from earlier in the intern term — i.e., compute the 5-tuple for each baseline method as worked-example validation
- Co-author the framework paper (named author, not acknowledgment)

**Deliverables:**
- `pwm-team/grand-challenges/low-dose-ct/dose-equivalence-framework.md`
- Framework paper (~25 pages, theoretical + empirical)
- L2 spec registered on PWMRegistry (the formal specification of the framework)
- Reference Python library `pwm_dose_equivalence` for computing the 5-tuple for any submitted method

### 9c — Reference solution development

**Goal:** Build a credible, open-source low-dose CT reconstruction method that scores in the top 25% of the (initially empty) leaderboard. Not the final SOTA — explicitly designed to be **improvable**, so the community has somewhere to compete.

**Technical approach (proposal — to be refined with continuing researcher in Phase B; Heyang Zhao's baseline reproductions in Phase A inform the architectural choice):**
- Backbone: unrolled iterative reconstruction (combines physical priors + learned priors)
- Pretrained on public datasets (LIDC-IDRI, NIH Chest CT, AAPM Low-Dose CT Grand Challenge 2016)
- Fine-tuned on 9a dataset
- Inference-time uncertainty quantification via deep ensembles or MC dropout
- Open-source under Apache 2.0

**Why not aim for SOTA?** Two reasons:
1. As solo founder + 2 collaborators, you cannot outcompete Siemens / GE / Philips R&D budgets
2. A reference solution that's *almost* SOTA but improvable is more valuable than a solution that's hard to beat — it draws competition and citations

**Effort:** 12 months parallel with 9a + 9b.

**Director actions:**
- Architect the approach
- Co-author method paper
- Submit as first L4 cert against the 9a benchmark on PWM mainnet

**Deliverables:**
- `pwm-team/grand-challenges/low-dose-ct/reference-method/` (open-source Python repo)
- Method paper (target: MICCAI or IEEE TMI)
- L4 cert on PWMRegistry, scored against the 9a benchmark
- Reproducibility RunBundle on IPFS

### 9d — Permanent leaderboard + community competition

**Goal:** Convert the benchmark + reference method into a self-sustaining competition that becomes the citation standard for the field.

| Item | Target |
|---|---|
| Launch venue | RSNA 2027 or ISBI 2027 (whichever fits 12-15mo post-mainnet) |
| Prize pool | $50-200K equivalent in PWM tokens (Reserve-funded) |
| Submission process | Containerized solution + RunBundle → S1-S4 verification → leaderboard ranking |
| Industrial outreach | Invite Siemens, GE, Philips, Canon to submit (anonymized if they prefer); their participation legitimizes the benchmark |
| Annual rhythm | Year-2 leaderboard refresh; "Yang Low-Dose CT Challenge" becomes a recurring event |
| Citations target | ≥100 within 24 months of launch |

**Effort:** Launch event: 3 months prep + 1 month execution. Ongoing leaderboard maintenance: ~2 hrs/week thereafter.

**Director actions:**
- Lead launch event at RSNA/ISBI (gives you a major-venue speaking slot)
- Curate submissions; coordinate with PWM scoring infrastructure (Track 8 outputs)
- Publish annual "State of Low-Dose CT" review citing leaderboard standings

**Deliverables:**
- Public web page at `lowdosect.physicsworldmodel.org` (or similar) — leaderboard + dataset + framework + reference method, all linked
- Annual review paper (Year 2 onwards)
- Reputation: "PWM Low-Dose CT Challenge" becomes a name in the field

### 9e — NIH R21 / R01 submission

**Goal:** Convert Track 9 work into NIH funding — first an R21 (with new UTSW PI as PI, Director as Co-I), then a follow-on R01 with Director as PI once faculty appointment is secured.

| Phase | When | Action |
|---|---|---|
| R21 LOI prep | D9 + 270 | Draft specific aims around 9a (dataset) + 9b (framework) |
| R21 submission | D9 + 365 | Submit with new UTSW PI as PI, Director as Co-I, on first available NIH cycle (Jun 16 or Oct 16, 2027) |
| R21 review | D9 + 540 | Standard NIH cycle |
| R21 funded | D9 + 730 (if successful) | $275K direct over 2 years; covers Track 9 collaborators continuation |
| R01 LOI prep | D9 + 730 | If R21 funded + faculty offer landed, begin R01 planning |
| R01 submission | D9 + 900 | Standard cycle |

**Hard prerequisite:** Track K (new UTSW PI) must complete by D9 + 270 for this timeline to hold. If Track K slips, Track 9e slips by the same amount.

---

## Critical path & dependencies

```
Track 9 dependencies (revised 2026-05-19 for Heyang intern integration)
═══════════════════════════════════════════════════════════════
[Mainnet deployed]  ← Track 1 must finish (D9 = 2026-05-19/20)
        │
        ▼
[Phase A: Heyang Zhao intern]  ← Already onboarded (papers/INTERN_WORKPLAN_..._HEYANG)
        │  Months 1-4 (weeks 1-12 baselines + dataset pipeline; weeks 13-16 handoff)
        │  ✅ NO recruitment dependency for Phase A
        │
        ├──────────── parallel ────────────┐
        ▼                                   ▼
[Track K: new UTSW PI confirmed]  [Phase B continuing-researcher recruitment]
  (target: D9 + 90 → D9 + 270)     (start outreach D9+60; onboard by D9+120)
   ← Required for 9a IRB + 9e R21    ← Postdoc OR Heyang extension OR PhD student
        │                                   │
        └───────────────┬───────────────────┘
                        ▼
[9a Dataset construction]  ── Months 1-9 ──▶ L3 benchmark on chain
   Phase A: Heyang builds pipeline + interim baselines (public datasets)
   Phase B: UTSW IRB + scan acquisition + radiologist annotations
        │                                            │
        ▼                                            │
[9b Dose-equivalence framework]  ── Months 6-12 ──▶ L2 spec on chain + framework paper
   Phase A: Heyang validates framework against reproduced baselines
   Phase B: Theoretical depth (Director-led) + framework paper drafting
        │                                            │
        ▼                                            │
[9c Reference solution]  ── Months 6-18 ──▶ L4 cert on chain + method paper
        │                                            │
        ▼                                            │
[9d Public competition launch]  ── Months 12-15 ──▶ RSNA/ISBI venue + leaderboard live
        │
        ▼
[9e R21 submission]  ── Month 12-15 ──▶ NIH review → funding decision Month 18-24
        │
        ▼
[Track 9 done — flagship grand challenge active; faculty narrative ready]
```

Total wall-clock: **~24 months** from mainnet to "Yang Low-Dose CT Challenge is the standard benchmark in the field."

---

## Budget (from Reserve)

| Item | Cost (USD) | PWM equivalent (at $1.30) | Phase |
|---|---|---|---|
| **Heyang Zhao intern stipend (16 weeks @ NextGen PlatformAI rate)** | **$10-25K** | **~8-19K PWM** | Phase A (months 1-4) |
| Continuing researcher — postdoc OR extended intern (months 5-24) | $90-130K | ~70-100K PWM | Phase B+C (20 months) |
| PhD student stipend (months 5-24 if recruited; new PI's dept covers half if possible) | $25-50K | ~19-38K PWM | Phase B+C |
| Cloud compute (training, leaderboard hosting, S3/Azure storage) | $50K | ~38K PWM | All phases |
| Annotation radiologist honoraria (4 radiologists × $5K) | $20K | ~15K PWM | Phase A-B (during 9a) |
| Conference travel + dataset hosting (PhysioNet, IPFS pinning) | $30K | ~23K PWM | All phases |
| IRB admin + data collection support | $30K | ~23K PWM | Phase A (with Track K) |
| Competition prize pool (paid Year 2 in PWM) | $50-200K | ~38-154K PWM | Phase C (9d) |
| Open-access publication fees (3-5 papers × $3K) | $15K | ~12K PWM | All phases |
| Contingency (10%) | $30-55K | ~23-42K PWM | — |
| **Total** | **$350-575K over 24 months** | **~270-440K PWM** | |

**Net savings vs. 2026-05-16 draft:** ~$50-70K because:
- Intern stipend ($10-25K for 16 weeks) replaces ~3 months of postdoc stipend ($37K)
- Phase B postdoc starts month 5 instead of month 1 (saves ~4 months of postdoc fringe-loaded cost)
- Recruitment urgency reduced — no scramble for month-1 postdoc start

Combined with Tracks 7+8 (~$215-475K), total Reserve commitment now runs **~$565K-$1.05M** against the 1.09M PWM Reserve. Slightly more comfortable than the 2026-05-16 estimate.

**Funding offset:** If R21 is funded (D9 + 18-24mo), it provides $275K direct that can backfill ~50% of Years 2-3 Reserve burn. This is the planned recycling: Reserve seeds Track 9; NIH funding sustains it; Reserve is replenished for future grand challenges.

---

## Definition of done

The track ships when all of the following are true:

- [ ] **9a:** Dataset published with ≥500 paired scans across ≥2 vendors and ≥2 sites; Nature Scientific Data or Medical Image Analysis Data Track paper accepted; L3 registered on PWM mainnet
- [ ] **9b:** Dose-equivalence framework paper accepted at IEEE TMI or Medical Image Analysis; `pwm_dose_equivalence` library published; L2 spec registered on PWM
- [ ] **9c:** Reference method open-sourced under Apache 2.0; method paper accepted at MICCAI/IEEE TMI/ISBI; first L4 cert registered on PWM mainnet
- [ ] **9d:** Public leaderboard live; ≥20 community submissions; ≥1 industrial vendor participated; named launch session at RSNA or ISBI
- [ ] **9e:** R21 submitted with new UTSW PI; (stretch) R21 funded
- [ ] **Citations:** ≥50 citations across the framework + dataset + method papers by D9 + 24mo
- [ ] **Faculty narrative:** Director's CV includes "Lead author, Yang Low-Dose CT Challenge benchmark + reference method; published framework; R21 PI/Co-I" — ready for assistant-professor applications Q1 2028

---

## Open questions (decide before starting)

| # | Question | Recommended default | When to decide |
|---|---|---|---|
| 1 | Which UTSW department (Radiology vs Radiation Oncology vs BME) for IRB + new PI? | Radiology (largest grant pool, most relevant clinicians) | Track K mentor selection (D9 + 90-270) |
| 2 | Single-task benchmark (e.g., lung nodule detection only) or multi-task (lung + liver + pancreas)? | Start single-task (chest / lung screening); extend in Year 2 | D9 + 30 |
| 3 | Cross-vendor partner — start with 1 institution or aim for 3? | Start with 1 partner; aim for 3 by competition launch | D9 + 60 |
| 4 | Industrial partner involvement — invite at benchmark launch or wait for organic interest? | Wait until 9c reference method is published; then invite as challengers | D9 + 540 |
| 5 | Open data or controlled access (PhysioNet credentialed)? | PhysioNet credentialed (HIPAA + IRB-compliant; standard in field) | D9 + 90 |
| 6 | Prize structure — 1st-place only or distributed top-10? | Distributed top-10 (encourages broad participation) | D9 + 365 |
| 7 | Pediatric subset — include or defer to Year 2? | Defer to Year 2 (separate IRB, different annotation pipeline) | D9 + 30 |
| 8 | Should reference method use diffusion / generative models? | No — start with unrolled iterative (better uncertainty quantification, more interpretable) | D9 + 180 |

---

## Risks

| Risk | Probability | Mitigation |
|---|---|---|
| **Track K slips past D9 + 270** | Medium (30%) | If Track K isn't on track by D9 + 90, defer Track 9 start by the same amount or longer; do not proceed without confirmed PI |
| **Phase A intern (Heyang Zhao) departs before Phase B handoff completes** | Low (10%) | Intern is already onboarded per 2026-05-17 workplan; 16-week commitment in writing. Mitigation: Phase A handoff documentation requirement (every deliverable + Docker image + README) catches knowledge before departure. Also: invite Heyang to extend past 16 weeks if performance is strong + mutually agreed |
| **Phase B continuing researcher recruitment fails (month 3-4 transition)** | Medium (30%) | Start outreach at month 2 (~D9+60); broad pool (UTSW + collaborator institutions + national applicants); first fallback = extend Heyang's intern term past 16 weeks; second fallback = senior PhD student already at UTSW |
| **IRB delays > 6 months** | Medium (40%) | Begin pre-submission consultation D9 + 30; parallel-track 9b (theoretical) which doesn't need IRB; if IRB stalls, use existing public datasets (LIDC-IDRI, AAPM 2016) as interim |
| **Industrial vendors refuse to participate** | High (60%) | Don't depend on them; design benchmark to be useful with academic methods only; vendor participation is bonus, not requirement |
| **Reference method (9c) underperforms** | Medium (30%) | Acceptable — explicit goal is "credible and improvable," not SOTA; if seriously below baseline, license existing open-source method and call it the "PWM baseline" |
| **R21 not funded first cycle** | High (75% — base rate) | Resubmit; R21 typical 2-3 cycle process to funding; not a failure mode |
| **Director's UTSW workload blocks Track 9** | Medium (40%) | This is why the postdoc + PhD student are non-negotiable; if you cannot delegate, the track collapses to evenings/weekends → won't ship in 24 months |
| **Vision audit gap re-emerges (PWM lacks AI-agent integration)** | Track 7 covers this | Track 9 is a *complement* to Track 7, not a replacement. Both run in parallel post-mainnet. |

---

## What success looks like at Year 3

**Director's CV by 2029-Q3 (D9 + 38 months):**

- Yang, C. et al. (2027). *PWM Low-Dose CT Benchmark: A Multi-Vendor Paired-Dose Dataset for Reproducible Reconstruction Research.* Nature Scientific Data.
- Yang, C. (2028). *A Probabilistic Framework for Diagnostic Dose-Equivalence in Low-Dose Computed Tomography.* IEEE Transactions on Medical Imaging.
- Yang, C. et al. (2028). *Unrolled Iterative Reconstruction with Uncertainty Quantification for Low-Dose CT.* MICCAI.
- Yang, C. et al. (2029). *State of the Field: Two Years of the PWM Low-Dose CT Challenge.* Medical Image Analysis.
- 1 R21 funded (Co-I); R01 in preparation as PI (pending faculty appointment)
- Invited speaker: RSNA 2027, MICCAI 2028, ISBI 2029
- Yang & Yuan 2026 FPB theorem paper (existing) + Yang dose-equivalence framework (new) = two cited contributions

**Faculty applications open Q1 2028 with this packet.** Realistic outcomes:
- Assistant Professor offer at top-25 medical school in radiology / medical imaging informatics / BME (~50% probability)
- Assistant Professor offer at top-50 medical school (~25% probability)
- Industry research scientist at major medical AI company (Siemens Healthineers, GE Research, Philips Research, etc.) (~20% probability — fallback)
- Continued Research Associate with R01 PI status (~5% — last-resort)

**For PWM:** Track 9 produces the protocol's first defensible "PWM secures the solution to X" claim. When AI agents (Track 7) query the PWM Low-Dose CT leaderboard, they get back real verified state-of-the-art methods, not an empty list. That's the moment the vision's killer-app sentence becomes literally true.

---

## What this track does NOT do

To stay scoped:

- **Does not require pivoting away from imaging anchor.** Track 9 is the imaging-anchored grand challenge; it's the deepest expression of the current vision, not a pivot away from it.
- **Does not replace Track 7 or Track 8.** Track 9 *is* the killer use case Tracks 7+8 make possible — they're complements, not substitutes.
- **Does not require Dr. Zaman's lab.** Intentionally chosen to be COI-clean; can proceed entirely under new UTSW PI's umbrella once Track K completes.
- **Does not commit Director to solving low-dose CT at SOTA level.** Goal is credible reference + framework + leaderboard, not "beat Siemens." Industry-beating performance is a stretch target, not a definition-of-done.
- **Does not depend on PWM-as-third-era marketing succeeding.** Track 9's scientific contribution is valuable even if PWM as a protocol fades; the dose-equivalence framework and benchmark are independent durable contributions.
- **Does not commit PWM to a fixed list of future grand challenges.** Tracks 10+ are aspirational targets dependent on Track 9 success, recruited PIs accepting offers, and external funding cycles. See `PWM_GRAND_CHALLENGE_LANDSCAPE_2026-05-16.md` for the full portfolio context.

---

## Cross-references

| Document | What it tells you |
|---|---|
| `pwm-team/long-term-vision/PWM_LONG_TERM_VISION_2026-05-12.md` §1.3, §3.4, §7, §8 | Vision context for grand-challenge framing; Track 9 operationalizes the §7 expansion path as a recruit-led portfolio rather than a domain-by-domain registry expansion |
| `pwm-team/plan/PWM_GRAND_CHALLENGE_LANDSCAPE_2026-05-16.md` | Portfolio landscape; Track 9 is Tier 1 candidate #1 of 6 |
| `pwm-team/plan/PWM_VISION_ALIGNMENT_AUDIT_2026-05-16.md` | Identifies the "no flagship killer app" gap that Track 9 closes |
| **`papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md`** | **Heyang Zhao's 16-week intern workplan — the Phase A staffing source for Track 9 months 1-4** |
| `pwm-team/plan/PLAN.md` § Track 5 | UTSW PI transition — hard dependency for 9a (IRB) and 9e (R21) |
| `pwm-team/plan/PLAN.md` § Track 8h | CT QC Platform paper revision — Heyang's intern work feeds this paper's empirical baselines |
| `pwm-team/plan/track_7/PWM_TRACK_7_AGENT_INFRASTRUCTURE_2026-05-16.md` | Track 9's reference solution becomes Track 7's first real cert to query |
| `pwm-team/plan/track_8/PWM_TRACK_8_SOLUTION_BOOTSTRAP_2026-05-16.md` | Track 9's benchmark is the canonical L3 that Track 8b's reference cert grants extend |
| `pwm-team/funds/PWM_RESEARCH_ASSOCIATE_AND_MENTOR_CONSTRAINTS_2026-05-13.md` | Track K constraint documentation |
| `pwm-team/funds/PWM_PI_TRANSITION_STRATEGY_2026-05-13.md` | Track K timeline that Track 9 depends on |
| Existing public datasets | LIDC-IDRI (NCI), AAPM 2016 Low-Dose CT Grand Challenge (Mayo), NIH Chest CT — usable as interim if 9a IRB stalls |
| `packages/pwm_core/contrib/modalities.yaml` | Existing CT modality definitions — Track 9 extends this |

---

## Timeline summary

| Date | Milestone |
|---|---|
| D9 = 2026-05-19/20 | Mainnet deployed (Track 1 complete) |
| D9 + 0 to D9 + 14 | **Heyang Zhao intern starts** (Phase A); kick-off + repo onboarding per `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` |
| D9 + 30 | Heyang's first reproduced baseline complete (Docker-reproducible); begin **Phase B continuing-researcher outreach** |
| D9 + 60 | Confirm new UTSW PI (Track K dependency); Heyang's 3 baselines complete; metadata schema drafted |
| D9 + 90 | IRB pre-submission consult; partner site MOU discussions begin; Heyang's dataset-cleaning pipeline on LIDC-IDRI + AAPM 2016 running |
| D9 + 120 | **Heyang's 16-week intern term ends** (Phase A → Phase B transition); handoff documentation complete; Phase B researcher onboarding OR Heyang extension confirmed |
| D9 + 180 | IRB approved; first scan acquisitions begin; 9b theoretical work in flight |
| D9 + 270 | 9b framework draft posted; 9c reference solution training underway |
| D9 + 365 | R21 submitted with new PI; 9a dataset paper submitted (Heyang co-author) |
| D9 + 540 | 9c reference solution v1 submitted as L4 cert; method paper submitted |
| D9 + 730 | Public competition launches at RSNA/ISBI 2028; ≥20 submissions; R21 funded (stretch) |
| D9 + 900 | Year-2 leaderboard refresh; R01 LOI submitted; faculty applications drafted |

---

## Decision points

| Date | Decision | Default if not made |
|---|---|---|
| D9 + 30 | **Phase B continuing-researcher** outreach started? Heyang's first baseline reproduction complete? | Defer Phase B start by 1 month per slip |
| D9 + 90 | New UTSW PI confirmed? | Defer Track 9 by Track K slippage |
| D9 + 90 | Track 9 green-lighted given (a) Track 7/8 progress, (b) NumFOCUS status, (c) Director bandwidth? | Defer by 1 quarter |
| D9 + 180 | IRB approved? | Use public datasets (LIDC-IDRI / AAPM 2016) as interim; resubmit IRB |
| D9 + 365 | Track 9 budget on plan? | Pause 9d competition launch; preserve runway for 9a-c |
| D9 + 540 | Reference method v1 acceptable? | Replace with licensed open-source method as PWM baseline |
| D9 + 730 | R21 funded? | Resubmit; continue Track 9 on Reserve until funded |

---

*Last revised: 2026-05-16. Track 9 is advisory until accepted into `PLAN.md`. Recommended acceptance gate: D9 + 7 days (one week post-mainnet), after Track 7/8 commitment is confirmed and Track K mentor candidates are identified.*
