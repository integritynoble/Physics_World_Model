# Bounty 7 — Genesis Principle Polish (Tiered)

- **Amount:** ~168,000 PWM total, tiered per-principle (see §Tiers below)
- **Opens:** Phase 2 mainnet launch
- **Structure:** NOT a single reference-impl bounty. A **per-principle claim
  bounty**: any external domain expert can claim any un-polished genesis
  principle, deliver the artifacts listed in §Deliverables, pass the 3
  verifier agents, and receive the per-principle bounty amount.
- **Reference implementation / acceptance harness:**
  - `papers/Proof-of-Solution/mine_example/cassi.md` (1,595 lines) — the
    structural template every submission must mirror
  - `papers/Proof-of-Solution/mine_example/cacti.md` — sibling-principle
    example (correct way to distinguish from an anchor without base-copying)
  - `pwm-team/pwm_product/genesis/l{1,2,3}/L{1,2,3}-003.json` — CASSI
    L1/L2/L3 JSON exemplar
  - `pwm-team/coordination/agent-cross-domain-verifier/patterns.md` — cluster
    conventions the submission must match
  - The 3 verifier agents (`agent-{physics,numerics,cross-domain}-verifier`)
    are the acceptance harness — identical to how they verified CASSI+CACTI.

## What you build

One or more **Genesis Principles** at cassi-depth. Unlike the infra bounties
(which produce one deliverable for one payout), this bounty is designed for
per-principle claims — claim the principles in your domain of expertise,
author them, submit for verification, get paid per accepted principle.

## Why this bounty exists

The PWM registry's 500 Genesis Principles span imaging, spectroscopy, medical
imaging, fluid dynamics, materials science, quantum mechanics, chemistry,
astrophysics, signal processing, and ~30 other domains. No founding team has
expertise across all. This bounty lets domain experts contribute their
specialty principles at cassi depth in exchange for PWM, parallelizing what
would otherwise be years of founding-team work.

## Tiers, weights, and pool allocation

### Allocation formula (pool-weight math, parity with PWMReward)

Per-principle payout comes from a **single 168,000 PWM Reserve pool** (`M_pool`)
distributed by the same weight formula used for L4 mining rewards:

```
payout_k_total = (M_pool − M(t)) × w_k / Σ_j(w_j over unclaimed principles)
```

where:
- `M(t)` = PWM already paid from Bounty #7 at time t (starts at 0, grows
  toward 168K as principles claim)
- `w_k` = tier weight (see table below)
- `Σ_j(w_j over unclaimed)` = sum of weights across principles not yet
  claimed + triple-verified. Shrinks as claims settle.

**Architectural benefit:** this formula is identical to the mining reward
pool in the core protocol — we reuse `PWMReward.distribute()` rather than
adding a new bounty-payout primitive. One less contract function to audit.

### Tier weights

| Tier | # principles | Weight `w_k` | Effective per-principle payout* |
|------|-------------|-------------|--------------------------------|
| A (founding team, done) | 2 | — | 0 (already done: CASSI #25, CACTI #27) |
| B (anchors) | ~20 | **20** | ~2,000 PWM |
| C (standard) | ~200 | **5** | ~500 PWM |
| D (specialty) | ~278 | **1** | ~100 PWM |

*Effective payout computed from the formula above with
`Σ(weights) = 20×20 + 200×5 + 278×1 = 1,678`. Because Σ shrinks as claims
settle, later claims within the same tier get the same payout as earlier
claims (unlike L4 mining, which has a fixed Σ and shrinking pool —
bounty semantics differ here by design).

### Tier B anchor shortlist

Bountied first because early-miner demand is expected here. Published in
full at bounty-open time in `pwm-team/bounties/07-genesis-priorities.md`.
Preliminary list:

- **A_microscopy** (sub-family anchors): #1 Widefield (done as pilot), #5 SIM,
  #11 PALM/STORM, #10 STED, #17 Phase Contrast, #8 FPM
- **C_medical_imaging** (sub-family anchors): #29 CT, #30 MRI, #32 Ultrasound,
  #33 PET, #42 OCT
- **G_electron_microscopy**: #85 TEM, #92 Cryo-ET
- **L_spectroscopy**: #146 NMR, #137 Raman
- **R_fluid_dynamics**: #170 Incompressible NS, #180 DNS
- **X_computational_chemistry**: DFT (principle # TBD per agent-chemistry's
  JSON numbering)

Expected effort: 10–30 hours per Tier B principle for a domain expert with
existing published work in the area.

### Rolling open schedule

Opens as Tier B first (Tier A already done), then Tier C after 50% of Tier B
closes, then Tier D. Rolling open prevents Reserve overspend and lets the
3 verifier agents scale review capacity.

## Two-stage payout (verify + minability)

**Each principle's `payout_k_total` splits 60/40:**

- **60% — Verification portion** — paid immediately on triple-verify merge.
  - Triggered by: all 3 verifier agents issue ACCEPT reviews → agent-coord
    merges PR → `PWMReward.distribute()` call pays 60% × payout_k_total to
    the author's wallet.
  - Rewards high-quality authoring. Niche principles that may never attract
    miners still get the bulk of the bounty.

- **40% — Minability portion** — paid when **≥ 2 independent L4 solutions**
  finalize against this principle's benchmarks within **12 months** of
  triple-verify.
  - Triggered by: the second finalized cert from a *distinct* SP wallet (not
    the author's) lands on-chain → agent-coord triggers
    `PWMReward.distribute()` for 40% × payout_k_total to the author.
  - Enforces minability: a gamed principle that looks good to verifiers but
    isn't actually solvable by real solvers never pays the second half.
  - Enforces demand: niche principles that nobody mines never pay the second
    half either.

**Unclaimed minability portion after 12 months:** returns to the Reserve
pool (increases M_pool for future bounty waves, or — per Phase 3 DAO
governance — funds adversarial-review bounties).

**Why 60/40, not 100/0 or 0/100:**

- **Not 100/0 on verify:** prevents a class of attacks where an author
  writes plausible-but-unmineable content that passes all 3 verifier gates
  (ε thresholds impossible to achieve in practice; baselines that don't
  exist in working code; spec yaml that seems complete but produces
  zero-signal data). The 40% holdback makes this attack unprofitable.
- **Not 0/100 on solutions:** creates chicken-and-egg — miners won't mine
  unverified principles, authors won't polish without upfront payout.
  60% on verify keeps the flywheel moving; the 40% tail rewards demand.
- **60/40 specifically** balances: 60% is the author's earned share for
  correct authoring; 40% is the market's vote on whether the authoring
  produced real value.

### Worked example — a Tier B anchor

CT principle (#29), Tier B, weight 20, effective payout 2,000 PWM.

- Day 0: author claims slot in `07-claims.md`.
- Day 14: author submits PR with `.md` walkthrough + L1/L2/L3 JSONs.
- Day 16: 3 verifier agents issue 3× ACCEPT. agent-coord merges.
  → `PWMReward.distribute()` pays **1,200 PWM** (60% of 2,000) to author.
  → Principle's `Verification: triple-verified`.
- Day 40: first miner (external SP) finalizes a cert against CT T1.
- Day 55: second miner (distinct SP) finalizes a cert against CT T1 (or any
  other CT benchmark).
  → `PWMReward.distribute()` pays **800 PWM** (40% of 2,000) to author.
- Total author earnings: 2,000 PWM over ~8 weeks.

### Worked example — a Tier D specialty principle (no market interest)

Obscure petroleum-engineering principle, Tier D, weight 1, effective payout
100 PWM.

- Day 0-14: author claims, submits, triple-verified.
- Day 16: **60 PWM paid** to author (60% of 100).
- Days 16-380: zero miners submit certs against this benchmark.
- Day 380 (12 months): minability window closes. **40 PWM returns to Reserve.**
- Author earned: 60 PWM for a niche principle. Fair — they wrote good text
  but the market didn't demand it.

## Deliverables (per principle claimed)

1. **`.md` walkthrough at cassi depth** — rewrite of
   `papers/Proof-of-Solution/mine_example/science/<domain>/<NNN>_<slug>.md`
   to mirror `cassi.md`'s section structure (1,400–1,700 lines). Remove the
   `Verification: deprecated-placeholder` or `draft` marker and replace with
   real per-principle physics.

2. **L1/L2/L3 JSON artifacts** — either create new ones at
   `pwm-team/pwm_product/genesis/l{1,2,3}/L{1,2,3}-<NNN>.json` OR polish
   existing ones authored by content-agents (check the directory first).
   Must include:
   - L1: `artifact_id`, `E.forward_model`, `G.dag` with `L_DAG` correctly
     derived per `STYLE_NOTES.md` §4.2, `W.condition_number_{sub,kappa,effective}`,
     `physics_fingerprint` with principle-unique `sensing_mechanism`,
     `difficulty_delta` and `difficulty_tier` consistent, `mismatch_parameters`
     specific to this principle (NOT inherited from a sibling).
   - L2: `spec_range`, `ibenchmark_range`, `epsilon_fn`, `tiers` (T1-T4 +
     P_benchmark).
   - L3: dataset manifest (`dataset_hash` if you provide data; otherwise
     cite a public dataset with its SHA-256), baseline solver table (3+
     rows matching the sub-family's canonical baselines per `patterns.md`),
     quality_scoring ladder (4+ thresholds).

3. **patterns.md cluster updates** — if your principle introduces a new
   primitive, baseline family, or dataset that's not yet in `patterns.md`
   for your cluster, include the proposed patterns.md diff in your PR.

4. **Verification field** — in the `.md` top-matter, set initially to:
   `Verification: draft (author: <your-github-handle>); awaiting triple-verify`

5. **PR to `main`** — one PR per principle, named
   `feat/principle-<NNN>-<slug>` — makes verifier review and payout tracking
   clean.

## Acceptance harness

Identical to Phase A's CASSI+CACTI verification:

1. `agent-physics-verifier` reviews → writes review file at
   `pwm-team/coordination/agent-physics-verifier/reviews/<NNN>_<slug>.md`.
   Verdict: ACCEPT / REVISE / REJECT. Minimum 3 specific findings per CLAUDE.md.

2. `agent-numerics-verifier` (if physics ACCEPTs) → consistency matrix +
   3 findings + verdict.

3. `agent-cross-domain-verifier` (parallel with numerics) → sibling
   comparison + matrix + 3 findings + verdict. Updates `patterns.md` if
   new conventions introduced.

4. `agent-coord` sees 3 ACCEPT verdicts → merges PR, updates principle's
   `Verification:` to `triple-verified`, triggers `PWMReward.depositBounty()`
   call for the claimed amount to the author's wallet.

Any REVISE blocks payout until addressed. Any REJECT with detailed physics
flaws → no payout; author can rework and resubmit as a new claim.

## Payout mechanics

### Stage 1 — Verification (60%)

1. Claim a principle by commenting on `pwm-team/bounties/07-claims.md`
   (first-come, first-served).
2. Submit PR with `.md` walkthrough + L1/L2/L3 JSONs + any `patterns.md`
   updates.
3. All 3 verifier agents issue ACCEPT → agent-coord merges the PR and
   calls:
   ```
   PWMReward.distribute(
     principle_hash = <L1-hash>,
     recipient       = <author_wallet>,
     amount          = (M_pool - M(t)) × w_k / Σ(w_j_unclaimed) × 0.60
   )
   ```
4. 60% payout hits the author's wallet in the same block finalization
   window as the merge commit.
5. Principle's weight `w_k` removed from `Σ` (to stabilize future
   per-principle payouts at the tier rate).

### Stage 2 — Minability (40%)

6. For 12 months following the merge, agent-coord monitors
   `PWMCertificate.Finalized` events filtered by `h_p = <this principle's L1-hash>`.
7. On the **second** finalized cert from a **distinct SP wallet**
   (`sp_wallet != author_wallet && != first_sp_wallet`):
   ```
   PWMReward.distribute(
     principle_hash = <L1-hash>,
     recipient       = <author_wallet>,
     amount          = prior_stage_1_amount × (40/60)
   )
   ```
8. The remaining 40% hits the author's wallet in the same finalization
   block.
9. If the 12-month window expires without 2 distinct-SP finalized certs,
   the 40% returns to Reserve via
   `PWMReward.returnToReserve(amount, reason="bounty_7_minability_window_expired")`.

## What you may NOT do

- Base-copy from CASSI or the widefield pilot and only rename. The 3
  verifier agents explicitly check for this (physics-verifier catches
  inherited κ triples; cross-domain-verifier checks Jaccard distance;
  both will REJECT base-copies).
- Fabricate solver names. Baselines must be citable in the open literature.
- Claim principles outside your expertise area. Nothing stops you
  mechanically, but the triple-verify process will REJECT physics that
  fails dimensional analysis, has wrong κ values for the modality, or
  cites wrong canonical baselines.
- Submit without running the acceptance checks locally first:
  ```bash
  diff <(grep -E '^##[^#]|^### ' cassi.md | sed "s|CASSI|<Your Principle>|g") \
       <(grep -E '^##[^#]|^### ' <your-principle.md>)
  # expected: empty (structural parity)
  ```

## Schedule

| Milestone | Date |
|-----------|------|
| Bounty Tier B opens | Phase 2 mainnet launch + 7 days |
| Tier C opens (rolling) | When 50% of Tier B claimed |
| Tier D opens (rolling) | When 50% of Tier C claimed |
| Bounty closes | All 500 principles triple-verified OR 12 months after launch (whichever first) |

Unclaimed principles after 12 months: agent-coord assigns to founding-team
content agents as cleanup backlog (reuses the content-agents' existing
fan-out mechanism).

## FAQ

**Q: I'm an expert in principle #X but the JSON was already authored by
agent-<whoever>. Can I still bounty it?**

A: Yes. Check whether the existing JSON passes triple-verify — if it does,
bounty isn't needed; if it has findings from the verifier agents, submit
a polish PR addressing them. You still get paid the tier amount for
converting draft → triple-verified.

**Q: What if two people claim the same principle?**

A: First-come-first-served per `07-claims.md`. If two PRs land within the
same 24h window, agent-coord reviews both and picks the one with fewer
verifier findings. The loser can rework a different principle.

**Q: Can I claim multiple principles?**

A: Yes, up to 10 concurrent claims per author (prevents hoarding while
allowing bulk submissions by experts with broad coverage).

**Q: What about updates after triple-verified?**

A: Post-triple-verified updates use the adversarial-review mechanism in
`VERIFIER_AGENTS.md` §8. Small fixes (typos, baseline additions) are
free. Substantive physics changes require a new principle ID (immutability
protects against retroactive history rewriting).

**Q: Where do I find cluster-specific guidance?**

A: `pwm-team/coordination/agent-cross-domain-verifier/patterns.md`. As of
the bounty-open date, 5 clusters have full pattern entries:
A_microscopy, B_compressive_imaging, G_electron_microscopy, L_spectroscopy,
C_medical_imaging, R_fluid_dynamics. The remaining 22+ clusters will be
seeded as anchor principles pass triple-verify; you can propose the
patterns.md entry as part of your principle PR if your cluster is
pattern-pending.

## Contact / questions

Open a GitHub issue at `integritynoble/pwm-public` with the `bounty-07` label.
agent-coord triages within 48 hours.
