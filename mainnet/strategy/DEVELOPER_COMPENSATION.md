# PWM Developer Compensation — Canonical Reference

**Date:** 2026-05-22
**Audience:** Director + co-founder candidates + future engineering hires + Foundation board (post-501(c)(3)) + grant reviewers
**Status:** Canonical reference; supersedes ad-hoc compensation discussions
**Purpose:** Single source of truth for "how much do PWM developers earn?"

---

## TL;DR

PWM developer compensation has **three layers**:

| Layer | Source | Amount | Mechanism |
|---|---|---|---|
| **L1: Founding team allocation** | 0.63M PWM (3% of 21M supply) | Single immutable beneficiary = Director's Ledger #1 at D9 deploy | `PWMVesting.sol` — 4-year linear, 1-year cliff, immutable beneficiary |
| **L2: External bounty pool** | 1.24M PWM (59% of Reserve) | 50K-500K PWM per bounty (8 OPEN/SPEC + 2 RESERVED) | Reserve escrow → reference-test gates → 30-day shadow → release |
| **L3: Foundation discretionary** | 0.86M PWM (41% of Reserve) | 100-50K PWM per grant; salary + PWM equity for hires; **co-founder economic stake also flows from this layer** | Governance vote (>50K) OR 3-of-5 multisig (≤50K) |

**Total developer-allocatable: ~2.73M PWM (13% of supply).** Plus founding-team vesting on top = 3.36M PWM (16% of supply) flowing to developers over 4 years.

**Path A bootstrap reality (canonical 2026-05-22 update):** The L1 allocation goes to a SINGLE immutable beneficiary at deploy — Director's Ledger #1. PWMVesting.sol is single-beneficiary with no admin function to redirect. Founder-rotation via `executeFounderChange` transfers MULTISIG SIGNING AUTHORITY only — it does NOT transfer PWM. Co-founder economic stake is therefore funded from **L3 (Foundation discretionary)**, not from automatic redistribution of L1. See §2 for the on-chain mechanism + §2.3 for the optional Director-deployed per-co-founder vesting alternative (Option B). **§2.2's prior 50/15/10/10/15 per-founder split is now historical/aspirational only — the contracts do not enforce it and Director's canonical recommendation 2026-05-22 is to retain 100% of L1; full §2.2 + §8 Decision Points rewrite pending.**

---

## 1. The Three Layers Explained

### 1.1 Why three layers?

Different developers need different incentive structures:

- **Founding team** needs long-term skin-in-the-game (multi-year vesting + significant share)
- **External bounty competitors** need clear deliverables + immediate payment after acceptance
- **Foundation discretionary** needs flexibility for one-off PRs, audit firms, conference travel, ongoing hires

A single compensation mechanism cannot serve all three. PWM uses three separate mechanisms with explicit policies for each.

### 1.2 The 21M supply breakdown (where developer comp comes from)

```
21,000,000 PWM total supply
├── 17,220,000 PWM (82%) — Minting pool (programmatic emission to L4 reproducers)
├──  2,100,000 PWM (10%) — Reserve allocation
│    ├── ~1,238,000 PWM (~59%) — External bounty pool (8 bounties)
│    └──   ~862,000 PWM (~41%) — Foundation discretionary (grants, hires, audits, bounties)
├──  1,050,000 PWM (5%) — Liquidity (Uniswap v3 PWM/USDC LP)
└──    630,000 PWM (3%) — Founding team (PWMVesting; 4yr linear, 1yr cliff)
```

**Developer-comp slice = 3% (founding team) + 10% (Reserve) = 13% of total supply.**

Plus: external developers can earn from the **17.22M Minting pool** by submitting verified L4 reproduction certificates as miners. This is structurally different — anyone with research expertise can mine PWM. Mining is its own compensation track separate from infrastructure development; covered in §6.

---

## 2. Layer 1 — Founding Team (0.63M PWM)

### 2.1 What the smart contract actually does

**Critical mechanism (verified against `PWMVesting.sol`):**

```solidity
address public immutable beneficiary;  // Line 29
// "No admin functions for changing schedule or beneficiary. This is intentional"
// "One contract is deployed per beneficiary."
```

At Phase 5 deploy (`deploy/erc20.js:144`):
- ONE `PWMVesting.sol` contract is deployed with ONE immutable beneficiary
- The deploy script sets `teamBeneficiary = process.env.PWM_TEAM_BENEFICIARY` = Director's Ledger #1
- The full 630,000 PWM is transferred to that single contract
- **The beneficiary cannot be changed after deploy.** No admin, no governance override, no upgrade path.

**Result on-chain:**

| Allocation | Amount | Vesting | Beneficiary | Modifiable? |
|---|---|---|---|---|
| `PWMVesting.sol` | **630,000 PWM** | 4-year linear, 1-year cliff | **Director's Ledger #1 (immutable)** | **NO** |

**`PWMGovernance.founders[]` is a SEPARATE mechanism.** It stores 5 signing addresses (governance rights). `executeFounderChange` rotates one slot, changing WHO can sign — but does **not** transfer PWM tokens. Founder rotation gives co-founders **governance power**, not **economic stake**.

### 2.2 What this means for co-founder economics

**At deploy: Director receives 100% of the 0.63M PWM founding-team allocation.** The 5-wallet bootstrap (Path A) means all 5 PWMGovernance signing slots are also Director-controlled.

**Co-founders joining via founder rotation receive governance rights but ZERO PWM automatically.** Any PWM they receive must come from one of four post-deploy mechanisms (§2.3 below), all of which are at Director's discretion + subject to legal agreement, not enforced by the founding-team smart contract.

### 2.3 Four mechanisms to share PWM with co-founders (all post-deploy)

| Option | Mechanism | Pro | Con |
|---|---|---|---|
| **A** | Direct transfer post-cliff (informal) | Simple; no extra contracts | Trust-based; co-founder has no on-chain protection |
| **B** | New `PWMVesting.sol` deployed post-cliff, funded from Director's unlocks | Formal; on-chain proof; matches 501(c)(3) audit needs | Co-founder's cliff starts at their contract deploy date (not at Phase 5) |
| **C** | Reserve grant via PWMGovernance (3-of-5 + 48h timelock for >5K PWM) | Doesn't dilute Director's 630K; institutional governance trail | Reduces Reserve discretionary pool (2.1M total) |
| **D** | No sharing — keep 100% | Maximum Director economic upside | NumFOCUS / Foundation credibility risk; co-founder recruitment difficult |

**Recommendation:** Hybrid B + C — Director shares part of own 630K via Option B (formal vesting) for co-founders who joined early and have substantial role; supplements with Option C (Reserve grants) for co-founders with bounded contributions. Option D not recommended for NumFOCUS-eligible Foundation trajectory.

### 2.4 PROPOSED voluntary redistribution targets

The 4 co-founder slots from Track 4a need a proposed economic-share target. This is what Director should commit to in writing (CLA + co-founder agreement) — NOT what the smart contract enforces. Director can break the written commitment; doing so opens legal + reputational risk.

| Founder slot | Role | Voluntary share of 630K | PWM target | Mechanism (A/B/C) |
|---|---|---|---|---|
| **Director (slot 0)** | Founder, sole spokesperson, technical architect | **50-70%** | **315K-441K PWM** | Retains via PWMVesting (immutable) |
| Co-founder #2 (slot 1) | Computational imaging researcher (non-UTSW) | 10-15% | 63K-94K PWM | B (new vesting post-cliff) |
| Co-founder #3 (slot 2) | Open-source contributor | 5-10% | 31K-63K PWM | B (new vesting post-cliff) |
| Co-founder #4 (slot 3) | Crypto/blockchain ecosystem contributor | 5-10% | 31K-63K PWM | B (new vesting post-cliff) |
| Co-founder #5 (slot 4, optional) | Domain expert | 10-15% | 63K-94K PWM | B (new vesting post-cliff) OR C (Reserve grant) |
| **Total redistributed** | | **30-50%** | **189K-315K PWM** | |
| **Director retains** | | **50-70%** | **315K-441K PWM** | Direct via PWMVesting |

**Rationale for 50-70% Director retention:**

- Director has done ~24 months of work pre-deploy (alone) — substantial unmatched contribution
- Co-founders join post-mainnet with bounded role + responsibilities; their compensation reflects forward contribution
- 50-70% retention is the standard for "founder + later co-founders" in startups + nonprofit-tech (Mozilla, Wikimedia, Linux Foundation comparables)
- Lower bound (50%) is most defensible for NumFOCUS / grant reviewers; upper bound (70%) defensible given unmatched pre-deploy work

### 2.5 Critical: the written agreement is the actual commitment

Because the smart contract gives Director 100% control, **the only thing that binds Director's voluntary redistribution is the written co-founder agreement (CLA + per-co-founder vesting agreement).** Standard practice:

1. **Pre-recruitment**: Director publishes voluntary-redistribution policy (this doc + §8 decisions)
2. **At recruitment**: signed CLA + signed co-founder agreement specifying:
   - Exact PWM share target (within published range)
   - Mechanism (Option B vs C)
   - Timeline (when Option B contract will be deployed)
   - Departure terms + claw-back
3. **At Option B contract deploy** (post-cliff, ~Year 1+): Director deploys new PWMVesting for co-founder, funded from Director's unlocks
4. **On-chain proof**: PWMVesting contract address published in `coordination/agent-coord/co-founders/<name>-vesting.md`

**The smart contract enforces NOTHING for co-founder economics.** The legal + reputational structure is the binding mechanism. This is intentional — Path A bootstrap requires Director's economic control during the high-uncertainty Year 1 + 2.

### 2.6 Cliff timing for late-joining co-founders (Option B)

For co-founders receiving PWM via Option B (new vesting contract):

- **Their 1-year cliff starts at THEIR Option B contract deploy date**, not Phase 5 deploy date
- Director's PWMVesting has its OWN cliff at D9 + 365 days; Director must wait for the cliff before funding Option B contracts

So a co-founder joining at Month 3 might see:
- Month 3: signs co-founder agreement; receives governance rights via founder rotation
- Month 12: Director's PWMVesting cliff hits; Director can `release()` first tranche
- Month 12: Director deploys Option B vesting for co-founder, funded from released PWM
- Month 12 + 365 = Month 24: co-founder's cliff hits; their first PWM unlocks

This is ~21 months from co-founder join to their first PWM unlock — substantial commitment required from both sides.

### 2.7 Co-founder departures

If a co-founder leaves before their Option B vesting completes:

- **Voluntary departure**: keep all tokens vested so far in THEIR Option B contract (immutable beneficiary protects them); Director simply stops funding future Option B contracts
- **For-cause departure** (fraud, severe breach): vested tokens may be subject to Foundation reclamation per CLA — but ONLY if the CLA explicitly preserved this right; smart contract itself doesn't enforce reclamation
- **Pre-Option-B-deployment departure**: if co-founder leaves before their Option B contract is deployed, Director simply doesn't deploy it; no economic transfer occurred

**Key asymmetry:** Director's 630K is immutable + cannot be clawed back even if Director leaves. Co-founder vesting (via Option B) is also immutable once deployed. The economic asymmetry favors whichever party deploys their vesting contract first.

---

## 3. Layer 2 — External Infrastructure Bounties (~1.24M PWM)

### 3.1 The 8 bounties

Per `pwm-team/bounties/INDEX.md`:

| # | Bounty | PWM | Status | Reference impl |
|---|---|---|---|---|
| 1 | **Scoring engine** | **200,000** | ✅ OPEN | `infrastructure/agent-scoring/pwm_scoring/` |
| 2 | **Web UI / Explorer** | **80,000** | ✅ OPEN | `infrastructure/agent-web/` |
| 3 | **pwm-node CLI** | **100,000** | ✅ OPEN | `infrastructure/agent-cli/pwm_node/` |
| 4 | **Mining client (CP role)** | **80,000** | ✅ OPEN | `infrastructure/agent-miner/pwm_miner/` |
| 5 | **Smart contracts competing impl** | **500,000** | SPEC PUBLISHED (opens Phase 1 sign-off) | `infrastructure/agent-contracts/contracts/` |
| 6 | **IPFS benchmark pinning** | **50,000** | SPEC PUBLISHED (opens Phase 1 sign-off) | SLA-driven (no reference) |
| 7 | **Genesis Principle polish (tiered)** | **~168,000** | SPEC PUBLISHED (opens Phase 2 launch + 7 days) | `cassi.md`/`cacti.md` + L1/L2/L3 JSON exemplars |
| 8 | **LLM-routed NL matcher** | **60,000** | SPEC PUBLISHED (opens CASSI/CACTI MVP gate) | `agent-cli` + `agent-web` faceted matcher |
| **Total** | | **~1,238,000** | | |

### 3.2 Acceptance protocol (same for all 8)

1. **Scope check** — implements mandatory interface from bounty spec
2. **Reference test suite** — passes every test the founding team's reference passes
3. **Interface parity** — drop-in replacement for reference
4. **30-day shadow run** — runs alongside reference on testnet
5. **Payment** — PWM released from Reserve escrow to winning submission

### 3.3 Multiple winners per bounty?

**By default: single winner.** First competing implementation that passes all 5 gates gets the bounty. Second-place submissions earn reputation but not the bounty.

**Exception (Bounty 7): tiered per-principle claim model** — multiple authors can each claim individual principles. Total pool fixed at ~168K PWM distributed across hundreds of claims.

### 3.4 What's NOT bountied?

Some infrastructure work is intentionally NOT bountied because:

| Item | Why no bounty |
|---|---|
| Day-to-day maintenance of reference implementations | Foundation discretionary (Layer 3) |
| Documentation improvements | Foundation discretionary (small contributor grants per §4.3) |
| Cosmetic UI changes | Foundation discretionary |
| Compliance / legal work | USD-paid via grants, not PWM |
| Audit firm engagement | USD-paid; firm doesn't accept PWM as primary |

The bounty pool covers **competing reference-implementation builds**, not maintenance, polish, or operational work.

---

## 4. Layer 3 — Foundation Discretionary (~862K PWM)

### 4.1 The slack budget

Of the 2.1M Reserve, 1.24M is earmarked for external bounties. The remaining **862K PWM is Foundation discretionary** — funds:

- Engineering hires (post-grant landing, Year 2+)
- Open-source contributor grants (small PRs)
- Audit firm payments (when paid in PWM)
- Research grants (Track 8 / Track 9 specific work)
- Conference + workshop sponsorships
- Bug bounty payouts (post-deploy Immunefi)
- Marketing / outreach experiments
- Emergency incident response fund

### 4.2 Governance rules

| Amount | Approval required | Speed |
|---|---|---|
| ≤ 5,000 PWM | Director (Path A bootstrap) OR 3-of-5 multisig (post-rotation) | Same-day |
| 5,001-50,000 PWM | 3-of-5 multisig + 48h timelock | ~3 days |
| > 50,000 PWM | DAO vote (≥⅔ weight, 14-day window) — when DAO active | ~3 weeks |

**Pre-DAO activation (Year 1)**: all >5K PWM spends go through 3-of-5 multisig + 48h timelock. Per spec, the DAO doesn't activate until at least D9+12 months.

### 4.3 PROPOSED contributor-grant rate card

For small PRs / contributions that don't fit a bounty:

| Contribution size | PWM grant range | Examples |
|---|---|---|
| < 10 LoC documentation fix | 100-500 PWM | Typo fix, code comment, README update |
| < 100 LoC test or bug fix | 500-2,000 PWM | New unit test, minor bug fix |
| < 500 LoC feature | 2,000-10,000 PWM | New UI route, helper function, dashboard widget |
| > 500 LoC feature | 10,000-50,000 PWM | New explorer page, indexer optimization, agent SDK module |
| > 5,000 LoC project | Treat as new bounty (governance proposal) | New major subsystem |

**Approval flow:**

- < 5K PWM grants: Director's discretion (Path A); 3-of-5 multisig post-rotation
- > 5K PWM grants: multisig + timelock
- Public proposal log on `physicsworldmodel.org/grants` (Year 2)

### 4.4 PROPOSED Bug Bounty (Immunefi-style) tier card

Pre-deploy commitment for post-deploy bug bounty program:

| Severity | USDC payment (from grant funding when available) | PWM bonus (from Reserve, post-LP) |
|---|---|---|
| **CRITICAL** (active exploit; funds at risk; protocol-bricking) | **$5,000** | **50,000 PWM** |
| **HIGH** (vulnerability disclosed pre-exploit; bounded by caps) | **$1,500** | **20,000 PWM** |
| **MEDIUM** (griefing; DoS; recoverable issue) | **$500** | **5,000 PWM** |
| **LOW** (best-practice deviation; minor info leak) | **$100** | **1,000 PWM** |
| **INFO** (style, doc; not exploitable) | Public acknowledgment | **100 PWM** |

**Total Reserve commitment if all 5 tiers triggered once each: 76,100 PWM** — minor. Realistic Year 1 spend: 1-3 LOW/INFO + maybe 1 MEDIUM = ~10-20K PWM.

**Immunefi page setup**: ~$1-5K USD + ~30 hr Director time. Recommended Month 3-6 post-deploy when grant funding lands.

### 4.5 Future engineering hire structure

When Foundation grants land (Year 2+), salaried hires possible:

| Role | USD salary range | PWM equity (4-yr vest, 1-yr cliff) | Source |
|---|---|---|---|
| Senior Web3 engineer (Solidity) | $130-180K/yr | 15-30K PWM/yr vesting (60-120K total over 4 yr) | Foundation employs |
| Senior full-stack engineer | $110-150K/yr | 10-20K PWM/yr (40-80K total) | Foundation employs |
| Mid-level engineer | $80-120K/yr | 5-10K PWM/yr (20-40K total) | Foundation employs |
| Junior engineer | $60-100K/yr | 2-5K PWM/yr (8-20K total) | Foundation employs |
| Researcher (PhD-level, post-doc) | $50-90K/yr | 3-8K PWM/yr (12-32K total) | Grant-funded |
| Intern (e.g., Heyang Zhao) | $30-50/hr ($30-50K for 16 weeks) | 5-10K PWM completion bonus | Foundation directly |

**Hybrid USD + PWM**: Foundation pays cash salary from grants (Sloan / CZI / NSF) + PWM equity vesting from Reserve. This is the **standard nonprofit-tech structure** — Wikimedia, Mozilla, Linux Foundation all use it.

**PWM-only compensation is RISKY for hires:**

- US tax implications: receiving PWM as comp creates ordinary-income tax at receipt
- Liquidity constraint: PWM market makers + LP needed before PWM equity has cash-out path
- Living expense reality: most engineers need USD salary for rent/food, can't subsist on PWM
- Diversity risk: PWM-only attracts crypto-natives only; excludes domain experts

**Standard offer formula**: 60-80% USD salary + 20-40% PWM equity. Adjustable based on candidate seniority + risk tolerance.

### 4.6 Heyang Zhao intern compensation

Per `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md`:

| Item | Amount | Notes |
|---|---|---|
| Stipend | $19-32K USD (16 weeks full-time at ~$30-50/hr) | From NextGen PlatformAI or Foundation when active |
| PWM bonus | **5,000-10,000 PWM** | Completion bonus paid from Reserve; payable when caps lift + LP active |
| Co-authorship credit | 2 papers (Track 9 dataset + framework) | Non-monetary; high career value |
| Recognition | Public acknowledgment as Track 9 contributor | Long-term reputational asset |

Total economic value for 16-week intern: ~$22-40K USD-equivalent (cash + PWM-when-redeemable + career credit).

---

## 5. PWM-vs-USD Denomination Strategy

### 5.1 The denomination question

Should developer compensation be denominated in PWM, USD, or hybrid?

| Denomination | Pro | Con |
|---|---|---|
| **PWM-only** | Aligns developer + protocol success | High volatility; tax complexity; living-expense gap |
| **USD-only** | Predictable; standard; tax-simple | No protocol alignment; doesn't build community |
| **Hybrid** | Aligns + bounded volatility risk | Complexity in formula |

### 5.2 Recommended policy

**Founding team (Layer 1)**: PWM-only (already locked via PWMVesting). Founders accept token-economic risk in exchange for upside.

**External bounties (Layer 2)**: PWM-only. Bounty hunters explicitly opt into PWM-token compensation.

**Engineering hires (Layer 3)**: **Hybrid** — 60-80% USD + 20-40% PWM equity vesting. Funded from grants (USD) + Reserve (PWM).

**Small contributor grants (Layer 3)**: PWM-only — small amounts, contributor takes it knowing it's PWM.

**Audit firm payments (Layer 3)**: USD by default; PWM accepted at firm's discretion (rare in practice).

**Bug bounty (Layer 3)**: Hybrid — USDC for predictable payout + PWM bonus for upside.

### 5.3 Tax & legal considerations

For US-based recipients:

- PWM received as compensation → ordinary income at fair-market value at receipt
- PWM held → capital gains when sold (long-term if held >1 yr)
- Founders should consult a crypto-aware CPA (per `coordination/PRE_DEPLOY_RISK_AUDIT_2026-05-21.md` action item)

For non-US recipients: similar concepts; check local tax law.

**Foundation status (501(c)(3)) implications:**

- W-2 employees can be compensated in USD + PWM
- 1099 contractors face self-employment tax on PWM-as-comp
- International contractors face local-jurisdiction rules

---

## 6. The Mining Pool (Separate Track)

Beyond infrastructure compensation, anyone can earn from the **17.22M Minting pool** by submitting verified L4 reproduction certificates as miners:

| Allocation | Amount | Mechanism |
|---|---|---|
| Reward distribution (per draw) | 40% rank-1; 5% rank-2; 2% rank-3; 1% rank-4-10 | Programmatic emission from `PWMMintingERC20.mintFor` |
| Split per draw (after rank) | AC p×55%, CP (1-p)×55%, L3 15%, L2 10%, L1 5%, T_k 15% | Per `PWMRewardERC20.distribute` |

**Mining is NOT infrastructure development** — it's research reproduction. A researcher who reproduces a benchmark can earn PWM regardless of whether they contribute to the protocol code.

Realistic Year 1 mining-pool emission: programmatic; depends on per-event Zeno formula. Director-targeted strong-success scenario: 2.7M PWM/year initial emission, declining via Zeno decay.

**Mining-pool earnings are separate from infrastructure compensation.** Top miners could earn 100K-1M PWM/year if they reproduce many high-value benchmarks. This is a different track than the L1+L2+L3 compensation above.

---

## 7. Gap-Fix Action Items

### 7.1 BEFORE first co-founder onboards

| Item | Why | Time | Cost |
|---|---|---|---|
| **CLA template** | Required for OSS contribution legal clarity | 30 min (use Apache Foundation template) | $0 |
| **Per-co-founder vesting agreement template** | Specifies vesting %, cliff, equity-stake terms | 1-2 hr drafting + legal review | $500-2K legal (or use template) |
| **PWM tax-treatment policy doc** | What new recipients should know about US tax | 1 hr drafting | $0 |

### 7.2 BEFORE bug-bounty program launches (Month 3-6)

| Item | Why | Time | Cost |
|---|---|---|---|
| **Immunefi page setup** | Public bug bounty registration | 30 hr Director time | $1-5K |
| **Bug-bounty tier card publication** | Public commitment to severity-amount mapping | Use §4.4 above; publish on physicsworldmodel.org | $0 |
| **Reserve allocation pre-commit** for bounty payouts | Reserve governance proposal to earmark 100-200K PWM for Year 1 bounty payouts | Multi-sig proposal + 48h timelock | $0 |

### 7.3 BEFORE first salaried hire (Year 2+)

| Item | Why | Time | Cost |
|---|---|---|---|
| **Foundation 501(c)(3) status active** | Required for W-2 employment | NumFOCUS Round 4 → independent foundation | Multi-year |
| **Salary band publication** | Industry-standard transparency | Adapt §4.5 above; publish | $0 |
| **Equity terms template** | Standardized offer | 4-6 hr legal | $1-3K |
| **HR / payroll system** | Standard requirement | Setup at Foundation level | $1-5K/yr ongoing |

### 7.4 BEFORE Track 8 / 9 research grants disbursed

| Item | Why | Time | Cost |
|---|---|---|---|
| **Research grant policy doc** | Defines who can apply, criteria, amounts | 2-3 hr drafting | $0 |
| **Grant agreement template** | Standardizes IP/attribution/Foundation rights | 4-6 hr legal | $1-2K |

---

## 8. Decision Points for Director

Items for explicit decision in next 30-90 days:

| # | Decision | Default | Notes |
|---|---|---|---|
| 1 | Director's target retention of 630K PWM founding-team allocation? | **50-70%** (per §2.4) | Smart contract gives Director 100%; this is the voluntary-share commitment Director will write into co-founder agreements |
| 1b | Per-co-founder voluntary-share target (within §2.4 range)? | Negotiate at recruitment | Published policy band exists; exact number per individual co-founder |
| 2 | Per-co-founder mechanism: Option A (informal), Option B (new vesting), Option C (Reserve grant), or Option D (no share)? | **Option B for >50K PWM; Option C for ≤50K PWM** (§2.3) | Option B requires Director waits for own cliff (Year 1+); Option C usable from deploy day |
| 3 | Bug bounty tier amounts (§4.4): accept defaults or modify? | **Accept defaults** | Industry-standard ratios; adjustable later |
| 4 | Engineering hire structure: USD-only, PWM-only, or hybrid? | **Hybrid** (60-80% USD + 20-40% PWM) (§5.2) | Standard nonprofit-tech model |
| 5 | Bug bounty pre-commit Reserve allocation? | **100-200K PWM** for Year 1 (per §7.2) | Modest; replenishable |
| 6 | Contributor grant rate card (§4.3): accept defaults? | Accept; refine based on Year 1 observed contributions | |
| 7 | Heyang intern PWM bonus: 5K or 10K? | **5-10K** (negotiate with Heyang) | Within published policy |
| 8 | CLA template: Apache Foundation default or custom? | **Apache Foundation default** | $0; battle-tested; widely accepted |
| 9 | Future-hire compensation publication: when to publish? | Year 2 after Foundation lands + first hire negotiated | Sets industry-standard expectation |

---

## 9. Cross-References

- `pwm-team/bounties/INDEX.md` — Master external-bounty list
- `pwm-team/bounties/01-scoring-engine.md` through `08-llm-matcher.md` — Per-bounty specs
- `PWMVesting.sol` — Founding team smart contract (4-yr linear, 1-yr cliff)
- `pwm-team/funds/PWM_FOUNDATION_FORMATION_PLAYBOOK_2026-05-13.md` — Foundation 501(c)(3) trajectory
- `pwm-team/coordination/PRE_DEPLOY_RISK_AUDIT_2026-05-21.md` §6 (legal/regulatory) — Tax & securities concerns
- `pwm-team/coordination/CRISIS_COMMS_PLAYBOOK_2026-05-21.md` §11 — Immunefi setup as pre-staged asset
- `pwm-team/coordination/prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md` — Founder upside math at each tier
- `pwm-team/coordination/PWM_API_VS_WEBSITE_2026-05-20.md` — Engineering hire prioritization
- `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md` — Intern compensation framework
- `pwm-team/numfocus/application/CONTRIBUTORS.md` § Section 5 — Co-founder recruitment template

---

## 10. The Single Most Important Framing

**PWM developer compensation is a portfolio, not a single number.**

- Founding team: long-term equity (PWM vesting) — high upside, multi-year commitment
- External bounties: deliverable-based payment (PWM) — middle-term, project-based
- Foundation grants: flexible (USD + PWM hybrid) — short-term to long-term, role-based
- Mining pool: research-based emission — long-tail, depends on research output

**No single category dominates.** Different developers fit different categories. The structure exists to attract the right talent to each layer with the right incentives.

For a grant reviewer or co-founder candidate evaluating PWM: **the total developer-allocated supply is 16% (3% founding team + 10% Reserve + 3% from Foundation discretionary grants / hires over 4 years).** This is competitive with crypto-tech startup founder + employee equity (typically 10-20% combined) but with public-goods alignment (Foundation governance, not VC return).

---

## 11. Honest Caveat: Most of This Isn't Locked In Yet

Several items in this doc are PROPOSALS, not policy:

| Status | Items |
|---|---|
| ✅ LOCKED ON-CHAIN | 0.63M founding team allocation; 4-year vesting; 1-year cliff; 21M cap; bounty pool from Reserve; **PWMVesting beneficiary = Director's Ledger #1 (IMMUTABLE)** |
| ✅ PUBLIC POLICY | 8 external bounties (specs published); + Bounty 9 MCP server + Bounty 10 Mobile UX (this doc + §3) |
| 🟡 PROPOSED (this doc) | Voluntary per-co-founder share within 50-70% Director retention band; Option B vs C mechanism per co-founder; bug-bounty tier card; engineering hire salary bands; contributor grant rate card; tax/legal policy |
| ❌ DEPENDENT ON | Foundation status (NumFOCUS Round 4 → 501(c)(3)) for salaried hires; LP seeding for PWM-redeemability; grant funding for USD salary baseline; **written co-founder agreement (CLA) for any voluntary redistribution to be enforceable** |

**Until Director's explicit sign-off + Foundation activation, this doc is the strawman — not the legal commitment.** That said, it represents reasonable standard practice + reflects PWM's published constraints. Future hires should expect terms in this range; exceptions require Foundation board approval.

---

*This doc is the canonical reference for PWM developer compensation. Update §2.2-2.5 when co-founder negotiations begin. Update §4.4 if bug bounty amounts shift post-deploy. Update §4.5 when first salaried hire is offered. Update §6 when Mining pool first emissions occur. Keep cross-references current.*
