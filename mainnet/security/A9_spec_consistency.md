# A9 Spec vs Code Consistency — pwm_overview1.md vs deployed contracts

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A9 — in-session)
**Commit reviewed:** post-`41efadb8` (sub-GPU HIGH fix); pre-merge `c688f5d3`
**Spec source of truth:** `pwm-team/long-term-vision/pwm_overview1.md` (3,065 lines)
**CLAUDE.md** at `pwm-team/infrastructure/agent-contracts/CLAUDE.md` (treated as second source of truth)

---

## Summary

| Category | Count | Action |
|---|---|---|
| MATCH (core protocol invariants) | 11 | ✅ no action |
| **NEW (introduced by patches, not in spec)** | **8** | ⚠️ spec amendment required |
| DRIFT (operational / off-chain) | 2 | document |
| DEFERRED (in spec but not implemented) | 2 | acknowledge |
| TOTAL | 23 | — |

**A9 verdict:** Core protocol invariants (token supply, splits, staking, challenge period, multisig) all match the spec. The 8 NEW state variables introduced by the soft-launch patches (mintingPaused, transfersPaused, approvedSubmitter, submissionPermissionless, maxTotalStakeWei, maxBenchmarkPoolWei, thresholdReachedAt × 3, ExecProposal/executeCall flow) are **not documented in pwm_overview1.md**. This is acceptable IF Director publishes a soft-launch addendum or amendment to the spec before mainnet deploy. The deploy itself is NOT blocked by this drift.

---

## 1. Core invariants — MATCH (11)

### 1.1 Total token supply: 21M PWM

| Spec | Code | Match? |
|---|---|---|
| `pwm_overview1.md:1519`: "fixed total supply of 21M PWM" | `deploy/erc20.js:50`: `const TOTAL = ethers.parseEther("21000000")` | ✅ |

### 1.2 Pool allocation

| Spec | Code | Match? |
|---|---|---|
| Minting 82% / 17.22M | `MINTING_POOL = 17_220_000` | ✅ |
| Reserve 10% / 2.1M | `RESERVE = 2_100_000` | ✅ |
| Liquidity 5% / 1.05M | `LIQUIDITY = 1_050_000` | ✅ |
| Founding team 3% / 630K | `FOUNDING_TEAM = 630_000` | ✅ |
| Sum: 21M | `MINTING_POOL + RESERVE + LIQUIDITY + FOUNDING_TEAM === TOTAL` enforced in deploy/erc20.js:55 | ✅ |

### 1.3 Staking amounts (governance-tunable floors)

| Spec | Code | Match? |
|---|---|---|
| L1 Principle: 10 PWM | `stakeAmount[LAYER_PRINCIPLE] = 10 ether` | ✅ |
| L2 Spec: 2 PWM | `stakeAmount[LAYER_SPEC] = 2 ether` | ✅ |
| L3 Benchmark: 1 PWM | `stakeAmount[LAYER_BENCHMARK] = 1 ether` | ✅ |
| Governance-adjustable via `setStakeAmount` | `setStakeAmount(uint8 layer, uint256 amount) external onlyGovernance` | ✅ |

### 1.4 Reward distribution splits

| Spec (basis points × 100) | Code | Match? |
|---|---|---|
| AC + CP combined: 55% | `SPLIT_AC_CP = 5_500` | ✅ |
| L3: 15% | `SPLIT_L3 = 1_500` | ✅ |
| L2: 10% | `SPLIT_L2 = 1_000` | ✅ |
| L1: 5% | `SPLIT_L1 = 500` | ✅ |
| Treasury T_k: 15% | `SPLIT_TREASURY = 1_500` (implicit: drawAmt − Σ(splits) absorbs dust) | ✅ |
| AC/CP split via shareRatioP (`p`) | `acAmt = drawAmt * p * 5500 / BPS_DENOM^2; cpAmt = drawAmt * (BPS_DENOM-p) * 5500 / BPS_DENOM^2` | ✅ |

### 1.5 Ranked draw schedule

| Spec | Code | Match? |
|---|---|---|
| Rank 1: 40% | `if (rank == 1) return 4_000` | ✅ |
| Rank 2: 5% | `if (rank == 2) return 500` | ✅ |
| Rank 3: 2% | `if (rank == 3) return 200` | ✅ |
| Rank 4-10: 1% each | `if (rank >= 4 && rank <= MAX_RANK) return 100; MAX_RANK = 10` | ✅ |
| ~52% rolls over | Implicit: rank-rollover is `pool[bench] -= Σ(drawAmts for rank 1..10)` | ✅ |

### 1.6 Challenge windows

| Spec | Code | Match? |
|---|---|---|
| Standard: 7 days | `CHALLENGE_PERIOD_STANDARD = 7 days` | ✅ |
| Extended (delta ≥ 10): 14 days | `CHALLENGE_PERIOD_EXTENDED = 14 days; DELTA_EXTEND_THRESHOLD = 10` | ✅ |

### 1.7 Multisig + timelock

| Spec | Code | Match? |
|---|---|---|
| 3-of-5 founder multisig | `REQUIRED_APPROVALS = 3; NUM_FOUNDERS = 5` | ✅ |
| 48h timelock | `TIME_LOCK = 48 hours` | ✅ |

### 1.8 Founding team vesting

| Spec | Code | Match? |
|---|---|---|
| 12-month cliff | `CLIFF = 365 * 86400` | ✅ |
| 4-year linear | `DURATION = 4 * 365 * 86400` | ✅ |

### 1.9 Treasury bounty cap

| Spec | Code | Match? |
|---|---|---|
| 50% of T_k per bounty | `require(amount * 2 <= balance, "PWMTreasuryERC20: exceeds 50% cap")` | ✅ |

### 1.10 Stake fates

| Spec | Code | Match? |
|---|---|---|
| Graduation: 50% returned to staker, 50% → reward pool | `half = s.amount / 2; other = s.amount - half; safeTransfer(staker, half); reward.seedBPool(b, other)` | ✅ |
| Challenge upheld: 50% burned, 50% to challenger | `safeTransfer(BURN_SINK, half); safeTransfer(challenger, other)` | ✅ |
| Fraud: 100% burned | `safeTransfer(BURN_SINK, s.amount)` | ✅ |

### 1.11 Zeno minting formula

| Spec | Code | Match? |
|---|---|---|
| `A_k = (M_POOL - M_emitted) × w_k / Σ(w_j)` | Line 198 of PWMMintingERC20: `A_k = (rem * wK) / sumW` where `rem = M_POOL - M_emitted` | ✅ |
| `A_{k,j,b} = A_k × w_{k,j,b} / Σ(w_{k,j',b'})` | Line 204: `A_kjb = (A_k * wB) / sumBW` | ✅ |
| `w_k = δ_k × max(activity_k, 1)` | `_principleWeight`: `p.delta * max(p.activity, 1)` | ✅ |
| `w_{k,j,b} = ρ_{j,b} × max(activity_{k,j,b}, ρ_{j,b})` | `_benchmarkWeight`: `b.rho * max(b.activity, b.rho)` | ✅ |

---

## 2. NEW state introduced by patches — NOT IN SPEC (8)

These were added during the 2026-05-18 multi-agent security review and are deploy-day state that pwm_overview1.md does NOT mention. They constitute the "soft-launch posture" — a 30-day operational window before the protocol matches the spec at full liberty.

| # | Variable / function | Contract | Default at deploy | Spec doc gap |
|---|---|---|---|---|
| 1 | `mintingPaused: bool` | PWMMintingERC20 | `true` | Spec says "M_POOL emits A_{k,j,b} per finalize call". Doesn't mention paused state. |
| 2 | `transfersPaused: bool` | PWMTreasuryERC20 | `true` | Spec says "payAdversarialBounty pays out up to 50% of T_k". Doesn't mention paused state. |
| 3 | `approvedSubmitter: mapping(addr=>bool)` | PWMCertificate | `{}` (empty) | Spec says "miners submit L4 certs". Doesn't mention approval gate. |
| 4 | `submissionPermissionless: bool` | PWMCertificate | `false` | Same as above. |
| 5 | `maxTotalStakeWei: uint256` | PWMStakingERC20 | `100 ether` (set in deploy script) | Spec mentions "USD floors" (which this is NOT exactly — it's a PWM TVL cap). |
| 6 | `maxBenchmarkPoolWei: uint256` | PWMRewardERC20 | `100 ether` (set in deploy script) | Spec mentions "B-pool seeding" without mentioning a per-pool cap. |
| 7 | `ExecProposal struct + proposeExec/approveExec/executeExec/cancelExec` | PWMGovernance | n/a (new code path) | Spec says "setParameter(bytes32 key, uint256 value)" — but doesn't describe how PWMGovernance reaches sibling contracts. **This is the A1-v2 CRITICAL fix, now implemented.** |
| 8 | `thresholdReachedAt: uint64` (on ExecProposal struct only; pre-existing flow uses proposedAt) | PWMGovernance | 0 at propose-time; set on live-threshold crossing | Spec says "48h timelock" — doesn't distinguish from-proposedAt vs from-threshold. The new behavior is the secure interpretation. |

**Recommendation:**

Add a `SOFT_LAUNCH_POSTURE_2026-05-18.md` appendix to pwm_overview1.md (or as a sibling doc in `pwm-team/long-term-vision/`) that documents:
  - The soft-launch caps (5 state variables) and their default values
  - The executeCall primitive as the canonical pre-DAO-activation governance flow
  - The lifecycle: deploy with caps → governance proposals via executeCall → eventual activateDAO transitions to contribution-weighted voting (deferred)
  - The 30-day soft-launch window expectations vs. the post-audit unfettered protocol

This is **documentation-only work**, no code changes. Estimated time: ~1 hour to draft.

---

## 3. DRIFT — operational / off-chain (2)

### 3.1 Reserve multisig: spec says 4-of-7, code is environment-dependent

`CLAUDE.md` line 130: "Reserve controlled by 4-of-7 multisig"

`deploy/erc20.js:61`: `const reserveAddr = process.env.PWM_RESERVE_MULTISIG || (isLive ? null : founders[1])`

The protocol contracts do NOT enforce the Reserve being a 4-of-7 multisig. Director sets the Gnosis Safe address via env var and the deploy script trusts it. Operationally, Director must:
  1. Pre-deploy: configure a Gnosis Safe with 4-of-7 founders + advisors
  2. Set `PWM_RESERVE_MULTISIG=<safe addr>` in the deploy env
  3. Verify post-deploy: 2.1M PWM lands at that address

If Director skips step 1 and uses a single EOA, the spec is violated but the contract layer doesn't catch it.

**Recommendation:** Add `post_deploy_verify.js` check that `code.length > 0` at `PWM_RESERVE_MULTISIG` address (proves it's a contract, not EOA). Doesn't prove "4-of-7" but proves "smart contract" — Safe-detection signal.

### 3.2 DAO governance for grants ≥ 50,000 PWM not enforced on-chain

`pwm_overview1.md:1598`: "Spending above 50,000 PWM requires a DAO governance vote (≥2/3 weight, 14-day window). Smaller grants approved by multisig."

The Reserve is held by a Gnosis Safe (per 3.1), not by PWMGovernance. The "≥ 50K PWM requires DAO vote" rule is **operational governance**, not enforced by smart contracts. Directors must follow it manually.

**Recommendation:** Document this as off-chain governance in the operational runbook. No contract change needed.

---

## 4. DEFERRED — in spec but not implemented (2)

### 4.1 Contribution-weighted DAO voting

`CLAUDE.md` line 137: "voting_weight = w1×(Reserve grants) + w2×(upstream royalties) + w3×(best Q_p) + w4×√(PWM held)"

`PWMGovernance.activateDAO(uint256 id)` sets `daoActivated = true` and disables the multisig path forever. But the DAO voting implementation itself is NOT included in the contract — there's no `vote(uint256 proposalId)` function tied to the contribution-weighted formula.

This is explicitly DEFERRED per `CLAUDE.md` line 5: "the DAO voting implementation itself is deferred (post-M3 per roadmap)."

**Recommendation:** Acceptable. Document in deploy notes that mainnet ships with `activateDAO` as a one-way switch, but voting must be deployed via a v2 governance contract (or via a fresh deploy of a DAO contract) before `activateDAO` is called. NEVER call `activateDAO` until the DAO voting implementation is live, or governance becomes unreachable.

### 4.2 Reserve adversarial bounty math

`CLAUDE.md` line 121: "M4 adversarial reward: max_i(deltaQ_i) × T_k_balance (cap 50% of T_k)"

The math `max_i(deltaQ_i) × T_k_balance` is computed off-chain (governance proposes the absolute amount via `proposeExec(target=treasury, data=payAdversarialBounty(...).encode())`). The contract only enforces the 50% cap.

**Recommendation:** Acceptable — the on-chain check (50% cap) bounds risk; the actual payout amount is a governance decision. Document that "the formula is operational guidance for governance proposals; the contract only enforces the 50% ceiling."

---

## 5. Cross-validation with prior agents

A9's findings are consistent with prior agents:

| Spec gap | Prior agent verdict |
|---|---|
| Soft-launch caps not in spec | A2-v2 INFO ("combined kill-switch surface gives 3-of-5 governance full protocol-brick power") |
| executeCall primitive new | A1-v2 CRITICAL → A1-v3 verified fix |
| thresholdReachedAt new | A1-v3 HIGH (now fixed by sub-GPU) |
| Reserve operational, not on-chain | A2-v2 + my A8 noted operational dependency |
| DAO voting deferred | Acknowledged in CLAUDE.md |

No new contradictions discovered.

---

## 6. Sub-spec ("CLAUDE.md") consistency check

CLAUDE.md is the agent-facing spec subset. Cross-references:

| CLAUDE.md statement | Code reality | Match? |
|---|---|---|
| Line 99: "Three-tier fixed-PWM staking: L1 10 / L2 2 / L3 1" | matches | ✅ |
| Line 100-101: "L4 epoch emission via `epochEmit()` external" | **CODE HAS NO `epochEmit()`** — emission is per-cert via `mintFor`, not per-epoch | ⚠️ DRIFT |
| Line 104-107: "w_k = δ_k × max(activity_k, 1); A_k formula" | matches | ✅ |
| Line 121: "PWMTreasury.receive15pct" | matches | ✅ |
| Line 137: "voting_weight formula" | deferred (4.1 above) | ⚠️ deferred |

### Notable drift: `epochEmit()` does not exist

CLAUDE.md describes minting as `epochEmit()` called once per epoch (daily UTC midnight). The actual implementation is `mintFor(principleId, benchmarkHash)` called from `PWMCertificate.finalize()` — per-event, not per-epoch.

This is a **DOCUMENTATION drift in CLAUDE.md**, not a code bug. The per-event model is the right design (event-driven, not time-cron-driven). CLAUDE.md needs updating to reflect this.

**Recommendation:** Update CLAUDE.md to describe `mintFor(principleId, benchmarkHash)` called by PWMCertificate during finalize, removing the `epochEmit()` mention. Trivial doc-only edit.

---

## 7. What I did NOT check (out of A9 scope)

- The 6-tuple Spec definition formal precision (vs. natural-language code comments)
- The `epsilon_fn` distance formula (off-chain math, no contract enforcement)
- The "rank_p, rank_q" terminology in the spec vs. `rank` (single integer) in code
- Off-chain L_DAG complexity scoring (not contract code)
- The 500-Principle genesis registration script (`register_batch.py`) — out of contract scope
- Bench listing / discovery UX (web/explorer concerns)

---

## 8. Recommendation summary

| Severity | Action | Owner | Time |
|---|---|---|---|
| INFO | Draft `SOFT_LAUNCH_POSTURE_2026-05-18.md` documenting the 8 new state variables | Director or future doc-update agent | ~1 hr (docs only) |
| INFO | Update CLAUDE.md: replace `epochEmit()` with `mintFor()` description | Director | ~10 min |
| LOW | Add Gnosis Safe code-length check to `post_deploy_verify.js` for Reserve address | Could bundle with A8 verifier patches | ~5 min |
| LOW | Document "operational off-chain governance for grants ≥ 50K PWM" in runbook | Director | ~10 min |

**None block deploy.** All can be addressed pre-mainnet or in the launch announcement.
