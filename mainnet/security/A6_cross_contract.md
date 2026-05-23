# A6 Cross-Contract Review

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A6 — cross-contract interaction safety)
**Commit reviewed:** `c688f5d3` on `main`
**Scope:** All 9 protocol contracts in deployment order from `deploy/erc20.js`, focused on cross-contract invariants — not per-contract review (A1/A2/A3 already covered each contract twice). Specific focus on the three soft-launch security patches:

  - `fe3ba529` — `executeCall` primitive on `PWMGovernance` (A1-v2 CRITICAL fix)
  - `41efadb8` — `thresholdReachedAt` on `ExecProposal` (A1-v3 HIGH fix)
  - `203df847` — 5 soft-launch flag patches (`mintingPaused`, `transfersPaused`, `approvedSubmitter`+`submissionPermissionless`, `maxTotalStakeWei`, `maxBenchmarkPoolWei`)

---

## Summary

| Severity | Count | Blocks deploy? |
|---|---|---|
| CRITICAL | 0 | NO |
| HIGH | 1 | **NO** (cross-validates A8 HIGH from a cross-contract perspective) |
| MEDIUM | 4 | NO (operational hazards, not security bugs) |
| LOW | 2 | NO |
| INFO | 2 | NO |

**Headline:** The protocol works coherently end-to-end under the soft-launch posture, BUT three cross-contract correctness hazards deserve attention before unpause:

  1. **HIGH (cross-validates A8):** `PWMCertificate.submit()` depends on `registry.exists(benchmarkHash)` which can only be populated by the registry owner. With deploy/erc20.js not transferring registry ownership, governance cannot register benchmarks post-handoff — meaning the entire L4 lifecycle is gated on the deployer EOA.
  2. **MED (new finding):** `PWMCertificate.finalize()` unconditionally calls `minting.mintFor()` when `address(minting) != address(0)`. With `setMinting(certAddr)` called at deploy AND `mintingPaused=true`, finalize() reverts hard. The certificate-source-of-truth comment ("optional: if unset, finalize skips mintFor") is misleading because deploy ALWAYS sets minting. This is a soft-launch landmine — currently masked by the fact that no approved submitters exist (no certs can be submitted), but is the first thing to break when governance starts approving submitters before un-pausing minting.
  3. **MED (new finding):** A graduating stake calls `reward.seedBPool` which is itself capped by `maxBenchmarkPoolWei`. A bench pool at 99/100 PWM rejects a 1-PWM seed and **reverts the entire `graduate()` call**, leaving the stake locked in Active status indefinitely. Recoverable via governance lifting the cap, but operationally surprising.

Beyond these three, the executeCall primitive's cross-contract attack surface is contained by the founder-only check on PWMGovernance reentrancy. Soft-launch caps (5 of them) interact correctly with the lifecycle as long as the **un-pause sequence is followed in the right order** (this is operational, not contractual).

---

## End-to-end L4 lifecycle walkthrough

Tracing ONE artifact (benchmark `B`, principle `K`) through the protocol from registration → staking → submission → finalization → reward distribution → treasury accrual. At each contract boundary I checked: caller authorization correct, trust assumptions justified, state invariants preserved, re-entrant paths blocked.

### Step 1: Benchmark registration in PWMRegistry

  - Caller: `registry.register(hash, parentHash, layer=3, creator)` — `onlyOwner`.
  - **Soft-launch reality:** Owner = deployer EOA, NOT governance. See HIGH-1 below. PWMGovernance has no path to call `register` after handoff because PWMRegistry was never transferred via `transferOwnership(govAddr)`.
  - Trust: PWMRegistry trusts only Ownable's owner. No back-references to other PWM contracts. Clean.

### Step 2: Staking against benchmark `B` in PWMStakingERC20

  - Caller: anyone calls `stake(layer=3, B)`.
  - Permission check: `pwmToken.safeTransferFrom(msg.sender, this, 1 ether)` — caller must hold ≥ 1 PWM and approve.
  - Cap check: `totalActiveStakeWei + 1 ether ≤ maxTotalStakeWei (100 PWM)`. ✓
  - Trust: PWMStakingERC20 trusts only its `governance` (set by deploy) for setX and slash flows. It does NOT need to know whether `B` is registered in PWMRegistry. **Note:** This is by design — staking can happen against unregistered benchmarks. The L4 cert lifecycle catches it later (`PWMCertificate.submit` enforces `registry.exists`).

### Step 3: L4 cert submission in PWMCertificate

  - Caller: address `S` calls `submit(certHash, B, ...)`.
  - Permission: `onlySubmitter` modifier — soft-launch requires `approvedSubmitter[S] == true` (`submissionPermissionless = false`). ✓
  - Registry check: `registry.exists(B)` must be true. ✓ → couples to Step 1.
  - State: `certificates[certHash].status = Pending`, `submittedAt = block.timestamp`.
  - Trust: PWMCertificate trusts `governance` for setX + `resolveChallenge`. Trusts `registry`, `reward`, `minting` as wired addresses (data sources, not callers). Clean.

### Step 4: Challenge window

  - Anyone calls `challenge(certHash, proof)` within 7 days (or 14 if delta ≥ 10). Status → Challenged.
  - `resolveChallenge(certHash, upheld)` is `onlyGovernance`. Returns to Pending or moves to Rejected.
  - Trust: no cross-contract calls; clean.

### Step 5: Finalize → mintFor → distribute → treasury

  - Caller: anyone calls `finalize(certHash)` after window closes.
  - Status check: must be `Pending`.
  - **CROSS-CONTRACT CALL 1:** `minting.mintFor(K, B)` — if `address(minting) != address(0)`.
    - Caller auth: PWMMintingERC20 has `onlyCertificate` modifier; msg.sender = PWMCertificate. ✓
    - Soft-launch failure path: `mintingPaused == true` → revert "PWMMintingERC20: minting paused" → entire `finalize()` reverts. See MED-1.
    - Non-promoted failure path: if `K.promoted == false`, mintFor reverts → finalize reverts.
    - On success: pulls `A_kjb` PWM via approve+depositMinting into PWMRewardERC20.
  - **CROSS-CONTRACT CALL 2 (within Call 1):** `reward.depositMinting(B, A_kjb)`.
    - Caller auth: PWMRewardERC20 has `msg.sender == minting` check. ✓
    - Cap check: `pool[B] + A_kjb ≤ maxBenchmarkPoolWei`. **If exceeded, reverts back up through finalize.** See MED-3.
    - Pool credited. Returns.
  - **CROSS-CONTRACT CALL 3:** `reward.distribute(certHash, Draw)`.
    - Caller auth: `onlyCertificate`. ✓
    - Splits pool by rank schedule (rank 1 → 40%; rank 11+ → 0%; rollover).
    - **CROSS-CONTRACT CALL 4:** `treasury.receive15pct(K, tkAmt)`.
      - Caller auth: `onlyReward`. ✓
      - Uses `safeTransferFrom(msg.sender, ...)` after `forceApprove(treasury, tkAmt)` in PWMRewardERC20 → atomic transfer. ✓
      - Soft-launch note: `treasury.transfersPaused` does NOT gate `receive15pct` (only `payAdversarialBounty`). So inflows work; only payouts blocked. ✓
    - Payouts to AC/CP/L1/L2/L3 via `safeTransfer`. No reentrancy risk because PWM is standard non-callback OZ ERC20.

### Step 6: M4 bounty payout (separate flow, not L4)

  - Caller: governance via executeCall → `treasury.payAdversarialBounty(K, winner, amount)`.
  - Soft-launch check: `!transfersPaused`. ✓ (blocks during soft-launch).

### Boundary issues found in lifecycle

| # | Boundary | Issue | Severity |
|---|---|---|---|
| BI-1 | Registry → Certificate | Registry owner is deployer EOA, not governance. Governance can never register benchmarks via 3-of-5+timelock. | HIGH (= A8 HIGH from cross-contract view) |
| BI-2 | Certificate → Minting | `finalize` unconditionally invokes `mintFor` when minting wired, but `mintFor` aborts on `mintingPaused=true` OR `!promoted`. Finalize is not graceful. | MEDIUM |
| BI-3 | Staking → Reward | `staking.graduate()` calls `reward.seedBPool` which is cap-gated; cap exceeded reverts the whole graduate(). | MEDIUM |
| BI-4 | Reward → Treasury | Cross-contract approve+transferFrom uses forceApprove (OZ helper); SafeERC20 prevents stuck-approval on non-zero balance. Pattern is clean. | OK |

---

## Trust web verification

Each contract's `onlyX` modifiers and external wires:

### PWMToken
  - Trusts: nobody (no onlyX modifiers reachable post-genesis-mint).
  - Trusted by: all 4 ERC20-sibling protocol contracts (Staking, Reward, Treasury, Minting) reference it as `immutable IERC20`. ✓
  - Note: PWMToken is `Ownable`, but the Ownable owner has no exploitable powers (cap fully minted at genesis, no further mint). A1 INFO-1 + A8 LOW-1 recommend renounce. ACK.

### PWMGovernance
  - Trusts: only its founders set (5 immutable on construction; rotatable via FounderChange 3-of-5+48h).
  - Trusted by: 5 sibling contracts (Reward, Staking, Certificate, Minting, Treasury) as `governance` address. **Not** trusted by Registry (which is Ownable-deployer-owned).
  - **executeCall trust:** governance trusts itself to gate all sibling setX via the 3-of-5+48h flow. ✓
  - Re-entrancy: `executeExec` sets `p.executed = true` BEFORE `p.target.call(p.data)` → same-id re-execution blocked.
  - **Cross-contract reentrancy attempt:** target.call could invoke `PWMGovernance.proposeExec/approveExec/etc.` But all of these are `onlyFounder`, and `msg.sender` during the re-entered call would be the **target contract**, not a founder. Sibling contracts (Reward, Staking, Cert, Minting, Treasury) are never registered as founders, so the re-entrancy attempt fails at the `onlyFounder` check. ✓

### PWMRegistry
  - Trusts: Ownable.owner only.
  - Trusted by: PWMCertificate references it as a data source (`registry.exists`). ✓
  - **CRITICAL DRIFT:** Owner stays as deployer EOA post-deploy. Governance is NOT in the trust web. See A8 HIGH and HIGH-1 here.

### PWMTreasuryERC20
  - Trusts: `onlyGovernance` (setX, payAdversarialBounty, setTransfersPaused) + `onlyReward` (receive15pct).
  - Trusted by: PWMRewardERC20 calls `treasury.receive15pct` after forceApprove. ✓
  - All trust wires correctly point to PWMGovernance and PWMRewardERC20.
  - **Trust expansion check:** Does TreasuryERC20 trust ANYTHING else? grep shows no other modifiers. ✓

### PWMRewardERC20
  - Trusts: `onlyGovernance` (setX + depositBounty + setMaxBenchmarkPoolWei) + `onlyCertificate` (distribute) + `msg.sender == staking` (seedBPool) + `msg.sender == minting` (depositMinting).
  - Trusted by: PWMCertificate (calls distribute), PWMStakingERC20 (calls seedBPool), PWMMintingERC20 (calls depositMinting), PWMTreasuryERC20 (receives via forceApprove → safeTransferFrom flow). ✓
  - **All 4 trust paths (`onlyCertificate`, `onlyStaking-equivalent`, `onlyMinting-equivalent`, `onlyGovernance`) are wired correctly** by deploy/erc20.js lines 164-174.
  - Lockout risk: After handoff, only governance can re-wire. If any setX was missed, it's permanently locked. **Check deploy/erc20.js:** `setCertificate`, `setStaking`, `setMinting`, `setTreasury` all present. ✓

### PWMStakingERC20
  - Trusts: `onlyGovernance` (setX, slash, graduate).
  - Trusted by: nobody (it pulls from caller, pushes to reward).
  - No cross-trust concerns.

### PWMCertificate
  - Trusts: `onlyGovernance` (setX, setApprovedSubmitter, setSubmissionPermissionless, resolveChallenge) + `onlySubmitter` (submit).
  - Trusted by: PWMRewardERC20 (via onlyCertificate) and PWMMintingERC20 (via onlyCertificate).
  - **External wires (data sources):** registry, reward, minting. All set by deploy. ✓
  - Trust expansion: a malicious governance executeCall could `setReward(maliciousAddr)` and redirect distribute() into the attacker's contract. But that requires 3-of-5+48h. Out of scope.

### PWMMintingERC20
  - Trusts: `onlyGovernance` (setX + setDelta/setPromotion/setBenchmark/etc + setMintingPaused) + `onlyCertificate` (mintFor).
  - Trusted by: PWMRewardERC20 (`msg.sender == minting` in depositMinting).
  - All wires set by deploy. ✓

### PWMVesting
  - Trusts: **NOBODY** (no governance, no admin, no setX functions at all).
  - All schedule + beneficiary parameters are `immutable` (set in constructor).
  - **Cross-contract attack surface: ZERO.** PWMVesting is isolated by design — A malicious governance executeCall cannot redirect or accelerate the vesting because PWMVesting has no externally-callable mutators except `release()` (which only the beneficiary can usefully call, and only releases vested amounts). ✓
  - One indirect attack: a malicious governance executeCall could call `PWMToken.transfer(...)` — but governance contract holds NO PWM tokens (the 21M went to: PWMMintingERC20 [82%], Reserve EOA [10%], Liquidity EOA [5%], PWMVesting [3%]). Governance has no PWM to move. Vesting safe.

### Circular trust check

Looking for `A trusts B AND B trusts A`:

  - PWMRewardERC20 trusts PWMCertificate (onlyCertificate) AND PWMCertificate trusts PWMRewardERC20 (calls `reward.distribute`). Is this circular?
    - PWMCertificate doesn't *trust* PWMRewardERC20 — it calls it as a consumer. PWMRewardERC20 trusts PWMCertificate to be authorized to call distribute. One-way trust.
  - PWMRewardERC20 trusts PWMTreasuryERC20 (calls `treasury.receive15pct`) AND PWMTreasuryERC20 trusts PWMRewardERC20 (`onlyReward`). Same pattern: caller is gated; callee is used. One-way trust. ✓
  - PWMMintingERC20 trusts PWMCertificate (onlyCertificate) AND PWMCertificate uses PWMMintingERC20. One-way. ✓

**No circular trust relationships.** ✓

---

## executeCall primitive cross-contract attack surface

`PWMGovernance.executeExec(id)` can call any function on any of the 8 other contracts (target != address(this) is blocked). I walked through the following attack scenarios:

### Scenario E1: 3-colluding-founders propose to break supply cap

  - `proposeExec(target=PWMToken, data=mint(...).calldata)`?
    - PWMToken inherits `ERC20Capped`. Even if PWMToken Ownable owner = governance (it's NOT — owner is deployer EOA), `_mint(amount)` would revert via `ERC20Capped._update` because TOTAL_SUPPLY is already minted at genesis. **Supply cap unbreakable.** ✓

### Scenario E2: 3-colluding-founders propose to drain a benchmark pool

  - `proposeExec(target=PWMRewardERC20, data=...)`? PWMRewardERC20 has no `onlyGovernance` payout function. All outflows are gated by `onlyCertificate`. **Drain via executeCall requires also re-wiring `certificate` to a malicious cert contract via `setCertificate(maliciousCert)`**. After that, 3 colluders own the reward pool. Mitigation: 48h timelock gives non-colluding founders veto window. **Risk class: 3-of-5 collusion fundamental risk; not a bug.**

### Scenario E3: 3-colluding-founders propose to brick a pause flag

  - `proposeExec(target=PWMMintingERC20, data=setMintingPaused(false).calldata)`?
    - Legitimate use case (governance unpauses minting after audit).
    - After execution, mintingPaused = false, mintFor works. ✓ Expected behavior, not exploit.

### Scenario E4: executeCall via sibling that re-enters governance

  - `proposeExec(target=PWMCertificate, data=setGovernance(0xdeadbeef).calldata)`? After 48h timelock execution, PWMCertificate.governance = 0xdeadbeef.
    - But 0xdeadbeef cannot make PWMCertificate call back into PWMGovernance — PWMCertificate has no function that calls PWMGovernance.proposeExec/approveExec.
    - Even if it did, msg.sender during the re-entered call would be PWMCertificate, which is not a founder. `onlyFounder` check rejects. ✓

### Scenario E5: Reentrancy during the external call in executeExec

  - executeExec sets `p.executed = true` BEFORE `p.target.call(p.data)`. Re-entrant call cannot re-execute the same id.
  - Re-entrant call to `proposeExec`/`approveExec`/`cancelExec` from msg.sender = target contract fails at `onlyFounder`. ✓
  - **Edge case:** what if target IS a founder address? An EOA cannot call back into a contract during a tx — only contracts can. So this only matters if `target` is a contract that happens to also be in `isFounder`. The constructor enforces 5 non-zero non-duplicate founders, but doesn't enforce they're EOAs. If a founder slot is a contract (e.g., a Gnosis Safe), then `executeExec(target=thatSafe, data=...)` could in principle have the safe re-enter PWMGovernance. The safe would still need to pass `onlyFounder` (it does — it's a founder), but the multisig safe could not unilaterally vote (the safe's owners would need to trigger it). **This is the same trust as the multisig itself — not new attack surface.**
  - **One real concern:** if `target` is a founder Gnosis Safe AND the Safe's owners include the same 3 colluders, then the colluders could chain `proposeExec → approveExec → executeExec` from a single Safe re-entry. But that's the colluders re-using their existing approve power, not a new attack. ✓

### Scenario E6: Re-execute via a separate proposal id

  - `proposeExec` with identical (target, data) at id=1, id=2, id=3 — each requires its own 3-of-5+48h. Replay-by-design. ✓

### Scenario E7: A1-v3 HIGH fix coverage gap — proposedAt-timelock on Proposal and FounderChange

  - `executeProposal` still uses `proposedAt + TIME_LOCK` (line 150). With proposed-at start, 3 colluders can approve in the last second of 48h, leaving non-colluders < 1 block to veto.
    - Exploitability via cross-contract: `Proposal` only modifies `parameters[key]`. No sibling contract reads from `PWMGovernance.parameters` directly. Verified by grep: no `governance.getParameter(` calls in any other contract. So the rapid-approve attack on `Proposal` doesn't change any cross-contract state. **Not exploitable cross-contract.**
  - `executeFounderChange` also uses `proposedAt + TIME_LOCK` (line 247). Same vulnerability shape: 3 colluders can rapid-approve in the last second of 48h.
    - Cross-contract impact: founder change replaces a slot's address. This directly changes who can approve future proposals. **The dissent window collapses for any subsequent proposal that requires the displaced founder's veto.**
    - **Concrete attack:** Colluders C1+C2+C3 propose to replace honest founder H with malicious M at hour 0. At hour 47:59, they execute the founder change. Now C1+C2+C3+M is a 4-of-5 — they can rapid-execute anything afterward. Honest founder H has effectively 1 block of veto window once the proposal-to-replace-them is approved-and-timelock-elapsed.
    - **Soft-launch mitigation:** This requires 3 colluding founders. The soft-launch caps limit downstream blast radius to 100 PWM per pool and paused minting/transfers. So even if colluders pull off the founder rotation attack, the protocol value at risk is bounded.
    - **Recommendation:** Apply the `thresholdReachedAt` fix to `FounderChange` and (for spec parity) `Proposal` as well, before unpausing soft-launch caps. See MED-2 below.

---

## Soft-launch invariant verification

Reading each soft-launch flag against the cross-contract flow that uses it:

### Inv-1: `mintingPaused = true` blocks all PWM emission

  - PWMMintingERC20.mintFor() line 197: `require(!mintingPaused, ...)`.
  - mintFor is `onlyCertificate`. The only caller is PWMCertificate.finalize() line 221.
  - **Confirmed: mintingPaused=true → mintFor reverts → finalize() REVERTS** (not graceful skip).
  - Cross-contract effect: with mintingPaused=true AND certificate.minting wired, **finalize is dead for ALL certs during soft-launch**. The contract comment "optional: if unset, finalize skips mintFor" is misleading — once setMinting is called (which deploy DOES), the optionality is gone.
  - **Why this doesn't break the soft-launch immediately:** No certs can be submitted because no `approvedSubmitter` exists. So finalize is moot.
  - **Why this is a landmine:** When governance later does `setApprovedSubmitter(X, true)` to start accepting certs, finalize will silently fail until **either** mintingPaused is set false **or** all submitted certs happen to be against non-promoted principles (which also reverts in mintFor on `require(p.promoted)`). The un-pause sequencing matters. See MED-1.

### Inv-2: `transfersPaused = true` blocks treasury bounty payouts

  - PWMTreasuryERC20.payAdversarialBounty line 81: `require(!transfersPaused, ...)`.
  - Inflows (receive15pct) NOT gated by this flag. ✓ (intentional — soft-launch can still accrue treasury balance).
  - Cross-contract effect: governance cannot pay bounties during soft-launch. Intended. ✓

### Inv-3: `submissionPermissionless = false` + no `approvedSubmitter` blocks all submissions

  - PWMCertificate.onlySubmitter line 87-91: `require(submissionPermissionless || approvedSubmitter[msg.sender])`.
  - Both `submissionPermissionless = false` (default) and `approvedSubmitter` mapping empty (deploy doesn't seed) → ALL submit() calls revert.
  - Cross-contract effect: no certificates → no finalize calls → no PWM emission → no royalty distribution. The entire L4 lifecycle is dormant. **This is the strongest soft-launch invariant.** ✓

### Inv-4: `maxTotalStakeWei = 100 ether` caps total active staked PWM

  - PWMStakingERC20.stake line 107-111: `totalActiveStakeWei + required ≤ maxTotalStakeWei`.
  - Cross-contract effect: bounded TVL. After cap is hit, no new stakes accepted. Existing stakes can graduate (frees cap) or be slashed (frees cap).
  - **Subtle:** the cap is on `totalActiveStakeWei`. `graduate()` decrements it before `safeTransfer(staker, half)` + `seedBPool(other)`. If `seedBPool` reverts (pool cap exceeded), the entire graduate() reverts and `totalActiveStakeWei` is rolled back to pre-decrement. State integrity preserved by EVM atomicity. ✓
  - However: the **stake is permanently stuck in Active** until governance lifts the reward pool cap. See MED-3.

### Inv-5: `maxBenchmarkPoolWei = 100 ether` caps per-benchmark reward pool

  - PWMRewardERC20._credit lines 137-140: `newBalance ≤ maxBenchmarkPoolWei`.
  - Applied to ALL three credit paths: `seedBPool` (from staking graduate), `depositMinting` (from cert finalize), `depositBounty` (from governance).
  - Cross-contract effect: per-benchmark pool grows are capped. Once cap is hit, NO further inflows (seed/mint/bounty) work. This means a bench at 99/100 PWM:
    - Cannot have new stakes graduate against it (revert).
    - Cannot accept finalize() mintFor injections (revert).
    - Cannot accept governance bounty top-ups (revert).
  - This is by design (limit blast radius), but the **interaction with graduate() bricks the staking lifecycle for any benchmark that fills its pool.** See MED-3.

### Inv-6: All 5 caps interact correctly

  - Caps are independent (no shared state, no coupling math). Each one is a `require(newValue ≤ cap)` gate.
  - **No combinatorial issue found** — disabling any one cap (governance setMax(big) or setPaused(false)) doesn't break the others.

---

## PWMRegistry ownership impact (A8 HIGH cross-validation)

A8 found that `deploy/erc20.js` does NOT call `registry.transferOwnership(govAddr)`. I confirm this from the cross-contract perspective and add lifecycle observations:

  - PWMRegistry.register is `onlyOwner` (line 38). After deploy, owner = deployer EOA.
  - PWMCertificate.submit requires `registry.exists(benchmarkHash)` (line 157).
  - PWMMintingERC20 references benchmarks too, but via its OWN registration: `registerBenchmark(principleId, benchmarkHash, rho)` — this is `onlyGovernance`. So minting has its own benchmark registry that IS governance-controlled.
  - **The two registries are NOT synced.** A benchmark can be registered in PWMRegistry (deployer EOA) without being registered in PWMMintingERC20 (governance). Cross-contract effect: a cert submitted against a registry-registered-but-not-minting-registered benchmark will pass submit() but revert in finalize() at `b.registered` check (PWMMintingERC20.mintFor line 201).

  - **Operational impact:** to fully wire up a benchmark, BOTH calls are needed:
    1. `registry.register(B, ...)` — by deployer EOA (not governance!)
    2. `minting.registerBenchmark(K, B, rho)` — by governance via executeCall

  - **A8 HIGH cross-confirmation:** Yes, this is real. The deployer EOA is a single point of failure for the registration flow. If the deployer key is lost/compromised after deploy:
    - Existing benchmarks are immutable (registry is append-only). Existing certs can still finalize for existing benchmarks.
    - But NO new benchmarks can ever be registered. Protocol cannot expand beyond the deploy-time genesis set.
    - Governance has no remediation path. transferOwnership is also onlyOwner (Ownable).

  - **A8's recommended one-line fix is correct and necessary.** Adding `await registry.transferOwnership(govAddr)` to deploy/erc20.js between lines 247 and 248 puts registry under the same 3-of-5+48h gate as the other 5 admin-bearing contracts.

  - **Important sequencing note:** any genesis benchmarks must be registered BEFORE the handoff (while deployer is still owner), OR registered post-handoff via `proposeExec(target=registry, data=register(...))`. The current deploy script registers NO genesis benchmarks — Director or a separate script must handle this. **If the runbook does post-handoff registration via governance proposals, expect ~48h delay per benchmark.**

---

## Findings

### [HIGH] CC-1: PWMRegistry ownership not handed off (cross-validates A8 HIGH)

**Files:**
  - `infrastructure/agent-contracts/deploy/erc20.js:237-249` (missing transferOwnership)
  - `infrastructure/agent-contracts/contracts/PWMRegistry.sol:38` (onlyOwner gate)
  - `infrastructure/agent-contracts/contracts/PWMCertificate.sol:157` (registry.exists dependency)

**Description:**

PWMCertificate.submit() requires `registry.exists(benchmarkHash)`. PWMRegistry.register() is `onlyOwner`. Without the missing `registry.transferOwnership(govAddr)`, the deployer EOA permanently owns the registry. After deploy:
  - PWMGovernance has NO path to register a new benchmark (executeCall to register fails because owner != governance).
  - The L4 lifecycle's first dependency (a registered benchmark) is gated by a single hot key, contradicting the 3-of-5+48h security model.
  - If the deployer key is lost, no new benchmarks can ever be onboarded.

**Impact:**

Operationally severe: single-key bottleneck for protocol expansion. Bricks governance's ability to manage protocol growth post-handoff.

**Recommendation:**

Apply A8's recommended patch — add `await (await registry.transferOwnership(govAddr)).wait();` to `deploy/erc20.js` between lines 247 and 248. Confirm runbook order: any genesis-batch benchmarks must be registered before this line (while deployer is still owner), OR a governance proposal must register them post-handoff.

**Soft-launch cap mitigation:** N/A — orthogonal to the 5 caps.

---

### [MEDIUM] CC-2: finalize() reverts hard under mintingPaused or non-promoted principle

**Files:**
  - `infrastructure/agent-contracts/contracts/PWMCertificate.sol:209-222` (finalize function)
  - `infrastructure/agent-contracts/contracts/PWMMintingERC20.sol:192-202` (mintFor preconditions)

**Description:**

PWMCertificate.finalize() conditionally calls `minting.mintFor()` based on `address(minting) != address(0)`. The contract comment on line 41 says "optional: if unset, finalize skips mintFor" — implying graceful behavior. But:
  1. deploy/erc20.js line 172 always calls `certificate.setMinting(mintingAddr)`, so the address is ALWAYS set in production.
  2. PWMMintingERC20.mintFor reverts in three places: `mintingPaused`, `!p.promoted`, `!b.registered`. Any of these reverts the entire finalize() call.

During soft-launch with `mintingPaused=true`, **every finalize() call will revert** if any cert reaches that stage.

**Currently masked because:** soft-launch has no approvedSubmitter, so no certs are submitted, so finalize is never called.

**Why this is a landmine:** The moment governance does `setApprovedSubmitter(X, true)` to allow first cert submissions (a single executeCall), users can submit certs but **cannot finalize them**. To make finalize work, governance must EITHER:
  - Unpause minting first (`setMintingPaused(false)`), AND register the principle in PWMMintingERC20 (`setPromotion(K, true)` + `setDelta(K, ...)` + `registerBenchmark(K, B, rho)`).
  - OR unwire minting from certificate (`certificate.setMinting(address(0))` — but `setMinting` requires `x != address(0)`, so this is impossible without contract surgery).

Because there is no way to set `certificate.minting` back to zero, **the protocol is locked into "mint or fail"** once setMinting is called the first time.

**Impact:**

Operational hazard: a sequencing error during un-pause can produce certs stuck at Pending → window-closed → still cannot finalize. Recoverable by completing the un-pause sequence, but confusing.

**Recommendation:**

Either:
  - (a) **Allow `certificate.setMinting(address(0))`** — relax the zero-check so governance can disable mint coupling for soft-launch / future audit pauses. Trivial change in PWMCertificate.sol:115.
  - (b) **Make finalize gracefully skip mintFor on failure** — wrap the mint call in try/catch and continue to distribute(). This preserves the cert-settlement lifecycle even if minting is paused. Pattern:
    ```solidity
    if (address(minting) != address(0)) {
        try minting.mintFor(c.principleId, c.benchmarkHash) {} catch {}
    }
    ```
  - (c) **Document the un-pause sequencing requirement** in the runbook: "setApprovedSubmitter MUST be preceded by setMintingPaused(false) AND principle promotion."

Path (b) is the safest because it makes the contract behavior match the source-of-truth comment. Path (c) is zero-code but requires operator discipline.

**Soft-launch cap mitigation:** N/A — this is about the un-pause path, not the cap math.

---

### [MEDIUM] CC-3: graduate() bricks when reward pool cap exceeded

**Files:**
  - `infrastructure/agent-contracts/contracts/PWMStakingERC20.sol:127-143` (graduate)
  - `infrastructure/agent-contracts/contracts/PWMRewardERC20.sol:116-120,133-143` (seedBPool + _credit cap check)

**Description:**

`staking.graduate(artifactHash, benchmarkHash)` calls `reward.seedBPool(benchmarkHash, half)` after `forceApprove`. `_credit` enforces `newBalance ≤ maxBenchmarkPoolWei`.

If the target benchmark's pool is at, say, 99 / 100 PWM and the seed would push it over, **the seedBPool call reverts and rolls back the entire graduate() transaction.** The stake's `Status.Active` is preserved (no state mutation), but governance cannot transition this stake to Graduated until **either**:
  - The benchmark's pool drains via distribute() (requires a cert finalize, which is also blocked during soft-launch — see CC-2).
  - Governance lifts `maxBenchmarkPoolWei` via executeCall.
  - Governance changes the benchmark slot — but graduate's benchmarkHash arg is per-call, so governance can simply graduate to a different (uncapped) benchmark on retry.

**Why this matters:** During soft-launch with cap=100 PWM, a single benchmark filled with 50 L3 stakes (50 × 1 PWM each via the stake() flow seeding 0.5 PWM via graduate × 100 grads, actually) — math: 1 PWM stake → 0.5 PWM seed. So 200 graduations into the same benchmark fills the pool. That's plausible on day-30 of soft-launch.

**Impact:**

Operational: graduating stakes can get DoS'd by a full reward pool. Not a security exploit (no value loss); a usability footgun.

**Recommendation:**

Two patterns:

  - (a) Bound the seed at the cap inside `seedBPool` — accept the part that fits, return the rest to the staker. Requires changing `seedBPool` signature.
  - (b) Document: graduate() must check `pool[B] + 0.5*stake ≤ cap` off-chain before calling. Operator burden.
  - (c) Skip the seed if it would exceed cap — `if (pool[B] + amount > cap) { pwmToken.safeTransfer(staker, amount); }` — but this changes the graduation economics (full return vs half-seed).

For soft-launch, (b) is acceptable. For mainnet maturity, (a) is the right shape.

**Soft-launch cap mitigation:** YES — the cap is what causes this issue. Raising the cap removes the brick.

---

### [MEDIUM] CC-4: Proposal + FounderChange timelock-from-proposedAt (A1-v3 HIGH fix incomplete)

**Files:**
  - `infrastructure/agent-contracts/contracts/PWMGovernance.sol:150` (executeProposal)
  - `infrastructure/agent-contracts/contracts/PWMGovernance.sol:247` (executeFounderChange)

**Description:**

The A1-v3 HIGH fix in commit `41efadb8` added `thresholdReachedAt` ONLY to ExecProposal. The same vulnerability shape exists on `Proposal` and `FounderChange`:

  - **Proposal:** timelock starts at proposedAt. 3 colluders can wait 47h59m, then approve, leaving honest founders < 1 block to veto. **Cross-contract impact: low** — Proposal only writes to `parameters[key]`, and no other contract reads from `parameters`. Limited to internal governance state.

  - **FounderChange:** timelock starts at proposedAt. 3 colluders can wait 47h59m, then approve, then execute. **Cross-contract impact: HIGH** — once a founder slot is rotated to a malicious address, the colluders have a 4-of-5 majority and can rapid-approve any future Proposal/ExecProposal/FounderChange without dissent. Compounding attack.

**Why this is MEDIUM not HIGH:**
  - During soft-launch, all 5 caps + pause flags are active. Even if colluders pull off the founder-rotation attack + chained exec attacks, the value at risk is bounded by the caps (100 PWM staking, 100 PWM per pool, paused minting, paused treasury payouts).
  - The attack still requires 3 colluding founders.

**Recommendation:**

Pre-audit, apply the `thresholdReachedAt` pattern to both `Proposal` and `FounderChange`. Same code shape as `approveExec` lines 320-323 → set `thresholdReachedAt = block.timestamp` when approvals first cross REQUIRED_APPROVALS. Same shape as `executeExec` line 342 → require `thresholdReachedAt + TIME_LOCK`.

The fix is mechanical (~20 LoC), well-tested for ExecProposal, and removes the rapid-approve attack on the 2 remaining proposal types.

**Soft-launch cap mitigation:** YES — caps bound the blast radius if exploited.

---

### [MEDIUM] CC-5: No way to set certificate.minting back to zero

**Files:**
  - `infrastructure/agent-contracts/contracts/PWMCertificate.sol:115-119` (setMinting requires non-zero)

**Description:**

`certificate.setMinting(address x)` requires `x != address(0)`. Once deploy/erc20.js calls `certificate.setMinting(mintingAddr)`, there is no path to disable minting integration short of pointing it at a no-op contract (which would still require deploying that no-op).

Cross-contract implication: the comment on PWMCertificate.sol:41 ("optional: if unset, finalize skips mintFor") describes a state that **cannot be re-entered** post-deploy. The optionality only exists pre-setMinting.

**Impact:**

Inflexibility. Combined with CC-2 (finalize reverts on mintingPaused), this means governance has no quick lever to make finalize work during a minting-emergency pause.

**Recommendation:**

Relax the zero-check in `setMinting`. Same pattern is needed for `setReward` (line 110) — though reward unsetting would break distribute() entirely, which is more dangerous. Apply only to setMinting; leave setReward strict.

Alternative: add an explicit `unsetMinting()` function (onlyGovernance) that sets `minting = address(0)`. More explicit.

**Soft-launch cap mitigation:** N/A.

---

### [LOW] CC-6: Governance has no PWM balance for depositBounty flow

**Files:**
  - `infrastructure/agent-contracts/contracts/PWMRewardERC20.sol:128-131` (depositBounty)

**Description:**

`depositBounty(benchmark, amount)` is `onlyGovernance` and does `safeTransferFrom(msg.sender = governance, ...)`. For this to work:
  1. PWMGovernance contract must hold ≥ amount PWM.
  2. PWMGovernance must have approved PWMRewardERC20 for ≥ amount.

But (per deploy/erc20.js lines 213-225) the entire 21M PWM supply goes to: PWMMintingERC20, Reserve EOA, Liquidity EOA, PWMVesting, PWMFaucet. Governance gets ZERO PWM. There is no executeCall pattern that gives governance PWM (it would require Reserve EOA to transfer in).

So **depositBounty is operationally a 3-step flow:**
  1. Reserve EOA transfers PWM to PWMGovernance contract.
  2. Governance executeCall → PWMToken.approve(PWMRewardERC20, amount).
  3. Governance executeCall → PWMRewardERC20.depositBounty(benchmark, amount).

Each step 2 and 3 needs its own 3-of-5+48h proposal. This is a usability hazard, not a security issue.

**Impact:**

Bounty top-ups are slow (48h × 2 = 96h minimum) and require Reserve coordination. May surprise operators expecting a simpler flow.

**Recommendation:**

Document the multi-step process in the bounty operations runbook. No code change required.

**Soft-launch cap mitigation:** N/A.

---

### [LOW] CC-7: PWMToken.transferOwnership/renounceOwnership not exercised (A1 INFO carry-over)

**Files:**
  - `infrastructure/agent-contracts/contracts/PWMToken.sol:27` (Ownable inheritance)
  - `infrastructure/agent-contracts/deploy/erc20.js:75-77` (token deploy)

**Description:**

PWMToken.Ownable owner is deployer EOA. The owner has no exploitable powers (cap fully minted at genesis). However, leaving the Ownable role active is confusing to integrators and external reviewers (who may assume it has mint/recovery powers). A1 and A8 both flagged.

**Cross-contract impact:** The Ownable role is dead but visible in block explorers / etherscan. Could mislead users into thinking the deployer can mint more.

**Recommendation:**

Add `await token.renounceOwnership()` to deploy/erc20.js after genesis distribution. Permanently removes the misleading dead role.

**Soft-launch cap mitigation:** N/A.

---

## Cross-contract invariants summary

Distilled list of cross-contract invariants verified during this review:

| # | Invariant | Status |
|---|---|---|
| I-1 | PWM total supply is fixed at 21M post-genesis | ✓ enforced by ERC20Capped |
| I-2 | Only PWMCertificate can call `reward.distribute` and `minting.mintFor` | ✓ enforced by `onlyCertificate` modifiers |
| I-3 | Only PWMStakingERC20 can call `reward.seedBPool` | ✓ enforced by `msg.sender == staking` |
| I-4 | Only PWMMintingERC20 can call `reward.depositMinting` | ✓ enforced by `msg.sender == minting` |
| I-5 | Only PWMRewardERC20 can call `treasury.receive15pct` | ✓ enforced by `onlyReward` |
| I-6 | Only PWMGovernance can call setX on Reward/Staking/Cert/Minting/Treasury | ✓ post-handoff (5 of 6 — Registry missing, see CC-1) |
| I-7 | PWMGovernance.executeExec cannot re-execute the same proposal | ✓ p.executed=true BEFORE call |
| I-8 | PWMGovernance.executeExec cannot target itself | ✓ require(target != address(this)) |
| I-9 | Cross-contract reentrancy from target back into PWMGovernance is blocked | ✓ onlyFounder rejects contract-callers |
| I-10 | All 5 soft-launch caps independently enforce | ✓ no coupling between caps |
| I-11 | Staking total cap and reward pool cap reclaim space on resolution | ✓ decrements on graduate/slash |
| I-12 | PWMVesting has zero cross-contract attack surface | ✓ no setters, immutable schedule |
| I-13 | Trust web has no circular trust relationships | ✓ caller-callee one-way only |
| I-14 | Genesis distribution sums to 21M | ✓ deploy script asserts |
| I-15 | Registry append-only | ✓ no delete/update in PWMRegistry |
| I-16 | Registry ownership matches governance trust | ✗ **VIOLATED** — see CC-1 |
| I-17 | finalize() is graceful when minting is unavailable | ✗ **VIOLATED** — see CC-2 |
| I-18 | graduate() succeeds whenever stake is Active | ✗ **VIOLATED** — see CC-3 |
| I-19 | Timelock starts when proposal becomes executable (not proposed) | partial ✓ — only ExecProposal, see CC-4 |

---

## Confidence

Reviewed **deeply** (every line, cross-referenced with deploy and prior findings):
  - PWMGovernance.sol (368 lines)
  - PWMCertificate.sol (251 lines)
  - PWMRewardERC20.sol (204 lines)
  - PWMTreasuryERC20.sol (97 lines)
  - PWMStakingERC20.sol (184 lines)
  - PWMMintingERC20.sol (299 lines)
  - PWMRegistry.sol (75 lines)
  - PWMVesting.sol (91 lines)
  - PWMToken.sol (49 lines)
  - deploy/erc20.js (283 lines)

Reviewed **structurally** (skimmed for control flow only):
  - The non-ERC20 sibling contracts (PWMMinting.sol, PWMReward.sol, PWMStaking.sol, PWMTreasury.sol — the native-coin versions) — not in scope for deploy.
  - PWMFaucet.sol — testnet only, out of scope.

Cross-referenced findings from A1-v3, A2-v2, A3-v2, A4, A8.

## What I did NOT check

  - **Mythril/symbolic-execution coverage:** A4 covered Slither; Mythril overnight scan is pending. Symbolic execution could find edge-case integer overflows in the reward math (`drawAmt * d.shareRatioP * SPLIT_AC_CP / (BPS_DENOM * BPS_DENOM)`) — Solidity 0.8 reverts on overflow, so any such finding would be DoS not theft, but worth checking.
  - **The actual genesis benchmark registration flow:** deploy/erc20.js doesn't register any. A separate `register_genesis.js` likely exists; not in my scope.
  - **The Phase 5 mainnet runbook MD itself** — A9 spec consistency check covers that.
  - **Gas / DoS via long benchmark arrays in PWMMintingERC20.removeBenchmark** — A3 covered the swap-and-pop pattern.
  - **The non-ERC20 sibling contracts** (PWMMinting.sol, PWMReward.sol, PWMStaking.sol, PWMTreasury.sol) — these exist in the contracts/ folder but deploy uses the ERC20 versions. Out of scope.
  - **Off-chain rank-assignment logic** for certificates — `c.rank` is set by submitter. Trust the indexer / off-chain process to assign correctly; out of cross-contract scope.
  - **Concrete proposal-id collision** between Proposal, FounderChange, ExecProposal — they have separate counters and mappings, so no collision possible. Spot-checked, no issue.
