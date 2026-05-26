# A1-v3 Security Review — PWMGovernance executeCall primitive

**Scope:** Single contract — `infrastructure/agent-contracts/contracts/PWMGovernance.sol`
**Patch under review:** commit `fe3ba529` on `release/d9-soft-launch-2026-05-18`
**Pass:** 3rd (focused, post-fix)
**Reviewer:** Agent A1-v3
**Date:** 2026-05-18

---

## Severity counts

| Severity | Count | Delta vs A1-v2 | Deploy-blocker |
|----------|-------|----------------|----------------|
| CRITICAL | 0     | **−1** (the A1-v2 CRITICAL is RESOLVED) | NO |
| HIGH     | 1     | +1 (carry-over MEDIUM-1 elevated — see "Carried-over findings") | **YES — requires Director sign-off as acceptable for soft-launch, or fix before mainnet** |
| MEDIUM   | 2     | =     | NO (operational mitigations exist) |
| LOW      | 1     | +1   | NO |
| INFO     | 2     | +2 | NO |

**Headline:** the patch correctly resolves A1-v2's CRITICAL. The day-31 unpause flow works end-to-end (verified by reading the test and confirming the test executes the intended assertion path). No new CRITICAL / HIGH-severity bugs introduced by the patch itself. The two carry-over MEDIUM findings from A1-v2 (timelock-from-propose-time and rotated-founder-stale-approvals) now apply to the new exec proposal flow as well — and their blast radius is expanded from "internal `parameters` map" to "any state on any sibling contract that trusts governance," which is why MEDIUM-1 is escalated to HIGH per A1-v2's stated cross-pollination rule (line 182 of A1-v2 report).

**ABORT_DECISION re-trigger:** NOT triggered. The A1-v2 CRITICAL is genuinely fixed; the remaining HIGH is the *same* HIGH that A1-v2 already predicted would surface upon adoption of `proposeExec`, and it has been a known limitation since v1. Director can accept it for soft-launch with explicit awareness; for mainnet, recommend resolving before audit.

---

## Verification of A1-v2 CRITICAL fix

A1-v2's CRITICAL was: "PWMGovernance has no on-chain mechanism to call the new `setX` functions added by the patch. Once `setGovernance(address(this))` is called on the 5 sibling contracts, all post-patch setters become permanently unreachable; the recovery option is 'redeploy from scratch'."

### Walk-through of the day-31 mintingPaused unpause flow

Verified against contract code lines 268-333 and the integration test `test/PWMGovernance_exec.test.js` lines 244-280.

```
Day 0:    PWMMintingERC20 deployed with initialGovernance = f0 (deployer EOA).
          f0 calls minting.setMintingPaused(true).
          f0 calls minting.setGovernance(<PWMGovernance contract addr>).
          (Now mintingPaused=true, governance=PWMGovernance.)
          From here, only PWMGovernance can flip mintingPaused.

Day 29:   Founder f0 calls
            governance.proposeExec(
              target = <minting addr>,
              data   = minting.interface.encodeFunctionData("setMintingPaused", [false])
            )
          → returns id=0. p.approvals=1 (f0 auto-approved). p.proposedAt = block.timestamp.
          Events: ExecProposalCreated(0, minting, calldata, f0)
                  ExecProposalApproved(0, f0, 1)

Day 29:   Founder f1 calls governance.approveExec(0). p.approvals=2.
Day 29:   Founder f2 calls governance.approveExec(0). p.approvals=3 ≥ REQUIRED_APPROVALS.

Day 31:   block.timestamp ≥ proposedAt + 48h.
          Any founder (test uses f0) calls governance.executeExec(0).
            - require(p.proposedAt != 0)                       ✓ (line 314)
            - require(!p.executed && !p.cancelled)             ✓ (line 315)
            - require(p.approvals >= 3)                        ✓ (line 316)
            - require(now >= proposedAt + 48h)                 ✓ (line 317)
            - p.executed = true                                ← CEI set BEFORE call (line 319)
            - (success, returnData) = minting.call(setMintingPaused(false))
                                                               ← external call (line 320)
              Inside minting: onlyGovernance passes because msg.sender == governance.
              State change: mintingPaused = false.
              Event: MintingPausedUpdated(false).
            - success = true, no revert path.
            - emit ExecProposalExecuted(0, minting, returnData)

Post:     minting.mintingPaused() returns false. Token unpause is durable.
```

**Confirmed: the day-31 unpause flow works end-to-end.** This is also what the integration test on lines 244-280 of `PWMGovernance_exec.test.js` asserts, and the test passes (14/14 in the new suite — verified by running `npx hardhat test test/PWMGovernance_exec.test.js`, all 14 green).

The test on lines 251-280 deliberately exercises the **post-handoff** flow (lines 267-268: `minting2.setGovernance(governance)` before the exec proposal) — meaning the test correctly verifies that an EOA founder cannot directly call `setMintingPaused` after handoff and that the only path is via `proposeExec → approveExec ×3 → wait 48h → executeExec`. **The test is correct, not just passing.**

### Generalization: every `onlyGovernance` setter on every sibling is now reachable

I enumerated the `onlyGovernance` functions across the 5 sibling contracts (via grep across `contracts/PWM{Staking,Reward,Treasury,Minting,Certificate}ERC20.sol`):

- **PWMStakingERC20:** `setGovernance`, `setReward`, `setStakeAmount`, `setMaxTotalStakeWei`, `graduate`, `slashForChallenge`, `slashForFraud`
- **PWMRewardERC20:** `setGovernance`, `setCertificate`, `setMinting`, `setStaking`, `setTreasury`, `setMaxBenchmarkPoolWei`, `depositBounty`
- **PWMTreasuryERC20:** `setGovernance`, `setReward`, `setTransfersPaused`, `payAdversarialBounty`
- **PWMMintingERC20:** `setGovernance`, `setCertificate`, `setReward`, `setMintingPaused`, `setDelta`, `setPromotion`, `registerBenchmark`, `setBenchmarkRho`, `removeBenchmark`
- **PWMCertificate:** `setGovernance`, `setRegistry`, `setReward`, `setMinting`, `setApprovedSubmitter`, `setSubmissionPermissionless`, `resolveChallenge`

All of these are now reachable via `proposeExec(target, calldata) → approveExec×3 → wait 48h → executeExec`. The protocol is no longer soft-bricked.

**A1-v2 CRITICAL: VERIFIED RESOLVED.** No further action required for the original deadlock.

---

## New findings on patched code

I traced every line of the 4 new entry points (`proposeExec`, `approveExec`, `executeExec`, `cancelExec`) plus the struct & mapping additions. Beyond the carry-over findings below, here is what I found that is **specific to the patch itself**:

### [LOW] `cancelExec` requires no minimum approvals — a single rogue founder can grief any exec proposal indefinitely (NEW)

**Location:** `cancelExec` (lines 336-342).

**Issue:** Any single founder can cancel any pending exec proposal at any time before execution, including one they themselves did not propose and which has already received 4 of 5 approvals. This is the same model as `cancelProposal` and `cancelFounderChange`, so consistent with the existing pattern. But the **blast radius is now bigger**: a single compromised founder key can DoS *all cross-contract governance actions indefinitely* by spamming `cancelExec(id)` on every proposal as soon as it lands.

A1-v2 already raised a closely-related concern (lines 249-251 of A1-v2 report, the "compromised single founder can indefinitely DoS all governance, including the very founder-rotation needed to evict them" point). That earlier framing was about `cancelFounderChange`. The exec primitive now adds a second equally-broad DoS surface: the same compromised founder can also block any cross-contract setter call.

**Why LOW not MEDIUM:** the attacker cannot *cause* state change, only prevent it. The DoS is recoverable by rotating the compromised founder out via `executeFounderChange` (which the compromised founder can also cancel — but the rotation flow has its own race that A1-v2 already flagged; this is mostly orthogonal). For a 30-day soft-launch with known operators on Sepolia, this is acceptable.

**Recommendation (defer to mainnet):** require 2-of-5 to cancel an exec proposal (or block proposer-self-cancellation in tandem with founder-rotation hardening). Same fix pattern as A1-v2's recommendation for `cancelFounderChange`.

### [INFO] EOA targets silently succeed (NEW)

**Location:** `executeExec` (line 320).

**Issue:** A founder can propose `proposeExec(target=<some EOA>, data=<any 4+ bytes>)`. After 3 approvals + 48h, `executeExec` calls the EOA. EVM rule: a `call` to an EOA with non-empty calldata returns `(success=true, returnData=0x)` and does nothing. The proposal is marked executed, the event fires, no error.

**Why INFO not LOW:** founders explicitly chose the target and the data; there is no third-party harm. The only "loss" is a wasted 48h timelock cycle. The pattern matches industry-standard Timelock/Safe modules (Compound, OpenZeppelin TimelockController) which also do not block EOA targets.

**Recommendation:** document in operator runbook ("verify target is a contract before approving"). Optional code-level mitigation is `require(target.code.length > 0, "PWMGovernance: target not contract")` in `proposeExec`. Not blocking.

### [INFO] Re-entrancy from target into PWMGovernance is bounded but worth documenting (NEW)

**Location:** `executeExec` line 320 (`p.target.call(p.data)`).

**Issue:** If the target's invoked function makes a callback into PWMGovernance, what can it do? I traced every external entry point:

1. **`executeExec(sameId)` re-entry:** blocked by `p.executed = true` set on line 319 BEFORE the call. The re-entrant call hits the `!p.executed && !p.cancelled` check on line 315 and reverts. **Safe.** (Verified.)

2. **`proposeExec` from inside the target call:** the target would need to be a founder address, which it is not unless a founder accidentally chose a founder EOA as the target. Even if it were, the new proposal still needs 3 approvals and 48h. **Safe (assuming target is not a founder EOA).**

3. **`executeFounderChange` from inside the target call:** the target would need to be a founder (msg.sender check). Same constraint as #2. Even if so, the founder change would need to be pre-approved 3-of-5 with timelock elapsed. **Safe under same assumption.**

4. **`cancelExec(differentId)`:** target would need to be a founder. **Safe under same assumption.**

5. **`executeProposal(differentId)` (parameter logbook):** target would need to be a founder. Even if so, no fund impact — only writes to `parameters` mapping. **Safe even if exploited.**

**Conclusion:** all callback paths require the target to *also* be a founder address. The system invariant "no founder address is also a contract that PWMGovernance calls via exec" is operationally enforceable: founders are EOAs (Safe wallets / hardware wallets), not contracts on the call-target list. If a founder is ever a contract (e.g., a Gnosis Safe used as a founder), then `proposeExec(target=<that Safe>, data=...)` would be possible AND that Safe could in principle re-enter PWMGovernance from inside its `execTransaction` handler. **In practice this is moot** because the Safe handler does not synchronously call back into the originator; but worth a note.

**Recommendation:** document in operator runbook that exec targets must NOT be any of the 5 founder addresses (or addresses that can act as founders).

### [INFO] `bytes data` is stored in storage — gas cost per propose grows with calldata size (NEW)

**Location:** `execProposals[id].data` (line 56 + line 285).

**Issue:** Storing arbitrary `bytes data` in storage costs ~20k gas/word for SSTORE. A founder proposing a large calldata blob (e.g., 10 KB) would pay ~6.4M gas to store it. Not a security bug — just operational. Standard pattern; OpenZeppelin TimelockController stores `bytes32 hash` of the calldata instead, requiring the executor to re-supply the full bytes (and verifying hash matches). That alternative is cheaper for huge blobs but more error-prone (mismatched bytes at execute time means re-propose + re-wait).

**Recommendation:** none for soft-launch. For mainnet, consider switching to hash-of-calldata pattern à la OZ TimelockController if storage costs become a concern. Founder-proposed setter calls are typically <100 bytes, so this is mostly theoretical.

---

## Carried-over findings from A1-v2

### [HIGH] Timelock measured from proposedAt, not from approval-threshold reached — **CARRIES TO `executeExec`; ELEVATED FROM MEDIUM TO HIGH per A1-v2 cross-pollination rule**

**Location:** `executeExec` line 317: `require(block.timestamp >= p.proposedAt + TIME_LOCK, ...)`.

**Issue:** identical to A1-v1 MEDIUM-1 and A1-v2 MEDIUM-1. The 48h timelock starts at proposal-creation time, not at the moment the 3rd approval lands. So if founders #1, #2, #3 are coordinated, founder #1 can `proposeExec` at t=0, founders #2 and #3 can each `approveExec` at t=47h59m, and founder #1 can `executeExec` at t=48h+1s — providing effectively only 1 second of "window" between approvals-met and execution.

**Why elevated:** A1-v2 explicitly stated (line 182): "If the new CRITICAL is fixed via Alternative 2 (proposeExec), then the late-approval rush problem would apply to arbitrary cross-contract calls — meaning a coordinated 3-of-5 could rush a `setApprovedSubmitter(attacker, true)` exec proposal, then immediately submit a self-dealing cert. This would elevate the severity to HIGH once an `execute()` primitive lands." That primitive has now landed. So per the auditor's own stated rule, this is now HIGH.

**Realistic exploit:** 3 colluding founders (≥3-of-5 of the multisig — which is the same threshold as the entire governance) propose `setApprovedSubmitter(colluder, true)` on PWMCertificate, all auto-approve within seconds, and execute at t=48h+1. Other 2 founders have no realistic chance to detect+cancel within a 1-second window.

**Why this is partially mitigated in practice:**

1. The attacker is already 3-of-5 of the multisig, which IS the governance threshold. They could equivalently `proposeExec` with separate well-timed proposals; the timelock is the *only* protection against same-tx-batch collusion, and it provides 48h advance warning ONLY if the proposal is *visible* but not yet 3-of-5 approved. If 3 colluders approve simultaneously at the end of the window, the 48h-advance-warning concept fails.
2. For an external observer with on-chain monitoring of `ExecProposalCreated` events, the warning starts at proposal-creation time, not approval-threshold. So the "1 second window" issue is specifically about the *intra-multisig veto* window for the 2 non-colluding founders.

**Recommendation:** for soft-launch (where founders are coordinated off-chain and the 5 keys are operated by Director + small trusted ops team), accept as known limitation. Document in Director sign-off. For mainnet, change to: `require(block.timestamp >= max(p.proposedAt, p.thresholdReachedAt) + TIME_LOCK, ...)`. Implementation: add `uint256 thresholdReachedAt` to `ExecProposal`, set it on the approval that brings `approvals` from 2→3 (line 303-area in `approveExec`).

**Status for deploy:** acceptance is a Director decision. Recommend: include in Director sign-off as explicit known limitation. Acceptable for Sepolia soft-launch; should be fixed before mainnet.

### [MEDIUM] Rotated founder's prior approvals still count — **CARRIES TO `approvedExec`**

**Location:** `approvedExec[id][addr]` is set true in `proposeExec` (line 291) and `approveExec` (line 302). It is **never cleared** when `executeFounderChange` (lines 233-247) rotates that founder out.

**Issue:** identical structural finding to A1-v2 MEDIUM-2. Consider:

```
t=0:   Founder f3 proposes exec id=0 to setApprovedSubmitter. approvals=1, approvedExec[0][f3]=true.
t=1h:  Founder f4 approves. approvals=2, approvedExec[0][f4]=true.
t=2h:  Founder f3's key is compromised. Other founders propose & execute a
       founderChange replacing f3 with f3_new. After 48h+, executeFounderChange runs:
         - founders[3] = f3_new
         - isFounder[f3] = false
         - isFounder[f3_new] = true
         (approvedExec[0][f3] is NOT touched.)
t=50h: p.approvals on exec id=0 still reads 2 (it stored a count, not a live recount).
t=51h: A second compromised-or-colluding founder f0 approves. approvals=3.
t=52h: executeExec(0) runs and sets the malicious approvedSubmitter.
```

The vote of the rotated-out (formerly compromised) f3 still counts, because `p.approvals` is a stored counter, not a live re-count over `isFounder[approver]`.

**Why MEDIUM not HIGH for soft-launch:** the timing is tight — the attacker must rotate-out the compromised founder AFTER they've already approved the malicious exec but BEFORE the second collaborator approves. In a 48h timelock, the rotation itself takes 48h, so the attacker would already have 48h of warning. In practice, an honest majority sees the malicious `ExecProposalCreated` event AND has 48h to cancel it (via `cancelExec`, which any single founder can do — see LOW finding above).

**Why MEDIUM not LOW:** A1-v2 (line 199) noted "with the new patches inert (per CRITICAL above), the impact is theoretical for now. Once governance has cross-contract reach, this becomes more pressing." It IS more pressing now. The blast radius is "fund theft enabler" — a malicious `setApprovedSubmitter(attacker, true)` opens a self-dealing certification path that could drain reward pools.

**Recommendation:** for soft-launch, accept as known limitation given the strong off-chain coordination and cancel-by-any-founder safety net. For mainnet, fix:
- Option A (local fix): on `executeFounderChange` (line 241-246), iterate over `nextExecProposalId` and decrement `p.approvals` for any pending exec proposal where `approvedExec[id][oldAddr]==true && !executed && !cancelled`. O(n) over pending proposals — bounded.
- Option B (better): change `executeExec` to live-recount approvals by iterating `founders[0..4]` and counting `approvedExec[id][founders[i]]`. O(5) loop, cheap. Same fix applies to `executeProposal` and `executeFounderChange`. Symmetric with A1-v2 MEDIUM-2 recommendation.

**Status for deploy:** Director decision; recommend include in known-limitations list.

### [MEDIUM] `parameters` has no key allow-list — STILL VALID

A1-v2 MEDIUM-3. No change. With the exec primitive added, `parameters` mapping is **arguably obsolete** for on-chain protocol effects (no other contract reads it). Continues as observability concern only. Recommendation: at mainnet refactor, decide whether to remove `parameters` entirely or to add an allow-list on both `(bytes32 key)` and `(address target, bytes4 selector)` pairs.

**Status for deploy:** not deploy-blocking. Already acknowledged in A1-v2.

### Not carried (LOW + INFO from A1-v2)

The LOWs and INFOs from A1-v2 that referenced minting/staking/treasury patches are out-of-scope for this pass. A1-v2's discussion of "single-founder DoS via cancel" (the cancelFounderChange one) is structurally similar to the new LOW finding I added above for `cancelExec`, but I treat them as **separate findings on separate code paths** (one is on the founder-rotation flow, one is on the exec flow) — both are real, both have the same recommended fix (require 2-of-5 to cancel, or block proposer-self-cancellation).

---

## Specific checks applied (per task prompt)

### A. Reentrancy and CEI
- **`p.executed = true` BEFORE external call:** ✓ verified at line 319 (set), line 320 (call). Correct CEI.
- **Re-entry into same id:** blocked by line 315 (`!p.executed`). ✓
- **Re-entry into NEW exec proposal:** possible but requires target to be a founder address. Operationally avoidable (see INFO finding #3 above). Not a vulnerability under stated invariant.
- **Re-entry into `executeFounderChange`:** same constraint as above; requires target to be a founder.
- **Re-entry into `executeProposal` (parameter logbook):** same constraint; no fund impact anyway.
- **Re-entry into `cancelExec(differentId)`:** same constraint. Note that this would be an *anti-attack* (attacker self-cancelling a parallel honest proposal) but again requires target to be founder.

### B. Authorization
- **`onlyFounder` on all 4 entry points:** ✓ verified — `proposeExec` (273), `approveExec` (297), `executeExec` (312), `cancelExec` (336).
- **`multisigActive` on all 4 entry points:** ✓ verified at same lines. Test on lines 220-240 of `PWMGovernance_exec.test.js` confirms post-DAO activation all 4 reject. (Note: this confirms a subtle correctness property — once DAO is activated, even the exec primitive is permanently disabled, so the DAO-side governance must independently implement an equivalent execute primitive for the post-DAO era. This is by design per the contract comment lines 157-160; recommend Director confirm DAO implementation roadmap includes equivalent exec capability.)
- **Same founder double-approve:** ✓ blocked by `approvedExec[id][msg.sender]` check on line 301.
- **Rotated founder stale approvals:** ✗ NOT addressed — same issue as A1-v2 MEDIUM-2; see carry-over above.

### C. Target validation
- `target != address(0)`: ✓ line 278.
- `target != address(this)`: ✓ line 279.
- **Proxy-back attack:** a target contract that on `call` re-enters PWMGovernance is theoretically possible, but as analyzed in INFO #3 above, all callback paths require the calling target to be a founder address. Under the operational invariant that exec targets are sibling contracts (Staking/Reward/Treasury/Minting/Certificate) and NOT founder addresses, this is safe.
- **EOA target:** silently succeeds. See INFO #2 above. Documented as non-blocking footgun.
- **Predecessor contract:** N/A — not upgradable, no proxy pattern, no migration path in scope.

### D. Calldata validation
- `data.length >= 4`: ✓ line 280. Sufficient to ensure a selector is present.
- **Wrong-selector accident:** target's fallback (if any) might catch this. For the 5 sibling contracts, NONE have a fallback function (verified by grep — `grep -n "fallback\|receive" contracts/PWM{Staking,Reward,Treasury,Minting,Certificate}*.sol` returns nothing). So a wrong-selector call to a sibling reverts with empty returndata → `"PWMGovernance: exec call failed (no reason)"` per line 329. Safe.
- **Storage-slot manipulation:** non-upgradable contract; no proxy; nothing to manipulate. ✓

### E. Timelock
- Measured from `proposedAt`: ✓ line 317. **Inherits MEDIUM-1; elevated to HIGH for this exec primitive.** See carry-over above.

### F. Revert handling
- Inline assembly bubble-up (lines 322-330): ✓ correct. Reads length from first 32 bytes of `returnData` mem ptr, reverts with the payload from offset 32 of length `sz`. Standard pattern.
- On failure, `p.executed = true` is rolled back: ✓ confirmed — the entire transaction reverts due to the `revert(...)` on line 326 or 329, so all state changes including line 319 are reverted. The test on lines 165-180 of `PWMGovernance_exec.test.js` verifies this: after a bubbled revert, `p.executed` is still `false` and the proposal can be re-attempted. **Important behavior: this is "fail-open for retry", not "fail-closed."** If a target reverts (e.g., due to a transient state mismatch), founders can simply retry without re-proposing. This is the correct choice for soft-launch.
- Huge returnData: yes, target can return arbitrary-size bytes. Cost is paid by the founder executing — they can cap their gas limit if concerned. Acceptable.

### G. Event correctness
- `ExecProposalCreated(id, target, data, proposer)`: ✓ line 292. All 4 fields populated.
- `ExecProposalApproved(id, approver, approvals)`: ✓ lines 293, 304. Emits on auto-approve too.
- `ExecProposalExecuted(id, target, returnData)`: ✓ line 332. Includes returnData blob.
- `ExecProposalCancelled(id)`: ✓ line 341.
- `data` and `returnData` are `bytes` and `bytes` indexed-as-data (not indexed-as-topic), so they are stored in event data, paid by emitter, not by readers. ✓

### H. Cross-contract trust
- Confirmed: governance now has the audited-Safe-with-Timelock pattern, which is the industry-standard architecture for this exact use case. The 3-of-5 multisig + 48h timelock concentrates protocol risk, but in a way that has years of precedent (Compound governance, Optimism's protocol multisig, etc.). **INFO-level note for Director:** before mainnet, ensure the 5 founder keys are operated with **independent custody** (no single ops team holding all 5 keys; HSM or Safe with 3 different signers each, etc.).

### I. Acceptance of A1-v2 CRITICAL fix
**VERIFIED RESOLVED.** Full walk-through above. Test at lines 244-280 of `PWMGovernance_exec.test.js` is correct (not just passing): it sets up the realistic post-handoff state where the EOA founder no longer has direct call access to `setMintingPaused`, and proves the only working path is through `proposeExec → approveExec×3 → wait 48h → executeExec`. All 14 tests in the new suite pass (`npx hardhat test test/PWMGovernance_exec.test.js`).

---

## Confidence

**Deep-traced (line-by-line, both code and test):**
- `proposeExec` (lines 272-294)
- `approveExec` (lines 297-305)
- `executeExec` (lines 312-333) — including the assembly revert bubble-up
- `cancelExec` (lines 336-342)
- `ExecProposal` struct + `execProposals`, `approvedExec`, `nextExecProposalId` storage layout
- All 4 new events
- `test/PWMGovernance_exec.test.js` (all 14 cases, end-to-end)

**Cross-referenced:**
- `multisigActive` modifier and DAO activation flow (lines 89-92, 161-172, 220-240 of test)
- `onlyGovernance` setters across all 5 sibling contracts (grep + spot-read of `PWMMintingERC20.setMintingPaused` and `PWMStakingERC20.setMaxTotalStakeWei`)
- A1-v2 findings (`A1_v2_token_governance_vesting_2026-05-18.md`), specifically the MEDIUM-1 (timelock) and MEDIUM-2 (rotated approvals) findings — confirmed both apply structurally to `executeExec` / `approvedExec`

**Structural-only (not line-by-line) — unchanged from A1-v2:**
- Parameter / founder-change flows (lines 104-257). I re-read for cross-pollination with exec flow but did not re-audit; A1-v2's findings on these stand.

---

## What I did NOT check

- **Other 8 contracts** — explicitly out of scope per task prompt (A2/A3 patches, ABIs, deploy script, vesting contract, etc.). If `proposeExec` is misused to call functions on those contracts that have *other* bugs, those bugs would still exist; this pass only verifies that the exec primitive itself is sound.
- **Gas-grief upper bounds** — did not stress-test how large a `data` blob can be before storage SSTORE costs exceed block gas limit. Founders pay this themselves, so non-issue for protocol security.
- **DAO-era replacement** — once `activateDAO()` is called, the exec primitive is permanently disabled. Whatever DAO implementation lands post-M3 must independently implement an equivalent exec capability. I did NOT verify the DAO roadmap; recommend Director confirm.
- **Slither/Mythril re-run** — did not run static analyzers on the patched contract. The patch is small (~70 LoC) and the patterns are all well-known (CEI, multisig, timelock); a static-analyzer pass is recommended before mainnet but not deploy-blocking for soft-launch.
- **Formal proof of the CEI invariant** — relied on reading. A formal proof (e.g., Certora) is appropriate for mainnet, not soft-launch.
- **Multi-call atomicity** — did not check what happens if a founder wants to atomically do two cross-contract calls (e.g., `setReward` on staking AND `setStaking` on reward in one shot). The current primitive is single-call; a multi-call extension would require batched proposals or a separate `proposeExecBatch`. Not a bug; just a feature gap. Documenting in case Director needs it.
- **Reentrancy across all reachable paths from sibling contracts back into governance via OTHER sibling contracts** — e.g., target A calls into B, which calls back into governance. This is the same "target-must-be-founder" invariant I analyzed in INFO #3, but only verified at the first-hop level. A multi-hop adversarial graph analysis was not performed.

---

## Summary

A1-v2's CRITICAL is **genuinely fixed** by commit `fe3ba529`. The exec primitive is well-constructed: CEI is correct, authorization is consistent with the rest of the contract, calldata and target validation are reasonable, revert bubble-up works, all 14 new tests pass, and the day-31 unpause flow works end-to-end. The protocol is no longer soft-bricked.

**Remaining risk (HIGH severity, single):** the timelock-from-propose-time issue (carry-over MEDIUM-1 from A1-v2, elevated to HIGH per A1-v2's own cross-pollination rule now that the exec primitive expands blast radius). Requires Director sign-off as known-limitation for soft-launch, or fix before mainnet.

**Remaining risk (MEDIUM):** stale rotated-founder approvals on `approvedExec` (analogous to A1-v2 MEDIUM-2 for parameter proposals). Bounded by 48h timelock + any-founder cancel. Acceptable for soft-launch; fix before mainnet.

**ABORT_DECISION:** does NOT re-trigger. Recommend Director proceed with soft-launch deploy subject to explicit sign-off acknowledging the two carry-over findings as known limitations.
