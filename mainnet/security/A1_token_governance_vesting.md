# A1 Security Review — PWMToken, PWMGovernance, PWMVesting

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A1)
**Scope:** 3 contracts (PWMToken.sol 49 LoC, PWMGovernance.sol 232 LoC, PWMVesting.sol 91 LoC) — 372 total lines
**Soft-launch caps in effect:** STAKING_TVL_CAP_USD=$1000, MINTING_PAUSED=true, TREASURY_TRANSFERS_PAUSED=true

## Summary

| Severity | Count | Blocks deploy? |
|---|---|---|
| CRITICAL | 0 | NO |
| HIGH     | 0 | NO |
| MEDIUM   | 3 | NO |
| LOW      | 4 | NO |
| INFO     | 4 | NO |

No CRITICAL or HIGH findings. Three MEDIUM findings worth fixing before the unfettered (post-audit) launch, but all are bounded by the soft-launch posture or by economic reality (the three founder addresses signing every action). Deploy is not blocked.

## Findings

### [MEDIUM] Timelock measured from proposal-time, not from approval-threshold reached

**File:** `contracts/PWMGovernance.sol:110-120`
**Function:** `executeProposal(uint256 id)` (and identically `executeFounderChange`, `activateDAO`)
**Description:** The 48-hour timelock check is `block.timestamp >= p.proposedAt + TIME_LOCK`. `proposedAt` is set once at proposal creation and never refreshed when approvals arrive. So a proposal can sit at 1 approval for 47h59m, receive its 2nd and 3rd approvals in the last minute, and become executable within the same block (or within a few blocks). The effective review window between "threshold reached" and "executable" can collapse to ~0.
**Impact:** A coordinated 3-of-5 attacker (or three founders whose keys were compromised together but only revealed late) can bypass the spirit of the 48h timelock. The other two founders may not have enough time to call `cancelProposal` after seeing the threshold reached. For parameter changes this can immediately re-set sensitive values (staking floors, challenge periods); for `activateDAO` it permanently disables the multisig path; for founder rotation it instantly evicts a slot.
**Reproducibility:** (1) Compromise/coordinate 3 founder keys. (2) Founder A calls `proposeFounderChange(slot=0, newAddr=attacker)` at t=0. (3) Wait until t=47h59m. (4) Founders B and C call `approveFounderChange` in two transactions in the same block at t=47h59m. (5) Founder A immediately calls `executeFounderChange` at t=48h00m. Total window in which the honest two founders see the third approval and can react: under one block (~2s on Base).
**Recommendation:**
```diff
-   require(block.timestamp >= p.proposedAt + TIME_LOCK, "PWMGovernance: timelock not elapsed");
+   require(block.timestamp >= p.thresholdReachedAt + TIME_LOCK, "PWMGovernance: timelock not elapsed");
```
Add `uint64 thresholdReachedAt` to `Proposal` and `FounderChange`. In `approveProposal`/`approveFounderChange`, after incrementing `approvals`, set `p.thresholdReachedAt = uint64(block.timestamp)` only when `p.approvals == REQUIRED_APPROVALS` and `p.thresholdReachedAt == 0`. This matches industry convention (Compound Timelock, OpenZeppelin TimelockController) and gives the dissenting minority a full 48h to call `cancelProposal` after the threshold is met. Alternatively, simply reset `proposedAt` each time a new approval comes in (simpler but slightly more invasive UX-wise).
**Soft-launch cap mitigation:** PARTIAL. Parameter changes during the soft-launch period cannot break the $1k staking cap (the cap is hard-coded constants in PWMStakingERC20, not governance-settable per the spec). But founder rotation and DAO activation can be rushed; these are outside the soft-launch caps.

---

### [MEDIUM] Revoked founder's prior approvals still count toward execution threshold

**File:** `contracts/PWMGovernance.sol:99-107` (and `194-202` for founder changes)
**Function:** `approveProposal`, `executeProposal`, `approveFounderChange`, `executeFounderChange`
**Description:** Approvals are stored as `mapping(uint256 => mapping(address => bool)) public approved`. When a founder is rotated out via `executeFounderChange`, their `isFounder[oldAddr]` is set false, but `approved[<anyPendingId>][oldAddr]` is NOT cleared and `p.approvals` is NOT decremented. Any pending proposal still carries that revoked founder's vote, and execution checks only `p.approvals >= REQUIRED_APPROVALS`, not whether the approvers are still founders.
**Impact:** If founders detect that founder A is compromised and rotate A out, any in-flight proposal that A pre-approved (including malicious ones A may have planted) keeps A's vote. The attacker needs only 2 collaborating remaining founders (instead of 3) to push it through. Worse, A could plant many proposals before being rotated and they all remain pre-approved-by-A in perpetuity until cancelled.
**Reproducibility:** (1) Compromised founder A calls `proposeFounderChange(slot=1, newAddr=attacker_alt)`. Auto-approves (1/3). (2) Compromised founder A also calls `proposeParameter(<sensitive key>, <malicious value>)`. Auto-approves (1/3). (3) Honest founders detect compromise and execute `proposeFounderChange(slot=0, newAddr=fresh_addr_for_A)`, rotating A out 48h later. (4) `isFounder[A_oldKey] = false`. (5) But `approved[<malicious_proposalId>][A_oldKey] = true` and `approvals = 1` persist. (6) If two other founder keys are then phished/coerced, they hit approvals=3 and execute, bypassing the intended "after rotation, A's votes don't count" protection.
**Recommendation:** On `executeFounderChange`, iterate pending proposals and clear approvals from `oldAddr` (gas-expensive, not recommended). Better: change the execute check to recount on-the-fly by iterating `founders[0..4]` and checking each is approved AND `isFounder[founders[i]]`. Since NUM_FOUNDERS=5, this is O(5) gas. Specifically:

```solidity
function _liveApprovals(uint256 id) internal view returns (uint8 n) {
    for (uint256 i = 0; i < NUM_FOUNDERS; i++) {
        if (approved[id][founders[i]]) n++;
    }
}
// Then in executeProposal / executeFounderChange:
require(_liveApprovals(id) >= REQUIRED_APPROVALS, "PWMGovernance: insufficient approvals");
```

This ensures only currently-seated founders count toward the threshold. Drop the `p.approvals` field or keep it for events only.
**Soft-launch cap mitigation:** NO. This is an architectural correctness gap that does not interact with the staking/treasury caps. However, it requires a multi-step social compromise scenario, not a single-call exploit.

---

### [MEDIUM] `parameters` has no key allow-list — director-facing spec drift

**File:** `contracts/PWMGovernance.sol:17, 117-118, 148-151`
**Function:** `executeProposal` (writes `parameters[p.key]`), `getParameter`
**Description:** The review spec ("Specific check D, item 5") explicitly states: "bytes32 parameter key allow-list prevents arbitrary writes." The code has no such allow-list — any `bytes32` key may be proposed and, on 3-of-5 + 48h, written into `parameters`. There is no `mapping(bytes32 => bool) public allowedKeys` and no `require(allowedKeys[p.key])` gate.
**Impact:** Spec drift, not direct theft. Because `parameters` is a `mapping(bytes32 => uint256)`, writing an "unknown" key cannot cause storage collisions with other state vars (mappings hash their keys into a disjoint slot space). The risk is governance-process: consumers of PWMGovernance (other contracts that read `getParameter(SOME_KEY)`) implicitly trust the allow-list documentation. If a future consumer naively reads `parameters[<arbitrary key>]` for a critical decision and a malicious 3-of-5 has pre-poisoned that key, behavior diverges from the spec.
**Reproducibility:** Any 3-of-5 collusion can write to any 256-bit key. The lack of allow-list means there is no on-chain record of which keys were ever sanctioned. Off-chain governance docs become the only source of truth.
**Recommendation:**
```diff
+ mapping(bytes32 => bool) public allowedKey;
+ bytes32 public constant KEY_STAKE_AMOUNT_L1     = keccak256("STAKE_AMOUNT_L1");
+ bytes32 public constant KEY_STAKE_AMOUNT_L2     = keccak256("STAKE_AMOUNT_L2");
+ // ... etc for every sanctioned key
+ constructor(address[NUM_FOUNDERS] memory _founders) {
+     // ... existing founder loop ...
+     allowedKey[KEY_STAKE_AMOUNT_L1] = true;
+     allowedKey[KEY_STAKE_AMOUNT_L2] = true;
+     // ... etc
+     allowedKey[keccak256("ACTIVATE_DAO")] = true;
+ }
  function proposeParameter(bytes32 key, uint256 value) external onlyFounder multisigActive returns (uint256 id) {
+     require(allowedKey[key], "PWMGovernance: key not whitelisted");
      // ... rest unchanged
  }
```
Alternatively the spec line should be amended to say "no allow-list; off-chain governance docs are authoritative."
**Soft-launch cap mitigation:** N/A — this is a spec-vs-code conformance gap.

---

### [LOW] `executeProposal` does not exclude the `ACTIVATE_DAO` key — can permanently brick a DAO-activation proposal

**File:** `contracts/PWMGovernance.sol:110-120` vs `135-146`
**Function:** `executeProposal`, `activateDAO`
**Description:** Both `executeProposal` and `activateDAO` operate on the same `proposals[id]` storage. `activateDAO` requires `p.key == keccak256("ACTIVATE_DAO")`. `executeProposal` has no such restriction. So a proposal whose key is `keccak256("ACTIVATE_DAO")` can be executed via the wrong path: `executeProposal` sets `p.executed = true` and writes `parameters[keccak256("ACTIVATE_DAO")] = p.value`, but does NOT set `daoActivated = true`. Subsequently, `activateDAO(id)` reverts because `!p.executed` is now false.
**Impact:** A single founder (call them malicious or accidental) can grief by calling `executeProposal(id)` on a fully-approved ACTIVATE_DAO proposal, burning that proposal id. Founders must then re-propose and wait another 48h. No funds at risk.
**Reproducibility:** (1) Founder A proposes `proposeParameter(keccak256("ACTIVATE_DAO"), 0)`. (2) 3 approvals collected, 48h elapses. (3) Any founder calls `executeProposal(id)` instead of `activateDAO(id)`. (4) Proposal is consumed; DAO not activated. Founders must restart the process.
**Recommendation:** Add to `executeProposal`:
```diff
+   require(p.key != keccak256("ACTIVATE_DAO"), "PWMGovernance: use activateDAO()");
```
Cheap, defensive.
**Soft-launch cap mitigation:** N/A — pure griefing, no fund impact.

---

### [LOW] PWMToken's `Ownable` role is unused — vestigial but accepted attack surface

**File:** `contracts/PWMToken.sol:6, 27, 32-40`
**Function:** Constructor (sets `initialOwner` via `Ownable(initialOwner)`)
**Description:** `PWMToken` inherits `Ownable`, but the contract defines NO `onlyOwner`-gated functions of its own. The 21M supply is fully minted in the constructor, and `ERC20Capped._update` enforces the cap on every transfer. So `owner` only has the OZ-default `transferOwnership` and `renounceOwnership` powers — both of which do nothing useful here because there's no privileged action to gate. The `initialOwner` parameter is, in effect, dead weight.
**Impact:** No direct exploit. But: (a) reviewers and integrators may mistakenly believe the owner has some recovery/mint power and act accordingly off-chain. (b) The owner key remains a target with no upside. (c) `renounceOwnership` is callable; if invoked it's harmless but emits a confusing event.
**Reproducibility:** Inspection only. Anyone reading the contract may assume `owner` can mint or pause; they cannot.
**Recommendation:** Either (a) remove `Ownable` inheritance entirely and drop the `initialOwner` constructor argument, or (b) immediately call `renounceOwnership()` in the constructor after `_mint`. Option (a) is cleaner:
```diff
- import {Ownable} from "@openzeppelin/contracts/access/Ownable.sol";
  ...
- contract PWMToken is ERC20, ERC20Capped, Ownable {
+ contract PWMToken is ERC20, ERC20Capped {
      ...
-     constructor(address initialHolder, address initialOwner)
+     constructor(address initialHolder)
          ERC20("Physics World Model", "PWM")
          ERC20Capped(TOTAL_SUPPLY)
-         Ownable(initialOwner)
      { ... }
  }
```
The contract docstring already says "the practical contract behavior is non-inflationary after deployment" — making the code match the docs is good hygiene.
**Soft-launch cap mitigation:** N/A — no fund impact.

---

### [LOW] `cancelProposal` and `cancelFounderChange` are single-signer kill switches (spec confirms this is intended, but worth flagging the asymmetry)

**File:** `contracts/PWMGovernance.sol:123-129, 225-231`
**Function:** `cancelProposal`, `cancelFounderChange`
**Description:** Any single founder can unilaterally cancel any pending proposal. The spec ("Specific check D, item 3") explicitly states this is intended ("single-signer kill switch (intended)"), so this is documentation, not a bug. However, the asymmetry — 3-of-5 to ACT but 1-of-5 to BLOCK — means any compromised founder key can DoS all governance for the soft-launch window. The other 4 founders cannot override the cancellation (they must re-propose, which the same compromised key can then re-cancel, forever).
**Impact:** Indefinite governance DoS by a single compromised key. Cannot steal funds, but can permanently halt all parameter changes and the DAO activation until the compromised founder is rotated out.
**Reproducibility:** A compromised founder runs a script that calls `cancelProposal(id)` and `cancelFounderChange(id)` on every newly-created proposal. The other founders need to call `proposeFounderChange` to remove the compromised key — but the compromised key cancels that proposal too. **EXCEPTION**: `proposeFounderChange` auto-approves (1 vote at creation), `approveFounderChange` adds 2 more (3 votes total), and only then `cancelFounderChange` can still cancel it pre-execution. So the rotation IS blockable indefinitely unless the rotation proposal's `approveFounderChange` calls and `executeFounderChange` are batched in a way the canceller can't intercept. On L2 Base with ~2s blocks and mempool visibility, a determined canceller bot will always win.
**Recommendation:** This is an architectural tradeoff. To fix, change cancellation to require 2-of-5 (or N-of-5 where N > 1), or specifically prevent the proposer of a pending founder-change from cancelling it (so a compromised founder can't kill their own rotation-out). At minimum, document the live-ness risk explicitly. Suggested:
```diff
- function cancelFounderChange(uint256 id) external onlyFounder multisigActive {
+ function cancelFounderChange(uint256 id) external onlyFounder multisigActive {
      FounderChange storage p = founderChangeProposals[id];
+     // A compromised founder cannot cancel their own rotation-out
+     require(founders[p.slot] != msg.sender, "PWMGovernance: cannot cancel own rotation");
      require(p.proposedAt != 0,            "PWMGovernance: unknown proposal");
```
Note: this only fixes the rotation case, not general parameter DoS. Full fix needs N-of-5 cancellation.
**Soft-launch cap mitigation:** N/A — affects governance liveness, not funds. During soft-launch the system runs with hardcoded caps, so parameter-DoS is even less impactful for 30 days.

---

### [LOW] Vesting accepts post-deployment deposits and back-vests them retroactively

**File:** `contracts/PWMVesting.sol:77-90`
**Function:** `_vestedAt`, `totalAllocation`, `releasable`
**Description:** `totalAllocation()` returns `balanceOf(this) + released`. So the "total" used in the linear vesting formula `(total * (timestamp - start)) / duration` is recomputed on every call from the current contract balance plus what's already been released. If anyone transfers additional PWM to the vesting contract AFTER `start`, those tokens are treated as if they had been there from `start` and are immediately back-vested up to `(timestamp - start) / duration` of the new total.
**Impact:** Intentional behavior for a one-time initial deposit, but if used iteratively (multiple top-ups) the schedule becomes confusing and front-runnable. Specifically: an attacker who learns a large top-up is incoming can call `release()` immediately before to lock in the pre-top-up vested fraction, then the post-top-up balance vests proportionally larger going forward. Actually that's NOT exploitable — `released` is per-beneficiary cumulative; the beneficiary gets the same total either way. The real concern: someone donating tokens to the contract gives them away forever — which is a feature, but worth surfacing.
**Reproducibility:** (1) Deploy vesting with 100k PWM at t=start. (2) At t=start+30 days (well into cliff), donor sends 100k more PWM to contract. (3) At t=cliff (12 months), `releasable()` = `(200k * 365days / 1460days) - 0` = 50k PWM, not 25k. Beneficiary receives the donor's tokens fully back-vested up to the cliff.
**Recommendation:** If the protocol expects top-ups: this is fine, document it. If only the initial deposit is intended: snapshot the total in storage:
```diff
+ uint256 public totalDeposit;  // set in constructor or first deposit
- function totalAllocation() public view returns (uint256) {
-     return token.balanceOf(address(this)) + released;
- }
+ function totalAllocation() public view returns (uint256) {
+     return totalDeposit;
+ }
```
With an explicit `function deposit(uint256 amount)` that increments `totalDeposit` and pulls tokens. This makes top-ups explicit and removes the surprise back-vest.
**Soft-launch cap mitigation:** N/A — affects vesting math only, not main protocol funds.

---

### [INFO] PWMToken's docstring claims `mint()` exists; contract has none

**File:** `contracts/PWMToken.sol:13-15`
**Description:** Doc comment says "After that, the owner can call mint() only up to the cap." There is no `mint()` function defined — only the constructor `_mint` call. The cap is reached at deploy so a mint() would always revert, but the missing function is still a docs/code mismatch.
**Recommendation:** Remove the misleading sentence from the docstring, or add an explicit `function mint(...) external onlyOwner { _mint(...); }` that always reverts at the cap (documentation by code).

---

### [INFO] PWMVesting beneficiary is a contract that can't accept tokens → stuck

**File:** `contracts/PWMVesting.sol:25-55`
**Description:** `beneficiary` is `immutable`. If set to a contract that lacks a way to forward / sweep ERC20 (e.g., a Gnosis Safe that's been left to misconfigure), funds released to it can become permanently inaccessible. The PWMVesting contract itself cannot redirect. Standard tradeoff for "cryptographically unstoppable" vesting.
**Recommendation:** Document explicitly in deploy runbook: "beneficiary MUST be a Safe or EOA capable of receiving and forwarding ERC20." Consider a `pendingBeneficiary` / `acceptBeneficiary` 2-step pattern if the team ever wants migration capability (currently disallowed by design).

---

### [INFO] PWMVesting trusts `token` to be a non-malicious ERC20

**File:** `contracts/PWMVesting.sol:28, 67-90`
**Description:** `releasable()` is called by `release()` BEFORE the state update (`released += amount`). Within `releasable()`, `totalAllocation()` calls `token.balanceOf(address(this))`. If `token` were malicious, it could re-enter on the balanceOf call (since IERC20 has no view-marker enforcement at runtime). However, the constructor cannot validate that token is OZ ERC20; it's an immutable address chosen by the deployer. In practice the team will set token to PWMToken, which is non-malicious. Risk only materializes if the constructor argument is wrong.
**Recommendation:** Either reorder `release()` to compute `releasable()` ONLY using stored state (impossible without a stored total), or add a `nonReentrant` modifier as belt-and-suspenders. Since `token` is set by trusted deployer to PWMToken, this is INFO. For completeness:
```diff
+ import {ReentrancyGuard} from "@openzeppelin/contracts/utils/ReentrancyGuard.sol";
- contract PWMVesting {
+ contract PWMVesting is ReentrancyGuard {
  ...
- function release() external {
+ function release() external nonReentrant {
```

---

### [INFO] Genesis-allocation amounts in PWMToken docstring (lines 19-22) sum correctly but are not enforced on-chain

**File:** `contracts/PWMToken.sol:17-22`
**Description:** Doc lists 17_220_000 + 2_100_000 + 1_050_000 + 630_000 = 21_000_000 — correct. The constructor mints the full 21M to `initialHolder`; the four sub-allocations are executed off-chain by the multisig. There is no on-chain check that the multisig actually does this. If the initial-holder multisig retains all 21M instead of distributing, no on-chain consumer can object. This is by design (simple constructor) but worth noting for the deploy runbook.
**Recommendation:** Have the deploy runbook include the four expected post-genesis Transfer events in addresses.json with mempool-watch verification. No code change.

---

## Confidence

**Deeply reviewed (every line, every branch, every storage transition):**
- `PWMToken.sol` — all 49 lines. Constructor mint logic and `_update` override traced through OZ v5 inheritance chain.
- `PWMGovernance.sol` — all 232 lines. Both proposal types (parameter, founder-change) and all four lifecycle states (propose/approve/execute/cancel) traced. Activate-DAO interaction with executeProposal verified. Founder-rotation race conditions analyzed (same-slot, mid-flight, oldAddr==newAddr edge case).
- `PWMVesting.sol` — all 91 lines. Cliff/linear math traced for boundary cases (t<cliff, t=cliff, t=cliff+1, t=duration-1, t>=duration). uint64 overflow analyzed for plausible (start, duration) values.

**Cross-checked against OpenZeppelin v5 source for:** `ERC20`, `ERC20Capped._update`, `Ownable` constructor, `SafeERC20.safeTransfer`. No regressions in OZ v5.0 / v5.1.

**Did NOT compile or run tests** — I did not execute Hardhat / Foundry. Findings are static-analysis only, based on Solidity 0.8.24 semantics. If team has time, the recommended diffs in MEDIUM-1 (timelock-from-threshold) and MEDIUM-2 (live-approval recount) should be unit-tested.

## What I did NOT check

- **PWMMinting, PWMStakingERC20, PWMRewardERC20, PWMCertificate, PWMTreasuryERC20, PWMRegistry, PWMFaucet** — out of A1 scope; reviewed by A2/A3/A4.
- **Gas griefing / MEV ordering** — out of A1 scope; A5 covers.
- **Front-running of `approveProposal` / `cancelProposal` sequencing on Base** — out of A1 scope (A5).
- **Cross-contract reentrancy via PWMToken hooks** — PWMToken has no transfer hooks (no ERC777, no callbacks); only `ERC20Capped._update` which is internal and pure. Out of concern.
- **Storage-layout collision with future upgradeable proxy** — these three contracts are NOT upgradeable (no proxy pattern, no `__gap`); standard storage-layout rules apply. Out of concern unless team plans to retrofit upgradeability.
- **Signature replay** — neither contract uses signatures; all auth is `msg.sender` via `isFounder` map. N/A.
- **OZ v5 dependency version pinning in package.json** — not in scope; deploy-team task.
- **Off-chain genesis distribution from `initialHolder`** — see INFO finding; runbook concern not contract concern.
- **`PWMRegistry`-style replay / id collision** — N/A for these three contracts.

## Bottom line for deploy

No CRITICAL or HIGH findings. The three MEDIUM findings are real but bounded:
1. **Timelock-from-proposedAt** — mitigated by the fact that 3-of-5 keys would need to coordinate the late-approval rush; honest founders should be running cancel-bots during soft-launch.
2. **Stale approvals from rotated founders** — requires both a compromise AND a planted proposal; can be mitigated by cancelling all pending proposals before executing any rotation.
3. **No parameter-key allow-list** — spec-drift; consumers all read specific keys, so unknown writes are inert. Document or implement.

The LOW and INFO findings are quality-of-life and runbook concerns. Soft-launch caps (paused minting, paused treasury, $1k staking cap) do not directly interact with these contracts but bound any downstream blast radius. **A1 says: deploy is safe to proceed pending the team's review of the three MEDIUM items.**
