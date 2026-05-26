# A1-v2 Security Re-Review — PWMToken, PWMGovernance, PWMVesting

**Date:** 2026-05-18 (second pass)
**Reviewer:** Claude Opus 4.7 (Agent A1-v2)
**Commit:** `203df847` on `release/d9-soft-launch-2026-05-18`
**Scope:** Same 3 contracts as A1 first pass
  - `contracts/PWMToken.sol` (49 LoC)
  - `contracts/PWMGovernance.sol` (232 LoC)
  - `contracts/PWMVesting.sol` (91 LoC)
**Patch context:** Patches landed in OTHER contracts (PWMCertificate, PWMMintingERC20, PWMTreasuryERC20, PWMRewardERC20, PWMStakingERC20). None of my 3 contracts were modified in `203df847`. Re-review must verify (a) no regression, (b) cross-contract impact of the new governance privileges, (c) anything a second pass might catch.

---

## Summary

| Severity | Count | Δ from A1 v1 | Blocks deploy? |
|---|---|---|---|
| CRITICAL | 1 | **+1 (NEW)** | **YES** |
| HIGH     | 0 | 0  | NO |
| MEDIUM   | 3 | 0  | NO |
| LOW      | 4 | 0  | NO |
| INFO     | 4 | 0  | NO |

**Headline change vs. v1:** ONE new CRITICAL surfaced on second-pass cross-contract inspection. **PWMGovernance has no on-chain mechanism to call the new `setX` functions added by the patch.** Once the deploy script hands off `governance` on the 5 sibling ERC20 contracts to the PWMGovernance address (`deploy/erc20.js:243-248`), ALL of the following become permanently unreachable:

  - `PWMMintingERC20.setMintingPaused` (the deploy-time pause cannot be lifted)
  - `PWMTreasuryERC20.setTransfersPaused` (treasury can never be unfrozen)
  - `PWMCertificate.setApprovedSubmitter` (no submitter can ever be approved)
  - `PWMCertificate.setSubmissionPermissionless` (cannot transition to permissionless)
  - `PWMRewardERC20.setMaxBenchmarkPoolWei`, `depositBounty`
  - `PWMStakingERC20.setMaxTotalStakeWei`, `setStakeAmount`
  - `PWMMintingERC20.setDelta`, `setPromotion`, `registerBenchmark`, `setBenchmarkRho`, `removeBenchmark`
  - Every `setGovernance`, `setCertificate`, `setReward`, `setMinting`, `setStaking`, `setTreasury`, `setRegistry` (i.e., **the governance address itself is forever locked to PWMGovernance**, which cannot use it)

This is the inverse problem to the soft-launch caps being correctly baked in: **the soft-launch posture is locked in *too well*** — there is no on-chain key that can ever turn the pauses off, raise the caps, promote a principle, register a benchmark, approve a submitter, or rotate any wiring.

The 3 prior MEDIUM findings remain valid; the 4 LOW and 4 INFO remain valid. No false positives identified.

---

## Findings

### [CRITICAL] PWMGovernance has no `execute(target, data)` primitive — all post-patch governance setters are unreachable after deploy handoff

**File:** `contracts/PWMGovernance.sol` (entire contract) vs. `deploy/erc20.js:243-248`
**Function:** Whole contract; specifically the absence of an arbitrary-call mechanism.

**Description:**

PWMGovernance's only state-changing primitive is `executeProposal(uint256 id)`, which on success runs exactly one operation:

```solidity
parameters[p.key] = p.value;
emit ProposalExecuted(id, p.key, p.value);
```

That is, it writes a (key, value) tuple into an internal `mapping(bytes32 => uint256) public parameters`. It does **not**:
  - call any external contract;
  - execute arbitrary calldata;
  - construct a transaction;
  - emit anything that an off-chain bot could forward as `msg.sender == governance`.

Meanwhile, every sibling protocol contract uses the pattern:

```solidity
modifier onlyGovernance() { require(msg.sender == governance, "...: not governance"); _; }
function setX(...) external onlyGovernance { ... }
```

— where `governance` is a plain `address`. After `deploy/erc20.js` calls `setGovernance(govAddr)` on each of the 5 sibling contracts (lines 243-248), `msg.sender == governance` can **only** be satisfied by the PWMGovernance contract itself making an external call. Because PWMGovernance has no such primitive, the satisfying caller does not exist.

I cross-checked all references: `grep -rn "PWMGovernance\|IPWMGovernance\|getParameter" contracts/` returns NOTHING outside of `PWMGovernance.sol` itself. No sibling contract reads `getParameter(key)`. So the `parameters` mapping is a write-only logbook with no on-chain consumers.

**What this means in practice:**

After the deploy script completes (assuming it runs without `PWM_SKIP_GOVERNANCE_HANDOFF=1`), the protocol is frozen in the soft-launch posture **forever**:

  1. `minting.mintingPaused = true` cannot be lifted → the 17.22M PWM emission pool is permanently locked. The whole purpose of PWMMintingERC20 (epoch emission) is bricked.
  2. `treasury.transfersPaused = true` cannot be lifted → adversarial bounties can never pay out.
  3. `certificate.submissionPermissionless = false` cannot be flipped → no certificate can ever be submitted (no submitter has been approved, and no submitter can ever be approved without `setApprovedSubmitter`, which is unreachable).
  4. The L4 reward economy never starts; the protocol is a museum-piece.
  5. Even worse: if a CRITICAL bug were found in any of these contracts, governance has **no on-chain remediation path**. They cannot pause-then-fix; they cannot rotate any address; they cannot do anything. The recovery option is "redeploy the entire protocol from scratch."

**Impact:**

  - **Liveness loss across the entire protocol.** The patch that made soft-launch caps enforceable simultaneously removed the only path to ever unmake them.
  - **17.22M PWM permanently locked in PWMMintingERC20.** That is 82% of the supply.
  - **No incident-response capability whatsoever.** Even the founder-rotation that PWMGovernance does support (executeFounderChange) only updates PWMGovernance's own internal `founders[]` — it doesn't change anything in the sibling contracts.
  - **The 48h timelock and 3-of-5 multisig are reduced to symbolic value** — they can vote on proposals, but the vote winning has no on-chain consequence beyond writing a key/value the rest of the protocol ignores.

This is exactly the kind of issue a second pass should catch and the first pass would not: A1-v1 reviewed PWMGovernance in isolation and confirmed its internal 3-of-5 + 48h logic is sound. It is — but the contract is a soundly-implemented null oracle.

**Reproducibility:**

  1. Director runs `npx hardhat run deploy/erc20.js --network base` per the deploy runbook.
  2. The script wires `setGovernance(govAddr)` on staking, reward, certificate, minting, treasury (lines 243-248).
  3. Director attempts to lift the soft-launch caps on day 31 by going through the multisig.
  4. Founders propose `proposeParameter(keccak256("UNPAUSE_MINTING"), 1)`. After 3-of-5 + 48h, `executeProposal` writes `parameters[keccak256("UNPAUSE_MINTING")] = 1`. **Nothing in `PWMMintingERC20` reads that key.** The minting contract continues to revert in `mintFor` because `mintingPaused == true` and `setMintingPaused` is callable only by `governance == PWMGovernance address`, which has no way to call it.
  5. Founders try `proposeFounderChange`, succeed in rotating internal founders — irrelevant; sibling contracts don't read founders.
  6. No on-chain action lifts the pause. Protocol stays frozen.

**Recommendation:**

The minimal viable fix is to add an `execute(address target, bytes calldata data)` style primitive gated by the same 3-of-5 + 48h timelock and re-use the existing `Proposal` shape. Concretely (sketch — must be unit-tested before merging):

```solidity
struct ExecProposal {
    address target;
    bytes   data;
    uint256 proposedAt;
    uint8   approvals;
    bool    executed;
    bool    cancelled;
}
mapping(uint256 => ExecProposal) public execProposals;
mapping(uint256 => mapping(address => bool)) public execApproved;
uint256 public nextExecProposalId;

function proposeExec(address target, bytes calldata data) external onlyFounder multisigActive returns (uint256 id) {
    require(target != address(0), "PWMGovernance: zero target");
    id = nextExecProposalId++;
    execProposals[id] = ExecProposal({
        target: target, data: data,
        proposedAt: block.timestamp, approvals: 1,
        executed: false, cancelled: false
    });
    execApproved[id][msg.sender] = true;
    // emit ...
}

function approveExec(uint256 id) external onlyFounder multisigActive { /* mirrors approveProposal */ }
function cancelExec(uint256 id)  external onlyFounder multisigActive { /* mirrors cancelProposal */ }

function executeExec(uint256 id) external onlyFounder multisigActive {
    ExecProposal storage p = execProposals[id];
    require(p.proposedAt != 0, "unknown");
    require(!p.executed && !p.cancelled, "finalised");
    require(p.approvals >= REQUIRED_APPROVALS, "insufficient approvals");
    require(block.timestamp >= p.proposedAt + TIME_LOCK, "timelock not elapsed");
    p.executed = true;
    (bool ok, bytes memory ret) = p.target.call(p.data);
    require(ok, _revertReason(ret));
    emit ExecExecuted(id, p.target, p.data);
}
```

Considerations:
  - Should **not** allow target = address(this) (re-entrancy into own governance state is dangerous).
  - May want to block calls to PWMToken (no need; supply is fixed) and especially to the registry/certificate `setGovernance` to prevent governance from accidentally orphaning itself.
  - Should reuse same MEDIUM-2 live-approval recount (see prior findings) for `approvals` accounting.

**Alternative fixes:**

  1. **Use a Gnosis Safe (3-of-5) as the `governance` address** on the sibling contracts, and demote PWMGovernance to a logbook for parameters that protocol consumers actually read. This is industry standard (Compound, Aave, ENS, Uniswap all do this). It requires the deploy script to NOT call `setGovernance(govAddr)` and instead point to the Safe.

  2. **Make the sibling contracts read `PWMGovernance.getParameter(key)` for everything currently behind `onlyGovernance`.** Heavy refactor; touches every sibling contract; orthogonal to soft-launch caps.

  3. **Delay the `setGovernance(govAddr)` calls** until the protocol is ready to truly handover (i.e., do soft-launch caps then handoff later when an `execute()` primitive exists). The deploy script as written does NOT support this — handoff is gated by `PWM_SKIP_GOVERNANCE_HANDOFF=1`, but that just leaves the deployer EOA as admin, which is worse than a Safe.

I recommend **Alternative 1 (Gnosis Safe)** because it's the minimum-code-change path and the team can ship `proposeExec` later as a v2 governance.

**Soft-launch cap mitigation:**

NO. This is the exact opposite of cap mitigation: the soft-launch caps cannot be removed. The protocol cannot enter its post-soft-launch operating state. Director **must** verify before mainnet deploy that either (a) `governance` is NOT set to the PWMGovernance address but to a Safe, or (b) the missing `execute` primitive is added to PWMGovernance.

**Severity:** CRITICAL — protocol bricked at month-2; 17M PWM permanently locked; no remediation path.

---

### Prior findings re-verification

Below, each of the 11 prior A1 findings is re-evaluated against the patched commit `203df847`. The patches did NOT touch `PWMToken`, `PWMGovernance`, or `PWMVesting`, so the analysis below is largely confirmation rather than re-derivation.

---

### [MEDIUM] Timelock measured from proposal-time, not from approval-threshold reached — **STILL VALID**

**File:** `contracts/PWMGovernance.sol:110-120, 207-221, 135-146`

**Re-review:** Lines unchanged. `executeProposal`, `executeFounderChange`, and `activateDAO` all still check `block.timestamp >= p.proposedAt + TIME_LOCK`, where `proposedAt` is set once at proposal creation and never updated on later approvals. The reproducibility scenario from v1 (proposal sits at 1 approval for 47h59m, jumps to 3 approvals in the last block, executes one block later) still works.

**Cross-pollination check:** The new patch added 5+ governance privileges in sibling contracts (`setMintingPaused`, `setTransfersPaused`, `setApprovedSubmitter`, `setSubmissionPermissionless`, `setMaxBenchmarkPoolWei`, `setMaxTotalStakeWei`, `setStakeAmount`, etc.). HOWEVER, the new CRITICAL above makes this point moot **today**: those setters are unreachable, so the timelock issue cannot manifest through them. If the new CRITICAL is fixed via Alternative 2 (proposeExec), then the late-approval rush problem would apply to arbitrary cross-contract calls — meaning a coordinated 3-of-5 could rush a `setApprovedSubmitter(attacker, true)` exec proposal, then immediately submit a self-dealing cert. This would **elevate the severity to HIGH** once an `execute()` primitive lands.

**Recommendation:** Same as v1 — switch to `thresholdReachedAt`-based timelock. Must be in the v2 governance contract.

---

### [MEDIUM] Revoked founder's prior approvals still count — **STILL VALID**

**File:** `contracts/PWMGovernance.sol:99-107, 194-202, 215-219`

**Re-review:** Lines unchanged. `executeFounderChange` still does:
```solidity
isFounder[oldAddr] = false;
isFounder[p.newAddr] = true;
```
without clearing `approved[*][oldAddr]` or `approvedFounderChange[*][oldAddr]`. The execution check still reads `p.approvals >= REQUIRED_APPROVALS` not a live recount.

**Cross-pollination check:** With the new patches inert (per CRITICAL above), the impact is theoretical for now. Once governance has cross-contract reach, this becomes more pressing: a rotated-out compromised founder's pre-approved `proposeExec(target=PWMCertificate, data=setApprovedSubmitter(attacker, true))` would still carry their vote forward, requiring only 2 (instead of 3) collaborating remaining founders to execute. **The cross-contract reach magnifies this finding's blast radius from "spec drift" to "fund theft enabler."**

**Recommendation:** Same as v1 — implement `_liveApprovals(id)` recount over current `founders[]`.

---

### [MEDIUM] `parameters` has no key allow-list — **STILL VALID, AND ELEVATED CONTEXT**

**File:** `contracts/PWMGovernance.sol:17, 117-118, 148-151`

**Re-review:** Lines unchanged. Any `bytes32` key may still be proposed and written.

**Cross-pollination check:** Per the new CRITICAL above, **no other contract reads `parameters`**. So the practical exposure of an arbitrary-key write is 0 — it's an indexer/observability concern only. The original "consumers of getParameter might naively trust" concern is moot because there are no on-chain consumers.

This finding becomes **secondary to the new CRITICAL**. Once an `execute()` primitive is added, the allow-list concept should be reconsidered: if `execute(target, data)` is the right model, then `parameters` may be deletable entirely. If a hybrid model is used (parameters + execute), then an allow-list over both `bytes32 parameters key` and `address target` is appropriate.

**Recommendation:** Defer this finding to the v2 governance redesign. Decide whether `parameters` stays or is replaced by `execute(target, data)`. If it stays, add the allow-list per v1 recommendation.

---

### [LOW] `executeProposal` does not exclude the `ACTIVATE_DAO` key — **STILL VALID**

**File:** `contracts/PWMGovernance.sol:110-120, 135-146`

**Re-review:** Lines unchanged. A proposal with `key == keccak256("ACTIVATE_DAO")` can still be consumed by the wrong path (`executeProposal` instead of `activateDAO`), griefing the activation flow.

**Cross-pollination check:** None — `ACTIVATE_DAO` is an internal PWMGovernance concept; no sibling contract reads it.

**Recommendation:** Same as v1 — add `require(p.key != keccak256("ACTIVATE_DAO"), ...)` to `executeProposal`.

---

### [LOW] PWMToken's `Ownable` role is unused — **STILL VALID**

**File:** `contracts/PWMToken.sol:6, 27, 32-40`

**Re-review:** Lines unchanged. `PWMToken` still inherits `Ownable` but exposes no `onlyOwner` functions. `initialOwner` remains a dead parameter.

**Cross-pollination check:** None — the patches did not touch the token. PWMToken is still a plain ERC20Capped with no callbacks; no new threat surface.

**Recommendation:** Same as v1 — drop `Ownable` inheritance or call `renounceOwnership()` in constructor. Strictly hygiene.

---

### [LOW] `cancelProposal` and `cancelFounderChange` are single-signer kill switches — **STILL VALID, AND ELEVATED CONTEXT**

**File:** `contracts/PWMGovernance.sol:123-129, 225-231`

**Re-review:** Lines unchanged. Any single founder can cancel any pending proposal.

**Cross-pollination check:** With the new CRITICAL, an attacker who compromises one founder key **cannot do much** today — there is no proposal whose execution affects the protocol meaningfully (proposals only write to a logbook). Once the new CRITICAL is fixed and PWMGovernance gains cross-contract reach (`proposeExec`), this finding becomes much more dangerous: a compromised single founder can indefinitely DoS all governance, including the very founder-rotation needed to evict them. The mitigation suggested in v1 (block proposer of a pending founder-change from cancelling it) becomes **required**, not optional.

**Recommendation:** Same as v1, with elevated urgency to be implemented in the v2 governance contract: require 2-of-5 to cancel, OR specifically block self-rotation cancellation by the rotated-out founder.

---

### [LOW] Vesting accepts post-deployment deposits and back-vests them — **STILL VALID**

**File:** `contracts/PWMVesting.sol:77-90`

**Re-review:** Lines unchanged. `totalAllocation()` still computes `balanceOf(this) + released`. Top-ups are back-vested.

**Cross-pollination check:** None — vesting has no interaction with governance, certificate, or staking. The 630_000 PWM allocation goes directly into the vesting contract per `deploy/erc20.js:224`.

**Note:** I rechecked the v1 finding's "front-runnable" analysis and confirm it's not an exploit — back-vesting is symmetric (the beneficiary gets the same total either way). The risk is purely a usability/spec-clarity concern.

**Recommendation:** Same as v1 — explicitly document or move to stored-total via `deposit()`.

---

### [INFO] PWMToken's docstring claims `mint()` exists — **STILL VALID**

**File:** `contracts/PWMToken.sol:13-15`

**Re-review:** Comment unchanged. No `mint()` function exists.

**Recommendation:** Same — fix the docstring.

---

### [INFO] PWMVesting beneficiary as a contract that can't accept tokens → stuck — **STILL VALID**

**File:** `contracts/PWMVesting.sol:25-55`

**Re-review:** Lines unchanged. `beneficiary` is still immutable. Burns funds if the beneficiary contract can't forward ERC20.

**Recommendation:** Same — document in deploy runbook.

---

### [INFO] PWMVesting trusts `token` to be non-malicious — **STILL VALID**

**File:** `contracts/PWMVesting.sol:28, 67-90`

**Re-review:** Lines unchanged. `releasable()` reads `token.balanceOf` before state update; PWMToken has no callbacks, so no exploit today.

**Recommendation:** Same — add `nonReentrant` as defense-in-depth.

---

### [INFO] Genesis-allocation sub-amounts not enforced on-chain — **STILL VALID**

**File:** `contracts/PWMToken.sol:17-22`

**Re-review:** Constructor unchanged. The deploy script (`deploy/erc20.js`) does perform the four distributions explicitly (lines 213-225), so the off-chain enforcement is in the deploy script. The on-chain constructor still cannot verify them.

**Cross-pollination check:** I verified the deploy script does the four transfers correctly:
  - `mintingAddr ← 17_220_000` (line 213)
  - `reserveAddr ← 2_100_000` (line 217)
  - `liquidityAddr ← 1_050_000` (line 221)
  - `vestingAddr ← 630_000` (line 224)
  
Plus a 210k testnet faucet seed. Mainnet should run with `faucetAddr == undefined` so that branch is skipped.

**Recommendation:** Same — verify on-chain via mempool watch during deploy. Confirmed already by A8 deploy-script audit per ABORT_DECISION doc.

---

## New issues found on second pass (besides the CRITICAL)

I traced every external/public function in all 3 contracts a second time. Beyond the CRITICAL above, I found **0 new findings**.

Specifically I rechecked:

  - **PWMToken constructor.** `_mint(initialHolder, TOTAL_SUPPLY)` is called once. `ERC20Capped` rejects future mints (none exist). `_update` correctly chains through both parents. No new issues.
  - **PWMToken transfer paths.** Plain ERC20 inheritance from OZ v5; no `_beforeTokenTransfer` hook; no callback surface. Transfers are pure storage updates. No new issues.
  - **PWMGovernance constructor.** Loops over 5 founders, requires non-zero and non-duplicate. The for-loop bound is `NUM_FOUNDERS = 5`; no gas-bomb risk. No new issues.
  - **PWMGovernance proposal lifecycle.** Both parameter and founder-change paths traced through propose → approve → execute → cancel. No new race conditions beyond v1.
  - **PWMGovernance `daoActivated` flag.** Once flipped, the `multisigActive` modifier reverts every state-changing function. The DAO replacement is "off-chain switchover" per docstring; no further on-chain logic. Confirmed: `daoActivated == true` permanently freezes the multisig.
  - **PWMVesting math edge cases.** Rechecked `timestamp == cliff - 1` (returns 0), `timestamp == cliff` (returns >0 if cliff < start+duration, else returns full total), `timestamp == start + duration` (returns full total), `timestamp == start + duration + 1` (returns full total). Division `(total * elapsed) / duration` rounds down — beneficiary receives slightly less than mathematically vested, residue stays in contract and is included in `totalAllocation()` on the next call. No off-by-one exploit.
  - **PWMVesting `released` accounting.** Increments by `releasable()` before transfer (CEI-correct in this case because PWMToken has no callbacks). No double-release.

---

## Cross-contract interactions affecting my 3 contracts

Beyond the CRITICAL, the only cross-contract paths that touch my contracts are:

  1. **PWMToken → PWMMintingERC20, PWMRewardERC20, PWMTreasuryERC20, PWMStakingERC20, PWMVesting** as the underlying ERC20. Plain ERC20 semantics; no callbacks. Verified.
  2. **PWMGovernance ← founders' EOAs**, exclusively. No external contract calls in. No external contract calls out (other than `parameters[key] = value` writes that no one reads — see CRITICAL).
  3. **PWMVesting ← PWMToken transfer at deploy**. One-time funding of 630k PWM. No further interaction.

There is NO scenario in which the new patches in A2/A3 contracts (PWMCertificate, PWMMintingERC20, PWMTreasuryERC20, PWMRewardERC20, PWMStakingERC20) create a malicious-proposal vector against my 3 contracts, BECAUSE the proposal mechanism cannot reach those contracts (per CRITICAL). If/when the CRITICAL is fixed via `proposeExec`, the existing MEDIUM findings (timelock, rotated approvals) will need to be re-evaluated as HIGH because their blast radius expands from "internal governance state" to "all 5 sibling contracts."

---

## Bottom line for deploy

**DEPLOY MUST BE PAUSED** until the CRITICAL is resolved. The first-pass A1 review missed this because it analyzed PWMGovernance in isolation. The patch in commit `203df847` is the trigger: it added on-chain pause/cap controls in sibling contracts that depend on `governance` being callable, but the deploy script then hands `governance` to a contract that **cannot make those calls**. The combination of these two correct-in-isolation changes produces a soft-bricked protocol.

**Recommended path forward (smallest blast radius):**

  1. **Do NOT call `setGovernance(govAddr)` on the 5 sibling contracts in the deploy script.** Instead, set `governance` to a Gnosis Safe (3-of-5 over the same founder keys). The Safe can transact to any of the 5 sibling contracts and so can exercise `setApprovedSubmitter`, `setMintingPaused`, etc.
  2. Keep PWMGovernance deployed as-is for now; it can serve as the on-chain proposal/approval logbook for off-chain Safe operators (founders propose in PWMGovernance, approve in PWMGovernance, then once 3-of-5 + 48h is reached, ONE founder executes the corresponding Safe transaction off-chain). This preserves the 48h timelock UX without changing any contract code.
  3. Defer the addition of `proposeExec(target, data)` to a v2 PWMGovernance shipped post-audit. At that point, also fix MEDIUM-1 (threshold-based timelock) and MEDIUM-2 (live-approval recount) — both of which become HIGH-severity in a contract that has cross-contract reach.

**Three prior MEDIUM findings remain valid but are not deploy-blockers** (they are bounded by the no-cross-contract-reach property created accidentally by the CRITICAL). After the CRITICAL is fixed, MEDIUM-1 and MEDIUM-2 should be reassessed at HIGH and addressed before any `proposeExec` ships.

**Four LOW + four INFO findings remain valid** with no severity change.

---

## Confidence

**Deeply re-reviewed** (every line, every storage transition, every external call site):
  - `PWMToken.sol` (49 lines)
  - `PWMGovernance.sol` (232 lines)
  - `PWMVesting.sol` (91 lines)

**Cross-referenced for cross-contract impact:**
  - `PWMMintingERC20.sol` — `setMintingPaused`, `setGovernance`, `mintingPaused` storage var (lines 38, 108-111)
  - `PWMTreasuryERC20.sol` — `setTransfersPaused`, `transfersPaused` storage var
  - `PWMCertificate.sol` — `setApprovedSubmitter`, `setSubmissionPermissionless` (lines 121-130)
  - `PWMRewardERC20.sol` — `depositBounty`, `setMaxBenchmarkPoolWei`
  - `PWMStakingERC20.sol` — `setMaxTotalStakeWei`, `setStakeAmount`
  - `deploy/erc20.js` — confirmed `setGovernance(govAddr)` handoff at lines 243-248
  - `A2_staking_reward_treasury_2026-05-18.md` — read in full; no cross-mediation findings via my 3 contracts.
  - `A3_minting_registry_certificate_2026-05-18.md` — read in full; C-1 mitigation depends on `setApprovedSubmitter` being callable, which the CRITICAL above shows is NOT the case.

**Did NOT execute** Hardhat or Foundry tests. The CRITICAL above is reproducible by static reading and by a 3-line Hardhat snippet (deploy, hand off, try `pwmGovernance.executeProposal(...)`; observe that `PWMMintingERC20.mintingPaused` stays true). Recommend the team write that integration test before any further deploy steps.

## What I did NOT check

Same exclusions as A1-v1. Specifically:
  - **A4 / A5 / A6 / A7 reports** — not yet written; I only had A1, A2, A3.
  - **Existence of off-chain Safe wiring** — not present in `deploy/erc20.js`. If a separate operations runbook exists that uses a Safe as `governance` (and the script lines 243-248 are dead/skipped on mainnet), the CRITICAL is moot. I could not find such a runbook in `deploy/`; recommend Director confirm.
  - **Whether `PWM_SKIP_GOVERNANCE_HANDOFF=1` is being used on mainnet** — if so, `governance` stays as the deployer EOA, which means the setters are callable but by an EOA, not a multisig. That is a DIFFERENT problem (single-key control of the protocol) and the v1 trust model assumed a real multisig. Director should confirm which path is being taken.
