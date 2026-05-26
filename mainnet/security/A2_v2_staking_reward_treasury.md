# A2-v2 Re-Review — PWMStakingERC20, PWMRewardERC20, PWMTreasuryERC20

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A2-v2)
**Commit:** `203df847` on `release/d9-soft-launch-2026-05-18`
**Scope:** 3 contracts (full re-read on patched code)
  - `contracts/PWMStakingERC20.sol` (185 lines)
  - `contracts/PWMRewardERC20.sol` (205 lines)
  - `contracts/PWMTreasuryERC20.sol` (97 lines)

**Patches verified in this pass:**
  1. `PWMTreasuryERC20` — `transfersPaused` storage + setter + check in `payAdversarialBounty`. (Was HIGH.)
  2. `PWMStakingERC20.setMaxTotalStakeWei` — `require(newMax > 0)`. (Was MED.)
  3. `PWMRewardERC20.setMaxBenchmarkPoolWei` — `require(newMax > 0)`. (Was MED.)
  4. `PWMRewardERC20.depositBounty` — added `onlyGovernance`. (Was MED.)

**Deploy-script verification:** `deploy/erc20.js:189-208` correctly calls
`setMaxTotalStakeWei(100 ether)`, `setMaxBenchmarkPoolWei(100 ether)`, and
`setTransfersPaused(true)` on the deployer's admin run. The construction-time
default (`transfersPaused == false`, `maxTotalStakeWei == 0`,
`maxBenchmarkPoolWei == 0`) does **not** match the soft-launch posture, but
the gap is closed inside the same deploy transaction sequence. No
third-party caller can reach `stake`, `payAdversarialBounty`, or
`depositBounty` in that window because only `governance` (= deployer at
that moment) holds the privileges that matter, and stake/seedBPool/depositMinting
have not yet been wired to a `staking`/`minting` address.

---

## Summary

| Severity   | v1 count | v2 count | Δ      |
|-----------:|:--------:|:--------:|:------:|
| CRITICAL   |    0     |    0     |   =    |
| HIGH       |    1     |    0     | −1     |
| MEDIUM     |    3     |    1     | −2     |
| LOW        |    6     |    6     |   =    |
| INFO       |    5     |    6     | +1     |

All four patches are correct and the HIGH + two of the three MEDs are
resolved. The remaining v1 MED (no global `paused` flag on
`PWMStakingERC20`) was not addressed by this patch round; it carries over.
One new MEDIUM is introduced by the `depositBounty → onlyGovernance` change:
it routes every bug-bounty top-up through the 3-of-5 + 48 h timelock, which
likely does not match the operational intent. One new INFO is added
covering the combined kill-switch surface (`setMintingPaused +
setTransfersPaused + setSubmissionPermissionless + setApprovedSubmitter`).

The patches did **not** regress any existing finding and did **not** break
any economic invariant. CEI ordering is unchanged; rounding and dust
behavior are unchanged; the `distribute` sum check
(`tkAmt = drawAmt − acAmt − cpAmt − l3Amt − l2Amt − l1Amt`) still preserves
the per-cert accounting identity exactly.

---

## Patch verification

### Patch 1 — PWMTreasuryERC20 transfer-pause (resolves v1 HIGH)

**Status:** RESOLVED.

`bool public transfersPaused` (line 25), `setTransfersPaused(bool)`
(lines 62-65), and the gate `require(!transfersPaused, ...)` at the top of
`payAdversarialBounty` (line 81) are all present and correctly composed.
The setter is `onlyGovernance`. The event `TransfersPausedUpdated` (line 31)
is emitted on every change.

**Walk-through:**
1. `payAdversarialBounty(principleId, winner, amount)` — first instruction
   after the `onlyGovernance` modifier is `require(!transfersPaused, …)`.
   With `transfersPaused == true` at deploy, every call reverts before
   touching `treasury[]` or transferring tokens. Correct.
2. With `transfersPaused == false`, the prior checks
   (`winner != 0`, `amount > 0`, `amount ≤ balance`, `amount*2 ≤ balance`)
   still execute and the 50 % cap is still enforced. No regression.
3. The deploy script (`deploy/erc20.js:204`) sets `transfersPaused = true`
   immediately after `setReward()`. The on-chain default at construction is
   `false`, but no external caller can reach `payAdversarialBounty` during
   that window because `governance == deployer` at that moment and the
   deployer does not exploit itself. Acceptable.

**Intentional non-gating of `receive15pct`:** confirmed by the in-code
comment at lines 21-24. Inflows are *not* gated by `transfersPaused`. This
is documented and is the right call for distribute's `forceApprove +
receive15pct` flow to keep working when the pause is enabled (see
[INFO-NEW-1] below).

---

### Patch 2 — PWMStakingERC20.setMaxTotalStakeWei zero-guard (resolves v1 MED)

**Status:** RESOLVED for the targeted scenario.

Line 92: `require(newMax > 0, "PWMStakingERC20: zero max disables cap; use governance proposal explicitly");`
The error string accurately describes the failure mode. Once governance has
set a non-zero cap, the cap cannot be silently disabled by passing 0.

**Residual gap (carries over from v1):** the guard only protects against
*post-set* zeroing. The constructor never initializes `maxTotalStakeWei`, so
the deploy-time default is `0` and the cap-check at line 109
(`if (cap != 0)`) short-circuits to "unlimited". The deploy script does
call `setMaxTotalStakeWei(100 ether)` (line 195 of `deploy/erc20.js`)
inside the same admin-run, but this window remains exploitable if any
external caller can reach `stake()` before that line executes. Today no
external caller can, because `pwmToken` allowances must be granted to the
staking contract address and that address only becomes known after deploy
completes. The window is therefore protected by social ordering only, not
by code. See [LOW-NEW-1].

---

### Patch 3 — PWMRewardERC20.setMaxBenchmarkPoolWei zero-guard (resolves v1 MED)

**Status:** RESOLVED for the targeted scenario.

Line 109 mirrors the staking patch. Identical analysis applies; identical
residual gap (constructor does not initialize, deploy script sets after).

---

### Patch 4 — PWMRewardERC20.depositBounty onlyGovernance (resolves v1 MED)

**Status:** RESOLVED for the v1 attack ("anyone can dump PWM into any
pool"), but the patch **over-corrects** and likely breaks the intended
bounty-funding flow. See [MED-NEW-1].

---

## Findings on patched code

### [MED-NEW-1] `depositBounty` is now timelock-gated; bug-bounty top-ups require 48 h governance proposals

**File:** `contracts/PWMRewardERC20.sol:128-131`
**Function:** `depositBounty(bytes32 benchmarkHash, uint256 amount)`
**Description:**
The v1 review correctly flagged that permissionless `depositBounty` lets
anyone push tokens into any pool. This patch gates the function with
`onlyGovernance`. That fixes the v1 issue but raises an operational concern:
`PWMGovernance` is a 3-of-5 multisig with a 48 h timelock (per A1's
governance review). The intended use case for `depositBounty` is to
top up a B-pool for a benchmark when an external sponsor (a sub-DAO, a
challenge prize donor, a project office that has won a contract requiring
adversarial bounties) wants to fund extra reward weight on a specific
benchmark hash. Routing every such deposit through the multisig:

  - couples bounty funding to the same timelock as parameter changes;
  - requires the multisig itself (or whatever address it delegates `msg.sender`
    to via a wrapper, which adds an extra trust hop) to hold and pre-approve
    the PWM being deposited;
  - prevents the obvious "sponsor sends PWM to a designated bounty wallet,
    that wallet calls `depositBounty`" pattern;
  - blocks the simplest sponsor-funded growth path entirely during the
    30-day soft-launch (governance can deposit, but every deposit takes
    48 h to clear).

**Impact:**
Loss of an operational flow. Not a fund-loss bug. The function now exists
but is impractical to use; bounty pools must instead grow via `seedBPool`
(only callable by the staking contract after a graduation) or via
`depositMinting` (only callable by the minting contract after an epoch).
Both alternative paths are upstream of distribute and not always
appropriate for "this specific benchmark needs more $".

**Reproducibility:**
Conceptual; depends on intended operational role of `depositBounty`. If the
function is in fact *only* meant to be called by governance — e.g., to
top up a bounty for an adversarial-defense round that the multisig itself
authorized — then the patch is correct as-is and this finding downgrades
to INFO. The contract source and CLAUDE.md do not document the intended
caller of `depositBounty`; the function's `kind == "B-pool-bounty"` label
in `_credit` is ambiguous.

**Recommendation:**
Two options:
  (a) Keep `onlyGovernance` and explicitly document that bounty top-ups
      are governance-managed (acknowledge the 48 h cadence).
  (b) Introduce a `mapping(address => bool) public approvedBountyFunders`
      and an `onlyGovernance` setter. `depositBounty` becomes
      `require(approvedBountyFunders[msg.sender], "...")`. Governance
      whitelists trusted sponsor addresses without paying the timelock per
      deposit. Mirrors the `approvedSubmitter` pattern in
      `PWMCertificate`.

The choice depends on whether bounty funding is a per-deposit governance
action or a per-funder one. Director should confirm the intent before
mainnet launch.

**Soft-launch cap mitigation:** YES — even if the function is locked out
during soft-launch, the protocol still functions; only the bounty-growth
optionality is lost.

---

### [MED-CARRYOVER] `PWMStakingERC20` still has no global pause flag

**File:** `contracts/PWMStakingERC20.sol:101-123`
**Function:** `stake`, `graduate`, `slashForChallenge`, `slashForFraud`

This v1 MED was not patched in this round and remains open. Recap:
incident-response inflexibility — the only way to halt new staking is to
crank `maxTotalStakeWei` down to a value below `totalActiveStakeWei + 1
ether`, which has the side-effect of locking out legitimate top-ups even
during a "fast freeze" response. A clean `setPaused(bool)` modifier on
`stake()` (resolution functions intentionally unpaused) is the
operationally correct posture.

The decision to defer this patch is consistent with v1's
"soft-launch cap mitigation: YES for steady-state" — loss is still bounded
by the 100 PWM cap during the 30-day window, and the incident-response
gap is a known and accepted cost. Flag for the post-soft-launch backlog.

---

### [LOW-NEW-1] Soft-launch caps are deploy-script-enforced, not constructor-enforced

**File:** `contracts/PWMStakingERC20.sol:63-71` (constructor),
         `contracts/PWMRewardERC20.sol:74-79` (constructor),
         `contracts/PWMTreasuryERC20.sol:43-48` (constructor)
**Description:**
None of the three constructors initialize the soft-launch state
(`maxTotalStakeWei`, `maxBenchmarkPoolWei`, `transfersPaused`). The deploy
script does set them, but a future mis-edit of the deploy script — or
a non-script deploy path (e.g., a forensic re-deploy via `hardhat console`
after an incident) — could yield a contract instance that is live on
mainnet with zero-cap and pause-off.

**Impact:**
Operational. The contract code does not encode its own safe defaults;
correctness depends on the deploy script staying in sync with the policy.

**Recommendation:**
Take constructor args for the genesis cap values and the genesis pause
posture:

```solidity
constructor(
    address token_,
    address initialGovernance,
    uint256 initialMaxBenchmarkPoolWei
) {
    ...
    maxBenchmarkPoolWei = initialMaxBenchmarkPoolWei;
    ...
}
```

For `PWMTreasuryERC20`, default `transfersPaused = true` in the constructor
body. The deploy script can still call the setter to change it, but the
contract is born safe.

**Soft-launch cap mitigation:** YES (deploy script is correct as written);
mitigation is operational.

---

### [INFO-NEW-1] Distribute does not pause when treasury is paused — and that is intentional

**File:** `contracts/PWMRewardERC20.sol:155-199`,
         `contracts/PWMTreasuryERC20.sol:69-74`
**Description:**
When `PWMTreasuryERC20.transfersPaused == true`, the only blocked path is
`payAdversarialBounty`. `receive15pct` is **not** blocked. Therefore
`PWMRewardERC20.distribute` continues to execute end-to-end during the
pause: it calls `treasury.receive15pct(principleId, tkAmt)` (line 194), the
15 % cut is credited to `treasury[principleId]`, and the
`treasury[principleId]` balance grows on every settled cert.

Funds are not lost — they are accumulated. Once governance unpauses, the
balance is available for `payAdversarialBounty`. This is the documented
intent (see lines 21-24 of `PWMTreasuryERC20.sol`).

**Cross-contract trust assumption:** `PWMRewardERC20.distribute` blindly
calls `treasury.receive15pct`. If a future upgrade adds pause-checking to
`receive15pct`, `distribute` will start reverting at line 194 mid-settlement,
which would *unsettle* the cert (because `settled[certHash] = true` is at
line 163, before the call, distribute would have set settled = true and
then reverted — but a revert rolls back the storage write too, so this is
actually safe). The current design is correct; flagging only because the
mental model is non-obvious and a future maintainer might "fix" the pause
asymmetry without realizing it would brick `distribute`.

**Impact:** None today. Documentation/mental-model concern.

**Recommendation:**
Add an inline `/// @dev` note on `receive15pct` explaining that this path
is intentionally *not* pause-gated so that `distribute` remains atomic.
Alternatively, if future policy wants distribute to short-circuit the 15 %
when paused, add a `treasury.transfersPaused()` view check inside
`distribute` and route the 15 % to rollover instead. Current design is
acceptable.

**Soft-launch cap mitigation:** YES.

---

### [INFO-NEW-2] Combined governance kill-switch surface

**File:** `contracts/PWMTreasuryERC20.sol:62-65` (setTransfersPaused),
         `contracts/PWMMintingERC20.sol:108-110` (setMintingPaused, out of A2 scope but composes),
         `contracts/PWMCertificate.sol` (setSubmissionPermissionless, setApprovedSubmitter — out of A2 scope but composes),
         `contracts/PWMStakingERC20.sol:91-95` (setMaxTotalStakeWei),
         `contracts/PWMRewardERC20.sol:108-112` (setMaxBenchmarkPoolWei)

**Description:**
The patches in this round (+ the prior `setMintingPaused`,
`setSubmissionPermissionless`, `setApprovedSubmitter` patches in sibling
contracts) collectively give the 3-of-5 governance multisig the power to:

  1. Halt PWM emission (`setMintingPaused(true)`).
  2. Halt T_k payouts (`setTransfersPaused(true)`).
  3. Halt new staking (set `maxTotalStakeWei` to a value below the current
     `totalActiveStakeWei`).
  4. Halt new benchmark-pool deposits (set `maxBenchmarkPoolWei` to a value
     below the current per-benchmark balance).
  5. Halt new cert submissions (`setSubmissionPermissionless(false)`
     + no approved submitters).

Plus the historically-existing power to swap any peer-contract address
via `setReward`, `setCertificate`, `setStaking`, `setTreasury`,
`setMinting`, `setGovernance`. Together, a compromised 3-of-5 can put the
entire protocol into a state where:

  - no new economic activity occurs,
  - existing pools and T_k balances are frozen in place,
  - the only way out is `setGovernance` to a new address — also requiring
    the same compromised 3-of-5 keys.

**Impact:**
This is a deliberate soft-launch design choice; the kill-switch power is
the point. The concern is asymmetric: governance can brick the protocol
faster than it can revive it (any "unbrick" path requires the same 3-of-5
threshold + 48 h timelock). If governance keys are *lost* (not
compromised), the protocol is permanently frozen.

The risk is mitigated by:
  - the social posture of the 3-of-5 founders;
  - the 48 h timelock on parameter changes (gives time for community
    response before a brick lands);
  - the planned DAO activation post-soft-launch (`activateDAO()` in
    PWMGovernance), which shifts trust to contribution-weighted voting.

**Recommendation:**
This is a governance-design question, not a contract-code question. The
contracts are correct as written; the kill-switch is justified for the
30-day soft-launch. Two operational mitigations worth considering:

  (1) Document the recovery RACI for a "founder key lost" scenario:
      under what circumstances and on what timeline can the remaining
      governance signers rotate `setGovernance` to a new address?
  (2) Consider a `dead-man's switch` on the pause flags: if `setX(true)`
      has been in effect for > N days (e.g., 90), an unprivileged caller
      can flip it back to `false`. This adds liveness defense against a
      lost-key scenario at the cost of a small abuse surface.

Neither is required for the soft-launch ship.

**Soft-launch cap mitigation:** YES (cap-bounded losses during the 30-day
window).

---

## Carry-over of v1 LOW + INFO findings

All v1 LOW and INFO findings were re-checked against the patched source.
None were resolved by this patch round; all remain at their original
severity. No new evidence to escalate any of them. Brief restatement:

| v1 Finding                                                                                        | Status | Notes                                                              |
|---------------------------------------------------------------------------------------------------|--------|--------------------------------------------------------------------|
| [LOW] `receive15pct` CEI ordering (external call before state update)                              | OPEN   | Unchanged at lines 71-73. Defense-in-depth; PWMToken has no hooks. |
| [LOW] No `nonReentrant` guard anywhere                                                            | OPEN   | Unchanged. Cheap to add OZ ReentrancyGuard; recommend post-launch.  |
| [LOW] `rankBps(0)` and `rankBps(>10)` silently return 0; ambiguous `DrawSettled` events            | OPEN   | Unchanged at lines 147-153, 165-176.                                |
| [LOW] `slashForChallenge`/`graduate` 1-wei dust on odd stake amounts                              | OPEN   | Unchanged. Inert until `setStakeAmount` to an odd value.            |
| [LOW] `forceApprove` race window — allowance not reset to 0 after the call                       | OPEN   | Unchanged at lines 139-140 (staking) and 193-194 (reward).          |
| [LOW] `setStakeAmount` accepts 1 wei (storage-spam griefing)                                      | OPEN   | Unchanged at lines 85-90.                                           |
| [INFO] Rank is caller-supplied; no on-chain randomness                                            | OPEN   | Unchanged.                                                          |
| [INFO] Double-claim prevention is `settled[certHash]` only                                        | OPEN   | Unchanged.                                                          |
| [INFO] $1 K → PWM-wei cap conversion correctness                                                  | OPEN   | Deploy script uses `parseEther("100")` = 100 PWM; verify USD price. |
| [INFO] Contracts are not upgradable                                                               | OPEN   | Unchanged — correct posture.                                        |
| [INFO] `SafeERC20` everywhere                                                                      | OPEN   | Unchanged — correct posture.                                        |

---

## Economic-invariant re-check

### Distribute sum identity
At `PWMRewardERC20.sol:185`:
```solidity
uint256 tkAmt = drawAmt - acAmt - cpAmt - l3Amt - l2Amt - l1Amt;
```
By construction, `acAmt + cpAmt + l3Amt + l2Amt + l1Amt + tkAmt == drawAmt`,
exactly. The split bps (5500 + 1500 + 1000 + 500 + 1500 == 10_000) sum to
`BPS_DENOM`, so the *intended* `tkAmt` is `drawAmt * 1500 / BPS_DENOM` and
the assigned-by-subtraction value captures any per-bucket rounding dust
into the T_k bucket. This is the right place for the dust (treasury can
afford imprecision; royalty recipients should get exactly their fraction).
**Invariant preserved post-patch.**

### Stake fate identity (graduation / slash-challenge / fraud)
| Path                  | `s.amount` distribution                                                                          | Sum check                          |
|-----------------------|--------------------------------------------------------------------------------------------------|------------------------------------|
| Graduate              | `half` → staker, `other = s.amount − half` → reward.seedBPool (B-pool)                            | `half + other == s.amount` ✓       |
| Slash for challenge   | `half` → BURN_SINK, `other = s.amount − half` → challenger                                       | `half + other == s.amount` ✓       |
| Slash for fraud       | `s.amount` → BURN_SINK                                                                            | exact ✓                            |

`totalActiveStakeWei` is decremented by `s.amount` in all three paths
(lines 134, 151, 166). `Status` is set to terminal (Graduated/Slashed/Fraud)
before any transfer; second-call reverts at `s.status == Active` check.
**Invariant preserved post-patch.**

### Treasury T_k balance accounting
`treasury[principleId] += amount` on inflow (line 72), `treasury[principleId]
= balance − amount` on outflow (line 88). No path increments or decrements
without a matching transfer. **Invariant preserved post-patch.**

### Cap accounting
- `totalActiveStakeWei` correctly incremented at line 112 *before* the
  external `safeTransferFrom`; if the transfer reverts, the staking call
  reverts and the increment is rolled back. Same for the cap check at
  lines 107-111: cap is read off the freshly-incremented `newTotal`. No
  TOCTOU window.
- `pool[benchmarkHash]` cap check at lines 137-140 uses the freshly-summed
  `newBalance`, same pattern. No TOCTOU.
**Invariant preserved post-patch.**

No economic-invariant regressions introduced by the patches.

---

## Cross-contract trust (unchanged from v1)

Same as v1. The new `transfersPaused` flag does not introduce any new
external trust dependency. The `onlyGovernance` gate on `depositBounty`
*reduces* the set of trusted callers (was permissionless, now governance
only); see [MED-NEW-1].

---

## Confidence

Re-read every line of all three patched contracts and the deploy script
section that configures them (`deploy/erc20.js:175-208`). Cross-referenced
with `PWMMintingERC20.setMintingPaused` and the v1 review report. Did not
re-read `PWMGovernance`, `PWMCertificate`, `PWMToken`, or the native-coin
siblings; their trust posture is unchanged from v1.

---

## What I did NOT check (same as v1, restated for clarity)

- Off-chain rank-assignment process.
- Property-based / fuzzing pass with Echidna or Foundry.
- Whether the multisig actually holds enough PWM to call `depositBounty`
  (operational, not contract-level).
- Whether `setGovernance` rotation under a "lost key" scenario has a
  documented recovery RACI (governance-design question; flagged in
  [INFO-NEW-2]).
- Base L2 sequencer-level concerns.

---

## Bottom line

  - All four patches are correct and resolve their respective v1 issues.
  - No regressions introduced.
  - One new MED ([MED-NEW-1]) raised by `depositBounty` over-restriction;
    director should confirm intent (per-deposit governance vs. per-funder
    whitelist).
  - One v1 MED ([MED-CARRYOVER]) deferred — staking still lacks a global
    pause; acceptable for soft-launch, flag for post-launch backlog.
  - Two new INFOs around distribute/pause asymmetry (intentional but
    non-obvious) and the combined governance kill-switch surface.
  - All six v1 LOWs and five v1 INFOs carry forward unchanged.
  - Economic invariants verified preserved.

The contracts are ready for the soft-launch posture as patched. The single
unresolved question is the intended operational model for `depositBounty`
— if the answer is "governance only", this report is effectively clean.
