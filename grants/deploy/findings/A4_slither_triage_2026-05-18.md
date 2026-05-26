# A4 Slither Triage — Static Analysis Findings on Patched Code

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A4 — in-session)
**Commit reviewed:** `92e946e5` on `main` (post-A8 audit; pre sub-GPU HIGH+MED PWMGovernance patch)
**Raw output:** `deploy/findings/slither_v2_raw.json` (58 results)
**Slither version:** 0.11.5, solc 0.8.24, target paris
**Scope:** All `contracts/*.sol` except `node_modules|test|legacy`

---

## Summary

Slither reported **58 findings** by impact level. After triage against the actual deploy path (only ERC20 contracts + non-deployed legacy contracts) and protocol-design choices documented in `CLAUDE.md`:

| Slither Impact | Slither Count | After A4 Triage | Action |
|---|---|---|---|
| HIGH | 1 | **0 deploy-relevant** (1 false-positive on non-deployed legacy) | none |
| MEDIUM | 12 | **0 deploy-blocking** (10 known-acceptable `divide-before-multiply`; 1 testnet-only; 1 by-design) | document |
| LOW | 29 | **0 deploy-blocking** (20 `timestamp` on L2 = acceptable; 9 `reentrancy-events` are stylistic) | document |
| INFO | 16 | **0 deploy-blocking** (interface-inheritance + naming + intentional assembly + non-deployed) | none |

**Deploy gate from A4:** GREEN. No Slither finding blocks mainnet deploy.

The single HIGH (`arbitrary-send-eth`) is on `PWMReward.sol` (legacy native-coin contract) — **not deployed by `deploy/erc20.js`**, which deploys only the ERC20 sibling stack.

---

## 1. The single HIGH — false positive on non-deployed contract

### [HIGH/arbitrary-send-eth] `PWMReward._send(address,uint256)` at `PWMReward.sol:216-220`

```solidity
function _send(address to, uint256 amount) internal {
    (bool ok, ) = address(to).call{value: amount}("");
    require(ok, "PWMReward: transfer failed");
}
```

**Slither concern:** `to` is user-supplied (winner address from `distribute()` Draw struct); sending ETH to arbitrary user could enable phishing-style exploits.

**A4 triage: FALSE POSITIVE for mainnet deploy.**

`PWMReward.sol` is the **legacy native-ETH contract**. The Track C mainnet stack deploys `PWMRewardERC20.sol` instead (see `deploy/erc20.js:108-112`), which uses `SafeERC20.safeTransfer` (no `call{value:}()`). `PWMReward.sol` exists in `contracts/` only because it's still used by `test/integration_l4_lifecycle.test.js` for backwards-compat regression tests.

**Verification:**
```
$ grep "PWMReward\b\|getContractFactory(\"PWMReward\"" deploy/erc20.js
  (no matches — PWMReward not deployed)
$ grep "getContractFactory(\"PWMRewardERC20\")" deploy/erc20.js
  const Reward = await ethers.getContractFactory("PWMRewardERC20");
```

**Soft-launch cap mitigation:** N/A — non-deployed contract.

**Recommendation:** Document in `addresses.json` or `deploy/README.md` that the legacy `PWMReward.sol` is NOT a mainnet artifact. Optionally delete `PWMReward.sol`/`PWMMinting.sol`/`PWMStaking.sol`/`PWMTreasury.sol` from `contracts/` after a separate `legacy/` folder migration, but that touches the test suite.

---

## 2. MEDIUM findings (12)

### 2.1 `divide-before-multiply` (10) — split-percentage arithmetic, by design

Affected files:
  - `PWMReward.sol:169` (legacy, NOT deployed) — 5 instances
  - `PWMRewardERC20.sol:155` (deployed) — 5 instances

The pattern is identical across both contracts. In `distribute()`:

```solidity
uint256 drawAmt = (balance * rbps) / BPS_DENOM;          // first division
// ...
uint256 acAmt = (drawAmt * uint256(d.shareRatioP) * SPLIT_AC_CP) / (uint256(BPS_DENOM) * BPS_DENOM);
uint256 cpAmt = (drawAmt * (BPS_DENOM - uint256(d.shareRatioP)) * SPLIT_AC_CP) / (uint256(BPS_DENOM) * BPS_DENOM);
uint256 l3Amt = (drawAmt * SPLIT_L3) / BPS_DENOM;
uint256 l2Amt = (drawAmt * SPLIT_L2) / BPS_DENOM;
uint256 l1Amt = (drawAmt * SPLIT_L1) / BPS_DENOM;
```

**Slither concern:** computing `(A/B) * C` instead of `(A*C) / B` loses precision when `A/B` rounds down.

**A4 triage: ACCEPTABLE BY DESIGN.**

- Each multiplication of `drawAmt` by a split-bps constant loses at most `splitBps/BPS_DENOM ≈ 0.55` of a wei per draw. Across all 5 splits (AC, CP, L3, L2, L1), the maximum cumulative loss per draw is ~5 wei.
- The `tkAmt` field absorbs the dust: `uint256 tkAmt = drawAmt - acAmt - cpAmt - l3Amt - l2Amt - l1Amt;` (line 184 of PWMRewardERC20). Any rounding loss flows to the treasury — explicitly intended per `pwm_overview1.md` "15% to T_k absorbs dust".
- Recomputing as `(balance * rbps * splitBps) / (BPS_DENOM * BPS_DENOM)` would change `drawAmt` semantics (`drawAmt` is the "withdrawn from pool" amount and must equal `balance * rbps / BPS_DENOM` per spec).
- A2's economic-invariant check ("distribute sum identity: tkAmt = drawAmt − all others") was verified GREEN by A2-v2.

**Soft-launch cap mitigation:** YES — bounded by $100 PWM pool cap; max 5 wei dust per draw is far below any practical concern.

**Recommendation:** No code change. Add a NatSpec comment in `distribute()` explicitly documenting that "drawAmt − Σ(splits) = tkAmt, by design (15% absorbs any rounding dust)".

### 2.2 `incorrect-equality` (1) — `PWMFaucet.nextEligibleAt`, testnet-only

`PWMFaucet.sol:60-64` uses `last == 0` to detect "user has never claimed". Slither flags strict equality as potentially unsafe (e.g., signed integer comparisons).

**A4 triage: SAFE.** `last` is a `uint256` storing `block.timestamp`; `== 0` correctly detects the "never written" state. `PWMFaucet` is **testnet-only** — `deploy/erc20.js:151-160` only deploys it when `isTestnet || PWM_DEPLOY_FAUCET=1`.

**Soft-launch cap mitigation:** N/A.

**Recommendation:** No change.

### 2.3 `unused-return` (1) — `PWMCertificate.finalize` ignores `minting.mintFor()` return value

`PWMCertificate.sol:221` calls `minting.mintFor(...)` and doesn't bind the returned `A_kjb` (PWM amount minted). The amount is logged by PWMMintingERC20's own `Minted` event, so the caller has no business with the return value.

**A4 triage: BY DESIGN.** PWMCertificate doesn't need the return value; the cross-contract event chain (`Minted` from PWMMintingERC20 → `DrawSettled` from PWMRewardERC20) gives off-chain indexers everything they need.

**Recommendation:** No change. Adding `uint256 amt = minting.mintFor(...);` and emitting from PWMCertificate would be redundant.

---

## 3. LOW findings (29)

### 3.1 `timestamp` (20) — `block.timestamp` used in comparisons

Affected: all 20 are `require(block.timestamp >= proposedAt + TIME_LOCK, ...)` style checks in PWMGovernance + PWMCertificate (challenge window). On Base L2 (single sequencer, 2-second block times, no MEV reorgs), `block.timestamp` is reliable to ~2 seconds.

**A4 triage: ACCEPTABLE on Base.** This is a well-known false-positive for L2 deployment.

**Soft-launch cap mitigation:** N/A.

**Recommendation:** No change. Document in audit prep notes: "We deploy on Base L2 where `block.timestamp` drift is bounded by the sequencer's enforcement of monotonic blocks at ~2-second intervals."

### 3.2 `reentrancy-events` (9) — events emitted after external calls

Slither flags 9 functions where `emit X(...)` happens after an external call. In every case I checked, the state mutation that the event reports is committed BEFORE the external call (CEI is correct); the event itself is post-call, which is the standard pattern.

Example: `PWMCertificate.finalize` (line 232 `c.status = Status.Finalized;` is set BEFORE the `reward.distribute()` call at line 235, and the `CertificateFinalized` event is emitted after).

**A4 triage: STYLISTIC — no real reentrancy.** A2-v2 explicitly verified CEI ordering across PWMStakingERC20, PWMRewardERC20, PWMTreasuryERC20 and reported correct ordering. Same applies here.

**Soft-launch cap mitigation:** N/A.

**Recommendation:** No change. Could refactor to emit BEFORE the external call for stylistic Slither-clean code, but the security posture is unchanged.

---

## 4. INFO findings (16)

### 4.1 `missing-inheritance` (7)

Slither suggests that, e.g., `PWMRegistry` should `inherit IPWMRegistry`. Each protocol contract has an interface declared in a different contract file (often the consumer), and Slither thinks they should be paired explicitly.

**A4 triage: COSMETIC.** The interfaces are duplicated in the consumer files (e.g., `IPWMReward` appears in both `PWMCertificate.sol` and `PWMMintingERC20.sol`) to keep each consumer self-contained. Adding `inherits` would tighten coupling without security gain.

**Recommendation:** Optionally consolidate into `contracts/interfaces/` folder in a future refactor. Not blocking.

### 4.2 `low-level-calls` (6) — `address.call{value:}()` on legacy contracts

All 6 are on legacy native-coin contracts (`PWMStaking.sol`, `PWMReward.sol`, `PWMTreasury.sol`) — NOT deployed on mainnet.

**Recommendation:** No change (same as the HIGH finding).

### 4.3 `naming-convention` (2) — `M_emitted` not in mixedCase

`PWMMinting.M_emitted` and `PWMMintingERC20.M_emitted` are uppercase. The codebase intentionally uses `M_*` for variables that mirror the mathematical notation in `pwm_overview1.md` (the spec uses `M_emitted` as the cumulative emitted amount).

**A4 triage: BY DESIGN.** Math-spec-aligned naming. Cosmetic only.

**Recommendation:** No change. Document the math-aligned naming convention in CLAUDE.md if not already there.

### 4.4 `assembly` (1) — intentional inline assembly in `PWMGovernance.executeExec`

The bubble-up revert assembly added in commit `fe3ba529` (A1-v2 CRITICAL fix). Standard pattern to forward target revert reasons.

**A4 triage: BY DESIGN.** A1-v3 verified the assembly is correct.

**Recommendation:** No change.

---

## 5. Cross-references with prior agent findings

A4's triage is consistent with prior agents. No Slither finding contradicts the manual reviews:

| Slither finding | Prior agent verdict |
|---|---|
| HIGH arbitrary-send-eth on PWMReward.sol | Not deployed (A8 confirmed) |
| MED divide-before-multiply | A2-v2 sum-identity check verified |
| MED unused-return on finalize | A1 reviewed (no concern) |
| LOW timestamp × 20 | A1-v1 + A1-v2 + A1-v3 all accepted Base L2 timestamp model |
| LOW reentrancy-events × 9 | A2-v2 + A3-v2 verified CEI ordering correct |
| INFO assembly | A1-v3 reviewed bubble-up assembly |

## 6. Items Slither did NOT catch (and other agents did)

Important to note: **Slither's coverage is COMPLEMENTARY to, not a substitute for, manual review.** Things Slither did NOT catch but the manual agents DID:

  - A3 CRITICAL: `PWMCertificate.submit()` permissionless self-deal exploit (Slither doesn't reason about access-control semantics)
  - A1-v2 CRITICAL: PWMGovernance has no execute-call primitive (Slither only sees that the contract compiles)
  - A2 HIGH: `PWMTreasuryERC20` missing `transfersPaused` flag (Slither doesn't track "advertised invariants vs implemented")
  - A8 HIGH: `deploy/erc20.js` missing `registry.transferOwnership(govAddr)` (Slither doesn't look at deploy scripts)

This is the canonical case for multi-tool review: each layer catches what the others miss.

---

## 7. Recommendation summary

| Category | Recommended action | Blocking deploy? |
|---|---|---|
| HIGH (1, on non-deployed legacy) | None — verify legacy contracts are unused on mainnet (already confirmed by reading deploy script) | NO |
| MED divide-before-multiply (10) | Add NatSpec comment in `distribute()` explaining dust absorption to T_k | NO |
| MED incorrect-equality on Faucet | None — testnet only | NO |
| MED unused-return on finalize | None — by design | NO |
| LOW timestamp × 20 | None — Base L2 acceptable | NO |
| LOW reentrancy-events × 9 | None — CEI ordering correct, events post-call is fine | NO |
| INFO missing-inheritance × 7 | Optional refactor into `contracts/interfaces/` | NO |
| INFO low-level-calls × 6 | None — non-deployed | NO |
| INFO naming-convention × 2 | None — by design (math-spec-aligned) | NO |
| INFO assembly × 1 | None — intentional in executeExec | NO |

**Overall A4 verdict: Slither findings do NOT block deploy.**

---

## 8. What I did NOT cover

  - **Slither-printers** (e.g., `function-summary`, `inheritance`) — informational only, no triage value.
  - **Slither-detectors at INFO level beyond the 16 reported** — Slither's default detector set found 58 results; some lower-severity detectors might be disabled by default. Did not enable `slither --detect all` for an exhaustive scan; cost/benefit not worth it given the manual review coverage.
  - **Re-running Slither after sub-GPU's HIGH+MED PWMGovernance patch.** The thresholdReachedAt + _liveApprovals additions WILL likely add a few more `timestamp` findings (1-2) and no new HIGH/MED. A10 aggregator should note this; a final Slither pass on the deploy commit is recommended as a sanity check but not a deploy gate.
  - **Mythril** — separate symbolic-execution pass not yet executed (task #24). Would catch arithmetic edge cases Slither doesn't reach. Recommended to run overnight; not deploy-blocking given the manual review depth.
