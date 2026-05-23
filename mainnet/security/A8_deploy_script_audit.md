# A8 Deploy-Script Audit — `deploy/erc20.js` + `scripts/post_deploy_verify.js`

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A8 — in-session)
**Commit reviewed:** `fe3ba529` on `release/d9-soft-launch-2026-05-18`
**Scope:**
  - `infrastructure/agent-contracts/deploy/erc20.js` (283 lines)
  - `infrastructure/agent-contracts/scripts/post_deploy_verify.js` (158 lines)
  - `infrastructure/agent-contracts/scripts/preflight_mainnet.sh` (234 lines — not deep-reviewed; spot-checked)
  - `infrastructure/agent-contracts/scripts/transfer_admin_to_governance.js` (legacy native-coin path; out of scope but confirmed not used in ERC20 deploy)

---

## Summary

| Severity | Count | Blocks deploy? |
|---|---|---|
| CRITICAL | 0 | NO |
| **HIGH** | **1** | **YES — verifier #13 will fail on mainnet without this fix** |
| MEDIUM | 3 | NO (operational/coverage gaps, not security bugs) |
| LOW | 3 | NO |
| INFO | 2 | NO |

**Headline:** the deploy script correctly bakes in the soft-launch caps and hands off governance for 5 of 6 admin-bearing contracts — but **forgets to transfer PWMRegistry ownership**. Since the verifier `post_deploy_verify.js` Check #13 explicitly asserts `PWMRegistry.owner() == PWMGovernance`, the 20-check verifier will fail on mainnet with the current code. Fix: add one `await (await registry.transferOwnership(govAddr)).wait()` line to `deploy/erc20.js`. Trivial.

Beyond that one HIGH, the deploy script is in good shape. The genesis split is correct, the soft-launch caps are baked in before handoff, mainnet env vars are enforced, and the wiring order is internally consistent.

---

## [HIGH] deploy/erc20.js does not call `registry.transferOwnership(govAddr)`

**File:** `infrastructure/agent-contracts/deploy/erc20.js:237-249` (governance handoff section)
**Line range:** 237-249
**Conflict with:** `scripts/post_deploy_verify.js:126` Check #13 — "PWMRegistry.owner() == PWMGovernance"

**Description:**

The governance handoff loop at lines 243-247 transfers `governance` ownership on 5 contracts (Reward, Staking, Certificate, Minting, Treasury) via `setGovernance(govAddr)`. PWMRegistry is conspicuously missing.

PWMRegistry inherits from OpenZeppelin's `Ownable` (see `contracts/PWMRegistry.sol:12`). Its `register()` function is `onlyOwner`. After deploy:

  - The deployer EOA (Director's hot key) holds Ownable ownership of PWMRegistry forever.
  - PWMGovernance cannot register any artifact.
  - Any artifact registration post-deploy requires the deployer's private key.

The `post_deploy_verify.js` script was written with the expectation that this transfer happens. Check #13 reads:

```js
expect("13. PWMRegistry.owner() == PWMGovernance",     await reg.owner(),       A.PWMGovernance);
```

On the current `fe3ba529` deploy script, this check fails:

```
✗ 13. PWMRegistry.owner() == PWMGovernance   (got <DEPLOYER>, expected <PWMGov>)
```

**Impact:**

  - Verifier exit non-zero → Phase 5 post-deploy step is RED → deploy day is incomplete per playbook.
  - Operationally: registry write power tied to a single EOA (the deployer hot key). If that key is lost/compromised, no new artifacts can be registered (and the protocol is bricked downstream because PWMCertificate.submit() requires `registry.exists(benchmarkHash)`).
  - Defeats the spirit of the 3-of-5 + 48h multisig: a critical write path is gated by a 1-key signature.

**Reproducibility:**

1. Run `npx hardhat run deploy/erc20.js --network base` per Phase 5 runbook.
2. Run `npx hardhat run scripts/post_deploy_verify.js --network base`.
3. Output: 19/20 PASS, 1 FAIL on Check #13.

**Recommendation:**

Add to `deploy/erc20.js` between lines 247 and 248 (inside the `else` branch of the governance-handoff check):

```diff
   await (await reward.setGovernance(govAddr)).wait();
   await (await staking.setGovernance(govAddr)).wait();
   await (await certificate.setGovernance(govAddr)).wait();
   await (await minting.setGovernance(govAddr)).wait();
   await (await treasury.setGovernance(govAddr)).wait();
+  await (await registry.transferOwnership(govAddr)).wait();
   console.log("  governance handoff complete.");
```

PWMRegistry inherits OpenZeppelin Ownable so `transferOwnership(govAddr)` works. After the transfer, only PWMGovernance (via `proposeExec(target=registry, data=registry.register(...).calldata)`) can register artifacts.

Note: this means the **first 500 genesis Principles MUST be registered BEFORE the handoff**, while the deployer still owns the registry. The current deploy script doesn't register any genesis Principles — there's a separate `scripts/register_batch.py` or `scripts/register_genesis.js` for that. Director should confirm that genesis registration happens BEFORE this new `transferOwnership` line in the runbook order, OR the registration script is run via a `proposeExec` after handoff.

**Soft-launch cap mitigation:** NO — this gap is independent of the soft-launch posture.

---

## [MEDIUM] post_deploy_verify.js does not verify the 5 soft-launch caps

**File:** `scripts/post_deploy_verify.js` (whole file — caps not checked anywhere)

**Description:**

The post-deploy verifier covers token supply, genesis distribution, founder list, and governance handoff (20 checks). It does NOT verify the 5 soft-launch cap state variables that `deploy/erc20.js:194-208` sets:

| Soft-launch invariant | Set in deploy/erc20.js | Verified post-deploy? |
|---|---|---|
| `staking.maxTotalStakeWei == 100 ether` | line 195 | **NO** |
| `reward.maxBenchmarkPoolWei == 100 ether` | line 197 | **NO** |
| `minting.mintingPaused == true` | line 200 | **NO** |
| `treasury.transfersPaused == true` | line 204 | **NO** |
| `certificate.submissionPermissionless == false` (default) | not explicit | **NO** |

**Impact:**

If the deploy script had a bug (e.g., env var override `PWM_MINTING_PAUSED=false` set accidentally; or one of the setX calls silently reverted; or the deploy step was interrupted between cap-setting and handoff), the protocol would deploy with the caps NOT enforced. The 20-check verifier would still pass green. Operationally invisible until a user submits a stake exceeding the (missing) cap.

**Reproducibility:**

1. Modify env to `PWM_MINTING_PAUSED=false` before running deploy/erc20.js.
2. Deploy → handoff complete.
3. Run post_deploy_verify.js → 20/20 GREEN.
4. The protocol is live with minting ACTIVE on day 1, contradicting the soft-launch posture.

**Recommendation:**

Extend `post_deploy_verify.js` with a `[E] Soft-launch caps` section (5 new checks → 25-check verifier):

```js
console.log("\n[E] Soft-launch caps");
expectBigInt("21. staking.maxTotalStakeWei == 100 PWM",
             await staking.maxTotalStakeWei(),
             ethers.parseEther("100"));
expectBigInt("22. reward.maxBenchmarkPoolWei == 100 PWM",
             await reward.maxBenchmarkPoolWei(),
             ethers.parseEther("100"));
expect("23. minting.mintingPaused == true",
       await minting.mintingPaused(),  true);
expect("24. treasury.transfersPaused == true",
       await treas.transfersPaused(),  true);
expect("25. certificate.submissionPermissionless == false",
       await cert.submissionPermissionless(), false);
```

The constants 100 PWM should come from env vars (`PWM_STAKING_MAX_WEI`, etc.) so verifier and deploy share the source of truth.

**Soft-launch cap mitigation:** N/A — this is the VERIFIER for the cap mitigation.

---

## [MEDIUM] No approved submitter at deploy → no certificate can be submitted until governance proposes

**File:** `deploy/erc20.js` (no `setApprovedSubmitter` call anywhere)

**Description:**

The deploy script doesn't call `certificate.setApprovedSubmitter(someAddress, true)` for any address. After handoff to PWMGovernance, the only way to add an approved submitter is via `proposeExec(target=certificate, data=setApprovedSubmitter(addr, true))` — 3-of-5 + 48h timelock.

**Impact:**

On deploy day, **no PWM L4 certificate can be submitted by anyone for at least 48 hours** (until the first governance proposal completes its timelock). The whole "miners earn rewards by submitting certs" loop is dormant by design during this window.

This is *intentional* per the soft-launch posture, but it is *not documented* in the deploy script or runbook. Director may be surprised when a miner tries to submit a cert on day 1 and gets `PWMCertificate: not approved submitter`.

**Reproducibility:**

1. Deploy.
2. Founders propose `setApprovedSubmitter(<first-miner>, true)` via `proposeExec`.
3. 3-of-5 approve → wait 48h → execute.
4. ONLY AFTER step 3 can `<first-miner>` call `submit()`.

**Recommendation (low-friction):**

Add a deploy-script env var `PWM_INITIAL_SUBMITTER`. If set, the deploy script calls `certificate.setApprovedSubmitter(<env>, true)` before handoff. Documents that this address has elevated trust during soft-launch (it can self-deal a rank-1 cert, bounded by the $100 PWM pool cap).

Alternatively: document the 48h post-deploy delay explicitly in the launch announcement / runbook. State that "the first 48h is operational warm-up; the first cert submission is gated on governance vote."

**Soft-launch cap mitigation:** YES — the per-pool $100 PWM cap bounds the loss even if the initial submitter behaves badly.

---

## [MEDIUM] `PWM_SKIP_GOVERNANCE_HANDOFF=1` is a footgun on mainnet

**File:** `deploy/erc20.js:238-249`

**Description:**

The deploy script has a debug-only escape hatch:

```js
if (process.env.PWM_SKIP_GOVERNANCE_HANDOFF === "1") {
    console.log("\nSkipping governance handoff (PWM_SKIP_GOVERNANCE_HANDOFF=1).");
    console.log("  ⚠ admin remains the deployer — DO NOT do this on mainnet.");
} else {
    ...
}
```

The warning message is correct, but the script does NOT enforce the "DO NOT do this on mainnet" rule. A Director (or sub-GPU server with a stale env file) running `PWM_SKIP_GOVERNANCE_HANDOFF=1 npx hardhat run deploy/erc20.js --network base` would deploy with the deployer EOA as eternal governance — single-key control over $millions of PWM.

**Impact:**

  - Direct theft path: anyone with the deployer key can drain all setX-gated functions.
  - The 3-of-5 multisig + 48h timelock is bypassed.
  - The soft-launch cap mechanism is still in effect (caps were set when deployer = admin), but the cap-LIFTING flow can be triggered by the deployer EOA alone instead of governance.

**Reproducibility:** Director or operator accidentally exports the wrong env file from a testnet rehearsal session.

**Recommendation:**

Add a hard guard near line 238:

```diff
+  const LIVE_NETWORKS_HARD_GUARD = new Set(["base", "mainnet"]);
+  if (process.env.PWM_SKIP_GOVERNANCE_HANDOFF === "1"
+      && LIVE_NETWORKS_HARD_GUARD.has(network.name)) {
+      throw new Error("Refusing to skip governance handoff on " + network.name +
+                       ". Unset PWM_SKIP_GOVERNANCE_HANDOFF or use a testnet.");
+  }
   if (process.env.PWM_SKIP_GOVERNANCE_HANDOFF === "1") {
```

This pattern is already present elsewhere in the codebase (`scripts/transfer_admin_to_governance.js:74-84` uses `PWM_MAINNET_CONFIRM=1` for analogous protection). Mirror it here.

**Soft-launch cap mitigation:** NO — orthogonal to caps.

---

## [LOW] PWMToken Ownable not renounced or transferred

**File:** `deploy/erc20.js:75-79` (PWMToken deploy)

**Description:**

`PWMToken` is deployed with `Token.deploy(deployer.address, adminAddr)`. The second argument is the OpenZeppelin Ownable initial owner. After deploy, the deployer EOA permanently owns PWMToken. Carries A1's INFO-1 finding: PWMToken has no `onlyOwner` functions of its own (the cap-on-supply enforcement is via inherited `ERC20Capped._update`), so the owner can only `transferOwnership` or `renounceOwnership` — neither of which does anything useful here.

**Impact:**

  - No direct exploit — there's nothing the owner can do.
  - Operationally: the deployer hot key remains a target with zero upside. If compromised, attacker gains `renounceOwnership` (harmless) or could `transferOwnership` to themselves (also harmless). They cannot mint or pause.
  - Reviewers and integrators may MISTAKENLY think the owner has recovery or mint powers — confusing.

**Recommendation:**

Either:
  - (a) Add `await (await token.renounceOwnership()).wait()` after genesis distribution (line ~234 of deploy/erc20.js), or
  - (b) Add `await (await token.transferOwnership(govAddr)).wait()` alongside the other handoffs.

Option (a) is cleaner — removes the misleading dead role entirely. Per A1's INFO-1 recommendation.

**Soft-launch cap mitigation:** N/A.

---

## [LOW] Vesting start uses `Math.floor(Date.now() / 1000)` (server time, not block.timestamp)

**File:** `deploy/erc20.js:141`

**Description:**

`const now = Math.floor(Date.now() / 1000)` uses the deploy-server's local clock for `PWMVesting.start`. On mainnet, this gets passed as the `start` constructor arg. The contract then uses `start + CLIFF` (1 year) and `start + DURATION` (4 years) for vesting math against `block.timestamp`.

If the deploy server clock drifts (say, 30 minutes ahead), the cliff fires 30 minutes earlier than wall-clock day-365. If clock is behind, cliff fires 30 minutes later. Bounded drift; not a vulnerability.

**Impact:**

  - Sub-minute drift: imperceptible.
  - Multi-hour drift (broken server clock): vesting cliff fires up to several hours off. Still well-bounded.
  - Worst case: a malicious deployer could set `now = block.timestamp - 365 days` to make all founder vesting immediately past-cliff. But the deployer is the one running deploy; if the deployer is malicious, they could just mint to themselves.

**Recommendation:**

Pass `0` for `start` and have `PWMVesting` use `block.timestamp` as start internally. Or use `await ethers.provider.getBlock("latest")`'s timestamp before the deploy tx. Defensive only; current code is reasonable.

**Soft-launch cap mitigation:** N/A.

---

## [LOW] No `nextProposalId == 0` sanity check in verifier

**File:** `scripts/post_deploy_verify.js`

**Description:**

The verifier doesn't check that no proposals have been made yet (nextProposalId == 0, nextFounderChangeId == 0, nextExecProposalId == 0). A fresh deploy should have all three at zero. If any is non-zero, something happened during deploy that shouldn't have.

**Impact:** Sanity check, low risk. A non-zero counter would indicate the deploy ran into an unexpected state (e.g., a re-run on an existing deploy slot).

**Recommendation:**

Add 3 more checks to the verifier (would bring total to 28):

```js
expectBigInt("26. nextProposalId == 0",          await gov.nextProposalId(),          0n);
expectBigInt("27. nextFounderChangeId == 0",     await gov.nextFounderChangeId(),     0n);
expectBigInt("28. nextExecProposalId == 0",      await gov.nextExecProposalId(),      0n);
```

**Soft-launch cap mitigation:** N/A.

---

## [INFO] Cap is denominated in PWM wei, not USD

**File:** `deploy/erc20.js:189-197` + cap setter calls

**Description:**

The soft-launch caps default to `100 PWM` (in wei). The dispatch playbook talks about `STAKING_TVL_CAP_USD = $1,000` but the actual contract value is fixed in PWM units. The USD-equivalence depends on the post-LP-seed market price of PWM. At deploy time (before Uniswap v3 LP is seeded), PWM has no market price, so $-equivalence is moot. Once LP is seeded at, say, $0.10/PWM, the cap is $10 USD-equivalent. At $1.00/PWM, $100. At $10.00/PWM, $1,000.

**Impact:**

Documentation drift between dispatch playbook ("$1,000 cap") and reality ("100 PWM cap"). If PWM trades higher than $10 at launch, the cap is more permissive than $1K; if lower, more restrictive. Bounded either way.

**Recommendation:**

Either:
  - Document the cap in PWM units explicitly: "The soft-launch cap is **100 PWM regardless of USD value**", or
  - Add an off-chain conversion step pre-deploy: Director picks the PWM/USDC LP seed price, then deploy script computes `cap = $1000 / seedPrice * 1e18`.

Path A (document) is consistent with the dispatch playbook's explicit "100 PWM (deliberately tight)" choice.

---

## [INFO] Founder rotation sentinel check correctly verifies bytecode versioning

**File:** `scripts/post_deploy_verify.js:117-122`

The verifier wraps `await gov.nextFounderChangeId()` in a try/catch and treats a revert as "rotation code NOT in deployed bytecode". This is a clever sanity check — it confirms the deployed bytecode includes the founder-rotation flow (added 2026-05-11) and isn't a pre-rotation snapshot. Good pattern; consider adding similar try/catch for `nextExecProposalId` and other newly-added functions.

---

## Confidence

Reviewed **deeply** (every line):
  - `deploy/erc20.js` (283 lines)
  - `scripts/post_deploy_verify.js` (158 lines)

Reviewed **structurally** (skimmed for control flow + assumptions; did not exhaustively trace each branch):
  - `scripts/transfer_admin_to_governance.js` (legacy native path; confirmed not used)
  - `scripts/preflight_mainnet.sh` (smoke-tested for env-var checks; not all 234 lines re-walked)

**NOT reviewed:**
  - `scripts/preflight_proposal_flow.js`, `scripts/verify_governance_owns_admin.js`, `scripts/verify-all.js`, `scripts/export-abis.js` — out of scope.
  - The Phase 5 mainnet runbook MD itself (`pwm-team/coordination/wallet/PWM_PHASE_5_PROGRESS_2026-05-17.md`) — covered by A9 (spec consistency).

## What I did NOT check (out of A8 scope)

  - The actual addresses Director will use for Reserve / Liquidity / Team Beneficiary at deploy time. Operational — depends on Director's Gnosis Safe setup.
  - Gas budget — the cap-setting calls add ~5 transactions worth of gas (~0.001 ETH on Base). Tiny.
  - Reorg safety — the deploy script issues many transactions sequentially. On a deep reorg, intermediate state could be observable. Bounded risk on Base (single sequencer, no MEV reorgs).
  - The output `addresses.json` file format — verified slot pattern but didn't audit JSON correctness across all networks.
  - Faucet seeding — testnet only, out of mainnet scope.

---

## Recommended deploy-script fix patch (one PR)

Combine all HIGH + MEDIUM fixes into one small patch (~30 LoC):

```diff
   await (await reward.setGovernance(govAddr)).wait();
   await (await staking.setGovernance(govAddr)).wait();
   await (await certificate.setGovernance(govAddr)).wait();
   await (await minting.setGovernance(govAddr)).wait();
   await (await treasury.setGovernance(govAddr)).wait();
+  await (await registry.transferOwnership(govAddr)).wait();  // HIGH fix
   console.log("  governance handoff complete.");
```

And a hard guard at top of governance handoff block:

```diff
   } else {
+    if (network.name === "base" || network.name === "mainnet") {
+      // No-op — let it through; PWM_SKIP_GOVERNANCE_HANDOFF=1 already blocked above
+    }
     console.log("\nHanding off admin to PWMGovernance…");
```

And in `post_deploy_verify.js`, add the 5+3 = 8 new checks (Section [E] Soft-launch caps and Section [F] Proposal counters).

After this patch:
  - Deploy verifier coverage: 20 → 28 checks
  - Deploy script: 1 missing handoff fixed
  - Operational footgun: mainnet-handoff-skip blocked

Estimated time to implement + test: ~30 min. Could be bundled with sub-GPU's HIGH+MED PWMGovernance patch for atomic commit.
