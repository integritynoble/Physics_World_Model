# A3 Security Review — PWMMintingERC20, PWMRegistry, PWMCertificate

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A3)
**Scope:** 3 contracts
  - `contracts/PWMMintingERC20.sol` (288 lines) — 17.22M PWM allocation pool (82% of total supply)
  - `contracts/PWMRegistry.sol` (75 lines) — append-only artifact hash store
  - `contracts/PWMCertificate.sol` (227 lines) — L4 certificate submission + challenge + settlement dispatch

## Summary

| Severity | Count |
|----------|-------|
| CRITICAL | 1 |
| HIGH     | 2 |
| MEDIUM   | 3 |
| LOW      | 5 |
| INFO     | 6 |

The single CRITICAL finding (C-1) sits in `PWMCertificate.submit()` — submission is fully permissionless and accepts attacker-supplied creator/AC/CP wallets and rank. This makes the entire economic settlement layer untrusted at submission time and is the highest-impact issue in scope. The HIGH findings concern the absence of a global mint kill-switch and reentrancy-guard hygiene around the `finalize → mintFor → reward.distribute` external-call chain. Several premise mismatches with the review prompt (no `mintingPaused` flag, no ERC1155, no separate `registerPrinciple/Spec/Benchmark` functions) are documented as INFO so subsequent reviewers don't repeat the search.

## Findings

---

### C-1 (CRITICAL) — Permissionless `PWMCertificate.submit()` accepts attacker-controlled rank, creator, and payout wallets

**File / function:** `PWMCertificate.sol` :: `submit(SubmitArgs calldata a)` (lines 125–154)

**Description:**
`submit` has no access modifier and validates only that:
  - `a.certHash != 0` and not already submitted,
  - `a.shareRatioP ∈ [1000, 9000]`,
  - all addresses non-zero,
  - the supplied `a.benchmarkHash` exists in `PWMRegistry`.

It does NOT verify any of the following attacker-controlled fields:
  - `a.l1Creator, a.l2Creator, a.l3Creator` — the addresses that receive the L1/L2/L3 royalty split (5%/10%/15% of the rank draw) in `PWMRewardERC20.distribute`.
  - `a.acWallet, a.cpWallet` — receives 55% of the rank draw, split by `shareRatioP`.
  - `a.rank` — directly chooses the draw bucket: rank 1 = 40% of the benchmark pool, rank 2 = 5%, etc.
  - `a.principleId` — must merely be a `(principleId, benchmarkHash)` pair that is registered in `PWMMintingERC20`. The actual parent-of-benchmark relationship from `PWMRegistry` is never consulted.
  - `a.delta` — the submitter chooses their own challenge window length (≥10 = 14 days, else 7 days).

After the 7-day challenge window with no governance-resolved challenge, `finalize()` may be called by anyone and:
  1. invokes `PWMMintingERC20.mintFor(a.principleId, a.benchmarkHash)` — emits real PWM into the benchmark pool, and
  2. invokes `PWMRewardERC20.distribute(certHash, Draw{ ... attacker-supplied wallets ..., rank: a.rank })` — pays the attacker-supplied wallets.

**Impact:**
Direct theft of ranked-draw rewards. An attacker who watches a freshly funded benchmark pool can submit a self-dealing certificate with `rank=1`, `acWallet=cpWallet=attacker`, `shareRatioP=9000`, and after 7 days collect up to 40% of the benchmark pool (capped by `maxBenchmarkPoolWei`). They also displace the legitimate L1/L2/L3 creators by injecting their own addresses into those roles. Because `finalize` additionally drives `mintFor`, the attacker also drains the Zeno A-pool emission for that `(principleId, benchmarkHash)` into the same pool they intend to drain — so a single submission both inflates the pool and steals from it.

The protocol's stated defense is the 7-day challenge window plus governance review of disputed certs. This is an *off-chain* trust assumption. Until the watcher/challenger toolchain is verified live in production, every L4 submission is unauthenticated.

**Reproducibility:**
Deterministic. Anyone can call `submit()`. No revert path blocks an attacker who picks valid `(principleId, benchmarkHash)` from public state and supplies their own wallets / rank.

**Recommendation:**
At minimum one of the following before mainnet:
  1. Require the certHash be registered in `PWMRegistry` as `layer == 4` and require `msg.sender == registry.getArtifact(certHash).creator`. The registry already records the artifact creator and is `onlyOwner` (Director's EOA), so this would gate submission to creators the Director has whitelisted. This is the most natural fit given the existing registry contract.
  2. Or, require a signature from a trusted "submission verifier" key over the SubmitArgs payload.
  3. And, decouple `rank` from submission entirely: derive it from an oracle or compute it deterministically from `Q_int` + benchmark statistics, rather than letting the submitter type a number.
  4. And, require `acWallet`, `cpWallet`, and the L1/L2/L3 creator addresses to match the corresponding `registry.getArtifact(parentHash).creator` chain.

**Soft-launch cap mitigation:**
`PWMRewardERC20.maxBenchmarkPoolWei` directly caps per-benchmark exposure. Setting `maxBenchmarkPoolWei` so that 40% × cap × peak USD ≤ $1,000 (USD-equivalent) bounds a single self-dealing rank-1 cert. Director should also keep the certificate `governance` key online for the full 7-day window of every Phase-1 cert so that `resolveChallenge(upheld=true)` can reject malicious submissions before `finalize` is callable.

---

### H-1 (HIGH) — No global mint kill-switch in `PWMMintingERC20`; review prompt's `mintingPaused=true at deploy` premise is unmet by code

**File / function:** `PWMMintingERC20.sol` (entire contract)

**Description:**
The review prompt states "PWMMintingERC20 … `mintingPaused=true` at deploy. Emits PWM rewards when miners produce valid L1/L2/L3 submissions." No `mintingPaused` storage variable, modifier, or function exists in the contract. The de-facto pause is implicit: at deploy no principle has `promoted=true`, so `mintFor` reverts on `require(p.promoted)`. The first `setPromotion(id, true)` call (governance-only) globally enables minting for that principle.

There is no single-action kill-switch the governance multisig can hit if an exploit is discovered. To halt all emissions, governance would need to:
  - call `setPromotion(id, false)` on every promoted principle, OR
  - call `setCertificate(address(0))` — except `setCertificate` requires `x != address(0)`, so this path is blocked, OR
  - rotate `setReward(address(0))` — also blocked by zero-check, OR
  - in PWMCertificate, call `setMinting(address(0))` — also blocked by zero-check.

The cleanest practical kill-switch today is `setReward` to a no-op contract or `setPromotion(false)` per principle. Neither is atomic or obvious in an incident.

**Impact:**
In an incident (e.g., wallet compromise, or C-1 above being actively exploited), governance lacks a one-call STOP. Each minute of delay can leak more of the 17.22M PWM pool.

**Reproducibility:**
Code-inspection. No exploit primitive in itself, but a defense-in-depth gap.

**Recommendation:**
Add a `bool public mintingPaused` defaulting to `true` and `function setMintingPaused(bool) external onlyGovernance`. Guard `mintFor` with `require(!mintingPaused)`. Also relax `setCertificate / setReward / setMinting` zero-checks (or add a dedicated `clearReward()` etc.) so governance has a true off-switch.

**Soft-launch cap mitigation:**
With `M_POOL = 17.22M` PWM, this is the largest single tokens-at-stake target in the system. For Phase-1 mainnet, governance should pre-promote only ONE principle, set its `delta` modestly, and watch every `mintFor` event. The maxBenchmarkPoolWei cap downstream limits per-event exposure but does not bound the cumulative emission from this contract.

---

### H-2 (HIGH) — No `nonReentrant` modifier on `PWMCertificate.finalize` despite multi-contract external-call chain

**File / function:** `PWMCertificate.sol` :: `finalize(bytes32 certHash)` (lines 185–214)

**Description:**
`finalize` makes the following external calls in order:
  1. `minting.mintFor(c.principleId, c.benchmarkHash)` — inside which: `pwmToken.forceApprove(reward)` then `reward.depositMinting(...)` which in turn calls `pwmToken.safeTransferFrom(minting → reward)`.
  2. `reward.distribute(certHash, d)` — inside which up to 6 token transfers and a `treasury.receive15pct` external call.

`c.status = Status.Finalized` is set on line 191 BEFORE the external calls, which correctly blocks same-certHash re-entry. However:
  - There is no `ReentrancyGuard` on the contract.
  - A malicious `treasury` (governance-set, so governance trust required) could re-enter `PWMCertificate.finalize(otherCertHash)` for an UNRELATED certificate during the `receive15pct` call. State for `otherCertHash` is independent, so the practical exploit requires governance to set a malicious treasury — outside threat model.
  - More relevant: if PWMToken were ever replaced with an ERC-777-style token with transfer hooks (not the current `PWMToken.sol` which is plain ERC20), the chain could be reentered through token callbacks.
  - SafeERC20 + the current `PWMToken` (plain ERC20, no hooks) make practical reentrancy hard today.

Per scope check A: the *minting path* update-before-call ordering IS correct in `PWMMintingERC20.mintFor`:
  - line 209: `M_emitted += A_kjb;`
  - line 213: `_incrementActivity(...)` (mutates totalPrincipleWeight, sumBenchmarkWeight)
  - line 218–220: external calls (`forceApprove`, `reward.depositMinting`).
This is CEI-compliant. Good.

**Impact:**
No exploitable reentrancy today against the deployed PWMToken. But this is "no guard, code is currently safe" — fragile. A future token upgrade, a treasury swap, or an integration with a hook-bearing reward token would silently introduce a vulnerability.

**Reproducibility:**
Inspection.

**Recommendation:**
Add OpenZeppelin's `ReentrancyGuard` to PWMCertificate and apply `nonReentrant` to `submit`, `challenge`, `resolveChallenge`, `finalize`. Same for `PWMMintingERC20.mintFor`. Cost: ~2.3k gas per call, trivial.

**Soft-launch cap mitigation:**
Not directly cap-limited; current token is plain ERC20, so practical risk is low under the Phase-1 stack. But the `maxBenchmarkPoolWei` cap bounds any single-cert loss if a reentrancy bug surfaces post-deploy.

---

### M-1 (MEDIUM) — `PWMRegistry` ownership is a single-key EOA; loss = arbitrary artifact registration

**File / function:** `PWMRegistry.sol` :: inherits `Ownable` (line 12); `register` is `onlyOwner` (lines 33–59)

**Description:**
The registry is owned by `msg.sender` at deployment ("Director's deployer EOA" per comment, line 11). All artifact registration is permissioned to that single key. Comment recommends `transferOwnership` to rotate; nothing in the code requires a multisig.

If the deployer EOA is compromised:
  - attacker registers arbitrary L1/L2/L3 hashes pointing to themselves as `creator` (and any L1/L2/L3 chain).
  - those creators then become eligible to receive the L1/L2/L3 royalty splits on every cert that references those parents — but only if the cert references the malicious hashes. Practical theft requires also pulling off C-1 (submit), where attacker can supply their own l1/l2/l3 addresses anyway, so the registry compromise is less critical than expected.
  - `renounceOwnership()` from OZ Ownable is callable by the owner with no timelock — could brick all future registration with one transaction.

**Impact:**
Single point of failure for protocol curation. Combined with C-1, low marginal lift; standalone, MEDIUM since artifact provenance is the source of truth for off-chain consumers.

**Reproducibility:**
Standard EOA compromise.

**Recommendation:**
Rotate ownership to the 3-of-5 governance multisig immediately after deployment + genesis batch. Override or disable `renounceOwnership()` (OZ pattern: `function renounceOwnership() public override onlyOwner { revert("disabled"); }`).

**Soft-launch cap mitigation:**
Registry holds no funds, so cap analysis is indirect — relies on C-1's mitigation.

---

### M-2 (MEDIUM) — `PWMCertificate.submit` does not bind certHash to a `PWMRegistry` L4 entry; certHash squatting is possible

**File / function:** `PWMCertificate.sol` :: `submit` (lines 125–154)

**Description:**
`submit` checks only `certificates[a.certHash].status == Status.None` — i.e., that no one has previously called `submit` with this certHash. It does NOT verify that `certHash` was registered as a layer-4 artifact in `PWMRegistry`, and it does not check that `certHash` is the hash of the supplied SubmitArgs.

This means:
  1. An attacker can pre-empt a known-to-be-coming certHash (if predictable) by calling `submit` first with adversarial parameters.
  2. The "certHash" stored is decoupled from any verifiable content — there's no on-chain way to confirm what the certHash hashes.

The legitimate submitter would then be locked out (`"already submitted"`) and would have to rely on `governance` calling `resolveChallenge(certHash, upheld=true)` after a challenge — which works, but is a manual step.

**Impact:**
Combined with C-1, gives attackers a denial-of-service primitive against any honest miner whose certHash leaks before submission. Standalone, low-likelihood (certHash is presumably content-addressed and not predictable without seeing the payload), but worth fixing.

**Reproducibility:**
Requires predicting certHash before legitimate submission.

**Recommendation:**
Require `registry.exists(a.certHash) && registry.getArtifact(a.certHash).layer == 4` before accepting. Then the registry's `onlyOwner` gate doubles as a submission gate. Also consider requiring `msg.sender == registry.getArtifact(a.certHash).creator` (closes the C-1 path too).

**Soft-launch cap mitigation:**
Same as C-1 — `maxBenchmarkPoolWei` bounds the financial damage of a squatted+self-dealing cert.

---

### M-3 (MEDIUM) — Governance can desync `PWMMintingERC20` benchmark registry from `PWMRegistry`

**File / function:** `PWMMintingERC20.sol` :: `registerBenchmark` (130–144), `removeBenchmark` (157–177); `PWMRegistry.register`

**Description:**
PWMMintingERC20 maintains its own `_benchmarks[principleId][benchmarkHash]` mapping populated by `registerBenchmark()`. There is no on-chain check that the same `benchmarkHash` is also registered in `PWMRegistry` as `layer == 3` with the corresponding parent principle. The two stores can drift, e.g.:
  - benchmark registered in minting but not in registry → `PWMCertificate.submit` would still revert (`registry.exists` check), so this direction is mostly inert.
  - benchmark registered in registry but not in minting → `mintFor` reverts; cert finalize would also revert, freezing the cert in `Pending`. After the challenge window, finalize is still callable but will revert until governance calls `registerBenchmark` on minting. Operational, not security.
  - benchmark `removeBenchmark`'d from minting after certs are mid-window → finalize permanently reverts for those certs. Funds permanently locked unless governance re-registers.

**Impact:**
Operational fragility, governance mistake → stuck certs whose pool funds become unrecoverable. Not theft, but loss-of-availability.

**Reproducibility:**
Governance must coordinate two registrations manually.

**Recommendation:**
Either (a) make `registerBenchmark` on minting also write/check the registry, or (b) add a recovery path in PWMCertificate so governance can refund a stuck cert.

**Soft-launch cap mitigation:**
Bounded by per-cert pool exposure (`maxBenchmarkPoolWei`).

---

### L-1 (LOW) — `setDelta` accepts arbitrary `principleId` with no registry cross-check

**File / function:** `PWMMintingERC20.sol` :: `setDelta` (104–114), `setPromotion` (116–128)

**Description:**
`setDelta(principleId, delta)` will accept any `uint256 principleId` and store delta. `setPromotion(principleId, true)` then only requires `p.delta > 0` and `p.benchmarks.length > 0`. There is no link from `principleId` (a uint256) to the L1 artifact hash in `PWMRegistry`. The mapping is governance-curated and effectively trusted.

**Impact:**
Governance footgun — if Director sets delta on a wrong principleId, the wrong principle gets weight on promotion. Not exploitable without governance compromise.

**Recommendation:**
Maintain an on-chain `mapping(uint256 principleId => bytes32 registryHash)` and validate `registry.exists(hash) && registry.getArtifact(hash).layer == 1` on first delta-set.

**Soft-launch cap mitigation:**
Low marginal risk; under M_POOL cap.

---

### L-2 (LOW) — `PWMCertificate.challenge` lacks bond; spam-challenges are free; later challengers overwrite

**File / function:** `PWMCertificate.sol` :: `challenge` (160–167)

**Description:**
Anyone may call `challenge(certHash, proof)` once per cert. No bond is required; `proof` bytes are emitted but not validated. Once status becomes `Challenged`, subsequent challenges revert (require status == Pending). The first challenger is recorded; later challengers cannot piggyback.

**Impact:**
A malicious actor can challenge every honest cert to halt finalization until governance resolves each one (cheap DoS). Governance must then resolve each — resource burn.

**Recommendation:**
Require a small PWM bond on challenge, refundable if upheld, forfeit to treasury otherwise.

**Soft-launch cap mitigation:**
None needed — DoS, not theft.

---

### L-3 (LOW) — `PWMCertificate.submit` allows attacker-chosen `rank=11+` to permanently zero-out a cert

**File / function:** `PWMCertificate.sol` :: `submit` (lines 125–154) + `PWMRewardERC20.distribute` rank handling

**Description:**
If a cert is submitted with `rank=0` or `rank>=11`, `PWMRewardERC20.rankBps(rank) == 0` and `distribute` early-returns with `settled[certHash]=true`. The cert is forever marked settled with zero distribution. If the cert was submitted by an attacker who chose rank=11 (combined with C-1), the legitimate distribution path is permanently blocked because settled=true.

**Impact:**
Denial of legitimate reward; subset of C-1.

**Recommendation:**
On submit, require `a.rank >= 1 && a.rank <= 10`.

**Soft-launch cap mitigation:**
Same as C-1.

---

### L-4 (LOW) — `PWMMintingERC20.forceApprove` leaves dangling allowance on revert in `depositMinting`

**File / function:** `PWMMintingERC20.sol` :: `mintFor` (lines 218–220)

**Description:**
`pwmToken.forceApprove(address(reward), A_kjb)` is set right before `reward.depositMinting(...)`. If `depositMinting` reverts post-approval (it doesn't with current code — but if reward were upgraded), the outer call reverts and the approval is rolled back. If a future version moves to an approval-stays semantic, dangling approval = pool drain by anyone who can call transferFrom from the reward address — but only reward can transferFrom because the approval is to reward. So practical risk is low.

**Impact:**
Defensive note. No current exploit.

**Recommendation:**
Pattern is fine. Optionally reset to 0 after the call.

---

### L-5 (LOW) — `PWMRegistry.renounceOwnership` from OZ Ownable is callable; could brick the registry

**File / function:** `PWMRegistry.sol` :: inherited from OZ `Ownable`

**Description:**
OZ Ownable provides `renounceOwnership()` callable by the current owner with no timelock. A compromised key OR a fat-fingered Director can permanently disable all `register` calls. Registry is then frozen — no new artifacts can be added, ever.

**Impact:**
Irreversible loss of registry write capability. Existing artifacts unaffected (mapping is preserved), but protocol cannot grow.

**Recommendation:**
Override `renounceOwnership` to revert.

**Soft-launch cap mitigation:**
Not financial; operational.

---

### I-1 (INFO) — Prompt premise mismatches: no `mintingPaused`, no ERC1155, no `registerPrinciple/Spec/Benchmark`

**Files:** all three

**Description:**
The review-spec prompt for A3 contains several premise claims that do not match the deployed code. Documenting here so subsequent reviewers don't waste cycles searching:

  1. PWMMintingERC20 has NO `mintingPaused` flag (covered by H-1 above).
  2. PWMCertificate is NOT ERC1155, NOT ERC721 — it's a plain contract with a `mapping(bytes32 => Certificate)`. There are no tokens minted, no `_beforeTokenTransfer`, no `uri()`. Scope checks H (ERC1155 mint permissions), I (soulbound transferability), and J (metadata URI mutability) are therefore not applicable.
  3. PWMRegistry has a single `register(hash, parentHash, layer, creator)` function, NOT separate `registerPrinciple`, `registerSpecification`, `registerBenchmark`. The layer is a parameter (1=Principle, 2=Spec, 3=Benchmark, 4=Solution). Access is `onlyOwner`, not "anyone permissionless".

**Recommendation:**
Update the A3 prompt or the audit checklist to match the actual contract shapes, especially for downstream reviewers.

---

### I-2 (INFO) — `PWMRegistry` cannot detect UI-side bytes32 right-padding corruption

**File / function:** `PWMRegistry.sol` :: `register`

**Description:**
Per the Phase-3 Sepolia rehearsal bug (62-hex-digit input right-padded with `0x00` to make a valid 64-hex bytes32), the contract has no way to detect this — Solidity bytes32 is a fixed 32-byte word, and any 32 bytes is a valid input. The contract correctly validates `hash != bytes32(0)`, but cannot distinguish a legitimate hash with trailing zeros from a UI-corrupted one. This MUST be caught client-side. Confirming: no obvious validation is missing at the contract level.

**Recommendation:**
Frontend / off-chain only.

---

### I-3 (INFO) — `M_emitted` cannot exceed `M_POOL` (cap-safety verified)

**File / function:** `PWMMintingERC20.sol` :: `mintFor` lines 192–210

**Description:**
Verified by inspection:
  - `rem = M_POOL - M_emitted` (always nonneg because `M_emitted` only grows by `A_kjb`).
  - `A_k = rem * wK / sumW`; with `wK ≤ sumW` (since `wK` is one term of `sumW`), `A_k ≤ rem`.
  - `A_kjb = A_k * wB / sumBW`; with `wB ≤ sumBW`, `A_kjb ≤ A_k ≤ rem`.
  - Therefore `M_emitted += A_kjb ≤ M_emitted + rem = M_POOL`.

Additionally, line 208 (`require(pwmToken.balanceOf(this) >= A_kjb)`) is a defense-in-depth check against accidental drainage from outside the formula.

**Conclusion:** The 17.22M cap is mathematically enforced. No supply-cap overflow path identified.

---

### I-4 (INFO) — External call return values are checked via SafeERC20 / OZ patterns

**Files:** PWMMintingERC20, PWMCertificate

**Description:**
All `IERC20` calls go through `SafeERC20.safeTransfer` / `forceApprove` / `safeTransferFrom`, which revert on failure or false-return. The minting → reward.depositMinting and reward → treasury.receive15pct calls revert on failure (no `(bool ok, ) = .call(...)` returns to ignore). Scope check L is satisfied.

---

### I-5 (INFO) — No `unchecked` blocks in PWMMintingERC20 / PWMRegistry / PWMCertificate; Solidity ^0.8.24 overflow protection is in effect

**Description:**
No `unchecked` arithmetic anywhere in the three contracts. All math is checked. Per-weight terms (`p.delta * a`, `b.rho * a`) could theoretically overflow uint256, but require activities > 2^128 to do so — infeasible. Scope check K is satisfied.

---

### I-6 (INFO) — `setPromotion(false)` decrement uses current weight; consistency verified

**File / function:** `PWMMintingERC20.sol` :: `setPromotion`, `_incrementActivity`

**Description:**
Invariant: `totalPrincipleWeight == Σ _principleWeight(p) for all p with p.promoted == true`. Verified by tracing all mutation paths:
  - `setPromotion(true)`: adds `_principleWeight(p)` (current value at promotion). ✓
  - `_incrementActivity` while promoted: swaps `oldPW → newPW`. ✓
  - `setDelta` while promoted: swaps `oldW → newW`. ✓
  - `setPromotion(false)`: subtracts `_principleWeight(p)` (current value). ✓

No accounting drift identified.

---

## Confidence

**Deep-reviewed (line-by-line, all paths traced):**
  - PWMMintingERC20.sol — every external function, every internal function, all math invariants, all access paths.
  - PWMRegistry.sol — entire file (75 lines).
  - PWMCertificate.sol — every external function, status transitions, finalize-call chain into PWMMintingERC20 and PWMRewardERC20.

**Cross-referenced (read in full but not exhaustively traced):**
  - PWMRewardERC20.sol — read in full to verify the `finalize → distribute` data flow, splits, and that the rank-bps + cap interaction matches the C-1 / L-3 exploit description.
  - PWMToken.sol — read in full to confirm it is a plain ERC20 with no transfer hooks (relevant to H-2 reentrancy assessment).

**Spot-checked / context-only:**
  - PWMMinting.sol (predecessor) — function signatures only, to confirm PWMMintingERC20 is a structural copy.

## What I did NOT check

  1. **Hardhat/Foundry test coverage.** I did not open `test/` to verify whether the C-1 self-dealing path is exercised. If it is, the test should be reviewed; if it isn't, a test demonstrating C-1 should be added before mainnet.
  2. **Compiler / metadata pinning.** I did not run `solc` or inspect `hardhat.config.{ts,js}` for `evmVersion`, optimizer settings, or via-IR mode. Scope of this review was 3 .sol files.
  3. **Gas / DOS at registration scale.** The "Genesis 500 Principle hashes batch-registered" requirement (from `CLAUDE.md`) was not stress-tested. Each `register` is independent so no obvious DOS, but a batch wrapper may be in scope elsewhere.
  4. **PWMGovernance interactions.** I did not read PWMGovernance.sol; the assumption "governance is a 3-of-5 multisig" comes from the spec. If `governance` is currently an EOA, all `onlyGovernance` checks degrade to single-key trust. Other agents should confirm.
  5. **Front-running on `submit`.** Mempool-level frontrunning of legitimate submitters is plausible (especially with C-1) but I did not quantify Base mainnet mempool exposure.
  6. **Cross-contract state with PWMStakingERC20 / PWMTreasuryERC20.** Out of scope for A3.
  7. **Event-indexer / off-chain monitoring assumptions.** The 7-day challenge window's effectiveness depends entirely on off-chain watchers that I did not audit.
  8. **Upgrade paths.** None of these contracts are upgradeable (no proxy patterns observed) — assumed intentional.
