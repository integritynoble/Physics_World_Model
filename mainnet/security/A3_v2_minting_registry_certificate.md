# A3-v2 Re-Review — PWMMintingERC20, PWMRegistry, PWMCertificate

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A3-v2)
**Commit:** `203df847` of `release/d9-soft-launch-2026-05-18`
**Scope (3 contracts, full re-read on patched commit):**
  - `contracts/PWMMintingERC20.sol` (299 lines) — 17.22M PWM allocation pool
  - `contracts/PWMRegistry.sol` (75 lines) — append-only artifact hash store
  - `contracts/PWMCertificate.sol` (251 lines) — L4 certificate submission + challenge + settlement dispatch

## Summary

| Severity | v1 Count | v2 Count | Delta                                         |
|----------|----------|----------|-----------------------------------------------|
| CRITICAL |    1     |    0     | C-1 RESOLVED (with residual trust assumption) |
| HIGH     |    2     |    1     | H-1 RESOLVED; H-2 still valid                 |
| MEDIUM   |    3     |    4     | +1 NEW (M-4 deploy-script default mismatch)   |
| LOW      |    5     |    5     | L-3 RESOLVED-via-modifier; L-6 NEW            |
| INFO     |    6     |    8     | +2 NEW (I-7 trust model, I-8 audit trail)     |

**Patch verdict.** The CRITICAL fix (`onlySubmitter` gate on `submit()`) is correctly implemented and has no detectable bypass — the modifier is applied to the only externally callable mutator of `certificates[]`, and the flag/mapping plumbing is consistent. The HIGH-1 fix (`mintingPaused` + `setMintingPaused`) is correctly implemented and covers the single `mintFor` entry path. No new CRITICAL or HIGH issues introduced.

**Residuals the Director must accept knowingly:**
  1. The C-1 fix is "governance vouches for submitters," NOT "no self-dealing possible" — an approved submitter still freely chooses `a.rank`, `a.acWallet`, `a.cpWallet`, and all L1/L2/L3 creator wallets. They can rank-1 self-deal exactly as before. The defence is *who* gets the key, not *what* the key can do. See I-7.
  2. `submissionPermissionless` is a single-bool kill of the whitelist. One bad governance tx re-introduces the original CRITICAL with no timelock at the contract layer. See M-4 / L-6.
  3. The `mintingPaused` storage default at construction is `false`. The protection that mainnet ships paused depends entirely on the deploy script calling `setMintingPaused(true)`. Env var `PWM_MINTING_PAUSED=false` silently disables the soft-launch posture. See M-4.

---

## Patch verification

### CRITICAL fix C-1 — `PWMCertificate.submit` access control

**Patch applied:**
  - `mapping(address => bool) public approvedSubmitter` (line 46)
  - `bool public submissionPermissionless` (line 47, default `false` per constructor leaving storage zero)
  - `modifier onlySubmitter` (lines 87–91)
  - `setApprovedSubmitter(submitter, approved)` `onlyGovernance` (lines 121–125)
  - `setSubmissionPermissionless(bool)` `onlyGovernance` (lines 127–130)
  - Modifier applied to `submit(SubmitArgs calldata)` (line 149)

**Bypass search (exhaustive walk of all functions and storage writes):**

  1. Other writers of `certificates[]`?
     - `submit` (sole write of new entry; gated by `onlySubmitter`). ✓
     - `challenge`, `resolveChallenge`, `finalize` all read existing entries and require non-`None` status; they cannot create a new entry. ✓
     - No fallback, no receive, no assembly. ✓ (Verified line-by-line; the contract has no `fallback()` / `receive()` declared, so cold ETH calls revert.)
  2. Other writers of `approvedSubmitter[]` or `submissionPermissionless`?
     - Only `setApprovedSubmitter` (writes mapping; `onlyGovernance`) and `setSubmissionPermissionless` (writes bool; `onlyGovernance`). No constructor write; no library; no inheritance shadow. ✓
  3. Can the modifier be tricked?
     - `submissionPermissionless || approvedSubmitter[msg.sender]` — short-circuits on the bool first. If governance never flips the bool, every caller must be in the mapping. ✓ No view-vs-state inconsistency. ✓
  4. Initialization race?
     - `governance = initialGovernance` is set in the constructor (line 95). The first transaction after deploy is by the deployer (now governance). There is no window where `governance == address(0)` and modifier checks fail. ✓
     - The default storage value for `submissionPermissionless` is `false`, and `approvedSubmitter[]` defaults to empty. So immediately after construction, `submit` is callable by NOBODY until governance acts. This is the strongest possible default. ✓
  5. Re-entrant flip during a sensitive call?
     - `setSubmissionPermissionless` and `setApprovedSubmitter` are pure storage writes with no external calls. There's no way for a re-entry to flip the flag mid-finalize. ✓ Even if it did, `submit` and `finalize` operate on independent `certHash` storage slots, so a flip can't retroactively unauthorize an existing cert.
  6. Other entry points that *circumvent* the cert pipeline (e.g., does `finalize` accept a fabricated Draw)?
     - `finalize` only reads from `certificates[certHash]`, populated by `submit`. There's no way to inject a `Certificate` struct without going through `submit`. ✓

**Verdict:** **C-1 RESOLVED at the contract layer.** No bypass found.

**Residual — read closely:**

The fix *closes the public-permissionless path* but does NOT add any rank verification, AC/CP wallet binding, or L1/L2/L3 chain-of-creator binding. The exploit primitive described in v1 ("submit a rank-1 cert with attacker-controlled wallets") still works *if you have governance approval as a submitter*. The trust model has shifted from "open mempool → adversary" to "approved submitter → adversary," and the protection now lives entirely off-chain (governance's KYC/vetting of submitter EOAs).

See finding I-7 for the explicit trust-model statement that should be in the soft-launch announcement.

---

### HIGH fix H-1 — `PWMMintingERC20.mintingPaused`

**Patch applied:**
  - `bool public mintingPaused` (line 38)
  - `event MintingPausedUpdated(bool paused)` (line 62)
  - `setMintingPaused(bool paused) external onlyGovernance` (lines 108–111)
  - `require(!mintingPaused, "PWMMintingERC20: minting paused")` at the head of `mintFor` (line 197)

**Coverage of all mint paths:**

  1. `mintFor(uint256 principleId, bytes32 benchmarkHash)` — only externally callable function that emits `Minted` and increments `M_emitted`. Now guarded by `require(!mintingPaused)` as the first executable statement. ✓
  2. Any other function that increments `M_emitted` or moves tokens out of the contract?
     - `setGovernance`, `setCertificate`, `setReward`, `setMintingPaused`, `setDelta`, `setPromotion`, `registerBenchmark`, `setBenchmarkRho`, `removeBenchmark` — none touch `M_emitted` or token balances. ✓
     - `_incrementActivity` mutates activity counters but does not move tokens. ✓
     - No `withdraw`, no `rescueTokens`, no `sweep` functions exist. (See L-6 below — this is actually a different problem.)
  3. Could governance be tricked into flipping `mintingPaused = false` accidentally?
     - `setMintingPaused` requires `onlyGovernance`. If `governance` is the 3-of-5 multisig with 48h timelock (per PWMGovernance.sol spec), then no single founder can flip it. **This depends on the deploy script correctly handing `governance` to the multisig — not verified here, but tracked in the A1 review.**
  4. Default-pause semantics:
     - At construction, `mintingPaused` defaults to `false` (Solidity zero-init).
     - Mainnet protection therefore depends on the deploy script calling `setMintingPaused(true)` BEFORE governance handoff.
     - `deploy/erc20.js` lines 191/199–202: `MINTING_PAUSED = (process.env.PWM_MINTING_PAUSED ?? "true") === "true"`, then `if (MINTING_PAUSED) setMintingPaused(true)`. So the default env behavior is correct.
     - **BUT:** if anyone runs the script with `PWM_MINTING_PAUSED=false` or any non-"true" string, the contract ships unpaused. There is no contract-level invariant that prevents this. See M-4.

**Verdict:** **H-1 RESOLVED at the contract layer.** Single mint path covered. The deploy-script default behavior is correct on `process.env.PWM_MINTING_PAUSED ?? "true"`. Caveat: trust the deploy script (M-4).

---

## Findings

### H-2 (HIGH) — STILL VALID — No `nonReentrant` on `PWMCertificate.finalize`

**Status:** Unchanged from v1. The patch did not address reentrancy hygiene. Recommendation unchanged: add OpenZeppelin `ReentrancyGuard` to PWMCertificate and PWMMintingERC20, apply `nonReentrant` to `submit`, `challenge`, `resolveChallenge`, `finalize`, and `mintFor`. Cost ~2.3k gas per call. Current `PWMToken` is plain ERC20 (no hooks) so no practical exploit today, but the dependency on token-class for safety is fragile.

(Full description retained in v1 report — see `A3_minting_registry_certificate_2026-05-18.md` H-2.)

---

### M-1 (MEDIUM) — STILL VALID — `PWMRegistry` ownership is a single-key EOA

**Status:** Unchanged. Registry's `Ownable` is unmodified in this patch cycle. Recommendation unchanged: rotate ownership to 3-of-5 multisig immediately after genesis batch, override `renounceOwnership()` to revert.

---

### M-2 (MEDIUM) — STILL VALID — `PWMCertificate.submit` does not bind `certHash` to a registered L4 entry

**Status:** Unchanged. The patch added access control but did NOT add the recommended check that `certHash` is a layer-4 entry in `PWMRegistry`. CertHash squatting by an *approved* submitter against another approved submitter is still possible. Recommendation unchanged.

---

### M-3 (MEDIUM) — STILL VALID — Governance can desync `PWMMintingERC20` benchmark registry from `PWMRegistry`

**Status:** Unchanged. No patch in this cycle. The two stores still drift on governance mistake. Recommendation unchanged.

---

### M-4 (MEDIUM) — **NEW** — `submissionPermissionless` and `mintingPaused` are single-tx flips with no time-lock or two-step confirmation at the contract layer

**Files / functions:**
  - `PWMCertificate.sol` :: `setSubmissionPermissionless(bool)` (lines 127–130)
  - `PWMMintingERC20.sol` :: `setMintingPaused(bool)` (lines 108–111)

**Description:**

Two new governance privileges introduced by the patch each toggle a single bool that radically changes the protocol's security posture:

  1. `setSubmissionPermissionless(true)` re-introduces the original C-1 CRITICAL state (any address can `submit()`). A single passing `onlyGovernance` call is the entire defense between soft-launch posture and pre-patch attack surface.
  2. `setMintingPaused(false)` re-enables the 17.22M PWM emission pipeline. A single passing `onlyGovernance` call is the entire defense between soft-launch and full mint.

Each of these is, by intent, a governance-controlled lever — that's appropriate. The concern is two-fold:

  - The contract layer does NOT enforce a time-lock or two-step confirmation. The 48h timelock referenced in PWMGovernance.sol spec is enforced one level up; if `governance` ends up being an EOA or a low-threshold multisig (out of scope for this review but cross-reference A1), the contract layer accepts the flip instantly.
  - Both events are emitted, but there is no on-chain "intent-to-unpause" → "delay" → "execute" pattern. Watchers must rely on monitoring the governance contract's queue, not these contracts.

**Impact:**
  - If the 3-of-5 multisig is compromised (3 keys lost or coerced), both flags can be flipped in a single transaction without the natural friction of a timelock.
  - If governance accidentally calls `setSubmissionPermissionless(true)` during soft-launch (e.g., fat-finger, copy-paste of test config), the entire patch is undone in one tx.

**Reproducibility:**
Deterministic given governance access.

**Recommendation:**
For the bool-flip-to-dangerous-state direction (`setSubmissionPermissionless(true)`, `setMintingPaused(false)`), add a two-step "propose + confirm after N hours" pattern at the contract layer. The bool-flip-to-safe-state direction (`setSubmissionPermissionless(false)`, `setMintingPaused(true)`) should remain instant — that's the emergency stop.

```solidity
// sketch
uint256 public submissionPermissionlessUnlockAt;
function proposeSubmissionPermissionless() external onlyGovernance {
    submissionPermissionlessUnlockAt = block.timestamp + 24 hours;
}
function setSubmissionPermissionless(bool x) external onlyGovernance {
    if (x) require(submissionPermissionlessUnlockAt != 0 &&
                   block.timestamp >= submissionPermissionlessUnlockAt, "timelocked");
    submissionPermissionless = x;
    submissionPermissionlessUnlockAt = 0;
    emit SubmissionPermissionlessUpdated(x);
}
```

**Soft-launch cap mitigation:**
If A1 confirms `governance` is wired to PWMGovernance (3-of-5 + 48h timelock), this finding drops to LOW. If `governance` is a raw multisig with no timelock, it stays MEDIUM. If `governance` is an EOA, it escalates to HIGH.

---

### L-1 (LOW) — STILL VALID — `setDelta` accepts arbitrary `principleId` with no registry cross-check

**Status:** Unchanged.

---

### L-2 (LOW) — STILL VALID — `PWMCertificate.challenge` lacks bond; spam-challenges are free

**Status:** Unchanged.

---

### L-3 (LOW) — DOWNGRADED — Attacker-chosen `rank=11+` zeroing out a cert

**Status:** Still a code-level fact, but the exploit is now gated behind the `onlySubmitter` whitelist. An approved submitter can still do this either by mistake or maliciously, but it's no longer a "drive-by" griefing primitive. Severity stays LOW; impact narrowed to "approved submitter misconfiguration." Recommendation unchanged: enforce `a.rank >= 1 && a.rank <= 10` at submit-time.

---

### L-4 (LOW) — STILL VALID — `forceApprove` dangling-allowance defensive note

**Status:** Unchanged.

---

### L-5 (LOW) — STILL VALID — `PWMRegistry.renounceOwnership` from OZ Ownable is callable

**Status:** Unchanged. Patch did not touch registry.

---

### L-6 (LOW) — **NEW** — `approvedSubmitter` mapping has no on-chain enumeration; only event log

**File / function:** `PWMCertificate.sol` :: `approvedSubmitter` mapping (line 46), `setApprovedSubmitter` (lines 121–125)

**Description:**

The list of currently-approved submitters is only inspectable by:
  1. Knowing the address up front and calling `approvedSubmitter(addr)` (per-address read, fine for verification of a single known address).
  2. Replaying every `ApprovedSubmitterUpdated` event from genesis to "now" and tracking the latest value per address — i.e., an off-chain indexer.

There is no `approvedSubmitterList()`, no `numApprovedSubmitters()`, and no Enumerable mapping. For the soft-launch phase where the universe of approved submitters is small (probably 1–5 addresses), this is acceptable. As the set grows, an event-replay dependency for "who can submit certs" becomes brittle:
  - A new ops engineer cannot answer "who are the currently approved submitters?" from contract state alone.
  - A bug in the indexer (missed event, fork ambiguity) can desync the apparent list from on-chain truth.
  - In an incident response window, faster on-chain enumeration would help.

**Impact:**
Operational/observability only. Not a security vulnerability per se.

**Reproducibility:**
By inspection of the contract.

**Recommendation:**
Add an `EnumerableSet.AddressSet private _approvedSubmitters` and maintain it in `setApprovedSubmitter`. Expose `approvedSubmitterCount()` and `approvedSubmitterAt(uint256)`. Cost: ~15k gas per add, similar per remove. Alternative for now: document the indexer/event-replay dependency in the runbook.

**Acceptable for soft-launch?** Yes, with the caveat that the runbook should explicitly state "approved submitter list is event-log only" and the operator should screenshot the post-deploy approval transactions.

---

### I-1 — RESOLVED — Prompt premise mismatches

The v1 INFO finding noting missing `mintingPaused`, ERC1155 absence, and `register*` shape is now partially obsolete: `mintingPaused` exists. The ERC1155 / `registerPrinciple/Spec/Benchmark` mismatches are still correct premise corrections — please ensure the audit-firm spec sheet has been updated, per the v1 recommendation.

---

### I-2..I-6 — STILL VALID, unchanged

  - I-2: UI-side bytes32 right-padding — frontend only.
  - I-3: `M_emitted ≤ M_POOL` proof — still holds, no math touched by patch.
  - I-4: SafeERC20 return-value checks — still holds.
  - I-5: No `unchecked` arithmetic — still holds.
  - I-6: `setPromotion(false)` weight-decrement invariant — still holds.

---

### I-7 (INFO) — **NEW** — Trust model after the C-1 patch: "governance vouches" not "no self-dealing"

**Files:** `PWMCertificate.sol` (entire submission pipeline)

**Description (for Director and auditor consumption):**

The C-1 patch closes the *open-mempool* attack but does NOT close the *insider* attack. Every approved submitter still has the following capabilities at `submit()` time:

  - Choose `a.rank` directly. Rank 1 = 40% of the benchmark pool. There is no rank-derivation, no Q_int → rank mapping, no oracle.
  - Choose `a.acWallet` and `a.cpWallet` freely. These receive 55% of the rank draw weighted by `shareRatioP ∈ [10%, 90%]`. There is no registry check that the AC/CP wallets match the artifact creator chain.
  - Choose `a.l1Creator`, `a.l2Creator`, `a.l3Creator` freely. These receive 5%/10%/15% of the rank draw. There is no chain-of-creator verification.
  - Choose `a.shareRatioP` freely within `[1000, 9000]`. Sets the AC/CP split.
  - Choose `a.principleId` freely; the only check is that `(principleId, benchmarkHash)` is registered in PWMMintingERC20.
  - Choose `a.delta` to control challenge-window length (7 vs 14 days).

**Trust statement (please paste into soft-launch announcement):**

  > During soft-launch, certificate submission is restricted to a governance-approved
  > allow-list. Approved submitters are trusted to (i) supply correct rank, AC/CP, and
  > creator wallets matching the actual contribution chain, and (ii) not self-deal.
  > Self-dealing by an approved submitter remains technically possible and is defended
  > only by (a) governance's vetting of who gets approved, (b) the 7-day (or 14-day)
  > challenge window during which any address may file `challenge()`, and (c) the
  > `maxBenchmarkPoolWei` per-benchmark cap, which bounds the dollar size of any
  > single self-dealing attempt during Phase 1.

**What the patch achieves:**
  - Eliminates drive-by exploitation by arbitrary mempool actors. ✓
  - Creates an on-chain record (events) of who was authorized to submit. ✓
  - Provides a single-call emergency revoke via `setApprovedSubmitter(addr, false)`. ✓

**What the patch does NOT achieve:**
  - Does not bind submitted data to verifiable on-chain truth.
  - Does not prevent rank/wallet substitution by an approved submitter.
  - Does not defend against approved-submitter key compromise.

**Recommendation:**
For mainnet promotion (not soft-launch), implement the v1 C-1 recommendations 1, 3, 4 (bind `certHash` to registry layer-4 entry; derive `rank` from `Q_int` + benchmark stats; bind AC/CP/L1/L2/L3 to the registry creator chain). The current patch is appropriate for *soft-launch with caps* but should be considered a stop-gap.

---

### I-8 (INFO) — **NEW** — Default-state asymmetry: `submissionPermissionless` defaults SAFE, `mintingPaused` defaults UNSAFE

**Files:** `PWMCertificate.sol`, `PWMMintingERC20.sol`

**Description:**

The two new bool flags have opposite default safety:

  - `submissionPermissionless`: default `false` → submit is locked to whitelist → SAFE default. ✓
  - `mintingPaused`: default `false` → mint is enabled → UNSAFE default. ✗

The deploy script `deploy/erc20.js` reconciles the second one by explicitly calling `setMintingPaused(true)` (line 200, when `MINTING_PAUSED` env defaults to "true"). But the contract-layer default is the *opposite* of the soft-launch posture. If anyone ever uses this contract outside the project's deploy script (e.g., re-deploys for a forked testnet, or audit firm spins up a local instance), they'll get an unpaused minting contract by default.

The comment on `PWMMintingERC20.sol` line 36–38 even claims `// mint flows from PWMCertificate.finalize → mintFor are blocked while paused.` — which is true, but the default of `mintingPaused` itself is `false`, not the more conservative `true`.

**Impact:**
Defense-in-depth. No exploit primitive on the main deploy path; just an "if you forget the script step, it ships unsafe" footgun.

**Recommendation:**
Either (a) change the constructor to initialize `mintingPaused = true` so the deploy script's `setMintingPaused(true)` becomes a redundant confirmation, OR (b) add a one-liner in the constructor: `mintingPaused = true; emit MintingPausedUpdated(true);`. Symmetric with the `submissionPermissionless = false` default in PWMCertificate, which is the correct posture.

```solidity
// PWMMintingERC20 constructor (suggested)
constructor(address token_, address initialGovernance) {
    require(token_ != address(0), "PWMMintingERC20: zero token");
    require(initialGovernance != address(0), "PWMMintingERC20: zero governance");
    pwmToken   = IERC20(token_);
    governance = initialGovernance;
    mintingPaused = true;                  // NEW
    emit MintingPausedUpdated(true);       // NEW
}
```

This makes the contract "deploy-script-independent" — even a fresh `new PWMMintingERC20(...)` from a Foundry script or REPL ships paused.

**Soft-launch cap mitigation:**
Current deploy script already handles this; finding is about robustness against future deploys.

---

## Confidence

**Deep-reviewed (line-by-line on patched commit):**
  - PWMMintingERC20.sol (299 lines) — every storage write, every modifier, every external call path. Traced patch additions and ensured no regressions in the existing CEI ordering, weight math, and cap enforcement.
  - PWMRegistry.sol (75 lines) — entire file; unchanged in this cycle.
  - PWMCertificate.sol (251 lines) — every external function, status transitions, finalize-call chain, patch additions, modifier order, no-fallback/no-receive verified.

**Cross-referenced:**
  - `deploy/erc20.js` lines 184–208 — confirmed the soft-launch posture is `setMintingPaused(true)` by default, `setSubmissionPermissionless` not called (relies on storage default `false`).
  - PWMRewardERC20.sol — confirmed `rankBps`, `distribute`, `settled[]` semantics relevant to L-3 unchanged.
  - PWMToken.sol — confirmed plain ERC20 (no hooks) so the H-2 reentrancy concern remains theoretical.

**Did NOT re-verify in this pass (assumed unchanged from v1):**
  - Hardhat/Foundry test coverage.
  - `governance` address handoff (cross-reference A1 review).
  - PWMGovernance.sol timelock semantics (cross-reference A1 review — directly affects M-4 severity).
  - Cross-contract interactions with Staking/Treasury beyond the finalize-call chain.

---

## Comparison to v1 — issue-by-issue

| ID  | v1 sev   | v2 sev   | Status   | Note                                                          |
|-----|----------|----------|----------|---------------------------------------------------------------|
| C-1 | CRITICAL | RESOLVED | RESOLVED | onlySubmitter modifier correctly applied; no bypass found.    |
| H-1 | HIGH     | RESOLVED | RESOLVED | mintingPaused + setMintingPaused correctly applied to mintFor.|
| H-2 | HIGH     | HIGH     | OPEN     | Reentrancy guard not added in this cycle.                     |
| M-1 | MEDIUM   | MEDIUM   | OPEN     | Registry Ownable single-key; unchanged.                       |
| M-2 | MEDIUM   | MEDIUM   | OPEN     | certHash not bound to registry layer-4 entry.                 |
| M-3 | MEDIUM   | MEDIUM   | OPEN     | Mintingbenchmark / registry desync risk.                      |
| M-4 |   —      | MEDIUM   | NEW      | Single-tx flip of two governance bools, no contract-timelock. |
| L-1 | LOW      | LOW      | OPEN     | setDelta accepts arbitrary principleId.                       |
| L-2 | LOW      | LOW      | OPEN     | Bondless challenge spam.                                      |
| L-3 | LOW      | LOW      | NARROWED | Now gated by onlySubmitter; insider misconfig only.           |
| L-4 | LOW      | LOW      | OPEN     | forceApprove dangling note (no current exploit).              |
| L-5 | LOW      | LOW      | OPEN     | renounceOwnership not overridden.                             |
| L-6 |   —      | LOW      | NEW      | approvedSubmitter has no on-chain enumeration.                |
| I-1 | INFO     | PARTIAL  | UPDATE   | mintingPaused now exists; other premise mismatches stand.     |
| I-2 | INFO     | INFO     | OPEN     | Bytes32 right-pad — frontend only.                            |
| I-3 | INFO     | INFO     | OK       | M_emitted ≤ M_POOL proof still holds.                         |
| I-4 | INFO     | INFO     | OK       | SafeERC20 return-checks.                                      |
| I-5 | INFO     | INFO     | OK       | No unchecked blocks.                                          |
| I-6 | INFO     | INFO     | OK       | setPromotion weight invariant.                                |
| I-7 |   —      | INFO     | NEW      | Trust model after C-1 patch — please publish.                 |
| I-8 |   —      | INFO     | NEW      | mintingPaused unsafe default; deploy script papers over.      |

---

## Soft-launch GO/NO-GO recommendation

**GO with conditions** (assuming A1 confirms governance is wired to PWMGovernance multisig with 48h timelock):

  1. Verify deploy script runs with `PWM_MINTING_PAUSED=true` (default) and the post-deploy state shows `mintingPaused == true`.
  2. Verify `approvedSubmitter[]` is empty at handoff (no auto-approvals in deploy script — checked, none present).
  3. Publish the I-7 trust statement in the soft-launch announcement so external users understand what the patch does and does not defend against.
  4. Keep `maxBenchmarkPoolWei = 100 PWM` (per `deploy/erc20.js` line 190) for Phase 1 to cap any insider self-deal.
  5. Set up off-chain monitoring for: `ApprovedSubmitterUpdated`, `SubmissionPermissionlessUpdated`, `MintingPausedUpdated` events — page Director on any change.
  6. Before mainnet promotion (post soft-launch), address M-4 (timelock for unsafe-direction flips) and the v1 C-1 deeper recommendations (rank derivation, AC/CP binding to registry).

**NO-GO if any of:**
  - `governance` ends up being an EOA (not multisig).
  - `mintingPaused` is not `true` at handoff.
  - The trust model in I-7 is not communicated to early users.
