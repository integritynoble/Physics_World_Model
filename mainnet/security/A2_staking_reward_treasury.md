# A2 Security Review — PWMStakingERC20, PWMRewardERC20, PWMTreasuryERC20

**Date:** 2026-05-18
**Reviewer:** Claude Opus 4.7 (Agent A2)
**Scope:** 3 contracts
  - `contracts/PWMStakingERC20.sol` (183 lines)
  - `contracts/PWMRewardERC20.sol` (203 lines)
  - `contracts/PWMTreasuryERC20.sol` (84 lines)
**Soft-launch caps in effect:** STAKING_TVL_CAP_USD=$1000, MINTING_PAUSED=true, TREASURY_TRANSFERS_PAUSED=true

---

## Summary

| Severity   | Count |
|-----------:|:-----:|
| CRITICAL   |   0   |
| HIGH       |   1   |
| MEDIUM     |   3   |
| LOW        |   6   |
| INFO       |   5   |

The contracts are small, idiomatic OpenZeppelin-style ERC20 code with strict
`onlyGovernance` / `onlyCertificate` / `onlyReward` / `onlyStaking` gating on
every state-changing path that moves money. CEI ordering is largely correct.
The single most important issue is that **PWMTreasuryERC20 has no
`treasuryTransfersPaused` flag at all**, despite the soft-launch plan listing
`TREASURY_TRANSFERS_PAUSED=true` as an in-effect cap. That gap is rated HIGH
because the soft-launch mitigation cannot be enacted by the contract as
written — the cap exists only as an off-chain promise.

Aside from that, residual risk on Base mainnet is well-bounded by the
`maxTotalStakeWei` cap (10 PWM at deploy, ≈ $1K equivalent at the genesis
price assumption) and by the fact that `PWMToken` is a plain
OZ ERC20-Capped — no ERC777 hooks, no reentrancy vector through transfers.

---

## Findings

### [HIGH] PWMTreasuryERC20 has no pause flag — soft-launch `TREASURY_TRANSFERS_PAUSED=true` cannot be enforced on-chain

**File:** `contracts/PWMTreasuryERC20.sol:1-84`
**Function:** entire contract; specifically `payAdversarialBounty(...)` (lines 65-79)
**Description:**
The deploy plan declares `TREASURY_TRANSFERS_PAUSED=true` as an in-effect cap
at mainnet launch, but `PWMTreasuryERC20` defines no such state variable. The
only path out of the treasury is `payAdversarialBounty`, gated by
`onlyGovernance` and by `amount * 2 <= balance` (50 % cap). There is no
boolean that governance can flip to globally freeze withdrawals, and there
is no view function returning a pause state.
**Impact:**
The promised pause-by-default behavior does not exist. If a `principleId`
ever accumulates T_k (via `receive15pct`) during the soft-launch window —
which it cannot today only because `MINTING_PAUSED=true` and the upstream
reward draw rarely triggers — the only thing preventing a payout is
governance discipline (a 3-of-5 multisig choosing not to call
`payAdversarialBounty`). If a governance key is compromised at the multisig
threshold, an attacker can drain up to 50 % of every per-principle balance
per call with no on-chain stop. The compounding-with-time risk is also
present: the contract has no upper bound at all on `treasury[principleId]`
beyond the global PWM supply (21 M ether).
**Reproducibility:**
1. Reward contract calls `receive15pct(0, 10 ether)` → `treasury[0] = 10e18`.
2. Three founder keys collude (or are compromised); they call
   `payAdversarialBounty(0, attacker, 5 ether)` — succeeds.
3. Repeat the call; each call drains 50 % of the remainder. No on-chain
   guard exists to stop this short of a `setGovernance` rotation, which
   itself requires the same threshold and a 48 h timelock window upstream.
**Recommendation:**
Add a `bool public transfersPaused` storage variable, a
`setTransfersPaused(bool)` setter gated by `onlyGovernance`, and a
`require(!transfersPaused, "PWMTreasuryERC20: paused")` check at the top of
`payAdversarialBounty`. Default `transfersPaused = true` in the constructor
(or set it as part of the deploy script) so that genesis state matches the
declared soft-launch cap. Also consider adding the same flag and check to
`receive15pct` if the design intends to block inflows during pause; today
inflows depend on the upstream `PWMRewardERC20` pause posture.
**Soft-launch cap mitigation:** PARTIAL — `MINTING_PAUSED=true` keeps the
reward pool from filling, which transitively keeps T_k empty during the
first 30 days. But once minting is unpaused on day 31 the gap reopens.
Also, `depositBounty()` in `PWMRewardERC20` is permissionless (see [MEDIUM] below)
and can independently fund pool balances that eventually flow to T_k.

---

### [MEDIUM] `PWMRewardERC20.depositBounty` is permissionless and bypasses `MINTING_PAUSED`

**File:** `contracts/PWMRewardERC20.sol:127-130`
**Function:** `depositBounty(bytes32 benchmarkHash, uint256 amount)`
**Description:**
Unlike `seedBPool` (gated to `staking`) and `depositMinting` (gated to
`minting`), `depositBounty` has no caller restriction. Anyone holding PWM
can push tokens into any benchmark pool. There is no guard preventing this
during the soft-launch period when `MINTING_PAUSED=true` is supposed to keep
pool growth at zero.
**Impact:**
Two issues. (1) Soft-launch accounting: total $-at-risk in `PWMRewardERC20`
during the 30-day window is no longer bounded by the staking cap alone —
any holder can dump arbitrary PWM into the pool and then trigger a payout
via the cert finalize path. The `maxBenchmarkPoolWei` cap in the contract
is not set at deploy (default 0 means "no cap" per the
`if (cap != 0)` short-circuit at line 137). (2) Pool-cap bypass on
mainnet: if governance later sets `maxBenchmarkPoolWei` to a low value to
bound per-pool exposure, the cap is enforced consistently in `_credit`, so
this part is OK — but a malicious actor can still fill multiple pools
across many `benchmarkHash` values.
**Reproducibility:**
1. Deployer mints 21 M PWM at genesis, distributes per the allocation plan.
2. Attacker holds 10 PWM, approves the reward contract, calls
   `depositBounty(arbitraryBenchmarkHash, 10e18)`.
3. `pool[arbitraryBenchmarkHash] = 10e18`. No cap check fires because
   `maxBenchmarkPoolWei == 0`.
**Recommendation:**
Either gate `depositBounty` to a whitelist (governance maintains a
`mapping(address => bool) public bountyFunders`), or **always** set
`maxBenchmarkPoolWei` to a non-zero soft-launch value in the deploy
script (e.g., 100 PWM ≈ $1000 worth, matching the staking cap). The latter
is the minimum-disruption fix and matches the symmetry of
`maxTotalStakeWei`. Also: add an explicit `setMaxBenchmarkPoolWei` call to
the deploy script — currently neither the constructor nor any in-source
defaulting forces it to a non-zero value at genesis.
**Soft-launch cap mitigation:** NO — the cap variable exists but defaults
to 0 (= unlimited). A deploy-script line must set it; this needs to be
verified in `deploy/` artifacts before mainnet launch.

---

### [MEDIUM] `setMaxTotalStakeWei` accepts 0, which disables the cap silently

**File:** `contracts/PWMStakingERC20.sol:91-94`
**Function:** `setMaxTotalStakeWei(uint256 newMax)`
**Description:**
Inside `stake()` at line 108 the contract reads `cap = maxTotalStakeWei` and
short-circuits the cap check with `if (cap != 0)`. This means setting
`maxTotalStakeWei = 0` does not enforce a "no staking" pause — it disables
the cap. A governance multisig intending to set the cap to its tightest
value (e.g., 1 wei or any small number) might fat-finger 0 and accidentally
remove the cap entirely.
**Impact:**
Single-key fat-finger by a governance proposer (3-of-5 still required to
execute, but the 48-hour timelock only protects against *unauthorized*
proposals, not against the wrong-value proposals that 3 sloppy approvers
ratify). The result is the staking TVL grows unbounded up to the available
supply of approved PWM. On mainnet pre-day-31 this could substantially
exceed the $1K cap the protocol promised.
**Recommendation:**
Reject `newMax == 0`:
```solidity
require(newMax > 0, "PWMStakingERC20: zero max disables cap");
```
If a "disable cap" path is genuinely desired, introduce an explicit
`bool public capEnabled` flag, and treat `newMax == 0` separately as a
typed governance action. Mirror the same fix in
`setMaxBenchmarkPoolWei` (line 108 of `PWMRewardERC20.sol`).
**Soft-launch cap mitigation:** PARTIAL — exploitable only by governance,
and the 3-of-5 + 48 h timelock provides defense in depth.

---

### [MEDIUM] `PWMStakingERC20` has no pause flag; soft-launch posture relies entirely on `maxTotalStakeWei`

**File:** `contracts/PWMStakingERC20.sol:96-122`
**Function:** `stake(uint8 layer, bytes32 artifactHash)`
**Description:**
There is no global pause for `stake()`, `graduate()`, `slashForChallenge()`,
or `slashForFraud()`. The cap-based bound is the only soft-launch safety
rail. If an exploit is discovered post-deploy, governance has two options:
(1) set `maxTotalStakeWei = 1` (effectively disables new stakes; existing
stakes still resolvable), or (2) rotate governance to a "null" address that
cannot call anything (but this also locks resolution paths forever).
Neither is a clean "pause + unpause" cycle.
**Impact:**
Incident-response inflexibility. Loss is bounded by the $1K cap during the
soft-launch, but the inability to cleanly freeze the contract while a fix
is prepared is a meaningful operational gap.
**Reproducibility:** N/A — design observation.
**Recommendation:**
Add a `bool public paused` and `whenNotPaused` modifier on `stake`. Allow
governance to flip via `setPaused(bool)`. The resolution functions
(`graduate`, `slash*`) can intentionally remain unpaused so in-flight
artifacts can still be wound down. This mirrors common OZ Pausable patterns
and aligns with the declared soft-launch posture.
**Soft-launch cap mitigation:** YES for steady-state — `maxTotalStakeWei`
bounds total exposure to $1K. The mitigation is operational-response only.

---

### [LOW] `receive15pct` violates checks-effects-interactions

**File:** `contracts/PWMTreasuryERC20.sol:57-62`
**Function:** `receive15pct(uint256 principleId, uint256 amount)`
**Description:**
The contract calls `pwmToken.safeTransferFrom(msg.sender, address(this),
amount)` (external call) **before** updating `treasury[principleId] +=
amount`. Standard CEI would update state first then make the external
call. PWMToken is a plain ERC20 with no callback hooks, so reentrancy is
not exploitable today. If the token is ever upgraded to one with hooks
(it cannot — PWMToken is non-upgradable), this becomes exploitable.
**Impact:**
None against current PWMToken. Style/defense-in-depth concern.
**Recommendation:**
Re-order:
```solidity
treasury[principleId] += amount;
pwmToken.safeTransferFrom(msg.sender, address(this), amount);
```
Or add a `nonReentrant` modifier — OZ `ReentrancyGuard` is already a small
dependency, and applying it across all of `receive15pct`, `payAdversarialBounty`,
`stake`, `graduate`, `slashForChallenge`, `slashForFraud`, `seedBPool`,
`depositMinting`, `depositBounty`, and `distribute` would add a uniform
defense-in-depth posture across all three contracts. This is the single
biggest hygiene improvement available.
**Soft-launch cap mitigation:** YES — fully bounded by $1K cap.

---

### [LOW] No `nonReentrant` guard on any external function across all three contracts

**File:** `contracts/PWMStakingERC20.sol`, `contracts/PWMRewardERC20.sol`, `contracts/PWMTreasuryERC20.sol`
**Function:** all state-changing externals
**Description:**
None of the three contracts import `ReentrancyGuard` or use a
`nonReentrant` modifier. PWMToken has no callback hooks (it is plain
OZ ERC20 + ERC20Capped), so reentrancy through token transfers cannot be
triggered today. The risk surface opens if:
  (a) a future migration ever swaps PWMToken to a callback-bearing
      implementation;
  (b) `forceApprove` + cross-contract `seedBPool`/`receive15pct` callbacks
      grow more complex.
**Impact:**
None today. Strict defense-in-depth concern. Note that
`PWMStakingERC20.graduate` already does three external calls in sequence
(`safeTransfer(s.staker, half)`, `forceApprove`, `reward.seedBPool`), and
state mutation is correctly performed *before* all three — so even with
hypothetical hooks the standard reentrancy patterns (drain via re-claim)
do not apply. Similarly `distribute` sets `settled[certHash] = true`
before any transfers. CEI is right; the guard would only matter if the
intra-protocol callee (`reward.seedBPool`, `treasury.receive15pct`) ever
re-entered a sibling write path.
**Recommendation:**
Adopt `ReentrancyGuard` and mark all state-changing externals
`nonReentrant`. Trivial cost, removes a whole class of future regressions.
**Soft-launch cap mitigation:** YES.

---

### [LOW] `rankBps(0)` and `rankBps(>10)` silently return 0 — no event indicates rollover reason

**File:** `contracts/PWMRewardERC20.sol:146-175`
**Function:** `distribute(certHash, Draw d)`
**Description:**
When `d.rank` is 0 or > MAX_RANK (10), `rankBps` returns 0 and
`distribute` emits `DrawSettled(certHash, benchmarkHash, rank, 0, pool[...])`
with no separate reason indicator. Off-chain indexers cannot distinguish
"rank too high" from "pool empty" from "draw amount rounded to 0" by event
shape alone (drawAmt == 0 case at line 172 has the same event payload).
**Impact:**
Indexer / accounting clarity. Not a security bug.
**Recommendation:**
Add a reason string or use a separate event for "rank not eligible" vs
"draw rounded down to zero" vs "rollover".
**Soft-launch cap mitigation:** N/A — not a loss vector.

---

### [LOW] `slashForChallenge` / `graduate` dust asymmetry

**File:** `contracts/PWMStakingERC20.sol:126-142, 144-158`
**Function:** `graduate`, `slashForChallenge`
**Description:**
Both functions split `s.amount` into `half = s.amount / 2` and
`other = s.amount - half`. When `s.amount` is odd, `other > half` by 1 wei.
In `graduate`, the staker receives `half` (the smaller side) and the reward
pool receives `other` (the larger side). In `slashForChallenge`, the burn
receives `half` (smaller) and the challenger receives `other` (larger).
With current per-layer amounts of 10/2/1 ether (all even-wei), this dust is
never actualized. If governance ever sets `setStakeAmount(layer, oddValue)`
the staker pays 1 wei more to the protocol on graduation and the burn
loses 1 wei to the challenger. Inconsequential, but worth noting.
**Impact:** Negligible (≤1 wei per resolution).
**Recommendation:**
Either explicitly document the rounding direction, or normalize:
```solidity
uint256 half = s.amount / 2;
uint256 other = half;          // give both sides the same; protocol keeps the 1-wei dust
require(half * 2 <= s.amount); // unused dust becomes contract-locked
```
Trivial fix; mainly hygiene.
**Soft-launch cap mitigation:** YES.

---

### [LOW] `forceApprove` race window between approve and call

**File:** `contracts/PWMStakingERC20.sol:137-139`, `contracts/PWMRewardERC20.sol:192-194`
**Function:** `graduate`, `distribute`
**Description:**
The pattern `pwmToken.forceApprove(callee, amt); callee.receiveX(...)`
relies on the callee immediately consuming the full allowance inside the
inner call. If `seedBPool` / `receive15pct` ever changes to pull less than
the full `amt`, residual allowance remains. A malicious or buggy callee
could later sweep the leftover via a separate `transferFrom`.
**Impact:**
Today both callees pull exactly `amount` via `safeTransferFrom(msg.sender,
address(this), amount)`. No residual. The risk is purely about callee
implementation drift.
**Recommendation:**
After the call, reset:
```solidity
reward.seedBPool(benchmarkHash, other);
pwmToken.forceApprove(address(reward), 0);
```
Same for the treasury approval in `distribute`. OZ `forceApprove(addr, 0)`
is the safe reset pattern. Trivial defense-in-depth.
**Soft-launch cap mitigation:** YES.

---

### [LOW] `setStakeAmount` allows lowering below 1 wei effectiveness

**File:** `contracts/PWMStakingERC20.sol:85-90`
**Function:** `setStakeAmount(uint8 layer, uint256 amount)`
**Description:**
The setter requires `amount > 0` but accepts any positive value, including
1 wei. If governance sets a tier to 1 wei, the cap at `maxTotalStakeWei`
effectively permits ~10^18× more stakes than intended (because each entry
takes one wei), and the per-artifact bookkeeping mapping could be spammed
cheaply, growing storage forever (each `stakes[artifactHash]` slot ≈
160 + 8 + 256 + 8 = 432 bits = 2 storage slots).
**Impact:**
Griefing vector: cheap-to-create unbounded storage. Mitigated by the cap
on total stake wei but not by a cap on number of artifacts. Each entry
permanently occupies storage.
**Recommendation:**
Either (a) enforce a minimum stake amount (e.g., `require(amount >= 1
ether / 1000, ...)` so 0.001 PWM is the floor), or (b) add a per-staker /
per-layer count limit. Option (a) is simpler.
**Soft-launch cap mitigation:** PARTIAL — economically bounded but storage
growth could become a long-term gas cost burden.

---

### [INFO] No on-chain randomness — rank is caller-supplied; threat surface lives upstream

**File:** `contracts/PWMRewardERC20.sol:154-198`
**Function:** `distribute`
**Description:**
`distribute` takes `d.rank` from the caller (`PWMCertificate`). There is no
use of `block.prevrandao`, `block.timestamp`, or any oracle inside these
three contracts. The off-chain ranking logic is whatever
PWMCertificate.submit() does with `a.rank` (which is also caller-supplied
there — see `PWMCertificate.sol:122,146`). On Base L2, with Sequencer-driven
block production, miner/proposer manipulation of randomness is moot here
because no randomness is consumed.
**Impact:**
No randomness-manipulation attack surface inside the reviewed contracts.
The rank-assignment trust model lives in the off-chain process that
generates `submit()` inputs and the governance review during the challenge
window. That is a protocol-design question outside the contract scope.
**Recommendation:**
Document explicitly in NatSpec that `rank` is a trusted input and the
challenge window is the only on-chain check. Consider, in a later
revision, a Merkle-proof scheme so the rank is cryptographically committed
at benchmark seeding time.
**Soft-launch cap mitigation:** YES (cap-bounded).

---

### [INFO] Double-claim prevention is `settled[certHash]` only

**File:** `contracts/PWMRewardERC20.sol:155, 162`
**Function:** `distribute`
**Description:**
The same `certHash` cannot be settled twice. Different certs over the same
benchmark are settled independently and pull from the same `pool[bh]`
balance, which is the intended design. Each cert has its own `settled`
flag. No global per-rank uniqueness — rank 1 can be paid on many certs
over the lifetime of a benchmark; that is by design (multi-round).
**Impact:** No double-claim within a single cert.
**Recommendation:** None — design is correct.
**Soft-launch cap mitigation:** YES.

---

### [INFO] $1K cap → wei mapping correctness

**File:** `contracts/PWMStakingERC20.sol:91-94, 100-122`
**Function:** `setMaxTotalStakeWei`, `stake`
**Description:**
The cap is denominated in PWM wei (1 PWM = 1e18). At deploy, the soft-launch
plan asserts $1K equivalent. With genesis PWM trading on Uniswap v3 PWM/USDC
at the seed price, the implied PWM-per-USD must be computed off-chain at
deploy time and `setMaxTotalStakeWei` called with the resulting wei value.
The contract has no on-chain USD oracle (per CLAUDE.md: "No Chainlink
oracle. Flat amounts only."), so the $1K↔PWM-wei conversion is a deploy-
script responsibility. Verify the deploy script computes this correctly
and that the value is visible in the `MaxTotalStakeWeiUpdated` event before
any third-party staker can call `stake()`.
**Impact:** Operational — depends on deploy-script correctness.
**Recommendation:**
In the deploy script, after deploying `PWMStakingERC20`, *before* enabling
any staker access, call `setMaxTotalStakeWei(N * 1 ether)` where N matches
the $1K budget at the genesis price (e.g., if PWM/USD = $5 at seed,
N = 200 PWM, so `setMaxTotalStakeWei(200 ether)`). Emit the value on a
deploy-audit log.
**Soft-launch cap mitigation:** YES, if deploy script is correct.

---

### [INFO] Cap survives storage upgrades — contract is not upgradable

**File:** `contracts/PWMStakingERC20.sol`, `contracts/PWMRewardERC20.sol`, `contracts/PWMTreasuryERC20.sol`
**Description:**
None of the three contracts inherits from any upgradable proxy base
(no `Initializable`, no `UUPSUpgradeable`, no `TransparentUpgradeableProxy`
deployment indicated in source). State is committed to permanent EVM
storage at the deployed address. There is no storage-layout-change risk.
**Impact:** Cap and pause state are forever-stable.
**Recommendation:** None.
**Soft-launch cap mitigation:** N/A.

---

### [INFO] `transfer` / `transferFrom` return values handled via `SafeERC20`

**File:** all three contracts
**Description:**
All token movements use `SafeERC20.safeTransfer` / `safeTransferFrom` /
`forceApprove`, which check return values and revert on non-standard
return shapes. No bare `transfer` is used.
**Impact:** Correct.
**Recommendation:** None.
**Soft-launch cap mitigation:** N/A.

---

## Cross-contract trust

`PWMGovernance` (3-of-5 multisig, 48 h timelock) is treated as a fully
trusted address for `setX(...)` configuration paths in all three contracts.
This is justified: governance is in-protocol and operated by the founding
team. `PWMToken` is treated as a fully trusted ERC20 — also justified: it
is a plain OZ ERC20Capped, immutable supply, no upgrade path.

`PWMCertificate` is treated as fully trusted to call
`PWMRewardERC20.distribute(...)`. Justified by the `onlyCertificate`
modifier set via governance. The implication: a compromise of the
certificate contract (or of governance, which can swap the certificate
pointer) compromises reward distribution. The 48 h timelock on
`PWMGovernance.executeProposal` plus the certificate's own 7- or 14-day
challenge window provides ≥ 7-day defense in depth.

`PWMStakingERC20` is treated as the only legitimate caller of
`seedBPool`. Justified.

`PWMRewardERC20` is treated as the only legitimate caller of
`receive15pct`. Justified.

No external (non-PWM) contracts are trusted by any of the three reviewed
contracts. There are no oracle dependencies. There is no inter-chain
messaging.

---

## Confidence

Reviewed **deeply** (every line, every branch):
- `PWMStakingERC20.sol` — all 5 externals (`setGovernance`, `setReward`,
  `setStakeAmount`, `setMaxTotalStakeWei`, `stake`, `graduate`,
  `slashForChallenge`, `slashForFraud`).
- `PWMRewardERC20.sol` — all 11 externals plus `_credit` and `rankBps`.
- `PWMTreasuryERC20.sol` — all 4 externals.

Skimmed (cross-referenced for trust-model context only):
- `PWMGovernance.sol` (verified 3-of-5 + 48 h timelock semantics, founder
  rotation path).
- `PWMToken.sol` (verified ERC20Capped, no hooks, 21 M supply, immutable).
- `PWMCertificate.sol` (verified `distribute` is dispatched from `finalize`
  after the challenge window).

Read but not relied on for findings:
- `PWMStaking.sol` (native-coin sibling — same shape, different value
  movement; not in scope).
- `PWMReward.sol` (native-coin sibling — same shape, different value
  movement; not in scope).
- `PWMTreasury.sol` (native-coin sibling — same shape; not in scope).
- `PWMMintingERC20.sol` (relevant only as a `depositMinting` caller).

---

## What I did NOT check

- **Off-chain rank-assignment process.** `rank` is caller-supplied at
  `PWMCertificate.submit()`. The fairness of the off-chain process that
  generates rank values is out of scope for a Solidity review.
- **Gas-bomb / DoS at scale.** I did not run a fuzzer to find paths where
  governance calls revert under specific calldata. The contract surface is
  small enough that this is unlikely to hide much, but a property-based
  test pass with Echidna or Foundry invariants would add confidence.
- **Deploy-script correctness.** The $1K → PWM-wei conversion, the
  `setMaxBenchmarkPoolWei` defaulting, and the `setTreasury` ordering all
  live in `deploy/` and were not reviewed here. Recommend a separate pass
  on the deploy artifacts before mainnet.
- **Base L2 sequencer-level threats.** Base is a single-sequencer L2;
  censorship and reorg-on-sequencer-restart are network-level concerns
  outside the contract scope. None of the reviewed contracts depend on
  block-level entropy or strict sequencing across blocks.
- **Front-running of `stake()` under cap.** When `totalActiveStakeWei` is
  near the cap, two competing `stake()` calls in the same block could race
  and one will revert. This is an expected fail-open behavior, not a bug.
- **Token-supply economics.** Whether $1K is actually a sufficient cap at
  Base mainnet PWM market price on 2026-05-19 is a deploy-time question,
  not a contract-correctness question.
- **Integration with `PWMVesting` and `Reserve` multisig.** Those are
  upstream holders of PWM; their behavior does not affect the security of
  the staking/reward/treasury triangle.
