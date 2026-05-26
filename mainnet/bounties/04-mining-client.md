# Bounty 4 — Mining Client (CP role)

- **Amount:** 80,000 PWM
- **Opens:** when `agent-miner` reference merges to `main` (~Day 27)
- **Reference implementation:** `infrastructure/agent-miner/pwm_miner/`
- **Acceptance harness:** `infrastructure/agent-miner/tests/` +
  `infrastructure/agent-miner/docker/Dockerfile`

## What you build

The Compute Provider (CP) client — the infrastructure that lets any GPU
operator earn PWM by executing solvers submitted by SPs, without needing
physics knowledge. Wraps four responsibilities:

1. **Job discovery** — subscribe to `PWMCertificate` event stream, filter for
   jobs whose declared compute manifest fits local hardware.
2. **Commit-then-reveal** — implement the 60-second anti-front-running
   protocol against `PWMCertificate`.
3. **Sandboxed execution** — run untrusted solver code in a hardened sandbox
   (Docker + seccomp + network egress whitelisted to the benchmark data host).
4. **Result reveal + claim** — upload reconstruction, call reveal within
   window, wait for finalize, receive `(1−p) × 55%` of the draw.

## Commit-then-reveal protocol

From `pwm_overview1.md §Compute Provider, §Commit-then-Reveal, §Spot-Check`:

```
Phase 1 (0–30 s after JobPosted event):
  CP commits hash(solution_preview || nonce)
  PWMCertificate.commit(commitHash)

Phase 2 (30–60 s):
  CP reveals solution_preview and nonce
  PWMCertificate.reveal(solution_preview, nonce)
  Contract verifies keccak256(preview || nonce) == commitHash
  First valid reveal wins (1-p) × 55% of the rank draw for that cert

Spot-check (async):
  Any CP may submit a spot-check proof showing a prior winner's solution did
  not match its commit preview. Slashes the cheater, awards 20% of the slash
  to the spot-checker.
```

## Interface contract

- **Contract ABIs:** `interfaces/contracts_abi/PWMCertificate.json` —
  subscribe to `JobPosted`, call `commit`, `reveal`, `claimReward`.
- **Addresses:** `interfaces/addresses.json`.
- **Sandbox spec:** `infrastructure/agent-miner/docker/Dockerfile` — your
  image must pass the same seccomp / capability-drop checks the reference
  image passes.

## What must pass

1. **100 consecutive testnet jobs.** Miner runs 100 jobs on Sepolia without a
   failed commit/reveal cycle caused by the client itself. Network or RPC
   failures that the reference handles identically are not counted against.

2. **Commit-reveal correctness.** Three adversarial tests:
   - Late commit: miner submits at T=29s; must succeed.
   - Late reveal: miner submits at T=59s; must succeed.
   - Mismatched reveal: on a crafted mismatch, contract call must revert and
     the client must surface the error, not swallow it.

3. **Sandbox escape tests.** Reference harness runs 10 malicious solver
   payloads (fork-bomb, mount attempt, egress to a blocked IP, kernel module
   load attempt, etc.). Zero escapes allowed.

4. **Hardware-filter correctness.** Given a `compute_manifest` declaring
   `min_vram_gb=24`, a 12-GB GPU client must not commit. Tested with
   synthetic manifests.

5. **Spot-check helper.** Must be able to submit a spot-check slash if fed
   a recorded preview/reveal divergence. Reference ships the sample
   divergence fixture; your spot-check call must succeed on Sepolia.

6. **Reward arrival.** For each of the 100 test jobs where the miner wins
   reveal, the `(1-p) × 55%` of the rank draw must land in the configured
   wallet within 2 blocks of `finalize`.

## What you may change

- Sandbox technology: Docker is the reference. Firecracker / gVisor /
  Kata Containers all acceptable provided they pass the escape tests.
- Implementation language: reference is Python; Rust/Go submissions welcome.
- Job-selection policy: reference uses expected-reward sort; your policy may
  differ as long as it respects the hardware filter.
- Logging, metrics, observability.

## What you may not change

- The commit-then-reveal timing (30s commit, 30s reveal window).
- The hash preimage (`keccak256(preview || nonce)`).
- The wallet-signing path (must support OS keychain or hardware wallet;
  plain-text keys are rejected, same rule as Bounty 3).
- The spot-check payload schema.

## Shadow run

Your miner runs alongside the reference for 30 days on Sepolia. Every reveal
your miner wins is paired with a reveal the reference would have produced;
cross-miner divergence in `solution_preview` is a regression event (the two
implementations may use different solvers so this compares CP-side only).

## Payment

- Paid from Reserve.
- Wallet in PR description.
- Mainnet swap 1:1 at Phase 2.
