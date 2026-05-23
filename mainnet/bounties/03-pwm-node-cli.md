# Bounty 3 — pwm-node CLI

- **Amount:** 100,000 PWM
- **Opens:** when `agent-cli` reference merges to `main` (~Day 22)
- **Reference implementation:** `infrastructure/agent-cli/pwm_node/`
- **Acceptance harness:** `infrastructure/agent-cli/tests/` (pytest + testnet e2e)

## What you build

`pwm-node` — the command-line tool individual miners (SP role) use to discover
benchmarks, run solvers locally, register as a Solution Provider, and submit
L4 certificates. It wraps the scoring engine (Bounty 1) and contract calls
behind human-readable commands. Install with one command; runs on Linux,
macOS, Windows.

## Mandatory commands

```bash
# Discovery (no wallet required)
pwm-node benchmarks [--domain <domain>] [--spec <spec_id>]
pwm-node inspect <cert_hash | benchmark_id | spec_id | principle_id>
pwm-node balance [--address <wallet>]

# Local validation (offline, no chain interaction)
pwm-node verify <benchmark_yaml>            # S1-S4 pre-check on a solver
pwm-node validate-benchmark <dir>           # schema check benchmark before submit

# Mining (SP role)
pwm-node mine <benchmark_id> --solver <solve.py> [--dry-run]

# Solution Provider registration
pwm-node sp register \
  --entry-point <solve.py> \
  --share-ratio <0.10..0.90>

# Submission to testnet / mainnet
pwm-node submit-cert --cert <payload.json>  # calls PWMCertificate.submit()
```

Additional commands (governance, staking, history) may be added but the list
above is the minimum required set.

## Interface contract

- **Scoring engine:** `from pwm_scoring import score_solution, SolutionScore`
  (Bounty 1 or the reference implementation — interchangeable, both pass the
  same tests).
- **Contract ABIs:** `interfaces/contracts_abi/*.json`.
- **Addresses:** `interfaces/addresses.json` — select `testnet` or `mainnet`
  via `--network` flag (default testnet).
- **Cert schema:** `interfaces/cert_schema.json`.

## What must pass

1. **End-to-end mine in ≤ 5 minutes** on Sepolia testnet from a clean install:
   ```
   pip install pwm-node
   pwm-node benchmarks --domain imaging        # lists benchmarks
   pwm-node mine <bench_id> --solver demo_solver.py
   ```
   Final step must produce a finalized L4 certificate on Sepolia.

2. **Install-on-clean-machine** test — reference runs this in Docker from
   scratch. Pass criterion: `pip install <your-package>` then the 5-minute
   mine works with zero additional flags.

3. **Wallet key storage** — `pwm-node` must integrate with at least one of:
   OS keychain (libsecret / Keychain / Credential Manager), hardware wallet
   (Ledger or Trezor), or encrypted file with passphrase. Plain-text key files
   are grounds for rejection.

4. **Dry-run parity** — `mine --dry-run` must produce the same Q score and
   `cert_payload` as a live mine, differing only in that the submit-cert tx
   is never broadcast.

5. **Deterministic output** — `inspect <cert_hash>` must produce identical
   output across two runs against the same on-chain state.

6. **Three-platform smoke test** — CI runs the test suite on Linux + macOS
   + Windows. All three pass.

## What you may change

- Language: Python reference; Rust / Go / Node submissions equally welcome
  provided `pip`-equivalent install on all three OSes ships binaries that
  work identically.
- Internal caching, parallelism, RPC batching.
- Additional commands (governance, contribution-weight voting, etc.).
- UX ergonomics (colors, progress bars, completion-on-tab).

## What you may not change

- The mandatory command set above, including flag names and positional args.
- The exit code convention: 0 on success, 1 on input error, 2 on chain error,
  3 on scoring-engine error, 4 on config error. The reference tests assert
  exit codes.
- The JSON output schema emitted by `inspect` and `mine --json` — other tools
  parse it.

## Shadow run

Your `pwm-node` runs alongside the reference for 30 days. We run a nightly CI
job: 20 mines per night across all 5 content domains. Your and the reference's
`cert_payload` must agree field-for-field for the same `(benchmark_id, solver)`
pair. Disagreement is a regression event.

## Payment

- Paid from Reserve.
- Wallet in PR description.
- Mainnet swap 1:1 at Phase 2.
