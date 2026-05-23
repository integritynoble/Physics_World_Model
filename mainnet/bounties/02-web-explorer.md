# Bounty 2 — Web UI / Explorer

- **Amount:** 80,000 PWM
- **Opens:** when `agent-web` reference merges to `main` (~Day 17)
- **Reference implementation:** `infrastructure/agent-web/`
  (`frontend/`, `indexer/`, `api/`)
- **Acceptance harness:** `infrastructure/agent-web/tests/` (e2e + indexer unit tests)

## What you build

A public web app that makes the PWM protocol legible to anyone without running
a CLI: researchers browsing principles, miners picking benchmarks, investors
checking pool sizes, users inspecting certificates. Three subsystems:

1. **Indexer** — subscribes to on-chain events from `PWMRegistry`,
   `PWMCertificate`, and `PWMReward`; materializes them into a local DB.
2. **API** — REST endpoints serving data from the indexer DB.
3. **Frontend** — the user-facing site.

## Pages (mandatory)

| Route | Content |
|---|---|
| `/` | Protocol overview: total remaining pool, # active principles, last 24h draws |
| `/principles` | All promoted principles: domain, δ, Pool_k, active benchmarks, top PSNR |
| `/principles/<id>` | Principle detail: L-DAG, forward model, specs, benchmarks |
| `/benchmarks` | All benchmarks: pool sizes, Rank 1 holder, ε, ρ weight |
| `/benchmarks/<id>` | Benchmark detail: Ω tier, Track A/B/C pass rates, leaderboard |
| `/leaderboard/<benchmark_id>` | Ranks 1–10: PSNR, Q, SP wallet, draw per epoch |
| `/pools` | Live Pool_k per principle + epoch rollover + T_k balances |
| `/cert/<hash>` | Certificate: S1–S4 verdicts, Q, reward breakdown, challenge status |
| `/submit` | Guided L2/L3 submission (calls `pwm-node` under the hood) |
| `/bounties` | This directory, rendered — open Reserve bounties + acceptance |

Pages may use any routing framework. URL paths above are required.

## Interface contract

- **Contract ABIs:** `interfaces/contracts_abi/*.json` (read-only)
- **Addresses:** `interfaces/addresses.json` (Sepolia now; mainnet at launch)
- **Cert schema:** `interfaces/cert_schema.json`

Your indexer consumes event logs via any JSON-RPC Ethereum provider. Your
frontend must switch between Sepolia and mainnet by reading `addresses.json`
at build time or run time — hardcoding either network voids the award.

## What must pass

1. **Freshness.** Leaderboard and `/pools` must reflect a new certificate
   finalization within **10 minutes** of the on-chain `CertificateFinalized`
   event. Tested continuously during shadow run.

2. **Completeness.** `/principles` must list every promoted principle in
   `PWMRegistry` with no missing rows. Tested against the genesis 500 after
   M1.5.

3. **Correctness.** Every displayed pool number, Q score, and reward amount
   must match an on-chain read within 1 wei / 0.0001 Q.

4. **Availability.** ≥ 99% uptime measured over the 30-day shadow run.

5. **Search.** The top-bar search input must resolve `<principle_id>`,
   `<spec_id>`, `<benchmark_id>`, `<cert_hash>`, and wallet addresses.

6. **Certificate deep-link.** `/cert/<hash>` must render a full report for
   any finalized certificate, pulled straight from event logs (no cache
   staleness > 2 minutes).

## What you may change

- Framework: React / Next.js / Vue / Svelte / SolidJS all acceptable.
- Styling: Tailwind, vanilla CSS, CSS-in-JS; design quality is not graded.
- Indexer DB: SQLite, Postgres, DuckDB, anything queryable from the API.
- Hosting: free tier is fine (Vercel, Netlify, Fly.io, Render). Must be
  publicly reachable for the 30-day shadow run.

## What you may not change

- URL paths listed above.
- The set of contract events indexed (adding events is fine; skipping required
  ones is not).
- The contract ABIs or addresses.json — these are shared interfaces.

## Shadow run

Your site runs alongside the reference for 30 days. agent-coord samples 100
random certificates and 100 random principle pages per week and diff's your
rendered values against the reference. Disagreement > 0 wei on money or > 0.001
on Q triggers investigation; three unexplained regressions void the award.

## Payment

- Paid from Reserve.
- Wallet listed in PR description; confirm on Sepolia before opening.
- Mainnet swap: 1:1 at Phase 2 launch.
