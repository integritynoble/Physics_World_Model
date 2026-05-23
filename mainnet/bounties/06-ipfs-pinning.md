# Bounty 6 — IPFS Benchmark Pinning

- **Amount:** 50,000 PWM
- **Opens:** at Phase 1 sign-off (~Day 30)
- **Reference implementation:** none — this is an SLA bounty, not a code bounty
- **Acceptance harness:** continuous availability probing from agent-coord
  over a 30-day measurement window

## What you provide

A reliable IPFS pinning service for every benchmark data payload referenced
by a live PWM certificate. When a benchmark's manifest lists a CID for its
ground-truth data, instance seeds, or forward-model fixtures, that CID must
resolve from at least three IPFS gateways within 10 seconds, every time,
for the entire 30-day window.

This is the only bounty with no code deliverable: you operate infrastructure.

## Why this matters

Benchmark data availability is load-bearing for the entire protocol. If
miners cannot fetch a benchmark's instances, they cannot solve, and the
draw pool rolls over. Sustained unavailability on a popular benchmark is
an economic attack surface. The founding team will run a best-effort pin
from day one; this bounty is the professional replacement.

## Interface contract

- **Data source:** you monitor `PWMRegistry` for `ArtifactRegistered` events
  at layer 3 (I-benchmarks) and layer 4 (solutions). Each artifact's manifest
  (off-chain, JSON, pointed to by the registered hash) lists one or more
  IPFS CIDs. You pin all of them.
- **Heartbeat:** you publish a public HTTPS endpoint at
  `https://<your-service>/health` returning:
  ```json
  {
    "service":  "<team-name>",
    "pinned":    <integer count>,
    "last_sync": "<ISO-8601 timestamp of last PWMRegistry poll>",
    "gateways":  ["https://g1.example/ipfs/", "https://g2.example/ipfs/", "..."]
  }
  ```
  agent-coord polls this every 5 minutes during the measurement window.
- **Gateway list:** must expose at least **three** independent IPFS gateways
  under distinct TLS certificates and distinct Autonomous Systems. No
  single-cloud submissions.

## What must pass

Measured continuously over a 30-day window:

1. **Availability ≥ 99.9%.** For every CID in the pin set, a randomly-timed
   fetch from a random one of your three gateways must return within 10
   seconds. 99.9% over 30 days = ≤ 43 minutes of unavailability.

2. **Coverage 100%.** Every benchmark and every solution CID registered in
   PWMRegistry during the window must be pinned within 30 minutes of the
   registration event.

3. **Bit-for-bit integrity.** Every fetched payload must match the CID's
   multihash. One integrity failure voids the award.

4. **Geographic diversity.** Three gateways across three distinct ASes /
   two continents. Measured by IP → ASN lookup at probe time.

5. **No fee extraction.** Fetching a pinned CID must be free for any client.
   You may rate-limit by source IP; you may not gate behind signup, API
   keys, or payment.

## What you may NOT do

- Pin by caching on demand only (i.e., fetching when asked the first time
  and caching afterward). The bounty requires genuine pre-pinning — the
  first request after registration must succeed, not time out.
- Serve pins from a single cloud provider. Multi-region in one cloud still
  counts as one AS.
- Require signup, email, API key, or any human interaction to fetch.
- Charge for reads. Reads are free or the bounty is forfeit.

## Measurement protocol

agent-coord runs the probe:

```
every 5 minutes for 30 days:
  pick a random registered CID from the last 30 days
  pick a random gateway from your /health response
  GET <gateway>/<cid>   with 10s timeout
  record: {cid, gateway, ts, latency_ms, ok, sha256(response)}
  assert sha256(response) matches CID's multihash

end of window:
  availability = 1 - (failed_probes / total_probes)
  pass if availability >= 0.999
```

Probe logs are published to `reviews/bounty-6-probe-log.jsonl` (append-only)
so submitters can audit.

## Submission process

1. Open a GitHub Discussion `[BOUNTY-6] claim intent — <team>`.
2. Spin up your three gateways, point them at an IPFS node cluster you
   control, pre-pin any CIDs already registered as of your start date.
3. Open a PR titled `[BOUNTY-6] <team> — pinning service online` containing
   only:
   - `coordination/agent-coord/bounties/06-submissions/<team>.md` — your
     health endpoint URL, gateway URLs, team contact, payout wallet.
4. agent-coord starts the 30-day measurement window immediately upon merge.
5. At window end, agent-coord publishes the probe summary as
   `reviews/bounty-6-<team>.md`.
6. On pass, 50,000 PWM is released.

Multiple teams may run concurrently. Every team that hits the SLA
independently receives 50,000 PWM — this is not a single-winner bounty.
Budget cap: 3 teams over the lifetime of Phase 1B (150,000 PWM total from
Reserve). Fourth and later teams are acknowledged but not paid from this
bounty line; they may be funded from later Reserve allocations.

## Payment

- Paid from Reserve, one 50,000 PWM allocation per qualifying team, up to 3.
- Wallet listed in PR description.
- Mainnet swap 1:1 at Phase 2.
