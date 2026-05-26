# PWM Reserve Bounties

**Total active commitment:** ~1,238,000 PWM across **8 bounties** in OPEN or SPEC PUBLISHED state (7 infrastructure + 1 content), paid from the 2,100,000 PWM Reserve. **2 additional bounties** (#9 MCP server, #10 Mobile UX) are in **RESERVED** state — spec docs published as planning artifacts; not yet claimable; gate conditions documented below. External developers/authors compete against a reference implementation (or, for Bounty #7, a reference content exemplar) published by the PWM founding team; submissions must pass the reference test harness or verifier-agent gates to qualify.

Published by: agent-coord
Last updated: **2026-05-22** — reclassified Bounty 9 (MCP server, 25K PWM) + Bounty 10 (Mobile UX, 40K PWM) from `SPEC PUBLISHED` → `RESERVED` per the 2026-05-20 website-first phasing decision (`coordination/PWM_API_VS_WEBSITE_2026-05-20.md`): agent infrastructure (MCP) is Year 3+ unless the walled-garden trigger fires; mobile UX is Foundation-discretionary, not bounty-board scope. Spec docs at `09-mcp-server.md` + `10-mobile-ux.md` retained as planning artifacts. Added "Founder vs external — who builds what" doctrine section below. Prior updates: 2026-05-22 (bounty docs created from `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` §3 by main-CPU); 2026-05-01 (Bounty 7 v2/v3 scope refresh).

> **Note on Reserve resizing (2026-04-30).** Director is considering a pre-mainnet rebalance of the Reserve from 10% (2.1M PWM) to 13% (2.73M PWM) per `pwm-team/pwm_product/genesis/PWM_RESERVE_RESIZING_FOR_V2_V3.md`. If approved, the bounty totals below stay the same (no per-bounty change), but the un-allocated Reserve pool increases by ~630K PWM. Decision pending Director sign-off as part of mainnet Step 7b.

## Status overview

| # | Bounty | PWM | Opens when | Reference impl | Status |
|---|---|---|---|---|---|
| 1 | [Scoring engine](01-scoring-engine.md)         | 200,000 | agent-scoring merged | `infrastructure/agent-scoring/pwm_scoring/` | **OPEN** — reference impl on main 2026-04-21 (81 tests pass) |
| 2 | [Web UI / Explorer](02-web-explorer.md)         |  80,000 | agent-web merged     | `infrastructure/agent-web/`                 | **OPEN** — reference impl on main 2026-04-21 (17 tests pass) |
| 3 | [pwm-node CLI](03-pwm-node-cli.md)              | 100,000 | agent-cli merged     | `infrastructure/agent-cli/pwm_node/`        | **OPEN** — reference impl on main 2026-04-21 (73 tests: 68 mocked + 5 opt-in e2e). Bounty acceptance procedure: `agent-cli/tests/e2e/README.md` |
| 4 | [Mining client (CP role)](04-mining-client.md)  |  80,000 | agent-miner merged   | `infrastructure/agent-miner/pwm_miner/`     | **OPEN** — reference impl on main 2026-04-21 (87 tests across all 6 modules). Bounty acceptance procedure: `agent-miner/tests/e2e/README.md` |
| 5 | [Smart contract security program (tiered)](05-contracts-competing.md) | 500,000 (5A 200K · 5B 100K · 5C 100K · 5D 100K) | D9+0 for 5A/5C/5D; D9+30 for 5B | `infrastructure/agent-contracts/contracts/` at audit-v3 baseline (9 contracts, 199 tests) | **SPEC PUBLISHED** — reframed 2026-05-22 from "competing impl" to tiered security program per canonical `pwm_overview1.md` line 1579 "best audited implementation". Tiers: 5A formal audit firm · 5B formal verification · 5C continuous fuzzing · 5D Immunefi bug-bounty pool seed |
| 6 | [IPFS benchmark pinning](06-ipfs-pinning.md)    |  50,000 | Phase 1 sign-off     | (no reference impl — SLA-driven)            | SPEC PUBLISHED — opens Phase 1 sign-off |
| 7 | [Genesis Principle Polish (tiered)](07-genesis-principle-polish.md) | ~168,000 | Phase 2 launch + 7 days | `cassi.md`/`cacti.md` + L1/L2/L3-003/004 JSON as exemplars | SPEC PUBLISHED — per-principle claim model; opens post-mainnet |
| 8 | [LLM-routed / NL matcher](08-llm-matcher.md)     |  60,000 | CASSI/CACTI §10.2.1 gate passes | `agent-cli` + `agent-web` faceted matcher (deterministic, LLM-free) | SPEC PUBLISHED — opens once CASSI/CACTI hit `TWO_ANCHOR_MVP_LOCKED` per `MVP_FIRST_STRATEGY.md` |
| 9 | [MCP server](09-mcp-server.md) | 25,000 | **RESERVED — opens only if walled-garden trigger fires** per `coordination/PWM_API_VS_WEBSITE_2026-05-20.md` | None (MCP spec is the reference at https://modelcontextprotocol.io) | **RESERVED** — spec preserved as planning artifact; agent infrastructure is Year 3+ per canonical phasing |
| 10 | [Mobile UX](10-mobile-ux.md) | 40,000 | **RESERVED — opens only on Foundation board approval** (out of bounty-board scope per AUTH_AND_WALLET_STRATEGY.md; non-crypto-native mobile users are SpecLab scope, not explorer scope) | Bounty 2 reference impl extended for mobile | **RESERVED** — spec preserved as planning artifact; Foundation discretionary pool (~862K PWM, Layer 3) is the right funding channel if a strong mobile proposal lands post-launch |

Bounties open **rolling**, not all at once. Infrastructure bounties (1-4) open
the day the reference implementation merges to `main`, so external developers
have a concrete harness to build against. Bounty 5's tiers open per the
schedule in `05-contracts-competing.md` (5A/5C/5D at D9+0; 5B at D9+30) — the
tiered security program is reframed from the original "competing impl" spec
and runs in parallel with the soft-launch window. Bounty 6 opens at Phase 1
sign-off (~Day 30). **Bounty 7 is structurally different** — it's a per-principle
claim model (not one-PR-one-payout) opening at Phase 2 launch.

**Bounty 7 scope refreshed 2026-05-01.** The original 2026-04-23 framing
(2 done CASSI/CACTI + ~498 un-polished baseline) is updated for the
2026-04-29/30 v2/v3 expansion. New scope:
- **Tier A (founder-authored, no Bounty 7 cost): ~30 anchors** = CASSI + CACTI
  + 8 v3 standalone multi-physics medical imaging Principles (L1-503..L1-510)
  + 2 newly-authored analytical cores (L1-511 PillCam, L1-518 XRD)
  + 19 v2 PWDR Principles (L1-512..L1-517, L1-519..L1-531).
  These 30 are already at v2/v3 schema depth and don't consume Bounty 7 funds.
- **Tier B (anchors): ~15-20** at 2,000 PWM each — pool of next-round anchors
  Director hasn't yet picked from `PWM_V3_MEDICAL_IMAGING_CANDIDATES.md` or
  related candidate docs; 23 candidates remain after the 28 v2/v3 promoted.
- **Tier C (standard): ~200** at 500 PWM each — long-tail v1 baseline
  Principles awaiting external-author re-polishing.
- **Tier D (specialty): ~280** at 100 PWM each — niche / specialty
  Principles after Director's per-Principle removal review per
  `pwm-team/pwm_product/genesis/PWM_V1_REMOVAL_CANDIDATES.md` (currently
  139 of 502 baseline are flagged as Category A widefield-template fan-out
  artifacts; some subset may be removed before mainnet, freeing those slots).

Net Bounty 7 budget unchanged at ~168,000 PWM. Effective per-Principle
payouts within each tier remain stable (the σ-over-unclaimed formula).
The expansion is **net-positive for Bounty 7 economics:** more polished
anchors at launch, same budget for long-tail polish post-launch.

**Bounty 8** pairs with the PWM-native
reference matcher ("faceted floor") that ships as part of `agent-cli` and
`agent-web`: PWM intentionally does NOT host an LLM-routed matcher itself,
so Bounty 8 is third-party territory from day-one of its opening. Opens only
*after* CASSI + CACTI clear the §10.2.1 promotion gate in
`MVP_FIRST_STRATEGY.md` — publishing earlier would invite over-fitting to a
harness that only covers 2 anchors.

## Bounties 9 and 10 — RESERVED (rationale)

Bounties 9 (MCP server, 25K PWM) and 10 (Mobile UX, 40K PWM) have spec docs
published as planning artifacts but are **not currently claimable**. They sit
in `RESERVED` state pending explicit trigger conditions:

**Bounty 9 — MCP server.** Per the canonical 2026-05-20 website-first phasing
decision (`pwm-team/coordination/PWM_API_VS_WEBSITE_2026-05-20.md`),
PWM's roadmap is: Year 1 website → Year 2 API exposure → Year 3+ agent
infrastructure. MCP is agent infrastructure (it exposes PWM as tool-callable
to AI assistants). The ONLY override is the **walled-garden trigger**: if a
major AI lab (Anthropic / Google / OpenAI / Microsoft / state-funded
equivalent) announces or ships a captive verification registry, Track 7
agent SDK (including MCP) accelerates from Year 3+ to Year 1.5. Until that
trigger fires, Bounty 9 stays RESERVED. Opening it earlier would conflict
with the website-first phasing and signal a roadmap PWM does not intend to
ship for ~3 years.

**Bounty 10 — Mobile UX.** PWM's Year 1 audience per the phasing doc =
researchers, miners, grant reviewers, journalists — predominantly desktop
users for the relevant use-cases (reading proofs, inspecting leaderboards,
writing grant copy). The "60% mobile" general-web statistic does not apply
to technical research tooling. Mobile-first onboarding is most valuable for
**SpecLab non-crypto-native users** (clinicians, students), which
`coordination/strategy/AUTH_AND_WALLET_STRATEGY.md` explicitly assigns to
the SpecLab product surface — **out of the public-explorer bounty scope**.
If a strong mobile-UX proposal arrives post-launch from a developer who
wants to build it anyway, fund it from the **Foundation discretionary pool**
(Layer 3, ~862K PWM available per `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md`),
not from this bounty board. Pre-publishing as an OPEN bounty creates the
obligation without the strategic need.

These RESERVED entries preserve the spec work already done (the 09-* and 10-*
docs are good planning artifacts) without distorting the strategic signal of
the OPEN bounty board.

## Founder vs external — who builds what

Canonical doctrine per `long-term-vision/pwm_overview1.md` §Bootstrap Phase
(lines 1657-1681):

### Mandatory founder work (only founders can build)

| What | Why |
|---|---|
| Protocol specification (the paper) | Only founders can define the protocol rules |
| Genesis 500 Principles (L1+L2+L3 content) | Without content there is nothing to mine; no external dev builds a mining client for an empty protocol |
| Smart contracts (Bounty #5 reference impl) | Ground truth — a bug here loses people's money; the founding team must own this |

### Reference implementations founders SHOULD build (acceptance harness)

These are technically delegable to external bounties per the canonical
doctrine, but in practice the founding team builds reference implementations
for them anyway because **without a reference, there is no concrete test
harness for external bounty submissions to be evaluated against**. All five
already shipped:

| # | Reference impl path | Status |
|---|---|---|
| 1 | `infrastructure/agent-scoring/` | ✅ Shipped (81 tests pass) |
| 2 | `infrastructure/agent-web/` | ✅ Shipped (17+ tests; SIWE wallet shipped per Track 3a commit `b274aaef`) |
| 3 | `infrastructure/agent-cli/` | ✅ Shipped (73 tests) |
| 4 | `infrastructure/agent-miner/` | ✅ Shipped (87 tests) |
| 5 | `infrastructure/agent-contracts/` | ✅ Shipped (199 tests; audit-v3 tag; multi-agent review complete) |

### What founders should NOT build

| What | Why founders abstain |
|---|---|
| **Bounty #1-#4 competing implementations + Bounty #5 audit/verification work** | Founders write ONE reference for #1-#4; bounties pay for second/third independent implementations. Bounty #5 reframed 2026-05-22 from "competing impl" to tiered security program (audit firm + formal verification + continuous fuzzing + Immunefi pool) — Director cannot claim any 5A/5B/5C tier under Path A bootstrap (self-dealing rule). All restraints follow `pwm_overview1.md` §10 doctrine: *"the founding team takes a minimal bootstrapping allocation (3%) and earns ongoing income through the same protocol channels as everyone else."* |
| **Bounty #6 IPFS pinning reference impl** | Per the bounty spec: "no reference impl — SLA-driven." Founders **operate** at least one pinning node (Pinata / Web3.storage / self-hosted IPFS) but do not write the software. |
| **Bounty #7 Tier B/C/D Principle polish** | Founders authored Tier A (~30 anchors: CASSI/CACTI + 28 v2/v3 expansion) at zero Bounty-7 cost. Tier B/C/D (long-tail ~500 Principles at 100-2000 PWM each) is external-author scope by design. |
| **Bounty #8 LLM-routed matcher** | Per this INDEX line above + the bounty spec: *"PWM intentionally does NOT host an LLM-routed matcher itself, so Bounty 8 is third-party territory from day-one of its opening."* The deterministic faceted-floor matcher ships in agent-cli + agent-web as part of bounties #2+#3. |
| **Claiming the 3% Founding Team allocation AND Bounty #5 simultaneously** | The 3% allocation (`PWMVesting`, 0.63M PWM, 4-year vest with 1-year cliff) is the canonical pre-launch bootstrapping compensation. Director would in principle qualify for the smart-contract reference impl bounty (500K PWM) but **does not claim it** under Path A bootstrap — that would be self-dealing during the period the Director holds all 5 multisig keys. Post-rotations (Path A → Path B), retroactive Reserve grants are governance-decided per `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` §3. |

### Canonical build order

Per `pwm_overview1.md` §Phase 1A (contracts must be public + correct before
any bounty developer writes a line of code):

```
1. Smart contracts (#5)         ← ground truth; everything else depends on them
2. Genesis 500 Principles       ← without content, mining cannot start
3. Scoring engine (#1)          ← needed to verify L4 solutions on-chain
4. pwm-node CLI (#3)            ← human + machine interaction layer
5. Mining client (#4)           ← creates L4 activity (the demand-side metric)
6. Web Explorer (#2)            ← public visibility for researchers / journalists
7. IPFS pinning (#6) — operate  ← contract a service or self-host; do not build
8. LLM matcher (#8) — defer     ← post-CASSI/CACTI §10.2.1 gate; external only
```

At D9 the founder-build list is complete for items 1-6. Item 7 is an
operational task during D9+0-7. Item 8 stays deferred per the §10.2.1 gate.

After D9, the Director's role shifts from "build" to "operate + review +
paper" per the time-allocation table in
`memory/project_active_session_state.md`.

## Common acceptance structure

Every bounty follows the same evaluation path:

1. **Scope check** — submission implements the mandatory interface listed in
   the bounty spec. Anything off-spec is rejected at triage.
2. **Reference test suite** — submission must pass every test the founding
   team's reference implementation passes. Tests are in the reference's
   `tests/` directory and are the authoritative acceptance harness.
3. **Interface parity** — submission must be a drop-in replacement for the
   reference. The CLI calls the scoring engine as a library; if your scoring
   engine rewrite breaks the CLI, it fails.
4. **30-day shadow run** — accepted submissions run alongside the reference
   on testnet for 30 days. Any correctness or availability regressions surface
   here. Serious regressions void the award.
5. **Payment** — after the shadow run, PWM is released from Reserve escrow to
   the submission's designated wallet.

## Claiming a bounty

External developers follow this flow:

```
1. Read the bounty spec + the reference implementation's CLAUDE.md
2. Open a GitHub Discussion in integritynoble/pwm-public titled:
     "[BOUNTY-<N>] claim intent — <your team name>"
   listing intended tech stack + contact + estimated delivery
3. Fork the repo, build against the interface, open a PR titled:
     "[BOUNTY-<N>] <component> submission — <your team name>"
4. agent-coord runs the reference test suite against your PR
5. If tests pass, your PR enters the 30-day shadow run
6. On successful shadow run, agent-coord writes
     coordination/agent-coord/reviews/bounty-<N>.md with the approval
   and the multisig releases PWM to the wallet listed in your PR description
```

Multiple submissions per bounty are accepted. First qualifying submission
wins the full amount; later qualifying submissions receive runner-up rewards
only if the director opens them (not guaranteed).

## Security + IP

- All submissions must be MIT-licensed and publicly readable.
- Do not submit code that includes paid or closed-source dependencies
  unless they are free for research and commercial use without restriction.
- Bounty payment is in PWM (Sepolia testnet initially; mainnet PWM at launch).
- The PWM team does not collect equity, KYC, or personal data from bounty
  participants. Supply only the payout wallet.

## Questions

Open a GitHub Discussion under the `bounties` category in
`integritynoble/pwm-public`. agent-coord triages within 72h.
