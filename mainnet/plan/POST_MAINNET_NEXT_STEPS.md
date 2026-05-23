# PWM Post-Mainnet Next Steps

**Date:** 2026-05-22 (D9 — mainnet deployed 2026-05-22T18:52:09Z)
**Author:** Director + sub-GPU
**Status:** Active execution checklist. Update as items complete.
**Supersedes:** `PWM_OPTION_A_DEPLOY_STATUS_2026-05-22.md` post-deploy section; `PWM_NEXT_STEP_ITEMS_3_6_2026-05-22.md` (both now archived — Phase 5 complete)

PWM is live on Base mainnet (chainId 8453). 9 contracts deployed, 25/25 verifier GREEN ×2, 6 CASSI+CACTI genesis artifacts on-chain, independent 10/10 PASS. This doc tracks everything from D9 forward.

---

## Phase summary

| Phase | Window | Gate |
|---|---|---|
| **Phase 1a** (now) | D9 → D9+30 (~2026-06-21) | Soft-launch caps active; audit window; no minting |
| **Phase 1b** | D9+30 → D9+180 (~2026-11-18) | Mining ACTIVE; PWM-CI-1 open; 5-30 external submitters |
| **Phase 2** | Months 6-12 | LP seeded; multi-benchmark; T_k royalty distributions |
| **Phase 3** | Months 12-24 | Mine-to-use; users pay PWM for advanced verified runs |

---

## IMMEDIATE — Step 5.5: Explorer cutover (do today)

Source-side changes are already committed (commit `1fd068e3`). The `base` slot in `addresses.json` is now populated (this commit). What remains is the VM-side execution.

**SSH to GCP VM and run:**

```bash
ssh -i ~/.ssh/google_compute_engine spiritai@34.63.169.185
```

Then on the VM:

### a) Pull latest source

```bash
cd /opt/pwm-explorer          # or wherever the agent-web repo is checked out
git pull origin main
```

### b) Update nginx config

```bash
# Backup existing config
sudo cp /etc/nginx/sites-available/pwm-explorer /etc/nginx/sites-available/pwm-explorer.bak

# Copy new config (has 3 server blocks: redirect, primary, testnet subdomain)
sudo cp pwm-team/infrastructure/agent-web/nginx.example.conf \
        /etc/nginx/sites-available/pwm-explorer

# Edit to set your actual domain/cert paths if needed, then test
sudo nginx -t
```

### c) Issue Let's Encrypt certs for new domains

```bash
sudo certbot --nginx \
  -d physicsworldmodel.org \
  -d explorer.physicsworldmodel.org \
  -d test.physicsworldmodel.org
```

(If DNS for `physicsworldmodel.org` isn't pointed at 34.63.169.185 yet, do that first in your registrar — A record → 34.63.169.185.)

### d) Docker rebuild and restart

```bash
docker compose build
docker compose up -d
```

### e) Verify all 3 indexers start

```bash
docker logs pwm-explorer 2>&1 | grep -i "indexer\|starting\|base\|sepolia" | head -20
```

Expected output: Base mainnet indexer + Base Sepolia indexer + Eth Sepolia indexer all starting.

### f) Smoke test

```bash
curl -s https://physicsworldmodel.org/api/status | python3 -m json.tool
# Should show: {"chain": "base", "chainId": 8453, ...}

curl -s https://physicsworldmodel.org/api/artifacts?limit=5 | python3 -m json.tool
# Should show CASSI + CACTI L1/L2/L3 artifacts
```

**Done when:** `physicsworldmodel.org` serves Base mainnet (chainId 8453) as default; CASSI + CACTI visible on leaderboard; `test.physicsworldmodel.org` serves testnet view.

---

## IMMEDIATE — Step 5.6: Wave 1 announcements (after 5.5)

Paste-ready copy is in `coordination/PWM_ANNOUNCEMENT_BUNDLE_2026-05-22.md`.

**Strategy:** quiet Wave 1 now (Twitter/X thread only). Save HackerNews + Reddit + LinkedIn for **Wave 2 on 2026-06-21** (D9+30) when mining activates and there's a real CTA.

**Wave 1 message (Twitter/X):** "PWM mainnet live on Base. CASSI + CACTI compressive imaging principles registered. Soft-launch audit window through June 21. PWM-CI-1 benchmark opens then — top-3 methods win up to 5,000 PWM."

See `PWM_ANNOUNCEMENT_BUNDLE_2026-05-22.md` §1 for the full 15-tweet thread.

---

## THIS WEEK (D9+0 to D9+7)

### Track 7: Apply for audit funding

**Priority: HIGHEST.** Grant applications take weeks to process; the D9+30 audit window is tight.

| Funder | URL | Award | Time to draft |
|---|---|---|---|
| Base Builder Grant | https://base.org/grants | $5-25K | 8-10 hr |
| Ethereum Foundation ESP | https://esp.ethereum.foundation | $10-50K | 12-15 hr |

Start with Base Builder (PWM is Base-native; strongest fit). Draft + submit by D9+7.

Supporting evidence to cite in applications:
- 188/188 tests GREEN
- 10-agent multi-agent security review (2 CRITICAL + 4 HIGH + 6 MEDIUM all fixed)
- Slither + Mythril clean
- 25/25 post-deploy verifier GREEN ×2
- Independent 10/10 on-chain verification
- `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md`
- `infrastructure/agent-contracts/SECURITY_REVIEW.md`

Audit firm options once ≥$15K lands:
- $15-25K: Cantina, Code4rena private contest, Spearbit standard
- $25-40K: Spearbit, Zellic, OpenZeppelin, Halborn

### Track 6: Community setup

Required before NumFOCUS application (Track 4b, target Oct 15).

| Asset | Action |
|---|---|
| Discord server | Create server; set up #announcements, #mining, #dev channels; Code of Conduct pinned |
| GitHub Discussions | Enable on `pwm-public` repo |
| `conduct@physicsworldmodel.org` | Configure DNS MX + mail forwarding to Director email |

Estimate: 1-2 hours total.

### Track 4a: Co-founder recruitment outreach

**Hard prerequisite for NumFOCUS (Oct 15) and founder rotations (Months 1-6).**

Send outreach to 8-10 candidates using template in `numfocus/application/CONTRIBUTORS.md §Section 5`.

Target slots:
| Slot | Profile | Where |
|---|---|---|
| #2 | Computational imaging researcher (non-UTSW) | Conference network; collaborator referrals |
| #3 | Open-source contributor (no academic affiliation) | OSS communities, Discord servers |
| #4 | Crypto/blockchain ecosystem contributor | Base Builder community, Ethereum researcher community |

Estimate: 2 hours to draft + send outreach emails.

---

## D9+30 (~2026-06-21) — Phase 1b activation

This is the hard deadline for mining activation.

### Governance proposals (3 executeCall proposals, 48h timelock each)

Run sequentially via PWMGovernance (3-of-5 multisig). Each needs: propose → 48h wait → execute.

```
Proposal 1: PWMMintingERC20.setMintingPaused(false)
Proposal 2: PWMTreasuryERC20.setTransfersPaused(false)
Proposal 3: PWMRegistry.setSubmissionPermissionless(true)
```

Allow ~1 week (3 × 48h timelocks + buffer). Start proposals at D9+21 so execution lands at D9+28-30.

### Wave 2 announcements

Post when minting is unpaused and submissions are open:
- HackerNews "Show HN: Physics World Model — verified AI4Science benchmarks on Base"
- Reddit r/MachineLearning, r/ethereum
- LinkedIn post
- Targeted outreach to 10-15 CASSI researchers (per `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md §6.3`)

Prize announcement: **PWM-CI-1 CASSI benchmark open. Submit your reconstruction method. Top 3 win 5,000 / 2,500 / 1,500 PWM.**

---

## D9+30 to D9+90: PWM-CI-1 launch preparation

Owner: Director (strategic) + Heyang intern (execution, weeks 10-16).

### "One of each" launch checklist

| # | Component | Effort |
|---|---|---|
| 1 | Landing page at `physicsworldmodel.org/ci1` | 1-2 days |
| 2 | GitHub repo `integritynoble/pwm-ci-1` with data scripts + leaderboard | 2-3 days |
| 3 | Baseline method (GAP-TV on CASSI — already computed, 23.02 dB avg) | Done |
| 4 | Evaluation script (PSNR/SSIM, deterministic) | 2-3 days |
| 5 | Prize pool governance proposal: 10,000 PWM from Reserve | 48h timelock |
| 6 | Technical report (arXiv companion) | 5-10 days |
| 7 | Community channel (Discord #pwm-ci-1) | 1 day |
| 8 | Submitter guide | 1-2 days |

**Total: ~3-5 weeks of focused work.** Fits Heyang weeks 10-16.

Heyang checkpoint decision at D9+45: Scenario X (unchanged low-dose CT plan) vs Scenario Y (re-scope to PWM-CI-1 support). Per `papers/INTERN_WORKPLAN_LOW_DOSE_CT_HEYANG_ZHAO_2026-05-17.md`.

---

## D9+30-60: Engage audit firm (if Track 7 grants land)

| Grant landed | Action |
|---|---|
| ≥$15K | Engage Cantina / Code4rena / Spearbit for 2-week scoping |
| ≥$25K | Standard 5-week fieldwork with Spearbit / Zellic / OZ |
| <$15K at week 8 | Bug bounty fallback; maintain soft-launch caps |

After audit: governance proposes cap-raise via 3 executeCall proposals (48h × 3). Full protocol live.

---

## Months 1-6: Founder rotations (Track 1d)

Replace Director's 5 keys one-at-a-time with co-founder keys. Hard prerequisite: Track 4a co-founder recruitment confirmed.

| Month | Action |
|---|---|
| 1-2 | Rotation #1 (co-founder #2's key replaces 1 Director slot) |
| 2-3 | Rotation #2 (co-founder #3's key) |
| 3-4 | Rotation #3 (co-founder #4's key) |
| 4-5 | Rotation #4 — Director controls only 1 of 5; genuine 3-of-5 achieved |

Each rotation: governance proposal + 48h timelock + execute.

---

## Track 4: Foundation & Funding calendar

| Date | Action |
|---|---|
| 2026-06-15 | Check if NumFOCUS portal reopened |
| 2026-07-01 | Decision: Round 3 (Jul 15 deadline) vs Round 4 (Oct 15) — Round 4 recommended |
| **2026-08-25** | **Earliest date for crypto-foundation grant applications (locked policy)** |
| 2026-08-25 to 2026-10-12 | Apply Base Builder, EF ESP, 0xPARC, Filecoin, Gitcoin, Mantle, Optimism RetroPGF |
| 2026-10-15 | NumFOCUS Round 4 deadline |
| 2026-10-31 | Expected: $190K; upside: $400-700K from Tier 1 grants |

---

## Track 8: Papers (active)

| Paper | Venue | Submit by | Status |
|---|---|---|---|
| 8a InverseNet | ECCV/NeurIPS 2026 | 2026-Q3 | Polish + resubmit; visual examples (G1 gap) |
| 8b PWMI-CASSI | ECCV 2026 | 2026-Q3 | Final results + figure quality pass |
| 8c Universal Simulation | Nature | 2026-Q3 | Awaiting presubmission enquiry response |
| 8d PWM Flagship | Nature | 2026-Q4 | Needs: hardware validation, state-of-the-art comparison, 8a citable |
| 8e Finite Primitive Theorem | Nature SI / math | 2026-Q4 | Companion to 8d |

Papers 8a + 8b are the immediate priority — both are implementation-complete, need polish only.

---

## Track 5: UTSW PI transition (begin Aug 2026)

| Phase | When | Action |
|---|---|---|
| 1 | 2026-08 → 2026-10 | UTSW COI Office disclosure (confidential from Dr. Zaman) |
| 2 | 2026-08 → 2026-10 | Identify 2-3 supportive UTSW faculty as candidate new mentors |
| 3 | 2026-11 → 2026-12 | Initial conversations with candidate PIs |
| 4 | 2027-Q1 | Formalize new mentor; PI change paperwork |
| 5 | 2027-Q2 | Draft NIH R21 specific aims with new PI |
| 6 | 2027-Q3 | Submit R21 (Jun 16 or Oct 16 deadline) |

**Do NOT disclose PWM to Dr. Zaman before new mentor is in place.**

---

## Deployed contract addresses (Base mainnet, chainId 8453)

| Contract | Address |
|---|---|
| PWMToken | `0x7326781182b9cDc1eF9Fa147fB689862f893dA14` |
| PWMGovernance (3-of-5 multisig) | `0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15` |
| PWMRegistry | `0x9F91784c2fa884A79473304050C581424E006fbd` |
| PWMTreasuryERC20 | `0xe0FE4A050a926da763907dFA872fA51ba359b061` |
| PWMRewardERC20 | `0x06B341BBFB3435561986f7C1821551E56D909b3D` |
| PWMStakingERC20 | `0x88D7860d800Cc68d905751696C3c0B4875Af950b` |
| PWMCertificate | `0x014492dEfc66D5b58b86027cEB636d4c84289eAe` |
| PWMMintingERC20 | `0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235` |
| PWMVesting | `0x9c57BA6f844dAAecB050D83f31A8279E04a441a9` |

Soft-launch caps active until D9+30: `mintingPaused=true`, `transfersPaused=true`, `submissionPermissionless=false`, `STAKING_TVL_CAP=$100`.

Full deploy record: `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md`

---

## Decision points

| Date | Decision | Default |
|---|---|---|
| D9+14 | PWM-CI-1 dataset choice: CASSI vs public low-dose CT | CASSI (Director's stated preference) |
| D9+21 | Start Phase 1b governance proposals (3 × 48h timelock) | Required for D9+30 activation |
| D9+45 | Heyang Scenario X vs Y checkpoint | X unless CI-1 needs hands |
| 2026-06-15 | NumFOCUS portal check | Defer to Round 4 |
| 2026-07-01 | NumFOCUS Round 3 vs Round 4 | Round 4 (Oct 15) — recommended |
| 2026-08-25 | First Tier 1 grant applications (locked policy gate) | Apply Base Builder first |

---

*This doc is the single source of truth for post-D9 execution. Update as items complete. Supersedes per-phase walkthrough docs created during the pre-deploy sprint.*
