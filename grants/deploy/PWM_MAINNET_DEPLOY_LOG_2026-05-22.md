# PWM Mainnet Deploy Log — Phase 5 Complete (2026-05-22)

**Date:** 2026-05-22
**Status:** Phase 5 Steps 5.1-5.4c COMPLETE. PWM is LIVE on Base mainnet.
**Deployer:** Director (Chengshuai "Abraham" Yang) + sub-GPU
**Chain:** Base mainnet (chainId 8453)
**Deploy decision:** Option A (CASSI + CACTI only — per `coordination/PWM_GENESIS_PRINCIPLES_REALITY_CHECK_2026-05-22.md`)

This doc is the **canonical deploy record** — to be referenced for grant applications, NumFOCUS submissions, audit firm engagements, and future protocol governance.

---

## TL;DR

- **9 contracts deployed** on Base mainnet (Phase 5.2)
- **9/9 verified on Basescan** (Phase 5.4)
- **21M PWM minted + distributed** per genesis allocation (verified via `balanceOf()`)
- **6 CASSI + CACTI artifacts registered** (Phase 5.4a)
- **PWMRegistry handed off to PWMGovernance** 3-of-5 multisig (Phase 5.4b)
- **25/25 post-deploy verifier GREEN x 2** (Phase 5.3 + 5.4c)
- **Soft-launch caps active** for 30-day audit window: `mintingPaused=true`, `transfersPaused=true`, `submissionPermissionless=false`, `STAKING_TVL_CAP_USD=$100`

---

## Step-by-step execution record

### Step 5.1 — Load deployer key + verify derivation ✅

| Item | Value |
|---|---|
| Derived deployer address | `0xA5349f9E42CeC9612E10648609F6E29d0BA0f325` |
| Match against `.env DEPLOYER_ADDRESS`? | ✅ TRUE (rotated from initial `0x0c566f0F…87dEd`) |
| Source | Ledger HW wallet |

### Step 5.2 — Deploy 9 contracts (IRREVERSIBLE) ✅

| Item | Value |
|---|---|
| Deploy script | `deploy/erc20.js --network base` |
| Broadcast timestamp | **2026-05-22T18:52:09.550Z** |
| Total transactions | 33 |
| Exit code | 0 |
| Source commit | `0c731a7c` |

#### Contract addresses (Base mainnet, chainId 8453)

| Contract | Address | Basescan |
|---|---|---|
| **PWMToken** | `0x7326781182b9cDc1eF9Fa147fB689862f893dA14` | https://basescan.org/address/0x7326781182b9cDc1eF9Fa147fB689862f893dA14 |
| **PWMGovernance** | `0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15` | https://basescan.org/address/0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15 |
| **PWMRegistry** | `0x9F91784c2fa884A79473304050C581424E006fbd` | https://basescan.org/address/0x9F91784c2fa884A79473304050C581424E006fbd |
| **PWMTreasuryERC20** | `0xe0FE4A050a926da763907dFA872fA51ba359b061` | https://basescan.org/address/0xe0FE4A050a926da763907dFA872fA51ba359b061 |
| **PWMRewardERC20** | `0x06B341BBFB3435561986f7C1821551E56D909b3D` | https://basescan.org/address/0x06B341BBFB3435561986f7C1821551E56D909b3D |
| **PWMStakingERC20** | `0x88D7860d800Cc68d905751696C3c0B4875Af950b` | https://basescan.org/address/0x88D7860d800Cc68d905751696C3c0B4875Af950b |
| **PWMCertificate** | `0x014492dEfc66D5b58b86027cEB636d4c84289eAe` | https://basescan.org/address/0x014492dEfc66D5b58b86027cEB636d4c84289eAe |
| **PWMMintingERC20** | `0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235` | https://basescan.org/address/0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235 |
| **PWMVesting** | `0x9c57BA6f844dAAecB050D83f31A8279E04a441a9` | https://basescan.org/address/0x9c57BA6f844dAAecB050D83f31A8279E04a441a9 |

(Note: `PWMFaucet` is testnet-only; not deployed to mainnet.)

#### Genesis token distribution (verified via `balanceOf()`)

| Recipient | Amount | % of supply |
|---|---|---|
| PWMMintingERC20 (mining pool) | 17,220,000 PWM | 82% |
| PWM_RESERVE_MULTISIG (3-of-5 Safe) | 2,100,000 PWM | 10% |
| PWM_LIQUIDITY_ADDR (Ledger #1) | 1,050,000 PWM | 5% |
| PWMVesting (4-yr linear, 1-yr cliff to Ledger #1) | 630,000 PWM | 3% |
| **Total** | **21,000,000 PWM** | **100%** |

#### Founder wallet addresses (PWMGovernance 5 founders)

```
0xf1Fa5803daAAaFf89932592ad54F4e7F5e3f7DEE  (Ledger #1)
0x1A84386863cd900DC2d61fCeeea1d09E876B977a  (HW #2)
0xde81b29E42F95C92c9A4Dc78882d0F05D2C81A29  (HW #3)
0x3CeA937cd8114Efa8120C011f1035c9b428C9d05  (HW #4)
0xa53F7e7Bc6B0Cc182d048217646082DDB2DacfE3  (HW #5)
```

Currently all 5 controlled by Director (Path A bootstrap). Founder rotations scheduled Months 1-6 post-mainnet.

### Step 5.2.5 — `publish_abi.js` (sync interfaces) ✅

| Item | Value |
|---|---|
| Output location | `coordination/agent-coord/interfaces/` |
| Files updated | `addresses.json` (canonical base_erc20 slot added) + 7 ABI files (PWMCertificate, PWMGovernance, PWMMinting, PWMRegistry, PWMReward, PWMStaking, PWMTreasury) |
| Source commit | `3dfcc4e7` |

### Step 5.3 — Post-deploy verify `--phase=5` ✅

| Item | Value |
|---|---|
| Checks | 25/25 GREEN |
| Phase | 5 (deployer-owned registry, as expected at this stage) |
| Verifier output | All 25 checks PASS — token distribution + governance handoff (5/6 contracts) + cross-contract wiring + rotation sentinel + 5 soft-launch caps |

### Step 5.4 — Basescan contract verification ✅

| Contract | Status |
|---|---|
| PWMToken | ✅ verified |
| PWMGovernance | ✅ verified |
| PWMRegistry | ✅ verified |
| PWMTreasuryERC20 | ✅ verified |
| PWMRewardERC20 | ✅ verified |
| PWMStakingERC20 | ✅ verified |
| PWMCertificate | ✅ verified |
| PWMMintingERC20 | ✅ verified |
| PWMVesting | ✅ verified |
| **Total** | **9/9 verified** |

### Step 5.4a — Genesis registration (CASSI + CACTI) ✅

Per Director's Option A decision 2026-05-22, registered ONLY 6 v3-verified artifacts (not all 1,591). Batch manifest: `infrastructure/agent-contracts/scripts/batches_genesis_curated/batch_00_cassi_cacti.json`.

| # | Artifact | Hash | Tx hash | Block | Gas |
|---|---|---|---|---|---|
| 1 | L1-025 (CASSI Principle) | `0x03b714a728cf9f81ea6445e73adcd02ed18420babc2a335421ed8d0f594cf19f` | `0xa5f3f882f15505b3eaf3d51030ad4ca10082cdbd9d534cc2701d37b58c826c7c` | 46344137 | 73,753 |
| 2 | L1-027 (CACTI Principle) | `0x358452e94a8000ae267129d34159166c22092f50ea4d2af9f2e91bf2a2839b6f` | `0x79816d94f9b269e6500c620e970f3d889a6a6b6bea2e1ad0bcedfdbcb43c70cb` | 46344137 | 73,741 |
| 3 | L2-025-001 (CASSI Spec) | `0xe08488dd8e1abbfd3e20c846ad2a40254b0651b0a51f3941f7a37c3fec302409` | `0x569982c61f6d79bc2c10c4c4db14dabc255669fb3e13c659c5c61a0ff3728e49` | 46344137 | 96,240 |
| 4 | L2-027-001 (CACTI Spec) | `0x2af46a93084cfb80a43097f4328ad961339e3417582ce9ea08ce93e2fd0ef1de` | `0xc948972c4153b164cfc62ef01304cd59956f46c5f1267db45b37ac5e4ccef15c` | 46344137 | 96,228 |
| 5 | L3-025-001-001 (CASSI Benchmark) | `0x617d3707468f4114715369a653ac2b05ad4c5db94a9346263290b85650a1971a` | `0xed6fc43e465ee6182cb01f328fd037ac73172ff1e7ae9039e913afeeaed90cc6` | 46344138 | 96,240 |
| 6 | L3-027-001-001 (CACTI Benchmark) | `0x185ff24d633974e4d9dbb658b687bd186cbf0ccf4d27145b678fc219d69a2c76` | `0x5ce626436d4c33ef54114ad9f4ac027c5d83714c450a23c8cdfeff51b26acbdf` | 46344138 | 96,240 |
| **Total gas** | | | | | **532,442** |

All 6 status = success.

### Step 5.4b — Registry handoff to PWMGovernance ✅

| Item | Value |
|---|---|
| Script | `scripts/transfer_registry_to_governance.js` |
| Tx hash | `0xbc1df0d2…969a26` (full: see Basescan) |
| Block | 46344140 |
| Status | `0x1` (success) |
| Registry owner before | `0xA5349f9E…0f325` (deployer EOA) |
| Registry owner after | **`0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15` (PWMGovernance 3-of-5 multisig)** |

**Effect:** All future Principle/Spec/Benchmark registrations on PWMRegistry now require governance proposal (3-of-5 multisig + 48h timelock). Deployer EOA can no longer unilaterally register artifacts.

### Step 5.4c — Post-deploy verify default `--phase=5.6` ✅

| Item | Value |
|---|---|
| Checks | 25/25 GREEN |
| Phase | 5.6 (PWMGovernance-owned registry, post-handoff) |
| Critical check | #13: `PWMRegistry.owner() == PWMGovernance` (0x83F210b9...) ✅ confirmed |

---

## Soft-launch caps (ACTIVE for 30-day audit window)

| Cap | Value | Effect |
|---|---|---|
| `STAKING_TVL_CAP_USD` | $100 | Limits total at-risk capital during audit |
| `MINTING_PAUSED` | `true` | `PWMMintingERC20.mintFor` cannot be called; no programmatic emission |
| `TREASURY_TRANSFERS_PAUSED` | `true` | Reserve treasury transfers blocked |
| `submissionPermissionless` | `false` | Only whitelisted submitters during audit |

Per Director's Option A + Phased Architecture Deployment doc:
- **Phase 1a (D9 to D9+30):** Caps active. No mining. Audit window.
- **Phase 1b (D9+30 to D9+180):** Caps lift. Mining ACTIVATES. Permissionless submissions. Full ranked-draw rewards.

---

## What ships at mainnet D9

**Live on Base mainnet:**
- ✅ 9 smart contracts (token + governance + registry + minting + staking + reward + treasury + certificate + vesting)
- ✅ 21M PWM fixed supply minted
- ✅ Genesis distribution complete (4 recipients)
- ✅ 6 verified artifacts on PWMRegistry (CASSI + CACTI L1/L2/L3)
- ✅ PWMRegistry governance-owned (3-of-5 multisig)
- ✅ Soft-launch caps protecting first 30 days
- ✅ All contracts Basescan-verified (public auditability)

**Pending (post-deploy housekeeping):**
- ⏳ Step 5.5: Explorer cutover (update pwm-explorer Docker image at `explorer.physicsworldmodel.org`)
- ⏳ Step 5.6: Announcements (Twitter, HackerNews, arXiv per landing page draft)
- ⏳ Step 5.7: Deploy log (this doc — DONE 2026-05-22)

---

## What ships LATER

- ❌ The other 1,591 stub-tier Principles (stay on Base Sepolia testnet; can be added via governance proposal as polished to v3 via Bounty 7)
- ❌ Mining-pool emission (Phase 1b activation D9+30)
- ❌ Soft-launch caps lift (Phase 1b activation D9+30)
- ❌ LP seeding (Phase 2 activation; Months 6-12)
- ❌ Mine-to-use / user payment for verified runs (Phase 3 activation; Months 12-24)

Per `coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md`.

---

## Director-rotated EOA reconciliation

The deployer address question raised earlier in the conversation is now resolved:

| Source | Address |
|---|---|
| Original `.env DEPLOYER_ADDRESS` | `0x0c566f0F87cD062C3DE95943E50d572c74A87dEd` |
| Derived from `DEPLOYER_PRIVATE_KEY` (Phase 5.1 actual) | **`0xA5349f9E42CeC9612E10648609F6E29d0BA0f325`** |
| `.env DEPLOYER_PRIVATE_KEY` was the rotated key | ✅ Yes (per sub-GPU's earlier `2df945ec` rotation) |

The `.env DEPLOYER_ADDRESS` field was stale (pre-rotation) but the private key was current. Phase 5.1 verification caught the mismatch and confirmed the private key derives the correct rotated address. `.env DEPLOYER_ADDRESS` should be updated to `0xA5349f9E…0f325` for future operational clarity (optional housekeeping).

---

## Total deploy gas cost

Approximate ETH spent on deploy day:

| Step | Tx count | Gas estimate | ETH cost @ ~0.1 gwei |
|---|---|---|---|
| 5.2 deploy (9 contracts + sibling handoffs) | 33 | ~8-12M gas total | ~$15-25 |
| 5.4a genesis registration (6 artifacts) | 6 | 532,442 gas | ~$0.10 |
| 5.4b registry handoff | 1 | ~50,000 gas | ~$0.01 |
| Step 5.4 Basescan verification (free; off-chain) | 0 | — | $0 |
| **Total** | **40 tx** | **~9-13M gas** | **~$15-25** |

Director's Coinbase ETH release covered this comfortably.

---

## Verification breadcrumbs (independent confirmation paths)

External parties can independently verify PWM is live + correctly deployed via:

1. **Basescan** — all 9 contracts verified (source visible)
2. **Tenderly** — deployment transactions traceable
3. **`addresses.json`** — committed in `infrastructure/agent-contracts/addresses.json` (slot `base_erc20`)
4. **post_deploy_verify.js** — anyone can re-run `node scripts/post_deploy_verify.js` with `PWM_RPC_URL` + `PWM_VERIFY_PHASE=5.6` → should see 25/25 GREEN
5. **PWMRegistry.owner()** — call returns `0x83F210b9...` (PWMGovernance), confirming handoff
6. **PWMToken.totalSupply()** — call returns `21,000,000 * 10**18` (21M fixed supply)

---

## Cross-references

- `infrastructure/agent-contracts/addresses.json` — Canonical contract addresses
- `coordination/agent-coord/interfaces/addresses.json` — Mirror for dependents
- `infrastructure/agent-contracts/batch_results/batch_00_cassi_cacti_base_2026-05-22.json` — Genesis registration tx hashes
- `coordination/agent-coord/reviews/batch_registration_batch_00_cassi_cacti_base_2026-05-22.md` — Genesis registration review
- `coordination/PWM_GENESIS_PRINCIPLES_REALITY_CHECK_2026-05-22.md` — Why Option A
- `coordination/PWM_OPTION_A_DEPLOY_STATUS_2026-05-22.md` — Pre-deploy blocker tracking
- `coordination/PWM_NEXT_STEP_ITEMS_3_6_2026-05-22.md` — Items 3 + 6 walkthrough
- `coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — Phase 1a/1b/2/3/4 sequencing
- `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — Demand-side strategy
- `coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — Marketing copy for Step 5.6
- `coordination/wallet/PWM_PHASE_5_PROGRESS_2026-05-17.md` — Phase 5 sub-sequence definition
- `coordination/wallet/PWM_PHASE_5_DEPLOY_DAY_2026-05-09.md` — Phase 5 procedure (canonical)
- `coordination/wallet/STEP_9_5_KILL_SWITCH_REHEARSAL_2026-05-19.md` — Pre-deploy rehearsal record
- `deploy/findings/SECURITY_REVIEW_2026-05-18.md` — Multi-agent security review (2 CRIT + 4 HIGH + 6 MED all fixed)
- `deploy/findings/A4_v2_mythril_triage_2026-05-18.md` — Mythril v1 (official); 0 issues across 9 contracts

---

## Next steps (post-deploy)

### Immediate (D9+0 to D9+7)
- Step 5.5: Explorer cutover (sub-GPU prep; Director SSH to GCP VM)
- Step 5.6: Announcements (Twitter, HackerNews, arXiv companion paper)
- Monitor: chain activity + community questions
- 30-day audit period begins (soft-launch caps active until D9+30)

### Phase 1b activation (D9+30)
- Unpause `MINTING_PAUSED`, `TREASURY_TRANSFERS_PAUSED`, `submissionPermissionless`
- Phase 1b mining ACTIVATES — full ranked-draw rewards live
- Outreach to first 30-50 mini-competition candidates per `PWM_USER_ACQUISITION_STRATEGY` §6.3

### Phase 2 (Months 6-12)
- LP seeded (Uniswap v3 PWM/USDC; 1.05M PWM + $50K USDC)
- Multi-benchmark expansion (PWM-CI-2, PWM-CI-3, PWM-MED-1)
- T_k royalty distributions
- Foundation 501(c)(3) outcome (NumFOCUS Round 4)

---

## Acknowledgments

This deploy was the result of:
- **Director (Chengshuai "Abraham" Yang)** — overall direction, founder rotation, key management, hardware wallet operations
- **Sub-GPU (Chengshuai's GPU server)** — Phase 5 sub-sequence revision, deploy script execution, Phase 4 preflight, Phase 5.4a/b/c execution
- **Main-CPU (Claude on main-CPU)** — Phase 5 progress documentation, Mythril overnight scans, security review aggregation
- **Multi-agent security review (2026-05-18)** — A1-A10 + Slither + Mythril + Echidna agents — 2 CRIT + 4 HIGH + 6 MED all fixed before deploy

---

## Bottom line

**PWM is live on Base mainnet as of 2026-05-22T18:52:09Z.**

The deployed protocol is a verified AI4Science platform with:
- 21M PWM fixed supply
- 6 founder-verified genesis artifacts (CASSI + CACTI)
- 3-of-5 multisig governance
- 30-day audit window via soft-launch caps
- Open path to Phase 1b mining activation at D9+30

This is the start, not the end. The next 6 months determine whether PWM follows the Chainlink / Arweave pattern (sustained) or the Helium / Filecoin / UMA pattern (~90% drop). Strategic docs in `coordination/` lay out the path. Director's Option A decision keeps the launch honest.

---

*This is the canonical deploy log for PWM mainnet 2026-05-22. Quote from this doc in grant applications, NumFOCUS submissions, future audit firm engagements, and protocol governance proceedings.*
