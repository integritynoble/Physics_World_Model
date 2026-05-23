# PWM Mainnet (Base) — Protocol Documentation

**Status:** LIVE on Base mainnet (chainId 8453) as of **2026-05-22T18:52:09Z**
**Soft-launch audit window:** through **2026-06-21** (D9+30)
**Phase 1b activation (mining open):** 2026-06-21 onwards
**Verification:** 10/10 independent on-chain checks PASS (see [DEPLOY_LOG.md](DEPLOY_LOG.md))

This directory contains the **protocol-layer artifacts** for PWM (Physics World Model) — separate from the evaluation harness / algorithm catalog in the rest of this repo. PWM is a verified AI4Science platform; the contracts below provide the cryptographic substrate that makes the benchmarks trustworthy.

---

## Quick links

- **🌐 Public landing:** https://physicsworldmodel.org
- **🔍 Block explorer:** https://explorer.physicsworldmodel.org
- **🧪 Testnet view:** https://test.physicsworldmodel.org

---

## Deployed contracts (Base mainnet, chainId 8453)

| Contract | Address | Source |
|---|---|---|
| **PWMToken** | [`0x7326781182b9cDc1eF9Fa147fB689862f893dA14`](https://basescan.org/address/0x7326781182b9cDc1eF9Fa147fB689862f893dA14) | [contracts/PWMToken.sol](contracts/PWMToken.sol) |
| **PWMGovernance** (3-of-5 multisig) | [`0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15`](https://basescan.org/address/0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15) | [contracts/PWMGovernance.sol](contracts/PWMGovernance.sol) |
| **PWMRegistry** | [`0x9F91784c2fa884A79473304050C581424E006fbd`](https://basescan.org/address/0x9F91784c2fa884A79473304050C581424E006fbd) | [contracts/PWMRegistry.sol](contracts/PWMRegistry.sol) |
| **PWMTreasuryERC20** | [`0xe0FE4A050a926da763907dFA872fA51ba359b061`](https://basescan.org/address/0xe0FE4A050a926da763907dFA872fA51ba359b061) | [contracts/PWMTreasuryERC20.sol](contracts/PWMTreasuryERC20.sol) |
| **PWMRewardERC20** | [`0x06B341BBFB3435561986f7C1821551E56D909b3D`](https://basescan.org/address/0x06B341BBFB3435561986f7C1821551E56D909b3D) | [contracts/PWMRewardERC20.sol](contracts/PWMRewardERC20.sol) |
| **PWMStakingERC20** | [`0x88D7860d800Cc68d905751696C3c0B4875Af950b`](https://basescan.org/address/0x88D7860d800Cc68d905751696C3c0B4875Af950b) | [contracts/PWMStakingERC20.sol](contracts/PWMStakingERC20.sol) |
| **PWMCertificate** | [`0x014492dEfc66D5b58b86027cEB636d4c84289eAe`](https://basescan.org/address/0x014492dEfc66D5b58b86027cEB636d4c84289eAe) | [contracts/PWMCertificate.sol](contracts/PWMCertificate.sol) |
| **PWMMintingERC20** | [`0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235`](https://basescan.org/address/0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235) | [contracts/PWMMintingERC20.sol](contracts/PWMMintingERC20.sol) |
| **PWMVesting** | [`0x9c57BA6f844dAAecB050D83f31A8279E04a441a9`](https://basescan.org/address/0x9c57BA6f844dAAecB050D83f31A8279E04a441a9) | [contracts/PWMVesting.sol](contracts/PWMVesting.sol) |

Full machine-readable address index: [`addresses.json`](addresses.json).

---

## Directory structure

```
mainnet/
├── README.md                       # This file
├── DEPLOY_LOG.md                   # Canonical deploy record (2026-05-22)
├── addresses.json                  # Machine-readable contract addresses
├── contracts/                      # 9 Solidity contracts (MIT licensed)
├── security/                       # Multi-agent security review trail (15 docs)
│   ├── SECURITY_REVIEW.md          # 10-agent aggregator report
│   ├── MYTHRIL_TRIAGE.md           # Mythril symbolic execution — 0 issues
│   ├── SLITHER_TRIAGE.md           # Slither static analysis triage
│   ├── ECONOMIC_ATTACK_REVIEW.md   # Economic-attack modeling (A5)
│   └── A1-A10*.md                  # Per-agent reports
├── strategy/                       # Strategic framing docs (2026-05-22)
│   ├── VALUE_FRAMING.md            # AI4Science is the product; verification is the moat
│   ├── USER_ACQUISITION_STRATEGY.md # Two-track demand-side strategy
│   ├── PHASED_ARCHITECTURE_DEPLOYMENT.md # Phase 1a/1b/2/3/4 sequencing
│   ├── TOKEN_UTILITY_AND_VALUE.md  # 6 value drivers + 8 use cases
│   ├── GENESIS_PRINCIPLES_REALITY_CHECK.md # Why 6 anchors (not 1,597)
│   ├── DEVELOPER_COMPENSATION.md   # Layer 1/2/3 framework
│   ├── LAUNCH_LANDING_PAGE_DRAFT.md # Implementation-ready marketing copy
│   └── ANNOUNCEMENT_BUNDLE.md      # Paste-ready Twitter/HN/Reddit/LinkedIn copy
├── grants/                         # Pending grant applications
│   ├── BASE_BUILDER_GRANT.md       # 5 ETH (~$15K) — submitted via Paragraph nomination
│   └── EF_ESP.md                   # $50K — to be submitted at esp.ethereum.foundation
├── plan/                           # Roadmap
│   ├── PLAN.md                     # Master execution plan (Tracks 1-8)
│   ├── POST_MAINNET_NEXT_STEPS.md  # Active checklist from D9 forward
│   └── TRACK_9_LOW_DOSE_CT.md      # Medical imaging flagship (RSNA/ISBI 2028)
└── bounties/                       # External developer bounty catalog (~1.3M PWM)
    ├── INDEX.md                    # 10 bounties, statuses, total pool
    └── 01-10-*.md                  # Per-bounty specs
```

---

## How to verify PWM is live (independent, no trust required)

Anyone can verify the protocol state from a public Base RPC. Sample checks:

### 1. PWMToken total supply
```bash
curl -s -X POST https://mainnet.base.org \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_call","params":[{
    "to":"0x7326781182b9cDc1eF9Fa147fB689862f893dA14",
    "data":"0x18160ddd"
  },"latest"],"id":1}'
# Expected result: 0x00000000000000000000000000000000000000000000045639018244f4000000
# = 21,000,000 × 10^18 wei
```

### 2. PWMRegistry owner (should be PWMGovernance)
```bash
curl -s -X POST https://mainnet.base.org \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_call","params":[{
    "to":"0x9F91784c2fa884A79473304050C581424E006fbd",
    "data":"0x8da5cb5b"
  },"latest"],"id":1}'
# Expected: 0x...83F210b9A8E5F0FAfE133c700F888b3A303f9b15 (PWMGovernance)
```

### 3. Soft-launch caps (mintingPaused should be true until 2026-06-21)
```bash
curl -s -X POST https://mainnet.base.org \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"eth_call","params":[{
    "to":"0x629190D88cdB0C4a2cFEe00Dd7EdD490c465B235",
    "data":"0xe1a283d6"
  },"latest"],"id":1}'
# Expected: 0x...0000000000000001 (true)
```

### 4. Genesis artifacts on PWMRegistry
The 6 founding artifacts (CASSI + CACTI L1/L2/L3) are registered with these content hashes:

| Artifact | Content hash |
|---|---|
| L1-025 CASSI Principle | `0x03b714a728cf9f81ea6445e73adcd02ed18420babc2a335421ed8d0f594cf19f` |
| L1-027 CACTI Principle | `0x358452e94a8000ae267129d34159166c22092f50ea4d2af9f2e91bf2a2839b6f` |
| L2-025-001 CASSI Spec | `0xe08488dd8e1abbfd3e20c846ad2a40254b0651b0a51f3941f7a37c3fec302409` |
| L2-027-001 CACTI Spec | `0x2af46a93084cfb80a43097f4328ad961339e3417582ce9ea08ce93e2fd0ef1de` |
| L3-025-001-001 CASSI Benchmark | `0x617d3707468f4114715369a653ac2b05ad4c5db94a9346263290b85650a1971a` |
| L3-027-001-001 CACTI Benchmark | `0x185ff24d633974e4d9dbb658b687bd186cbf0ccf4d27145b678fc219d69a2c76` |

Call `PWMRegistry.exists(bytes32)` (selector `0x38a699a4`) with each hash; all should return `true`.

---

## Token economics (21M PWM fixed supply, capped)

| Allocation | Amount | % | Mechanism |
|---|---|---|---|
| Minting pool | 17,220,000 PWM | 82% | Programmatic emission via `PWMMintingERC20.mintFor`; Zeno formula; never reaches zero asymptotically |
| Reserve | 2,100,000 PWM | 10% | 3-of-5 Gnosis Safe; grants distributed by multisig/DAO governance |
| Liquidity | 1,050,000 PWM | 5% | Uniswap v3 PWM/USDC LP (seeded Phase 2) |
| Founding team | 630,000 PWM | 3% | `PWMVesting.sol` — 4-year linear, 1-year cliff, immutable beneficiary |
| **Total** | **21,000,000 PWM** | **100%** | Cap enforced in `PWMToken` constructor |

All allocations verified on-chain via `balanceOf()` at deploy (2026-05-22).

---

## Security posture

Before mainnet deploy:
- **199/199 Hardhat tests passing**
- **Mythril** symbolic execution: 0 issues across 9 contracts (5h 18min wall, all detectors clean)
- **Slither** static analysis: 58 raw findings → 0 deploy-blocking after triage
- **Multi-agent security review** (10 specialized AI agents): 2 CRITICAL + 4 HIGH + 6 MEDIUM caught and fixed
- **Sepolia governance rehearsals**: full propose → 3-of-5 approve → 48h timelock → execute cycle validated

After mainnet deploy:
- **Post-deploy verifier 25/25 GREEN ×2** (pre-handoff + post-handoff)
- **Independent on-chain verification 10/10 PASS** (2026-05-22 evening)
- **All 9 contracts Basescan-verified** (public source code)

See [security/](security/) for full review trail.

---

## Roadmap (high level)

| Phase | Window | Status |
|---|---|---|
| **Phase 1a** — Soft-launch audit | D9 → D9+30 (~2026-06-21) | ✅ IN PROGRESS |
| **Phase 1b** — Mining active; PWM-CI-1 CASSI benchmark opens | D9+30 → D9+180 (~2026-11-18) | ⏳ Activation 2026-06-21 |
| **Phase 2** — LP seeded; multi-benchmark | Months 6-12 | ⏳ |
| **Phase 3** — Mine-to-use; users pay PWM for advanced runs | Months 12-24 | ⏳ |

See [plan/POST_MAINNET_NEXT_STEPS.md](plan/POST_MAINNET_NEXT_STEPS.md) for the active checklist.

---

## Funding pipeline

PWM operates as a public-goods nonprofit (NumFOCUS Round 4 sponsorship pending; Foundation 501(c)(3) trajectory). All grant applications are public for transparency:

- [grants/BASE_BUILDER_GRANT.md](grants/BASE_BUILDER_GRANT.md) — 5 ETH (~$15K) for bug bounty seed + partial audit funding
- [grants/EF_ESP.md](grants/EF_ESP.md) — $50K for formal third-party audit + 3 pre-cap-raise features

After audit clears and cap-raise governance executes (~Q4 2026), the protocol operates at production scale.

---

## License

MIT (consistent with the rest of this repo). Smart contracts use OpenZeppelin libraries (also MIT).

## Disclosures

- PWM is **independent of UT Southwestern Medical Center**. The Founder (Chengshuai "Abraham" Yang) is faculty at UTSW but operates PWM as a separate open-source / public-goods project. UTSW does not endorse, fund, or have any administrative role.
- PWM tokens are **utility tokens for protocol participation, NOT investment contracts**. The PWM Foundation does not market tokens as investments or engage in price-management activity. Token economics is the reward mechanism; the benchmark is the product.

---

## Contact

- Email: director@physicsworldmodel.org (TBD) | platformaiyang@gmail.com
- GitHub: [@integritynoble](https://github.com/integritynoble)
- ORCID: [0000-0003-2840-5344](https://orcid.org/0000-0003-2840-5344)

---

*This is the public mainnet documentation for PWM. For implementation-level details (deploy scripts, internal coordination, private wallet operations), see the team's private working repo. The contracts deployed here are immutable; future updates are governance-proposed only.*
