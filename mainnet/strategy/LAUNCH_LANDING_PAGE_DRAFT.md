# PWM D9 Launch Landing Page — Copy Draft

**Date:** 2026-05-22
**Audience:** sub-GPU + Bounty 2 winner (Web UI / Explorer) for implementation in `pwm_product/platform/pwm_platform/`
**Status:** Ready-to-implement copy; pending Director sign-off on framing decisions in `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` §10
**Purpose:** Concrete landing-page copy for D9 mainnet launch, leading with **PWM-CI-1 (Verified Computational Imaging Benchmark #1)** — the product that gives PWM a real first user vertical.

---

## Implementation Notes for sub-GPU

### Routing

- **Primary route:** `https://physicsworldmodel.org/` (homepage)
- **Benchmark detail route:** `https://physicsworldmodel.org/benchmarks/pwm-ci-1`
- **Leaderboard route:** `https://physicsworldmodel.org/benchmarks/pwm-ci-1/leaderboard`
- **Submission guide route:** `https://physicsworldmodel.org/benchmarks/pwm-ci-1/submit`
- **About route:** `https://physicsworldmodel.org/about`
- **FAQ route:** `https://physicsworldmodel.org/faq`

### Production location

Implementation in `pwm_product/platform/pwm_platform/` per `[[feedback_production_deploy]]`. NOT in legacy `platform/` directory.

### Design system

- Mobile-first responsive (per Bounty 10 spec); breakpoints 320 / 768 / 1024 / 1440
- WCAG 2.1 AA compliance for color contrast + touch targets
- Tailwind + shadcn/ui (recommended; consistent with Bounty 2 reference impl)
- Color palette: stick with PWM brand colors (TBD; suggest neutral grays + single accent)
- Avoid crypto / Web3 visual cues (no Ethereum logo, no token charts, no "buy now" CTAs)
- DO use academic / scientific visual cues (paper-like typography; sober color; data tables; charts)

### Voice / tone

- **Always:** scientific, precise, technical, sober, helpful
- **Never:** speculative, "moon," "wagmi," token-price-focused, hype-y, anti-establishment

### Conversion KPI

- Primary: clicks on "Submit Your Method to PWM-CI-1" CTA
- Secondary: GitHub repo stars; Discord/WeChat joins; arXiv preprint downloads
- Tertiary: time-on-page > 60 seconds (signal of genuine reader interest)

---

## SECTION 1: HERO

### Headline

> **Verified benchmarks for physics-grounded AI.**

### Sub-headline

> PWM is an open-source benchmark platform where researchers submit AI methods to **leak-proof, cryptographically-verified leaderboards**.
> Starting with **computational imaging**.
> Reproducibility, citation, and verification — built in.

### Primary CTA

> **[Submit Your Method to PWM-CI-1 →](https://physicsworldmodel.org/benchmarks/pwm-ci-1)**

### Secondary CTAs

> [View the Leaderboard](https://physicsworldmodel.org/benchmarks/pwm-ci-1/leaderboard) · [Read the Technical Report (arXiv)](https://arxiv.org/abs/pwm-ci-1) · [GitHub Repo](https://github.com/integritynoble/pwm-ci-1)

### Trust signals (subtle, below the fold)

> 🏛️ NumFOCUS sponsorship pending · 🔓 MIT licensed · ⛓️ Base mainnet · 📄 arXiv preprint

---

## SECTION 2: THE FIRST BENCHMARK — PWM-CI-1

### Section title

> **PWM-CI-1: CASSI Reconstruction Benchmark**

### What it is

> A **leak-proof, cryptographically-verified** benchmark for compressed sensing imaging spectrometer (CASSI) reconstruction methods. The test set is sealed on-chain. Submissions are scored against held-out ground truth that no submitter has seen.
>
> Submit your reconstruction method. Get a verified PSNR + SSIM score. Win up to **5,000 PWM**.

### At a glance

| Item | Detail |
|---|---|
| **Task** | CASSI reconstruction from compressed measurements |
| **Dataset** | [Insert: public CASSI dataset name + size] |
| **Metric** | PSNR (primary), SSIM (secondary) |
| **Test set** | Held-out scans; hash committed on-chain at benchmark launch (cannot be tampered with) |
| **Submission format** | Containerized inference script (Docker); standard interface defined in GitHub repo |
| **Submission window** | D9 to D9+90 (first round) |
| **Prize pool** | 10,000 PWM (Reserve discretionary) |
| **Entry fee** | None (Round 1 is sponsored) |
| **License** | MIT for all submissions; baseline + eval code MIT |

### How submissions are scored

> Each submission runs against the **sealed test set** in a controlled Docker environment. Output reconstructions are scored using PSNR + SSIM against held-out ground truth. Results are written as **L4 reproduction certificates** on PWM Registry — immutable, citable, queryable.
>
> Top-3 winners receive PWM token rewards. All ranked submissions receive a citable certificate hash (your method's verified score, permanently on-chain).

### Prize breakdown

| Rank | Prize | Notes |
|---|---|---|
| **Rank 1** | **5,000 PWM** | Plus paper co-authorship offer on the benchmark report |
| **Rank 2** | **2,500 PWM** | Plus paper co-authorship offer |
| **Rank 3** | **1,500 PWM** | Plus paper co-authorship offer |
| Rank 4-10 | 1,000 PWM total | Distributed proportionally |
| All ranked | Citable L4 certificate hash | Permanently on-chain; cite in your paper |

### Why PWM-CI-1 first?

> CASSI reconstruction is a real, active research problem. Existing benchmarks suffer from test-set leakage and irreproducibility. PWM-CI-1 fixes both with cryptographic commitment.
>
> If your method works, you get a citable proof. If it doesn't, you learn fast.

---

## SECTION 3: HOW IT WORKS

### Section title

> **How PWM works**

### Four-step explanation

> **1. Benchmark spec is committed on-chain.**
> The task, dataset, metric, and held-out test set hash are written to PWM Registry as immutable artifacts. No one — not even the benchmark authors — can modify the test set after commitment.
>
> **2. You submit your method.**
> Fork the GitHub repo. Implement the standard inference interface. Submit a Docker container running your method. No setup beyond standard ML tooling.
>
> **3. PWM verifies your submission.**
> Your container runs against the sealed test set in a reproducible environment. PSNR + SSIM scores are computed deterministically. Results are written as L4 reproduction certificates to PWM Registry.
>
> **4. The leaderboard auto-updates.**
> Your method appears on the leaderboard with its verified score, your cert hash, and a citation-ready snippet. If you're top-3, you receive PWM token rewards.

### Visualization (sub-GPU to design)

> [Flowchart showing: GitHub Repo → Submit → PWM Verification → L4 Cert + Leaderboard]

---

## SECTION 4: WHY USE PWM

### Section title

> **Why submit to PWM?**

### Five core features (the magnets, framed for academics)

#### 1. 🔒 Leak-proof benchmarks

> Test sets are committed on-chain at benchmark creation. No one can leak them, modify them, or accidentally train on them. Your PSNR is the **real** PSNR — not an overfitted-to-test-set number.

#### 2. 🪪 Reproduction certificates

> Every submission generates an L4 reproduction certificate — a cryptographically-signed record of your method's verified score. Cite the cert hash in your paper. Future readers can verify your result independently.

#### 3. ⏱️ Cryptographic priority timestamps

> Your submission is timestamped with an immutable block timestamp + your wallet address. If your method later gets contested or scooped, you have unbreakable proof of when you submitted it. Better than arXiv timestamps (which can be revised) or lab notebooks (which can be backdated).

#### 4. 📈 Citation graph as protocol

> PWM exposes a "this method was reproduced N times" graph queryable via API + MCP. Future researchers see how many independent labs reproduced your method. Higher reproduction = higher trust.

#### 5. 🪙 Verified contributions earn rewards

> Top-ranking submissions earn PWM tokens from the Foundation Reserve. PWM is a new token; current speculative value is modest, but if PWM follows the trajectory of similar verification protocols, this could be meaningful.

> **What PWM is NOT:** PWM is **not a cryptocurrency speculation platform**. The token is the reward mechanism for verified contributions, not the product. The product is the benchmark.

---

## SECTION 5: SUBMIT YOUR METHOD

### Section title

> **Submit your method to PWM-CI-1**

### Quickstart (30-second walkthrough)

> ```bash
> # 1. Clone the repo
> git clone https://github.com/integritynoble/pwm-ci-1
> cd pwm-ci-1
>
> # 2. Set up the environment
> docker build -t my-cassi-method -f Dockerfile.template .
>
> # 3. Implement your method
> # Edit src/method.py — implement the standard inference interface
>
> # 4. Test locally
> ./scripts/test_local.sh  # validates against a public mini test set
>
> # 5. Submit
> ./scripts/submit.sh --wallet 0xYourWallet
> ```
>
> Your submission is verified on-chain within ~10 minutes. Results appear on the leaderboard.

### Full submission guide

> See [github.com/integritynoble/pwm-ci-1/blob/main/SUBMISSION.md](https://github.com/integritynoble/pwm-ci-1/blob/main/SUBMISSION.md) for the complete walkthrough including:
>
> - Standard inference interface specification
> - Docker base image requirements
> - Reproducibility constraints (deterministic seeds, single-GPU, etc.)
> - Wallet setup (use any Ethereum-compatible wallet on Base mainnet)
> - Common pitfalls + debugging tips

### Requirements

- An Ethereum-compatible wallet on Base mainnet
- Docker installed
- ~10 minutes of compute time per submission (the verification environment runs your method on the sealed test set)
- Your method's reproducibility (deterministic outputs given fixed seeds)
- Your inference script (no training data required; training is your own job)

### Common questions

- **"Do I need to know blockchain?"** No. Submit via the script; wallet setup takes 5 minutes; the rest is standard ML tooling.
- **"Do I have to pay to submit?"** Round 1 is sponsored — no entry fee. Later rounds may have a small refundable stake (1-10 PWM).
- **"What if my method doesn't win?"** You still get a citable certificate hash with your verified score. Use it in your paper as evidence your method achieves X PSNR.
- **"Can I submit multiple methods?"** Yes. Each submission is scored independently. Only your best-scoring submission counts for ranking.

---

## SECTION 6: ROADMAP

### Section title

> **PWM's benchmark roadmap**

### What's coming

| Benchmark | When | Status |
|---|---|---|
| **PWM-CI-1: CASSI Reconstruction** | **D9 (launching now)** | ✅ LIVE |
| PWM-CI-2: Compressed Sensing | D9+90 | Spec drafting |
| PWM-CI-3: Spectral Imaging | D9+180 | Planning |
| **PWM-MED-1: Low-Dose CT (Public Data)** | D9+365 | Track 9 / Heyang Zhao workplan; planning |
| PWM-MED-2: Low-Dose CT (Clinical-Grade) | D9+730 | Track 9 RSNA / ISBI 2028 pre-launch |
| PWM-CI-N: TBD | Ongoing | Community + governance proposals |

### The medical imaging flagship

> **PWM-MED-2** (low-dose CT clinical) is the medical imaging flagship — launching at RSNA / ISBI 2028 with attending radiologist co-authorship, IRB approval, and clinical-grade ground truth. Comes after PWM-CI-1 + PWM-MED-1 validate the mechanism.

---

## SECTION 7: ABOUT PWM

### Section title

> **About PWM**

### What PWM is

> **PWM (Physics World Model)** is an open-source benchmark platform for physics-grounded AI. PWM combines:
>
> - **Cryptographic verification** (Base L2; PWM Registry contracts)
> - **Reproducible scoring** (deterministic Docker environments)
> - **Citation infrastructure** (cert hashes, immutable timestamps, reproduction graph)
> - **Community governance** (5-founder multisig now; DAO transition Year 1+)
> - **Token incentives** (PWM rewards verified contributions)
>
> PWM is **not** a token speculation platform. The token is the reward mechanism for the benchmark platform — not the other way around.

### Open-source + community-governed

> All PWM smart contracts, infrastructure code, and benchmark specs are MIT-licensed and publicly readable:
>
> - Smart contracts: [github.com/integritynoble/pwm](https://github.com/integritynoble/pwm) (`infrastructure/agent-contracts/`)
> - Benchmark reference: [github.com/integritynoble/pwm-ci-1](https://github.com/integritynoble/pwm-ci-1)
> - Indexer + frontend: open bounties (see `bounties/INDEX.md`)
> - Specifications: `specs/` directory in main repo

### Governance + foundation status

> PWM is bootstrapped by a small founding team. Current governance is a 5-multisig with Director (Zhang Yang) holding all 5 signing slots (Path A bootstrap). Co-founder recruitment is in motion.
>
> Foundation 501(c)(3) trajectory via **NumFOCUS Round 4 sponsorship application**. Trademark filings for "PWM" and "Physics World Model" filed with USPTO.
>
> **PWM is not affiliated with UTSW or any commercial entity.** The Director is a faculty member at UTSW Medical Center but operates PWM as an independent open-source / public-goods project. Track 9 medical imaging research is conducted under separate IRB approval and institutional review.

### Token economics summary

> PWM has a fixed 21,000,000 token supply:
>
> - **82% (17.22M):** Programmatic emission to L4 reproduction certificates (mining pool)
> - **10% (2.1M):** Foundation Reserve (bounties, grants, contributor rewards, mini-competitions)
> - **5% (1.05M):** Liquidity (Uniswap v3 PWM/USDC, seeded Year 1+)
> - **3% (630K):** Founding team (PWMVesting; 4-year linear, 1-year cliff)
>
> See `coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` for full economic structure.

---

## SECTION 8: FAQ

### Section title

> **Frequently asked questions**

### Selected FAQ items

**Q: How is PWM different from MLCommons, Kaggle, or Papers with Code?**

> All of those are valuable. PWM's differentiator: **cryptographic test-set commitment**. MLCommons, Kaggle, and Papers with Code rely on trust + server-side data hiding. PWM enforces leak-proof benchmarks via on-chain commitment. There's no trusted host that can leak the test set.

**Q: Why blockchain? Is this just a "crypto project"?**

> No. Blockchain is the **infrastructure**, not the product. The product is the benchmark platform.
>
> We use blockchain for three specific reasons:
> 1. **Immutable test-set commitment** — no one can modify the sealed test set, including PWM Foundation
> 2. **Cryptographic priority timestamps** — your submission's submission time is unforgeable
> 3. **Programmatic reward distribution** — rewards flow automatically based on verified scores, with no human discretion
>
> If a non-blockchain mechanism existed that achieved all three, we'd use it. None does.

**Q: Do I need to know crypto / web3 to submit?**

> No. The only blockchain interaction is wallet setup (5 minutes; one-time). Everything else is standard ML tooling (Docker, PyTorch, GitHub).

**Q: What is the PWM token worth?**

> The PWM token is brand-new. Token value at launch is modest and depends on demand-side adoption. PWM Foundation does not make price predictions. **The value of submitting to PWM-CI-1 is not the token reward.** The value is the **verified leaderboard score + citable certificate + visibility** for your method.
>
> Token economics is the **reward mechanism**, not the product. We say this repeatedly because the framing matters.

**Q: Is PWM affiliated with UTSW?**

> No. The Director is a faculty member at UTSW Medical Center but operates PWM as an independent open-source project. UTSW does not endorse, fund, or have any administrative role in PWM. Track 9 medical imaging research conducted under separate IRB approval is the only intersection.

**Q: How can I support PWM without submitting a method?**

> - Star the GitHub repos
> - Cite PWM in your papers (if you've used a PWM benchmark)
> - Join the Discord / WeChat community
> - Apply to be a Bounty winner (see `bounties/INDEX.md`)
> - Apply to be a co-founder if you're a relevant domain expert

**Q: What does "Path A bootstrap" mean in your docs?**

> Path A means the project is bootstrapped by a small founding team (Director only, currently) before recruiting co-founders. This is **transparent governance**: anyone can verify the 5-signing-slot multisig is currently Director-held. Co-founder recruitment is in motion; founder rotation will distribute signing rights as new co-founders join.

**Q: Is the PWM token a security?**

> PWM is structured as a utility token for protocol participation, not as an investment contract. PWM Foundation does not market the token as an investment. Specific securities law analysis is ongoing (see `coordination/PRE_DEPLOY_RISK_AUDIT_2026-05-21.md` §6). If you're a US person, consult a crypto-aware attorney before significant token acquisition.

**Q: What happens if PWM fails / disbands?**

> All smart contracts are immutable + autonomously functional. Even if PWM Foundation ceases to exist, the verification infrastructure continues to work. Test sets are sealed on-chain forever. Existing certificates are permanent. The protocol is **trust-minimized by design**.

---

## SECTION 9: GET INVOLVED

### Section title

> **Get involved**

### Community channels

| Channel | Link | Audience |
|---|---|---|
| 💬 **Discord** | [discord.gg/pwm](https://discord.gg/pwm) (TBD) | Primary; English; OSS-default |
| 🇨🇳 **WeChat** | [QR code TBD] | Chinese-language community |
| 📂 **GitHub Discussions** | [github.com/integritynoble/pwm/discussions](https://github.com/integritynoble/pwm/discussions) | OSS contributors |
| 🐦 **Twitter / X** | [@PhysicsWorldModel](https://twitter.com/PhysicsWorldModel) (TBD) | Announcements |
| 📰 **Newsletter** | [Sign up](https://physicsworldmodel.org/newsletter) | Monthly digest |

### Bounties

> PWM has an active bounty program for external contributors. See [`bounties/INDEX.md`](https://github.com/integritynoble/pwm/blob/main/pwm-team/bounties/INDEX.md) for the 10 current open / spec'd bounties (~1.3M PWM total pool).
>
> Highlights:
> - **Bounty 1: Scoring engine** (200K PWM) — competing reference impl
> - **Bounty 2: Web UI / Explorer** (80K PWM) — frontend
> - **Bounty 5: Smart contracts competing impl** (500K PWM) — alternate Solidity impl
> - **Bounty 9: MCP server** (25K PWM) — AI assistant integration
> - **Bounty 10: Mobile UX** (40K PWM) — mobile-first dApp

### Co-founder recruitment

> PWM is actively recruiting co-founders (Track 4a). Looking for:
>
> - Computational imaging researcher (non-UTSW lab; PI track or senior postdoc)
> - Open-source contributor with imaging / ML reproducibility experience
> - Crypto / blockchain ecosystem contributor with academic respect
> - Optionally: domain expert (medical imaging, spectroscopy, materials science)
>
> Email director@physicsworldmodel.org or apply via [github.com/integritynoble/pwm/discussions](https://github.com/integritynoble/pwm/discussions) under category "co-founder".

### Research partnerships

> Interested in posting a benchmark on PWM, integrating PWM verification into your lab workflow, or partnering on RSNA / ISBI 2028 medical imaging flagship? Email partnerships@physicsworldmodel.org (TBD).

---

## SECTION 10: FOOTER

### Footer content

```
PWM (Physics World Model)

Open-source benchmark platform for physics-grounded AI.

Started 2024. Mainnet launch 2026.
MIT licensed. Community governed.

[GitHub] [arXiv] [Discord] [Twitter] [Email]

Not affiliated with UTSW or any commercial entity.
PWM tokens are utility tokens for protocol participation, not investment contracts.

NumFOCUS Round 4 sponsorship pending.
Foundation 501(c)(3) trajectory.

Smart contracts on Base mainnet:
- PWMToken: 0x... (link to Basescan)
- PWMRegistry: 0x... (link to Basescan)
- PWMGovernance: 0x... (link to Basescan)
(Full contract addresses at /contracts)

© 2026 PWM Foundation. Released under MIT for all PWM Foundation-authored content.
```

---

## SECTION 11: PRE-LAUNCH CHECKLIST FOR SUB-GPU

Before D9 deploy, the following must be ready:

### Content
- [ ] Hero copy finalized (Section 1)
- [ ] PWM-CI-1 spec finalized (Section 2)
- [ ] arXiv preprint URL placeholder filled
- [ ] Token economics summary verified against `PWM_DEVELOPER_COMPENSATION_2026-05-22.md`
- [ ] UTSW non-affiliation language reviewed by Director
- [ ] FAQ items reviewed by Director (especially "Is PWM affiliated with UTSW?" and "Is PWM a security?")

### Infrastructure
- [ ] GitHub repo `pwm-ci-1` LIVE with baseline + eval + submission guide
- [ ] Discord server set up; invite link active
- [ ] WeChat group set up; QR code generated
- [ ] GitHub Discussions enabled for `integritynoble/pwm`
- [ ] Twitter / X account created (or rebranded from existing)
- [ ] Newsletter signup form (basic; sub-GPU's choice of tool)
- [ ] Email aliases (director@, partnerships@, security@) configured

### Frontend
- [ ] Routes implemented per implementation notes above
- [ ] Mobile-responsive (Bounty 10 compatible)
- [ ] WCAG 2.1 AA verified (Lighthouse score ≥ 90 accessibility)
- [ ] Lighthouse performance ≥ 80
- [ ] No tracking / analytics that violate academic norms (avoid Google Analytics; use privacy-respecting alternative if any)
- [ ] Open Graph / Twitter card metadata configured

### Contract integration
- [ ] Leaderboard reads from on-chain PWMCertificate events
- [ ] Wallet connection (Privy or RainbowKit) integrated
- [ ] Submission flow tested end-to-end on testnet
- [ ] Base mainnet contract addresses populated in `addresses.json`

### Legal / compliance
- [ ] Terms of service drafted (basic; standard OSS terms + token disclaimer)
- [ ] Privacy policy drafted (basic; data collection minimal)
- [ ] Cookie banner (only if cookies are actually used)
- [ ] UTSW non-affiliation language in footer + about page

### Outreach prep (Director-led)
- [ ] 30-50 outreach list assembled
- [ ] 3-5 outreach email templates drafted
- [ ] Twitter/X launch thread (12-15 tweets) drafted
- [ ] HackerNews post drafted
- [ ] Reddit posts drafted (r/MachineLearning, r/PhD, r/computervision)
- [ ] arXiv companion paper submitted (cs.DL or stat.ML)

---

## SECTION 12: COPY VARIATIONS FOR DIFFERENT AUDIENCES

### For arXiv comments / academic outreach

> Subject: Invitation to PWM-CI-1, a new cryptographically-verified CASSI reconstruction benchmark
>
> Dear [name],
>
> I noticed your recent paper on [topic] and wanted to invite you to submit your method to PWM-CI-1, a new leak-proof CASSI reconstruction benchmark we're launching this month.
>
> PWM-CI-1 fixes test-set leakage cryptographically: the test set is committed on-chain at benchmark launch and cannot be modified. Your method's PSNR is the real PSNR.
>
> Submission is free in Round 1. Top-3 winners receive PWM token rewards + co-authorship offer on the benchmark report (target venue: MICCAI 2026 or NeurIPS 2026).
>
> Full details: https://physicsworldmodel.org/benchmarks/pwm-ci-1
>
> Best,
> Zhang Yang
> Director, PWM Foundation
> Faculty, UTSW Medical Center (independent of UTSW for this project)

### For Twitter/X launch thread (12 tweets)

> **Tweet 1:** PWM-CI-1 is live: a cryptographically-verified CASSI reconstruction benchmark. Test set sealed on-chain. Top-3 winners earn PWM token rewards + paper co-authorship. Submit your method: physicsworldmodel.org/benchmarks/pwm-ci-1
>
> **Tweet 2:** Why does this matter? Existing benchmarks suffer from test-set leakage. Once a test set is public, it's polluted: methods overfit. PWM-CI-1 cryptographically commits the test set on-chain. No one — including us — can leak it.
>
> **Tweet 3:** How it works: 1) Fork the GitHub repo. 2) Implement the standard inference interface. 3) Submit a Docker container. 4) PWM verifies against the sealed test set. 5) Your method appears on the leaderboard with a citable cert hash.
>
> **Tweet 4:** Round 1 is FREE. No entry fee. PWM Foundation sponsors all gas costs for first 100 submissions. We pay you to participate (top-3: 5K/2.5K/1.5K PWM).
>
> **Tweet 5:** What's a "verified cert hash"? Every submission generates an L4 reproduction certificate — cryptographically signed proof of your method's verified score. Cite it in your paper. Future readers verify your result independently.
>
> **Tweet 6:** Is this just a "crypto project"? No. Blockchain is the infrastructure (immutable test sets, unforgeable timestamps). The product is the benchmark. The token is the reward mechanism, not the product.
>
> **Tweet 7:** Worked example: You're a PhD student with a novel CASSI reconstruction method. You submit to PWM-CI-1. Your method scores PSNR=32.5 dB (rank 4). You cite cert hash 0xabc... in your paper. Reviewers verify your result on-chain.
>
> **Tweet 8:** What's coming next: PWM-CI-2 (compressed sensing, D9+90). PWM-CI-3 (spectral imaging, D9+180). PWM-MED-1 (low-dose CT public data, D9+365). PWM-MED-2 (clinical low-dose CT at RSNA/ISBI 2028).
>
> **Tweet 9:** Smart contracts on Base mainnet: github.com/integritynoble/pwm. MIT-licensed. All open. Bounty hunters: see bounties/INDEX.md for 10 open/spec'd bounties (~1.3M PWM pool).
>
> **Tweet 10:** Governance: 5-multisig (Path A bootstrap; Director currently holds all 5; co-founder recruitment in motion). NumFOCUS Round 4 sponsorship pending. Foundation 501(c)(3) trajectory.
>
> **Tweet 11:** Not affiliated with UTSW. Director is faculty there but operates PWM as independent open-source project. UTSW does not endorse, fund, or administer PWM. Track 9 medical research conducted under separate IRB approval.
>
> **Tweet 12:** Questions? Discord (link), WeChat (link for Chinese-language community), GitHub Discussions, or email director@physicsworldmodel.org. Want to submit a benchmark on your domain? partnerships@physicsworldmodel.org.

### For HackerNews "Show HN" post

> **Title:** Show HN: PWM-CI-1 — cryptographically-verified CASSI reconstruction benchmark
>
> **Body:**
>
> Hi HN,
>
> I'm Zhang Yang, faculty at UTSW Medical Center. Today we're launching PWM-CI-1, a new computational-imaging benchmark with two technical novelties:
>
> 1. **Leak-proof test set:** Committed on-chain at benchmark launch. Cannot be modified. No trusted host can leak it.
>
> 2. **Verified reproduction certificates:** Every submission generates an L4 cryptographic certificate proving the method's verified score. Citable in papers.
>
> PWM (Physics World Model) is an open-source benchmark platform we've been building for the last 2 years. PWM-CI-1 is the first concrete benchmark. The broader roadmap: PWM-CI-2 (compressed sensing), PWM-CI-3 (spectral imaging), and PWM-MED-1/2 (low-dose CT, with RSNA/ISBI 2028 launch target).
>
> What we want feedback on:
> - The benchmark spec — is the metric (PSNR + SSIM) sufficient? Should we add perceptual metrics?
> - The submission interface — is the Docker contract reasonable? Anything missing?
> - The verification mechanism — does the L4 certificate flow make sense?
> - The token reward model — Round 1 is free + sponsored; later rounds may have small refundable stake. Is this the right cadence?
>
> Repo: github.com/integritynoble/pwm-ci-1
> Landing: physicsworldmodel.org/benchmarks/pwm-ci-1
> Tech report (arXiv): [link]
> Discord: [link]
>
> Smart contracts on Base mainnet. Foundation 501(c)(3) trajectory (NumFOCUS Round 4 pending). Not affiliated with UTSW.
>
> Happy to answer questions about technical decisions or strategy. Also actively recruiting co-founders (medical imaging PI, OSS contributor, crypto-ecosystem contributor).

### For Reddit r/MachineLearning post

> **Title:** [Project] PWM-CI-1: Cryptographically-Verified CASSI Reconstruction Benchmark (Test Set Sealed On-Chain)
>
> **Body:**
>
> TL;DR: We're launching a new computational imaging benchmark where the test set is cryptographically committed on-chain. No leakage, no overfitting to test set, no trusted host.
>
> Background: Test-set leakage has been a problem for ML benchmarks for a decade. ImageNet was leaked. SQuAD was leaked. Many ML benchmarks have been "polluted" by test-set exposure. The standard mitigation is server-side hiding, which depends on trust in the host org.
>
> Our approach: commit the test set hash to a smart contract on Base mainnet at benchmark launch. No one — including us — can modify the test set after that. Submissions run against the sealed test set in deterministic Docker containers.
>
> PWM-CI-1 (Compressed Sensing Imaging Spectrometer benchmark):
> - Task: CASSI reconstruction
> - Dataset: [public CASSI dataset]
> - Metric: PSNR + SSIM
> - Top-3 winners: 5K / 2.5K / 1.5K PWM token rewards
> - Round 1: free, sponsored (no entry fee)
> - Submission: Docker container; standard inference interface
> - Verification: deterministic; ~10 min per submission
>
> Links:
> - Landing page: physicsworldmodel.org/benchmarks/pwm-ci-1
> - GitHub repo: github.com/integritynoble/pwm-ci-1
> - Technical report (arXiv): [link]
> - Discord: [link]
>
> Discussion welcome on whether this approach scales beyond CASSI / computational imaging. We're planning PWM-MED-1/2 (low-dose CT) for 2027 with RSNA/ISBI 2028 as the flagship competition.

---

## SECTION 13: GROWTH / ANALYTICS PLAN

For sub-GPU + Director monitoring:

### Daily (D9 to D9+30)

- [ ] Landing-page visits (unique visitors / day)
- [ ] CTA clicks (Submit Your Method button)
- [ ] GitHub repo stars / forks
- [ ] Discord member growth
- [ ] WeChat member growth
- [ ] Newsletter signups
- [ ] arXiv preprint downloads

### Weekly (D9+7 to D9+90)

- [ ] PWM-CI-1 submission count (target: 5-10 by D9+90)
- [ ] Average submission latency (clone → submit)
- [ ] Top-of-leaderboard score (track improvement)
- [ ] Submission diversity (how many distinct submitters)
- [ ] Geographic distribution (US / Europe / China / rest of world)
- [ ] Twitter/X engagement (likes, retweets, replies)
- [ ] HackerNews position (if posted)

### Monthly (Month 1-6)

- [ ] Full submission count + cumulative
- [ ] Group A user funnel (outreach → expressed interest → submission)
- [ ] Co-founder #2 candidate pipeline status
- [ ] arXiv companion paper download / citation tracking
- [ ] Track B mentor search status
- [ ] Token-speculation discourse volume (qualitative)

### KPI targets (from PWM_USER_ACQUISITION_STRATEGY §3.4)

- **D9+30:** 3-5 internal-team submissions
- **D9+90:** 5-10 external submissions
- **D9+180:** 10-30 total submissions
- **D9+365:** 50+ submissions OR 5+ labs involved
- **D9+730:** users paying / staking to run verified evaluation

If KPIs underperform by 50% or more at any milestone, trigger strategy review per `PWM_USER_ACQUISITION_STRATEGY §9.2`.

---

## SECTION 14: CHANGE LOG

| Date | Change | Author |
|---|---|---|
| 2026-05-22 | Initial draft (PWM-CI-1 launch focused; two-track framing; product-leads-token sequence) | Director + Claude |
| | | |

---

*This doc is the implementation-ready landing page draft. Sub-GPU reviews + implements in `pwm_product/platform/pwm_platform/`. Director sign-off on framing decisions in `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` §10 unlocks implementation. Iterate based on Director feedback before D9 launch.*
