# PWM Announcement Bundle (Step 5.6) — 2026-05-22

**Date:** 2026-05-22
**Audience:** Director (paste-ready announcement copy)
**Status:** Ready for Director to post when Step 5.5 explorer cutover completes + Director chooses timing.
**Purpose:** Concrete copy for Twitter/X, HackerNews, Reddit, LinkedIn, arXiv, outreach emails. Lead with AI4Science framing per `PWM_VALUE_FRAMING_2026-05-22.md`.

This doc complements:
- `coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — Source for hero copy + framing
- `coordination/PWM_VALUE_FRAMING_2026-05-22.md` — AI4Science framing canonical
- `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — Target audience
- `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md` — Verified deploy facts

---

## Timing strategy

PWM mainnet went live 2026-05-22T18:52:09Z. **Soft-launch caps active until ~2026-06-21 (D9+30).** Two announcement waves:

| Wave | Date | What to say | Why |
|---|---|---|---|
| **Wave 1 (now)** | D9 (today / this week) | "PWM mainnet deployed. Soft-launch audit window underway. PWM-CI-1 CASSI benchmark opens 2026-06-21." | Establish credibility; build interest before mining activates |
| **Wave 2 (D9+30)** | 2026-06-21 | "PWM-CI-1 NOW OPEN. Submit your AI4Science CASSI reconstruction method. Top 3 win up to 5,000 PWM." | Concrete CTA; mining activated; live competition |

**Recommendation:** Do Wave 1 quietly (Twitter/X + maybe one academic mailing list) for now. Save the big push (HN Show HN + Reddit + LinkedIn + press) for Wave 2 when there's a real CTA users can act on.

---

## 1. Twitter/X launch thread — Wave 1 (15 tweets)

**Use this NOW** to announce deploy + tease the D9+30 PWM-CI-1 opening.

### Tweet 1/15
> PWM mainnet is live on Base.
>
> Two cornerstone AI4Science principles registered: CASSI + CACTI (compressive spectral + temporal imaging).
>
> Soft-launch audit period: now to June 21. PWM-CI-1 benchmark opens then.
>
> 🧵 What it is + why →

### Tweet 2/15
> PWM = Physics World Model.
>
> An open-source platform for **verified AI4Science methods** in physics-grounded problems.
>
> Researchers submit reconstruction methods → cryptographic verification → leaderboard → citation cert hash.
>
> No central trust required.

### Tweet 3/15
> What problem PWM solves:
>
> Test-set leakage. Once a benchmark's test set is public, methods overfit. Most "state-of-the-art" claims are corrupted.
>
> PWM commits the test set hash on-chain. The set is sealed. Your PSNR is the real PSNR.

### Tweet 4/15
> Today's deploy (Base mainnet, chainId 8453):
>
> • 9 smart contracts ✅ verified on Basescan
> • 21M PWM fixed supply
> • PWMRegistry owned by 3-of-5 multisig
> • 6 genesis artifacts: CASSI + CACTI L1/L2/L3
>
> Independent verification: 10/10 PASS

### Tweet 5/15
> First benchmark: PWM-CI-1.
>
> Task: CASSI reconstruction from compressive measurements.
> Dataset: public CASSI dataset (sealed test set on-chain).
> Metric: PSNR + SSIM.
> Prize: 10,000 PWM (Top-3: 5K/2.5K/1.5K).
> Entry: free (no fee in Round 1).

### Tweet 6/15
> Who this is for:
>
> PhD students + postdocs with novel reconstruction methods.
>
> Submit your method → get a verified PSNR/SSIM score → cite the cert hash in your paper.
>
> Your method's score is real, sealed, and unforgeable.

### Tweet 7/15
> How it works:
>
> 1. Fork github.com/integritynoble/pwm-ci-1
> 2. Implement the standard inference interface
> 3. Submit a Docker container
> 4. PWM verifies against sealed test set
> 5. Your method appears on leaderboard with citable cert hash

### Tweet 8/15
> What you DON'T need:
>
> • No crypto/Web3 background (wallet setup is 5 min)
> • No payment to submit (Round 1 is sponsored)
> • No KYC, no equity, no terms-of-service hostage situation
>
> All methods are MIT-licensed. Submit, download, run locally — free.

### Tweet 9/15
> Why blockchain?
>
> Not for tokens or speculation. For three specific things:
>
> 1. Cryptographic test-set commitment (no leakage possible)
> 2. Unforgeable submission timestamps (priority claims)
> 3. Programmatic reward distribution (no human discretion)
>
> The product is the benchmark. The blockchain is plumbing.

### Tweet 10/15
> Roadmap:
>
> • Today–June 21 (D9 to D9+30): Soft-launch audit window. Read access only.
> • June 21+: PWM-CI-1 submissions OPEN. Mining activates.
> • Months 6-12: PWM-CI-2 (compressed sensing), PWM-CI-3 (spectral), PWM-MED-1 (low-dose CT)
> • 2028: RSNA/ISBI medical imaging flagship

### Tweet 11/15
> Genesis content commitment:
>
> Just 2 fully v3-verified anchors at launch (CASSI + CACTI). Other 1,591 stub-tier principles stay on testnet until polished via Bounty 7.
>
> We chose narrow + verified over wide + low-quality. PWM grows with each polished benchmark.

### Tweet 12/15
> Governance:
>
> 3-of-5 multisig at deploy (Path A bootstrap). Founder rotations Months 1-6 post-mainnet.
>
> NumFOCUS Round 4 sponsorship pending. Foundation 501(c)(3) trajectory.
>
> Not affiliated with UTSW or any commercial entity.

### Tweet 13/15
> Contracts on Base mainnet:
>
> PWMToken: 0x7326781182b9cDc1eF9Fa147fB689862f893dA14
> PWMRegistry: 0x9F91784c2fa884A79473304050C581424E006fbd
> PWMGovernance: 0x83F210b9A8E5F0FAfE133c700F888b3A303f9b15
>
> Full list: physicsworldmodel.org/contracts

### Tweet 14/15
> Open-source: github.com/integritynoble/pwm
> MIT licensed. All smart contracts + indexer + explorer.
>
> Bounty program: 10 active bounties / ~1.3M PWM pool. Including Mobile UX (40K), MCP server (25K), Smart contracts competing impl (500K).
>
> bounties/INDEX.md

### Tweet 15/15
> Built by @platformaiyang + collaborators.
>
> Founded 2024. Mainnet 2026. Soft-launch through June 21.
>
> Questions: Discord (link soon) or director@physicsworldmodel.org
>
> If you've got an AI4Science method that wants a verified score — this is the platform.
>
> 🔬⛓️

---

## 2. HackerNews "Show HN" post — Wave 2 (use after PWM-CI-1 opens, D9+30+)

### Title (under 80 chars)
> Show HN: PWM-CI-1 — Cryptographically-verified CASSI reconstruction benchmark

### Body
> Hi HN,
>
> I'm Zhang Yang, faculty at UTSW Medical Center. After 2 years of building, we just opened PWM-CI-1, a new computational-imaging benchmark with two technical novelties:
>
> 1. **Leak-proof test set:** Committed on-chain at benchmark launch (sha256 hash to Base mainnet smart contract). Cannot be modified. No trusted host can leak it.
>
> 2. **Verified reproduction certificates:** Every submission generates a cryptographic L4 certificate proving the method's verified score. Citable as `0x617d3707…` in papers.
>
> PWM (Physics World Model) is the broader platform — an open-source benchmark substrate for verified AI4Science methods. PWM-CI-1 is the first concrete benchmark, anchored to the CASSI (Coded Aperture Snapshot Spectral Imaging) principle. Other benchmarks coming: compressed sensing, spectral imaging, low-dose CT (Track 9 / RSNA 2028).
>
> The smart contracts are on Base mainnet (chainId 8453), 9 contracts total, Basescan-verified. Genesis: CASSI L1-025 + CACTI L1-027 plus their L2 specs and L3 benchmarks. 21M PWM fixed supply. 3-of-5 multisig governance.
>
> **Specifically why I built this:**
>
> Test-set leakage is the #1 problem with ML benchmarks. ImageNet was leaked. SQuAD was leaked. NLP benchmarks have known leaks. The standard mitigation is "trust the host" — Kaggle, Hugging Face, MLPerf all rely on operator integrity. PWM removes the trust assumption via on-chain commitment.
>
> **What I'd love feedback on:**
>
> - Submission interface: is the Docker contract reasonable? Anything missing for typical CASSI methods?
> - Scoring: PSNR + SSIM. Should we add LPIPS / perceptual metrics?
> - Reward model: top-3 win 5K/2.5K/1.5K PWM (Reserve-sponsored). Round 1 is free; future rounds may require small refundable entry stake. Is this the right cadence?
> - Verification: deterministic Docker, single-GPU, fixed seeds. Are there edge cases I'm missing?
>
> **Roadmap:**
> - PWM-CI-2 (compressed sensing) — D9+90
> - PWM-CI-3 (spectral imaging) — D9+180
> - PWM-MED-1 (low-dose CT, public data) — D9+365
> - PWM-MED-2 (low-dose CT, clinical-grade) — 2028 RSNA/ISBI launch
>
> **Links:**
> - Landing: https://physicsworldmodel.org
> - Repo: https://github.com/integritynoble/pwm-ci-1
> - Technical report (arXiv): [link when published]
> - Discord: [link]
>
> **Disclosures:**
> - I'm faculty at UTSW Medical Center but PWM is operated independently. UTSW does not endorse or fund PWM.
> - PWM tokens are utility tokens for protocol participation. Not investment contracts. NumFOCUS Round 4 sponsorship pending; Foundation 501(c)(3) trajectory.
>
> Happy to answer questions about technical decisions, the verification mechanism, or strategy.

---

## 3. Reddit r/MachineLearning post — Wave 2

### Title
> [P] PWM-CI-1: Cryptographically-Verified CASSI Reconstruction Benchmark (sealed test set on-chain)

### Body
> **TL;DR:** New computational imaging benchmark where the test set hash is cryptographically committed to a smart contract on Base mainnet. No leakage possible, even by the benchmark host. Submissions earn verifiable certificate hashes citable in papers.
>
> **Why this exists**
>
> Test-set leakage is endemic in ML benchmarks. Every "SOTA" claim implicitly trusts that the benchmark host's test set hasn't been exposed to the model. As benchmarks age, this assumption fails — quietly. PWM-CI-1 makes the assumption cryptographically enforceable.
>
> **PWM-CI-1 specifics:**
> - **Task:** CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction from compressive measurements
> - **Dataset:** Public CASSI dataset, held-out test set with sealed hash on Base mainnet
> - **Metric:** PSNR + SSIM
> - **Prize:** 10,000 PWM (rank 1: 5K, rank 2: 2.5K, rank 3: 1.5K)
> - **Submission:** Docker container with standard inference interface
> - **License:** MIT for all submissions; baseline + eval code MIT
> - **Entry fee:** Free (Round 1 sponsored; future rounds may have refundable stake)
>
> **Mechanism:**
> 1. Fork github.com/integritynoble/pwm-ci-1
> 2. Implement standard inference interface
> 3. Submit Docker container
> 4. PWM runs against sealed test set → PSNR/SSIM computed deterministically
> 5. Cryptographic L4 certificate written to PWMRegistry contract
> 6. Cite the cert hash in your paper as proof of verified score
>
> **Open questions:**
> - Is PSNR + SSIM sufficient or should we add perceptual metrics (LPIPS, FID)?
> - The Docker contract is single-GPU deterministic seeds. Are there CASSI methods this constrains unfairly?
> - Beyond CASSI, what other computational imaging benchmarks would benefit from on-chain commitment?
>
> **Roadmap:**
> - PWM-CI-2 (compressed sensing) — D9+90
> - PWM-CI-3 (spectral imaging) — D9+180
> - PWM-MED-1 (low-dose CT, public data) — D9+365
>
> **Links:**
> - Landing: https://physicsworldmodel.org
> - Repo: https://github.com/integritynoble/pwm-ci-1
> - Technical report: https://arxiv.org/abs/[TBD]
> - Smart contracts on Base mainnet: https://basescan.org/address/0x9F91784c2fa884A79473304050C581424E006fbd
>
> Disclosures: I'm faculty at UTSW but PWM is independent. PWM tokens are utility tokens for protocol participation, not investment contracts. NumFOCUS Foundation track.

---

## 4. Reddit r/computervision post — Wave 2

### Title
> [Project] Verified CASSI reconstruction benchmark with cryptographically-sealed test set

### Body
> If you work in computational imaging (especially CASSI / compressive spectral / snapshot hyperspectral), we just opened PWM-CI-1 — a benchmark where the test set is cryptographically committed on Base mainnet and cannot be leaked, even by the benchmark host.
>
> **Why this matters for CV/imaging:**
>
> CASSI methods are typically evaluated on the KAIST CAVE dataset or similar. These datasets are years old, public, and have known leakage — many "state-of-the-art" claims include test-set contamination through hyperparameter tuning. PWM-CI-1 commits the test set hash to a smart contract; even the benchmark authors can't reveal it before submission deadline.
>
> **Test set sealed via:**
> - SHA256 hash of held-out scans committed to PWMRegistry contract
> - Contract: 0x9F91784c2fa884A79473304050C581424E006fbd on Base mainnet
> - Verifiable on Basescan
>
> **Submission:**
> - Standard inference interface in a Docker container
> - Single-GPU, deterministic seeds
> - PSNR + SSIM scoring (perceptual metrics planned for Round 2)
> - Round 1 sponsored (no entry fee); top-3 win 5K/2.5K/1.5K PWM
>
> Repo: github.com/integritynoble/pwm-ci-1
> Landing: physicsworldmodel.org
>
> Methods that earn a top-3 placement get a citable cryptographic certificate hash, plus optional co-authorship on the benchmark report (target venue: MICCAI 2026 or NeurIPS 2026 D&B).
>
> Happy to answer questions about the Docker contract, the scoring, or CASSI-specific quirks.

---

## 5. LinkedIn post — Wave 2 (more formal)

### Body
> After two years of building, we just opened **PWM-CI-1**, the first benchmark on Physics World Model (PWM) — an open-source verification platform for AI4Science methods.
>
> What makes PWM-CI-1 different from existing computational imaging benchmarks: the test set is **cryptographically committed on Base mainnet**. It cannot be leaked, even by the benchmark host. The "SOTA" claims you see on traditional ML benchmarks often include subtle test-set contamination; on PWM, that's impossible.
>
> 🎯 **Specifically:**
> • Task: CASSI reconstruction (Coded Aperture Snapshot Spectral Imaging)
> • Sealed test set hash on PWM Registry smart contract
> • Standard Docker inference interface
> • PSNR + SSIM scoring with deterministic verification
> • Top-3 win up to 5,000 PWM tokens (Reserve-sponsored)
> • All submissions MIT-licensed; baseline + eval code public
>
> 📐 **Why blockchain?**
>
> Not for tokens or speculation. For three specific properties:
> 1. **Immutable test-set commitment** — no central party can leak the test
> 2. **Cryptographic submission timestamps** — priority claims are unforgeable
> 3. **Programmatic reward distribution** — no human discretion
>
> The product is the benchmark. The blockchain is plumbing.
>
> 🔬 **Roadmap:**
> • PWM-CI-2 (compressed sensing), PWM-CI-3 (spectral) coming in Months 3-6
> • PWM-MED-1 (low-dose CT, public data) in Month 12
> • Medical imaging flagship at RSNA/ISBI 2028
>
> 🏛️ **Disclosure:** I'm faculty at UT Southwestern Medical Center but PWM is operated independently. UTSW does not endorse or fund the project. NumFOCUS sponsorship pending; Foundation 501(c)(3) trajectory.
>
> If you're a researcher, PhD student, or postdoc working on inverse problems — your method's score on a leak-proof benchmark is much more credible than its score on a leakable one. PWM-CI-1 gives you that score.
>
> Submit your method: github.com/integritynoble/pwm-ci-1
> Learn more: physicsworldmodel.org
>
> Happy to discuss the technical decisions or the broader vision in the comments.
>
> #AI4Science #ComputationalImaging #ReproducibleResearch #CASSI #ComputerVision #MachineLearning

---

## 6. arXiv companion paper plan — to draft Months 1-3 post-deploy

### Suggested title
> "PWM: A Verified Benchmark Substrate for AI4Science with Cryptographically-Sealed Test Sets"

### Abstract (draft)
> Machine learning benchmark integrity depends on test-set secrecy, but leakage is endemic. Once a test set is public, hyperparameter tuning and unconscious overfitting corrupt subsequent SOTA claims; once a host has exposed it, even unintentionally, no recovery is possible. We present **PWM (Physics World Model)**, an open-source benchmark substrate that addresses test-set leakage through cryptographic commitment on a public blockchain. PWM commits the SHA256 hash of held-out test data to an on-chain smart contract at benchmark creation. Submissions to PWM benchmarks are evaluated in a deterministic Docker environment; verified PSNR/SSIM scores are written as cryptographic L4 certificates citable in subsequent literature. We demonstrate the substrate with PWM-CI-1, a CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction benchmark, and discuss extensions to compressive sensing, spectral imaging, and low-dose CT reconstruction. PWM's contributions: (a) a substrate decoupling benchmark integrity from operator trust, (b) a cryptographic citation graph enabling reproducibility tracking across the literature, and (c) an open-source platform demonstrated with the first verified AI4Science methods on Base mainnet (2026-05-22).

### Outline
1. Introduction — test-set leakage as endemic ML problem
2. Related work — Kaggle, MLPerf, Hugging Face, Papers with Code limitations
3. PWM substrate — three layers (Principle / Spec / Benchmark)
4. Cryptographic verification — commit-and-reveal scheme
5. PWM-CI-1 case study — CASSI reconstruction
6. Worked example — submission to citation
7. Discussion — scope, limitations, future benchmarks
8. Conclusion

### Authors
- Zhang Yang (Director, lead author)
- Heyang Zhao (intern, weeks 10-16 contribution)
- Co-founder #2 (when signed)
- Track K mentor (when committed; for clinical sections)

### Target venue
- arXiv (cs.DL or stat.ML) for the preprint
- Submission to NeurIPS Datasets & Benchmarks track 2026 (deadline ~June 6)

---

## 7. Outreach email templates

### 7.1 Academic researcher invitation (cold)

> Subject: Inviting your CASSI method to a cryptographically-verified benchmark
>
> Dear [Researcher Name],
>
> I noticed your recent paper "[Paper Title]" on CASSI reconstruction. I'd like to invite you to submit your method to PWM-CI-1, a new leak-proof CASSI reconstruction benchmark we launched on 2026-06-21 (after a 30-day audit window post-deploy).
>
> Why this might interest you:
> - The test set is cryptographically committed on-chain; no central party can leak it
> - Your method's PSNR/SSIM score is verified deterministically and citable as a cert hash
> - Top 3 win 5K/2.5K/1.5K PWM tokens + co-authorship offer on the benchmark report
> - Round 1 is free (sponsored by PWM Foundation)
> - All submissions MIT-licensed
>
> If interested, the repo is at github.com/integritynoble/pwm-ci-1 — submission is a Docker container implementing a standard inference interface. ~30 minutes from clone to first submission.
>
> Happy to answer any questions or walk through the submission flow.
>
> Best,
> Zhang Yang
> Director, PWM Foundation
> Faculty, UT Southwestern Medical Center (independent of UTSW for this project)
> https://physicsworldmodel.org

### 7.2 AI lab partnership (Anthropic, HuggingFace, OpenAI eval teams)

> Subject: Verified AI4Science benchmark — partnership inquiry
>
> Dear [Team Lead Name],
>
> I'm Zhang Yang, founder of PWM (Physics World Model) — an open-source verification platform for AI4Science methods. We just launched on Base mainnet (2026-05-22) with our first benchmark (PWM-CI-1) opening 2026-06-21.
>
> Why this is relevant to [Anthropic / HuggingFace / OpenAI] eval work:
> - PWM benchmarks have **cryptographically sealed test sets** — no leakage possible, even by us
> - Submissions generate verified PSNR/SSIM scores citable as cert hashes
> - The substrate scales beyond CASSI to compressive sensing, spectral imaging, low-dose CT
> - We're MIT-licensed, NumFOCUS-track, Foundation 501(c)(3) trajectory
>
> Potential partnership angles:
> - PWM cert hashes as training data quality signal (verified methods + verified outputs)
> - Cross-link evals: [Anthropic / your] internal benchmarks reference PWM cert hashes for verified comparisons
> - MCP server integration: PWM is queryable from [Claude / your assistant] via Bounty 9
>
> Would be glad to schedule a 30-min call to discuss. Available next week — what's your preferred time zone?
>
> Best,
> Zhang Yang
> Director, PWM Foundation
> https://physicsworldmodel.org
>
> P.S. NumFOCUS Round 4 sponsorship is pending; Foundation 501(c)(3) trajectory. We're not a token project — we're an open-source verification platform that happens to use blockchain for the cryptographic substrate.

### 7.3 PI / lab head invitation (medium-warm contact)

> Subject: Invitation to host a verified benchmark on PWM
>
> Dear Prof. [Name],
>
> I've followed your work on [specific imaging problem] with interest — particularly [specific paper or contribution]. After 2 years of building, I just launched PWM (Physics World Model), an open-source verified benchmark platform for AI4Science. Our first benchmark (PWM-CI-1) opens 2026-06-21 with CASSI reconstruction as the inaugural task.
>
> I'd love to discuss the possibility of hosting a benchmark in your domain on PWM. The mechanics:
> - You define the benchmark (task, dataset, scoring metric, test set)
> - PWM commits the test set hash on-chain; submissions are verified deterministically
> - Your lab is recognized as the benchmark author; PWM royalties flow to a per-benchmark T_k pool over time
> - All MIT-licensed; no equity, no IP transfer, no surrender of academic credit
>
> Domain candidates I think would benefit:
> - [Their specific subfield]
> - [Adjacent problem with reproducibility concerns]
>
> Would you be open to a 15-minute call to discuss the technical model and what hosting a benchmark would involve?
>
> Best,
> Zhang Yang
> Director, PWM Foundation
> Faculty, UT Southwestern (independent of UTSW for this project)
> https://physicsworldmodel.org

---

## 8. Press / journalist pitch

### Subject
> First mainnet-verified AI4Science benchmark goes live on Base

### Body (2 paragraphs)
> Today marks the launch of PWM (Physics World Model), an open-source benchmark platform for AI4Science with a novel approach to a long-standing problem in machine learning: test-set leakage. PWM commits benchmark test sets cryptographically to a public blockchain at creation time — meaning no central party, including the benchmark authors themselves, can leak the test data prior to evaluation. The first benchmark, PWM-CI-1, opens 2026-06-21 with CASSI (Coded Aperture Snapshot Spectral Imaging) reconstruction; future benchmarks will cover compressive sensing, spectral imaging, and low-dose CT reconstruction.
>
> PWM was deployed to Base mainnet on 2026-05-22 with a 30-day soft-launch audit window. The platform is operated by an independent Foundation (NumFOCUS Round 4 sponsorship pending), distinct from any commercial entity. Smart contracts are open-source (MIT-licensed) and verified on Basescan. PWM's value proposition is not token speculation but verification infrastructure — the protocol's 21M-supply PWM tokens are used as participation incentives, not investment vehicles.
>
> Founder Zhang Yang (faculty at UT Southwestern, operating PWM independently from his university role) is available for interviews on the technical mechanism, the broader AI4Science vision, or the cryptographic verification approach.
>
> Contact: director@physicsworldmodel.org
> Landing: https://physicsworldmodel.org

---

## 9. Post-Wave-2 follow-ups (week-by-week)

### Week 1 post-launch (D9+30 to D9+37)
- Daily Twitter/X engagement (responding to comments, new threads)
- Personal follow-ups to 30-50 candidate users from outreach list
- HN post (Tuesday morning ET; highest visibility window)
- Reddit posts (Wednesday)
- LinkedIn post (Thursday)

### Week 2-4 (D9+37 to D9+57)
- Track first submissions; publicly highlight top-3 cert hashes
- Reach out to journalists/podcasters who covered HN
- Workshop submission to MICCAI 2026 (deadline ~Jun 7)
- arXiv companion paper submission (week 3)

### Month 2-3 (D9+60 to D9+90)
- Co-founder #2 introduction (when signed); their lab contributes 2-5 method submissions
- AI/data partnership outreach (Anthropic, HuggingFace, OpenAI eval teams)
- First interim benchmark report on arXiv

---

## 10. What to NOT say

❌ "Earn PWM tokens by mining" — sounds like Helium pattern; scares academics
❌ "Tokenize your research" — sounds extractive
❌ "PWM will moon" or any price speculation
❌ "Decentralized science" or "DeSci" — generic Web3 buzzword
❌ "We have 1,591 principles" — only 6 on mainnet; rest are stub-tier on testnet
❌ "UTSW endorses PWM" — UTSW is independent; mention only as Director's affiliation
❌ "PWM is an investment opportunity" — securities law concern

---

## 11. Cross-references

- `coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — Hero copy + framing
- `coordination/PWM_VALUE_FRAMING_2026-05-22.md` — AI4Science framing canonical
- `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — Target audiences
- `coordination/PWM_TOKEN_UTILITY_AND_VALUE_2026-05-22.md` — Securities-aware language
- `deploy/PWM_MAINNET_DEPLOY_LOG_2026-05-22.md` — Verified contract addresses + tx hashes
- `coordination/CRISIS_COMMS_PLAYBOOK_2026-05-21.md` — Response templates if announcement attracts criticism

---

## 12. Bottom line

**Wave 1 (now, low-key):** Twitter/X thread + maybe one academic mailing list. Sets up Wave 2.

**Wave 2 (D9+30, big push):** HN Show HN + Reddit (r/ML + r/computervision) + LinkedIn + email outreach + arXiv companion paper.

**Don't post Wave 2 until:**
- ✅ Phase 1b activated (caps lifted; mining live)
- ✅ Landing page (physicsworldmodel.org) fully serves the explorer
- ✅ PWM-CI-1 submission flow tested end-to-end
- ✅ At least 1-2 internal-team submissions populate the leaderboard (so it doesn't look empty)
- ✅ arXiv preprint submitted (referenceable from HN/Reddit posts)

---

*This doc is the canonical announcement bundle for PWM launch. Update if facts change. Director chooses which channels to post and when.*
