# PWM Value Framing — Canonical Reference

**Date:** 2026-05-22
**Audience:** Director + sub-GPU + marketing + co-founders + grant reviewers + bounty winners
**Status:** Canonical reference for "What is PWM's value? What do we sell?"
**Purpose:** Resolves Director's 2026-05-22 framing question: *"Among (verified leaderboard, citable cert hash) and (AI4Science, solution + mismatch), which is better as the value of PWM?"*

This doc complements:
- `PWM_TOKEN_UTILITY_AND_VALUE_2026-05-22.md` — Token economics value drivers
- `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — Two-track demand-side strategy
- `PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — Phased feature activation
- `PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — Launch marketing copy

This doc answers: **WHAT is PWM's value framing?**

---

## TL;DR

1. **PWM's value is AI4Science solutions, not the verification mechanism.** Director's 2026-05-22 framing question is correctly answered by Option B (AI4Science + Solution + Mismatch), not Option A (verified leaderboard + cert hash).

2. **The distinction is WHAT vs HOW:**
   - **WHAT (value):** AI4Science solutions — actual AI methods that solve physics-grounded problems
   - **HOW (mechanism):** Verified Mismatch + cert hash + leaderboard — the quality guarantee

3. **Lead with WHAT, not HOW.** Successful protocols (AlphaFold, Chainlink, arXiv) market their PRODUCTS, not their mechanisms.

4. **Canonical value framing:**
   > **"PWM is the verified AI4Science platform for physics-grounded problems."**

   "Verified" is the adjective. "AI4Science" is the noun. "Platform" is the substrate.

5. **Phase 1 vertical-specific framing:**
   > **"PWM-CI-1 is the verified CASSI reconstruction benchmark. The leading AI4Science methods compete here for ranked rewards."**

6. **Existing docs need updates** to lead with AI4Science instead of verification mechanisms (§7).

---

## 1. The Framing Question

Director asked on 2026-05-22:

> "You neglect the solution (AI4Science). Can we use AI4Science as the value, especially solution? Among (verified leaderboard, citable cert hash) and (AI4Science, solution and mismatch) which is better as the value of PWM?"

### 1.1 The two candidate framings

**Option A — Verified leaderboard + citable cert hash**
- PWM's value is the *verification infrastructure*
- What users get: cryptographic proof their method works, citation hash for paper
- Compared to: arXiv with cryptographic timestamps; rigorous MLCommons
- Demand driver: researchers wanting credibility / reproducibility verification

**Option B — AI4Science (solution + Mismatch)**
- PWM's value is the *actual AI methods*
- What users get: ready-to-use AI4Science solutions for their scientific problems
- Compared to: HuggingFace for AI4Science; verified compute services
- Demand driver: scientists, clinicians, AI engineers wanting solutions they can USE

### 1.2 Which is correct?

**Option B is correct.** Director's intuition is right. Here's why.

---

## 2. WHAT vs HOW — The Marketing Principle

### 2.1 The principle

Successful protocols + products market their PRODUCT, not their MECHANISM. The mechanism is a moat — it supports the product, but it's not the lead.

| Successful protocol | WHAT they sell (value / product) | HOW they do it (mechanism) |
|---|---|---|
| **AlphaFold** | "Protein structures" | Deep learning with attention |
| **Chainlink** | "Reliable price feeds" | Decentralized oracle network |
| **arXiv** | "Open access research papers" | Pre-print server with peer review |
| **Stripe** | "Accept payments easily" | Payment processing API |
| **AWS** | "Cloud computing for any workload" | Distributed infrastructure |
| **HuggingFace** | "Models for every AI task" | Model hub + transformers library |

Notice: in every case, what you remember (the value) is the WHAT, not the HOW. AlphaFold isn't "deep learning with attention"; it's "protein structures."

### 2.2 PWM's current framing problem

My earlier framing (`PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md`):

> "Verified benchmarks for physics-grounded AI."

This is the HOW (verification mechanism). It's accurate but it's not compelling. It describes the moat, not the product.

### 2.3 The corrected framing

Lead with WHAT:

> **"PWM is the verified AI4Science platform for physics-grounded problems."**

Or, more concrete for Phase 1:

> **"PWM-CI-1 is the verified CASSI reconstruction benchmark. The leading AI4Science methods compete here."**

The mechanism (verification) is now the adjective ("verified") — supporting, not leading. The product (AI4Science methods) is the noun — the actual value.

---

## 3. Why AI4Science Is the Better Lead

### 3.1 It maps to a recognized academic field

"AI4Science" is established terminology:
- NeurIPS workshops: "AI4Science", "AI for Science"
- ICML workshops: "ML4Sci", "AI4Science"
- NSF program: "AI for Scientific Research"
- DOE program: "AI for Science"
- Conferences: AAAI Bridge programs, NeurIPS ML4PS
- Foundations: Schmidt Futures, Sloan, CZI all fund AI4Science

By using "AI4Science," PWM positions itself within an existing academic conversation that researchers + grant reviewers + AI lab evaluators all recognize. By using "verified benchmark platform," PWM positions itself in a category that's harder to evaluate.

### 3.2 It scales to all user groups

Different user groups care about different things, but ALL care about AI4Science:

| User group | Why they care about AI4Science |
|---|---|
| **PhD students / postdocs (miners; Group A)** | Want to produce AI4Science solutions that get recognized |
| **PIs / lab heads (Group B)** | Want their lab to be associated with AI4Science methods |
| **Clinicians / radiologists (Group C subset)** | Want AI4Science solutions they can use clinically |
| **AI labs (Anthropic / HuggingFace / OpenAI; Group C)** | Want AI4Science solutions as benchmarks + training data |
| **Grant reviewers** | Want to fund AI4Science (an established priority) |
| **Investors / funders** | Want to invest in AI4Science (a recognized growth area) |

By contrast, "verified leaderboard" only resonates with Group A submitters. Other groups don't care about leaderboards per se — they care about AI4Science.

### 3.3 It describes a PRODUCT, not infrastructure

People buy products, not infrastructure. Examples:

- Customers buy "iPhones" (product), not "ARM-based mobile computing" (infrastructure)
- Researchers buy "AlphaFold structures" (product), not "neural network outputs" (infrastructure)
- Users buy "Chainlink price feeds" (product), not "oracle network calls" (infrastructure)

PWM should sell "AI4Science solutions" (product), not "verified leaderboard" (infrastructure).

### 3.4 It avoids crypto/blockchain connotations

"Verified leaderboard" can sound like a crypto-grift to academic audiences. "Token", "blockchain", "verification" all carry baggage. "AI4Science" carries no baggage — it's pure academic positioning.

### 3.5 It opens demand-side revenue

The biggest demand-side revenue eventually will come from Group C (AI/data consumers, clinicians, AI labs). These users pay for AI4Science SOLUTIONS, not for abstract verification. They want to:

- Run a verified AI4Science method on their own data
- Integrate AI4Science methods into their workflow
- Get verified inference results from established methods

**They will NOT pay for "leaderboard access" or "verification services."** They will pay for AI4Science solutions that solve their problems.

If PWM's value framing leads with verification, demand-side revenue is hard to materialize. If PWM's value framing leads with AI4Science, demand-side revenue has a clear path (Phase 3 mine-to-use mechanism).

---

## 4. But Verification IS Still the Moat

This is important: AI4Science as the value framing doesn't mean verification is unimportant. Verification is what makes PWM's AI4Science TRUSTWORTHY. Without verification, PWM's AI4Science would just be random GitHub repos.

### 4.1 What verification provides

1. **Quality guarantee** — Mismatch ensures methods actually achieve claimed performance on held-out test sets
2. **Trust layer** — cert hash provides cryptographic proof of verified score
3. **Reproducibility** — L4 reproduction certificates show methods can be reproduced
4. **Cryptographic timestamps** — priority claims are unforgeable
5. **Citation infrastructure** — papers can cite specific cert hashes
6. **Anti-leakage** — test sets sealed on-chain prevent overfitting

### 4.2 Without verification, PWM is just HuggingFace

If PWM dropped the verification layer, it would be a model hub for AI4Science methods. That's a useful service but not unique — HuggingFace + Papers with Code + Kaggle already exist.

PWM's UNIQUE moat is verification. But the moat is what defends the product (AI4Science solutions), not the product itself.

### 4.3 The combined framing

**PWM's full value statement:**

> "PWM is the verified AI4Science platform for physics-grounded problems. AI4Science solutions on PWM are cryptographically verified via leak-proof benchmarks + reproducible certificates + citation graph. Researchers submit methods; clinicians + AI labs use them with proven verification."

- **Lead (the WHAT):** AI4Science platform for physics-grounded problems
- **Support (the moat):** Cryptographically verified
- **How it works (HOW):** Leak-proof benchmarks + reproducible certs + citation graph
- **Who uses it:** Researchers (submitters/miners) + clinicians + AI labs (users)

This framing leads with product, but explicitly establishes verification as the differentiating moat.

---

## 5. Phase-Specific Value Framing

The value framing should adapt by phase + user audience.

### 5.1 Phase 1 (Months 1-6) — Bootstrapping Submitters/Miners

**Primary audience:** PhD students, postdocs, AI imaging researchers (Group A; submitters/miners)

**What they want:** Verified leaderboard rank for their AI4Science method + citation cert hash for their paper

**Phase 1 framing:**

> "PWM-CI-1 is the verified CASSI reconstruction benchmark. Submit your AI4Science method. Get a cryptographically-verified score. Win up to 5,000 PWM. Cite the cert hash in your paper."

**Note:** Phase 1 framing DOES emphasize the leaderboard + cert hash because submitters care about those. But the value PROPOSITION is "verified AI4Science benchmark," not "verification infrastructure."

### 5.2 Phase 2 (Months 6-12) — Bringing in Verifiers/Miners

**Primary audience:** Compute providers, reproducers (Group D; CP role miners)

**What they want:** PWM rewards for verified reproduction work + reputation as reliable reproducer

**Phase 2 framing:**

> "PWM-CI-1's leading AI4Science methods need independent reproduction. Run verifications on test data. Generate L4 certificates. Earn PWM via ranked-draw rewards."

**Note:** Phase 2 framing emphasizes the AI4Science solutions exist + can be reproduced. The CP miners are SUPPORTING the AI4Science ecosystem, not building a standalone verification service.

### 5.3 Phase 3 (Year 2+) — Bringing in Users (Demand-Side)

**Primary audience:** Clinicians, AI labs, AI/data consumers (Group C)

**What they want:** Use verified AI4Science methods on their own data

**Phase 3 framing:**

> "PWM hosts the world's leading verified AI4Science methods for physics-grounded problems. Run any verified method on your data. Get cryptographically-verified outputs. Pay per inference."

**Note:** Phase 3 framing is the FULL value proposition — "verified AI4Science solutions you can use." This is the demand-side revenue story.

---

## 6. Vertical-Specific Framing Examples

PWM's AI4Science is best marketed via specific verticals, not abstract category.

### 6.1 Computational imaging vertical

**Bad framing (too abstract):**
> "PWM is a verification platform for AI4Science"

**Good framing (vertical-specific):**
> "PWM has the verified leading AI4Science methods for compressed sensing imaging (CASSI). Submit your reconstruction method to PWM-CI-1; benchmark against the top 10 verified methods; cite the verified cert hash."

### 6.2 Medical imaging vertical (Track 9 / Track B)

**Bad framing:**
> "PWM verifies medical AI methods"

**Good framing:**
> "PWM hosts the verified leading AI4Science methods for low-dose CT reconstruction. PWM-MED-1 is the cryptographically-verified benchmark. Top methods reduce CT dose while preserving image quality, with verified PSNR + SSIM scores."

### 6.3 Generic landing page (high-level)

**Bad framing (current PWM_LAUNCH_LANDING_PAGE §1):**
> "Verified benchmarks for physics-grounded AI."

**Good framing (revised):**
> "Verified AI4Science solutions for physics-grounded problems. Starting with computational imaging."

The "verified" stays — it's the moat. But "AI4Science solutions" replaces "benchmarks" because that's the PRODUCT.

---

## 7. Updates Required to Existing Docs

This value-framing decision triggers updates to multiple canonical docs:

### 7.1 `PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md`

**Hero section update:**

OLD:
> "Verified benchmarks for physics-grounded AI."

NEW:
> "Verified AI4Science solutions for physics-grounded problems. Starting with computational imaging."

**Section 4 (Why use PWM) — refocus magnets on AI4Science context:**

- Magnet #1 (Leak-proof benchmarks) → "Leak-proof benchmarks ensure verified AI4Science scores"
- Magnet #2 (Reproduction certificates) → "Verified reproduction proves your AI4Science method works"
- Magnet #5 (Compute-for-tokens) → "Earn PWM by contributing AI4Science methods"

**Section 7 (About PWM) update:**

OLD:
> "PWM (Physics World Model) is an open-source benchmark platform for physics-grounded AI."

NEW:
> "PWM (Physics World Model) is an open-source platform for verified AI4Science solutions in physics-grounded problems."

### 7.2 `PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` (sub-GPU version on main)

**Two-track section update:**
- Track A current: "Fast academic/research wedge"
- Track A revised: "Fast AI4Science research wedge"

**Framing section (§8):**
- Already uses "verified benchmark platform" — update to "verified AI4Science platform"

### 7.3 `PWM_TOKEN_UTILITY_AND_VALUE_2026-05-22.md`

**§11 (The Single Most Important Framing) update:**

OLD:
> "PWM is not a token mining protocol that adds science later. PWM is a science verification platform with a sophisticated token economy."

NEW:
> "PWM is not a token mining protocol that adds science later. PWM is the verified AI4Science platform for physics-grounded problems, with a sophisticated token economy supporting verified solution creation + consumption."

### 7.4 `PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md`

**§3.3 (Mine-first inverts trust direction) update:**

OLD:
> "First users need PWM to prove value to THEM. Mine-first asks users to give first (mine, expend compute, generate verifications) before they receive value (a verified leaderboard score for their own method)."

NEW:
> "First users need PWM to prove value to THEM. Mine-first asks users to give first (mine, expend compute, generate verifications) before they receive value (a verified AI4Science solution catalog they can submit to and benefit from). The value users seek is AI4Science recognition, not abstract verification."

### 7.5 Director's outbound communication

**arXiv companion paper title update:**

OLD:
> "PWM: A Cryptographic Substrate for Scientific Priority and Reproducibility"

NEW:
> "PWM: A Verified AI4Science Platform for Physics-Grounded Problems"

**Twitter/X bio update:**

OLD:
> "Verified benchmarks for physics-grounded AI"

NEW:
> "Verified AI4Science solutions for physics-grounded problems"

**HackerNews + Reddit titles** should also use "AI4Science" framing.

---

## 8. The Three Audiences (Refined)

PWM's value framing must work for three distinct audiences:

### 8.1 Researchers (Submitters / Miners)

**What they want:** Recognition for their AI4Science method.

**Value framing they hear:** "Your AI4Science method gets cryptographically verified. Top-3 win PWM. Citable cert hash for your paper."

**Why this works:** Verification + leaderboard + cert hash all directly serve their goal (recognition).

### 8.2 Funders + Grant Reviewers

**What they want:** Confidence that PWM is real AI4Science infrastructure (not crypto-grift).

**Value framing they hear:** "PWM is the verified AI4Science platform for physics-grounded problems. NumFOCUS sponsorship pending. Foundation 501(c)(3) trajectory. Major imaging conferences citing PWM cert hashes."

**Why this works:** "AI4Science" is a recognized priority area (NSF, NIH, DOE all fund it). "Verified" + "Foundation" + "Conference citations" all signal substance.

### 8.3 Users (Clinicians, AI Labs — Phase 3+ Demand-Side)

**What they want:** Use AI4Science solutions on their own data, with proven verification.

**Value framing they hear:** "Run any verified AI4Science method on your data via PWM. Pay per inference. Get cryptographically-verified outputs."

**Why this works:** Direct utility — a real product they can use. Verification + payment are the trust + access mechanisms.

---

## 9. The Antipattern: "Verification Platform" as Sole Identity

If PWM markets itself ONLY as a "verification platform," it falls into multiple antipatterns:

1. **Too abstract.** What's being verified? Why does anyone care?
2. **No clear user.** Who is a "verification platform user"?
3. **No clear revenue.** Verification alone doesn't generate fees.
4. **Doesn't differentiate.** Everyone claims to "verify" something.
5. **Sounds like compliance software.** Boring positioning.

By contrast, "verified AI4Science platform" tells:
- What's being verified (AI4Science solutions)
- Why users care (AI4Science is a known field)
- How revenue flows (users pay for AI4Science solutions)
- How PWM differentiates (verified AI4Science vs unverified)
- Sounds like research infrastructure (academic positioning)

---

## 10. Cross-References

- `pwm-team/coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` — Update hero + §4 + §7 per §7.1
- `pwm-team/coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` (sub-GPU canonical) — Update two-track wording per §7.2
- `pwm-team/coordination/PWM_TOKEN_UTILITY_AND_VALUE_2026-05-22.md` — Update §11 per §7.3
- `pwm-team/coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — Update §3.3 per §7.4
- `pwm-team/coordination/prevent_copy/PWM_COMPETITIVE_DEFENSE_2026-05-20.md` — Verification is the moat (already aligned)
- `pwm-team/coordination/prevent_copy/PWM_TOKEN_VALUE_DEFENSE_2026-05-20.md` — Concrete demand vertical (AI4Science aligns)
- `pwm-team/coordination/prevent_copy/PWM_REALISTIC_VALUATION_2026-05-20.md` — Token value estimates
- `pwm-team/coordination/PWM_DEVELOPER_COMPENSATION_2026-05-22.md` — Bounty / compensation (already aligned)
- External: NSF AI4Science program; Schmidt Futures AI4Science; NeurIPS AI4Science workshop

---

## 11. The Single Most Important Framing

**PWM is not a verification platform. PWM is the verified AI4Science platform for physics-grounded problems.**

Verification is the moat. AI4Science solutions are the product. The product leads. The moat defends.

---

## 12. The Single Sentence

**PWM sells verified AI4Science solutions (the WHAT), defended by leak-proof benchmarks + reproducible certificates + cryptographic timestamps (the HOW) — Director's 2026-05-22 framing question is correctly answered by leading with AI4Science (Option B), not with verification mechanisms (Option A).**

---

## 13. Director's Decision Points

| # | Decision | Default | Notes |
|---|---|---|---|
| 1 | Approve lead framing: "verified AI4Science platform for physics-grounded problems"? | YES (recommended) | This is the central decision |
| 2 | Approve updating `PWM_LAUNCH_LANDING_PAGE_DRAFT` per §7.1? | YES | sub-GPU executes |
| 3 | Approve updating `PWM_USER_ACQUISITION_STRATEGY` per §7.2? | YES | sub-GPU executes (since sub-GPU owns canonical version) |
| 4 | Approve updating `PWM_TOKEN_UTILITY_AND_VALUE` per §7.3? | YES | Claude executes |
| 5 | Approve updating `PWM_PHASED_ARCHITECTURE_DEPLOYMENT` per §7.4? | YES | Claude executes |
| 6 | Approve external comms updates (arXiv title, Twitter bio, HN/Reddit titles)? | YES | Director executes when launching |
| 7 | Approve vertical-specific framing (PWM-CI-1: CASSI reconstruction; PWM-MED-1: low-dose CT)? | YES (recommended) | Standard pattern for all benchmarks |

---

*This doc is the canonical reference for PWM's value framing. Update when major positioning shifts. Update at Phase 1 → Phase 2 transition if value framing needs adjustment.*
