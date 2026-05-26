# PWM Genesis Principles — Reality Check (2026-05-22)

**Date:** 2026-05-22
**Audience:** Director (deploy day decision)
**Status:** Diagnostic doc — answers "how many Principles are actually v3-verified vs stub-tier?"
**Purpose:** Resolves Director's 2026-05-22 question: "what are the 1,597 genesis Principles? which files are they in?"

This doc complements:
- `coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` — Phase 1a deliverables
- `coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` — references "30 v3 anchors at launch"
- `plan/PLAN.md` Track 2 — claims "Genesis Content materially complete (1,591 on Base Sepolia)"
- `deploy/findings/REGISTRY_HANDOFF_DECISION_2026-05-18.md` — `register_batch.py` Step 5.4a

This doc is the **operational reality check** against those strategic claims before deploying to mainnet.

---

## TL;DR

1. **The "1,597 genesis Principles" claim is true on testnet** — 1,591 artifacts are registered on Base Sepolia per Track 2 status (+ 6 founder-vetted on Eth Sepolia).

2. **But on closer inspection, only 2 of those 1,591 are at v3-polish quality** — CASSI (L1-003) + CACTI (L1-004) in `pwm_product/genesis/`.

3. **The other 1,589 are all `"registration_tier": "stub"`** — auto-generated metadata files in `content/<agent>/principles/<sub-domain>/L*.json`. Not personally reviewed for science correctness.

4. **The "30 v3 anchors at launch" referenced in strategic docs does NOT match the committed filesystem.** Only 2 v3 anchors exist as polished files; the other ~28 (L1-503..L1-531 medical imaging series, etc.) appear to be aspirational plans or unfinished drafts.

5. **Decision needed:** Deploy 2 verified + transparent narrative, OR deploy ~30 with mixed quality + revisable launch claims, OR pause to polish, OR deploy all 1,591 stubs with caveats.

6. **My recommendation: Option A (2 Principles, transparent narrative).** Most honest with reality; PWM-CI-1 launch only needs CASSI on mainnet.

---

## 1. What's actually in the codebase (verified 2026-05-22)

### 1.1 Layer counts

| Layer | Type | Count in `content/` | Files at `"registration_tier": "stub"` |
|---|---|---|---|
| **L1** | Principle (physics-grounded model) | 529 | 529 (100%) |
| **L2** | Spec (executable specification) | 531 | 531 (100%) |
| **L3** | Benchmark (with test data) | 531 | 531 (100%) |
| **Total** | | **1,591** | **1,591 (all stub-tier)** |

(+ 6 founder-vetted on Eth Sepolia per Track 2 plan; not yet examined for tier)

### 1.2 Where files live

```
content/
├── agent-physics/principles/    147 L1, 148 L2, 148 L3
├── agent-imaging/principles/    164 L1, 165 L2, 165 L3
├── agent-signal/principles/      39 L1,  39 L2,  39 L3
├── agent-chemistry/principles/   67 L1,  67 L2,  67 L3
└── agent-applied/principles/    112 L1, 112 L2, 112 L3
```

Each file structure:
```
content/<agent>/principles/<sub-domain>/L<level>-<num>_<name>.json
```

### 1.3 Polished v3 anchors (the REAL Tier A)

The only files at v3 quality:

```
pwm_product/genesis/
├── l1/
│   ├── L1-003.json    (CASSI canonical v3)
│   └── L1-004.json    (CACTI canonical v3)
├── l2/
│   ├── L2-003.json    (CASSI spec)
│   └── L2-004.json    (CACTI spec)
└── l3/
    ├── L3-003.json    (CASSI benchmark)
    └── L3-004.json    (CACTI benchmark)
```

**Total polished v3 anchors: 2** (CASSI + CACTI).

### 1.4 How content/ stubs relate to v3 anchors

The stub files in `content/` reference the v3 anchors. Example:

```
content/agent-imaging/principles/B_compressive_imaging/L1-025_cassi.json:
{
  "artifact_id": "L1-025",
  "registration_tier": "stub",                              ← stub
  "title": "Coded Aperture Snapshot Spectral Imaging (CASSI)",
  "canonical_reference": "pwm_product/genesis/l1/L1-003.json"  ← points to v3 polished
}
```

So `content/` is the registration INDEX (with stubs); `pwm_product/genesis/` is the polished CONTENT for the 2 Tier A anchors only.

### 1.5 Walkthroughs + demos exist (separate from registration tier)

```
pwm_product/walkthroughs/cassi.md, cacti.md
pwm_product/demos/cassi/, cacti/
pwm_product/reference_solvers/cassi/, cacti/
pwm_product/tests/test_cassi_quality.py, test_cacti_quality.py
pwm_product/explainers/cassi.md, cacti.md
```

These exist for CASSI + CACTI but NOT for the other 1,589 Principles.

---

## 2. The gap between strategic claims and reality

### 2.1 What strategic docs claim

`coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` and earlier docs state:

> "30 Tier A founder-authored v3 anchors are already at v2/v3 schema depth and don't consume Bounty 7 funds:
> - CASSI + CACTI
> - 8 v3 standalone multi-physics medical imaging Principles (L1-503..L1-510)
> - 2 newly-authored analytical cores (L1-511 PillCam, L1-518 XRD)
> - 19 v2 PWDR Principles (L1-512..L1-517, L1-519..L1-531)"

This is referenced in:
- `bounties/INDEX.md` Bounty 7 scope description
- `pwm-team/coordination/strategy/EXPLORER_PURPOSE.md`
- `numfocus/PWM_TOKEN_ECONOMY_AND_NUMFOCUS_COMPATIBILITY_2026-05-13.md`
- Multiple PWM_V3_*.md docs in `pwm_product/genesis/`

### 2.2 What the filesystem actually has

- ✅ CASSI v3-polished (L1-003)
- ✅ CACTI v3-polished (L1-004)
- ❌ L1-503..L1-510 medical imaging series — **no files at L1-503+ in content/ or pwm_product/genesis/**
- ❌ L1-511 PillCam — **no file**
- ❌ L1-518 XRD — **no file**
- ❌ L1-512..L1-531 PWDR series — **no files**

**Search verification:**
```bash
find pwm_product/ -name "L1-503*" -o -name "L1-511*" -o -name "L1-518*" 2>/dev/null
# Returns: nothing
```

### 2.3 Why this matters

The strategic docs' "30 v3 anchors at launch" assumption is incorrect against the current filesystem. Specifically:
- PWM-CI-1 launch (Phase 1b) claims "showcase 30 v3 anchors" — actually only 2 exist
- NumFOCUS application materials may reference 30 — they should be revised to 2 (or the polish work must complete first)
- Marketing copy in `PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` references PWM-CI-1 as if multiple v3 Principles exist as comparables

**Either:**
- (A) The polish work to bring 28 more anchors from stub → v3 is INCOMPLETE
- (B) The files exist somewhere not committed yet (uncommitted local work)
- (C) The strategic doc claims were aspirational, not factual

Director needs to clarify which.

---

## 3. What "registration_tier" means

PWM has a tier system for genesis content:

| Tier | What it means | Suitable for mainnet? |
|---|---|---|
| **v3** | Full schema depth; physics-correct; demos + tests + walkthroughs exist; Director or domain expert reviewed | ✅ YES — ready for mainnet showcase |
| **v2** | Improved schema; partially polished; may lack demos | 🟡 Acceptable with caveat |
| **v1** | Basic schema; original commit; not deeply reviewed | 🟡 Acceptable as long-tail catalog |
| **stub** | Auto-generated placeholder metadata; not science-reviewed | ❌ NOT suitable as flagship; OK as long-tail catalog with disclaimer |
| **draft** | Work-in-progress; not yet committed | N/A (don't deploy) |

**Current state: 1,591 stubs + 2 v3 polished + (6 founder-vetted on Eth Sepolia).** 0 v2-tier in `content/`.

This is **NOT** equivalent to "1,597 verified Principles ready for mainnet."

---

## 4. The four realistic deploy options

### Option A — 2 Principles only (CASSI + CACTI)

| | |
|---|---|
| **What ships** | L1/L2/L3-003 (CASSI) + L1/L2/L3-004 (CACTI) — 6 artifacts total |
| **Time at Step 5.4a** | ~5 min |
| **Gas cost** | ~$0.05 |
| **Launch narrative** | "PWM mainnet launches with 2 cornerstone Principles: CASSI and CACTI. More to follow as community polish work completes." |
| **Honesty score** | ✅ High — matches the truly v3-verified state |
| **NumFOCUS credibility** | Slight downgrade — must revise marketing claims of "1,597" |
| **PWM-CI-1 viability** | ✅ Works — only needs CASSI Principle |

### Option B — 2 v3 + ~30 selected stubs (Director picks Tier A candidates)

| | |
|---|---|
| **What ships** | CASSI + CACTI (v3 polished) + ~30 selected stubs Director has personally read |
| **Time at Step 5.4a** | 1-3 hours review + 10 min deploy |
| **Gas cost** | ~$0.30 |
| **Launch narrative** | "PWM launches with 32 founder-reviewed Principles, anchored by 2 fully-polished v3 anchors (CASSI + CACTI)." |
| **Honesty score** | 🟡 Medium — depends on what "reviewed" means; the stubs are auto-generated metadata, not deep science review |
| **NumFOCUS credibility** | Acceptable |
| **PWM-CI-1 viability** | ✅ Works |

### Option C — All 1,591 stubs + 2 v3

| | |
|---|---|
| **What ships** | All 1,591 currently on testnet → mainnet |
| **Time at Step 5.4a** | 1-3 hours |
| **Gas cost** | ~$5-15 |
| **Launch narrative** | "1,591 genesis Principles registered at launch (1 fully polished, 1,589 stub-tier awaiting community polish via Bounty 7)" |
| **Honesty score** | 🟡 Low if you don't add the caveat; medium if you do |
| **NumFOCUS credibility** | Mixed — looks impressive but reviewers may dig in |
| **PWM-CI-1 viability** | ✅ Works |
| **Risk** | Public catalog of 1,591 mostly-stub Principles could attract criticism ("PWM has 1,591 auto-generated artifacts, not 1,591 verified Principles") |

### Option D — Pause deploy; polish ~30 more anchors to v3 first

| | |
|---|---|
| **What ships** | Eventually 30 v3 anchors as strategic docs claim |
| **Time** | Weeks-months of polish work |
| **Gas cost** | Same as Option B at eventual deploy |
| **Launch narrative** | "PWM launches with 30 polished v3 anchors covering computational + medical imaging." |
| **Honesty score** | ✅ High when complete |
| **NumFOCUS credibility** | High |
| **PWM-CI-1 viability** | ✅ Strong (matches strategic doc claim) |
| **Cost** | Deploy day delays by weeks-months |

---

## 5. My honest recommendation

**Option A (2 Principles, transparent narrative) is the best fit for reality.**

Reasons:

1. **Truthful.** Only 2 are truly v3-verified. Deploying 1,591 stubs sends a misleading quality signal.

2. **Reversible.** You can add more Principles later via governance proposal (Phase 1b+).

3. **PWM-CI-1 launch unblocked.** The Phase 1b mining launch only needs CASSI Principle on mainnet (CASSI L1 + CASSI L2 + CASSI L3 benchmark spec).

4. **Honest with grant reviewers.** NumFOCUS reviewers will appreciate a clear "starting cornerstone + ongoing polish program" story over "1,591 mostly-stub artifacts."

5. **Matches Director's verification preference.** Director has personally reviewed (in fact, authored) CASSI + CACTI. The 1,589 stubs would need fresh review work.

6. **Cheapest deploy.** $0.05 gas at Step 5.4a; ~5 minutes.

**The trade-off:** PWM-CI-1 launch landing page won't be able to claim "30 v3 anchors at launch" — that claim was wrong against current filesystem reality. Landing page should be revised to "Launches with CASSI (compressive spectral imaging) + CACTI (compressive temporal imaging). More AI4Science Principles added as community polish completes (Bounty 7)."

---

## 6. If Director prefers Option B

If you want to deploy more than 2 but less than 1,591:

**Steps to take:**

1. **Identify the ~30 stubs you want at launch.** Open each one in `content/<agent>/principles/<sub-domain>/L1-*.json` and confirm:
   - Title makes sense
   - Domain / sub_domain correct
   - Forward model description is correct
   - Solver class is reasonable
   - Promote `"registration_tier": "stub"` → `"v2"` or `"v3"` in the JSON

2. **Generate v2/v3 supplement content** for each:
   - Walkthrough markdown (like `pwm_product/walkthroughs/cassi.md`)
   - Demo code (like `pwm_product/demos/cassi/`)
   - Test (like `pwm_product/tests/test_cassi_quality.py`)
   - Explainer (like `pwm_product/explainers/cassi.md`)

3. **Update marketing copy** to reflect the actual count (e.g., "32 v3 anchors").

Realistic time: **30-60 minutes per Principle × 30 = 15-30 hours of polish work** before deploy day.

---

## 7. If Director prefers Option D (pause + polish first)

If you want to fully achieve the "30 v3 anchors" strategic claim:

**Steps:**

1. **Pause Phase 5 mainnet deploy** by ~2-4 weeks
2. **Director (or hire 1-2 helpers) polishes 28 stubs → v3 quality**:
   - Open each at L1-503..L1-531
   - Write E/G/W/C content
   - Generate walkthrough + demo + test + explainer for each
   - Promote `"registration_tier"` → `"v3"`
3. **Re-run testnet registration** with the polished versions
4. **Then proceed with deploy day Option B/C with 30+ truly-verified Principles**

Realistic time: 2-4 weeks of focused Director work, OR ~$10-25K to hire short-term help.

---

## 8. Action items + questions for Director

### Question 1: Where are the missing v3 anchors?

The strategic docs assume 30 v3 anchors exist. Filesystem has 2.

- (A) Are the other 28 polished v3 files committed somewhere I haven't searched? (e.g., uncommitted local work, separate branch)
- (B) Were the strategic doc claims aspirational rather than factual?
- (C) Did this work get done in a different repo / location?

### Question 2: Which option do you want for deploy day?

- ☐ Option A: 2 Principles (CASSI + CACTI). Honest. Quick. Recommend.
- ☐ Option B: ~30 stubs reviewed + 2 v3 polished. Requires 15-30 hr polish work.
- ☐ Option C: 1,591 stubs + 2 v3. Misleading without caveat.
- ☐ Option D: Pause + polish 30 to v3 first (2-4 weeks delay).

### Question 3: Should I revise the strategic docs?

If Option A or D: strategic docs need to be revised to match reality (currently claim 30 v3 anchors; should claim 2 + ongoing polish).

If Option B: strategic docs can be revised to match what's actually shipping (32 anchors with 2 fully polished + 30 reviewed).

### Question 4: What about the 1,591 stubs on Base Sepolia testnet?

They're currently visible at `explorer.physicsworldmodel.org`. Visitors see "1,591 Principles" but they're auto-generated stubs. Three options:

- (A) Leave them on testnet (no action; matches Track 2 plan)
- (B) Mark them as "draft" or "incomplete" in the UI (UI work; modest)
- (C) Migrate to a "long-tail catalog" view separate from "Verified v3 anchors" (UI redesign; significant)

---

## 9. Bottom line for deploy day decision

**You can deploy mainnet today with CASSI + CACTI only (Option A).** That's the honest state.

You CANNOT deploy "30 v3 anchors" today because they don't exist as committed files.

If the "30 v3 anchors" claim is important to you, **pause the deploy by 2-4 weeks** (Option D) and complete the polish work first. Otherwise, **deploy with 2 + revise marketing to match reality** (Option A).

The decision is honesty vs. perceived launch scale. I'd choose honesty.

---

## 10. Cross-references

- `pwm-team/plan/PLAN.md` Track 2 status — claims "1,591 on Base Sepolia"
- `pwm-team/coordination/PWM_USER_ACQUISITION_STRATEGY_2026-05-22.md` §6.3 — references "30 v3 anchors"
- `pwm-team/coordination/PWM_LAUNCH_LANDING_PAGE_DRAFT_2026-05-22.md` Section 2 — describes PWM-CI-1 (only needs CASSI)
- `pwm-team/coordination/PWM_PHASED_ARCHITECTURE_DEPLOYMENT_2026-05-22.md` §5 — "30 v3 anchors define hardware embedded"
- `pwm-team/bounties/INDEX.md` Bounty 7 scope description — refs "Tier A ~30 anchors founder-authored"
- `pwm-team/numfocus/PWM_TOKEN_ECONOMY_AND_NUMFOCUS_COMPATIBILITY_2026-05-13.md` — refs 1,597
- `pwm-team/deploy/findings/REGISTRY_HANDOFF_DECISION_2026-05-18.md` — Step 5.4a `register_batch.py`
- `content/<agent>/principles/<sub-domain>/L*.json` — 1,591 stub-tier files
- `pwm_product/genesis/l1/L1-003.json` + `L1-004.json` — only v3 anchors

---

*This doc is the reality-check on genesis Principle verification status. Update if new v3 anchors are committed; update after Director's deploy decision; update if the 1,591 stubs are promoted to v2/v3 quality.*
