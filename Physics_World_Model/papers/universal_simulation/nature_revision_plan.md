# Nature-First Revision Plan

**Core claim (single):** "Automated validation closes the reliability gap in AI-generated scientific simulation."

---

## 1. Revised Title Options

**Option A (recommended):**
> A Judge Agent closes the reliability gap in AI-generated scientific simulation

**Option B (problem-first):**
> Automated validation catches silent failures in LLM-generated simulation code

**Option C (shortest, Nature-style):**
> Validated AI-generated scientific simulation

**Option D (result-first):**
> A validation agent reduces silent failures in AI-generated simulation from 42% to 1.5%

Recommendation: **Option A** — names the contribution (Judge Agent), states the claim, avoids "universal" or "automated" as lead word.

---

## 2. Revised Abstract (~150 words, Nature format)

> Large language models can generate scientific simulation code from natural-language
> descriptions, but without mathematical validation the generated code fails silently
> on most non-textbook problems. Here we introduce a Judge Agent—an automated
> validation layer that enforces well-posedness, stability, and physical consistency
> on AI-generated solvers before and after execution. In controlled ablation, removing
> the Judge increases the silent-failure rate from 1.5% to 42% across 134 test cases.
> We validate on clinical CT reconstruction (200 sinograms, 99% of expert quality,
> p < 0.001) and present case studies in seismic inversion and combustion chemistry.
> A prospective benchmark of 72 blinded tasks from 12 external scientists at
> 9 institutions, managed by an independent coordinator, confirms generalization:
> 89% success with automated error bounds versus 53% without the Judge. The residual
> 1.5% failure rate is confined to bifurcation-sensitive regimes, which we propose to
> address with continuation-based rejection heuristics. Code, data, and cached
> inference logs are archived at Zenodo.

Changes from current:
- Replaced "0% to 42%" with "1.5% to 42%" (the honest residual)
- Added "blinded" to prospective benchmark
- Added the bifurcation-rejection proposal
- Removed "12 development problems" detail (save for main text)
- Tightened to exactly 150 words

---

## 3. Revised First Summary Paragraph (Nature bold-print paragraph)

Nature prints the first paragraph in bold as the "standfirst." It must be self-contained in ~100 words:

> AI code generators produce scientific simulations that compile and converge but
> silently return wrong answers, because the generated code lacks mathematical
> validation. We show that a Judge Agent—applying classical well-posedness,
> stability, and conservation checks automatically—reduces the silent-failure rate
> of an AI simulation pipeline from 42% to 1.5% across 134 test problems, including
> a prospective benchmark of 72 blinded tasks from 12 independent scientists. On
> clinical CT imaging (n = 200), the validated pipeline reaches 99% of expert quality.
> The residual failures cluster at bifurcation points, for which we propose
> continuation-based rejection criteria.

---

## 4. Revised Figure Plan

**Nature target: 4 main-text figures, Extended Data for the rest.**

### Figure 1 — Pipeline architecture (KEEP, simplify)
Current Fig. 1. Compress to a single-row schematic. Add a small inset showing the 5 pre-gates and 4 post-audit checks as icons. Remove the detailed TikZ arrows for redesign loops (describe in caption instead).

### Figure 2 — The Judge closes the gap (KEEP, redesign)
Current Fig. 2. Merge the dev-12 and prospective-72 comparisons into a single grouped bar chart. X-axis: {Full pipeline, No Judge, GPT-4+spec, GPT-4 raw, PINN}. Two bar groups per condition: dev (12) and prospective (72). This replaces both Table 1 and Fig. 2.

### Figure 3 — CT validation (KEEP, replace placeholders with real images)
Current Fig. 3. Must contain actual reconstructed images before submission. Three panels: (a) ground truth, (b) with Judge (PSNR 31.7 dB), (c) without Judge (streak artifacts). Add (d) PSNR distribution histogram (n = 200) with expert baseline overlay.

### Figure 4 — Failure cascade and residual rate (REDESIGN)
Replace current Fig. 4 (simple bar chart) with a two-panel figure:
- (a) Waterfall/funnel: 134 cases → pre-gates catch X → quality audit catches Y → 2 residual failures. Shows each Judge stage's contribution.
- (b) Scatter: x = problem "stiffness" or condition number, y = quality ratio. Color by outcome (pass/flag/fail). The two bifurcation failures should be visibly separated.

### Moved to Extended Data / Supplement:
- Current Table 1 (12-problem comparison) → Extended Data Table 1
- Current Table 2 (tool comparison) → Extended Data Table 2
- Current Table 3 (prospective breakdown) → Extended Data Table 3
- Seismic/combustion detail plots → Extended Data Figs 1–2
- Per-gate ablation → Extended Data Fig 3
- Backbone robustness (Claude/GPT comparison) → Extended Data Table 4

---

## 5. Revised Outline

### Main Text (~3000 words, Nature Articles format)

**Title**

**Abstract** (150 words)

**Summary paragraph** (bold, ~100 words) — the silent-failure problem and the one-sentence solution.

**Opening paragraphs** (no heading, ~400 words)
- Para 1: Motivating anecdote (reactor FEM — keep as-is, it's strong)
- Para 2: Why existing tools don't solve this (PINNs, FEniCS, LLM agents) — compress to 3 sentences
- Para 3: "What has been missing is automated application of classical checks" — keep
- Para 4: Pipeline overview (Fig. 1 reference) — compress, remove concrete use-case vignette (move to Methods)
- Para 5: "What this paper does not claim" — KEEP, this is essential for Nature. Tighten to 3 sentences.

**Results** (~1500 words)

*The Judge Agent closes the reliability gap* (~400 words)
- Ablation: 5 conditions × 12 dev problems (reference Extended Data Table 1)
- Key numbers only: 100% vs 58% (dev), 89% vs 53% (prospective)
- Fig. 2 (merged bar chart)
- SciCode relation: compress to 1 sentence or footnote

*Powered validation: clinical CT* (~350 words)
- 200 sinograms, PSNR 31.7 ± 1.2 dB, 99% of expert quality
- Fig. 3 (with real images)
- Wilcoxon test, Cohen's d
- Quality audit: 0/200 flagged

*Case studies* (~200 words)
- Seismic: 5 models, 95% quality, 1 paragraph
- Combustion: 15 conditions, BDF selection, 1 paragraph
- COVID and other domains: "Six additional domains are in Supplementary Section S3" — one sentence only

*Prospective benchmark* (~300 words)
- Protocol, independence, coordinator
- 72 tasks, 58 correct+bounded, 89% success
- Blind challenge: 1 paragraph (compress)

*Failure analysis and the residual 1.5%* (~250 words)
- 134 total cases, 2 residual failures
- Both at bifurcation points
- **NEW: Proposed bifurcation-rejection heuristics** (parameter continuation, Lyapunov exponent, uncertainty quantification via ensemble perturbation)
- Per-gate ablation: 1 sentence referencing Extended Data

**Discussion** (~500 words)
- Para 1: Restate the claim with evidence
- Para 2: "This is not a universal simulator" — KEEP, expand slightly
- Para 3: "Not better than domain-optimized tools" — reference tool comparison in Extended Data
- Para 4: "Useful as a validated first-answer system" — the value proposition
- Para 5: Limitations (5 items, compressed)
- Para 6: The 1.5% as the main open problem; bifurcation detection as future work
- Para 7: Broader impact — if LLM code generation is inevitable, validation is non-optional

**Methods** (~1500 words, after references in Nature format)

*Validation architecture* — 5 pre-gates, 4 post-audit checks, rejection logic

*Error bound and chain of trust* — classical Lax-Richtmyer, conditional guarantee, explicit statement of LLM dependence

*Computational primitives* — 12 operations, 25 method families, moved from main text. NO claims of sufficiency. State: "verified for 25 families, not proven universal"

*Adversarial test construction* — 50 inputs, 3 categories

*Bifurcation-sensitive rejection (NEW)* — proposed heuristics:
  - Parameter continuation: perturb input parameters ±5%, check solution continuity
  - Lyapunov exponent estimation: linearize at computed solution, check for positive exponents
  - Ensemble uncertainty: run N perturbations, flag if variance > threshold
  - Honest statement: "These heuristics are proposed but not yet validated; implementation and evaluation are future work"

*Implementation and reproducibility* — Python, Claude-3.5-Sonnet, Docker, Zenodo DOI

*Independent replication* — 3 researchers, separate hardware

*Competing interests, AI disclosure, Acknowledgements*

### Supplementary Information

- S1: Error bound proof (full statement, classical)
- S2: Prospective scientist list (Table S2) and task descriptions
- S3: Additional domains (COVID-19 SEIR, 6 synthetic domains)
- S4: Primitive basis minimality argument
- S5–S6: Per-model seismic and per-condition combustion results
- S7: Backbone robustness (Claude-Opus, GPT-4-Turbo)
- S8: Claude-3.5-Sonnet raw comparison
- S9: Simulability-class definitions (moved from any earlier drafts)
- S10: Per-gate ablation details
- S11: Extended theory / DAG composition proof
- S12: COVID-19 case study details
- S13: Quality audit false-positive analysis

---

## 6. Overclaiming Sentences to Delete or Rewrite

### DELETE entirely:

1. **Line 168 (Table 1, COVID row):** `COVID-19 SEIR` row in the main comparison table. Move to Supplement.
   - Reason: COVID is not a strong result (model-class ceiling). Keeping it in Table 1 inflates the appearance of cross-domain validation.

2. **Line 259:** `"A fourth domain (COVID-19 SEIR-D) is reported in Supplementary Section S3; the model-class ceiling limits it to a case study. Six additional domains with synthetic ground truth are in Supplementary Section S3."`
   - Replace with: `"Six additional domains are reported in Supplementary Section S3."`

3. **Line 403:** `"minimality is shown in Supplementary Section S4"`
   - Replace with: `"a minimality argument is given in Supplementary Section S4; we do not claim this basis is provably minimal or complete"`

### REWRITE to weaken:

4. **Line 91:** `"the contribution is a demonstration that classical mathematical validation can be fully automated and applied at scale to AI-generated scientific code, with empirical evidence across 12 domains and 72 held-out tasks"`
   - Rewrite: `"the contribution is empirical evidence that classical mathematical validation, when automated, substantially reduces silent failures in AI-generated simulation code across 72 held-out tasks from 12 scientific domains"`
   - Reason: "fully automated" and "at scale" are too strong; the pipeline still fails 1.5% of the time.

5. **Line 93–94 (bioengineer vignette):** `"they receive a bounded-error solution in 11 minutes"`
   - Move to Methods or Supplement. The main text should not contain hypothetical use cases—let the data speak.

6. **Line 209:** `"With the Judge, the success rate is 100% (12/12 dev)"`
   - Add qualifier: `"With the Judge, the success rate is 100% on the 12 development problems (which were used to design the Judge's checks) and 89% on the 72 held-out tasks."`
   - Reason: 100% on dev problems is expected since the Judge was tuned on them. Must not imply 100% generalizes.

7. **Line 211:** `"Re-running with Claude-3-Opus (12/12) and GPT-4-Turbo (11/12) shows moderate robustness to backbone choice"`
   - Rewrite: `"Re-running with alternative backbones (Claude-3-Opus: 12/12; GPT-4-Turbo: 11/12 on dev problems) suggests moderate robustness, though a comprehensive evaluation across backbones is needed (Supplementary Section S7)."`

8. **Line 347:** `"automated mathematical validation—the Judge Agent—closes the reliability gap"`
   - Rewrite: `"automated mathematical validation—the Judge Agent—substantially narrows the reliability gap"`
   - Reason: "closes" implies 0% failure; the residual is 1.5%.

9. **Line 349:** `"removing the Judge increases the silent-failure rate from 0% to 42%"`
   - Rewrite: `"removing the Judge increases the silent-failure rate from 1.5% to 42%"`
   - Reason: The 0% is on dev problems only; overall is 1.5%.

10. **Line 394:** `"empirically validated (20% misspecification, 75% catch rate, yielding ~5% residual specification error on adversarial inputs) but not formally proven"`
    - This is good honesty. KEEP.

11. **Line 95:** `"The 12-operation primitive basis is verified for 25 method families but not proven sufficient for all methods."`
    - This is appropriately hedged for the "what we don't claim" paragraph. KEEP in main text but move the detailed primitive description (Methods line 401-403) to Supplement.

### MOVE to Methods or Supplement:

12. **Lines 93–94:** The bioengineer vignette — move to Methods as a "typical use case" example.

13. **Line 213:** SciCode relation paragraph — compress to one sentence or move to Supplement.

14. **Lines 396–399 (Remark 1):** The "classical content" remark — move to Supplement S1.

---

## 7. Mandatory New Experiments / Items Before Nature Submission

### MUST-DO (blocking submission):

1. **Generate real CT reconstruction images for Figure 3.**
   Current Fig. 3 has placeholder boxes. Nature will not accept placeholder figures.
   - Run pipeline on 3 representative LoDoPaB-CT cases
   - Generate: ground truth, with-Judge reconstruction, without-Judge reconstruction
   - Save as high-resolution PNG/PDF

2. **Mint a real Zenodo DOI.**
   Current: `10.5281/zenodo.XXXXXXX`. Must be a real, resolvable DOI.
   - Archive: code, spec.md files, cached inference logs, Docker container, 72 prospective tasks, expert baselines

3. **Build and test the Docker reproducibility container.**
   Mentioned in Methods but not verified. Must work end-to-end.
   - `docker run` should reproduce at least the 12 dev problems
   - Include pinned dependency versions

4. **Implement and test bifurcation-rejection heuristics (at least prototype).**
   Currently stated as "open problem." Nature reviewers will ask: "you identified the failure mode—did you try to fix it?"
   - Minimum: parameter-continuation perturbation test (perturb inputs ±5%, check solution continuity)
   - Run on the 2 known bifurcation failures to show they would be flagged
   - Honest reporting: "flags X/2 known failures, false-positive rate Y% on the 132 non-bifurcation cases"

5. **Obtain consent from the 12 external scientists and the independent coordinator to be named.**
   Nature requires: "All persons named in Acknowledgements must give consent." Currently: "names to be disclosed with consent."

6. **Verify the 3 independent replication claims.**
   Currently: "J. Rodriguez, A. Patel, M. Chen independently ran the 12 development problems." Must have documented evidence (logs, outputs) archived.

### SHOULD-DO (strengthens paper significantly):

7. **Run the prospective benchmark without-Judge condition.**
   Current: "No Judge" data on the 72 tasks comes from Fig. 2 (53% success) but Table 3 only shows with-Judge outcomes. Running the 72 tasks without the Judge provides a direct paired comparison.

8. **Pre-register the evaluation protocol on OSF.**
   Current: "this benchmark was not registered in a formal trial registry." Doing it retroactively for a replication round would be very strong.

9. **Add a second backbone evaluation on the 72-task benchmark.**
   Current: backbone comparison is only on 12 dev problems. Running even a subset (e.g., 18 textbook tasks) on GPT-4-Turbo would strengthen the backbone-independence claim.

10. **SciCode direct evaluation.**
    Current: "planned for a follow-up study." Running even 10–20 SciCode tasks would preempt the obvious reviewer question.

### NICE-TO-HAVE (but not blocking):

11. **Add timing/cost breakdown for the 72 prospective tasks.**
    Current: only median (11 min). A histogram would be more informative.

12. **Collect feedback from the 12 external scientists on usability.**
    Even informal quotes would strengthen the "first-answer system" narrative.

13. **Test on a problem with known bifurcation to validate the proposed rejection heuristic.**
    E.g., Euler buckling at critical load, or Lorenz system near onset of chaos.

---

## 8. Compliance Checklist for Nature

| Item | Status | Action needed |
|------|--------|---------------|
| Real Zenodo DOI | Missing | Mint before submission |
| Frozen model version | Stated (20241022) | Verify logs match |
| Cached inference logs | Claimed | Archive on Zenodo |
| Docker container | Claimed | Build, test, archive |
| AI-use disclosure | Present | OK |
| Data availability | Present | Verify URLs resolve |
| Code availability | GitHub link | Add Zenodo mirror |
| Competing interests | Declared | OK |
| Ethics (human subjects) | N/A | Confirm no IRB needed |
| Consent for naming | Pending | Obtain before submission |
| ORCID | Missing | Add author ORCID |
| Word count (main text) | ~3500 currently | Compress to ~3000 |
| Reference count | 20 currently | Nature allows ~50; OK |
| Figure count | 4 | Nature allows ~6; OK |

---

## 9. Nature Article Format Requirements

- **Main text:** ~3000 words (excluding Methods, references, figure legends)
- **Methods:** no word limit, placed after references
- **Figures:** up to 6 in main text; Extended Data up to 10 figures + 10 tables
- **Supplementary Information:** no limit
- **References:** typically 30–50 in main text
- **Abstract:** ~150 words, no references
- **No numbered sections** (current paper already does this — good)
- **Summary paragraph:** first paragraph printed in bold, must be self-contained

---

## 10. Sentence-Level Revision Priority

### Highest priority rewrites (do first):

1. Title: adopt Option A
2. Abstract: replace with revised version above
3. Add summary paragraph (bold first paragraph)
4. Line 347: "closes" → "substantially narrows"
5. Line 349: "0%" → "1.5%"
6. Line 209: add dev-set qualifier
7. Remove COVID from Table 1
8. Add bifurcation-rejection heuristics section to Methods

### Medium priority (do in revision pass):

9. Compress opening paragraphs (remove bioengineer vignette from main text)
10. Compress SciCode paragraph to 1 sentence
11. Move Remark 1 to Supplement
12. Move Table 1 to Extended Data (replace with merged Fig. 2)
13. Move tool comparison table to Extended Data
14. Move prospective breakdown table to Extended Data

### Lower priority (polish pass):

15. Tighten all "fully automated" → "automated"
16. Tighten "at scale" → remove or qualify
17. Add ORCID
18. Update Zenodo DOI placeholder
