# Nature Publication: Weaknesses & Suggestions

## CRITICAL ISSUES (likely rejection triggers)

### 1. No Truly Controlled Hardware Experiment

The real-data experiments apply *software-simulated* mask perturbations to existing measurements -- the mask is never physically displaced and re-acquired. The paper even describes the proper protocol (Methods, lines 576--591) but doesn't execute it. Nature reviewers will see this as the #1 weakness: the central claim ("mismatch dominates") rests on simulated mismatch even for "hardware validation."

**Suggestion:** Either (a) physically displace the CASSI/CACTI mask by known amounts (micrometer stage), re-acquire, and validate, or (b) substantially soften the hardware validation claims and relabel the section (e.g., "Semi-synthetic validation on real measurements"). A multi-unit comparison (2+ cameras of same design) would be powerful.

---

### 2. CASSI Real Data Contradicts the Central Narrative

The CASSI real-data results show MST-L residual ratio of 0.9x (mismatch *improves* the fit), and GAP-TV only 1.8x. The paper correctly attributes this to pre-existing manufacturing errors, but this means the headline claim -- "sub-pixel mismatch erases a decade of solver progress" -- is only true in simulation. On real hardware, mismatch is absorbed by existing imperfections. Nature reviewers will see this as undermining the core thesis.

**Suggestion:** Reframe the narrative: the *cumulative* burden of many small uncontrolled errors (manufacturing + assembly + drift) is the real problem, not a single sub-pixel shift. The simulation results show *sensitivity*, while the real results show *baseline calibration quality matters*. This is actually a more nuanced and publishable finding.

---

### 3. No Comparison with Existing Calibration Methods

The paper never benchmarks its beam-search/gradient correction against established calibration approaches (ESPIRiT for MRI coils, blind ptychographic methods, auto-focus for CT CoR, etc.). Nature requires demonstrating improvement *over the state of the art*, not just improvement over doing nothing.

**Suggestion:** Add a comparison table: for each modality, show the standard calibration approach and its recovery ratio vs. PWM's approach. Even if PWM doesn't beat specialized methods, showing it achieves comparable results *without modality-specific tuning* is the real contribution.

---

## MAJOR ISSUES (will draw strong reviewer criticism)

### 4. Self-Referential Validation (InverseNet)

The primary experimental validation comes from "yang2026inversenet" -- a technical report from the same first author at the same institution. This is circular. Nature reviewers expect independent benchmarks.

**Suggestion:** Either (a) submit InverseNet to a peer-reviewed venue first (ECCV deadline may work), or (b) reframe InverseNet results as part of this paper's own experimental methodology rather than citing it as external validation. Option (b) means expanding the Methods/Supplementary to be self-contained.

---

### 5. Recovery Ratios Are Moderate for the Flagship Modality

CASSI, the most deeply validated modality, shows rho = 22% (GAP-TV) to 46% (MST-L). The paper frames correction as transformative, but recovering less than half the degradation undercuts the argument. The strong results (CACTI 93%, SPC 89%) use simpler modalities.

**Suggestion:** (a) Be more honest in framing -- acknowledge that multi-parameter mismatch is only partially correctable and this is itself a finding. (b) Show that even partial correction (46% for MST-L = +6.50 dB) exceeds recent solver improvements. (c) Consider running CASSI with a simpler mismatch (shift-only) to show higher recovery, then present the 5-parameter case as the harder problem.

---

### 6. Overstated Scope: "26 Validated Modalities"

Only 8 have any PSNR numbers, only 7 have correction results, and only 2 have real data. The other 18 have only adjoint checks. Calling all 26 "validated" is misleading.

**Suggestion:** Use clear, honest tier labels in Table S3: "Full validation (7)", "Scenario I baseline (1)", "Template validated (18)". Replace "26 validated modalities" in the abstract and text with "26 modality templates, 7 with full end-to-end correction validation."

---

### 7. MRI +48.25 dB Result Appears Unrealistic

A 48.25 dB correction gain from a 5% coil sensitivity mismatch is extraordinary. This likely reflects a pathological setup (e.g., single-coil sensitivity completely nulling certain regions). Nature reviewers will question whether this is representative.

**Suggestion:** (a) Use a more realistic MRI mismatch (e.g., partial coil sensitivity error on multi-coil data) and report a more modest but credible gain. (b) If you keep the extreme result, add a realistic-mismatch comparison alongside it to show the typical case.

---

### 8. CT QC Copilot Feels Bolted On

The clinical CT QC section (main Discussion + Supp Note 7) uses simulated scanner fleets with no real clinical data. It's essentially a separate paper crammed into this one. Nature will see it as diluting focus.

**Suggestion:** Either (a) remove it entirely and save it for a separate clinical paper, or (b) obtain real clinical phantom data from even one scanner to validate the ACR metrics, making it a genuine translational contribution rather than a simulation exercise.

---

## MODERATE ISSUES (will be noted by reviewers)

### 9. Limited Theoretical Novelty

The three gates (information loss, noise, model mismatch) are well-known in the inverse problems community. The Triad is a useful *formalization* but not a theoretical advance. The OperatorGraph DAG is an engineering contribution, not a scientific one.

**Suggestion:** Strengthen the theoretical section: (a) prove a formal decomposition theorem showing the three gates are necessary and sufficient (not just useful categories), (b) derive bounds showing when Gate 3 must dominate (e.g., for well-designed instruments), or (c) reframe the contribution as an *empirical law* -- showing across 7 modalities that Gate 3 consistently dominates is itself a significant finding, like a scaling law.

---

### 10. Only 2 Authors

Nature papers spanning 7 modalities + clinical CT + hardware validation typically involve larger teams. Two authors from a single company + one university raises concerns about validation breadth.

**Suggestion:** Consider adding collaborators who contributed hardware access, specific domain expertise, or independent validation. Even a clinical physicist co-author for the CT QC section would help.

---

### 11. Missing Visual Reconstruction Comparisons

The figure captions describe comparison figures, but compelling visual evidence of mismatch artifacts and correction recovery is essential for Nature. Reviewers need to *see* the failure mode.

**Suggestion:** Add a prominent figure showing: (a) ideal reconstruction, (b) mismatched reconstruction with visible artifacts, (c) corrected reconstruction, for at least CASSI, CACTI, and one non-photon modality. Nature readers respond to visual evidence.

---

## TEXT INCONSISTENCIES TO FIX

### 12. Methods Section Still Says "MIT License"

`methods.tex` line 672 says "open-source software under the MIT license" -- contradicts the fix already made to `main.tex` (PWM Noncommercial Share-Alike License v1.0).

---

### 13. Methods SPC Description Outdated

`methods.tex` lines 538--539 describe mismatch as "multiplicative gain bias" and solver as "ADMM-TV" -- should be "exponential gain drift" and "FISTA-TV" per updated main text.

---

### 14. Table S5 CACTI Delta-PSNR Stale

`supplementary.tex` Table S5 (Computational Cost, line 646) shows CACTI delta-PSNR as +22.94 -- should be +10.21 per updated Table S1.

---

### 15. Broader Benchmark Paragraph SPC Number Stale

`main.tex` line 167: "from 23.35 dB (SPC)" should be 28.06 dB per updated Table S3.

---

### 16. Table S5 SPC Delta-PSNR Stale

`supplementary.tex` Table S5 (line 649) shows SPC delta-PSNR as +12.21 -- should be +7.71 to match updated Table S1.
