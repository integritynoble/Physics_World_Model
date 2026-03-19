# Co-author Invitation Email Template

**Paper:** "Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"
**Target journal:** Nature (Article)
**Authors:** Chengshuai Yang (NextGen PlatformAI C Corp), Xin Yuan (Westlake University)

---

## Subject Line

```
Invitation to collaborate: Nature paper on universal structure of computational imaging — [MODALITY] validation
```

---

## Email Body

Dear [TITLE] [LAST_NAME],

I am writing to invite you to collaborate on a Nature submission that establishes the universal structure of computational imaging forward models and identifies operator mismatch as the dominant reconstruction bottleneck across modalities.

**Paper summary.** We prove two theorems: (1) the Finite Primitive Basis Theorem, showing that every imaging forward model admits an approximate DAG representation over exactly 11 physically typed primitives --- sufficient and minimal; and (2) the Triad Decomposition, proving that every reconstruction failure decomposes into three root causes (information deficiency, carrier noise, operator mismatch), with a formal condition under which mismatch dominates. We validate both results across 7 modalities spanning 4 carrier families (optical photons, X-ray photons, electrons, nuclear spins), with hardware validation on 5 real instruments. Forward-model correction recovers +0.8 to +10.7 dB of PSNR, with Cohen's d > 2.0 for every modality.

**Why you.** Your expertise in [EXPERTISE_AREA] is directly relevant to [CONTRIBUTION_AREA]. Specifically, we would like to invite you to [SPECIFIC_TASK_DESCRIPTION]. This contribution would [VALUE_TO_PAPER].

**What is involved.** The estimated effort is [EFFORT_ESTIMATE]. We handle all PWM pipeline analysis, data processing, and manuscript preparation. Your contribution would be limited to [SCOPE_BOUNDARY]. A detailed contribution package is attached describing the specific protocol, data format, expected outcomes, and ICMJE authorship criteria.

**Timeline.** We are targeting submission in Q2 2026. We would need your contribution within [TIMELINE] of agreement. The full timeline is:

| Milestone | Target date |
|-----------|------------|
| Agreement to collaborate | [DATE] |
| Contribution completed | 2--4 weeks after agreement |
| Draft shared for review | 1 week after contribution received |
| Final manuscript approval | 1 week after draft review |
| Submission | Q2 2026 |

The complete manuscript draft, supplementary information, and open-source codebase (https://github.com/integritynoble/Physics_World_Model) are available for your review upon expression of interest.

I would welcome the opportunity to discuss this further at your convenience. Please feel free to suggest a time for a brief call, or I am happy to provide additional details by email.

Best regards,

Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

---

## Contribution Areas

| Area | Description | Typical effort | Example collaborators |
|------|------------|----------------|----------------------|
| Hardware validation (CASSI/CACTI) | Physical mask displacement experiment on coded aperture instruments | 1--2 days lab time | Instrument PIs with DD-CASSI or CACTI systems |
| Hardware validation (CT) | CT phantom scan with controlled center-of-rotation offset | 1 day + phantom access | Clinical medical physicists, micro-CT researchers |
| Hardware validation (MRI) | Multi-coil brain scan with controlled coil repositioning | 2--3 days | MRI physicists with research protocol access |
| Hardware validation (ptychography) | 4D-STEM scan with controlled stage drift or probe jitter | 1--2 days beamline time | Electron microscopists with 4D-STEM capability |
| Hardware validation (SIM/OCT) | Structured illumination or OCT scan testing falsifiable predictions | 1--2 days | Optical microscopists |
| Algorithm review | Validate reconstruction algorithm parameters and experimental protocol | 2--3 days desk work | Solver developers (GAP-TV, PnP, unrolled networks) |
| Clinical translation | Validate CT QC Copilot or MRI correction on clinical phantom data | 3--5 days | Clinical medical physicists |
| Theoretical strengthening | Tighter error bounds, algebraic closure, or category-theoretic structure | 1--2 weeks | Inverse problems theorists, applied mathematicians |
| Independent replication | Run the open-source pipeline on independently acquired data | 3--5 days | Any computational imaging group |

---

## ICMJE Authorship Criteria Checklist

All co-authors must satisfy all four ICMJE criteria. Check each item that applies to the proposed contribution.

### Criterion 1: Substantial contribution to conception/design OR acquisition/analysis/interpretation of data

- [ ] Designed or executed a hardware validation experiment
- [ ] Acquired real instrument data under the controlled mismatch protocol
- [ ] Provided independent analysis or interpretation of results
- [ ] Contributed to the theoretical framework (proofs, bounds, formalism)
- [ ] Developed or validated a reconstruction algorithm used in the study

### Criterion 2: Drafted the article OR revised it critically for important intellectual content

- [ ] Drafted one or more sections of the manuscript
- [ ] Provided critical scientific review and substantive revision suggestions
- [ ] Reviewed and validated modality-specific experimental descriptions

### Criterion 3: Final approval of the version to be published

- [ ] Reviewed and approved the final manuscript before submission

### Criterion 4: Agreement to be accountable for all aspects of the work

- [ ] Agreed to ensure that questions related to accuracy or integrity of any part of the work are appropriately investigated and resolved

---

## Notes for the Corresponding Author

- Send the contribution package (contribution_[name].md) as an attachment with the invitation email.
- Customize all [PLACEHOLDER] fields before sending.
- For hardware experimentalists, emphasize that no software development is needed --- the PWM pipeline processes their data without modification.
- For theorists, emphasize the companion SIAM paper opportunity (Finite Primitive Theorem).
- Track responses and contribution status in the project management system.
- All collaborators receive the full manuscript draft upon agreement.
