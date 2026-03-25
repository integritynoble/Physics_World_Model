# Workshop Proposal: Trustworthy Evaluation of Computational Imaging Algorithms

## Proposed for MICCAI 2026

### Organizers
- [Your Name], [Affiliation]
- [Co-organizer], [Affiliation]

### Workshop Title
**PWM: Physics World Model — Toward Certifiable, Reproducible Evaluation of Medical Imaging Algorithms**

### Summary (250 words)

The reproducibility crisis in computational medical imaging persists: reported
metrics often cannot be verified, forward-model assumptions are undocumented,
and algorithmic comparisons lack standardized protocols. We propose a
half-day workshop introducing the Physics World Model (PWM) — an open-source
evaluation framework that makes imaging algorithms executable, comparable,
and certifiable through a universal trust protocol.

PWM's core contribution is the trust ratchet: every algorithm evaluation
produces a RunBundle (immutable audit record with SHA-256 provenance), passes
through a universal Judge kernel (R1-R4 operational gates verifying spec
completeness, reproducibility, metric integrity, and budget compliance), and
emits a machine-readable Certificate carrying a trust tier (Draft →
Author-confirmed → Reproduced → Certified).

The workshop will:
1. **Present the PWM protocol** — CoreSpec six-tuple, OperatorGraph compilation,
   4-scenario evaluation, and the S1-S4 / R1-R4 dual-gate architecture
2. **Live demonstration** — participants run `pwm evaluate` on their laptops
   against CT, MRI, and CASSI modalities, producing real Certificates
3. **Challenge track** — "Beat the Harness" competition where participants
   attempt to produce high-scoring results that fail diagnostic gates
4. **Panel discussion** — reproducibility standards for MICCAI publications,
   with editors and reviewers

The workshop addresses MICCAI's growing emphasis on reproducibility and
algorithmic fairness. PWM's 172 modalities, 2,732 pre-cataloged algorithms,
and open trust kernel provide immediate infrastructure for standardized
evaluation.

### Relevance to MICCAI

- Directly supports MICCAI's reproducibility initiative
- Provides infrastructure for standardized algorithm comparison
- CT QC Copilot addresses clinical quality assurance
- Open-source: any group can adopt without licensing

### Format

- **Duration**: Half-day (4 hours)
- **Session 1** (90 min): Keynote + PWM protocol tutorial
- **Session 2** (60 min): Hands-on challenge (participants run PWM locally)
- **Session 3** (30 min): Challenge results and leaderboard
- **Session 4** (60 min): Panel — "What should reproducible imaging evaluation look like?"

### Expected Attendance
50-80 participants (based on interest in reproducibility + imaging benchmarks)

### Technical Requirements
- WiFi for participant laptops
- Projector for live demo
- No special hardware needed (PWM runs on CPU)

### Previous Related Workshops
- No previous PWM workshop (this would be the inaugural event)
- Related: MICCAI reproducibility challenges, Learn2Reg, BraTS

### Platform and Data
- Open-source: github.com/integritynoble/Physics_World_Model
- Live platform: pwm.platformai.org
- 12 golden reference bundles at Certified tier
- No proprietary data required

### Key References
1. [Universal Simulation paper — the PWM theoretical foundation]
2. [InverseNet — calibration and mismatch correction]
3. [CT QC Copilot — operations-flywheel QC system]
