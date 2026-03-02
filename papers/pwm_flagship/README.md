# PWM Flagship Paper — Nature Submission

**Title:** Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging

**Target:** Nature

**Authors:** Chengshuai Yang, Xin Yuan, David J. Brady, Steve B. Jiang

---

## File Structure

```
papers/pwm_flagship/
  main.tex                    # Main Nature manuscript (~4000 words)
  methods.tex                 # Online Methods (unlimited length)
  supplementary.tex           # Supplementary Information (47 pages, 21 notes)
  pwm_flagship.bib            # Bibliography (~50 references)
  preamble.tex                # Packages and commands
  README.md                   # This file
  figures/                    # Main figures (Fig 1-9, PDF+PNG)
  extended_data/              # Extended Data figures (ED1-ED10)
  original/                   # Original drafts (reference only)
  scripts/                    # Experiment & figure generation scripts
    run_*_4scenario.py        # Single-phantom 4-scenario validation
    run_*_multiphantom.py     # Multi-phantom validation (N=4-5, bootstrap CI)
    run_ct_512x512_scalability.py  # 512x512 CT scalability experiment
    aggregate_all_results.py  # Combine all modality results into summary JSON
    generate_all_figures.py   # Nature-quality figure generation (300 DPI)
  results/
    real_data_4scenario/      # Per-modality JSON results
    fluorescence_4scenario/   # Fluorescence experiment results
    combined/                 # Aggregated cross-modality summary
```

## Building

```bash
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

## Reproducing Multi-Phantom Experiments

All experiments run on CPU (no GPU required):

```bash
cd /path/to/Physics_World_Model
export PYTHONPATH=packages/pwm_core:$PYTHONPATH

# Run individual modality experiments
python3 papers/pwm_flagship/scripts/run_cryoem_multiphantom.py      # ~40s, N=5
python3 papers/pwm_flagship/scripts/run_fluorescence_multiphantom.py # ~40s, N=5
python3 papers/pwm_flagship/scripts/run_ultrasound_multiphantom.py   # ~100s, N=5
python3 papers/pwm_flagship/scripts/run_compholo_multiphantom.py     # ~120s, N=4
python3 papers/pwm_flagship/scripts/run_cbct_multiphantom.py         # ~15min, N=5

# 512x512 CT scalability experiment
python3 papers/pwm_flagship/scripts/run_ct_512x512_scalability.py    # ~13min

# Aggregate all results
python3 papers/pwm_flagship/scripts/aggregate_all_results.py

# Regenerate figures
python3 papers/pwm_flagship/scripts/generate_all_figures.py
```

## Key Numbers

| Metric | Value |
|--------|-------|
| Modalities compiled | 168 (across 19 categories) |
| Unique DAG patterns | 37 |
| OperatorGraph templates | 168 |
| Physical carriers | 5 families (photons, electrons, spins, acoustic, X-rays) |
| Modalities fully validated (Phase 1) | 9 (single-phantom) |
| Modalities validated (Phase 2) | 5 (multi-phantom, bootstrap CI) |
| Total validated configurations | 14 |
| Correction range | +0.2 to +48.25 dB |
| Gate 3 dominant | 14/14 configurations (100%) |
| 168-modality registry | All covered by 11 primitives |

### Phase 2 Multi-Phantom Results (5 carrier families)

| Modality | Carrier | N | Gate 3 Parameter | Max Delta (dB) | Recovery |
|----------|---------|---|------------------|----------------|----------|
| Fluorescence | Photon | 5 | PSF sigma | +8.35 +/- 3.58 | 0.53 |
| CT (offset) | X-ray | 5 | Detector offset | +6.53 +/- 1.82 | 1.000 |
| Cryo-EM | Electron | 5 | Defocus | +3.30 +/- 1.03 | 1.000 |
| Comp. Holography | Photon | 4 | Prop. distance | +1.04 +/- 0.48 | 1.10 |
| Ultrasound | Acoustic | 5 | Speed of sound | +0.20 +/- 0.10 | --- |

## Main Figures

1. **PWM Overview** — End-to-end pipeline from 168 modalities to 11 primitives
2. **OperatorGraph IR** — DAG examples + Physics Fidelity Ladder (4 tiers)
3. **Triad Law** — Decision tree + gate binding heatmap across carriers
4. **14-Configuration Correction** — Bar chart across 5 carrier families (Phase 1 + Phase 2)
5. **CASSI/CACTI Deep Dive** — 4-scenario comparison with state-of-the-art solvers
6. **Zero-Shot Generalization** — Cross-carrier transfer across 12 modalities
7. **Hardware Validation** — Real CASSI + CACTI instrument results
8. **Visual Comparison** — Reconstruction quality across scenarios
9. **Basis Growth** — Primitive count vs modality coverage (N=168, K=11)

---

## Co-Author Contributions

| Co-Author | Affiliation | Contribution | Effort |
|-----------|-------------|-------------|--------|
| **Chengshuai Yang** | NextGen PlatformAI | Conceived project, proved theorems, built PWM platform, all experiments, wrote manuscript | Primary |
| **Xin Yuan** | Westlake University | GAP-TV solver, EfficientSCI, CASSI/CACTI forward models & mismatch characterization, real-data protocols | Active |
| **David J. Brady** | University of Arizona | Validate physics-stage mapping, review system design implications, validate holography & CASSI/CACTI sections | ~3–5 hours |
| **Steve B. Jiang** | UT Southwestern | Validate clinical relevance, review clinical translation, advise on medical physics framing, optional: real CBCT data | ~3–5 hours |

---

## Invitation Letter: David J. Brady

**To:** djbrady@arizona.edu
**Subject:** Invitation to co-author Nature manuscript — your imaging system design work as a foundation

Dear Professor Brady,

I am Chengshuai Yang, founder of NextGen PlatformAI. I am writing to invite you to co-author a manuscript we are preparing for Nature:

**"Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"**

### What the paper proves

We prove two results that unify computational imaging across all carrier families:

1. **The Finite Primitive Basis Theorem** — every imaging forward model (from coded aperture cameras to cryo-EM to MRI) admits an ε-approximate representation as a typed DAG over exactly 11 canonical primitives. This basis is both sufficient and minimal. The primitive library saturates at 11 as modality coverage grows past 168 registered systems across 19 categories, with no new primitive required for the most recent 138 additions.

2. **The Triad Decomposition** — every reconstruction failure decomposes into exactly three independent root causes: information deficiency, carrier noise, and operator mismatch. Hardware validation across 12 modalities spanning all five carrier families (photons, electrons, X-rays, nuclear spins, acoustic waves) confirms that operator mismatch — not information deficiency or noise — is the dominant reconstruction bottleneck, recoverable by +0.8 to +10.7 dB through forward-model correction alone.

### Why your work is foundational to this paper

Your contributions are already deeply embedded in this framework:

- **Optical Imaging and Spectroscopy (Wiley, 2009):** Your textbook formalizes the source-medium-sensor decomposition that we extend into the six physics-stage families underlying the 11 primitives. We cite this as the classical foundation for our physics-stage mapping.

- **Multiscale gigapixel photography (Nature, 2012):** Your AWARE camera with dozens of jointly calibrated sub-apertures is a canonical example of why tolerance-aware co-design of hardware and reconstruction is essential. We cite this in our Discussion on imaging system design.

- **Compressive holography (Optics Express, 2009):** Your compressive holographic sensing framework directly informs our holography validation, where we analyse Gate 3 sensitivity to propagation distance error.

- **CASSI and CACTI instruments:** These instruments — which you co-invented — are two of our primary validation platforms. Our paper includes hardware validation on real CASSI (TSA scenes) and CACTI data confirming Triad predictions on physical measurements.

### What your contribution would involve (~3-5 hours)

Intellectual guidance and validation — no new experiments or coding required:

1. **Validate the physics-stage mapping** — confirm that the six physics-stage families and their primitive assignments are physically sound from a system-design perspective.

2. **Review the imaging system design implications** — we have a Discussion paragraph on how the Finite Primitive Basis implies that imaging system design is a combinatorial optimization over 11 typed primitives; your perspective on multi-scale and compressive architectures would strengthen this.

3. **Provide feedback on the holography and CASSI/CACTI sections** — since these instruments originate from your lab, your validation of our experimental protocols and interpretation adds significant credibility.

### What is PWM

PWM (Physics World Model) is an open-source platform for computational imaging autonomy: https://github.com/integritynoble/Physics_World_Model

It turns any imaging system into a self-diagnosing, self-correcting pipeline:

- Input: natural-language prompt, structured spec, or measured data + imperfect operator
- Output: OperatorGraph DAG -> Triad diagnosis -> autonomous correction -> reconstructed image + audit trail

The platform covers 168 modalities across 19 categories, with a built-in adversarial evaluation harness (LIP-Arena). It is public, extensible, and designed for reproducibility.

### Author list

Chengshuai Yang, Xin Yuan, David J. Brady, Steve B. Jiang

### Timeline

We aim to submit within 4-6 weeks. Your review of the relevant sections could be completed in a few hours at your convenience. I have attached the current manuscript PDF and would be delighted to discuss any questions — a brief Zoom call to walk through the relevant sections is also very welcome.

Best regards,
Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

---

## Invitation Letter: Steve B. Jiang

**To:** Steve.Jiang@utsouthwestern.edu
**Subject:** Invitation to co-author Nature manuscript — connecting forward-model mismatch to clinical radiation therapy QA

Dear Professor Jiang,

I am Chengshuai Yang, founder of NextGen PlatformAI. I am writing to invite you to co-author a manuscript we are preparing for Nature:

**"Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging"**

### What the paper proves

We prove two results that unify computational imaging across all carrier families:

1. **The Finite Primitive Basis Theorem** — every imaging forward model admits an ε-approximate representation as a typed DAG over exactly 11 canonical primitives. This basis is both sufficient and minimal, covering 168 registered modalities across 19 categories.

2. **The Triad Decomposition** — every reconstruction failure decomposes into exactly three independent root causes: information deficiency, carrier noise, and operator mismatch. Hardware validation across 12 modalities spanning all five carrier families confirms that operator mismatch is the dominant reconstruction bottleneck, recoverable by +0.8 to +10.7 dB through forward-model correction alone.

### Why your work is directly relevant

Your research on CBCT image quality in radiation therapy addresses the exact clinical problem our theoretical framework explains:

- **Monte Carlo scatter correction (Xu et al., PMB 2015):** Your work on physics-based CBCT scatter correction is a textbook example of what we call Gate 3 (operator mismatch) — the assumed scatter-free forward model diverges from the scatter-contaminated measurement. Our Triad framework formalizes this as a specific type of forward-model mismatch and provides a universal correction pipeline.

- **Deep RL for CT parameter tuning (Shen et al., IEEE TMI 2018):** Your reinforcement-learning approach to automated regularization parameter tuning is directly analogous to our autonomous calibration pipeline — both address the problem of optimizing reconstruction operator parameters, which is itself a form of Gate 3 correction.

- **CycleGAN CBCT-to-CT (Liang et al., PMB 2019):** Your unpaired image-domain correction demonstrates how learned mappings can compensate for aggregate forward-model mismatch without explicit physics decomposition — a complementary approach to our model-based Gate 3 correction.

We cite all three papers in our Discussion, connecting them to the Triad framework, and frame CBCT results against AAPM TG-142 and TG-66 clinical QA standards.

### Clinical validation already in the paper

- CBCT detector offset: 2.5-3.9 dB degradation at 5-20 pixel offset with 100% oracle recovery (128x128 and 512x512)
- CT real sinograms: 8-9 dB loss on public walnut micro-CT and Helsinki Tomography Challenge datasets
- MRI multi-coil: 1.75-7.14 dB degradation under clinically realistic sensitivity mismatch
- Clinical QA framing: Results connected to AAPM TG-142/TG-66 mechanical tolerance specifications

### What your contribution would involve (~3-5 hours)

Clinical validation and interpretation — no new experiments or coding required:

1. **Validate clinical relevance** — confirm that the CBCT mismatch magnitudes we test (5-20 pixel detector offset) are clinically realistic and that our results have practical implications for radiation therapy QA.

2. **Review the clinical translation paragraph** — we discuss how automated Gate 3 monitoring could complement existing QA protocols (TG-142, TG-66); your perspective as a leading medical physicist would strengthen this argument.

3. **Advise on clinical framing** — help ensure the paper speaks correctly to the radiation oncology and medical physics communities.

4. **Optional: provide real CBCT data** — if you have access to CBCT datasets with known geometric calibration states, we could add a real-data validation that would significantly strengthen the clinical impact.

### What is PWM

PWM (Physics World Model) is an open-source platform for computational imaging autonomy: https://github.com/integritynoble/Physics_World_Model

The platform covers 168 modalities across 19 categories, includes a clinical CT QC Copilot module, and is designed for audit-grade reproducibility — relevant to clinical deployment standards.

### Author list

Chengshuai Yang, Xin Yuan, David J. Brady, Steve B. Jiang

### Timeline

We aim to submit within 4-6 weeks. Your review of the clinical sections could be completed in a few hours at your convenience. I have attached the current manuscript PDF and would be happy to walk through the relevant sections on a brief Zoom call.

Best regards,
Chengshuai Yang
NextGen PlatformAI C Corp
integrityyang@gmail.com

---

## Tips for Sending the Invitations

1. **Attach the PDF** — send the compiled main.pdf, don't just describe the paper
2. **Their papers are already cited** — they can see their names in the bibliography
3. **CC Xin Yuan on Brady's email** — Yuan and Brady have co-authored extensively (CACTI, rank minimization, SCI survey), which adds credibility
4. **Offer a Zoom call** — walking through the 2-3 relevant sections makes it easy for them to say yes
5. **Follow up** — if no response in 1 week, send a brief follow-up
6. **What to point them to:**
   - For Brady: Section 2 (physics-stage mapping), Discussion (system design paragraph), compressive holography paragraph, CASSI/CACTI hardware validation
   - For Jiang: Discussion (clinical translation paragraph, CBCT prediction), CT detector offset validation, MRI multi-coil validation
