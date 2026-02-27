# PWM Revenue Strategy

**How Physics World Models generates revenue**

---

## Table of Contents

1. [Revenue Overview](#1-revenue-overview)
2. [Revenue Stream 1: Clinical CT/CBCT QA SaaS](#2-revenue-stream-1-clinical-ctcbct-qa-saas)
3. [Revenue Stream 2: Semiconductor Inspection](#3-revenue-stream-2-semiconductor-inspection)
4. [Revenue Stream 3: Smartphone Camera SDK](#4-revenue-stream-3-smartphone-camera-sdk)
5. [Revenue Stream 4: Microscopy Deconvolution Plugin](#5-revenue-stream-4-microscopy-deconvolution-plugin)
6. [Revenue Stream 5: MRI QA / Calibration SaaS](#6-revenue-stream-5-mri-qa--calibration-saas)
7. [Revenue Stream 6: Defense & Intelligence](#7-revenue-stream-6-defense--intelligence)
8. [Revenue Stream 7: Automotive Camera Calibration](#8-revenue-stream-7-automotive-camera-calibration)
9. [Revenue Stream 8: AR/VR Camera Calibration](#9-revenue-stream-8-arvr-camera-calibration)
10. [Revenue Stream 9: Cryo-EM CTF Estimation](#10-revenue-stream-9-cryo-em-ctf-estimation)
11. [Revenue Stream 10: Camera Diagnosis SaaS (DxOMark Competitor)](#11-revenue-stream-10-camera-diagnosis-saas-dxomark-competitor)
12. [Prioritized Roadmap](#12-prioritized-roadmap)
13. [Total Revenue Projections](#13-total-revenue-projections)
14. [Market Size Summary](#14-market-size-summary)

---

## 1. Revenue Overview

PWM monetizes a single insight: **calibration mismatch (Gate 3) is the dominant bottleneck in imaging, and forward-model correction recovers more quality than upgrading the algorithm.** Every revenue stream sells a version of this: automated mismatch diagnosis + correction for a specific imaging market.

### Revenue Principle

| What PWM Sells | In Every Market |
|----------------|-----------------|
| **Diagnosis** | Identify which Gate is limiting image quality |
| **Localization** | Pinpoint which primitive node has the mismatch |
| **Correction** | Auto-calibrate the forward model parameters |
| **Monitoring** | Continuous drift detection over time |

### Revenue Summary Table

| # | Revenue Stream | Time to Revenue | Year 1–2 ARR | Year 3–5 ARR | TAM |
|---|---------------|-----------------|--------------|--------------|-----|
| 1 | Clinical CT/CBCT QA SaaS | 12–18 months | $5–10M | $30–50M | $2–4B |
| 2 | Semiconductor Inspection | 18–24 months | $3–5M | $20–50M | $15B |
| 3 | Smartphone Camera SDK | 6–12 months | $10–20M | $50–100M | $1–2B |
| 4 | Microscopy Plugin | **3–6 months** | $2–5M | $10–20M | $500M |
| 5 | MRI QA SaaS | 12–18 months | $3–5M | $10–20M | $1B |
| 6 | Defense (SAR, Hyperspectral) | 6–12 months | $2–5M | $5–15M | $500M |
| 7 | Automotive Camera Calibration | 12–18 months | $3–5M | $10–30M | $2–5B |
| 8 | AR/VR Camera Calibration | 12–24 months | $1–3M | $20–50M | $1–2B |
| 9 | Cryo-EM CTF Estimation | 12 months | $1–3M | $5–15M | $300M |
| 10 | Camera Diagnosis SaaS | **3–6 months** | $2–5M | $10–20M | $200M |
| | **Total** | | **$32–66M** | **$170–370M** | **$24–31B** |

---

## 2. Revenue Stream 1: Clinical CT/CBCT QA SaaS

### The Opportunity

Hospitals are **legally required** to perform quality assurance on CT and CBCT scanners. Annual ACR accreditation, routine monthly/quarterly QA, and post-service recalibration checks are mandatory. The current process is manual, time-consuming, and detects problems only after they've degraded image quality.

### Market Size

- ~50,000 CT scanners in the US
- ~10,000 CBCT-equipped linear accelerators in the US
- ~$2B CT quality assurance market globally
- Medical physicists spend 2–4 hours per scanner per month on QA
- Medical physicist salary: $150K–$250K/year (expensive labor)

### Product: PWM CT/CBCT QA Copilot

**What it does:**
1. **Automated ACR phantom analysis** — processes phantom scans, extracts 10 ACR metrics (CT number accuracy, uniformity, geometric accuracy, noise, resolution, slice thickness)
2. **Triad diagnosis** — for every metric failure, identifies whether it's Gate 1 (protocol), Gate 2 (dose), or Gate 3 (calibration drift)
3. **Root-cause localization** — identifies which physical parameter drifted (CoR offset, HU drift, gain variation, tube output change)
4. **Predictive drift detection** — tracks calibration parameters over time, predicts when a scanner will fail ACR accreditation 3–6 months before it happens
5. **CBCT scatter correction** — automatically estimates and corrects scatter contamination for each patient body habitus

**The paper already has:** CT validation with CoR mismatch showing 8–9 dB degradation and 100% oracle recovery. The CT QC Copilot concept is described in the supplementary material.

### Revenue Model

| Tier | Price | Features |
|------|-------|----------|
| Basic | $5,000/scanner/year | Automated ACR analysis + Triad report |
| Pro | $10,000/scanner/year | + Predictive drift detection + longitudinal trending |
| Enterprise | $15,000/scanner/year | + CBCT scatter correction + multi-site fleet dashboard |

### Revenue Projection

| Year | Scanners | ARPU | Revenue |
|------|----------|------|---------|
| Year 1 | 200 | $8K | $1.6M |
| Year 2 | 1,000 | $10K | $10M |
| Year 3 | 3,000 | $12K | $36M |
| Year 5 | 5,000 | $12K | $60M |

### Go-to-Market

1. **Jiang connection** — validate on UTSW clinical scanners. UTSW has ~20 CT/CBCT scanners.
2. **AAPM channel** — present at AAPM annual meeting, publish in Medical Physics journal
3. **Partner with service vendors** — Siemens, GE, Philips service engineers could use PWM for post-service calibration verification
4. **FDA pathway** — QA software (not diagnostic) has a lighter regulatory path (510(k) Class II or exempt)

### Competitive Advantage

| Competitor | What They Do | PWM Advantage |
|-----------|-------------|---------------|
| Sun Nuclear (Mirion) | Phantom + analysis software | PWM diagnoses *why* a metric fails (which Gate, which parameter) |
| Standard Imaging | Dosimetry + QA | PWM adds forward-model-based root cause analysis |
| Radformation | Automation of RT workflows | PWM adds physics-based calibration diagnosis |
| Manual QA | Physicist visual inspection | PWM is automated, quantitative, predictive |

---

## 3. Revenue Stream 2: Semiconductor Inspection

### The Opportunity

Semiconductor manufacturing requires sub-nanometer calibration accuracy. Every wafer fab has 100+ inspection tools (electron microscopes, metrology stations), each needing continuous calibration. A 0.1% yield improvement on a $20B fab saves $20M/year.

### Market Size

- ~$15B semiconductor inspection and metrology market (KLA, ASML, Applied Materials, Hitachi High-Tech)
- Electron ptychography is already validated in the paper (4D-STEM SrTiO₃)
- EUV lithography metrology needs better forward-model calibration

### Product: PWM Metrology Calibration Engine

**What it does:**
1. **Probe calibration** for electron ptychography (position errors, aberrations)
2. **SEM image quality diagnosis** (astigmatism, focus, contamination = Gate 3)
3. **Automated aberration correction** via OperatorGraph-based forward model
4. **Continuous tool monitoring** — detect calibration drift between scheduled maintenance

### Revenue Model

- Per-tool license: $100,000–$500,000/year
- Or per-wafer SaaS: $0.01–$0.10/wafer

### Revenue Projection

| Year | Tool Licenses | ARPU | Revenue |
|------|--------------|------|---------|
| Year 1 | 10 | $200K | $2M |
| Year 2 | 50 | $200K | $10M |
| Year 3 | 100 | $250K | $25M |
| Year 5 | 200 | $300K | $60M |

### Go-to-Market

1. **Partner with one equipment maker** (KLA, ASML, Hitachi High-Tech, or JEOL)
2. **Demonstrate yield improvement** on a pilot production line
3. **SBIR/STTR** grant from NSF or DOE for semiconductor metrology R&D
4. **Publish at SPIE Advanced Lithography + Patterning** conference

---

## 4. Revenue Stream 3: Smartphone Camera SDK

### The Opportunity

Mid-range smartphone OEMs cannot afford in-house camera R&D comparable to Apple/Google. They buy camera processing SDKs from companies like ArcSoft ($200–400M/year revenue at $0.10–$0.80/device). PWM's physics-based approach can deliver flagship-quality features on mid-range hardware.

### Market Size

- ~$800M–$1.5B camera SDK market (ArcSoft, Morpho, Almalence, Vidhance)
- ~1.26 billion smartphones shipped annually
- 89% of consumers cite camera quality as top purchase factor
- $16.8B computational photography market (2025), projected $49.9B by 2035

### Product: PWM Camera SDK

| Module | Feature | Replaces | Primitive Chain |
|--------|---------|----------|----------------|
| PWM-HDR | Multi-frame HDR with Triad ghost detection | ArcSoft HDR | `[M→C→Λ→D] × K` |
| PWM-Night | Night mode with motion-kernel-aware merge | OEM night mode | `C→M→C→D` |
| PWM-Portrait | Forward-model bokeh with depth diagnosis | ArcSoft Bokeh | `C(h(z))→Σ→D` |
| PWM-Zoom | Physics-based SR with Gate 1 boundary detection | ArcSoft SuperZoom | `S→C→D` |
| PWM-Stab | Rolling-shutter-aware stabilization | Vidhance, Morpho | `M→C→S→D` |
| PWM-Fuse | Cross-camera forward model alignment | ArcSoft multi-cam | `[M→C→M→D] × cams` |
| PWM-Color | Illuminant-aware color science | Built-in ISP color | `M→Σ→C→S→D` |

**Unique advantage over ArcSoft/Morpho:** PWM is physics-based, not black-box neural. Benefits:
- **Smaller models** — physics constrains the solution space, reducing neural network size
- **Better generalization** — works across sensor models without per-device training
- **Explainable failures** — Triad diagnosis tells you *why* an image looks bad
- **Gate 1 detection** — can formally detect when the ISP is hallucinating (anti-Samsung zoom controversy)

### Revenue Model

- Per-device royalty: $0.10–$0.50 per device (module-dependent)
- Minimum annual license: $1M per OEM

### Revenue Projection

| Year | Devices | ARPU | Revenue |
|------|---------|------|---------|
| Year 1 | 50M | $0.15 | $7.5M |
| Year 2 | 200M | $0.20 | $40M |
| Year 3 | 400M | $0.25 | $100M |

### Go-to-Market

1. **Target one mid-tier OEM** (Xiaomi, Oppo, Vivo, Realme, Motorola, Nothing)
2. **Demonstrate A/B comparison** — same hardware, ArcSoft ISP vs. PWM ISP
3. **Ship one module first** (PWM-Night or PWM-Portrait — highest user impact)
4. **Expand module by module** as trust builds

### Manufacturing Calibration Add-On

PWM can also improve the manufacturing calibration process:
- Current: 5–15 seconds per device, calibrating parameters independently
- PWM: calibrates the **entire forward model** simultaneously, focusing time on dominant Gate 3 sources
- Saves $0.05–$0.20 per device in calibration time/cost
- Reduces camera-related returns (currently 8–15% of all returns)

**Additional revenue:** $0.05–$0.20 per device for factory calibration optimization

---

## 5. Revenue Stream 4: Microscopy Deconvolution Plugin

### The Opportunity

Every confocal, widefield, and super-resolution microscope needs PSF deconvolution. Current tools (Huygens, AutoQuant) use fixed PSF models — a textbook Gate 3 mismatch. PWM's adaptive PSF estimation corrects for sample-induced aberrations.

### Market Size

- ~$8B microscopy market (Nikon, Zeiss, Leica, Olympus/Evident)
- Thousands of core imaging facilities worldwide
- Each facility serves 50–200 researchers
- Huygens (SVI) charges ~$5,000/seat for deconvolution

### Product: PWM Deconvolution Plugin

**What it does:**
1. **Adaptive PSF estimation** — estimates the actual PSF from the image data itself (blind deconvolution with physics-informed priors)
2. **Triad diagnosis** — identifies whether resolution is limited by Gate 1 (Nyquist/sampling), Gate 2 (photon budget), or Gate 3 (PSF mismatch)
3. **Forward-model correction** — corrects for spherical aberration, refractive index mismatch, and chromatic aberration
4. **Multi-channel support** — handles fluorescence crosstalk between channels

**Format:** ImageJ/Fiji plugin (open-source ecosystem) + standalone application

### Revenue Model

| Tier | Price | Features |
|------|-------|----------|
| Academic | $3,000/seat/year | Deconvolution + Triad report |
| Core Facility | $8,000/seat/year | + Multi-user license + batch processing |
| Enterprise (OEM) | $15,000/seat/year | + API integration + Nikon/Zeiss/Leica plugin |

### Revenue Projection

| Year | Seats | ARPU | Revenue |
|------|-------|------|---------|
| Year 1 | 300 | $5K | $1.5M |
| Year 2 | 800 | $6K | $4.8M |
| Year 3 | 2,000 | $7K | $14M |

### Go-to-Market

**This is the fastest path to revenue (3–6 months):**
1. Ship as Fiji/ImageJ plugin (immediate distribution to millions of users)
2. Free tier with basic deconvolution (drives adoption)
3. Paid tier for adaptive PSF estimation + Triad diagnosis
4. Partner with Nikon (NIS-Elements), Zeiss (ZEN), or Leica (LAS X) for OEM integration
5. Present at Biophysical Society, ASCB, and Photonics West conferences

### Competitive Advantage

| Competitor | Price | PWM Advantage |
|-----------|-------|---------------|
| Huygens (SVI) | ~$5K/seat | PWM estimates PSF adaptively; Huygens uses theoretical PSF |
| AutoQuant (Media Cybernetics) | ~$3K/seat | PWM provides Triad diagnosis (why resolution is limited) |
| DeconvolutionLab2 (free) | Free | PWM adds physics-informed adaptive correction |
| Richardson-Lucy (built-in) | Free | PWM handles spatially-varying PSF + aberrations |

---

## 6. Revenue Stream 5: MRI QA / Calibration SaaS

### The Opportunity

MRI calibration is complex (coil sensitivities, B0/B1 mapping, gradient calibration). The paper shows ESPIRiT degrades -9.29 dB at 24 ACS lines. PWM's cross-modality approach could outperform for multi-vendor hospital fleets.

### Market Size

- ~40,000 MRI scanners in the US, ~70,000 globally
- ~$7B MRI market
- Quantitative MRI (qMRI) demands accurate forward models
- ACR MRI accreditation is mandatory

### Product: PWM MRI Calibration Engine

**What it does:**
1. **Coil sensitivity estimation** — physics-informed alternative to ESPIRiT, robust at low ACS line counts
2. **B0/B1 inhomogeneity correction** — forward-model-based field map estimation
3. **Gradient nonlinearity correction** — geometric distortion from gradient imperfections
4. **QA automation** — automated ACR MRI phantom analysis with Triad diagnosis

### Revenue Model

- $8,000–$20,000/scanner/year

### Revenue Projection

| Year | Scanners | ARPU | Revenue |
|------|----------|------|---------|
| Year 1 | 200 | $10K | $2M |
| Year 2 | 500 | $12K | $6M |
| Year 3 | 1,500 | $14K | $21M |

---

## 7. Revenue Stream 6: Defense & Intelligence

### The Opportunity

NGA, NRO, DARPA pay premium for automated sensor calibration. SAR autofocus = Gate 3 correction (platform motion error causes phase error in SAR image).

### Market Size

- ~$500M addressable market for imaging sensor calibration in defense
- SAR, hyperspectral (CASSI variant), infrared imaging
- Export-controlled, high-margin, long contract cycles

### Product: PWM Sensor Calibration for ISR

**What it does:**
1. **SAR autofocus** — forward-model-based motion compensation
2. **Hyperspectral calibration** — spectral response drift correction (uses CASSI/CACTI expertise)
3. **Multi-sensor fusion** — cross-modal alignment (EO/IR/SAR)

### Revenue Model

- Government contracts (SBIR Phase I: $275K, Phase II: $1.15M, Phase III: production)
- Direct procurement contracts: $2–5M each

### Revenue Projection

| Year | Contracts | Revenue |
|------|-----------|---------|
| Year 1 | 2 SBIR Phase I | $550K |
| Year 2 | 2 SBIR Phase II + 1 direct | $4.3M |
| Year 3 | 3 production contracts | $10M |

### Go-to-Market

1. **Apply to SBIR/STTR** (NSF, DOD, NGA topics on imaging sensor calibration)
2. **Present at SPIE Defense + Commercial Sensing**
3. **Partner with a defense prime** (Raytheon, L3Harris, Northrop Grumman)

---

## 8. Revenue Stream 7: Automotive Camera Calibration

### The Opportunity

ADAS cameras require sub-pixel accuracy for safety-critical applications. Cameras must be recalibrated after windshield replacement, collision repair, or wheel alignment.

### Market Size

- 5–8 cameras per L2 ADAS vehicle, 12–16 for L3–L4
- Automotive camera module market: $15B (2024) → $25–30B (2030)
- Aftermarket recalibration: $100–200 per event (millions of events/year)
- Automotive camera calibration TAM: $2–5B by 2028

### Product: PWM ADAS Calibration

**What it does:**
1. **Factory end-of-line calibration** — multi-camera alignment using OperatorGraph DAG
2. **Aftermarket recalibration** — post-windshield-replacement camera alignment
3. **Continuous self-calibration** — runtime monitoring for calibration drift
4. **Triad diagnosis** — distinguishes miscalibration (Gate 3) from damage (Gate 1) from environmental degradation (Gate 2)

### Revenue Model

- Factory: $1–5 per vehicle (OEM integration)
- Aftermarket: $50–200 per calibration event (tool licensing)

### Revenue Projection

| Year | Revenue Source | Revenue |
|------|--------------|---------|
| Year 1 | 50K aftermarket events × $50 | $2.5M |
| Year 2 | 200K events + 1M vehicles × $2 | $12M |
| Year 3 | 500K events + 5M vehicles × $3 | $40M |

---

## 9. Revenue Stream 8: AR/VR Camera Calibration

### The Opportunity

AR/VR headsets have 6–12+ cameras requiring precise calibration. Apple Vision Pro's 12-camera calibration contributes significantly to its $3,499 price.

### Market Size

- AR/VR camera calibration: $200M (2025) → $1–2B (2030), CAGR 50–70%
- Apple Vision Pro, Meta Quest, Samsung XR, Sony PSVR

### Product: PWM XR Calibration Engine

**What it does:**
1. **Factory multi-camera calibration** — stereoscopic passthrough, eye tracking, hand tracking
2. **Runtime self-calibration** — continuous recalibration from natural head motion (like super-resolution from hand tremor)
3. **Comfort diagnosis** — detect when calibration drift causes nausea-inducing distortion

### Revenue Model

- $1–10 per headset (volume licensing)

### Revenue Projection

| Year | Headsets | ARPU | Revenue |
|------|----------|------|---------|
| Year 1 | 1M | $2 | $2M |
| Year 2 | 5M | $3 | $15M |
| Year 3 | 10M | $5 | $50M |

---

## 10. Revenue Stream 9: Cryo-EM CTF Estimation

### The Opportunity

CTF estimation is the rate-limiting calibration step in cryo-EM. Labs would pay to automate and improve CTF fitting — the difference between 2Å and 3Å resolution determines publishability and drug discovery success.

### Market Size

- ~2,000 cryo-EM instruments worldwide
- ~$5M per instrument
- Pharma companies running cryo-EM for drug discovery have large budgets
- ~$300M addressable market for cryo-EM software tools

### Product: PWM Cryo-EM CTF Engine

**What it does:**
1. **Physics-informed CTF estimation** — uses the full OperatorGraph forward model (`P → M → P → C → D`) for CTF fitting
2. **Triad diagnosis per particle** — identifies whether resolution is limited by ice thickness (Gate 1), beam damage (Gate 2), or defocus error (Gate 3)
3. **Adaptive defocus refinement** — refines per-particle CTF parameters during 3D reconstruction
4. **Integration with RELION / cryoSPARC** — plugin for existing workflows

### Revenue Model

| Tier | Price | Features |
|------|-------|----------|
| Academic | $10,000/lab/year | CTF estimation + Triad report |
| Pharma | $25,000/lab/year | + Per-particle refinement + API |
| Enterprise | $50,000/lab/year | + On-premise deployment + priority support |

### Revenue Projection

| Year | Labs | ARPU | Revenue |
|------|------|------|---------|
| Year 1 | 50 | $15K | $750K |
| Year 2 | 150 | $18K | $2.7M |
| Year 3 | 300 | $20K | $6M |

---

## 11. Revenue Stream 10: Camera Diagnosis SaaS (DxOMark Competitor)

### The Opportunity

DxOMark charges OEMs $100K–$500K per engagement for camera tuning consulting. There's no automated, physics-based tool that diagnoses *why* camera quality is limited.

### Market Size

- DxOMark: ~$20–50M/year revenue
- Camera QA tool market: ~$200M
- ~25 active smartphone OEMs globally
- Thousands of camera reviewers and enthusiast photographers

### Product: PWM Camera Diagnosis Platform

**What it does:**
1. User uploads image(s) + EXIF metadata
2. PWM constructs the appropriate OperatorGraph DAG
3. Triad analysis: decompose image quality into Gate 1 / Gate 2 / Gate 3 contributions
4. Per-primitive diagnosis: which node is the bottleneck
5. Quantitative "PWM Score" — like DxOMark but with physics-based root cause

### Revenue Model

| Tier | Price | Users |
|------|-------|-------|
| Free | $0 | 5 scans/month (drives traffic) |
| Pro | $500–$2,000/year | Reviewers, enthusiasts, small teams |
| Enterprise | $100K–$300K/year | OEM R&D, competitive benchmarking |

### Revenue Projection

| Year | Enterprise | Pro | Revenue |
|------|-----------|-----|---------|
| Year 1 | 10 × $150K | 1K × $1K | $2.5M |
| Year 2 | 25 × $200K | 3K × $1K | $8M |
| Year 3 | 50 × $200K | 5K × $1.5K | $17.5M |

---

## 12. Prioritized Roadmap

### Phase 1: Fastest Revenue (Months 0–6)

| Priority | Product | Time | Revenue Y1 | Why first |
|----------|---------|------|-----------|-----------|
| **1** | **Microscopy Plugin** | 3–6 months | $1.5M | Fastest to ship. No regulatory barrier. Existing open-source ecosystem (Fiji). Every biology lab is a customer. |
| **2** | **Camera Diagnosis SaaS** | 3–6 months | $2.5M | Web app, no hardware. Free tier drives viral adoption. Enterprise sales to OEMs. |
| **3** | **Defense SBIR** | 3 months (application) | $550K | Non-dilutive funding. Validates SAR/hyperspectral use case. |

**Phase 1 total: ~$4.5M revenue + non-dilutive funding**

### Phase 2: Clinical & Smartphone (Months 6–18)

| Priority | Product | Time | Revenue Y2 | Why second |
|----------|---------|------|-----------|------------|
| **4** | **Clinical CT/CBCT QA SaaS** | 12–18 months | $10M | Leverage Jiang connection. CT QC Copilot is half-built. Clear regulatory path. |
| **5** | **Smartphone Camera SDK** | 6–12 months | $7.5M | Massive market. Start with one OEM, one module (Night Mode or Portrait). |
| **6** | **MRI QA SaaS** | 12–18 months | $2M | Extension of CT QA platform to MRI. |

**Phase 2 total: ~$19.5M revenue**

### Phase 3: High-Value Enterprise (Months 12–36)

| Priority | Product | Time | Revenue Y3 | Why third |
|----------|---------|------|-----------|-----------|
| **7** | **Semiconductor Inspection** | 18–24 months | $25M | Highest per-unit value. Requires partnership. |
| **8** | **Automotive Calibration** | 12–18 months | $40M | Massive market. Safety-critical = premium pricing. |
| **9** | **AR/VR Calibration** | 12–24 months | $50M | Emerging market. Scales with headset volume. |
| **10** | **Cryo-EM CTF** | 12 months | $6M | Niche but high-value. Drug discovery budgets. |

**Phase 3 total: ~$121M revenue**

---

## 13. Total Revenue Projections

| Year | Phase | Cumulative Products | ARR |
|------|-------|-------------------|-----|
| Year 1 | Phase 1 | Microscopy + SaaS + SBIR | **$4.5M** |
| Year 2 | Phase 1+2 | + CT QA + Smartphone SDK + MRI QA | **$24M** |
| Year 3 | Phase 1+2+3 | + Semiconductor + Automotive + AR/VR + Cryo-EM | **$145M** |
| Year 5 | Full portfolio | All 10 products mature | **$300–400M** |

### Revenue by Category

| Category | Year 3 ARR | % of Total |
|----------|-----------|------------|
| Medical imaging (CT, CBCT, MRI QA) | $71M | 49% |
| Consumer cameras (Smartphone SDK, Diagnosis SaaS) | $32M | 22% |
| Industrial/scientific (Semiconductor, Microscopy, Cryo-EM) | $45M | 31% |
| Defense | $10M | 7% |
| Automotive + AR/VR | $90M | — |

---

## 14. Market Size Summary

### Camera Calibration & QA

| Segment | 2025 TAM | 2030 TAM | CAGR |
|---------|----------|----------|------|
| Smartphone camera calibration (manufacturing) | $2.5–4.0B | $3.5–5.0B | 8–10% |
| Automotive camera calibration | $1.5–2.5B | $3.0–5.0B | 18–22% |
| AR/VR camera calibration | $0.1–0.3B | $0.8–2.0B | 50–70% |
| Medical imaging QA (CT, MRI, CBCT) | $2.0–3.0B | $3.0–4.0B | 8–10% |
| Security/industrial camera calibration | $0.3–0.5B | $0.5–0.8B | 12–15% |

### Computational Photography & Imaging Software

| Segment | 2025 TAM | 2030 TAM | CAGR |
|---------|----------|----------|------|
| Camera SDKs (ArcSoft, Morpho, etc.) | $0.8–1.5B | $2.0–3.5B | 15–20% |
| Semiconductor inspection software | $3.0–5.0B | $5.0–8.0B | 12–15% |
| Microscopy deconvolution | $0.3–0.5B | $0.5–0.8B | 10–12% |
| Cryo-EM software tools | $0.2–0.3B | $0.4–0.6B | 15–18% |
| Camera QA tools (Imatest, DxOMark) | $0.1–0.3B | $0.3–0.5B | 15–18% |

### Key Competitive Players

| Company | Revenue (est.) | What They Sell | PWM Advantage |
|---------|---------------|----------------|---------------|
| ArcSoft | $200–400M/yr | Camera SDK | Physics-based vs. black-box neural |
| DxOMark | $20–50M/yr | Camera benchmarking | Root-cause diagnosis (which Gate, which primitive) |
| Huygens (SVI) | ~$30M/yr | Microscopy deconvolution | Adaptive PSF estimation |
| Sun Nuclear | ~$200M/yr | Medical QA | Predictive drift + Triad diagnosis |
| Imatest | $10–20M/yr | Image quality testing | Forward-model diagnosis on top of metrics |
| KLA | ~$10B/yr | Semiconductor inspection | PWM calibration as add-on to KLA tools |

### Grand Total Addressable Market

| Category | 2025 TAM | 2030 TAM |
|----------|----------|----------|
| Camera calibration & QA | $6.4–10.3B | $10.8–16.8B |
| Imaging software & SDKs | $4.4–7.6B | $8.2–13.4B |
| **Grand Total** | **$10.8–17.9B** | **$19.0–30.2B** |
