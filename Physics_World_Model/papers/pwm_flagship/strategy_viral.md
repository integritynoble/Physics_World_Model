# PWM Virality Strategy

**How to make Physics World Models famous**

---

## Table of Contents

1. [The Core Viral Insight](#1-the-core-viral-insight)
2. [Viral Modalities — Ranked by Audience Size](#2-viral-modalities--ranked-by-audience-size)
3. [Smartphone Computational Photography — The Biggest Opportunity](#3-smartphone-computational-photography--the-biggest-opportunity)
4. [Viral Content Pieces](#4-viral-content-pieces)
5. [Media Strategy](#5-media-strategy)
6. [Timeline](#6-timeline)

---

## 1. The Core Viral Insight

PWM's viral potential comes from one counter-intuitive finding:

> **Your iPhone's Portrait Mode and your hospital's MRI scanner use the same 11 physics primitives — and both fail for exactly the same 3 reasons.**

This is surprising, cross-disciplinary, and immediately understandable. The unification of a $500 consumer device with a $3M clinical scanner under the same mathematical framework is inherently newsworthy.

### Why This Goes Viral

1. **Universality is inherently shareable.** "One framework for all cameras" is a clean narrative.
2. **The number 11 is memorable.** "11 primitives" is a concrete, finite, surprising result.
3. **It's contrarian.** The community believes better algorithms are the answer; PWM says better calibration is the answer.
4. **It touches everyone.** Smartphones, medical scans, telescopes — everyone has a personal connection to at least one imaging modality.
5. **It has visual proof.** Before/after correction images are immediately compelling.

---

## 2. Viral Modalities — Ranked by Audience Size

| Rank | Modality | Audience | Headline Appeal | Effort | Viral Score |
|------|----------|----------|----------------|--------|------------|
| **1** | **Smartphone photography** | 4B users | "Why your phone camera fails" | Medium | **10/10** |
| **2** | **Medical CT/MRI** | 100K+ radiologists + patients | "Your scanner is solving the wrong problem" | Already done | **9/10** |
| **3** | **Cryo-EM** | 50K structural biologists + pharma | "The same physics limits drug discovery" | Medium | **9/10** |
| **4** | **JWST** | Astronomy + general public | "JWST uses the same 11 primitives" | Medium | **8/10** |
| **5** | **Autonomous driving** | AV industry + investors | "LiDAR mismatch = your MRI mismatch" | Medium-high | **8/10** |
| **6** | **Brain MRI (fMRI)** | 100K neuroscientists | "Every brain scan has Gate 3 error" | Low | **7/10** |
| **7** | **Samsung 100x Zoom** | Tech press + consumers | "Samsung's zoom is lying — here's the physics" | Low | **7/10** |
| **8** | **LIGO** | Physics community + public | "Gravitational wave detection as Gate 3" | High | **6/10** |

---

## 3. Smartphone Computational Photography — The Biggest Opportunity

### 8 Features, Same 11 Primitives

Every major smartphone camera feature is an inverse problem that decomposes into PWM's primitives:

| Feature | OperatorGraph DAG | Primitives Used | Dominant Gate 3 |
|---------|-------------------|-----------------|-----------------|
| **HDR fusion** | `[M→C→Λ→D] × K frames` | M, C, Λ, D | Scene motion between exposures |
| **Night Mode** | `C→M→C→D per frame` | M, C, D | Motion blur kernel estimation |
| **Portrait Mode** | `C(h(z))→Σ→D` | C, Σ, D | Depth map error at boundaries |
| **Computational Zoom** | `S→C→D per frame` | S, C, D | Sub-pixel registration |
| **Video Stabilization** | `M→C→S→D` | M, C, S, D | Rolling shutter + parallax |
| **Multi-Camera Fusion** | `[M→C→M→D] × cameras` | M, C, D | Inter-camera parallax + color |
| **RAW / Color Science** | `M→Σ→C→S→D` | M, Σ, C, S, D | White balance (illuminant) |
| **Panorama** | `[M→C→M→D] × K frames` | M, C, D | Translation parallax |

### Key Findings (Viral-Worthy)

1. **All 8 features use only 6 of 11 primitives** ({M, C, S, D, Σ, Λ}). Smartphones never need wave propagation, Radon projection, Fourier encoding, dispersion, or scattering — they sit in the geometric-optics incoherent subset.

2. **Gate 3 dominates in 6 of 8 features** under normal conditions. The same finding as MRI, CT, CASSI.

3. **The dominant Gate 3 parameter is geometric in 6 of 8 cases** (registration, depth, parallax) — explaining why the industry has invested in hardware feature matching, LiDAR, and neural depth estimation.

4. **Samsung's 100x zoom crosses the Gate 1 boundary** — beyond ~5x, the system lacks sufficient information and the ISP starts hallucinating detail. PWM can formally detect this transition.

### The "Periodic Table of Cameras"

A single shareable visual showing every camera system plotted by which primitives it uses:

```
                    The Periodic Table of Cameras

         P   M   Π   F   C   Σ   D   S   W   R   Λ
iPhone   .   *   .   .   *   *   *   *   .   .   *   (6/11)
MRI      .   *   .   *   .   .   *   *   .   .   .   (4/11)
CT       .   .   *   .   .   .   *   .   .   .   *   (3/11)
CASSI    .   *   .   .   .   *   *   .   *   .   .   (4/11)
Cryo-EM  *   *   .   .   *   .   *   .   .   .   .   (4/11)
JWST     *   *   .   .   *   .   *   .   .   .   .   (4/11)
LiDAR    *   .   .   .   .   .   *   *   .   .   .   (3/11)
US       .   .   .   .   *   .   *   *   .   .   .   (3/11)
Holo     *   *   .   .   .   .   *   .   .   .   .   (3/11)
```

---

## 4. Viral Content Pieces

### Piece 1: "The Periodic Table of Cameras" (Visual / Infographic)

**Format:** High-res infographic, Twitter/X thread, Nature News & Views figure
**Core message:** All cameras use the same 11 building blocks; the specific combination determines the modality
**Target:** 100K+ impressions on Twitter/X, picked up by tech press
**Effort:** 1–2 days design

### Piece 2: "Why Your Night Photos Look Blurry (And Your Radiologist's MRI Looks Fine)"

**Format:** Blog post (2,000 words) + Twitter thread
**Core message:** Both Night Mode and MRI face Gate 3 (operator mismatch), but MRI has better calibration infrastructure. The smartphone industry spends seconds per device on calibration; radiology spends hours per scanner. The difference between a great and mediocre phone camera is primarily a calibration gap, not an algorithm gap.
**Angle:** Contrarian — challenges the assumption that AI/algorithms are the bottleneck
**Target:** The Verge, Ars Technica, Hacker News front page
**Effort:** 1 day writing

### Piece 3: "Samsung's 100x Zoom Is Lying to You — Here's the Physics"

**Format:** Blog post + YouTube explainer
**Core message:** At 100x, Samsung's Space Zoom crosses the Gate 1 boundary — there isn't enough information in the measurements for real super-resolution. The ISP switches from physics-based inversion to AI hallucination (generating plausible but fabricated detail). PWM can formally detect this transition: the measurement residual diverges from what the forward model predicts.
**Angle:** Builds on existing Samsung zoom controversy with rigorous physics explanation
**Target:** PetaPixel, DPReview, tech YouTubers (MKBHD, Marques Brownlee)
**Effort:** 2–3 days

### Piece 4: Interactive Web Demo — "Diagnose Your Phone Camera"

**Format:** Web application at `pwm.platformai.org/diagnose`
**How it works:**
1. User uploads a photo
2. PWM analyzes the image metadata (EXIF: camera model, exposure, focal length, ISO)
3. PWM constructs the appropriate OperatorGraph DAG for that camera/feature
4. PWM diagnoses which Gate is the dominant limitation:
   - Gate 1: "You need more light / a longer exposure / a better lens"
   - Gate 2: "Sensor noise is your bottleneck — use a larger sensor or brighter conditions"
   - Gate 3: "Your camera's calibration is the bottleneck — here's what calibration correction could improve"
5. Shows before/after simulation of Gate 3 correction

**Target:** 50K users in first month, viral sharing on photography forums
**Effort:** 2–4 weeks development

### Piece 5: "11 Primitives in 11 Minutes" (YouTube Video)

**Format:** Animated explainer video (11 minutes)
**Structure:**
- Minute 1: Hook — "Every camera ever made uses the same 11 building blocks"
- Minutes 2–5: The 11 primitives with visual animations
- Minutes 6–8: The 3 Gates with real failure examples
- Minutes 9–10: Smartphone examples (Night Mode, Portrait, Zoom)
- Minute 11: "What this means for AI, medicine, and your next phone"

**Target:** 500K views (physics/tech YouTube audience)
**Effort:** 2–3 weeks production

### Piece 6: "One Framework, Every Camera" (Nature News & Views Companion)

**Format:** 1,500-word accessible summary for Nature News & Views or Nature Briefing
**Core message:** Computational imaging has a universal grammar — and understanding it reveals that calibration, not AI, is the underinvested lever
**Target:** Nature's editorial team (pitch alongside paper submission)
**Effort:** 1 day writing

### Piece 7: Twitter/X Thread — "The 11 Primitives"

**Format:** 15-tweet thread with visuals
**Structure:**
1. Hook: "We proved that EVERY imaging system — from your iPhone to a $5M cryo-EM — uses the same 11 physics primitives."
2. Tweets 2–6: One primitive per tweet with visual example
3. Tweets 7–9: The 3 Gates with before/after images
4. Tweet 10: "Here's the twist: Gate 3 (calibration mismatch) dominates in 100% of cases"
5. Tweets 11–13: Smartphone examples
6. Tweet 14: "This means better calibration > better AI for image quality"
7. Tweet 15: Link to paper + code

**Target:** 200K impressions, 5K likes, picked up by AI/science accounts
**Effort:** 2 hours

---

## 5. Media Strategy

### Tier 1 — Science Press (Coincides with Paper Publication)

| Outlet | Angle | Contact Method |
|--------|-------|---------------|
| **Nature News & Views** | Companion piece to the paper | Pitch to Nature editors with submission |
| **Science Magazine News** | "Universal structure discovered in imaging physics" | Press release + direct pitch |
| **Physics Today** | "11 primitives unify computational imaging" | Author-submitted news piece |
| **Optica / SPIE News** | "Finite Primitive Basis Theorem for imaging" | Society press channels |

### Tier 2 — Tech Press (1–2 Weeks After Publication)

| Outlet | Angle | Format |
|--------|-------|--------|
| **The Verge** | "The physics behind your phone camera" | Feature article |
| **Ars Technica** | "Why calibration beats AI for image quality" | In-depth technical |
| **Wired** | "11 building blocks describe every camera" | Magazine feature |
| **IEEE Spectrum** | "Universal diagnostic for imaging failures" | Technical feature |
| **Hacker News** | Paper link + blog post | Community submission |

### Tier 3 — Photography / Camera Press (2–4 Weeks After)

| Outlet | Angle | Format |
|--------|-------|--------|
| **PetaPixel** | "Why calibration matters more than megapixels" | Guest article |
| **DPReview** | "A new framework for camera quality" | Technical deep-dive |
| **DxOMark Blog** | "Physics-based camera diagnosis" | Partnership pitch |
| **Imaging Resource** | "The science of why some cameras fail" | Explainer |

### Tier 4 — YouTube / Social (Ongoing)

| Creator | Audience | Pitch |
|---------|----------|-------|
| **3Blue1Brown** | Math/science (5M+ subscribers) | "The math behind every camera" |
| **Veritasium** | Science (14M+ subscribers) | "Why your phone camera lies to you" |
| **MKBHD** | Tech (18M+ subscribers) | "The physics of smartphone zoom" |
| **Linus Tech Tips** | Tech enthusiasts (16M+ subscribers) | "Why calibration > megapixels" |
| **Two Minute Papers** | AI/research (1.5M+ subscribers) | "11 primitives for all of imaging" |
| **Steve Mould** | Science education (4M+ subscribers) | "The 3 reasons every camera fails" |

### Tier 5 — Academic Social Media (Continuous)

| Platform | Strategy |
|----------|----------|
| **Twitter/X** | Thread on publication day; weekly "Primitive of the Week" series |
| **LinkedIn** | Post targeting medical physics, semiconductor, microscopy professionals |
| **Reddit** | r/physics, r/MachineLearning, r/MedicalPhysics, r/photography |
| **Mastodon** | Academic science community |

---

## 6. Timeline

### Pre-Publication (Now → Paper Submission)

| Week | Action |
|------|--------|
| 1 | Draft "Periodic Table of Cameras" infographic |
| 2 | Draft Twitter/X thread (ready to post on pub day) |
| 3 | Draft blog posts (Night Mode, Samsung Zoom) |
| 4 | Pitch Nature News & Views companion piece |

### Publication Week

| Day | Action |
|-----|--------|
| Day 0 | Paper goes live. Post Twitter thread. Submit to Hacker News. |
| Day 1 | Publish "Periodic Table of Cameras" infographic. LinkedIn post. |
| Day 2 | Publish "Night Mode vs. MRI" blog post. Pitch The Verge / Ars Technica. |
| Day 3 | Publish "Samsung 100x Zoom" blog post. Pitch PetaPixel. |
| Day 5 | Pitch YouTube creators (3Blue1Brown, Veritasium, Two Minute Papers). |

### Post-Publication (Weeks 2–12)

| Week | Action |
|------|--------|
| 2–3 | Launch interactive web demo ("Diagnose Your Phone Camera") |
| 4–6 | Produce "11 Primitives in 11 Minutes" YouTube video |
| 6–8 | Write "Primitive of the Week" Twitter series (11 weeks) |
| 8–12 | Seek YouTube collaborations; conference talks (CVPR, MICCAI, SIGGRAPH) |

### Metrics to Track

| Metric | Target (3 months post-pub) | Target (12 months) |
|--------|---------------------------|---------------------|
| Paper citations | 10–20 | 50–100 |
| Twitter impressions | 500K | 2M |
| GitHub stars | 500 | 2,000 |
| Blog post views | 100K total | 500K total |
| Web demo users | 50K | 200K |
| YouTube views | 100K (own) + collab views | 1M total |
| Media articles | 10 | 30 |
| Conference invitations | 3 | 10 |
