# Benchmark 1 — LLM Spec Router

## Overview

| Field | Value |
|-------|-------|
| **Task** | Given a natural-language description of an imaging system, select the correct spec from 4 choices |
| **Format** | JSON, 50 samples per variant |
| **Metric** | spec-match accuracy (% correct) |
| **Input** | Natural-language description + 4 candidate specs (A/B/C/D) |
| **Output** | JSON: `{"answer": "A"}` (one letter) |

## Data Structure

Each sample in the JSON file contains:

```json
{
  "id": 1,
  "description": "A spectral imaging system that uses a coded aperture mask followed by a prism to disperse light across wavelengths, summing spectral channels onto a 2D detector.",
  "correct_spec": "M(mask) → W(α, a) → Σ_λ → D(g, η₄)",
  "correct_dag": [
    {"primitive": "M",     "params": "mask",   "label": "Coded Aperture"},
    {"primitive": "W",     "params": "α, a",   "label": "Prism Dispersion"},
    {"primitive": "Sigma", "params": "λ",      "label": "Spectral Sum"},
    {"primitive": "D",     "params": "g, η₄",  "label": "Detector"}
  ],
  "distractors": [
    "Π(fan) → D(g, η₂)",
    "F(spiral) → C(psf) → D(g, η₃)",
    "P(fresnel) → |·|² → D(g, η₁)"
  ]
}
```

### Field Descriptions

| Field | Description |
|-------|-------------|
| `id` | Sample index (1–50) |
| `description` | Natural-language description of the imaging system. All 50 samples describe the same modality but with different phrasings. |
| `correct_spec` | The correct spec string using the primitive notation |
| `correct_dag` | The spec decomposed into an ordered list of primitives, each with parameters and a human-readable label |
| `distractors` | 3 incorrect specs from other modalities |

### Spec Notation

Each spec is a chain of physics primitives:

| Primitive | Meaning | Example Params |
|-----------|---------|----------------|
| `M(·)` | Mask / spatial modulation | `mask` (binary coded aperture) |
| `W(·)` | Wavelength dispersion | `α, a` (prism angle, aperture) |
| `Σ_λ` | Spectral summation | `λ` (wavelength axis) |
| `D(·)` | Detector (gain + noise) | `g, η` (gain, noise level) |
| `Π(·)` | Projection | `fan`, `cone`, `ray` |
| `F(·)` | Fourier / k-space sampling | `k-traj`, `spiral` |
| `C(·)` | Convolution (PSF/blur) | `psf`, `CTF` |
| `P(·)` | Propagation | `fresnel`, `e⁻` |
| `\|·\|²` | Intensity (magnitude squared) | — |
| `R(·)` | Rotation | `θ` |

---

## Prompt Sent to the LLM

### System Message

```
You are an expert in computational imaging and physics-based forward models.
You will be given a natural-language description of an imaging system and four
candidate spec strings (labelled A, B, C, D). Exactly one is correct.

Reply with ONLY a JSON object: {"answer": "<letter>"}
where <letter> is one of A, B, C, or D.
```

### User Message — Example 1 (Correct: B)

```
### Imaging System Description
A spectral imaging system that uses a coded aperture mask followed by a prism
to disperse light across wavelengths, summing spectral channels onto a 2D detector.

### Candidate Specs
A. P(fresnel) → |·|² → D(g, η₁)
B. M(mask) → W(α, a) → Σ_λ → D(g, η₄)
C. F(spiral) → C(psf) → D(g, η₃)
D. Π(fan) → D(g, η₂)

Which spec correctly describes this imaging system?
```

### Expected LLM Response

```json
{"answer": "B"}
```

### User Message — Example 2 (Correct: D)

```
### Imaging System Description
An imaging system for capturing hyperspectral data in a single snapshot by
modulating the scene with a binary mask and then dispersing the modulated
signal with a prism.

### Candidate Specs
A. Π(fan) → D(g, η₂)
B. F(spiral) → C(psf) → D(g, η₃)
C. P(fresnel) → |·|² → D(g, η₁)
D. M(mask) → W(α, a) → Σ_λ → D(g, η₄)

Which spec correctly describes this imaging system?
```

### User Message — Example 3 (Correct: B)

```
### Imaging System Description
A compressive spectral imager where a coded aperture spatially modulates the
scene, a dispersive element shifts each wavelength band, and the spectral
channels are summed on a focal plane array.

### Candidate Specs
A. P(fresnel) → |·|² → D(g, η₁)
B. M(mask) → W(α, a) → Σ_λ → D(g, η₄)
C. F(spiral) → C(psf) → D(g, η₃)
D. Π(fan) → D(g, η₂)

Which spec correctly describes this imaging system?
```

---

## How It Works

1. The **correct spec** stays the same across all 50 samples (it is the variant's true spec).
2. The **description** changes each sample — 50 different natural-language paraphrases of the same imaging system.
3. The **distractors** are specs from 3 other unrelated modalities.
4. The 4 choices (1 correct + 3 distractors) are **shuffled randomly** per sample, so the correct answer letter (A/B/C/D) varies.
5. The LLM must pick the correct letter based on understanding the physics described in the text.

## Scoring

```
spec_match_accuracy = (number of correct answers) / 50
```

Per variant, per model. Results are aggregated across all 65 variants for overall model ranking.

---

## Data File Location

```
platform/pwm_platform/static/benchmark-data/v1.0/{variant}_b1_public.json
```

Example: `sd_cassi_b1_public.json` (50 samples for the SD-CASSI variant)
