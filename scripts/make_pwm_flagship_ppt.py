#!/usr/bin/env python3
"""Generate PowerPoint presentation for PWM Flagship paper.

Title: Eleven Primitives and Three Gates: The Universal Structure of Computational Imaging
Target: 40-minute talk for Prof. Xin Yuan's group at Westlake University
"""

import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "papers" / "pwm_flagship"
FIG = OUT / "figures"

# ── Color palette ─────────────────────────────────────────────────
WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
BLACK   = RGBColor(0x00, 0x00, 0x00)
DARK    = RGBColor(0x1A, 0x1A, 0x2E)
BLUE    = RGBColor(0x2E, 0x5F, 0xA1)
ORANGE  = RGBColor(0xE8, 0x71, 0x2B)
GREEN   = RGBColor(0x3D, 0x9E, 0x3F)
RED     = RGBColor(0xC9, 0x3C, 0x3C)
PURPLE  = RGBColor(0x7B, 0x4E, 0xA3)
GRAY    = RGBColor(0x6B, 0x6B, 0x6B)
LTGRAY  = RGBColor(0xE8, 0xE8, 0xE8)
GATE1   = RGBColor(0x4A, 0x90, 0xD9)  # blue
GATE2   = RGBColor(0xE8, 0xA8, 0x38)  # amber
GATE3   = RGBColor(0xD9, 0x4A, 0x4A)  # red
BG_DARK = RGBColor(0x0F, 0x17, 0x2A)
BG_SLIDE = RGBColor(0xF5, 0xF7, 0xFA)

prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)


# ── Helper functions ──────────────────────────────────────────────
def add_bg(slide, color=BG_SLIDE):
    bg = slide.background
    fill = bg.fill
    fill.solid()
    fill.fore_color.rgb = color

def add_text_box(slide, left, top, width, height, text, font_size=18,
                 bold=False, color=BLACK, align=PP_ALIGN.LEFT, font_name='Calibri'):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(font_size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font_name
    p.alignment = align
    return tf

def add_bullet_slide(slide, left, top, width, height, items, font_size=18,
                     color=BLACK, spacing=Pt(6)):
    txBox = slide.shapes.add_textbox(Inches(left), Inches(top),
                                     Inches(width), Inches(height))
    tf = txBox.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.text = item
        p.font.size = Pt(font_size)
        p.font.color.rgb = color
        p.font.name = 'Calibri'
        p.space_after = spacing
        p.level = 0
    return tf

def add_title_bar(slide, title, subtitle=None):
    """Add a colored title bar at the top of a content slide."""
    shape = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, Inches(0), Inches(0),
        Inches(13.333), Inches(1.1))
    shape.fill.solid()
    shape.fill.fore_color.rgb = BLUE
    shape.line.fill.background()
    add_text_box(slide, 0.6, 0.15, 12, 0.7, title,
                 font_size=32, bold=True, color=WHITE)
    if subtitle:
        add_text_box(slide, 0.6, 0.7, 12, 0.4, subtitle,
                     font_size=16, color=RGBColor(0xCC, 0xDD, 0xFF))

def add_section_slide(slide, section_title, section_number=None):
    """Full-page section divider."""
    add_bg(slide, DARK)
    if section_number:
        add_text_box(slide, 0.6, 2.0, 12, 0.8, section_number,
                     font_size=24, color=GATE1, bold=False)
    add_text_box(slide, 0.6, 2.7, 12, 1.5, section_title,
                 font_size=48, bold=True, color=WHITE)

def add_slide_number(slide, num, total):
    add_text_box(slide, 12.3, 7.0, 1.0, 0.4, f"{num}/{total}",
                 font_size=12, color=GRAY, align=PP_ALIGN.RIGHT)

def new_slide():
    return prs.slides.add_slide(prs.slide_layouts[6])  # blank


# ── Total slide count for numbering ──────────────────────────────
TOTAL = 30
sn = 0

# ══════════════════════════════════════════════════════════════════
# SLIDE 1: Title
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide, BG_DARK)

add_text_box(slide, 0.8, 1.0, 11.7, 1.5,
             "Eleven Primitives and Three Gates:",
             font_size=44, bold=True, color=WHITE)
add_text_box(slide, 0.8, 2.0, 11.7, 1.0,
             "The Universal Structure of Computational Imaging",
             font_size=40, bold=True, color=GATE1)

add_text_box(slide, 0.8, 3.5, 11.7, 0.5,
             "Chengshuai Yang, David J. Brady, Xin Yuan",
             font_size=24, color=RGBColor(0xBB, 0xBB, 0xBB))
add_text_box(slide, 0.8, 4.2, 11.7, 0.8,
             "NextGen PlatformAI C Corp  |  University of Arizona  |  Westlake University",
             font_size=18, color=GRAY)

add_text_box(slide, 0.8, 5.5, 11.7, 0.5,
             "Talk for Prof. Yuan's Group  |  March 2026",
             font_size=18, color=GRAY)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 2: Outline
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Talk Outline", "~40 minutes")
items = [
    "1.  Motivation: Why do reconstruction algorithms fail on real instruments?",
    "2.  Key Idea: 11 primitives + 3 gates = universal grammar",
    "3.  The Finite Primitive Basis Theorem (Theorem 1)",
    "      - OperatorGraph representation",
    "      - 11 canonical primitives",
    "      - Basis-growth saturation",
    "4.  The Triad Decomposition (Theorem 2)",
    "      - Three gates: Recoverability, Carrier Budget, Operator Mismatch",
    "      - Four-Scenario evaluation protocol",
    "5.  Experimental Validation (12 modalities, 5 carrier families)",
    "      - CASSI deep dive + hardware validation",
    "      - Cross-carrier results: CT, MRI, cryo-EM, ultrasound",
    "6.  Discussion and Future Directions",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=20, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 3: Motivation - The Problem
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "The Problem: Silent Reconstruction Failure")

items = [
    "Computational imaging has made enormous progress in algorithms:",
    "  - Compressed sensing, Plug-and-Play, deep unrolling, vision transformers...",
    "",
    "But algorithms routinely fail on deployed instruments.",
    "",
    "The root cause is often NOT the algorithm --- it's the forward model.",
    "",
    "When the assumed operator H_nom differs from the true physics H_true,",
    "the solver faithfully solves the WRONG problem.",
    "",
    "This failure is silent and systematic:",
    "  - Produces structured artifacts that mimic signal content",
    "  - Practitioners blame the solver, not the model",
    "  - No general framework to diagnose or fix it",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=20, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 4: Motivation - Diversity vs. Universality
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Diverse Systems, Shared Structure")

items = [
    "A coded-aperture camera, an MRI scanner, and a cryo-electron microscope",
    "  appear to share nothing: different carriers, geometries, detectors.",
    "",
    "Yet each converts a physical scene into measurements by composing",
    "  a small chain of transformations:",
    "     propagation  ->  interaction  ->  encoding  ->  detection",
    "",
    "Key insight: This universality is not accidental --- it reflects the",
    "finite structure of physical measurement processes.",
    "",
    "Question: Can we formalize this?",
    "  - What are the minimal building blocks?",
    "  - Why do reconstructions fail, and how to diagnose them?",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=20, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 5: Big Picture / Overview
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Our Answer: 11 Primitives + 3 Gates", "A universal grammar for computational imaging")

# Two boxes side by side
# Left: Theorem 1
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.8), Inches(1.5),
    Inches(5.5), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
shape.line.color.rgb = BLUE
add_text_box(slide, 1.1, 1.6, 5.0, 0.5,
             "Theorem 1: Finite Primitive Basis",
             font_size=22, bold=True, color=BLUE)
items1 = [
    "Every imaging forward model decomposes",
    "into a DAG over exactly 11 primitives.",
    "",
    "Sufficient + Minimal basis.",
    "Covers all 5 carrier families.",
    "170 modalities validated.",
]
add_bullet_slide(slide, 1.1, 2.2, 5.0, 2.0, items1, font_size=17, color=DARK)

# Right: Theorem 2
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5),
    Inches(5.7), Inches(2.8))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xFE, 0xF0, 0xE8)
shape.line.color.rgb = ORANGE
add_text_box(slide, 7.1, 1.6, 5.2, 0.5,
             "Theorem 2: Triad Decomposition",
             font_size=22, bold=True, color=ORANGE)
items2 = [
    "Every reconstruction failure has exactly",
    "3 independent root causes (gates):",
    "",
    "Gate 1: Information deficiency",
    "Gate 2: Carrier noise",
    "Gate 3: Operator mismatch",
]
add_bullet_slide(slide, 7.1, 2.2, 5.2, 2.0, items2, font_size=17, color=DARK)

# Bottom: Impact
items3 = [
    "Together: a compositional language for designing, diagnosing, and correcting ANY imaging system.",
    "Validation: 12 modalities, 5 carrier families, +0.8 to +13.9 dB recovery on real instruments.",
]
add_bullet_slide(slide, 0.8, 4.7, 11.7, 2.0, items3, font_size=20, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 6: Figure 1 - Overview
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 1: The Universal Grammar of Computational Imaging")
if (FIG / "fig1_overview.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig1_overview.png"),
        Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.8))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 7: Section Divider - Finite Primitive Basis
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_section_slide(slide, "The Finite Primitive Basis", "PART I")
add_text_box(slide, 0.6, 4.0, 12, 1.0,
             "Every imaging forward model decomposes into a DAG over exactly 11 primitives",
             font_size=22, color=GRAY)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 8: OperatorGraph
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "The OperatorGraph Representation")

items = [
    "Every imaging forward model encoded as a typed DAG:",
    "  - Each node wraps a single primitive physical operator",
    "  - Edges define data flow from source to detector",
    "",
    "Every primitive implements:",
    "  - forward(x) -> y    (physics simulation)",
    "  - adjoint(y) -> x    (validated: <Hx,y> = <x,H*y>)",
    "",
    "Key properties:",
    "  - Adjoint consistency: randomized dot-product test, delta < 10^-6",
    "  - Lipschitz bounds: nonlinear primitives restricted to 5 canonical families",
    "  - Not a universal approximator (unlike neural networks)",
    "",
    "This makes reconstruction both interpretable and certifiable.",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=19, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 9: The 11 Primitives
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "The 11 Canonical Primitives")

# Table header
header = "  #    Primitive       Notation          Physical Action"
primitives = [
    "  1    Propagate       P(d, lambda)       Free-space wave propagation",
    "  2    Modulate        M(m)              Element-wise multiplication (mask, coil, absorption)",
    "  3    Project         Pi(theta)          Radon line-integral projection",
    "  4    Encode          F(k)              Fourier-domain encoding (k-space)",
    "  5    Convolve        C(h)              Spatial convolution (PSF)",
    "  6    Accumulate      Sigma             Summation over spectral/temporal axis",
    "  7    Detect          D(g, eta)          Detector response (5 canonical families)",
    "  8    Sample          S(Omega)          Sub-sampling on index set",
    "  9    Disperse        W(alpha, a)        Wavelength-dependent spatial shift",
    " 10    Scatter         R(sigma, dE)       Direction change and/or energy shift",
    " 11    Transform       Lambda(f, theta)   Pointwise nonlinear physics operation",
]

tf = add_text_box(slide, 0.5, 1.3, 12.3, 0.5, header,
                  font_size=16, bold=True, color=BLUE, font_name='Consolas')
add_bullet_slide(slide, 0.5, 1.8, 12.3, 5.0, primitives,
                 font_size=15, color=DARK, spacing=Pt(2))

items_bottom = [
    "9 linear primitives (have adjoint) + 2 nonlinear (Detect, Transform)",
    "Detect: 5 families (linear-field, log, sigmoid, intensity |x|^2, coherent) -- max 2 params each",
    "Transform: 5 families (Beer-Lambert, log, phase wrap, polynomial, saturation) -- bounded Lipschitz",
]
add_bullet_slide(slide, 0.5, 5.8, 12.3, 1.5, items_bottom,
                 font_size=15, color=GRAY, spacing=Pt(2))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 10: Figure 2 - OperatorGraph Examples
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 2: OperatorGraph DAGs + Fidelity Ladder + Basis Growth")
if (FIG / "fig2_operatorgraph.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig2_operatorgraph.png"),
        Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.8))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 11: DAG Examples
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Example DAG Decompositions")

examples = [
    "CASSI (photon):       M -> W -> Sigma -> D         (4 nodes, depth 4)",
    "    mask * disperse * sum_spectral * detect",
    "",
    "CT (X-ray):           Pi -> D                       (2 nodes, depth 2)",
    "    radon_project * detect",
    "",
    "MRI (nuclear spin):   M -> F -> S -> D              (4 nodes, depth 4)",
    "    coil_modulate * fourier_encode * k-space_sample * detect",
    "",
    "Ptychography (e-):    M -> P -> D                   (3 nodes, depth 3)",
    "    probe_modulate * fresnel_propagate * detect",
    "",
    "Ultrasound:           P -> Sigma -> D               (3 nodes, depth 3)",
    "    wave_propagate * delay_sum * detect",
    "",
    "Median: 3 nodes, depth 3.  Max observed: 5 nodes, depth 5.",
    "Theoretical bounds: N_max = 20, D_max = 10 (very conservative).",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, examples, font_size=17, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 12: Theorem Statement
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Theorem 1: Finite Primitive Basis")

# Theorem box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(1.0), Inches(1.5),
    Inches(11.3), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
shape.line.color.rgb = BLUE

items_thm = [
    "For every H in C_img (the imaging operator class),",
    "there exists a typed DAG G with nodes from B = {P, M, Pi, F, C, Sigma, D, S, W, R, Lambda}",
    "that is an epsilon-approximate representation of H.",
    "",
    "The library B has exactly 11 elements and is minimal:",
    "removing any primitive leaves at least one modality with error > epsilon.",
]
add_bullet_slide(slide, 1.3, 1.6, 10.7, 2.2, items_thm, font_size=20, color=DARK)

items_proof = [
    "Proof sketch:",
    "  1. H = H_K ... H_1 with K <= N_max stages",
    "  2. Each stage classified into 6 physics families",
    "  3. Each family mapped to 1-2 primitives",
    "  4. Linear composition: sub-multiplicative error ||H - H_G|| <= K * max(eps_k) * B^(K-1)",
    "  5. Nonlinear: Lipschitz bound ||H(x) - H_G(x)|| <= sum eps_k * prod gamma_j",
    "",
    "Minimality: 8 primitives strictly necessary; 3 more needed under complexity bounds.",
    "Witness modality for each removal (e.g., without Pi, CT has e_img >> 0.01).",
]
add_bullet_slide(slide, 0.8, 4.3, 11.7, 3.0, items_proof, font_size=17, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 13: Decomposition Table
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Decomposition Results: All e_img < 0.01")

items = [
    "Full-validation modalities (12 with 4-scenario correction):",
    "  CASSI (photon):      M->W->Sigma->D        e < 10^-4",
    "  CACTI (photon):      M->Sigma->D            e < 10^-4",
    "  CT (X-ray):          Pi->D                   e < 10^-5",
    "  MRI (spin):          M->F->S->D              e < 10^-6",
    "  Ptychography (e-):   M->P->D                 e = 4.2x10^-4",
    "  + 7 more modalities (lensless, SPC, fluorescence, holography, CBCT, cryo-EM, ultrasound)",
    "",
    "Held-out modalities (frozen library, 0 new primitives):",
    "  OCT, photoacoustic, SIM, phase-contrast X-ray, electron ptychography",
    "  All pass: e_img < 0.01",
    "",
    "Exotic modalities:",
    "  Ghost imaging, THz-TDS, Compton*, Raman, fluorescence, DOT, Brillouin",
    "  *Compton triggered extension protocol: e=0.34 without Scatter R -> e<0.01 with R",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=17, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 14: Basis Saturation
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Basis-Growth Saturation: K=11 Suffices")

if (FIG / "fig9_basis_growth.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig9_basis_growth.png"),
        Inches(1.0), Inches(1.5), Inches(5.5), Inches(5.0))

items = [
    "9 of 11 primitives introduced by",
    "  first 10 modalities",
    "",
    "K = 11 at N = 35 modalities",
    "No new primitive for next 135",
    "",
    "Sublinear, saturating growth:",
    "  New modalities compose existing",
    "  primitives rather than requiring",
    "  new ones",
    "",
    "Consistent with theorem:",
    "  Once all 6 physics-stage families",
    "  are covered, basis is complete.",
]
add_bullet_slide(slide, 7.0, 1.4, 5.5, 5.5, items, font_size=18, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 15: Section Divider - Triad
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_section_slide(slide, "The Triad Decomposition", "PART II")
add_text_box(slide, 0.6, 4.0, 12, 1.0,
             "Every reconstruction failure has exactly 3 independent root causes",
             font_size=22, color=GRAY)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 16: Why Exactly 3 Gates
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Why Exactly Three Gates?")

items = [
    "The reconstruction pipeline has exactly 3 inputs:",
    "",
    "  1. A sensing geometry H  -->  determines what information is captured",
    "  2. A physical carrier      -->  determines the noise floor",
    "  3. A forward model H_nom  -->  what the solver inverts",
    "",
    "Any reconstruction error must trace to one of these:",
    "",
    "  Gate 1: Information never captured (null space of H)",
    "  Gate 2: Information captured but corrupted (carrier noise)",
    "  Gate 3: Information captured but misinterpreted (H_nom != H_true)",
    "",
    "This trichotomy is exhaustive:",
    "  MSE <= MSE^(G1) + MSE^(G2) + MSE^(G3)     (no residual term)",
    "",
    "Solver suboptimality is subsumed by Gate 3 (solver's implicit model).",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=19, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 17: Three Gates Details
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "The Three Gates in Detail")

# Gate 1 box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.3),
    Inches(4.0), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE0, 0xEE, 0xFA)
shape.line.color.rgb = GATE1
add_text_box(slide, 0.7, 1.4, 3.6, 0.4,
             "Gate 1: Recoverability", font_size=20, bold=True, color=GATE1)
items_g1 = [
    "Is enough information",
    "captured?",
    "",
    "Compression ratio gamma=m/n",
    "Null space dimension",
    "PSNR ceiling from theory",
    "Fix: acquire more data",
]
add_bullet_slide(slide, 0.7, 1.9, 3.6, 1.8, items_g1, font_size=14, color=DARK, spacing=Pt(2))

# Gate 2 box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(4.7), Inches(1.3),
    Inches(4.0), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xFA, 0xF0, 0xE0)
shape.line.color.rgb = GATE2
add_text_box(slide, 4.9, 1.4, 3.6, 0.4,
             "Gate 2: Carrier Budget", font_size=20, bold=True, color=GATE2)
items_g2 = [
    "Is the SNR sufficient?",
    "",
    "Shot noise (photon counting)",
    "Thermal noise (electronic)",
    "T1/T2 relaxation (MRI)",
    "Fix: source/detector",
    "      improvement",
]
add_bullet_slide(slide, 4.9, 1.9, 3.6, 1.8, items_g2, font_size=14, color=DARK, spacing=Pt(2))

# Gate 3 box
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(8.9), Inches(1.3),
    Inches(4.0), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xFA, 0xE0, 0xE0)
shape.line.color.rgb = GATE3
add_text_box(slide, 9.1, 1.4, 3.6, 0.4,
             "Gate 3: Operator Mismatch", font_size=20, bold=True, color=GATE3)
items_g3 = [
    "Does H_nom match H_true?",
    "",
    "Geometric misalignment",
    "Parameter drift",
    "Model simplification",
    "Fix: calibration /",
    "      operator correction",
]
add_bullet_slide(slide, 9.1, 1.9, 3.6, 1.8, items_g3, font_size=14, color=DARK, spacing=Pt(2))

# Bottom: Key insight
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(4.2),
    Inches(12.3), Inches(2.5))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xFE, 0xE8)
shape.line.color.rgb = GREEN
add_text_box(slide, 0.8, 4.3, 11.7, 0.4,
             "Key Insight: Gate 3 dominates in well-designed instruments",
             font_size=20, bold=True, color=GREEN)
items_key = [
    "Gates 1 & 2 are addressed at design time (sampling geometry, carrier selection).",
    "Gate 3 (operator mismatch) is the deployment-stage bottleneck: 14/14 configurations show Gate 3 dominant.",
    "This is why algorithms often fail on real instruments: the model is wrong, not the solver.",
    "The Triad tells you WHERE the error is, and WHAT to fix.",
]
add_bullet_slide(slide, 0.8, 4.8, 11.7, 1.8, items_key, font_size=16, color=DARK, spacing=Pt(2))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 18: Figure 3 - Triad
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 3: Triad Decomposition Validated Across All Carriers")
if (FIG / "fig3_triad.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig3_triad.png"),
        Inches(0.5), Inches(1.3), Inches(12.3), Inches(5.8))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 19: Four-Scenario Protocol
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Evaluation: The Four-Scenario Protocol")

items = [
    "Every modality evaluated under 4 standardized scenarios:",
    "",
    "  Scenario I  (Ideal):     y = H_true * x,  reconstruct with H_true",
    "       --> Oracle upper bound. Best possible quality.",
    "",
    "  Scenario II (Mismatch):  y = H_true * x,  reconstruct with H_nom (H_nom != H_true)",
    "       --> Standard operating condition. The model is WRONG.",
    "",
    "  Scenario III (Oracle):   Same y as Sc. II, reconstruct with H_true",
    "       --> Correction ceiling. What's achievable if you knew the truth.",
    "",
    "  Scenario IV (Corrected): Same y as Sc. II, reconstruct with H_hat (estimated)",
    "       --> Our contribution. Autonomous calibration recovers quality.",
    "",
    "Recovery ratio: rho = (PSNR_IV - PSNR_II) / (PSNR_I - PSNR_II)",
    "  rho=0: calibration useless.  rho=1: fully closes the gap.",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=18, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 20: Section Divider - Results
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_section_slide(slide, "Experimental Results", "PART III")
add_text_box(slide, 0.6, 4.0, 12, 1.0,
             "12 modalities, 5 carrier families, +0.8 to +13.9 dB recovery",
             font_size=22, color=GRAY)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 21: Figure 4 - Correction Bar Chart
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 4: Correction Results Across 14 Configurations")
if (FIG / "fig4_correction_bar.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig4_correction_bar.png"),
        Inches(0.3), Inches(1.3), Inches(6.0), Inches(5.5))

items = [
    "14 configurations across 5 carriers:",
    "",
    "Optical photons:",
    "  CASSI: +6.68 dB (10 scenes)",
    "  CACTI: +4.37 dB (6 videos)",
    "  SPC: +2.31 dB (11 images)",
    "  Lensless: +13.9 dB",
    "  Holography: +7.41 dB",
    "  Fluorescence: +0.80 dB",
    "",
    "X-ray: CT +3.10 dB, CBCT +5.0 dB",
    "Electron: Ptycho +6.77 dB, Cryo +2.4 dB",
    "Spin: MRI +2.5 dB",
    "Acoustic: Ultrasound +1.8 dB",
    "",
    "Gate 3 dominant in ALL 14/14 cases!",
]
add_bullet_slide(slide, 6.5, 1.3, 6.3, 6.0, items, font_size=16, color=DARK, spacing=Pt(2))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 22: CASSI Deep Dive
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "CASSI Deep Dive: Coded Aperture Spectral Imaging")

items = [
    "CASSI forward model:  M(mask) -> W(dispersion) -> Sigma(spectral sum) -> D(detect)",
    "",
    "5-parameter mismatch model:",
    "  (dx, dy, theta, a1, alpha) = spatial shifts + rotation + dispersion slope + axis angle",
    "  True values: dx=0.5 px, dy=0.3 px, theta=0.1 deg, a1=2.02, alpha=0.15 deg",
    "",
    "Correction algorithms:",
    "  Algorithm 1: Hierarchical beam search (~300s/scene)",
    "    1D sweeps -> 3D beam search -> 2D beam search -> coordinate descent",
    "    Accuracy: +/-0.1-0.2 pixels",
    "",
    "  Algorithm 2: Joint gradient refinement (~3200s/scene)",
    "    Differentiable forward model + Adam optimizer + multi-start",
    "    Accuracy: +/-0.05-0.1 pixels (3-5x improvement)",
    "",
    "Both are SOLVER-AGNOSTIC: corrected operator benefits any downstream solver",
    "  (TwIST, GAP-TV, DGSMP, MST-L, CST-L) without retraining!",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=17, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 23: Figure 5 - CASSI Deep Dive + Hardware
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 5: CASSI Results + Hardware Validation")
if (FIG / "fig5_deepdive.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig5_deepdive.png"),
        Inches(0.3), Inches(1.3), Inches(6.0), Inches(5.5))
if (FIG / "fig5_hardware.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig5_hardware.png"),
        Inches(6.5), Inches(1.3), Inches(6.3), Inches(5.5))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 24: Cross-Carrier Validation
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Cross-Carrier Validation: 5 Carrier Families")

items = [
    "OPTICAL PHOTONS (6 modalities):",
    "  CASSI:  Sc.II = 27.63 dB -> Sc.IV = 34.31 dB  (recovery ratio 0.95)",
    "  CACTI:  +4.37 dB correction, EfficientSCI solver",
    "  SPC:    3 compression ratios (10%, 25%, 50%), exponential gain drift",
    "",
    "X-RAY PHOTONS (2 modalities):",
    "  CT: center-of-rotation offset correction, FBP + SART solvers",
    "  CBCT: detector offset, 5 real images (LoDoPaB-CT, FIPS, HTC)",
    "",
    "ELECTRONS (2 modalities):",
    "  Ptychography: probe position jitter correction via ePIE",
    "  Cryo-EM: defocus error correction, CTF-based Wiener filtering",
    "",
    "NUCLEAR SPINS (1 modality):",
    "  MRI: coil sensitivity error correction, ESPIRiT calibration adopted",
    "",
    "ACOUSTIC WAVES (1 modality):",
    "  Ultrasound: speed-of-sound error correction, delay-and-sum beamforming",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=17, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 25: Figure 6 - Zero-shot
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 6: Cross-Carrier Zero-Shot Generalization")
if (FIG / "fig6_zeroshot.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig6_zeroshot.png"),
        Inches(1.0), Inches(1.3), Inches(11.0), Inches(5.8))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 26: Visual Comparison
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Figure 8: Visual Comparison Across All Scenarios")
if (FIG / "fig8_visual_comparison.png").exists():
    slide.shapes.add_picture(
        str(FIG / "fig8_visual_comparison.png"),
        Inches(0.3), Inches(1.3), Inches(12.7), Inches(5.8))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 27: Solver-Agnostic Nature
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Key Property: Solver-Agnostic Correction")

items = [
    "Critical distinction from previous work:",
    "",
    "  Traditional approach:  Fix the algorithm (better solver, more training data)",
    "  Our approach:          Fix the operator (calibrate H_nom -> H_true)",
    "",
    "The corrected operator benefits ANY downstream solver:",
    "  - TwIST, GAP-TV (optimization-based)",
    "  - DGSMP, MST-L, CST-L, HDNet (deep learning)",
    "  - All improve without retraining!",
    "",
    "Why this matters:",
    "  - Solver design and operator correction are orthogonal",
    "  - Better solvers + better operators = compounding gains",
    "  - One calibration serves the entire solver ecosystem",
    "",
    "Relationship to Prof. Yuan's SCI work:",
    "  - The SCI framework unifies coded-aperture modalities",
    "  - Our 11 primitives provide the structural explanation of why SCI works",
    "  - Gate 3 analysis extends SCI to deployment-stage robustness",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=18, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 28: Discussion
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Discussion and Broader Impact")

items = [
    "What this paper establishes:",
    "  1. Universal structure: 11 primitives cover 170+ modalities (all 5 carrier families)",
    "  2. Complete diagnosis: 3 gates exhaust all root causes of reconstruction failure",
    "  3. Actionable correction: +0.8 to +13.9 dB on 12 validated modalities",
    "",
    "Relationship to existing work:",
    "  - Not a better solver: COMSOL/FEniCS solve single problems better",
    "  - Complementary: fixes the operator so that ANY solver works better",
    "  - Extends SCI (Yuan et al.) from optical to all 5 carrier families",
    "",
    "Generalization beyond imaging:",
    "  - Seismology: wave propagation + attenuation + sensor response",
    "  - Radio astronomy: aperture synthesis + ionospheric distortion + noise",
    "  - Particle physics: interaction model + detector response + pile-up",
    "  - All exhibit the same 3-gate failure structure",
    "",
    "Limitation: formal proof in companion paper [yang2026fpt]",
]
add_bullet_slide(slide, 0.8, 1.4, 11.7, 5.5, items, font_size=18, color=DARK)
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 29: Summary
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide)
add_title_bar(slide, "Summary")

# Two highlight boxes
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(1.5),
    Inches(6.0), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xF0, 0xFE)
shape.line.color.rgb = BLUE
items_s1 = [
    "11 Primitives (Theorem 1):",
    "  Every imaging forward model = DAG over 11 primitives",
    "  Minimal, sufficient, covers 170 modalities",
    "  Basis-growth saturates at K=11",
]
add_bullet_slide(slide, 0.7, 1.6, 5.6, 1.8, items_s1, font_size=17, color=DARK, spacing=Pt(2))

shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(6.8), Inches(1.5),
    Inches(6.0), Inches(2.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xFE, 0xF0, 0xE8)
shape.line.color.rgb = ORANGE
items_s2 = [
    "3 Gates (Theorem 2):",
    "  Every failure = G1 (info) + G2 (noise) + G3 (mismatch)",
    "  Gate 3 dominates in deployed instruments (14/14)",
    "  Autonomous correction: +0.8 to +13.9 dB",
]
add_bullet_slide(slide, 7.0, 1.6, 5.6, 1.8, items_s2, font_size=17, color=DARK, spacing=Pt(2))

# Key numbers
shape = slide.shapes.add_shape(
    MSO_SHAPE.ROUNDED_RECTANGLE, Inches(0.5), Inches(3.8),
    Inches(12.3), Inches(3.0))
shape.fill.solid()
shape.fill.fore_color.rgb = RGBColor(0xE8, 0xFE, 0xE8)
shape.line.color.rgb = GREEN

items_nums = [
    "Key Numbers:",
    "  - 11 primitives, minimal and sufficient",
    "  - 3 gates, exhaustive decomposition (MSE = MSE_G1 + MSE_G2 + MSE_G3)",
    "  - 170 modality templates, 5 carrier families",
    "  - 12 modalities with full 4-scenario validation (including 2 Nobel Prize techniques)",
    "  - +0.8 to +13.9 dB recovery on deployed instruments",
    "  - Gate 3 dominant in 14/14 validated configurations (100%)",
    "  - Solver-agnostic: one calibration benefits all downstream solvers",
    "",
    "The first universal grammar for designing, diagnosing, and correcting",
    "computational imaging systems.",
]
add_bullet_slide(slide, 0.7, 3.9, 11.9, 2.8, items_nums, font_size=18, color=DARK, spacing=Pt(2))
add_slide_number(slide, sn, TOTAL)

# ══════════════════════════════════════════════════════════════════
# SLIDE 30: Thank You
# ══════════════════════════════════════════════════════════════════
sn += 1
slide = new_slide()
add_bg(slide, BG_DARK)

add_text_box(slide, 0.8, 2.0, 11.7, 1.0,
             "Thank You!", font_size=56, bold=True, color=WHITE,
             align=PP_ALIGN.CENTER)
add_text_box(slide, 0.8, 3.2, 11.7, 0.8,
             "Questions & Discussion",
             font_size=32, color=GATE1, align=PP_ALIGN.CENTER)

add_text_box(slide, 0.8, 4.5, 11.7, 0.5,
             "Chengshuai Yang  |  integrityyang@gmail.com",
             font_size=20, color=GRAY, align=PP_ALIGN.CENTER)
add_text_box(slide, 0.8, 5.0, 11.7, 0.5,
             "Code: github.com/integritynoble/Physics_World_Model",
             font_size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_text_box(slide, 0.8, 5.5, 11.7, 0.5,
             "Acknowledgments: Prof. David Brady (U. Arizona), Prof. Xin Yuan (Westlake)",
             font_size=18, color=GRAY, align=PP_ALIGN.CENTER)
add_slide_number(slide, sn, TOTAL)

# ── Save ──────────────────────────────────────────────────────────
out_path = OUT / "talk_pwm_flagship.pptx"
prs.save(str(out_path))
print(f"\nPresentation saved: {out_path}")
print(f"Total slides: {sn}")
print(f"Estimated duration: ~40 minutes (1.3 min/slide)")
