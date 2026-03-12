#!/usr/bin/env python3
"""
Full usage demo — simulates the complete three-agent pipeline for CT.

This demo shows exactly what the real pipeline does:
  Period 1 (Forward):  Plan Agent → Judge Agent → Performance Agent → sinogram
  Period 2 (Recon):    Plan Agent → Judge Agent → Performance Agent → x_hat + metrics

All forward simulation and reconstruction use the REAL code.
The LLM agent calls are replaced with hand-crafted plans to run without an API key.
"""
from __future__ import annotations
import sys, os
import numpy as np

# Ensure package imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from system_design.schemas.plan import (
    PlanDocument, ForwardAction, ReconAction,
    FlowchartElement, NoiseSpec, MismatchSpec,
    AlgorithmStep, PlanDemands,
)
from system_design.schemas.judgment import JudgmentResult, JudgmentIssue
from system_design.pipeline.forward_simulator import ForwardSimulator
from system_design.pipeline.reconstructor import Reconstructor
from system_design.agents.performance_agent import PerformanceAgent, ForwardResult, ReconResult
from system_design.database import get_modality_context_for_prompt, get_algorithm_context_for_prompt
from system_design.utils.metrics import psnr, ssim
from system_design.config import OUTPUT_DIR, PSNR_TARGET, SSIM_TARGET

# ══════════════════════════════════════════════════════════════════════════════
# HELPER
# ══════════════════════════════════════════════════════════════════════════════
def banner(title: str):
    w = 70
    print(f"\n{'='*w}")
    print(f"  {title}")
    print(f"{'='*w}\n")


def section(title: str):
    print(f"\n--- {title} ---\n")


# ══════════════════════════════════════════════════════════════════════════════
# STEP 0: Show database context (what the Plan Agent sees)
# ══════════════════════════════════════════════════════════════════════════════
banner("STEP 0: Database Context (injected into Plan Agent prompt)")

print("Modality context for CT:")
print(get_modality_context_for_prompt("ct"))
print()
print("Algorithm context for CT:")
print(get_algorithm_context_for_prompt("ct"))


# ══════════════════════════════════════════════════════════════════════════════
# PERIOD 1: FORWARD DESIGN
# ══════════════════════════════════════════════════════════════════════════════
banner("PERIOD 1: FORWARD — Design a sparse-view low-dose CT system")

# ------ Step 1: Plan Agent generates a plan ------
section("Step 1: Plan Agent → PlanDocument (forward)")

print("User prompt: 'Design a sparse-view CT system with 60 projection angles,")
print("              low dose (I0=1e4), for pediatric chest imaging.'\n")

forward_plan = PlanDocument(
    modality="ct",
    period="forward",
    version=1,
    iteration=1,
    task=(
        "Design a complete forward model for sparse-view X-ray CT with only "
        "60 projection angles and low photon flux (I0=1e4) for pediatric "
        "chest imaging. The system must model all physical elements from "
        "X-ray source through detector digitization, including realistic "
        "noise and calibration mismatch sources."
    ),
    plan_steps=[
        "Configure polychromatic X-ray tube source at 80 kVp with 1.5mm Al filtration",
        "Model Beer-Lambert attenuation through soft tissue phantom",
        "Define parallel-beam acquisition geometry with 60 projection angles over 180°",
        "Simulate flat-panel CsI:Tl detector with Poisson noise (I0=1e4) and Gaussian readout noise",
        "Apply 12-bit ADC digitization with dark current",
        "Identify beam hardening, scatter, and center-of-rotation mismatch sources",
    ],
    action=ForwardAction(
        flowchart_ascii=(
            "[X-ray Tube 80kVp] → [Soft Tissue Phantom] → [Parallel-Beam 60 angles]\n"
            "       ↓                      ↓                        ↓\n"
            "  [Polychromatic         [Beer-Lambert           [CoR offset\n"
            "   beam hardening]        attenuation]            mismatch]\n"
            "                                                       ↓\n"
            "                              → [CsI:Tl Flat Panel Detector] → [12-bit ADC] → y\n"
            "                                        ↓\n"
            "                                  [Poisson I0=1e4]\n"
            "                                  [Gaussian σ=3 e⁻]\n"
            "                                  [Dark current 0.05 e⁻/s]"
        ),
        elements=[
            FlowchartElement(
                id="xray_source",
                name="X-ray Tube Source (80 kVp)",
                type="source",
                parameters={
                    "energy_kVp": 80,
                    "flux_photons_per_s": 5e5,
                    "focal_spot_mm": 0.4,
                    "filtration": "1.5mm Al",
                    "spectrum": "polychromatic",
                },
                noise=[],
                mismatch=[
                    MismatchSpec(
                        type="beam_hardening",
                        description="Polychromatic spectrum causes cupping artifacts in soft tissue",
                        severity="high",
                        correction_method="2nd-order polynomial linearization from water phantom calibration",
                    ),
                ],
                connects_to=["tissue_attenuation"],
            ),
            FlowchartElement(
                id="tissue_attenuation",
                name="Soft Tissue Attenuation",
                type="interaction",
                parameters={
                    "model": "beer_lambert",
                    "mu_water_cm": 0.184,  # at 80 kVp effective energy
                    "material": "pediatric_soft_tissue",
                },
                noise=[],
                mismatch=[
                    MismatchSpec(
                        type="scatter",
                        description="Compton scatter adds low-frequency background (SPR ~0.3 for pediatric chest)",
                        severity="medium",
                        correction_method="Scatter kernel estimation with 1D convolution correction",
                    ),
                ],
                connects_to=["geometry"],
            ),
            FlowchartElement(
                id="geometry",
                name="Parallel-Beam Acquisition (60 angles)",
                type="geometry",
                parameters={
                    "scan_type": "parallel_beam",
                    "num_angles": 60,
                    "angular_range_deg": 180,
                    "detector_pixels": 256,
                    "pixel_pitch_mm": 0.4,
                },
                noise=[],
                mismatch=[
                    MismatchSpec(
                        type="center_of_rotation_offset",
                        description="Mechanical misalignment causes ring artifacts (estimated ±0.5 px)",
                        severity="medium",
                        correction_method="Cross-correlation of 0°/180° projection pair",
                    ),
                ],
                connects_to=["detector"],
            ),
            FlowchartElement(
                id="detector",
                name="CsI:Tl Flat Panel Detector",
                type="detector",
                parameters={
                    "scintillator": "CsI:Tl",
                    "pixels": [256, 256],
                    "pixel_pitch_mm": 0.4,
                    "quantum_efficiency": 0.75,
                },
                noise=[
                    NoiseSpec(type="poisson", parameters={"I0": 1e4}),
                    NoiseSpec(type="gaussian", parameters={"sigma_electrons": 3.0}),
                    NoiseSpec(type="dark_current", parameters={"electrons_per_s": 0.05, "exposure_s": 0.02}),
                ],
                mismatch=[
                    MismatchSpec(
                        type="detector_response_nonuniformity",
                        description="Per-pixel gain variations up to ±2%",
                        severity="low",
                        correction_method="Flat-field correction with air scan",
                    ),
                ],
                connects_to=["adc"],
            ),
            FlowchartElement(
                id="adc",
                name="12-bit ADC",
                type="digitization",
                parameters={"bit_depth": 12, "dynamic_range_db": 72},
                noise=[],
                mismatch=[],
            ),
        ],
        measurement_shape="(256, 60)",
        total_noise_model="y ~ Poisson(I0 * exp(-H*x)) + N(0, σ_readout²) + Poisson(dark * t_exp)",
    ),
    demands=PlanDemands(
        feasibility="yes",
        budget_feasible="yes",
        algorithm_convergence="N/A",
        comments="Low-dose (I0=1e4) will produce noisy sinograms (estimated SNR ~17 dB). "
                 "Sparse view (60 angles) will cause streak artifacts in FBP but is "
                 "recoverable with iterative reconstruction.",
    ),
)

# Save and display the plan markdown
os.makedirs(OUTPUT_DIR, exist_ok=True)
forward_plan.markdown_path = os.path.join(OUTPUT_DIR, "ct_forward_v1_iter1.md")
md_text = forward_plan.to_markdown()
with open(forward_plan.markdown_path, "w") as f:
    f.write(md_text)

print(f"Plan saved to: {forward_plan.markdown_path}")
print(f"Plan length: {len(md_text)} chars, {md_text.count(chr(10))} lines")
print("\n" + "·"*50)
print("PLAN MARKDOWN PREVIEW (first 60 lines):")
print("·"*50)
for line in md_text.split("\n")[:60]:
    print(f"  {line}")
print("  ...")


# ------ Step 2: Judge Agent evaluates the plan ------
section("Step 2: Judge Agent → JudgmentResult")

forward_judgment = JudgmentResult(
    feasible=True,
    confidence=0.88,
    summary=(
        "The forward model is physically sound. All major elements (source, "
        "attenuation, geometry, detector, ADC) are present with appropriate noise "
        "models. The low photon flux (I0=1e4) will produce moderately noisy "
        "sinograms but is realistic for low-dose pediatric CT. Beam hardening "
        "and scatter mismatch are correctly identified. The 60-angle sparse view "
        "is aggressive but feasible with iterative reconstruction."
    ),
    issues=[
        JudgmentIssue(
            category="noise_level",
            severity="warning",
            element_id="detector",
            description="Estimated SNR ~17 dB at I0=1e4 is low; FBP will produce noisy images",
            suggestion="Use iterative reconstruction (TV-ADMM or PnP-ADMM) instead of FBP",
        ),
        JudgmentIssue(
            category="physics",
            severity="warning",
            element_id="xray_source",
            description="Polychromatic beam hardening correction should be applied before reconstruction",
            suggestion="Include beam hardening polynomial correction in the mismatch pipeline",
        ),
    ],
    snr_estimate_db=17.2,
    cost_estimate_usd=380000.0,
    budget_ok=True,
    redesign_prompt="",
)

print(f"Verdict:    {'PASS ✓' if forward_judgment.feasible else 'FAIL ✗'}")
print(f"Confidence: {forward_judgment.confidence:.2f}")
print(f"SNR est.:   {forward_judgment.snr_estimate_db:.1f} dB")
print(f"Budget est.: ${forward_judgment.cost_estimate_usd:,.0f}")
print(f"Issues:     {len(forward_judgment.issues)} warnings, {len(forward_judgment.critical_issues)} critical")
for issue in forward_judgment.issues:
    print(f"  [{issue.severity}] {issue.category} ({issue.element_id}): {issue.description}")
print(f"\nSummary: {forward_judgment.summary}")


# ------ Step 3: Performance Agent runs the forward model ------
section("Step 3: Performance Agent → Forward Simulation")

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

phantom = resize(shepp_logan_phantom(), (256, 256), anti_aliasing=True).astype(np.float32)
phantom /= phantom.max()

print(f"Phantom: Shepp-Logan {phantom.shape}, range=[{phantom.min():.3f}, {phantom.max():.3f}]")

perf_agent = PerformanceAgent()
fwd_result: ForwardResult = perf_agent.run(plan=forward_plan, phantom=phantom)

y = fwd_result.measurements
print(f"\nMeasurements (sinogram): {y.shape}")
print(f"  range = [{y.min():.4f}, {y.max():.4f}]")
print(f"  mean  = {y.mean():.4f}")
print(f"  Element outputs: {list(fwd_result.element_outputs.keys())}")

# Save forward result
np.savez(
    os.path.join(OUTPUT_DIR, "ct_forward_result.npz"),
    measurements=y,
    ground_truth=phantom,
)
print(f"\nSaved → {OUTPUT_DIR}/ct_forward_result.npz")


# ══════════════════════════════════════════════════════════════════════════════
# PERIOD 2: RECONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════
banner("PERIOD 2: RECONSTRUCTION — Recover image from sparse, noisy sinogram")

# ------ Step 1: Plan Agent generates reconstruction plan ------
section("Step 1: Plan Agent → PlanDocument (reconstruction)")

print("User prompt: 'Reconstruct the 60-angle low-dose CT sinogram using TV-ADMM")
print("              with beam hardening and scatter correction.'\n")

recon_plan = PlanDocument(
    modality="ct",
    period="reconstruction",
    version=1,
    iteration=1,
    task=(
        "Reconstruct the original attenuation map from a sparse-view (60 angles), "
        "low-dose (I0=1e4) CT sinogram. The inverse problem is severely ill-posed "
        "due to angular undersampling and high noise. The sinogram is also corrupted "
        "by beam hardening and Compton scatter. Use TV-ADMM with mismatch corrections "
        "to regularize the reconstruction."
    ),
    plan_steps=[
        "Apply beam hardening polynomial correction to the sinogram",
        "Apply scatter kernel subtraction to remove low-frequency bias",
        "Initialize with Filtered Back-Projection (FBP, ramp filter)",
        "Run TV-ADMM: alternate data fidelity gradient step and TV proximal step",
        "Enforce non-negativity constraint at each iteration",
        "Check convergence: ||x_{k+1} - x_k|| / ||x_k|| < 1e-4 or max 100 iterations",
    ],
    action=ReconAction(
        algorithm_name="TV-ADMM",
        algorithm_type="Variational",
        steps=[
            AlgorithmStep(
                step=1,
                name="Mismatch Pre-Correction",
                description="Apply beam hardening polynomial linearization and scatter subtraction to the raw sinogram before reconstruction.",
                equation="y_corr = a0 + a1*y + a2*y^2 - 0.1*G_sigma(y)",
                parameters={"a0": 0.0, "a1": 1.0, "a2": -0.05, "scatter_sigma": 20.0},
            ),
            AlgorithmStep(
                step=2,
                name="FBP Initialization",
                description="Compute initial estimate via filtered back-projection with a Ram-Lak filter.",
                equation="x_0 = R^{-1}_{FBP}(y_corr)",
                parameters={"filter": "ramp"},
            ),
            AlgorithmStep(
                step=3,
                name="Data Fidelity Gradient",
                description="Compute the gradient of 0.5*||Rx - y_corr||^2 where R is the Radon operator.",
                equation="grad = R^T(Rx_k - y_corr)",
                parameters={},
            ),
            AlgorithmStep(
                step=4,
                name="TV Proximal Step",
                description="Apply the proximal operator of the isotropic total variation penalty. This preserves edges while denoising flat regions.",
                equation="x_{k+1} = prox_{lambda*TV}(x_k - eta * grad)",
                parameters={"lambda_tv": 0.01, "eta": 0.005},
            ),
            AlgorithmStep(
                step=5,
                name="Non-Negativity Projection",
                description="Project onto the non-negative orthant since attenuation coefficients are non-negative.",
                equation="x_{k+1} = max(x_{k+1}, 0)",
                parameters={},
            ),
        ],
        convergence_criterion="||x_{k+1} - x_k||_2 / ||x_k||_2 < 1e-4, or max 100 iterations",
        mismatch_corrections=[
            MismatchSpec(
                type="beam_hardening",
                description="Polychromatic cupping artifact from 80 kVp source",
                severity="high",
                correction_method="2nd-order polynomial linearization: y_corr = y - 0.05*y^2",
            ),
            MismatchSpec(
                type="scatter",
                description="Compton scatter low-frequency bias (SPR ~0.3)",
                severity="medium",
                correction_method="Subtract 10% of Gaussian-blurred sinogram (sigma=20 px)",
            ),
        ],
        hyperparameters={
            "lambda_tv": 0.01,
            "step_size": 0.005,
            "num_iterations": 100,
            "filter": "ramp",
        },
        expected_runtime_s=5.0,
        references=[
            "Sidky & Pan, 'Image reconstruction in circular cone-beam CT by constrained TV minimization', Phys. Med. Biol. 53(17), 2008",
            "Kak & Slaney, 'Principles of Computerized Tomographic Imaging', IEEE Press 1988",
        ],
    ),
    demands=PlanDemands(
        feasibility="yes",
        budget_feasible="N/A",
        algorithm_convergence="yes",
        comments="TV-ADMM converges reliably for sparse-view CT. The TV penalty "
                 "effectively suppresses streak artifacts from angular undersampling. "
                 "Beam hardening and scatter pre-corrections are essential at this "
                 "noise level.",
    ),
)

# Save and display
recon_plan.markdown_path = os.path.join(OUTPUT_DIR, "ct_reconstruction_v1_iter1.md")
recon_md = recon_plan.to_markdown()
with open(recon_plan.markdown_path, "w") as f:
    f.write(recon_md)

print(f"Plan saved to: {recon_plan.markdown_path}")
print(f"Plan length: {len(recon_md)} chars\n")
print("·"*50)
print("PLAN MARKDOWN PREVIEW (first 60 lines):")
print("·"*50)
for line in recon_md.split("\n")[:60]:
    print(f"  {line}")
print("  ...")


# ------ Step 2: Judge Agent evaluates reconstruction plan ------
section("Step 2: Judge Agent → JudgmentResult")

recon_judgment = JudgmentResult(
    feasible=True,
    confidence=0.92,
    summary=(
        "The reconstruction plan is algorithmically sound. TV-ADMM is a "
        "well-established method for sparse-view CT with proven convergence "
        "guarantees under convexity (TV penalty is convex). The beam hardening "
        "polynomial correction and scatter subtraction are appropriate for the "
        "identified mismatch sources. Step sizes and regularization weight are "
        "within reasonable ranges. No pre-training required — this is a pure "
        "optimization method suitable for test-time execution."
    ),
    issues=[
        JudgmentIssue(
            category="convergence",
            severity="warning",
            element_id="",
            description="lambda_tv=0.01 may over-smooth fine structures; consider adaptive tuning",
            suggestion="Start with lambda_tv=0.005 and increase if streak artifacts persist",
        ),
    ],
    convergence_likely=True,
    mismatch_handled=True,
    redesign_prompt="",
)

print(f"Verdict:        {'PASS' if recon_judgment.feasible else 'FAIL'}")
print(f"Confidence:     {recon_judgment.confidence:.2f}")
print(f"Convergence:    {'likely' if recon_judgment.convergence_likely else 'unlikely'}")
print(f"Mismatch:       {'handled' if recon_judgment.mismatch_handled else 'NOT handled'}")
print(f"Issues:         {len(recon_judgment.issues)} warnings")
for issue in recon_judgment.issues:
    print(f"  [{issue.severity}] {issue.category}: {issue.description}")
print(f"\nSummary: {recon_judgment.summary}")


# ------ Step 3: Performance Agent runs reconstruction ------
section("Step 3: Performance Agent → Reconstruction")

recon_result: ReconResult = perf_agent.run(
    plan=recon_plan,
    measurements=y,
    x_true=phantom,
)

x_hat = recon_result.x_hat
print(f"\nReconstruction: {x_hat.shape}")
print(f"  range = [{x_hat.min():.4f}, {x_hat.max():.4f}]")
print(f"  PSNR  = {recon_result.psnr_db:.2f} dB")
print(f"  SSIM  = {recon_result.ssim_val:.4f}")
print(f"  Time  = {recon_result.runtime_s:.2f} s")
print(f"  Iters = {len(recon_result.convergence_history)}")

# Save reconstruction result
np.savez(
    os.path.join(OUTPUT_DIR, "ct_reconstruction_result.npz"),
    x_hat=x_hat,
    x_true=phantom,
)
print(f"\nSaved → {OUTPUT_DIR}/ct_reconstruction_result.npz")


# ══════════════════════════════════════════════════════════════════════════════
# ALSO DEMONSTRATE: Judge REJECTS a plan (redesign loop)
# ══════════════════════════════════════════════════════════════════════════════
banner("BONUS: Demonstrating Judge REJECTION → Redesign Loop")

section("Iteration 1: Bad plan — physically impossible MRI design")

bad_plan = PlanDocument(
    modality="mri",
    period="forward",
    version=1,
    iteration=1,
    task="Design a 0.001T MRI system with 100x acceleration",
    plan_steps=["Use 0.001T magnet", "100x undersampling"],
    action=ForwardAction(
        flowchart_ascii="[0.001T Magnet] → [100x k-space] → [Single coil]",
        elements=[
            FlowchartElement(id="magnet", name="0.001T Magnet", type="source",
                parameters={"field_strength_T": 0.001}),
            FlowchartElement(id="kspace", name="100x Undersampled k-Space", type="geometry",
                parameters={"acceleration_factor": 100, "num_coils": 1}),
        ],
        measurement_shape="(256, 256)",
    ),
    demands=PlanDemands(feasibility="yes", budget_feasible="yes"),
)

bad_judgment = JudgmentResult(
    feasible=False,
    confidence=0.95,
    summary=(
        "REJECTED: 0.001T field produces virtually no signal — thermal noise "
        "will completely dominate. Additionally, 100x acceleration with a single "
        "coil violates the Nyquist sampling theorem (max acceleration ≈ num_coils). "
        "No reconstruction algorithm can recover meaningful images."
    ),
    issues=[
        JudgmentIssue(
            category="physics",
            severity="critical",
            element_id="magnet",
            description="0.001T field strength produces SNR < 0.1 dB — no usable signal",
            suggestion="Use at least 0.5T (ideally 1.5T or 3T) for clinical imaging",
        ),
        JudgmentIssue(
            category="physics",
            severity="critical",
            element_id="kspace",
            description="100x acceleration with 1 coil violates max_acceleration ≈ num_coils constraint",
            suggestion="Use 8-coil array (max 8x acceleration) or reduce acceleration factor",
        ),
    ],
    convergence_likely=False,
    mismatch_handled=False,
    redesign_prompt=(
        "CRITICAL: (1) Increase field strength to at least 1.5T for adequate SNR. "
        "(2) Reduce acceleration to at most num_coils (use 8-coil array with 4x acceleration). "
        "(3) Add coil sensitivity estimation (ESPIRiT) to handle parallel imaging."
    ),
)

print(f"Verdict:    {'PASS' if bad_judgment.feasible else 'FAIL ✗ REJECTED'}")
print(f"Confidence: {bad_judgment.confidence:.2f}")
print(f"Critical:   {len(bad_judgment.critical_issues)} critical issues")
for issue in bad_judgment.critical_issues:
    print(f"  [CRITICAL] {issue.category} ({issue.element_id}): {issue.description}")
print(f"\nRedesign prompt sent back to Plan Agent:")
print(f"  '{bad_judgment.redesign_prompt}'")

section("Iteration 2: Plan Agent redesigns with judge feedback → corrected plan")

fixed_plan = PlanDocument(
    modality="mri",
    period="forward",
    version=1,
    iteration=2,
    task="Design a 3T MRI system with 4x Cartesian acceleration using 8-coil array",
    plan_steps=[
        "Configure 3T main field with ±1 ppm homogeneity",
        "Apply 90° RF excitation (sinc pulse, BW=3kHz)",
        "Model Bloch equation tissue response (T1=1000ms, T2=80ms)",
        "Sample Cartesian k-space with 4x acceleration + ACS centre lines",
        "Receive with 8-channel phased array coil",
        "Apply 16-bit ADC digitization",
    ],
    action=ForwardAction(
        flowchart_ascii="[3T B0] → [RF 90°] → [Tissue] → [4x k-space] → [8-ch Coil] → [ADC]",
        elements=[
            FlowchartElement(id="b0", name="3T Main Field", type="source",
                parameters={"field_strength_T": 3.0, "homogeneity_ppm": 1.0},
                mismatch=[MismatchSpec(type="b0_inhomogeneity", severity="medium")],
                connects_to=["rf"]),
            FlowchartElement(id="rf", name="RF Excitation", type="source",
                parameters={"flip_angle_deg": 90, "bandwidth_hz": 3000},
                connects_to=["tissue"]),
            FlowchartElement(id="tissue", name="Tissue Response", type="interaction",
                parameters={"model": "bloch_equations", "T1_range_ms": [800, 1200]},
                connects_to=["kspace"]),
            FlowchartElement(id="kspace", name="4x Cartesian k-Space", type="geometry",
                parameters={"trajectory": "cartesian", "acceleration_factor": 4, "num_coils": 8},
                connects_to=["coil"]),
            FlowchartElement(id="coil", name="8-Channel Phased Array", type="detector",
                parameters={"num_coils": 8},
                noise=[NoiseSpec(type="thermal_johnson", parameters={"T_K": 310, "bandwidth_hz": 150000})],
                connects_to=["adc"]),
            FlowchartElement(id="adc", name="16-bit ADC", type="digitization",
                parameters={"bit_depth": 16}),
        ],
        measurement_shape="(256, 256) complex64",
    ),
    demands=PlanDemands(feasibility="yes", budget_feasible="yes"),
)

fixed_judgment = JudgmentResult(
    feasible=True,
    confidence=0.91,
    summary="Corrected plan is physically sound. 3T field provides adequate SNR. "
            "4x acceleration with 8-coil array is well within the parallel imaging limit. "
            "Plan now includes B0 inhomogeneity mismatch identification.",
    issues=[],
    convergence_likely=True,
    mismatch_handled=True,
)

print(f"Verdict:    {'PASS' if fixed_judgment.feasible else 'FAIL'}")
print(f"Confidence: {fixed_judgment.confidence:.2f}")
print(f"Summary:    {fixed_judgment.summary}")
print(f"\n→ Pipeline would now proceed to Performance Agent for execution.")


# ══════════════════════════════════════════════════════════════════════════════
# FINAL SUMMARY
# ══════════════════════════════════════════════════════════════════════════════
banner("SUMMARY")

print("Files generated:")
for f in sorted(os.listdir(OUTPUT_DIR)):
    path = os.path.join(OUTPUT_DIR, f)
    size = os.path.getsize(path)
    print(f"  {f:45s}  {size:>8,} bytes")

print(f"""
Results:
  Forward:
    Sinogram shape:    {y.shape}
    Sinogram range:    [{y.min():.3f}, {y.max():.3f}]

  Reconstruction:
    x_hat shape:       {x_hat.shape}
    PSNR:              {recon_result.psnr_db:.2f} dB   (target: {PSNR_TARGET} dB)
    SSIM:              {recon_result.ssim_val:.4f}   (target: {SSIM_TARGET})
    Iterations:        {len(recon_result.convergence_history)}
    Runtime:           {recon_result.runtime_s:.2f} s

Pipeline flow demonstrated:
  1. Plan Agent → structured markdown with 4 sections
  2. Judge Agent → feasibility verdict (PASS/FAIL)
     - PASS example: CT forward + reconstruction
     - FAIL example: Bad MRI design → redesign → PASS
  3. Performance Agent → real forward simulation + real reconstruction
""")
