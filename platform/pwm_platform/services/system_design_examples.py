"""Pre-built multi-round examples for the System Design Studio.

Each example demonstrates a full forward + reconstruction workflow with
multiple refinement rounds.  Data is static — no LLM calls needed.
"""

from __future__ import annotations

from typing import Any

# ── Types ───────────────────────────────────────────────────────────────────

Round = dict[str, Any]   # {user, assistant, judge, spec_judged}
Phase = dict[str, Any]   # {period, rounds, final_plan_md, final_spec_md}
Example = dict[str, Any] # {id, title, description, modality, phases}


def get_examples() -> list[Example]:
    return [_ct_example(), _mri_example(), _microscopy_example()]


def get_example_by_id(example_id: str) -> Example | None:
    for ex in get_examples():
        if ex["id"] == example_id:
            return ex
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Example 1 — Sparse-View CT
# ═════════════════════════════════════════════════════════════════════════════

def _ct_example() -> Example:
    return {
        "id": "ct_sparse",
        "title": "Sparse-View CT",
        "description": "60-angle low-dose CT: forward model design + TV-ADMM reconstruction with beam hardening correction",
        "modality": "ct",
        "phases": [
            # ── Forward Phase ───────────────────────────────────────────
            {
                "period": "forward",
                "rounds": [
                    {
                        "user": "Design a sparse-view CT system with 60 projection angles and low photon dose (I0=1e4) for pediatric chest imaging",
                        "assistant": (
                            "I've designed the forward model for: **Sparse-view X-ray CT with 60 angles, low-dose (I0=1e4) for pediatric chest imaging**\n\n"
                            "The signal chain has **5 elements**:\n\n"
                            "- **X-ray Tube Source 80 kVp** (source) | noise: none | mismatch: beam_hardening\n"
                            "- **Soft Tissue Attenuation** (interaction) | noise: none | mismatch: scatter\n"
                            "- **Parallel-Beam Geometry 60 angles** (geometry) | noise: none | mismatch: center_of_rotation_offset\n"
                            "- **CsI:Tl Flat Panel Detector** (detector) | noise: poisson, gaussian, dark_current\n"
                            "- **12-bit ADC** (digitization)\n\n"
                            "Composite noise: `y ~ Poisson(I0 * exp(-Hx)) + N(0, sigma^2) + Poisson(dark * t_exp)`\n"
                            "Output shape: `(256, 60)`\n\n"
                            "Feasibility: yes\n"
                            "Note: Low-dose (I0=1e4) will produce noisy sinograms (SNR ~17 dB). Sparse view (60 angles) causes streak artifacts in FBP but is recoverable with iterative methods."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 88%)\n\n"
                            "The forward model is physically sound. All five signal chain elements are present with realistic parameters.\n\n"
                            "[WARNING] **noise_level**: SNR ~17 dB is marginal for pediatric imaging. Consider increasing I0 to 5e4 if dose permits.\n"
                            "  -> Increase photon flux or add dose modulation\n"
                            "[WARNING] **mismatch**: Scatter correction uses simple kernel subtraction; consider Monte Carlo-based correction for more accuracy.\n"
                            "  -> Replace Gaussian kernel with patient-specific scatter estimation"
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "Increase detector pixels to 512 and add a more accurate scatter model using Monte Carlo estimation",
                        "assistant": (
                            "I've designed the forward model for: **Updated sparse-view CT with 512 detector pixels and MC scatter**\n\n"
                            "The signal chain has **5 elements**:\n\n"
                            "- **X-ray Tube Source 80 kVp** (source) | mismatch: beam_hardening\n"
                            "- **Soft Tissue Attenuation** (interaction) | mismatch: scatter (MC-based)\n"
                            "- **Parallel-Beam Geometry 60 angles** (geometry) | mismatch: center_of_rotation_offset\n"
                            "- **CsI:Tl Flat Panel 512px** (detector) | noise: poisson, gaussian, dark_current\n"
                            "- **14-bit ADC** (digitization)\n\n"
                            "Changes: detector_pixels 256->512, bit_depth 12->14, scatter correction updated to Monte Carlo estimation with patient-specific scatter kernels.\n\n"
                            "Output shape: `(512, 60)`\n\n"
                            "Feasibility: yes"
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 92%)\n\n"
                            "Improved design. Higher detector resolution (512px) and MC scatter model are good upgrades. The 14-bit ADC provides better dynamic range for the increased pixel count."
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "What if we use fan-beam geometry instead of parallel-beam? Add a 40cm source-to-isocenter distance",
                        "assistant": (
                            "I've designed the forward model for: **Fan-beam CT with 40cm SID, 60 views, 512 detector pixels**\n\n"
                            "The signal chain has **5 elements**:\n\n"
                            "- **X-ray Tube Source 80 kVp** (source) | mismatch: beam_hardening, focal_spot_blur\n"
                            "- **Soft Tissue Attenuation** (interaction) | mismatch: scatter (MC-based)\n"
                            "- **Fan-Beam Geometry** (geometry) | params: SID=40cm, SDD=80cm, fan_angle=45deg | mismatch: center_of_rotation_offset\n"
                            "- **CsI:Tl Curved Detector 512px** (detector) | noise: poisson, gaussian, dark_current\n"
                            "- **14-bit ADC** (digitization)\n\n"
                            "Key change: Parallel-beam -> fan-beam. Added focal spot blur mismatch. Detector is now curved to match fan geometry.\n\n"
                            "Output shape: `(512, 60)`\n\n"
                            "Feasibility: yes\n"
                            "Note: Fan-beam is more realistic for clinical CT. The short SID (40cm) increases cone-beam artifacts for off-axis slices."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 90%)\n\n"
                            "Fan-beam geometry is more clinically realistic. Good addition of focal spot blur.\n\n"
                            "[WARNING] **physics**: SID of 40cm is unusually short for clinical CT (typical 50-60cm). This increases magnification and may cause spatial resolution loss.\n"
                            "  -> Consider increasing SID to 50cm"
                        ),
                        "spec_judged": True,
                    },
                ],
                "final_plan_md": (
                    "# Task\n\nDesign a fan-beam X-ray CT system with 60 projection angles and low photon dose for pediatric chest imaging.\n\n"
                    "# Plan\n\n"
                    "1. Configure polychromatic X-ray tube at 80 kVp with 1.5mm Al filtration\n"
                    "2. Model Beer-Lambert attenuation through pediatric soft tissue\n"
                    "3. Define fan-beam geometry with 60 views, SID=40cm, SDD=80cm\n"
                    "4. Simulate curved CsI:Tl detector with Poisson noise (I0=1e4)\n"
                    "5. Apply 14-bit ADC digitization\n"
                    "6. Identify beam hardening, scatter, CoR, and focal spot mismatch sources\n\n"
                    "# Demands\n\n"
                    "- **feasibility**: yes\n"
                    "- **budget_feasible**: yes\n"
                    "- **algorithm_convergence**: N/A\n"
                ),
                "final_spec_md": (
                    "# Task\n\nDesign a fan-beam X-ray CT system with 60 projection angles and low photon dose for pediatric chest imaging.\n\n"
                    "# Plan\n\n"
                    "1. Configure polychromatic X-ray tube at 80 kVp\n"
                    "2. Model Beer-Lambert attenuation\n"
                    "3. Define fan-beam geometry (60 views, SID=40cm)\n"
                    "4. Simulate curved detector with noise\n"
                    "5. Apply 14-bit ADC\n"
                    "6. Identify all mismatch sources\n\n"
                    "# Action\n\n"
                    "## System Flowchart\n\n"
                    "```\n"
                    "[X-ray Tube 80kVp] -> [Soft Tissue] -> [Fan-Beam 60 views]\n"
                    "       |                    |                    |\n"
                    "  [beam_hardening]      [scatter]         [CoR offset]\n"
                    "  [focal_spot_blur]                            |\n"
                    "                          -> [CsI:Tl Curved 512px] -> [14-bit ADC] -> y\n"
                    "                                    |\n"
                    "                              [Poisson I0=1e4]\n"
                    "                              [Gaussian sigma=3]\n"
                    "                              [Dark current 0.05]\n"
                    "```\n\n"
                    "### Element: X-ray Tube Source 80 kVp (`xray_source`)\n\n"
                    "- **Type**: source\n"
                    "- **Parameters**:\n"
                    "  - `energy_kVp`: 80\n"
                    "  - `flux_photons_per_s`: 500000.0\n"
                    "  - `focal_spot_mm`: 0.4\n"
                    "  - `filtration`: 1.5mm Al\n"
                    "- **Mismatch sources**:\n"
                    "  - `beam_hardening` [high]: Polychromatic cupping -> correction: 2nd-order polynomial\n"
                    "  - `focal_spot_blur` [low]: Finite focal spot causes penumbra -> correction: Deconvolution\n\n"
                    "### Element: Soft Tissue Attenuation (`tissue_attenuation`)\n\n"
                    "- **Type**: interaction\n"
                    "- **Parameters**:\n"
                    "  - `model`: beer_lambert\n"
                    "  - `mu_water_cm`: 0.184\n"
                    "- **Mismatch sources**:\n"
                    "  - `scatter` [medium]: Compton scatter (SPR ~0.3) -> correction: MC estimation\n\n"
                    "### Element: Fan-Beam Geometry (`geometry`)\n\n"
                    "- **Type**: geometry\n"
                    "- **Parameters**:\n"
                    "  - `scan_type`: fan_beam\n"
                    "  - `num_angles`: 60\n"
                    "  - `SID_cm`: 40\n"
                    "  - `SDD_cm`: 80\n"
                    "  - `fan_angle_deg`: 45\n"
                    "  - `detector_pixels`: 512\n"
                    "- **Mismatch sources**:\n"
                    "  - `center_of_rotation_offset` [medium]: Mechanical misalignment -> correction: Cross-correlation\n\n"
                    "### Element: CsI:Tl Curved Detector (`detector`)\n\n"
                    "- **Type**: detector\n"
                    "- **Parameters**:\n"
                    "  - `scintillator`: CsI:Tl\n"
                    "  - `pixels`: 512\n"
                    "  - `pixel_pitch_mm`: 0.4\n"
                    "  - `quantum_efficiency`: 0.75\n"
                    "- **Noise**:\n"
                    "  - poisson: I0=10000.0\n"
                    "  - gaussian: sigma_electrons=3.0\n"
                    "  - dark_current: electrons_per_s=0.05\n\n"
                    "### Element: 14-bit ADC (`adc`)\n\n"
                    "- **Type**: digitization\n"
                    "- **Parameters**:\n"
                    "  - `bit_depth`: 14\n"
                    "  - `dynamic_range_db`: 84\n\n"
                    "## Composite Noise Model\n\n"
                    "```\ny ~ Poisson(I0 * exp(-Hx)) + N(0, sigma^2) + Poisson(dark * t_exp)\n```\n\n"
                    "**Measurement shape**: `(512, 60)`\n\n"
                    "# Demands\n\n"
                    "- **feasibility**: yes\n"
                    "- **budget_feasible**: yes\n"
                    "- **algorithm_convergence**: N/A\n\n"
                    "**Comments**: Fan-beam with 60 views at I0=1e4 is challenging but feasible with iterative reconstruction.\n"
                ),
            },
            # ── Reconstruction Phase ────────────────────────────────────
            {
                "period": "reconstruction",
                "rounds": [
                    {
                        "user": "Now design a TV-ADMM reconstruction for this sparse-view CT system",
                        "assistant": (
                            "I've designed a **TV-ADMM** (Variational) reconstruction plan.\n\n"
                            "**6 algorithm steps**:\n\n"
                            "1. **Beam Hardening Pre-Correction** - Apply 2nd-order polynomial linearization to sinogram\n"
                            "2. **Scatter Subtraction** - Subtract MC-estimated scatter from corrected sinogram\n"
                            "3. **FBP Initialization** - Compute initial estimate via filtered back-projection (Ram-Lak filter)\n"
                            "4. **Data Fidelity Gradient** - Compute gradient of 0.5*||Rx - y_corr||^2\n"
                            "5. **TV Proximal Step** - Apply proximal operator of isotropic TV penalty\n"
                            "6. **Non-Negativity Projection** - Project onto non-negative orthant\n\n"
                            "**2 mismatch corrections**:\n"
                            "- beam_hardening: 2nd-order polynomial linearization\n"
                            "- scatter: Monte Carlo kernel subtraction\n\n"
                            "Convergence: ||x_{k+1} - x_k||_2 / ||x_k||_2 < 1e-4, max 100 iterations"
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 91%)\n\n"
                            "TV-ADMM is well-suited for sparse-view CT. The algorithm steps are mathematically sound.\n\n"
                            "[WARNING] **algorithm**: Step size eta=0.005 may be too conservative for 60-view data. Consider adaptive step size.\n"
                            "  -> Use backtracking line search or Barzilai-Borwein step size"
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "Add the beam hardening polynomial coefficients and reduce lambda_tv to 0.005 for better detail preservation",
                        "assistant": (
                            "I've designed a **TV-ADMM** (Variational) reconstruction plan.\n\n"
                            "**6 algorithm steps**:\n\n"
                            "1. **Beam Hardening Pre-Correction** - Polynomial: y_corr = a0 + a1*y + a2*y^2, a0=0, a1=1.0, a2=-0.03\n"
                            "2. **Scatter Subtraction** - MC kernel, sigma=20px, fraction=0.08\n"
                            "3. **FBP Initialization** - Ram-Lak filter\n"
                            "4. **Data Fidelity Gradient** - grad = R^T(Rx_k - y_corr)\n"
                            "5. **TV Proximal Step** - lambda_tv=0.005, eta=0.005\n"
                            "6. **Non-Negativity Projection** - x_{k+1} = max(x_{k+1}, 0)\n\n"
                            "Changes: lambda_tv reduced 0.01->0.005, beam hardening a2=-0.05->-0.03 (less aggressive for pediatric), scatter fraction 0.10->0.08."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 93%)\n\n"
                            "Good refinements. Lower lambda_tv preserves more fine detail. Reduced beam hardening coefficient is appropriate for pediatric soft tissue."
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "Add Barzilai-Borwein adaptive step size to improve convergence speed",
                        "assistant": (
                            "I've designed a **TV-ADMM** (Variational) reconstruction plan.\n\n"
                            "**7 algorithm steps**:\n\n"
                            "1. **Beam Hardening Pre-Correction** - a0=0, a1=1.0, a2=-0.03\n"
                            "2. **Scatter Subtraction** - MC kernel subtraction\n"
                            "3. **FBP Initialization** - Ram-Lak filter\n"
                            "4. **Data Fidelity Gradient** - grad = R^T(Rx_k - y_corr)\n"
                            "5. **Barzilai-Borwein Step Size** - eta_k = (s^T s) / (s^T g) where s=x_k-x_{k-1}, g=grad_k-grad_{k-1}\n"
                            "6. **TV Proximal Step** - lambda_tv=0.005, adaptive eta_k\n"
                            "7. **Non-Negativity Projection** - x_{k+1} = max(x_{k+1}, 0)\n\n"
                            "Added BB step size (Step 5). Expected convergence in ~40-60 iterations instead of 100."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 95%)\n\n"
                            "Excellent addition. Barzilai-Borwein adaptive step size significantly accelerates ADMM convergence. The algorithm is now well-optimized for this problem."
                        ),
                        "spec_judged": True,
                    },
                ],
                "final_plan_md": (
                    "# Task\n\nReconstruct the attenuation map from a sparse-view (60 angles), low-dose (I0=1e4) fan-beam CT sinogram "
                    "with beam hardening and scatter corrections.\n\n"
                    "# Plan\n\n"
                    "1. Apply beam hardening polynomial correction\n"
                    "2. Subtract MC-estimated scatter\n"
                    "3. Initialize with FBP (Ram-Lak filter)\n"
                    "4. Run TV-ADMM with Barzilai-Borwein adaptive step size\n"
                    "5. Enforce non-negativity at each iteration\n"
                    "6. Check convergence: relative change < 1e-4 or max 100 iterations\n\n"
                    "# Demands\n\n"
                    "- **feasibility**: yes\n"
                    "- **budget_feasible**: N/A\n"
                    "- **algorithm_convergence**: yes\n"
                ),
                "final_spec_md": (
                    "# Task\n\nReconstruct the attenuation map from sparse-view fan-beam CT.\n\n"
                    "# Action\n\n"
                    "## Algorithm: TV-ADMM with Barzilai-Borwein\n\n"
                    "**Type**: Variational\n\n"
                    "### Algorithm Steps\n\n"
                    "**Step 1: Beam Hardening Pre-Correction**\n\n"
                    "$$\ny_corr = a0 + a1*y + a2*y^2\n$$\n"
                    "Parameters: a0=0, a1=1.0, a2=-0.03\n\n"
                    "**Step 2: Scatter Subtraction**\n\n"
                    "Subtract MC-estimated scatter kernel (sigma=20px, fraction=0.08)\n\n"
                    "**Step 3: FBP Initialization**\n\n"
                    "$$\nx_0 = FBP(y_corr)\n$$\n\n"
                    "**Step 4: Data Fidelity Gradient**\n\n"
                    "$$\ngrad = R^T(Rx_k - y_corr)\n$$\n\n"
                    "**Step 5: Barzilai-Borwein Step Size**\n\n"
                    "$$\neta_k = (s^T s) / (s^T g)\n$$\n"
                    "where s = x_k - x_{k-1}, g = grad_k - grad_{k-1}\n\n"
                    "**Step 6: TV Proximal Step**\n\n"
                    "$$\nx_{k+1} = prox_{lambda*TV}(x_k - eta_k * grad)\n$$\n"
                    "Parameters: lambda_tv=0.005\n\n"
                    "**Step 7: Non-Negativity Projection**\n\n"
                    "$$\nx_{k+1} = max(x_{k+1}, 0)\n$$\n\n"
                    "### Mismatch Corrections\n\n"
                    "- `beam_hardening` [high]: Polychromatic cupping\n"
                    "  Correction: 2nd-order polynomial (a2=-0.03)\n"
                    "- `scatter` [medium]: Compton scatter SPR~0.3\n"
                    "  Correction: MC kernel subtraction (sigma=20, frac=0.08)\n\n"
                    "**Convergence**: ||x_{k+1} - x_k||_2 / ||x_k||_2 < 1e-4, max 100 iterations\n\n"
                    "### Hyperparameters\n\n"
                    "- `lambda_tv`: 0.005\n"
                    "- `step_size`: adaptive (BB)\n"
                    "- `num_iterations`: 100\n"
                    "- `filter`: ramp\n\n"
                    "# Demands\n\n"
                    "- **feasibility**: yes\n"
                    "- **algorithm_convergence**: yes\n"
                ),
            },
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Example 2 — MRI Parallel Imaging
# ═════════════════════════════════════════════════════════════════════════════

def _mri_example() -> Example:
    return {
        "id": "mri_parallel",
        "title": "MRI Parallel Imaging",
        "description": "8-channel accelerated MRI: k-space undersampling forward model + GRAPPA reconstruction",
        "modality": "mri",
        "phases": [
            {
                "period": "forward",
                "rounds": [
                    {
                        "user": "Design an MRI forward model with 8-channel parallel imaging and 4x acceleration",
                        "assistant": (
                            "I've designed the forward model for: **8-channel parallel MRI with 4x undersampling**\n\n"
                            "The signal chain has **6 elements**:\n\n"
                            "- **1.5T Superconducting Magnet** (source) | mismatch: b0_inhomogeneity\n"
                            "- **RF Excitation** (source) | mismatch: b1_inhomogeneity\n"
                            "- **Tissue Relaxation** (interaction) | params: T1/T2 brain tissue\n"
                            "- **Cartesian k-space Sampling** (geometry) | params: R=4, ACS=24 lines | mismatch: eddy_currents\n"
                            "- **8-Channel Phased Array Coil** (detector) | noise: gaussian (thermal)\n"
                            "- **16-bit ADC** (digitization)\n\n"
                            "Composite noise: `y = M * F * S * x + N(0, sigma^2)` where M=mask, F=FFT, S=sensitivity maps\n"
                            "Output shape: `(8, 256, 64)` (8 coils, 256 readout, 64 phase-encode lines from 256)"
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 89%)\n\n"
                            "Correct parallel imaging model. 4x acceleration with 24 ACS lines is standard.\n\n"
                            "[WARNING] **physics**: B0 inhomogeneity at 1.5T can cause geometric distortion in EPI. Specify whether this is spin-echo or gradient-echo.\n"
                            "  -> Add sequence type parameter"
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "This is a spin-echo sequence. Add chemical shift artifact and specify brain tissue T1=900ms T2=80ms",
                        "assistant": (
                            "I've designed the forward model for: **Updated 8-channel SE-MRI with tissue parameters**\n\n"
                            "The signal chain has **6 elements**:\n\n"
                            "- **1.5T Superconducting Magnet** (source) | mismatch: b0_inhomogeneity\n"
                            "- **Spin-Echo RF** (source) | TE=15ms, TR=500ms | mismatch: b1_inhomogeneity\n"
                            "- **Brain Tissue Relaxation** (interaction) | T1=900ms, T2=80ms, PD=1.0 | mismatch: chemical_shift\n"
                            "- **Cartesian k-space R=4** (geometry) | ACS=24 | mismatch: eddy_currents\n"
                            "- **8-Channel Phased Array** (detector) | noise: gaussian sigma=0.01\n"
                            "- **16-bit ADC** (digitization)\n\n"
                            "Added: spin-echo sequence, tissue parameters, chemical shift artifact."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 93%)\n\n"
                            "Well-specified model. Spin-echo eliminates T2* effects. Chemical shift at 1.5T is ~220 Hz (3.5 ppm), which is manageable."
                        ),
                        "spec_judged": True,
                    },
                ],
                "final_plan_md": (
                    "# Task\n\nDesign a 1.5T 8-channel spin-echo MRI forward model with 4x parallel imaging acceleration.\n\n"
                    "# Plan\n\n"
                    "1. Model B0 field at 1.5T with inhomogeneity map\n"
                    "2. Apply spin-echo RF excitation (TE=15ms, TR=500ms)\n"
                    "3. Simulate tissue relaxation (T1=900ms, T2=80ms)\n"
                    "4. Sample Cartesian k-space with R=4, 24 ACS lines\n"
                    "5. Model 8-channel coil sensitivity maps\n"
                    "6. Add thermal noise and ADC quantization\n\n"
                    "# Demands\n\n- **feasibility**: yes\n"
                ),
                "final_spec_md": (
                    "# Task\n\n8-channel spin-echo MRI with 4x acceleration\n\n"
                    "# Action\n\n## System Flowchart\n\n"
                    "```\n[1.5T Magnet] -> [SE RF] -> [Tissue] -> [k-space R=4] -> [8ch Coil] -> [ADC] -> y\n```\n\n"
                    "### Element: 1.5T Magnet (`b0_field`)\n- **Type**: source\n- **Parameters**: field_strength_T=1.5\n"
                    "- **Mismatch**: `b0_inhomogeneity` [medium]\n\n"
                    "### Element: Spin-Echo RF (`rf_excitation`)\n- **Type**: source\n- **Parameters**: TE=15ms, TR=500ms, flip=90deg\n"
                    "- **Mismatch**: `b1_inhomogeneity` [low]\n\n"
                    "### Element: Brain Tissue (`tissue_relaxation`)\n- **Type**: interaction\n- **Parameters**: T1=900ms, T2=80ms, PD=1.0\n"
                    "- **Mismatch**: `chemical_shift` [low]: 220Hz at 1.5T\n\n"
                    "### Element: Cartesian k-space (`kspace_sampling`)\n- **Type**: geometry\n- **Parameters**: R=4, ACS_lines=24, matrix=256x256\n"
                    "- **Mismatch**: `eddy_currents` [low]\n\n"
                    "### Element: 8-Channel Coil (`phased_array`)\n- **Type**: detector\n- **Noise**: gaussian sigma=0.01\n\n"
                    "### Element: 16-bit ADC (`adc`)\n- **Type**: digitization\n- **Parameters**: bit_depth=16\n\n"
                    "**Measurement shape**: `(8, 256, 64)`\n\n"
                    "# Demands\n\n- **feasibility**: yes\n- **budget_feasible**: yes\n"
                ),
            },
            {
                "period": "reconstruction",
                "rounds": [
                    {
                        "user": "Design a GRAPPA reconstruction for this undersampled multi-coil MRI data",
                        "assistant": (
                            "I've designed a **GRAPPA** (Classical) reconstruction plan.\n\n"
                            "**5 algorithm steps**:\n\n"
                            "1. **ACS Extraction** - Extract 24 auto-calibration signal lines from center of k-space\n"
                            "2. **Kernel Fitting** - Fit GRAPPA weights from ACS data (kernel 5x4, 8 coils)\n"
                            "3. **Missing Line Interpolation** - Fill unacquired k-space lines using GRAPPA kernel\n"
                            "4. **Per-Coil IFFT** - Inverse FFT each coil image\n"
                            "5. **Root Sum-of-Squares Combination** - Combine coil images: x = sqrt(sum(|x_c|^2))\n\n"
                            "**1 mismatch correction**:\n"
                            "- coil_sensitivity_error: GRAPPA is self-calibrating, no explicit sensitivity maps needed\n\n"
                            "Convergence: N/A (non-iterative, single-pass)"
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 94%)\n\n"
                            "GRAPPA is the standard clinical choice for Cartesian parallel imaging at R=4. Kernel 5x4 with 24 ACS lines provides robust calibration."
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "Add noise pre-whitening step and use SENSE-GRAPPA hybrid for better SNR",
                        "assistant": (
                            "I've designed a **SENSE-GRAPPA Hybrid** (Classical) reconstruction plan.\n\n"
                            "**6 algorithm steps**:\n\n"
                            "1. **Noise Pre-Whitening** - Apply noise decorrelation matrix from noise-only prescan\n"
                            "2. **ACS Extraction** - Extract 24 central k-space lines\n"
                            "3. **GRAPPA Kernel Fitting** - Fit 5x4 kernel weights per coil\n"
                            "4. **k-space Interpolation** - Fill missing lines with GRAPPA weights\n"
                            "5. **SENSE Unfolding** - Apply coil sensitivity maps for final combination (better SNR than RSS)\n"
                            "6. **Residual Phase Correction** - Remove systematic phase from B0 inhomogeneity\n\n"
                            "Changes: Added noise pre-whitening (step 1), replaced RSS with SENSE combination (step 5), added phase correction (step 6)."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 96%)\n\n"
                            "SENSE-GRAPPA hybrid is an excellent choice. Noise pre-whitening improves g-factor, SENSE combination provides optimal SNR."
                        ),
                        "spec_judged": True,
                    },
                ],
                "final_plan_md": (
                    "# Task\n\nReconstruct fully-sampled brain image from 4x undersampled 8-channel k-space data.\n\n"
                    "# Plan\n\n"
                    "1. Pre-whiten noise across coil channels\n"
                    "2. Calibrate GRAPPA kernel from ACS lines\n"
                    "3. Interpolate missing k-space lines\n"
                    "4. Combine coil images with SENSE\n"
                    "5. Correct residual B0 phase\n\n"
                    "# Demands\n\n- **feasibility**: yes\n- **algorithm_convergence**: yes\n"
                ),
                "final_spec_md": (
                    "# Task\n\nReconstruct from 4x undersampled 8-channel MRI.\n\n"
                    "# Action\n\n## Algorithm: SENSE-GRAPPA Hybrid\n\n**Type**: Classical\n\n"
                    "### Algorithm Steps\n\n"
                    "**Step 1: Noise Pre-Whitening**\nApply decorrelation matrix W from prescan.\n\n"
                    "**Step 2: ACS Extraction**\n24 auto-calibration lines from k-space center.\n\n"
                    "**Step 3: GRAPPA Kernel Fitting**\nKernel size 5x4, fit per target coil.\n\n"
                    "**Step 4: k-space Interpolation**\nFill R-1 = 3 missing lines per acquired line.\n\n"
                    "**Step 5: SENSE Unfolding**\nOptimal coil combination with sensitivity maps.\n\n"
                    "**Step 6: Phase Correction**\nRemove B0-induced phase offset.\n\n"
                    "### Hyperparameters\n\n- `kernel_size`: 5x4\n- `acs_lines`: 24\n- `acceleration`: 4\n\n"
                    "# Demands\n\n- **feasibility**: yes\n- **algorithm_convergence**: yes\n"
                ),
            },
        ],
    }


# ═════════════════════════════════════════════════════════════════════════════
# Example 3 — Widefield Fluorescence Microscopy
# ═════════════════════════════════════════════════════════════════════════════

def _microscopy_example() -> Example:
    return {
        "id": "widefield_fluor",
        "title": "Widefield Fluorescence Microscopy",
        "description": "LED-excited fluorescence: forward model with aberrations + Richardson-Lucy deconvolution",
        "modality": "widefield",
        "phases": [
            {
                "period": "forward",
                "rounds": [
                    {
                        "user": "Design a widefield fluorescence microscopy forward model with LED excitation and CMOS camera",
                        "assistant": (
                            "I've designed the forward model for: **Widefield fluorescence microscopy with LED + CMOS**\n\n"
                            "The signal chain has **5 elements**:\n\n"
                            "- **LED Excitation 470nm** (source) | params: power=50mW, bandwidth=25nm | mismatch: illumination_nonuniformity\n"
                            "- **Fluorescent Sample** (interaction) | excitation=470nm, emission=525nm, QY=0.8 | mismatch: photobleaching\n"
                            "- **40x 0.75NA Objective** (geometry) | params: NA=0.75, magnification=40x | mismatch: spherical_aberration\n"
                            "- **Tube Lens 200mm** (processing) | focal_length=200mm\n"
                            "- **sCMOS Camera** (detector) | noise: poisson (shot), gaussian (read=1.5e-), dark_current\n\n"
                            "Composite noise: `y ~ Poisson(PSF * x + background) + N(0, sigma_read^2)`\n"
                            "Output shape: `(2048, 2048)`"
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 87%)\n\n"
                            "Standard widefield fluorescence model. All elements present.\n\n"
                            "[WARNING] **physics**: Out-of-focus fluorescence is not modeled. For thick samples, this dominates image degradation.\n"
                            "  -> Add defocus PSF model or specify thin sample assumption"
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "This is for thin cell monolayer (~5um). Add chromatic aberration and specify the PSF as Airy disk with defocus",
                        "assistant": (
                            "I've designed the forward model for: **Updated widefield fluorescence for thin cell monolayer**\n\n"
                            "The signal chain has **5 elements**:\n\n"
                            "- **LED Excitation 470nm** (source) | mismatch: illumination_nonuniformity\n"
                            "- **Thin Cell Monolayer** (interaction) | thickness=5um, QY=0.8 | mismatch: photobleaching\n"
                            "- **40x 0.75NA Objective** (geometry) | PSF: Airy disk with defocus model | mismatch: spherical_aberration, chromatic_aberration\n"
                            "- **Tube Lens 200mm** (processing)\n"
                            "- **sCMOS Camera** (detector) | read_noise=1.5e-, pixel_size=6.5um\n\n"
                            "Added: thin sample assumption, Airy+defocus PSF, chromatic aberration (Stokes shift 470->525nm causes axial shift)."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 92%)\n\n"
                            "Good refinement. Thin sample assumption is valid for cell monolayer. Chromatic aberration from Stokes shift is correctly identified."
                        ),
                        "spec_judged": True,
                    },
                ],
                "final_plan_md": (
                    "# Task\n\nDesign widefield fluorescence microscopy for thin cell monolayer imaging.\n\n"
                    "# Plan\n\n"
                    "1. Model LED excitation at 470nm with illumination field\n"
                    "2. Compute fluorescence emission at 525nm with photobleaching\n"
                    "3. Apply Airy disk PSF with defocus through 40x 0.75NA objective\n"
                    "4. Add chromatic and spherical aberration\n"
                    "5. Simulate sCMOS detector with shot noise and read noise\n\n"
                    "# Demands\n\n- **feasibility**: yes\n"
                ),
                "final_spec_md": (
                    "# Task\n\nWidefield fluorescence for thin cell monolayer\n\n# Action\n\n"
                    "## System Flowchart\n\n```\n"
                    "[LED 470nm] -> [Cell Monolayer] -> [40x 0.75NA Obj] -> [Tube Lens] -> [sCMOS] -> y\n"
                    "```\n\n"
                    "### Element: LED 470nm (`led_source`)\n- **Type**: source\n"
                    "- **Parameters**: wavelength=470nm, power=50mW, bandwidth=25nm\n"
                    "- **Mismatch**: `illumination_nonuniformity` [medium]\n\n"
                    "### Element: Cell Monolayer (`sample`)\n- **Type**: interaction\n"
                    "- **Parameters**: thickness=5um, excitation=470nm, emission=525nm, QY=0.8\n"
                    "- **Mismatch**: `photobleaching` [low]\n\n"
                    "### Element: 40x Objective (`objective`)\n- **Type**: geometry\n"
                    "- **Parameters**: NA=0.75, magnification=40, PSF=airy_with_defocus\n"
                    "- **Mismatch**: `spherical_aberration` [medium], `chromatic_aberration` [low]\n\n"
                    "### Element: sCMOS (`camera`)\n- **Type**: detector\n"
                    "- **Noise**: poisson (shot), gaussian sigma=1.5e-\n"
                    "- **Parameters**: pixel_size=6.5um, resolution=2048x2048\n\n"
                    "**Measurement shape**: `(2048, 2048)`\n\n"
                    "# Demands\n\n- **feasibility**: yes\n"
                ),
            },
            {
                "period": "reconstruction",
                "rounds": [
                    {
                        "user": "Design a Richardson-Lucy deconvolution for this widefield microscopy data",
                        "assistant": (
                            "I've designed a **Richardson-Lucy** (Classical) reconstruction plan.\n\n"
                            "**5 algorithm steps**:\n\n"
                            "1. **Flat-Field Correction** - Divide by illumination reference to correct nonuniformity\n"
                            "2. **Background Subtraction** - Estimate and subtract autofluorescence background\n"
                            "3. **PSF Estimation** - Compute theoretical Airy+defocus PSF from NA=0.75 at 525nm\n"
                            "4. **RL Iteration** - x_{k+1} = x_k * (PSF^T * (y / (PSF * x_k + eps)))\n"
                            "5. **Positivity Constraint** - Ensure x >= 0 (inherent in RL)\n\n"
                            "**2 mismatch corrections**:\n"
                            "- illumination_nonuniformity: Flat-field correction\n"
                            "- spherical_aberration: Included in measured/theoretical PSF\n\n"
                            "Convergence: KL divergence < 1e-6 or max 50 iterations"
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 90%)\n\n"
                            "RL deconvolution is the standard choice for fluorescence microscopy. Positivity is inherently maintained.\n\n"
                            "[WARNING] **algorithm**: RL amplifies noise after too many iterations. Add regularization or early stopping.\n"
                            "  -> Consider TV-regularized RL or monitor noise amplification"
                        ),
                        "spec_judged": True,
                    },
                    {
                        "user": "Add TV regularization to prevent noise amplification and set the regularization strength to 0.001",
                        "assistant": (
                            "I've designed a **TV-Regularized Richardson-Lucy** (Variational) reconstruction plan.\n\n"
                            "**6 algorithm steps**:\n\n"
                            "1. **Flat-Field Correction** - Divide by illumination reference\n"
                            "2. **Background Subtraction** - Subtract autofluorescence estimate\n"
                            "3. **PSF Estimation** - Theoretical Airy+defocus PSF (NA=0.75, 525nm)\n"
                            "4. **RL Update** - x_{k+1} = x_k * (PSF^T * (y / (PSF * x_k + eps)))\n"
                            "5. **TV Denoising** - x_{k+1} = prox_{lambda*TV}(x_{k+1}), lambda=0.001\n"
                            "6. **Positivity Projection** - x_{k+1} = max(x_{k+1}, 0)\n\n"
                            "Changes: Added TV proximal step after RL update. This prevents noise amplification while preserving edges."
                        ),
                        "judge": (
                            "**Judge Verdict**: PASS (confidence: 95%)\n\n"
                            "TV-RL is an excellent combination for fluorescence. lambda=0.001 provides moderate regularization suitable for cell imaging."
                        ),
                        "spec_judged": True,
                    },
                ],
                "final_plan_md": (
                    "# Task\n\nDeconvolve widefield fluorescence images to recover sharp cell structures.\n\n"
                    "# Plan\n\n"
                    "1. Apply flat-field correction for illumination uniformity\n"
                    "2. Subtract background autofluorescence\n"
                    "3. Estimate PSF from objective parameters\n"
                    "4. Run TV-regularized Richardson-Lucy deconvolution\n"
                    "5. Monitor convergence via KL divergence\n\n"
                    "# Demands\n\n- **feasibility**: yes\n- **algorithm_convergence**: yes\n"
                ),
                "final_spec_md": (
                    "# Task\n\nDeconvolve widefield fluorescence for cell monolayer.\n\n"
                    "# Action\n\n## Algorithm: TV-Regularized Richardson-Lucy\n\n"
                    "**Type**: Variational\n\n"
                    "### Algorithm Steps\n\n"
                    "**Step 1: Flat-Field Correction**\ny_corr = y / flat_field\n\n"
                    "**Step 2: Background Subtraction**\ny_clean = y_corr - bg_estimate\n\n"
                    "**Step 3: PSF Estimation**\nTheoretical Airy+defocus, NA=0.75, lambda_em=525nm\n\n"
                    "**Step 4: RL Update**\n$$\nx_{k+1} = x_k * (h^T * (y / (h * x_k + eps)))\n$$\n\n"
                    "**Step 5: TV Denoising**\n$$\nx_{k+1} = prox_{lambda*TV}(x_{k+1})\n$$\nlambda=0.001\n\n"
                    "**Step 6: Positivity**\nx_{k+1} = max(x_{k+1}, 0)\n\n"
                    "### Mismatch Corrections\n\n"
                    "- `illumination_nonuniformity` [medium]: Flat-field correction\n"
                    "- `spherical_aberration` [medium]: Included in PSF model\n\n"
                    "**Convergence**: KL divergence < 1e-6 or max 50 iterations\n\n"
                    "### Hyperparameters\n\n- `lambda_tv`: 0.001\n- `max_iterations`: 50\n- `epsilon`: 1e-10\n\n"
                    "# Demands\n\n- **feasibility**: yes\n- **algorithm_convergence**: yes\n"
                ),
            },
        ],
    }
