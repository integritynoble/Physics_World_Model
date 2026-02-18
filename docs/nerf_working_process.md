# NeRF Working Process

## End-to-End Pipeline for Neural Radiance Fields

This document traces a complete NeRF experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a 3D scene from multi-view photographs and render novel views.
 Images: images/ (10 views, 128x128), Poses: poses.npy, focal_length=111 px."
```

---

## 2. PlanAgent --- Intent Parsing & Modality Mapping

**File:** `agents/plan_agent.py` (1,076 lines)

### Step 2a: Parse Intent -> `PlanIntent`

```python
intent = plan_agent.parse_intent(prompt)
# PlanIntent(
#   mode=ModeRequested.auto,
#   has_measured_y=True,         # "images/" detected
#   operator_type=OperatorType.nonlinear_operator,
#   files=["images/", "poses.npy"],
#   params={"n_views": 10, "image_size": [128, 128], "focal_length_px": 111}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> nerf entry
nerf:
  keywords: [NeRF, neural_radiance, volumetric_rendering, novel_view, 3D_reconstruction]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="nerf",
#   confidence=0.94,
#   reasoning="Matched keywords: NeRF, novel_view, 3D_reconstruction"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the NeRF registry entry:

```python
system = plan_agent.build_imaging_system("nerf")
# ImagingSystem(
#   modality_key="nerf",
#   display_name="Neural Radiance Fields (NeRF)",
#   signal_dims={"x": [128, 128, 64], "y": [10, 128, 128, 3]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...5 elements...],
#   default_solver="nerf_mlp"
# )
```

**NeRF Element Chain (5 elements):**

```
Multi-View Camera Rig ---> Camera Lens ---> 3D Scene Volume ---> Volume Rendering ---> RGB Image Sensor
  throughput=1.0            throughput=0.90   throughput=1.0      throughput=1.0        throughput=0.85
  noise: none               noise: aberration  noise: none         noise: none           noise: shot+read+quant
  n_views=10                f=50mm, f/2.0     volume=[128,128,64] n_samples=64          8-bit, gamma=2.2
  FOV=60 deg                pinhole model      pos_encoding=10     importance=128        RGB, 3 channels
  coverage=360 deg
```

**Cumulative throughput:** `1.0 x 0.90 x 1.0 x 1.0 x 0.85 = 0.765`

**Forward model:** `C(r) = integral_0^inf T(t) * sigma(r(t)) * c(r(t), d) dt`

where T(t) = exp(-integral_0^t sigma(s) ds) is the accumulated transmittance, sigma is the volume density, c is the view-dependent color, and the integral is along each camera ray r.

---

## 3. PhotonAgent --- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  nerf:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+09
      qe: 0.95
      exposure_s: 0.016
  ```

### Computation

```python
# 1. Source photon count (ambient illumination, typical daylight)
N_source = 1.0e9 photons/frame

# 2. Apply cumulative throughput
N_effective = N_source * 0.765 = 7.65e8 photons/frame

# 3. Photons per pixel per color channel
# 128 x 128 image, 3 color channels (Bayer pattern)
N_per_pixel = N_effective / (128 * 128 * 3) = 15,564 photons/pixel/channel

# 4. Noise variances (consumer camera sensor)
shot_var   = N_per_pixel = 15564                # Poisson
read_var   = 3.0^2 = 9.0                        # Low-noise CMOS
dark_var   = 0.0                                 # Short exposure
total_var  = shot_var + read_var = 15573

# 5. SNR per pixel
SNR = N_per_pixel / sqrt(total_var) = 15564 / 124.8 = 124.7
SNR_db = 20 * log10(124.7) = 41.9 dB

# 6. After 8-bit quantization (gamma-corrected):
# Quantization noise: sigma_q = 1/(sqrt(12) * 255) ~ 0.00113
# This is negligible compared to shot noise in linear domain
# But 8-bit clipping limits dynamic range to 48 dB

# 7. Multi-view redundancy
# 10 views -> each 3D point seen from ~3-5 views on average
# Redundancy gain: sqrt(4) = 2x
SNR_effective_db = 41.9 + 20*log10(2) = 41.9 + 6.0 = 47.9 dB
```

### Output -> `PhotonReport`

```python
PhotonReport(
  n_photons_per_pixel=15564,
  snr_db=47.9,
  noise_regime=NoiseRegime.shot_limited,    # shot_var/total_var > 0.99
  shot_noise_sigma=124.7,
  read_noise_sigma=3.0,
  total_noise_sigma=124.8,
  feasible=True,
  quality_tier="excellent",                  # SNR > 30 dB
  throughput_chain=[
    {"Multi-View Camera Rig": 1.0},
    {"Camera Lens": 0.90},
    {"3D Scene Volume": 1.0},
    {"Volume Rendering": 1.0},
    {"RGB Image Sensor": 0.85}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime. Consumer cameras provide excellent photon budget. "
              "Multi-view redundancy adds ~6 dB. 8-bit quantization is the limiting factor."
)
```

---

## 4. MismatchAgent --- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"nerf"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  nerf:
    parameters:
      camera_pose_error:
        range: [0.0, 5.0]
        typical_error: 0.5
        unit: "pixels (reprojection)"
        description: "Camera extrinsic error from SfM/COLMAP pose estimation"
      focal_length_error:
        range: [-50.0, 50.0]
        typical_error: 10.0
        unit: "pixels"
        description: "Intrinsic focal length error from imprecise calibration"
    severity_weights:
      camera_pose_error: 0.60
      focal_length_error: 0.40
    correction_method: "gradient_descent"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.60 * |0.5| / 5.0       # pose:  0.060
  + 0.40 * |10.0| / 100.0    # focal: 0.040
S = 0.100  # Low severity (good SfM pipeline)

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.00 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="nerf",
  mismatch_family="gradient_descent",
  parameters={
    "camera_pose_error":   {"typical_error": 0.5, "range": [0, 5], "weight": 0.60},
    "focal_length_error":  {"typical_error": 10.0, "range": [-50, 50], "weight": 0.40}
  },
  severity_score=0.100,
  correction_method="gradient_descent",
  expected_improvement_db=1.00,
  explanation="Low mismatch severity. COLMAP provides sub-pixel pose accuracy. "
              "NeRF can jointly optimize poses during training (BARF-style)."
)
```

---

## 5. RecoverabilityAgent --- Can We Reconstruct?

**File:** `agents/recoverability_agent.py` (912 lines)

### Input
- `ImagingSystem` (signal_dims for CR calculation)
- `PhotonReport` (noise regime)
- Calibration table from `compression_db.yaml`:
  ```yaml
  nerf:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 0.50, noise: "shot_limited", solver: "siren",
         recoverability: 0.96, expected_psnr_db: 61.2,
         provenance: {dataset_id: "nerf_synthetic_blender_2023", ...}}
      - {cr: 0.25, noise: "shot_limited", solver: "siren",
         recoverability: 0.89, expected_psnr_db: 52.4, ...}
      - {cr: 0.10, noise: "shot_limited", solver: "siren",
         recoverability: 0.72, expected_psnr_db: 38.7, ...}
  ```

### Computation

```python
# 1. Compression ratio
# Measurements: 10 views x 128 x 128 x 3 = 491,520 pixel values
# Scene: 128 x 128 x 64 volume (density + 3-channel color)
# Signal DOF: 128 * 128 * 64 * 4 = 4,194,304
# CR = measurements / signal_DOF
CR = 491520 / 4194304 = 0.117
# Closest calibration entry: cr=0.10 (sparse views)

# 2. Operator diversity (multi-view angular coverage)
# 10 views over 360 degrees -> 36 deg angular spacing
# Good coverage but sparse
diversity = min(10 / 20, 1.0) = 0.50  # reference: 20 views = good

# 3. Condition number proxy
# Sparse views -> some viewing directions unobserved
kappa = 1 / (1 + diversity) = 0.667

# 4. Calibration table lookup (interpolating between cr=0.10 and cr=0.25)
#    cr=0.117 -> linear interpolation
#    recoverability = 0.72 + (0.89 - 0.72) * (0.117 - 0.10) / (0.25 - 0.10)
#                   = 0.72 + 0.17 * 0.113 = 0.739
#    expected_psnr = 38.7 + (52.4 - 38.7) * 0.113 = 40.2 dB

# 5. Best solver: "siren" (SIREN positional encoding MLP)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.117,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.50,
  condition_number_proxy=0.667,
  recoverability_score=0.739,
  recoverability_confidence=0.90,
  expected_psnr_db=40.2,
  expected_psnr_uncertainty_db=3.0,
  recommended_solver_family="siren",
  verdict="sufficient",              # 0.60 <= score < 0.85
  calibration_table_entry={...},
  explanation="Sufficient recoverability. 10 views is sparse; expect floater "
              "artifacts in unobserved regions. More views would improve quality."
)
```

---

## 6. AnalysisAgent --- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(47.9 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.100                        = 0.100   # Low mismatch
compression_score = 1 - 0.739                    = 0.261   # Moderate (sparse views)
solver_score      = 0.15                         = 0.15    # SIREN well-characterized

# Primary bottleneck
primary = "compression"  # max(0.0, 0.100, 0.261, 0.15) = compression

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.100*0.5) * (1 - 0.261*0.5) * (1 - 0.15*0.5)
  = 1.0 * 0.95 * 0.870 * 0.925
  = 0.764
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="compression",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.100, compression=0.261, solver=0.15
  ),
  suggestions=[
    Suggestion(
      text="Add more training views (20+ recommended for full 360 coverage)",
      priority="high",
      expected_gain_db=12.0
    ),
    Suggestion(
      text="Use Instant-NGP hash encoding for 100x faster training",
      priority="medium",
      expected_gain_db=0.0
    ),
    Suggestion(
      text="Apply depth regularization to reduce floaters in sparse regions",
      priority="medium",
      expected_gain_db=2.0
    )
  ],
  overall_verdict="sufficient",      # 0.60 <= P < 0.80
  probability_of_success=0.764,
  explanation="Sparse view coverage (10 views) is the primary bottleneck. "
              "3D scene has insufficient multi-view constraints for full recovery."
)
```

---

## 7. AgentNegotiator --- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" BUT verdict="sufficient" | No veto |
| Severe mismatch without correction | severity=0.100 < 0.7 | No veto |
| All marginal | photon=excellent, others mixed | No veto |
| Joint probability floor | P=0.764 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.739   # recoverability_score
P_mismatch       = 1.0 - 0.100 * 0.7 = 0.930

P_joint = 0.95 * 0.739 * 0.930 = 0.653
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],               # No vetoes
  proceed=True,
  probability_of_success=0.653
)
```

---

## 8. PreFlightReportBuilder --- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
total_pixels = 128 * 128 * 64 = 1,048,576  # volume voxels
dim_factor   = total_pixels / (256 * 256) = 16.0
solver_complexity = 5.0  # SIREN MLP (1000 iterations, ray sampling)
n_views = 10
n_iterations = 1000

# NeRF training is computationally intensive
runtime_s = 2.0 * dim_factor * solver_complexity * 0.125 * (n_iterations / 100)
          = 2.0 * 16.0 * 5.0 * 0.125 * 10.0
          = 200.0 seconds  # ~3.3 minutes (GPU)
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="nerf", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=200.0,
  proceed_recommended=True,
  warnings=[
    "Sparse views (10) may produce floater artifacts in unobserved regions"
  ],
  what_to_upload=[
    "multi-view RGB images ([N_views, H, W, 3])",
    "camera poses as 4x4 matrices ([N_views, 4, 4])",
    "focal length in pixels (float)"
  ]
)
```

---

## 9. Pipeline Execution

**File:** `core/runner.py`

After the pre-flight report is approved (interactively or auto), the pipeline runner executes:

### Step 9a: Build Physics Operator

```python
# NeRF forward model (volume rendering equation):
#   C(r) = integral_0^inf T(t) * sigma(r(t)) * c(r(t), d) dt
#   T(t) = exp(-integral_0^t sigma(r(s)) ds)
#
# Discretized with quadrature (N samples along each ray):
#   C_hat = sum_{i=1}^{N} T_i * alpha_i * c_i
#   T_i = prod_{j=1}^{i-1} (1 - alpha_j)
#   alpha_i = 1 - exp(-sigma_i * delta_i)
#
# Parameters:
#   poses:        (10, 4, 4) camera-to-world matrices
#   focal_length: 111 pixels
#   near, far:    0.1, 10.0 scene units
#   n_samples:    64 (coarse) + 128 (fine, importance sampling)
#
# Input:  scene = volumetric function (x,y,z,d) -> (sigma, r, g, b)
# Output: images = (10, 128, 128, 3) rendered RGB images

class NeRFOperator(PhysicsOperator):
    def forward(self, mlp):
        """Render all training views via volume rendering"""
        images = np.zeros((n_views, H, W, 3))
        for v in range(n_views):
            rays_o, rays_d = get_rays(poses[v], focal, H, W)
            for i in range(H):
                for j in range(W):
                    # Sample points along ray
                    t = np.linspace(near, far, n_coarse)
                    pts = rays_o[i,j] + rays_d[i,j] * t[:, None]
                    d_view = rays_d[i,j] / np.linalg.norm(rays_d[i,j])

                    # Query MLP: (x,y,z,d) -> (sigma, rgb)
                    sigma, rgb = mlp(pts, d_view)

                    # Volume rendering quadrature
                    delta = t[1:] - t[:-1]
                    alpha = 1 - np.exp(-sigma[:-1] * delta)
                    T = np.cumprod(np.concatenate([[1.0], 1 - alpha]))
                    weights = T[:-1] * alpha
                    images[v, i, j] = np.sum(weights[:, None] * rgb[:-1], axis=0)
        return images

    def adjoint(self, residuals):
        """Gradient of rendering loss w.r.t. MLP parameters"""
        # Computed via automatic differentiation (PyTorch autograd)
        # Not an explicit adjoint -- backpropagation through the volume renderer
        pass

    def check_adjoint(self):
        """Nonlinear operator: gradient consistency check"""
        # Verifies torch.autograd.gradcheck passes
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-3)
```

### Step 9b: Load Training Data

```python
# Load multi-view images
import glob
from PIL import Image

image_files = sorted(glob.glob("images/*.png"))
images = np.stack([np.array(Image.open(f)) for f in image_files])
images = images.astype(np.float32) / 255.0  # (10, 128, 128, 3)

# Load camera poses
poses = np.load("poses.npy")  # (10, 4, 4) camera-to-world matrices
focal = 111.0                  # focal length in pixels
```

### Step 9c: Reconstruction (Neural Network Optimization)

**SIREN-based Neural Implicit Representation:**

```python
import torch
import torch.nn as nn

class SirenLayer(nn.Module):
    def __init__(self, in_features, out_features, omega_0=30, is_first=False):
        super().__init__()
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features)
        # SIREN initialization (Sitzmann et al. 2020)
        with torch.no_grad():
            if is_first:
                self.linear.weight.uniform_(-1/in_features, 1/in_features)
            else:
                bound = np.sqrt(6/in_features) / omega_0
                self.linear.weight.uniform_(-bound, bound)

    def forward(self, x):
        return torch.sin(self.omega_0 * self.linear(x))

class NeuralRadianceField(nn.Module):
    def __init__(self, input_dim=256, hidden_dim=256, n_layers=4):
        super().__init__()
        # Positional encoding: gamma(p) = (sin(2^k*pi*p), cos(2^k*pi*p))
        # L=10 levels -> input_dim = 3*2*10 = 60 (position)
        #              + 3*2*4 = 24 (direction) = 84 total
        self.pos_enc_levels = 10
        self.dir_enc_levels = 4

        # Density network (position only)
        layers = [SirenLayer(60, hidden_dim, is_first=True)]
        for _ in range(n_layers - 1):
            layers.append(SirenLayer(hidden_dim, hidden_dim))
        self.density_net = nn.Sequential(*layers)
        self.sigma_head = nn.Linear(hidden_dim, 1)

        # Color network (position features + direction)
        self.color_net = nn.Sequential(
            nn.Linear(hidden_dim + 24, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 3),
            nn.Sigmoid()
        )

    def forward(self, pts, dirs):
        # Positional encoding
        pts_enc = positional_encoding(pts, self.pos_enc_levels)   # (N, 60)
        dirs_enc = positional_encoding(dirs, self.dir_enc_levels) # (N, 24)

        # Density
        h = self.density_net(pts_enc)
        sigma = torch.relu(self.sigma_head(h))  # non-negative density

        # Color (view-dependent)
        color = self.color_net(torch.cat([h, dirs_enc], dim=-1))

        return sigma, color
```

**Training loop:**

```python
model = NeuralRadianceField().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

n_iters = 1000
batch_rays = 1024  # rays per iteration

for it in range(n_iters):
    # Sample random batch of rays from training views
    view_idx = np.random.randint(0, n_views)
    pixel_idx = np.random.choice(H*W, batch_rays, replace=False)

    rays_o, rays_d = get_rays(poses[view_idx], focal, H, W)
    target_rgb = images[view_idx].reshape(-1, 3)[pixel_idx]

    # Render rays
    rgb_pred = render_rays(model, rays_o.reshape(-1,3)[pixel_idx],
                           rays_d.reshape(-1,3)[pixel_idx],
                           near=0.1, far=10.0, n_samples=64)

    # Photometric loss
    loss = ((rgb_pred - target_rgb)**2).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# Expected PSNR on training views: ~40.2 dB
# Expected PSNR on held-out views: ~32.0 dB (reference benchmark)
```

**Alternative solvers:**

| Solver | Type | PSNR (novel view) | GPU | Training Time |
|--------|------|-------------------|-----|---------------|
| SIREN MLP | Neural Implicit | 32.0 dB | Yes | ~200s |
| Instant-NGP | Hash Grid | 33.5 dB | Yes | ~20s |
| Mip-NeRF 360 | Anti-aliased | 34.2 dB | Yes | ~600s |

### Step 9d: Metrics

```python
# Per-view PSNR (on held-out test views)
for v in test_views:
    rendered = render_full_image(model, poses[v], focal, H, W)
    psnr_v = 10 * np.log10(1.0 / np.mean((rendered - images[v])**2))

# Average PSNR across test views
avg_psnr = np.mean(psnr_per_view)  # ~32.0 dB (benchmark)

# SSIM per view
avg_ssim = np.mean([ssim(rendered_v, gt_v) for v in test_views])

# LPIPS (Learned Perceptual Image Patch Similarity) -- NeRF-specific
# Measures perceptual quality using VGG features
avg_lpips = np.mean([lpips(rendered_v, gt_v) for v in test_views])

# Depth map quality (if ground truth depth available)
depth_rmse = np.sqrt(np.mean((depth_rendered - depth_gt)**2))
```

### Step 9e: RunBundle Output

```
run_bundle/
+-- meta.json              # ExperimentSpec + provenance
+-- agent_reports/
|   +-- photon_report.json
|   +-- mismatch_report.json
|   +-- recoverability_report.json
|   +-- system_analysis.json
|   +-- negotiation_result.json
|   +-- preflight_report.json
+-- arrays/
|   +-- images_train.npy   # Training images (10, 128, 128, 3) + SHA256
|   +-- poses.npy          # Camera poses (10, 4, 4) + SHA256
|   +-- novel_views.npy    # Rendered novel views (N, 128, 128, 3) + SHA256
|   +-- depth_maps.npy     # Rendered depth maps (N, 128, 128) + SHA256
+-- model/
|   +-- nerf_weights.pth   # Trained MLP weights + SHA256
+-- metrics.json           # PSNR, SSIM, LPIPS per view + average
+-- operator.json          # Operator params (poses hash, focal, near/far)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (SIREN -> Instant-NGP -> Mip-NeRF) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **NeRF-specific:** Nonlinear volumetric forward model, view-dependent color, positional encoding, LPIPS perceptual metric, training-time optimization instead of single-pass inference.
