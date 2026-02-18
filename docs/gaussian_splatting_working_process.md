# Gaussian Splatting Working Process

## End-to-End Pipeline for 3D Gaussian Splatting

This document traces a complete 3D Gaussian Splatting (3DGS) experiment through the PWM multi-agent system, from user prompt to final RunBundle output.

---

## 1. User Prompt

```
"Reconstruct a 3D scene using Gaussian splatting from multi-view images.
 Images: images/ (10 views, 128x128), Poses: poses.npy,
 Point cloud: sparse_points.ply, focal_length=111 px."
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
#   files=["images/", "poses.npy", "sparse_points.ply"],
#   params={"n_views": 10, "image_size": [128, 128], "focal_length_px": 111}
# )
```

### Step 2b: Keyword Match -> `ModalitySelection`

The `_keyword_match()` method scans the prompt against each modality's keyword list from `modalities.yaml`:

```yaml
# modalities.yaml -> gaussian_splatting entry
gaussian_splatting:
  keywords: [gaussian_splatting, 3DGS, real_time, novel_view, point_cloud]
```

```python
modality = plan_agent.map_prompt_to_modality(prompt)
# ModalitySelection(
#   modality_key="gaussian_splatting",
#   confidence=0.97,
#   reasoning="Matched keywords: gaussian_splatting, 3DGS, point_cloud"
# )
```

If keyword match fails (confidence < 0.6), falls back to LLM semantic matching.

### Step 2c: Build Imaging System -> `ImagingSystem`

Constructs the element chain from the Gaussian splatting registry entry:

```python
system = plan_agent.build_imaging_system("gaussian_splatting")
# ImagingSystem(
#   modality_key="gaussian_splatting",
#   display_name="3D Gaussian Splatting",
#   signal_dims={"x": [128, 128, 64], "y": [10, 128, 128, 3]},
#   forward_model_type=ForwardModelType.nonlinear_operator,
#   elements=[...5 elements...],
#   default_solver="gaussian_splatting_3dgs"
# )
```

**3DGS Element Chain (5 elements):**

```
Multi-View Camera Rig ---> Camera Lens ---> 3D Gaussian Cloud ---> Tile-Based Rasterizer ---> RGB Image Sensor
  throughput=1.0            throughput=0.90   throughput=1.0         throughput=1.0              throughput=0.85
  noise: none               noise: aberration  noise: none            noise: none                 noise: shot+read+quant
  n_views=10                f=50mm             N_init=100k            tile=16px                   8-bit, gamma=2.2
  FOV=60 deg                pinhole            SH_degree=3            radix sort                  RGB, 3 channels
                                               opacity=sigmoid        alpha_compositing
                                               scale=exp              front_to_back
```

**Cumulative throughput:** `1.0 x 0.90 x 1.0 x 1.0 x 0.85 = 0.765`

**Forward model (EWA splatting + alpha compositing):**

```
C(p) = sum_i alpha_i * c_i * prod_{j<i}(1 - alpha_j)
alpha_i = o_i * G_2d(p; mu_i, Sigma_i)
```

where each 3D Gaussian is defined by mean mu_i, 3D covariance Sigma_i (parameterized via quaternion rotation + scale), opacity o_i, and spherical harmonics color coefficients. The 3D Gaussians are projected to 2D via the EWA splatting equation:

```
Sigma_2d = J * W * Sigma_3d * W^T * J^T
```

where W is the world-to-camera transform and J is the Jacobian of the projective mapping.

---

## 3. PhotonAgent --- SNR & Noise Analysis

**File:** `agents/photon_agent.py` (872 lines)

### Input
- `ImagingSystem` (element chain with throughputs + noise kinds)
- Photon model from `photon_db.yaml`:
  ```yaml
  gaussian_splatting:
    model_id: "generic_detector"
    parameters:
      source_photons: 1.0e+09
      qe: 0.95
      exposure_s: 0.016
  ```

### Computation

```python
# 1. Source photon count (ambient illumination, daylight)
N_source = 1.0e9 photons/frame

# 2. Apply cumulative throughput
N_effective = N_source * 0.765 = 7.65e8 photons/frame

# 3. Photons per pixel per color channel
# 128 x 128 image, 3 color channels
N_per_pixel = N_effective / (128 * 128 * 3) = 15,564 photons/pixel/channel

# 4. Noise variances
shot_var   = N_per_pixel = 15564                # Poisson
read_var   = 3.0^2 = 9.0                        # Low-noise CMOS
dark_var   = 0.0                                 # Short exposure
total_var  = shot_var + read_var = 15573

# 5. SNR per pixel
SNR = N_per_pixel / sqrt(total_var) = 15564 / 124.8 = 124.7
SNR_db = 20 * log10(124.7) = 41.9 dB

# 6. Multi-view redundancy (same as NeRF)
# 10 views with average ~4x coverage per point
SNR_effective_db = 41.9 + 20*log10(2.0) = 47.9 dB
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
    {"3D Gaussian Cloud": 1.0},
    {"Tile-Based Rasterizer": 1.0},
    {"RGB Image Sensor": 0.85}
  ],
  noise_model="poisson",
  explanation="Shot-limited regime. Consumer camera images provide excellent photon "
              "budget. Identical photon model to NeRF (same camera hardware)."
)
```

---

## 4. MismatchAgent --- Operator Mismatch Severity

**File:** `agents/mismatch_agent.py` (422 lines)

### Input
- Modality key: `"gaussian_splatting"`
- Mismatch spec from `mismatch_db.yaml`:
  ```yaml
  gaussian_splatting:
    parameters:
      position_noise:
        range: [0.0, 0.1]
        typical_error: 0.02
        unit: "scene_units"
        description: "SfM point cloud position noise propagated to Gaussian centers"
      scale_error:
        range: [0.5, 2.0]
        typical_error: 0.15
        unit: "unitless"
        description: "Initial Gaussian scale error from sparse point density"
    severity_weights:
      position_noise: 0.55
      scale_error: 0.45
    correction_method: "gradient_descent"
  ```

### Computation

```python
# Severity score (weighted normalized errors)
S = 0.55 * |0.02| / 0.1     # position_noise: 0.110
  + 0.45 * |0.15| / 1.5     # scale_error:    0.045
S = 0.155  # Moderate-low severity

# Expected improvement from correction
improvement_db = clip(10 * S, 0, 20) = 1.55 dB
```

### Output -> `MismatchReport`

```python
MismatchReport(
  modality_key="gaussian_splatting",
  mismatch_family="gradient_descent",
  parameters={
    "position_noise": {"typical_error": 0.02, "range": [0.0, 0.1], "weight": 0.55},
    "scale_error":    {"typical_error": 0.15, "range": [0.5, 2.0], "weight": 0.45}
  },
  severity_score=0.155,
  correction_method="gradient_descent",
  expected_improvement_db=1.55,
  explanation="Moderate-low mismatch severity. SfM point noise is the dominant source. "
              "3DGS optimization inherently corrects Gaussian positions via gradient descent "
              "during training, making it self-correcting for moderate mismatch."
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
  gaussian_splatting:
    signal_prior_class: "deep_prior"
    entries:
      - {cr: 0.50, noise: "shot_limited", solver: "gaussian_opt",
         recoverability: 0.85, expected_psnr_db: 30.1,
         provenance: {dataset_id: "mipnerf360_outdoor_2023", ...}}
      - {cr: 0.25, noise: "shot_limited", solver: "gaussian_opt",
         recoverability: 0.72, expected_psnr_db: 26.3, ...}
      - {cr: 1.0, noise: "shot_limited", solver: "gaussian_opt",
         recoverability: 0.93, expected_psnr_db: 33.8, ...}
  ```

### Computation

```python
# 1. Compression ratio
# 3DGS signal: adaptive number of Gaussians (up to 5M)
# Each Gaussian: 3 pos + 4 quat + 3 scale + 1 opacity + 48 SH = 59 params
# With N_init=100k: 100000 * 59 = 5,900,000 parameters
# Measurements: 10 * 128 * 128 * 3 = 491,520 pixel values
# CR = measurements / parameters = 0.083
# But 3DGS adaptively prunes and densifies, so effective CR varies
# For calibration: use view fraction = 10 / N_total_views
# Assume 20 views would be "full" -> CR = 10/20 = 0.50

# 2. Operator diversity
# 10 views over 360 degrees, same as NeRF
diversity = 0.50  # Moderate coverage

# 3. Condition number proxy
kappa = 1 / (1 + diversity) = 0.667

# 4. Calibration table lookup
#    Match: noise="shot_limited", solver="gaussian_opt", cr=0.50
#    -> recoverability=0.85, expected_psnr=30.1 dB, confidence=1.0

# 5. Best solver: "gaussian_opt" (differentiable 3DGS optimization)
```

### Output -> `RecoverabilityReport`

```python
RecoverabilityReport(
  compression_ratio=0.50,
  noise_regime=NoiseRegime.shot_limited,
  signal_prior_class=SignalPriorClass.deep_prior,
  operator_diversity_score=0.50,
  condition_number_proxy=0.667,
  recoverability_score=0.85,
  recoverability_confidence=1.0,
  expected_psnr_db=30.1,
  expected_psnr_uncertainty_db=2.0,
  recommended_solver_family="gaussian_opt",
  verdict="excellent",              # score >= 0.85
  calibration_table_entry={...},
  explanation="Excellent recoverability. Sparse SfM point cloud provides good "
              "initialization for Gaussian centers. Expected 30.1 dB on MipNeRF360 benchmark."
)
```

---

## 6. AnalysisAgent --- Bottleneck Classification

**File:** `agents/analysis_agent.py` (489 lines)

### Computation

```python
# Bottleneck scores (lower = better)
photon_score      = 1 - min(47.9 / 40, 1.0)   = 0.0     # Excellent SNR
mismatch_score    = 0.155                        = 0.155   # Moderate-low
compression_score = 1 - 0.85                     = 0.15    # Good recoverability
solver_score      = 0.20                         = 0.20    # 3DGS relatively new

# Primary bottleneck
primary = "solver"  # max(0.0, 0.155, 0.15, 0.20) = solver

# Probability of success
P = (1 - 0.0*0.5) * (1 - 0.155*0.5) * (1 - 0.15*0.5) * (1 - 0.20*0.5)
  = 1.0 * 0.923 * 0.925 * 0.90
  = 0.768
```

### Output -> `SystemAnalysis`

```python
SystemAnalysis(
  primary_bottleneck="solver",
  bottleneck_scores=BottleneckScores(
    photon=0.0, mismatch=0.155, compression=0.15, solver=0.20
  ),
  suggestions=[
    Suggestion(
      text="Add more training views for better Gaussian densification coverage",
      priority="high",
      expected_gain_db=3.7
    ),
    Suggestion(
      text="Enable adaptive density control (clone + split + prune) for sharp edges",
      priority="medium",
      expected_gain_db=2.0
    ),
    Suggestion(
      text="Use anti-aliased 3DGS (Mip-Splatting) for multi-scale consistency",
      priority="low",
      expected_gain_db=1.0
    )
  ],
  overall_verdict="sufficient",      # 0.60 <= P < 0.80
  probability_of_success=0.768,
  explanation="System is adequate. Solver maturity is the marginal bottleneck; "
              "3DGS optimization requires careful hyperparameter tuning for "
              "densification thresholds and learning rate schedules."
)
```

---

## 7. AgentNegotiator --- Cross-Agent Veto

**File:** `agents/negotiator.py` (349 lines)

### Veto Conditions Checked

| Condition | Check | Result |
|-----------|-------|--------|
| Low photon + high compression | quality="excellent" AND verdict="excellent" | No veto |
| Severe mismatch without correction | severity=0.155 < 0.7 | No veto |
| All marginal | photon=excellent, others mixed | No veto |
| Joint probability floor | P=0.768 > 0.15 | No veto |

### Joint Probability

```python
P_photon         = 0.95    # tier_prob["excellent"]
P_recoverability = 0.85    # recoverability_score
P_mismatch       = 1.0 - 0.155 * 0.7 = 0.892

P_joint = 0.95 * 0.85 * 0.892 = 0.720
```

### Output -> `NegotiationResult`

```python
NegotiationResult(
  vetoes=[],               # No vetoes
  proceed=True,
  probability_of_success=0.720
)
```

---

## 8. PreFlightReportBuilder --- Final Gate

**File:** `agents/preflight.py` (645 lines)

### Runtime Estimate

```python
# 3DGS is faster than NeRF due to rasterization (not ray marching)
total_pixels = 128 * 128 = 16,384  # per view
n_views = 10
n_gaussians = 100000  # initial, grows during densification
n_iterations = 30000  # standard 3DGS training schedule

# Rasterization is much faster than ray marching
# ~150 FPS rendering -> training limited by backward pass
runtime_s = n_iterations * 0.003  # ~3ms per iteration on GPU
          = 90.0 seconds  # ~1.5 minutes
```

### Output -> `PreFlightReport`

```python
PreFlightReport(
  modality=ModalitySelection(modality_key="gaussian_splatting", ...),
  system=ImagingSystem(...),
  photon=PhotonReport(...),
  mismatch=MismatchReport(...),
  recoverability=RecoverabilityReport(...),
  analysis=SystemAnalysis(...),
  estimated_runtime_s=90.0,
  proceed_recommended=True,
  warnings=[],
  what_to_upload=[
    "multi-view RGB images ([N_views, H, W, 3])",
    "camera poses as 4x4 matrices ([N_views, 4, 4])",
    "sparse SfM point cloud (.ply, for Gaussian initialization)",
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
# 3DGS forward model (differentiable rasterization):
#
# Each Gaussian i is parameterized by:
#   mu_i:    (3,) position in world coordinates
#   q_i:     (4,) rotation quaternion -> R_i = quat_to_rot(q_i)
#   s_i:     (3,) log-scale -> S_i = diag(exp(s_i))
#   o_i:     (1,) logit-opacity -> alpha_base = sigmoid(o_i)
#   sh_i:    (48,) spherical harmonics coefficients (degree 3, RGB)
#
# 3D covariance:  Sigma_3d = R * S * S^T * R^T
# 2D projection:  Sigma_2d = J * W * Sigma_3d * W^T * J^T
# Pixel alpha:    alpha_i(p) = alpha_base * exp(-0.5 * (p-mu_2d)^T Sigma_2d^{-1} (p-mu_2d))
# Color:          c_i(d) = SH(sh_i, d)  (view-dependent via spherical harmonics)
#
# Alpha compositing (front to back, sorted by depth):
#   C(p) = sum_i T_i * alpha_i(p) * c_i(d)
#   T_i  = prod_{j<i} (1 - alpha_j(p))
#
# Input:  Gaussian parameters {mu, q, s, o, sh} for N Gaussians
# Output: rendered images (N_views, H, W, 3)

class GaussianSplattingOperator(PhysicsOperator):
    def forward(self, gaussians, camera):
        """Render image by splatting Gaussians onto image plane"""
        H, W = camera.image_size
        image = np.zeros((H, W, 3))
        depth_order = np.argsort(gaussians.depth(camera))  # front to back

        # Tile-based rasterization (16x16 tiles)
        for tile_y in range(0, H, 16):
            for tile_x in range(0, W, 16):
                # Find Gaussians overlapping this tile
                tile_gaussians = cull_to_tile(gaussians, tile_x, tile_y, camera)

                for py in range(tile_y, min(tile_y+16, H)):
                    for px in range(tile_x, min(tile_x+16, W)):
                        T = 1.0
                        for i in depth_order:
                            if i not in tile_gaussians:
                                continue
                            # 2D Gaussian evaluation
                            alpha = gaussians.alpha_2d(i, px, py, camera)
                            color = gaussians.sh_color(i, camera.view_dir(px, py))
                            image[py, px] += T * alpha * color
                            T *= (1 - alpha)
                            if T < 0.001:
                                break  # early termination
        return image

    def adjoint(self, residuals):
        """Gradient computed via differentiable rasterization (PyTorch autograd)"""
        # Kerbl et al. SIGGRAPH 2023: custom CUDA backward pass
        pass

    def check_adjoint(self):
        """Nonlinear operator: gradient consistency check"""
        # Returns AdjointCheckReport(passed=True, max_rel_error<1e-3)
```

### Step 9b: Load Training Data and Initialize Gaussians

```python
# Load multi-view images
import glob
from PIL import Image

image_files = sorted(glob.glob("images/*.png"))
images = np.stack([np.array(Image.open(f)) for f in image_files])
images = images.astype(np.float32) / 255.0  # (10, 128, 128, 3)

# Load camera poses
poses = np.load("poses.npy")  # (10, 4, 4)
focal = 111.0

# Initialize Gaussians from SfM point cloud
import trimesh
pcd = trimesh.load("sparse_points.ply")
points = np.array(pcd.vertices)           # (N_pts, 3)
colors = np.array(pcd.visual.vertex_colors)[:, :3] / 255.0  # (N_pts, 3)

# Initialize Gaussian parameters
n_gaussians = len(points)  # typically 10k-100k from SfM
means = torch.from_numpy(points).float().to(device)       # positions
quats = torch.zeros(n_gaussians, 4, device=device)        # identity rotation
quats[:, 0] = 1.0
log_scales = torch.zeros(n_gaussians, 3, device=device) + np.log(0.001)  # small
logit_opacities = torch.zeros(n_gaussians, device=device) - 2.0          # ~0.12
sh_coeffs = torch.zeros(n_gaussians, 48, device=device)   # SH degree 3
# Initialize DC component from point colors
sh_coeffs[:, 0] = rgb_to_sh_dc(colors)

# All parameters require gradients
for p in [means, quats, log_scales, logit_opacities, sh_coeffs]:
    p.requires_grad = True
```

### Step 9c: Optimization (Training)

```python
import torch
import torch.optim as optim

# Separate learning rates per parameter group (Kerbl et al. 2023)
optimizer = optim.Adam([
    {"params": [means],            "lr": 1.6e-4 * spatial_scale},
    {"params": [quats],            "lr": 1e-3},
    {"params": [log_scales],       "lr": 5e-3},
    {"params": [logit_opacities],  "lr": 5e-2},
    {"params": [sh_coeffs],        "lr": 2.5e-3},
])

n_iterations = 30000

for it in range(n_iterations):
    # Random training view
    view_idx = np.random.randint(0, n_views)
    gt_image = images_tensor[view_idx]
    camera = cameras[view_idx]

    # Differentiable rasterization (CUDA kernel)
    rendered = rasterize_gaussians(
        means, quats, log_scales, logit_opacities, sh_coeffs,
        camera, background_color=torch.zeros(3)
    )

    # Loss: L1 + D-SSIM (Kerbl et al. 2023)
    l1_loss = torch.abs(rendered - gt_image).mean()
    ssim_loss = 1 - ssim_fn(rendered, gt_image)
    loss = 0.8 * l1_loss + 0.2 * ssim_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Adaptive density control (every 100 iterations, after warmup)
    if it >= 500 and it % 100 == 0:
        # Clone: duplicate Gaussians with large positional gradients in
        # under-reconstructed regions (small scale)
        grad_mask = means.grad.norm(dim=-1) > grad_threshold
        small_mask = torch.exp(log_scales).max(dim=-1).values < scale_threshold
        clone_mask = grad_mask & small_mask
        clone_gaussians(clone_mask)

        # Split: subdivide large Gaussians with large gradients
        large_mask = grad_mask & ~small_mask
        split_gaussians(large_mask)

        # Prune: remove near-transparent Gaussians
        prune_mask = torch.sigmoid(logit_opacities) < 0.005
        prune_gaussians(prune_mask)

    # Opacity reset (every 3000 iterations)
    if it % 3000 == 0:
        logit_opacities.data.fill_(-2.0)  # Reset to low opacity
```

### Step 9d: Rendering Novel Views

```python
# After training, render from any camera pose
novel_pose = interpolate_poses(poses[0], poses[1], t=0.5)
novel_camera = Camera(pose=novel_pose, focal=focal, H=128, W=128)

with torch.no_grad():
    novel_view = rasterize_gaussians(
        means, quats, log_scales, logit_opacities, sh_coeffs,
        novel_camera, background_color=torch.zeros(3)
    )
# Rendering speed: ~150 FPS at 128x128 (real-time capable)
# Expected PSNR on novel views: ~30.1 dB (MipNeRF360 benchmark)
```

**Alternative solvers:**

| Solver | Type | PSNR | Training | Rendering |
|--------|------|------|----------|-----------|
| 3DGS (Kerbl 2023) | Gaussian Opt | 30.1 dB | ~90s | 150 FPS |
| Mip-Splatting | Anti-aliased | 31.2 dB | ~120s | 130 FPS |
| 2DGS (Huang 2024) | Surface-based | 29.5 dB | ~100s | 140 FPS |

### Step 9e: Metrics

```python
# Per-view PSNR on held-out test views
for v in test_views:
    rendered = rasterize_gaussians(..., cameras[v])
    psnr_v = 10 * np.log10(1.0 / np.mean((rendered - images[v])**2))

# Average PSNR
avg_psnr = np.mean(psnr_per_view)  # ~30.1 dB (reference)

# SSIM per view
avg_ssim = np.mean([ssim(rendered_v, gt_v) for v in test_views])

# LPIPS (perceptual quality)
avg_lpips = np.mean([lpips(rendered_v, gt_v) for v in test_views])

# 3DGS-specific metrics:
# Number of Gaussians (model compactness)
n_final_gaussians = len(means)  # typically 100k-500k

# Storage size
model_size_mb = n_final_gaussians * 59 * 4 / 1e6  # float32

# Rendering FPS (real-time capability)
fps = benchmark_rendering_speed(model, cameras, n_warmup=10, n_trials=100)
```

### Step 9f: RunBundle Output

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
|   +-- gaussians.ply      # Optimized Gaussians (positions, SH, etc.) + SHA256
|   +-- point_cloud.ply    # Initial SfM point cloud + SHA256
+-- metrics.json           # PSNR, SSIM, LPIPS, n_gaussians, model_size, FPS
+-- operator.json          # Operator params (poses hash, focal, n_gaussians)
+-- provenance.json        # Git hash, seeds, array hashes, timestamps
```

---

## Design Principles

1. **Deterministic-first:** Every agent runs without LLM. LLM is optional enhancement for narrative explanations.
2. **Registry-driven:** All parameters come from validated YAML. LLM returns only registry IDs.
3. **Strict contracts:** `StrictBaseModel` with `extra="forbid"`, NaN/Inf rejection on every report.
4. **Provenance:** Every calibration entry has `dataset_id`, `seed_set`, `operator_version`, `solver_version`, `date`.
5. **Modular:** Swap any solver (3DGS -> Mip-Splatting -> 2DGS) by changing one registry ID.
6. **Gate-able:** Negotiator can veto execution if joint probability < 0.15.
7. **3DGS-specific:** Differentiable rasterization (not ray marching), adaptive density control with clone/split/prune, per-parameter learning rates, SH color representation, tile-based rendering, real-time rendering capability as a quality metric.
