# InverseNet CASSI Validation: Benchmark Integration Guide

**Purpose:** Guide to properly integrating CASSI validation with PWM benchmark infrastructure

**Reference Plan:** `cassi_plan_inversenet.md`
**Implementation Status:** `IMPLEMENTATION_STATUS.md`

---

## Quick Start: The Right Approach

The PWM benchmark framework already has everything needed. Follow this pattern:

```python
from pwm_core.benchmarks.benchmark_helpers import build_benchmark_operator
from pwm_core.api import ExperimentSpec, build_operator

# 1. Create CASSI operator
operator = build_benchmark_operator("cassi", (256, 256, 28))

# 2. Apply mismatch parameters (for Scenario IV)
mismatch_params = {
    'mask_dx': 0.5,      # pixels
    'mask_dy': 0.3,      # pixels
    'mask_theta': 0.1,   # degrees
}
operator.set_theta(mismatch_params)

# 3. Generate measurement
y = operator.forward(x)  # Shape: (256, 310)

# 4. Reconstruct
x_recon = mst_reconstruct(y)  # MST handles shift_back internally

# 5. Evaluate
psnr = compute_psnr(x, x_recon)
```

---

## Key Benchmark Files to Reference

### 1. **benchmark_helpers.py** - Operator Factory
**Location:** `packages/pwm_core/benchmarks/benchmark_helpers.py`

**Key Function:**
```python
def build_benchmark_operator(
    modality: str,           # "cassi"
    dims: Tuple[int, ...],   # (256, 256, 28)
    theta: Optional[Dict] = None,
    assets: Optional[Dict] = None,
) -> BaseOperator:
    """Build operator via graph templates or legacy fallback."""
```

**How it works:**
1. Creates `ExperimentSpec` with modality and dimensions
2. Calls `build_operator()` which:
   - Tries graph template first (modern approach)
   - Falls back to legacy operator if template not available
3. Applies `theta` parameters if provided
4. Returns ready-to-use operator with `.forward()`, `.adjoint()`, `.set_theta()`

**Usage for InverseNet:**
```python
# Scenario I: Ideal
op_ideal = build_benchmark_operator("cassi", (256, 256, 28))
y_ideal = op_ideal.forward(scene)

# Scenario II: Baseline (no mismatch applied, just different measurement)
op_real = build_benchmark_operator("cassi", (256, 256, 28))
y_baseline = op_real.forward(scene)  # Uses real mask

# Scenario IV: Oracle with known mismatch
op_oracle = build_benchmark_operator("cassi", (256, 256, 28))
op_oracle.set_theta({'mask_dx': 0.5, 'mask_dy': 0.3, 'mask_theta': 0.1})
y_corrupt = op_oracle.forward(scene)
```

### 2. **test_operator_correction.py** - Correction Pattern
**Location:** `packages/pwm_core/benchmarks/test_operator_correction.py`

**Structure for reference:**
1. Builds operator via `build_benchmark_operator()`
2. Generates measurement with known parameters
3. Applies wrong parameters (assumes perfect when mismatch exists)
4. Performs correction via grid search / optimization
5. Evaluates improvement

**For InverseNet Scenarios:**
- Scenario I: Oracle baseline (no wrong assumptions)
- Scenario II: Wrong parameters assumed (uncorrected mismatch)
- Scenario IV: Correct parameters known (oracle correction)

**Key insight:** The framework demonstrates that operator knowledge improves reconstruction when applied correctly.

### 3. **_cassi_upwmi.py** - CASSI Reference Implementation
**Location:** `packages/pwm_core/benchmarks/_cassi_upwmi.py`

**Study these patterns:**
1. Forward model definition and measurement generation
2. How `set_theta()` is used to apply mismatch
3. Reconstruction method integration
4. Metrics computation

**Reusable code patterns:**
```python
# Load scene
scene = load_kaist_scene(...)

# Create operator
operator = build_benchmark_operator("cassi", (256, 256, 28))

# Forward model
y = operator.forward(scene)

# MST reconstruction
from pwm_core.recon.mst import create_mst
model = create_mst(variant='mst_l')
x_recon = mst_forward(y, model)

# Metrics
psnr = compute_psnr(scene, x_recon)
ssim = compute_ssim(...)
sam = compute_sam(...)
```

### 4. **run_all.py** - Benchmark Runner Template
**Location:** `packages/pwm_core/benchmarks/run_all.py`

**Structure:**
```python
class BenchmarkRunner:
    def run_modality(self, modality: str, dims: Tuple):
        """Run complete benchmark for one modality."""
        # Load scene
        # Create operator
        # Run forward model
        # Reconstruct
        # Compute metrics
        # Save results
```

**Adapt this for InverseNet:**
```python
class InverseNetCassiValidator:
    def validate_scene(self, scene: np.ndarray):
        """Validate across all 3 scenarios and 2 methods."""
        results = {
            'scenario_i': self.scenario_i(scene),
            'scenario_ii': self.scenario_ii(scene),
            'scenario_iv': self.scenario_iv(scene),
        }
        return results
```

---

## Step-by-Step Integration

### Phase 1: Operator Verification (1-2 hours)

**Goal:** Ensure operators work correctly

```python
# test_operators.py
from pwm_core.benchmarks.benchmark_helpers import build_benchmark_operator
import scipy.io as sio

# Load test scene
scene = load_kaist_scene('scene01')  # (256, 256, 28)

# Build operators
op_ideal = build_benchmark_operator("cassi", (256, 256, 28))
op_real = build_benchmark_operator("cassi", (256, 256, 28))

# Test forward models
y_ideal = op_ideal.forward(scene)
y_real = op_real.forward(scene)

print(f"Ideal measurement: {y_ideal.shape}, range [{y_ideal.min():.4f}, {y_ideal.max():.4f}]")
print(f"Real measurement: {y_real.shape}, range [{y_real.min():.4f}, {y_real.max():.4f}]")

# Verify expected shapes
assert y_ideal.shape == (256, 310), f"Wrong shape: {y_ideal.shape}"
assert y_real.shape == (256, 310), f"Wrong shape: {y_real.shape}"

# Check parameter setting
op_oracle = build_benchmark_operator("cassi", (256, 256, 28))
op_oracle.set_theta({'mask_dx': 0.5, 'mask_dy': 0.3, 'mask_theta': 0.1})
y_oracle = op_oracle.forward(scene)
print(f"Oracle measurement: {y_oracle.shape}")

# Verify degradation exists
diff = np.mean((y_ideal - y_oracle) ** 2)
print(f"Measurement difference: {diff:.6f} (should be > 0)")
```

**Success criteria:**
- ✓ Operators create (256, 310) measurements
- ✓ set_theta() works without errors
- ✓ Different operators produce different results

### Phase 2: Reconstruction Validation (2-3 hours)

**Goal:** Verify MST reconstruction works with proper inputs

```python
# test_reconstruction.py
from pwm_core.recon.mst import create_mst, shift_back_meas_torch
import torch

# Create measurements (from Phase 1)
y_ideal = ...  # (256, 310)

# MST-L reconstruction
model = create_mst(variant='mst_l')
model = model.to('cuda:0')
model.eval()

# Prepare tensor
y_tensor = torch.from_numpy(y_ideal).unsqueeze(0).float()  # (1, 256, 310)

# Shift back
y_shifted = shift_back_meas_torch(y_tensor, step=2, nC=28)  # (1, 28, 256, 256)

# Reconstruct
with torch.no_grad():
    x_recon = model(y_shifted)  # (1, 28, 256, 256)

# Convert to numpy
x_np = x_recon.squeeze(0).permute(1, 2, 0).cpu().numpy()  # (256, 256, 28)

# Verify
assert x_np.shape == (256, 256, 28)
assert x_np.min() >= 0 and x_np.max() <= 1

# Compute PSNR
psnr_val = compute_psnr(scene, x_np)
print(f"PSNR: {psnr_val:.2f} dB (expect 34-36 dB)")
```

**Success criteria:**
- ✓ MST models load successfully
- ✓ Tensor shapes transform correctly
- ✓ PSNR values in realistic range (20-36 dB)

### Phase 3: Scenario Implementation (4-6 hours)

**Goal:** Implement all 3 scenarios with proper operators

```python
# validate_scenarios.py
class CassiValidator:
    def __init__(self):
        self.scene = load_kaist_scene('scene01')
        self.device = 'cuda:0'

    def scenario_i(self):
        """Ideal: perfect operator, no mismatch."""
        op = build_benchmark_operator("cassi", (256, 256, 28))
        y = op.forward(self.scene)
        x_recon = self.reconstruct_mst(y)
        psnr = compute_psnr(self.scene, x_recon)
        return {'psnr': psnr, 'measurement': y}

    def scenario_ii(self):
        """Baseline: real mask, assumed perfect."""
        op = build_benchmark_operator("cassi", (256, 256, 28))
        # Note: Should use real mask here via parameter
        y = op.forward(self.scene)
        # Add noise to simulate mismatch
        y_noisy = self.add_noise(y)
        x_recon = self.reconstruct_mst(y_noisy)
        psnr = compute_psnr(self.scene, x_recon)
        return {'psnr': psnr, 'measurement': y_noisy}

    def scenario_iv(self):
        """Oracle: known mismatch parameters."""
        op = build_benchmark_operator("cassi", (256, 256, 28))
        op.set_theta({'mask_dx': 0.5, 'mask_dy': 0.3, 'mask_theta': 0.1})
        y = op.forward(self.scene)
        y_noisy = self.add_noise(y)
        x_recon = self.reconstruct_mst(y_noisy)
        psnr = compute_psnr(self.scene, x_recon)
        return {'psnr': psnr, 'measurement': y_noisy}

    def reconstruct_mst(self, y):
        # MST reconstruction
        ...

    def add_noise(self, y):
        # Poisson + Gaussian noise
        ...
```

### Phase 4: Full Validation (4-8 hours)

**Goal:** Run complete 10-scene × 3-scenario × 2-method validation

```python
# validate_cassi_inversenet_final.py
validator = CassiValidator()

for scene_name in SCENES:
    scene = load_kaist_scene(scene_name)

    for method in ['mst_s', 'mst_l']:
        results_i = scenario_i(scene, method)
        results_ii = scenario_ii(scene, method)
        results_iv = scenario_iv(scene, method)

        save_results(scene_name, method, {
            'i': results_i,
            'ii': results_ii,
            'iv': results_iv,
        })

# Aggregate and save
aggregate_results()
```

**Deliverables:**
- `cassi_validation_results.json` - Per-scene results
- `cassi_summary.json` - Aggregated statistics
- `cassi_figures/` - Visualization PNG files
- `VALIDATION_REPORT.md` - Analysis and findings

---

## Critical Parameters

### Scenario Configuration

| Parameter | Scenario I | Scenario II | Scenario IV |
|-----------|-----------|-----------|-----------|
| **Mask** | Ideal | Real | Real |
| **Mismatch** | dx=0, dy=0, θ=0 | None (but measurement has it) | dx=0.5, dy=0.3, θ=0.1 |
| **Noise** | None | Poisson + Gaussian | Poisson + Gaussian |
| **Operator** | `build_benchmark_operator()` | `build_benchmark_operator()` | `build_benchmark_operator()` + `set_theta()` |

### Noise Parameters

```python
# Poisson + Gaussian noise
photon_peak = 10000  # Sensor saturation
read_noise_sigma = 1.0  # dB

# Implementation:
y_scaled = y / np.max(y) * photon_peak
y_poisson = np.random.poisson(y_scaled)
y_noisy = y_poisson / photon_peak * np.max(y) + np.random.normal(0, read_noise_sigma, y.shape)
```

---

## Debugging Checklist

If results look wrong:

- [ ] **Measurement shape wrong?**
  - Should be (256, 310) for CASSI
  - Check operator.forward() output shape

- [ ] **PSNR too low (1-10 dB)?**
  - MST model not loaded properly
  - Check model.state_dict() has weights
  - Verify shift_back input/output shapes

- [ ] **PSNR too high (>50 dB)?**
  - Measurement not corrupted properly
  - Check noise is being added
  - Verify mismatch parameters applied via set_theta()

- [ ] **Scenario gaps wrong?**
  - Scenario I should be highest PSNR
  - Scenario II should be middle
  - Scenario IV should be between II and I
  - If not, check scenario implementations

- [ ] **Results not reproducible?**
  - Set random seeds at start: `np.random.seed(42)`
  - Verify same scenes loaded each time
  - Check operator uses same mask

---

## Success Metrics

**Final Results** (per plan):

```
| Method | Scenario I | Scenario II | Scenario IV | Gap I→II |
|--------|-----------|-----------|-----------|---------|
| MST-L  | 36.0 dB   | 32.3 dB   | 33.6 dB   | 3.7 dB  |
| MST-S  | 34.2 dB   | 30.5 dB   | 31.8 dB   | 3.7 dB  |
```

**When complete:**
- ✓ All 10 scenes processed
- ✓ 3 scenarios for each method
- ✓ PSNR values in expected ranges
- ✓ Results JSON files created
- ✓ Summary statistics computed
- ✓ Figures generated
- ✓ Documentation complete

---

## References

- PWM Core Benchmarks: `packages/pwm_core/benchmarks/`
- CASSI Plan: `papers/inversenet/cassi_plan_inversenet.md`
- Implementation Status: `papers/inversenet/IMPLEMENTATION_STATUS.md`
- Test Operator Correction: `packages/pwm_core/benchmarks/test_operator_correction.py`

---

**This guide provides the roadmap for completing InverseNet CASSI validation with proper benchmark integration.**

**Start with Phase 1 (operator verification) to validate the approach before scaling to full 10-scene validation.**
