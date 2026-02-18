# {{ modality_display_name }} — PWM Report

| Field             | Value                              |
|-------------------|------------------------------------|
| **Modality ID**   | `{{ modality_id }}`                |
| **Category**      | {{ category }}                     |
| **Dataset**       | `{{ dataset_id }}`                 |
| **Date**          | {{ date }}                         |
| **PWM version**   | {{ pwm_version }}                  |
| **Report author** | PWM automated pipeline             |

---

## Modality overview

{{ modality_description }}

**Forward model:** `{{ forward_model_equation }}`

**Key parameters:**

| Parameter            | Value            |
|----------------------|------------------|
| Signal dimensions    | {{ signal_dims }} |
| Wavelength range (nm)| {{ wavelength_range_nm }} |
| Default solver       | {{ default_solver }} |
| Forward model type   | {{ forward_model_type }} |

---

## Standard dataset

| Field          | Value                          |
|----------------|--------------------------------|
| Dataset ID     | `{{ dataset_id }}`             |
| Display name   | {{ dataset_display_name }}     |
| Source URL      | {{ source_url }}               |
| Citation        | {{ citation }}                 |
| License         | {{ license }}                  |
| Image count     | {{ n_images }}                 |
| Dimensions      | {{ dimensions }}               |
| Download SHA256 | `{{ download_sha256 }}`        |

---

## PWM pipeline flowchart (mandatory)

Every report MUST include the end-to-end flowchart for the modality pipeline.
Two canonical patterns are shown below. Choose the one that matches the modality
(or combine elements from both if the pipeline is hybrid).

### Pattern A — Linear chain (e.g. widefield, confocal, CT, MRI)

```
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────┐    ┌──────────┐
│  Source   │───>│  Optics /    │───>│  Detector  │───>│  Readout  │───>│  Recon   │
│ (illum.) │    │  Sample int. │    │  (sensor)  │    │  (ADC)    │    │ (solver) │
└──────────┘    └──────────────┘    └────────────┘    └───────────┘    └──────────┘
     │                │                   │                │                │
     │  throughput τ₁ │  throughput τ₂    │  QE, noise     │  bit depth     │  algorithm
     │  noise: —      │  noise: aberr.    │  noise: shot,  │  noise: quant. │  params
     │                │                   │  dark, read    │                │
```

### Pattern B — Coded/multiplexed acquisition (e.g. SPC, CASSI, CACTI, lensless)

```
┌──────────┐    ┌──────────────┐    ┌────────────┐    ┌───────────┐    ┌──────────┐
│  Source   │───>│  Encoding    │───>│  Detector  │───>│  Readout  │───>│  Recon   │
│ (illum.) │    │  (mask/code) │    │  (sensor)  │    │  (ADC)    │    │ (solver) │
└──────────┘    └──────────────┘    └────────────┘    └───────────┘    └──────────┘
     │                │                   │                │                │
     │  throughput τ₁ │  Φ encoding       │  QE, noise     │  bit depth     │  algorithm
     │  noise: —      │  CR = M/N         │  noise: shot,  │  noise: quant. │  params
     │                │  mask type        │  dark, read    │                │
```

### Modality-specific flowchart

```
{{ flowchart }}
```

---

## Element inventory & mismatch parameters

| # | Element name | Element type | Transfer kind | Throughput | Noise kinds | Mismatch parameters | Mismatch range |
|---|-------------|-------------|---------------|------------|-------------|--------------------:|---------------:|
| 1 | {{ element_1_name }} | {{ element_1_type }} | {{ element_1_transfer }} | {{ element_1_throughput }} | {{ element_1_noise }} | {{ element_1_mismatch_params }} | {{ element_1_mismatch_range }} |
| 2 | {{ element_2_name }} | {{ element_2_type }} | {{ element_2_transfer }} | {{ element_2_throughput }} | {{ element_2_noise }} | {{ element_2_mismatch_params }} | {{ element_2_mismatch_range }} |
| 3 | {{ element_3_name }} | {{ element_3_type }} | {{ element_3_transfer }} | {{ element_3_throughput }} | {{ element_3_noise }} | {{ element_3_mismatch_params }} | {{ element_3_mismatch_range }} |
| 4 | {{ element_4_name }} | {{ element_4_type }} | {{ element_4_transfer }} | {{ element_4_throughput }} | {{ element_4_noise }} | {{ element_4_mismatch_params }} | {{ element_4_mismatch_range }} |
| 5 | {{ element_5_name }} | {{ element_5_type }} | {{ element_5_transfer }} | {{ element_5_throughput }} | {{ element_5_noise }} | {{ element_5_mismatch_params }} | {{ element_5_mismatch_range }} |

> Add or remove rows as needed for the modality.

---

## Node-by-node trace (one sample)

Single-sample trace through the pipeline showing tensor shapes, value ranges, and timing at each node.

| Node | Operation | Input shape | Output shape | Value range (min, max) | Wall time (s) | Notes |
|------|-----------|-------------|-------------|----------------------:|---------------:|-------|
| 0 | Ground truth load | — | {{ gt_shape }} | {{ gt_range }} | {{ gt_time }} | {{ gt_notes }} |
| 1 | {{ node_1_op }} | {{ node_1_in }} | {{ node_1_out }} | {{ node_1_range }} | {{ node_1_time }} | {{ node_1_notes }} |
| 2 | {{ node_2_op }} | {{ node_2_in }} | {{ node_2_out }} | {{ node_2_range }} | {{ node_2_time }} | {{ node_2_notes }} |
| 3 | {{ node_3_op }} | {{ node_3_in }} | {{ node_3_out }} | {{ node_3_range }} | {{ node_3_time }} | {{ node_3_notes }} |
| 4 | {{ node_4_op }} | {{ node_4_in }} | {{ node_4_out }} | {{ node_4_range }} | {{ node_4_time }} | {{ node_4_notes }} |
| 5 | Reconstruction | {{ recon_in }} | {{ recon_out }} | {{ recon_range }} | {{ recon_time }} | {{ recon_notes }} |

> Add or remove rows as needed for the modality.

---

## Workflow W1: Prompt-driven simulation + reconstruction

**Description:** End-to-end pipeline from ground truth through simulated forward model
to reconstruction, driven by a natural-language prompt.

**Prompt used:** `{{ w1_prompt }}`

**Solver:** `{{ w1_solver }}`

**Solver parameters:** `{{ w1_solver_params }}`

### W1 metrics

| Metric    | Value          |
|-----------|----------------|
| PSNR (dB) | {{ w1_psnr }}  |
| SSIM      | {{ w1_ssim }}  |
| NRMSE     | {{ w1_nrmse }} |
| Runtime (s)| {{ w1_runtime }} |
| Iterations | {{ w1_iters }} |

---

## Workflow W2: Operator correction mode (measured y + operator A)

**Description:** Given a real or simulated measurement `y` and a (possibly mismatched)
forward operator `A`, PWM corrects the operator and reconstructs.

### Operator A specification

| Field                  | Value                              |
|------------------------|------------------------------------|
| A_definition           | {{ w2_a_definition }}              |
| A_extraction_method    | {{ w2_a_extraction_method }}       |
| A_sha256               | `{{ w2_a_sha256 }}`               |
| Linearity              | {{ w2_linearity }}                 |
| Notes (if linearized)  | {{ w2_linearization_notes }}       |

> **A_extraction_method** must be one of:
> - `graph_stripped` — operator extracted from the PWM graph by stripping noise nodes
> - `provided` — operator supplied externally by the user
> - `linearized` — non-linear operator linearized around an operating point

### W2 comparison

| Metric               | Before correction | After correction | Change (%) |
|----------------------|------------------:|-----------------:|-----------:|
| NLL                  | {{ w2_nll_before }} | {{ w2_nll_after }} | {{ w2_nll_change_pct }} |
| PSNR (dB)            | {{ w2_psnr_before }} | {{ w2_psnr_after }} | {{ w2_psnr_change }} |
| SSIM                 | {{ w2_ssim_before }} | {{ w2_ssim_after }} | {{ w2_ssim_change }} |
| NRMSE                | {{ w2_nrmse_before }} | {{ w2_nrmse_after }} | {{ w2_nrmse_change }} |
| Mismatch residual    | {{ w2_mismatch_before }} | {{ w2_mismatch_after }} | {{ w2_mismatch_change }} |
| Correction runtime (s)| —                 | {{ w2_correction_runtime }} | — |

---

## Test results summary

### Quick gate (pass/fail)

- [ ] Forward model runs without error
- [ ] Adjoint consistency check passes (|<Ax,y> - <x,A^T y>| / (||Ax|| ||y||) < 1e-5)
- [ ] Reconstruction PSNR >= modality acceptance threshold
- [ ] Reconstruction SSIM >= modality acceptance threshold
- [ ] W2 NLL decreases after correction
- [ ] W2 corrected PSNR >= uncorrected PSNR
- [ ] No NaN/Inf in any output tensor
- [ ] Runtime within 2x of benchmark budget

### Full metrics

| # | Test name | Status | Value | Threshold | Margin | Notes |
|---|-----------|--------|------:|----------:|-------:|-------|
| 1 | forward_no_error | {{ t1_status }} | — | — | — | {{ t1_notes }} |
| 2 | adjoint_consistency | {{ t2_status }} | {{ t2_value }} | < 1e-5 | {{ t2_margin }} | {{ t2_notes }} |
| 3 | w1_psnr | {{ t3_status }} | {{ t3_value }} | {{ t3_threshold }} | {{ t3_margin }} | {{ t3_notes }} |
| 4 | w1_ssim | {{ t4_status }} | {{ t4_value }} | {{ t4_threshold }} | {{ t4_margin }} | {{ t4_notes }} |
| 5 | w2_nll_decrease | {{ t5_status }} | {{ t5_value }} | < 0 | {{ t5_margin }} | {{ t5_notes }} |
| 6 | w2_psnr_improvement | {{ t6_status }} | {{ t6_value }} | >= 0 | {{ t6_margin }} | {{ t6_notes }} |
| 7 | no_nan_inf | {{ t7_status }} | — | — | — | {{ t7_notes }} |
| 8 | runtime_budget | {{ t8_status }} | {{ t8_value }} | {{ t8_threshold }} | {{ t8_margin }} | {{ t8_notes }} |
| 9 | operator_norm | {{ t9_status }} | {{ t9_value }} | {{ t9_threshold }} | {{ t9_margin }} | {{ t9_notes }} |
| 10 | reconstruction_convergence | {{ t10_status }} | {{ t10_value }} | {{ t10_threshold }} | {{ t10_margin }} | {{ t10_notes }} |

---

## Reproducibility

| Field                | Value                              |
|----------------------|------------------------------------|
| Seed                 | {{ seed }}                         |
| NumPy RNG state hash | `{{ numpy_rng_state_hash }}`       |
| PWM version          | {{ pwm_version }}                  |
| Python version       | {{ python_version }}               |
| Key package versions | {{ key_package_versions }}         |
| Platform             | {{ platform }}                     |
| Reproduction command | `{{ reproduction_command }}`       |
| Output hash          | `{{ output_hash }}`                |

---

## Saved artifacts

| Artifact              | Path                               | SHA256                          |
|-----------------------|------------------------------------|---------------------------------|
| Ground truth          | `{{ gt_path }}`                    | `{{ gt_sha256 }}`              |
| Measurement           | `{{ measurement_path }}`           | `{{ measurement_sha256 }}`     |
| Reconstruction (W1)   | `{{ recon_w1_path }}`              | `{{ recon_w1_sha256 }}`        |
| Reconstruction (W2)   | `{{ recon_w2_path }}`              | `{{ recon_w2_sha256 }}`        |
| Operator A (original) | `{{ operator_orig_path }}`         | `{{ operator_orig_sha256 }}`   |
| Operator A (corrected)| `{{ operator_corr_path }}`         | `{{ operator_corr_sha256 }}`   |
| Run bundle            | `{{ runbundle_path }}`             | `{{ runbundle_sha256 }}`       |
| Report (this file)    | `{{ report_path }}`                | `{{ report_sha256 }}`          |

---

## Next actions

- [ ] {{ next_action_1 }}
- [ ] {{ next_action_2 }}
- [ ] {{ next_action_3 }}
