# Prospective Head-to-Head Expert Study Protocol

## Objective

Replace estimated expert times in Table 4 (Expert Comparison) with **measured** times from a prospective study. Five AI agent "experts" independently design and reconstruct for 3 real-data modalities, using only a one-sentence task description (matching the Plan Agent's input format).

---

## 1. Expert Agents (n=5)

Each agent has a distinct expertise profile and system design philosophy. All agents receive the **same one-sentence design brief** per modality and must produce a complete imaging system design from scratch — forward model specification, validation, and reconstruction.

| Agent | Persona | Design Philosophy | Allowed Libraries |
|-------|---------|------------------|-------------------|
| **E1** | Medical imaging physicist | Physics-first: derive forward model from first principles, ADMM solver | numpy, scipy, scikit-image |
| **E2** | Signal processing engineer | Fourier-domain: design via spectral analysis, FBP/spectral methods | numpy, scipy, pywt |
| **E3** | Applied mathematician | Variational: formulate as optimization, TV/wavelet regularization | numpy, scipy, pylops |
| **E4** | Computational imaging researcher | Geometry-first: build measurement operator, CG/LSQR iterative solver | numpy, scipy, astra-toolbox |
| **E5** | Algorithm engineer | Modular: plug-and-play forward model + denoiser prior (BM3D/NLM) | numpy, scipy, bm3d |

**Each agent must independently:**
1. **Specify** the forward model (physics, geometry, noise model, parameters)
2. **Validate** the forward model (adjoint test, energy conservation, invertibility)
3. **Implement** the reconstruction pipeline (solver + regularization)
4. **Evaluate** on test data (PSNR, SSIM)

**Constraints on all agents:**
- No pre-trained neural networks (matches "NEVER train" standing rule)
- No access to ground truth during design or reconstruction
- Must write complete, runnable Python code
- Forward model must be derived from the one-sentence brief (not provided)
- Time measured from prompt receipt to final PSNR/SSIM output

---

## 2. Modalities (n=3)

Selected for ground-truth availability and carrier diversity:

### 2.1 CT (X-ray carrier)
- **Data**: LoDoPaB-CT from GCS (`gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/`)
- **Task prompt**: "Design a computational imaging system for parallel-beam CT with Poisson noise. Specify the forward model, select appropriate reconstruction algorithm, and produce reconstructed images."
- **Expected deliverables**: (1) Forward model specification (physics + parameters), (2) Validation of model correctness, (3) Working reconstruction pipeline, (4) Reconstructed images with quality metrics
- **Ground truth**: Available (x_true in dataset)
- **Paper reference**: Table 4 row 1 (CT real)

### 2.2 MRI (Spin carrier)
- **Data**: M4Raw-derived from GCS (`gs://pwm-benchmark-datasets/datasets/Benchmark/mri/public/`)
- **Task prompt**: "Design a computational imaging system for multi-coil MRI with 4× Cartesian undersampling. Specify the forward model, select appropriate reconstruction algorithm, and produce reconstructed images."
- **Expected deliverables**: (1) Forward model specification (k-space sampling + coil sensitivities), (2) Validation of model correctness, (3) Working reconstruction pipeline, (4) Reconstructed images with quality metrics
- **Ground truth**: Available (x_true in dataset)
- **Paper reference**: Table 4 row 2 (MRI real)

### 2.3 CASSI (Photon carrier)
- **Data**: KAIST TSA from GCS (`gs://pwm-benchmark-datasets/datasets/Benchmark/sd_cassi/public/`)
- **Task prompt**: "Design a computational imaging system for coded-aperture snapshot spectral imaging (CASSI). Specify the forward model, select appropriate reconstruction algorithm, and produce reconstructed images."
- **Expected deliverables**: (1) Forward model specification (coded aperture + spectral dispersion), (2) Validation of model correctness, (3) Working reconstruction pipeline, (4) Reconstructed images with quality metrics
- **Ground truth**: Available (x_true in dataset)
- **Paper reference**: Table 4 row 3 (CASSI real)

---

## 3. Measurement Protocol

### 3.1 Timing
- **Wall-clock time**: Measured via Python `time.perf_counter()` from prompt dispatch to final metric output
- **Includes**: Forward model derivation, specification writing, validation, solver implementation, parameter tuning, reconstruction execution
- **Excludes**: Data download time (pre-cached), system overhead

### 3.2 Quality Metrics
- **PSNR** (dB): `10 * log10(max_val^2 / MSE)` with `max_val = x_true.max()`
- **SSIM**: `skimage.metrics.structural_similarity` with `data_range=x_true.max() - x_true.min()`
- Both computed on full image (no ROI cropping)

### 3.3 Lines of Code (LoC)
- Count non-empty, non-comment lines in the complete design output (forward model spec + reconstruction code)
- Excludes imports and metric computation boilerplate
- Separately report: (a) forward model specification lines, (b) reconstruction code lines

### 3.4 Samples
- **Per modality**: Use the first 10 test samples from the public tier
- **Report**: Mean ± std across samples for PSNR and SSIM

---

## 4. Agent Pipeline Comparison

The **Agent pipeline** (Plan → Judge → Performance) runs on the same 3 prompts with the same data:
- Plan Agent generates `spec.md`
- Judge validates through 6-gate compiler
- Reconstructor executes the compiled pipeline
- Same timing methodology (wall-clock from prompt to metrics)

---

## 5. Execution Plan

### Phase 1: Data Preparation
1. Download test data for CT, MRI, CASSI from GCS (10 samples each)
2. Verify forward models match paper descriptions
3. Create standardized evaluation harness (`evaluate.py`)

### Phase 2: Expert Agent Runs (5 agents × 3 modalities = 15 runs)
For each (agent, modality) pair:
1. Start timer
2. Dispatch one-sentence prompt to agent
3. Agent generates complete reconstruction script
4. Execute reconstruction on all 10 samples
5. Stop timer
6. Compute PSNR, SSIM, LoC

### Phase 3: Agent Pipeline Runs (3 modalities)
1. Run Plan → Judge → Performance pipeline on each modality
2. Same timing, same metrics

### Phase 4: Analysis
1. Compute per-agent, per-modality results
2. Aggregate: mean ± std across agents for "expert" column
3. Compute quality ratio: `Agent_PSNR / mean(Expert_PSNR) × 100`
4. Compute speedup factor: `mean(Expert_time) / Agent_time`
5. Statistical tests: paired t-test (agent vs. best expert) per modality

---

## 6. Output Format

### Per-run JSON
```json
{
  "agent_id": "E1",
  "modality": "ct",
  "wall_clock_s": 1234.5,
  "psnr_mean": 31.2,
  "psnr_std": 1.1,
  "ssim_mean": 0.88,
  "ssim_std": 0.03,
  "lines_of_code": 85,
  "reconstruction_code": "path/to/script.py",
  "success": true,
  "error": null
}
```

### Aggregated results → `expert_study_results.json`
```json
{
  "modality": "ct",
  "agent_pipeline": {"psnr": 31.7, "ssim": 0.891, "time_min": 25, "loc": 12},
  "experts": {
    "mean_psnr": 30.8, "std_psnr": 1.5,
    "mean_ssim": 0.87, "std_ssim": 0.04,
    "mean_time_min": 45, "std_time_min": 15,
    "mean_loc": 120, "std_loc": 40
  },
  "quality_ratio": 103.0,
  "speedup": 1.8
}
```

---

## 7. Integration into Paper

### Table Update (Table 4)
Replace estimated expert times with measured values:
- Expert PSNR → `mean ± std` across 5 agents
- Expert time → `mean ± std` wall-clock time
- Add footnote: "Expert values are prospective measurements from 5 independent reconstruction agents"

### New Paragraph (Section: Expert Time Methodology)
Replace the current acknowledgment ("We acknowledge this is not a prospective head-to-head comparison...") with:
> "Expert times are measured prospectively: five independent design agents, each with a distinct system design philosophy (physics-first, Fourier-domain, variational, geometry-first, modular plug-and-play), received identical one-sentence design briefs matching the Plan Agent input. Each agent independently derived the forward model, validated its correctness, implemented reconstruction, and evaluated quality. Wall-clock time was measured from design brief to final metric output. All agents used classical/iterative methods without pre-trained networks, matching the complexity of expert manual system design. Results (Table 4) show [findings]."

---

## 8. Acceptance Criteria

- [ ] n ≥ 3 experts complete all 3 modalities successfully
- [ ] n ≥ 3 modalities have ground-truth PSNR comparison
- [ ] All times are measured (not estimated)
- [ ] Statistical comparison reported (mean ± std, quality ratio)
- [ ] Results integrated into paper.tex Table 4 and Methods section

---

## 9. Implementation Notes

- Each expert agent will be implemented as a Claude API call with a distinct system prompt defining its design philosophy
- The agent must output: (a) a forward model specification (structured text), (b) validation code, (c) reconstruction code
- Forward model physics are NOT provided — agents must derive them from the one-sentence design brief
- Only a standardized `load_data(modality)` function is provided (returns measurements + metadata, no physics hints)
- Max wall-clock timeout per agent per modality: 30 minutes
- This matches the paper's claim: the framework reduces "weeks of specialist effort per modality" to minutes of automated design
