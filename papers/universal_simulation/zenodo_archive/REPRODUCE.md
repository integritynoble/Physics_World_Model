# Reproducing the Results

## Requirements

- Python 3.10+
- NumPy, SciPy, scikit-image, matplotlib, PyTorch (optional, for DL solvers)
- Docker (optional, for containerized reproduction)

## Quick Start

```bash
# Install dependencies
pip install numpy scipy scikit-image matplotlib pyyaml h5py

# Run all 12 development benchmark experiments
python code/scripts/run_all_12_benchmarks.py

# Expected output: 12/12 PASS, median quality ratio >= 0.96
```

## Detailed Reproduction

### 1. Development Benchmarks (12 domains)

Each domain has a `spec.md` file in `benchmark_tasks/12_development_domains/`:

| Domain | Directory | Metric | Expected |
|--------|-----------|--------|----------|
| Clinical CT | 10_inverse_problems | PSNR | ~25.6 dB |
| Seismic FWI | 11_seismic | PSNR | ~20.5 dB |
| Combustion | 07_chemical_kinetics | Rel. error | < 0.01 |
| Granular flow | 01_classical_mechanics | L2 error | < 0.001 |
| Helium | 03_quantum_chemistry | Energy error | < 1 mHa |
| BFS flow | 04_fluid_dynamics | L2 error | < 0.04 |
| Topology opt. | 06_structural_mechanics | Compliance | Converged |
| Waveguide | 02_electromagnetics | Eigenvalue error | < 0.001 |
| Heat equation | 05_thermodynamics | L_inf error | < 1e-4 |
| Fresnel | 09_optics | L2 error | < 0.05 |
| Rossby waves | 08_epidemiology | Correlation | > 0.90 |
| Reaction-diffusion | 12_molecular_dynamics | L2 error | < 0.10 |

### 2. Prospective Benchmark (72 tasks)

The 72 prospective tasks from 12 external scientists are documented in:
- `benchmark_tasks/72_prospective_tasks.json`
- `data/table2_prospective.json`

Results: 89% success rate (64/72), 95% CI: [80%, 95%]

### 3. CT Figure Generation

```bash
python code/scripts/generate_ct_figure.py
# Outputs: figures/ct_groundtruth.png, ct_with_judge.png, ct_without_judge.png, ct_psnr_histogram.png
```

### 4. 168-Modality Imaging Benchmark

```bash
# Run single modality
python -m benchmarks.runners.run_benchmark --modality cassi --level M0

# Run all 168 modalities
python -m benchmarks.runners.run_all --level M0
```

## Data Sources

- LoDoPaB-CT: Leuschner et al. (2021), CC BY 4.0
- Marmousi-2: Martin et al. (2006), public domain (SEG)
- GRI-Mech 3.0: Smith et al. (1999), public domain

## Hardware

Development and testing performed on:
- CPU: AMD Ryzen / Intel Core (any modern x86-64)
- GPU: NVIDIA RTX (optional, for DL solvers)
- RAM: 16 GB minimum
- Total runtime for 12 benchmarks: ~5 minutes

## Model Backbone

The framework uses Claude Sonnet 4.6 (Anthropic, 2025) as the primary LLM backbone.
Alternative backbones tested: Claude Opus 4.6, GPT-5.4.
Cached inference logs are included for API-independent reproduction.

## License

MIT License. See LICENSE file.
