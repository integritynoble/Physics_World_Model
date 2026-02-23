# PWM Benchmarks – 168 Imaging Modalities

Config-driven benchmark framework for all 168 imaging modalities in the Physics World Model.

## Quick Start

```bash
# Run a single modality
python -m benchmarks.runners.run_benchmark --modality widefield --level M0

# Dry-run (validate config + operator build, no reconstruction)
python -m benchmarks.runners.run_benchmark --modality cassi --level M0 --dry-run

# Run all modalities in a category
python -m benchmarks.runners.run_category --category "Medical Imaging" --level M0

# List all categories
python -m benchmarks.runners.run_category --list-categories

# Run all 168 modalities
python -m benchmarks.runners.run_all --level M0 --dry-run
python -m benchmarks.runners.run_all --level M0 --parallel 4

# Regenerate configs from registries
python -m benchmarks.runners.generate_configs
```

## Architecture

```
benchmarks/
  framework/              # Core framework classes
    benchmark_config.py   # BenchmarkConfig dataclass + YAML loader
    base_benchmark.py     # BaseBenchmark orchestrator
    metrics.py            # PSNR, SSIM, SAM, NRMSE
    data_source.py        # Multi-strategy data acquisition
    source_attribution.py # Provenance tracking
    synthetic_data.py     # Phantom generators
    mismatch_engine.py    # Perturbation injection + grid search
    report_writer.py      # JSON + markdown output

  configs/                # 168 YAML configs (auto-generated)
    _template.yaml        # Reference template
    widefield.yaml        # ... one per modality
    cassi.yaml

  categories/             # Shared physics per category
    microscopy_psf.py     # ~30 microscopy modalities
    compressive_mask.py   # ~20 compressive modalities
    medical_ct_radon.py   # ~15 CT/projection modalities
    medical_mri_kspace.py # ~10 MRI modalities
    electron_ctf.py       # ~10 electron microscopy
    scanning_probe.py     # ~8 SPM modalities
    remote_sensing_sar.py # ~10 remote sensing
    nuclear_emission.py   # ~10 nuclear/emission

  runners/                # CLI entry points
    run_benchmark.py      # Single modality
    run_category.py       # By category
    run_all.py            # Full 168-modality sweep
    generate_configs.py   # Auto-generate configs

  tests/                  # Validation
    test_framework.py     # Unit tests (33 tests)
    test_configs.py       # Config validation (14 tests)

  results/                # Output (gitignored)
```

## Maturity Levels

| Level | Description |
|-------|-------------|
| M0 | Template: nominal parameters, forward + reconstruct |
| M1 | Single-parameter mismatch scenarios |
| M2 | Compound mismatch + grid-search correction |
| M3 | Real experimental data with measured mismatch |
| M4 | Adversarial worst-case mismatch injection |

## Data Sourcing Priority

1. **web** – Download from known dataset URLs
2. **experimental** – Load from local `datasets/` directory
3. **synthetic_web** – Download synthetic from repositories
4. **generated** – Create via category-module phantom generator

## Source Attribution

Every benchmark result tracks provenance:
- Ground truth: where the test data came from
- Forward model: which operator/graph template
- Solver: which reconstruction algorithm
- Mismatch ranges: source of perturbation parameters
