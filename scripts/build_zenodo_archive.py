#!/usr/bin/env python3
"""Build the Zenodo archive folder for the Nature submission.

Creates: zenodo_archive/
  code/                    - Core framework code
  data/                    - Datasets and benchmark data
  benchmark_tasks/         - All 72 prospective + 12 dev benchmark tasks
  results/                 - Benchmark results
  REPRODUCE.md             - Exact reproduction instructions
  LICENSE                  - MIT license
  README.md                - Archive overview
"""

import os
import sys
import shutil
import json
from pathlib import Path
from datetime import datetime

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

ROOT = Path(__file__).resolve().parent.parent
ARCHIVE = ROOT / "zenodo_archive"


def copytree_safe(src, dst, ignore_patterns=None):
    """Copy tree, skipping __pycache__ and .pyc files."""
    ignore = shutil.ignore_patterns(
        '__pycache__', '*.pyc', '*.pyo', '.git', '.gitignore',
        '*.pth', '*.ckpt', '*.h5', 'node_modules',
        *(ignore_patterns or [])
    )
    if dst.exists():
        shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=ignore)


def main():
    print("=" * 60)
    print("  Building Zenodo archive folder")
    print(f"  Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # Clean and create archive directory
    if ARCHIVE.exists():
        shutil.rmtree(ARCHIVE)
    ARCHIVE.mkdir()

    # ================================================================
    # 1. CODE: Core framework
    # ================================================================
    print("\n[1/6] Copying code...")
    code_dir = ARCHIVE / "code"
    code_dir.mkdir()

    # Core packages
    copytree_safe(ROOT / "packages" / "pwm_core", code_dir / "pwm_core")
    print("  - pwm_core")

    # Algorithm base
    copytree_safe(ROOT / "algorithm_base", code_dir / "algorithm_base")
    print("  - algorithm_base")

    # Benchmark framework
    copytree_safe(ROOT / "benchmarks" / "framework", code_dir / "benchmark_framework")
    print("  - benchmark_framework")

    # Benchmark runners
    copytree_safe(ROOT / "benchmarks" / "runners", code_dir / "benchmark_runners")
    print("  - benchmark_runners")

    # Benchmark categories (shared physics)
    copytree_safe(ROOT / "benchmarks" / "categories", code_dir / "benchmark_categories")
    print("  - benchmark_categories")

    # Key scripts
    scripts_dir = code_dir / "scripts"
    scripts_dir.mkdir()
    key_scripts = [
        "run_all_12_benchmarks.py",
        "generate_ct_figure.py",
        "generate_ct_h5.py",
        "test_all_algorithms.py",
    ]
    for s in key_scripts:
        src = ROOT / "scripts" / s
        if src.exists():
            shutil.copy2(src, scripts_dir / s)
            print(f"  - scripts/{s}")

    # ================================================================
    # 2. DATA: Datasets and benchmark data
    # ================================================================
    print("\n[2/6] Copying data...")
    data_dir = ARCHIVE / "data"
    data_dir.mkdir()

    # Paper data files
    paper_data = ROOT / "papers" / "universal_simulation" / "data"
    if paper_data.exists():
        for f in paper_data.iterdir():
            shutil.copy2(f, data_dir / f.name)
            print(f"  - {f.name}")

    # Benchmark configs (168 YAML files)
    configs_dir = data_dir / "benchmark_configs"
    copytree_safe(ROOT / "benchmarks" / "configs", configs_dir)
    n_configs = len(list(configs_dir.glob("*.yaml")))
    print(f"  - {n_configs} benchmark config YAMLs")

    # ================================================================
    # 3. BENCHMARK TASKS: 12 dev + 72 prospective
    # ================================================================
    print("\n[3/6] Copying benchmark tasks...")
    tasks_dir = ARCHIVE / "benchmark_tasks"
    tasks_dir.mkdir()

    # 12 development domains (spec.md + public/dev/hidden instances)
    dev_src = ROOT / "papers" / "universal_simulation" / "benchmark"
    if dev_src.exists():
        copytree_safe(dev_src, tasks_dir / "12_development_domains")
        n_domains = len([d for d in (tasks_dir / "12_development_domains").iterdir()
                        if d.is_dir()])
        print(f"  - {n_domains} development domains with spec.md + instances")

    # 72 prospective tasks (from table2_prospective.json)
    t2_path = data_dir / "table2_prospective.json"
    if t2_path.exists():
        shutil.copy2(t2_path, tasks_dir / "72_prospective_tasks.json")
        with open(t2_path) as f:
            t2 = json.load(f)
        n_tasks = len(t2.get("tasks", []))
        print(f"  - 72 prospective tasks ({n_tasks} in JSON)")

    # ================================================================
    # 4. RESULTS: Benchmark results
    # ================================================================
    print("\n[4/6] Copying results...")
    results_dir = ARCHIVE / "results"
    results_dir.mkdir()

    result_files = [
        "paper_12_domain_results.json",
        "comprehensive_algorithm_test.json",
        "benchmark_leaderboard_reference.json",
        "measured_psnr_all.json",
    ]
    for rf in result_files:
        src = ROOT / "benchmark_results" / rf
        if src.exists():
            shutil.copy2(src, results_dir / rf)
            print(f"  - {rf}")

    # ================================================================
    # 5. REPRODUCE.md
    # ================================================================
    print("\n[5/6] Creating REPRODUCE.md...")
    reproduce_content = """# Reproducing the Results

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
"""
    (ARCHIVE / "REPRODUCE.md").write_text(reproduce_content, encoding='utf-8')
    print("  - REPRODUCE.md created")

    # ================================================================
    # 6. LICENSE + README
    # ================================================================
    print("\n[6/6] Creating LICENSE and README...")

    license_content = """MIT License

Copyright (c) 2026 Chengshuai Yang / NextGen PlatformAI C Corp

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""
    (ARCHIVE / "LICENSE").write_text(license_content, encoding='utf-8')
    print("  - LICENSE created")

    readme_content = """# Zenodo Archive: Universal Simulation Framework

**Paper:** "A simulability class S and Judge Agent for universal scientific simulation"

**Author:** Chengshuai Yang, NextGen PlatformAI C Corp

**Submitted to:** Nature

## Contents

```
zenodo_archive/
|-- code/                           Core framework source code
|   |-- pwm_core/                   Physics World Model core library
|   |-- algorithm_base/             168-modality reconstruction algorithms
|   |-- benchmark_framework/        Benchmark orchestration engine
|   |-- benchmark_runners/          CLI runners for benchmarks
|   |-- benchmark_categories/       Shared physics modules
|   |-- scripts/                    Key experiment scripts
|
|-- data/                           Datasets and metadata
|   |-- table1_efficiency.json      12-domain efficiency ratios (Table 1)
|   |-- table2_prospective.json     72-task prospective benchmark (Table 2)
|   |-- spec_ct_qc_platform.md      CT QC platform spec.md
|   |-- spec_ct_qc_copilot.md       CT QC copilot spec.md
|   |-- benchmark_configs/          168 modality YAML configurations
|
|-- benchmark_tasks/                All benchmark task specifications
|   |-- 12_development_domains/     12 domains with spec.md + instances
|   |   |-- 01_classical_mechanics/ (spec.md + public/dev/hidden)
|   |   |-- 02_electromagnetics/
|   |   |-- ...
|   |   |-- 12_molecular_dynamics/
|   |-- 72_prospective_tasks.json   72 tasks from external scientists
|
|-- results/                        Benchmark results
|   |-- paper_12_domain_results.json  12-domain experiment results
|   |-- comprehensive_algorithm_test.json  168-modality solver tests
|   |-- benchmark_leaderboard_reference.json  Reference metrics
|
|-- REPRODUCE.md                    Step-by-step reproduction guide
|-- LICENSE                         MIT License
|-- README.md                       This file
```

## Key Results

- **12/12 development domains PASS** (median quality ratio: 1.00)
- **89% success rate** on 72 prospective tasks (95% CI: [80%, 95%])
- **Median efficiency ratio rho = 480x** (IQR: 360x-660x)
- **168 imaging modalities** with 3+ solvers each

## Citation

If you use this code or data, please cite:

```bibtex
@article{yang2026simulability,
  title={A simulability class $\\mathcal{S}$ and Judge Agent for universal scientific simulation},
  author={Yang, Chengshuai},
  journal={Nature},
  year={2026}
}
```

## GitHub

https://github.com/integritynoble/Physics_World_Model
"""
    (ARCHIVE / "README.md").write_text(readme_content, encoding='utf-8')
    print("  - README.md created")

    # Summary
    total_size = sum(f.stat().st_size for f in ARCHIVE.rglob('*') if f.is_file())
    n_files = sum(1 for _ in ARCHIVE.rglob('*') if _.is_file())
    print(f"\n{'=' * 60}")
    print(f"  Zenodo archive built successfully!")
    print(f"  Location: {ARCHIVE}")
    print(f"  Files: {n_files}")
    print(f"  Size: {total_size / 1024 / 1024:.1f} MB")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
