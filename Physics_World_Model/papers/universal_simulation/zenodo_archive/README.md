# Zenodo Archive: Universal Simulation Framework

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
  title={A simulability class $\mathcal{S}$ and Judge Agent for universal scientific simulation},
  author={Yang, Chengshuai},
  journal={Nature},
  year={2026}
}
```

## GitHub

https://github.com/integritynoble/Physics_World_Model
