# PWM Open-Core Boundary

## License Structure

### Open Source (MIT License)

The following components are freely available under the MIT license:

- **Core library** (`packages/pwm_core/`)
  - OperatorGraph IR and compiler
  - 26 imaging modality templates
  - All reconstruction solvers
  - Calibration algorithms (Algorithm 1: beam search, Algorithm 2: GPU differentiable)
  - Agent framework (deterministic path)
  - Analysis and diagnosis modules
  - CLI tools (`pwm run`, `pwm calib-recon`, `pwm demo`, `pwm doctor`)

- **Datasets** (`datasets/`)
  - InverseNet benchmark dataset (1,728 samples)
  - All generation scripts
  - Dataset cards and manifests

- **Community** (`community/`)
  - Challenge infrastructure
  - Leaderboard system
  - Validation pipeline

- **Documentation** (`docs/`)
  - QuickStart guide
  - Gallery
  - API reference

### Paid Services (Commercial License)

The following services are available under a commercial license:

| Service | Description | Target Customer |
|---------|-------------|-----------------|
| **Calibration Sprint** | Expert-guided calibration of customer's imaging system using PWM toolkit | Labs with custom hardware |
| **Hosted GPU Pipeline** | Cloud-hosted PWM pipeline with GPU acceleration, job queue, and result storage | Teams without GPU infrastructure |
| **Priority Support** | Dedicated support channel, SLA-backed response times | Enterprise users |
| **Custom Modality Development** | New modality template + calibration for non-standard imaging systems | R&D departments |
| **Training Workshops** | On-site or virtual training on PWM toolkit and calibration best practices | Academic groups, industry teams |

### Boundary Rules

1. **All algorithms are open source.** No algorithm is paywalled.
2. **Paid services are execution + expertise**, not code access.
3. **Community contributions** (challenges, leaderboard entries) are always free.
4. **Research use** is always free, including commercial research.
5. **The CLI and library** will never require a license key to run.
