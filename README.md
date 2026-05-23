# Physics World Model

**Live Platform: [physicsworldmodel.org](https://physicsworldmodel.org)** · **Block Explorer: [explorer.physicsworldmodel.org](https://explorer.physicsworldmodel.org)**

> 🪙 **PWM mainnet went live on Base on 2026-05-22.** 9 smart contracts deployed, Basescan-verified. Browse on-chain Principles, Specs, Benchmarks, and Certificates at [explorer.physicsworldmodel.org](https://explorer.physicsworldmodel.org).

[![Discussions](https://img.shields.io/github/discussions/integritynoble/Physics_World_Model?label=Discussions&logo=github)](https://github.com/integritynoble/Physics_World_Model/discussions)
[![Good First Issues](https://img.shields.io/github/issues/integritynoble/Physics_World_Model/good%20first%20issue?label=Good%20First%20Issues&color=7057ff)](https://github.com/integritynoble/Physics_World_Model/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
[![Contributing](https://img.shields.io/badge/Contributing-Guide-blue)](CONTRIBUTING.md)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/integritynoble/Physics_World_Model/blob/master/examples/PWM_Quickstart.ipynb)

---

**Physics World Model (PWM)** is an open protocol for AI4Science — providing verified **Solutions**, **Benchmarks**, **Specs**, and **Principles** for scientific domains where AI must respect physical laws.

PWM started from **Computational Imaging** — the richest domain for testing physics-aware AI — and is expanding to cover all domains where measurement physics governs what AI can know.

---

## The Four-Layer Protocol

Every PWM artifact lives in a strict hierarchy:

```
Principles
  └── Specs          (derived from Principles)
        └── Benchmarks     (measurable tests for each Spec)
              └── Solutions / Certificates   (verified implementations)
```

| Layer | What it is | Example |
|---|---|---|
| **Principle** | A physical law or invariant that constrains what AI solutions may do | "CASSI forward model: coded aperture × dispersive element" |
| **Spec** | A concrete, reproducible experimental specification derived from a Principle | Sensor geometry, mask pattern, noise model, wavelength range |
| **Benchmark** | A measurable test: dataset + metric + threshold | PSNR ≥ 32 dB on CASSI-28 at 4× compression |
| **Solution / Certificate** | A verified implementation that passes a Benchmark, cryptographically certified on-chain | `cert/0xabc…` — MST-L v2, PSNR 34.1 dB, verified 2026-05-22 |

All four layers are recorded on-chain on Base. Every Certificate is verifiable by anyone — no trust in a central authority required.

---

## Why Physics World Model?

Science needs benchmarks that cannot be gamed. Current AI leaderboards:

- Are hosted on private servers (can be edited or removed)
- Measure accuracy on static datasets (models overfit over time)
- Have no cryptographic verification of results
- Are siloed by domain (no shared protocol across physics disciplines)

PWM solves all four problems:

- **On-chain permanence** — Principles, Specs, Benchmarks, and Certificates are stored on Base. No central party can alter or erase them.
- **Prospective evaluation** — benchmarks are sealed before submission; data is not public until after the round closes.
- **Cryptographic certificates** — every verified Solution is a hash-linked on-chain Certificate. The verification is public and reproducible.
- **Domain-agnostic protocol** — the same four-layer stack applies to any physics domain. Computational Imaging is the founding domain; others follow.

---

## Founding Domain: Computational Imaging

Computational Imaging is the launching domain for PWM because it uniquely combines:

- **Forward physics** that is fully specifiable (optics, MRI, CT, acoustic, electron)
- **Inverse problems** where AI solutions must respect physical operators
- **Measurable ground truth** via simulation + calibrated lab instruments
- **Clinical stakes** (MRI, CT, OCT) that demand verified, auditable AI

### What this repo contains

This repository is the research toolkit and algorithm catalog for the Computational Imaging domain:

| Component | Description |
|---|---|
| **Harness** (OperatorGraph, 10 canonical primitives, 4-scenario protocol, LIP-Arena) | How imaging methods are tested |
| **Algorithm catalog** (172 modalities, 43+ solvers) | Current best methods |
| **Theoretical foundations** | FPB Theorem, Triad Decomposition |
| **Clinical CT QC Copilot** | Audit-grade CT quality assurance |
| **Benchmark results** | PSNR/SSIM tables for 26 modalities |

The on-chain protocol ([`mainnet/`](mainnet/)) and this research toolkit are complementary: the toolkit defines and evaluates methods; the protocol certifies and permanently records them.

---

## Getting Started

### Install

```bash
pip install -U pip
pip install -e packages/pwm_core
pip install -e packages/pwm_AI_Scientist
```

### Run a Reconstruction

```bash
# Microscopy
pwm run --prompt "SIM structured illumination, 3 angles, 3 phases, live cell"

# Compressive imaging
pwm run --prompt "CASSI spectral imaging, 28 bands, coded aperture"

# Medical imaging
pwm run --prompt "CT sparse view, 90 angles, low dose"
pwm run --prompt "MRI accelerated, 4x undersampling, parallel imaging"

# View results
pwm view runs/latest
```

### Evaluate a Method

```bash
# Score a method on a modality
pwm evaluate --method my_solver --modality cassi --track correct

# Run the full 4-scenario protocol
pwm evaluate --method my_solver --modality cassi --scenarios I,II,III,IV
```

---

## Modality Coverage

172 imaging modalities across 5 physical carriers:

| Carrier | Modalities (examples) |
|---|---|
| **Photon** | CT, OCT, FPM, SIM, CASSI, single-pixel, FLIM, light field, photoacoustic |
| **Spin (MRI)** | Parallel imaging, diffusion, spectroscopy, MRSI, MRF |
| **Electron** | SEM, TEM, STEM-EELS, 4D-STEM, ptychography, holography |
| **Acoustic** | Ultrasound, HIFU, full-waveform inversion |
| **Particle** | Neutron CT, muon tomography |

Full catalog: [`docs/modality_catalog.md`](docs/modality_catalog.md)

---

## Theoretical Foundations

PWM's algorithm toolkit is grounded in two theoretical results:

### Finite Primitive Basis (FPB) Theorem

Every imaging forward model admits an ε-approximate representation as a typed DAG over exactly **10 canonical primitives**: Propagate, Modulate, Project, Encode, Convolve, Accumulate, Detect, Sample, Disperse, Scatter.

Basis growth saturates at K=10 for N=31 modalities — no new primitive has been needed across 172 modalities.

### Triad Decomposition

Every reconstruction failure decomposes into three root causes:

| Gate | Name | Cause |
|---|---|---|
| Gate 1 | Recoverability | Null-space loss |
| Gate 2 | Carrier Budget | SNR floor |
| Gate 3 | Operator Mismatch | H_nominal ≠ H_true |

**Gate 3 dominates** across all validated modalities. Autonomous correction recovers +0.8 to +10.7 dB without retraining.

Reference: *"Ten Primitives and Three Gates: The Universal Structure of Computational Imaging"* (Yang & Yuan, 2026)

---

## On-Chain Protocol

The on-chain layer lives in [`mainnet/`](mainnet/):

- **9 smart contracts** on Base mainnet, Basescan-verified
- **531 genesis Principles** on Base Sepolia testnet
- **533 genesis Benchmarks** on Base Sepolia testnet
- Live explorer at [explorer.physicsworldmodel.org](https://explorer.physicsworldmodel.org)

Grant applications and funding strategy: [`grants/`](grants/)

---

## Roadmap

| Phase | Status | Description |
|---|---|---|
| Computational Imaging | **Live** | 531 Principles, 533 Benchmarks on-chain; 172-modality harness |
| Solution Submissions | **Testnet** | Open submission pipeline, prospective evaluation |
| Certificate Rewards | **Testnet** | Pool-weighted ETH rewards for rank-1 Solutions |
| Expansion Domains | **Planned** | Genomics, climate, materials science, particle physics |

Detailed roadmap at [physicsworldmodel.org/roadmap](https://physicsworldmodel.org/roadmap).

---

## Community & Contributing

Four ways to contribute:

| Level | What | Where |
|---|---|---|
| **Algorithm** | Submit a better solver | `contrib/solver_registry.yaml` + PR |
| **Modality** | Add a new imaging modality | `docs/modality_catalog.md` + PR |
| **Data** | Add calibration datasets | `DATA.md` + GCS bucket |
| **Protocol** | Propose a new Principle or Spec | [physicsworldmodel.org/contribute](https://physicsworldmodel.org/contribute) |

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

---

## Repository Layout

```
Physics_World_Model/
├── packages/           — pwm_core, pwm_AI_Scientist Python packages
├── mainnet/            — on-chain protocol: contracts, addresses, deploy logs
├── grants/             — grant applications and funding coordination
├── benchmarks/         — benchmark definitions and results
├── datasets/           — dataset adapters and registry
├── rails/              — SolveEverything gear implementations
├── docs/               — protocol and modality documentation
├── examples/           — notebooks and quickstart
├── contrib/            — solver registry, community contributions
└── papers/             — preprints and references
```

---

## License

MIT — see [LICENSE](LICENSE)

## Citation

```bibtex
@software{yang2026pwm,
  author  = {Yang, David},
  title   = {Physics World Model},
  year    = {2026},
  url     = {https://github.com/integritynoble/Physics_World_Model},
}
```
