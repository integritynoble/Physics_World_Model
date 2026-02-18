# Gear 5: Data Trusts -- The Lubricant

> Shared, safe data resource via legal data trusts.

**Status: FOUNDATION**

---

## The Principle

Data trusts provide a legal and technical framework for sharing data safely across organizations. Multiple labs can contribute measurements to a shared pool without losing control of their IP. The trust holds data on behalf of contributors and releases it under agreed conditions. This breaks the "data silo" problem: each lab has too few measurements to train robust models, but together they have enough.

---

## PWM Implementation

PWM's current data approach is **synthetic-first**: most benchmarks use on-the-fly generated data, requiring no data downloads or sharing agreements. For real-data benchmarks, PWM uses public datasets with proper citations. The foundation for data trusts exists in the registry and licensing infrastructure.

### Synthetic-First Policy

- All 64 modalities have synthetic data generators
- Benchmarks run without downloading external datasets
- Deterministic seeds ensure reproducibility across machines
- No large binaries in the repository

### Dataset Registry

The dataset registry maps modalities to their data sources:
- Public URLs with citations for real datasets (TSA_simu_data, LoDoPaB-CT, fastMRI, KAIST)
- Synthetic generation scripts for each modality
- Format specifications and shape/dtype declarations

### Open Core Licensing

PWM's open-core boundary defines what is freely available vs. what requires partnership:
- **Core (MIT)**: All operators, solvers, graph templates, registries, evaluation harness
- **Datasets**: Public datasets referenced by URL; contributed datasets under separate agreements
- **Challenge data**: Generated on-the-fly for each LIP-Arena round

### Data Provenance

Every RunBundle includes provenance fields tracking data origin:
- `dataset_id`: Which dataset was used
- `seed_set`: Random seeds for reproducibility
- `platform`: Hardware and software environment
- `git_hash`: Code version

---

## Key Files

| File | Description |
|------|-------------|
| `packages/pwm_core/contrib/modalities.yaml` | 64-modality registry (includes dataset references) |
| `community/OPEN_CORE_BOUNDARY.md` | Open-core licensing policy |
| `docs/contracts/runbundle_schema.md` | RunBundle schema with provenance fields |

---

## What's Built

- **64-modality dataset registry**: Every modality mapped to data sources (synthetic + real)
- **Synthetic data generators**: On-the-fly generation for all modalities; no download required
- **MIT-licensed core**: All evaluation infrastructure freely available
- **Data provenance in RunBundles**: dataset_id, seed_set, platform, git_hash tracked per run
- **Deterministic reproducibility**: Same seeds + same code = identical results

---

## What's Next

- **Data trust framework**: Define legal and technical terms for lab partnerships (contribution terms, access tiers, attribution requirements)
- **Privacy-preserving contributions**: Protocol for labs to contribute measurements while preserving patient/sample privacy (e.g., contribute calibration measurements without raw patient data)
- **Access tiers**: Public (synthetic, published datasets) / Partner (contributed real-data, under agreement) / Challenge (generated per-round, sequestered)
- **Data provenance chain**: Extend provenance beyond per-run to per-sample: who captured it, which instrument, when, what conditions
- **Federated evaluation**: Labs run the harness locally on their private data and submit only scores + RunBundles

---

## Connections

- **Gear 1 (Targeting System)**: The harness uses pooled data (synthetic + real) for evaluation
- **Gear 6 (Decision Logs)**: Data provenance is part of every RunBundle and DR-IS record
- **Gear 10 (Literacy)**: Documentation teaches users how to prepare and contribute datasets
