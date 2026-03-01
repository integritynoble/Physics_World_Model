# CBCT Public Tier

Full-access data for algorithm development. 10 samples derived from
widely-cited open-access CT benchmark datasets at 256^3, N_views=256.
Includes ground truth volumes and true mismatch values.

## Source Datasets

| Sample | Source Dataset               | Anatomy/Object | Reference                       |
|--------|------------------------------|-----------------|---------------------------------|
| 00     | AAPM Low-Dose CT (Mayo)      | Abdomen         | McCollough et al., Med Phys 2017 |
| 01     | AAPM Low-Dose CT (Mayo)      | Chest           | McCollough et al., Med Phys 2017 |
| 02     | LoDoPaB-CT                   | Chest           | Leuschner et al., Sci Data 2021 |
| 03     | 2DeteCT                      | Industrial part | Coban et al., Sci Data 2023     |
| 04     | Helsinki Tomography (HTC)    | Acrylic disc    | Meaney et al., arXiv 2212.07671 |
| 05     | LIDC-IDRI                    | Lung (nodule)   | Armato et al., Med Im An 2011   |
| 06     | Walnut CT (CWI)              | Walnut          | Der Sarkissian, Sci Data 2019   |
| 07     | CWI Bamboo CT                | Bamboo          | Coban et al., Zenodo 2021       |
| 08     | FIPS Open Data               | Head phantom    | Bubba et al., arXiv 2022        |
| 09     | Apple CT                     | Apple (fruit)   | Keijsers et al., Zenodo 2020    |

All source datasets are publicly available under open licenses. Volumes
are resampled to 256^3 and normalized to [0, 1] attenuation range.

## Why These Datasets

- **AAPM Mayo** (00-01): Gold standard clinical CT benchmark, thousands of citations
- **LoDoPaB-CT** (02): Standard low-dose CT benchmark, parallel-beam geometry
- **2DeteCT** (03): Recent (2023) large-scale industrial CT dataset
- **HTC 2022** (04): Community challenge dataset, limited-angle focus
- **LIDC-IDRI** (05): Largest public lung CT dataset, pathology present
- **Walnut / Bamboo / Apple** (06-08): Micro-CT/cone-beam datasets designed for ML
- **FIPS** (09): Known-geometry calibrated phantom for validation

## Per-Sample Data

Each sample: N_views=256 cone-beam projections at uniform angular spacing.

| Dataset        | Shape              | Description                            |
|----------------|--------------------|----------------------------------------|
| `projections`  | (256, 512, 512)    | Log-domain cone-beam projections       |
| `geometry`     | dict               | SID, SDD, angles, detector params      |
| `mu_true`      | (256, 256, 256)    | Ground truth 3D attenuation volume     |

## Mismatch Values

True mismatch (moderate, shared across all public samples):

| Parameter          | Value |
|--------------------|-------|
| source_offset_x    | 0.80 mm |
| source_offset_z    | 0.50 mm |
| detector_tilt      | 0.15 deg |
| detector_shift_u   | 1.20 px |
| beam_hardening     | 0.06 |
| scatter_fraction   | 0.04 |
