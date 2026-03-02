# CBCT Public Tier

Full-access data for algorithm development. 10 samples derived from
the most recent and widely-cited open-access CBCT/CT benchmark datasets
at 256^3, N_views=256. Includes ground truth volumes and true mismatch values.

## Source Datasets

| Sample | Source Dataset               | Year | Anatomy/Object   | Reference                              |
|--------|------------------------------|------|-------------------|----------------------------------------|
| 00     | AAPM Low-Dose CT (Mayo)      | 2017 | Abdomen           | McCollough et al., Med Phys 2017       |
| 01     | LIDC-IDRI / ICASSP 3D CBCT  | 2024 | Lung (nodule)     | Armato et al., 2011; Zhang et al., ICASSP 2024 |
| 02     | CBCTLiTS                     | 2024 | Liver (synth CBCT)| Huo et al., 2024                       |
| 03     | MMDental                     | 2025 | Dental CBCT       | Feng et al., Med Image Anal 2025       |
| 04     | CTooth+                      | 2022 | Dental CBCT       | Cui et al., 2022                       |
| 05     | 2DeteCT                      | 2023 | Industrial part   | Coban et al., Sci Data 2023            |
| 06     | Helsinki Tomography (HTC)    | 2022 | Acrylic disc      | Meaney et al., arXiv 2212.07671        |
| 07     | Walnut CT (CWI) / PCCT       | 2019 | Walnut            | Der Sarkissian, Sci Data 2019; Bauer et al., 2025 |
| 08     | CQ500                        | 2018 | Head CT           | Chilamkurthy et al., 2018              |
| 09     | DM4CT (Synchrotron)          | 2025 | Rock micro-CT     | Shao et al., 2025                      |

All source datasets are publicly available under open licenses. Volumes
are resampled to 256^3 and normalized to [0, 1] attenuation range.

## Why These Datasets

- **AAPM Mayo** (00): Gold standard clinical CT benchmark, thousands of citations
- **LIDC-IDRI / ICASSP 2024** (01): Largest public lung CT dataset (1010 CTs); used in the 2024 ICASSP 3D CBCT Reconstruction Challenge with simulated CBCT sinograms via ASTRA toolbox
- **CBCTLiTS** (02): 201 synthetic paired CBCT/CT volumes from LiTS (2024), 5 quality levels (32-490 projections), liver tumor segmentation benchmark
- **MMDental** (03): First and largest public CBCT dental dataset (660 patients, 2025), 3D dental CBCT with expert medical records
- **CTooth+** (04): 22 annotated + 146 unlabeled dental CBCT volumes for tooth segmentation; captures root/canal complexity
- **2DeteCT** (05): 5000-slice industrial CT (2023), 3 beam modes (high-fidelity, low-dose, beam-hardening), 750 OOD slices
- **HTC 2022** (06): Helsinki Tomography Challenge, 7 difficulty levels, limited-angle CT, known-geometry acrylic disc phantoms
- **Walnut CT / PCCT** (07): Micro-CT walnut dataset for ML (2019); also referenced in 2025 cone-beam photon-counting CT dataset (15 walnuts, dual energy thresholds, 172,800 raw projections)
- **CQ500** (08): 491 non-contrast head CT scans, open-access, hemorrhage/fracture/midline-shift labels
- **DM4CT** (09): Recent (Feb 2025) CT reconstruction benchmark with real-world rock samples from synchrotron, 10 diffusion methods + 7 classical baselines, data on Zenodo

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
