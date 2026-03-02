# CBCT Hidden Tier

Server-side evaluation only. 20 procedurally generated phantoms at 256^3,
N_views in {128, 256, 512}. Strongest mismatch. Never leaves PWM servers.

Hidden-tier phantoms are extreme stress-test versions of real CBCT dataset
characteristics. They push reconstruction algorithms to breaking point using
reaction-diffusion PDE solvers, gyroid/Schwarz-P TPMS surfaces, depth-6
L-system vascular trees, Worley noise with 300 seed points, fractal
boundary perturbation, and extreme multi-material dynamic range.

## Dataset Inspirations (Adversarial)

Hidden recipes take the most challenging features from recent CBCT datasets
and amplify them to create adversarial phantoms that stress-test every aspect
of reconstruction.

| Recipe              | Inspired By (extreme)                    | Adversarial Feature                               |
|---------------------|------------------------------------------|----------------------------------------------------|
| trabecular_micro    | Walnut CT / PCCT 2025                    | Gyroid + Schwarz-P + Worley porosity like walnut shell micro-structure |
| multi_metal         | AAPM Low-Dose CT + orthopedic implants   | Ti prosthesis + screws + plate + wire — extreme streaks |
| vascular_tree       | CBCTLiTS 2024 portal vasculature         | Depth-6 arterial + venous + portal + capillary bed  |
| lung_parenchyma     | LIDC-IDRI / ICASSP 2024 CBCT Challenge   | Bronchial tree + alveolar Worley + nodules + GGO    |
| fractal_membrane    | CBCTLiTS 2024 organ boundaries           | Fractal-boundary organs + thin peritoneal folds     |
| gyroid_scaffold     | 2DeteCT 2023 industrial + DM4CT 2025     | Multi-scale TPMS metamaterial + Worley cells + dense core |
| dental_metal        | MMDental 2025 / CTooth+ extreme          | Full dental arch + metal crowns + amalgam + ortho wire |
| cardiac_chambers    | AAPM Low-Dose CT cardiac                 | LV/RV/atria + coronary tree + calcifications        |
| multi_contrast      | 2DeteCT 2023 multi-beam modes            | All materials (air/fat/muscle/bone/metal) stacked   |
| reaction_diffusion  | DM4CT 2025 complex microstructure        | Turing-pattern 3D + periodic grid + Worley + metal  |

## Samples

| #  | Recipe              | Inspired By             | N_views | Difficulty | Key Features                                       |
|----|---------------------|-------------------------|---------|------------|----------------------------------------------------|
| 00 | trabecular_micro    | Walnut CT/PCCT          | 128     | hard       | Gyroid + Schwarz-P + Worley porosity + canals      |
| 01 | multi_metal         | AAPM + ortho implants   | 128     | hard       | Ti prosthesis + screws + plate + wire + amalgam    |
| 02 | vascular_tree       | CBCTLiTS 2024           | 128     | hard       | Depth-6 arterial + venous + portal + capillary bed |
| 03 | lung_parenchyma     | LIDC/ICASSP 2024        | 128     | hard       | Bronchial tree + alveolar Worley + nodules + GGO   |
| 04 | fractal_membrane    | CBCTLiTS 2024           | 256     | hard       | Fractal-boundary organs + thin peritoneal folds    |
| 05 | gyroid_scaffold     | 2DeteCT/DM4CT 2025      | 128     | hard       | Multi-scale TPMS + Worley + dense core             |
| 06 | dental_metal        | MMDental/CTooth+        | 256     | hard       | Dental arch + metal crowns + amalgam + ortho wire  |
| 07 | cardiac_chambers    | AAPM cardiac            | 128     | hard       | LV/RV/atria + coronary tree + calcifications       |
| 08 | multi_contrast      | 2DeteCT multi-beam      | 256     | hard       | All materials: air/fat/muscle/bone/metal stacked   |
| 09 | reaction_diffusion  | DM4CT 2025              | 128     | extreme    | Turing patterns + periodic grid + Worley + metal   |
| 10 | trabecular_micro    | Walnut CT/PCCT          | 256     | hard       | Gyroid trabecular (different seed, more views)     |
| 11 | multi_metal         | AAPM + ortho implants   | 256     | hard       | Multiple metal types (different seed)              |
| 12 | vascular_tree       | CBCTLiTS 2024           | 512     | hard       | Full vascular anatomy (different seed)             |
| 13 | lung_parenchyma     | LIDC/ICASSP 2024        | 256     | hard       | Lung with nodules (different seed)                 |
| 14 | fractal_membrane    | CBCTLiTS 2024           | 512     | hard       | Fractal organs (different seed)                    |
| 15 | gyroid_scaffold     | 2DeteCT/DM4CT 2025      | 256     | hard       | TPMS metamaterial (different seed)                 |
| 16 | dental_metal        | MMDental/CTooth+        | 128     | extreme    | Sparse-view dental metal — worst case              |
| 17 | cardiac_chambers    | AAPM cardiac            | 512     | hard       | Cardiac anatomy (different seed)                   |
| 18 | multi_contrast      | 2DeteCT multi-beam      | 128     | extreme    | Sparse-view extreme dynamic range                  |
| 19 | reaction_diffusion  | DM4CT 2025              | 256     | extreme    | Turing patterns (different seed)                   |

## Computational Cost per Sample

| Operation                         | Typical cost per phantom        |
|-----------------------------------|---------------------------------|
| fBm noise (8 octaves, 3-5 passes)| 3-5x gaussian_filter on 256^3   |
| Worley noise (100-300 pts)        | O(n_pts * N^3) distance evals   |
| Gyroid/Schwarz-P TPMS             | Trig on full 256^3 grid         |
| L-system branching (depth 5-6)    | 50-300 tube segment renders     |
| Reaction-diffusion PDE            | 250-400 iterations on 128^3     |
| Elastic deformation               | 3x gaussian_filter + interp     |
| Anisotropic noise (6 octaves)     | 6x directional gaussian_filter  |
| SDF eval + boolean ops            | Multiple 256^3 field evaluations|

~60-70% hard/extreme stress tests + ~30-40% hard with more views for fairness.

Per sample: `projections`, `geometry`, `mu_true` (256^3). True mismatch stored.
Mismatch: severe (source_offset_x=1.50, beam_hardening=0.12, scatter_fraction=0.08, etc.)
