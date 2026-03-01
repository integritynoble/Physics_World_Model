# CBCT Hidden Tier

Server-side evaluation only. 20 procedurally generated phantoms at 256^3,
N_views in {128, 256, 512}. Strongest mismatch. Never leaves PWM servers.

Hidden-tier phantoms use the most computationally intensive generation:
reaction-diffusion PDE solvers, gyroid/Schwarz-P TPMS surfaces, depth-6
L-system vascular trees, Worley noise with 300 seed points, fractal
boundary perturbation, and extreme multi-material dynamic range.

## Samples

| #  | Recipe              | N_views | Difficulty | Key Features                                       |
|----|---------------------|---------|------------|----------------------------------------------------|
| 00 | trabecular_micro    | 128     | hard       | Gyroid + Schwarz-P + Worley porosity + canals      |
| 01 | multi_metal         | 128     | hard       | Ti prosthesis + screws + plate + wire + amalgam    |
| 02 | vascular_tree       | 128     | hard       | Depth-6 arterial + venous + portal + capillary bed |
| 03 | lung_parenchyma     | 128     | hard       | Bronchial tree + alveolar Worley + nodules + GGO   |
| 04 | fractal_membrane    | 256     | hard       | Fractal-boundary organs + thin peritoneal folds    |
| 05 | gyroid_scaffold     | 128     | hard       | Multi-scale TPMS + Worley + dense core             |
| 06 | dental_metal        | 256     | hard       | Dental arch + metal crowns + amalgam + ortho wire  |
| 07 | cardiac_chambers    | 128     | hard       | LV/RV/atria + coronary tree + calcifications       |
| 08 | multi_contrast      | 256     | hard       | All materials: air/fat/muscle/bone/metal stacked   |
| 09 | reaction_diffusion  | 128     | extreme    | Turing patterns + periodic grid + Worley + metal   |
| 10 | trabecular_micro    | 256     | hard       | Gyroid trabecular (different seed, more views)     |
| 11 | multi_metal         | 256     | hard       | Multiple metal types (different seed)              |
| 12 | vascular_tree       | 512     | hard       | Full vascular anatomy (different seed)             |
| 13 | lung_parenchyma     | 256     | hard       | Lung with nodules (different seed)                 |
| 14 | fractal_membrane    | 512     | hard       | Fractal organs (different seed)                    |
| 15 | gyroid_scaffold     | 256     | hard       | TPMS metamaterial (different seed)                 |
| 16 | dental_metal        | 128     | extreme    | Sparse-view dental metal — worst case              |
| 17 | cardiac_chambers    | 512     | hard       | Cardiac anatomy (different seed)                   |
| 18 | multi_contrast      | 128     | extreme    | Sparse-view extreme dynamic range                  |
| 19 | reaction_diffusion  | 256     | extreme    | Turing patterns (different seed)                   |

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
