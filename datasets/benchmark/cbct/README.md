# CBCT — Cone-Beam Computed Tomography

## Overview

Cone-Beam Computed Tomography (CBCT) is a 3D imaging technique that
reconstructs a volumetric attenuation map from a set of 2D cone-beam
X-ray projections acquired at multiple angles around the object.

This package provides benchmark data for evaluating reconstruction algorithms
under **forward-model mismatch**.

## Forward Model

```
p(theta, u, v) = gain * integral[ mu(r(t)) dt ] + offset + noise
```

Where:
- `mu(x,y,z)` is the 3D attenuation volume (ground truth)
- `p(theta, u, v)` is the log-domain projection at angle theta, detector pixel (u, v)
- The integral is along the ray from X-ray source through detector pixel (u, v)
- `gain` and `offset` model detector non-uniformity
- `noise` is measurement noise (Poisson + Gaussian)

## Dataset Design

| Tier   | Source                        | Volume | Samples | N_views         | Access         |
|--------|-------------------------------|--------|---------|-----------------|----------------|
| Public | Real CT benchmark datasets    | 256^3  | 10      | 256             | Full (GT+spec) |
| Dev    | Procedural (anatomy-inspired) | 256^3  | 20      | 128, 256, 512   | Blind (p+spec) |
| Hidden | Procedural (adversarial)      | 256^3  | 20      | 128, 256, 512   | Server-only    |

**Public** uses slices/volumes from widely-cited open-access CT datasets.

**Dev/Hidden are generated procedurally** using a computationally intensive
generator (~1200 lines) that produces phantoms with: multi-scale fBm + Worley
cellular noise, gyroid/Schwarz-P TPMS surfaces, L-system vascular trees,
reaction-diffusion Turing patterns, elastic deformation fields, anisotropic
fiber textures, and SDF-based multi-material compositing. The generator
code + secret seeds (kept private on PWM servers) fully determine each sample.

## Procedural Phantom Types

### Dev Recipes (anatomy-inspired, medium-to-high complexity)

| ID | Name              | Features                                                    |
|----|-------------------|-------------------------------------------------------------|
| 0  | head_cranial      | Skull + cortical folds + ventricles + sinuses + scalp       |
| 1  | torso_thorax      | Ribs (tori) + spine + lung parenchyma + heart + aorta tree  |
| 2  | abdomen_organs    | Liver + kidneys + spleen + bowel loops + mesenteric vessels  |
| 3  | extremity_bone    | Cortex + marrow + 4 muscle bundles (fiber texture) + fat    |
| 4  | dental_arch       | 16 teeth (enamel/dentin/pulp) + mandible + tongue + nerves  |
| 5  | pelvis_hip        | Iliac bones + sacrum + femoral heads + bladder + vessels    |
| 6  | shoulder_complex  | Humerus + glenoid + scapula + clavicle + rotator cuff       |
| 7  | knee_joint        | Condyles + menisci + cartilage + patella + popliteal tree   |
| 8  | spine_segment     | 3-4 vertebrae + discs + canal + processes + paraspinal      |
| 9  | hand_wrist        | 8 carpals + 5 metacarpals + phalanges + tendons + forearm   |

### Hidden Recipes (adversarial stress-tests, extreme complexity)

| ID | Name                | Features                                                       |
|----|---------------------|----------------------------------------------------------------|
| 0  | trabecular_micro    | Gyroid + Schwarz-P + Worley porosity + Haversian canals        |
| 1  | multi_metal         | Ti prosthesis + screws + plate + wire + amalgam — streaks      |
| 2  | vascular_tree       | Aorta + veins + portal system (depth-6 branching) + capillary  |
| 3  | lung_parenchyma     | Bronchial tree + alveolar Worley + nodules + ground-glass      |
| 4  | fractal_membrane    | Fractal-boundary organs + ultra-thin peritoneal folds          |
| 5  | gyroid_scaffold     | Multi-scale TPMS metamaterial + Worley cells + dense core      |
| 6  | dental_metal        | Full dental arch + metal crowns + amalgam + orthodontic wire   |
| 7  | cardiac_chambers    | LV/RV/atria + coronary tree + valve calcifications             |
| 8  | multi_contrast      | All materials (air/fat/muscle/bone/metal) + fiber + trabecular |
| 9  | reaction_diffusion  | Turing-pattern 3D + periodic grid + Worley + metal inclusions  |

## Computational Complexity per Phantom

Each phantom involves multiple expensive operations stacked:
- 3-5 passes of 8-octave 3D fBm noise (256^3 per pass)
- Worley cellular noise with 100-300 seed points
- Gyroid/Schwarz-P TPMS surface evaluation on full grid
- L-system branching trees (depth 4-6, hundreds of tube segments)
- Reaction-diffusion PDE solver (250-400 iterations)
- Elastic deformation via 3D displacement field interpolation
- Anisotropic directional noise for fiber textures
- SDF evaluation and smooth boolean operations

## N_views Distribution

- **Dev**: 40% N=256, 30% N=512, 30% N=128
- **Hidden**: 40% N=128, 30% N=256, 30% N=512 (sparser views, harder)

## Mismatch Parameters

| Parameter          | Range            | Dev (mild)  | Hidden (severe) |
|--------------------|------------------|-------------|-----------------|
| `source_offset_x`  | [-2.0, 2.0] mm  | 0.50        | 1.50            |
| `source_offset_z`  | [-1.5, 1.5] mm  | 0.30        | 1.00            |
| `detector_tilt`    | [-0.5, 0.5] deg | 0.10        | 0.35            |
| `detector_shift_u` | [-3.0, 3.0] px  | 0.80        | 2.20            |
| `beam_hardening`   | [0.0, 0.15]     | 0.04        | 0.12            |
| `scatter_fraction` | [0.0, 0.10]     | 0.03        | 0.08            |

## Geometry

- Source-to-isocenter distance (SID): 600 mm
- Source-to-detector distance (SDD): 1200 mm
- Detector size: 512 x 512 pixels
- Detector pixel pitch: 0.8 mm
- Volume: 256 x 256 x 256 voxels
- Voxel size: 0.5 mm isotropic

## Attenuation Scale (normalized)

| Material        | mu value | Approx. HU  |
|-----------------|----------|--------------|
| Air             | 0.00     | -1000        |
| Lung            | 0.06     | -700         |
| Fat             | 0.18     | -100         |
| Water           | 0.22     | 0            |
| Soft tissue     | 0.25     | +40          |
| Blood           | 0.26     | +55          |
| Muscle          | 0.28     | +60          |
| Cartilage       | 0.30     | +100         |
| Cancellous bone | 0.55     | +400         |
| Cortical bone   | 0.80     | +1000        |
| Enamel          | 0.92     | +2500        |
| Metal           | 1.00     | +3000+       |

## Scoring

```
Score = 0.4 x PSNR_norm + 0.4 x SSIM + 0.2 x Consistency
```

## References

- Feldkamp, L.A., Davis, L.C., Kress, J.W. "Practical cone-beam algorithm." JOSA A 1984.
- Buzug, T.M. "Computed Tomography." Springer 2008.
- McCollough, C.H. et al. "Low-dose CT for the detection and classification of metastatic liver lesions: Results of the 2016 Low Dose CT Grand Challenge." Med. Phys. 2017.
- Coban, S.B. et al. "2DeteCT — A large 2D expandable, trainable, experimental CT dataset." Sci. Data 2023.
- Meaney, A. et al. "Helsinki Tomography Challenge 2022." arXiv 2212.07671.
- Leuschner, J. et al. "LoDoPaB-CT, a benchmark dataset for low-dose CT." Sci. Data 2021.
- Der Sarkissian, H. et al. "A cone-beam X-ray CT data collection designed for ML." Sci. Data 2019. (Walnut CT)
- PWM Benchmark: https://pwm.platformai.org/benchmark/cbct
