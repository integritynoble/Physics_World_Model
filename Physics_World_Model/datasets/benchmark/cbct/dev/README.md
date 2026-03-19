# CBCT Dev Tier

Blind evaluation. 20 procedurally generated phantoms at 256^3,
N_views in {128, 256, 512}. Ground truth and mismatch values hidden.

Each phantom is explicitly inspired by the structural characteristics found in
recent, high-impact CBCT/CT datasets from the public tier. The procedural
generator (~1700 lines) uses multi-layer synthesis: 8-octave fBm, Worley
cellular noise, SDF-based geometry with smooth booleans, L-system branching
trees, anisotropic fiber textures, elastic deformation, gyroid/Schwarz-P TPMS
surfaces, and realistic multi-material attenuation compositing.

## Dataset Inspirations

Dev recipes simulate the types of anatomy and objects found in real CBCT
datasets, enabling evaluation on anatomically-informed but novel test volumes.

| Recipe            | Inspired By                       | Key Characteristics Simulated                      |
|-------------------|-----------------------------------|----------------------------------------------------|
| head_cranial      | CQ500 (2018)                      | Skull vault + cortical folds + ventricles + sinuses |
| torso_thorax      | AAPM Low-Dose CT / ICASSP 2024    | Ribs + spine + lung parenchyma + heart + aorta      |
| abdomen_organs    | CBCTLiTS (2024)                   | Liver + kidneys + bowel loops + mesenteric vessels   |
| extremity_bone    | LoDoPaB-CT (2021)                 | Cortex + marrow + 4 muscle bundles (fiber texture)   |
| dental_arch       | MMDental (2025) / CTooth+         | 16 teeth (enamel/dentin/pulp) + mandible + nerves    |
| pelvis_hip        | AAPM Low-Dose CT (2017)           | Iliac bones + sacrum + femoral heads + bladder       |
| shoulder_complex  | AAPM Low-Dose CT (2017)           | Humerus + scapula + rotator cuff + clavicle          |
| knee_joint        | LoDoPaB-CT (2021)                 | Condyles + menisci + cartilage + patella              |
| spine_segment     | AAPM Low-Dose CT (2017)           | 3-4 vertebrae + discs + canal + paraspinal fibers    |
| hand_wrist        | Helsinki Tomography (2022)        | 8 carpals + 5 metacarpals + phalanges + tendons      |

## Samples

| #  | Recipe            | Inspired By         | N_views | Difficulty | Key Features                                    |
|----|-------------------|---------------------|---------|------------|-------------------------------------------------|
| 00 | head_cranial      | CQ500               | 256     | medium     | Skull + cortical folds + ventricles + sinuses   |
| 01 | torso_thorax      | AAPM/ICASSP 2024    | 256     | medium     | Ribs + spine + lungs + heart + aorta tree       |
| 02 | abdomen_organs    | CBCTLiTS 2024       | 256     | medium     | Liver + kidneys + bowel + mesenteric vessels    |
| 03 | extremity_bone    | LoDoPaB-CT          | 512     | medium     | Cortex + marrow + muscle fibers + fat lobules   |
| 04 | dental_arch       | MMDental/CTooth+    | 256     | medium-hard| 16 teeth + mandible + roots + tongue + nerves   |
| 05 | pelvis_hip        | AAPM Low-Dose       | 256     | medium     | Iliac bones + femoral heads + bladder + vessels |
| 06 | shoulder_complex  | AAPM Low-Dose       | 512     | medium     | Humerus + scapula + rotator cuff + vessels      |
| 07 | knee_joint        | LoDoPaB-CT          | 256     | medium     | Condyles + menisci + cartilage + patella        |
| 08 | spine_segment     | AAPM Low-Dose       | 512     | medium     | Vertebrae + discs + canal + paraspinal fibers   |
| 09 | hand_wrist        | HTC 2022            | 256     | medium     | Carpals + metacarpals + phalanges + tendons     |
| 10 | head_cranial      | CQ500               | 512     | medium     | Skull + brain folds (different seed)            |
| 11 | torso_thorax      | AAPM/ICASSP 2024    | 512     | medium     | Ribs + lungs + aorta branching (different seed) |
| 12 | abdomen_organs    | CBCTLiTS 2024       | 128     | medium-hard| Sparse-view abdomen reconstruction              |
| 13 | extremity_bone    | LoDoPaB-CT          | 128     | medium-hard| Sparse-view bone with fiber texture             |
| 14 | dental_arch       | MMDental/CTooth+    | 512     | medium     | Full dental anatomy (different seed)            |
| 15 | pelvis_hip        | AAPM Low-Dose       | 128     | medium-hard| Sparse-view pelvis                              |
| 16 | shoulder_complex  | AAPM Low-Dose       | 256     | medium     | Shoulder anatomy (different seed)               |
| 17 | knee_joint        | LoDoPaB-CT          | 128     | medium-hard| Sparse-view knee joint                          |
| 18 | spine_segment     | AAPM Low-Dose       | 256     | medium     | Spine segment (different seed)                  |
| 19 | hand_wrist        | HTC 2022            | 512     | medium     | Hand anatomy (different seed)                   |

Each recipe generates phantoms with:
- 20-60 structural elements (bones, organs, vessels)
- 3-5 noise texture layers at different scales
- Elastic deformation for organic boundary irregularity
- Anisotropic fiber textures for muscle/cortical bone
- L-system branching vascular trees (depth 3-4)
- Realistic HU-based multi-material attenuation

Per sample: `projections` (N_views, 512, 512), `geometry`. No ground truth.

Mismatch: mild (source_offset_x=0.50, beam_hardening=0.04, etc.)
