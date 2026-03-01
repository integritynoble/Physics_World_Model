# CBCT Dev Tier

Blind evaluation. 20 procedurally generated phantoms at 256^3,
N_views in {128, 256, 512}. Ground truth and mismatch values hidden.

Each phantom uses multi-layer procedural generation: 8-octave fBm,
Worley cellular noise, SDF-based geometry with smooth booleans,
L-system branching trees, anisotropic fiber textures, elastic
deformation, and realistic multi-material attenuation compositing.

## Samples

| #  | Recipe            | N_views | Difficulty | Key Features                                    |
|----|-------------------|---------|------------|-------------------------------------------------|
| 00 | head_cranial      | 256     | medium     | Skull + cortical folds + ventricles + sinuses   |
| 01 | torso_thorax      | 256     | medium     | Ribs + spine + lungs + heart + aorta tree       |
| 02 | abdomen_organs    | 256     | medium     | Liver + kidneys + bowel + mesenteric vessels    |
| 03 | extremity_bone    | 512     | medium     | Cortex + marrow + muscle fibers + fat lobules   |
| 04 | dental_arch       | 256     | medium-hard| 16 teeth + mandible + roots + tongue + nerves   |
| 05 | pelvis_hip        | 256     | medium     | Iliac bones + femoral heads + bladder + vessels |
| 06 | shoulder_complex  | 512     | medium     | Humerus + scapula + rotator cuff + vessels      |
| 07 | knee_joint        | 256     | medium     | Condyles + menisci + cartilage + patella        |
| 08 | spine_segment     | 512     | medium     | Vertebrae + discs + canal + paraspinal fibers   |
| 09 | hand_wrist        | 256     | medium     | Carpals + metacarpals + phalanges + tendons     |
| 10 | head_cranial      | 512     | medium     | Skull + brain folds (different seed)            |
| 11 | torso_thorax      | 512     | medium     | Ribs + lungs + aorta branching (different seed) |
| 12 | abdomen_organs    | 128     | medium-hard| Sparse-view abdomen reconstruction              |
| 13 | extremity_bone    | 128     | medium-hard| Sparse-view bone with fiber texture             |
| 14 | dental_arch       | 512     | medium     | Full dental anatomy (different seed)            |
| 15 | pelvis_hip        | 128     | medium-hard| Sparse-view pelvis                              |
| 16 | shoulder_complex  | 256     | medium     | Shoulder anatomy (different seed)               |
| 17 | knee_joint        | 128     | medium-hard| Sparse-view knee joint                          |
| 18 | spine_segment     | 256     | medium     | Spine segment (different seed)                  |
| 19 | hand_wrist        | 512     | medium     | Hand anatomy (different seed)                   |

Each recipe generates phantoms with:
- 20-60 structural elements (bones, organs, vessels)
- 3-5 noise texture layers at different scales
- Elastic deformation for organic boundary irregularity
- Anisotropic fiber textures for muscle/cortical bone
- L-system branching vascular trees (depth 3-4)
- Realistic HU-based multi-material attenuation

Per sample: `projections` (N_views, 512, 512), `geometry`. No ground truth.

Mismatch: mild (source_offset_x=0.50, beam_hardening=0.04, etc.)
