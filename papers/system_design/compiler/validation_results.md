# Constrained Primitive Compiler Validation Results

**Total modalities**: 36
**Pass rate**: 100.0%
**Chain fidelity**: 86.1%

| Modality | Expected Chain | Compiled Chain | Valid | Match | Carrier | Level |
|----------|---------------|----------------|-------|-------|---------|-------|
| brillouin | M -> R -> D | M -> R -> D | PASS | Yes | photon | exotic |
| cacti | M -> Sigma -> D | M -> D -> D | PASS | No | photon | full |
| cassi | M -> W -> Sigma -> D | M -> W -> D -> D | PASS | No | photon | full |
| cbct | Pi -> Lambda -> D | Pi -> Lambda -> D | PASS | Yes | xray | template |
| compton | M -> R -> D | M -> R -> D | PASS | Yes | xray | exotic |
| ct | Pi -> D | Pi -> D | PASS | Yes | xray | full |
| ct_polychromatic | Pi -> Lambda -> D | Pi -> Lambda -> D | PASS | Yes | xray | template |
| dexa | M -> Pi -> D | M -> Pi -> D | PASS | Yes | xray | template |
| doppler_us | P -> D | P -> D | PASS | Yes | acoustic | template |
| dot | M -> R -> P -> R -> D | M -> R -> P -> R -> D | PASS | Yes | photon | exotic |
| elastography | P -> P -> D | P -> P -> D | PASS | Yes | acoustic | template |
| electron_ptycho | M -> P -> D | M -> P -> D | PASS | Yes | electron | held_out |
| fluorescence | M -> R -> D | M -> R -> D | PASS | Yes | photon | exotic |
| fluorescence_saturated | M -> R -> Lambda -> D | M -> R -> Lambda -> D | PASS | Yes | photon | template |
| ghost_imaging | M -> Sigma -> D | M -> D -> D | PASS | No | photon | exotic |
| lensless | C -> D | C -> D | PASS | Yes | photon | full |
| mri | M -> F -> S -> D | M -> F -> S -> D | PASS | Yes | spin | full |
| mri_phase_wrapped | M -> F -> S -> Lambda -> D | M -> F -> S -> Lambda -> D | PASS | Yes | spin | template |
| muon_tomography | R -> Pi -> D | R -> Pi -> D | PASS | Yes | particle | template |
| oct | P + P -> Sigma -> D | P -> P -> D -> D | PASS | No | photon | held_out |
| palm_storm | M -> C -> D | M -> C -> D | PASS | Yes | photon | template |
| pet | Pi -> D | Pi -> D | PASS | Yes | xray | template |
| phase_contrast | Pi -> P -> M -> D | Pi -> P -> M -> D | PASS | Yes | xray | held_out |
| photoacoustic | M -> P -> D | M -> P -> D | PASS | Yes | acoustic | held_out |
| proton_therapy | Lambda -> Pi -> D | Lambda -> Pi -> D | PASS | Yes | particle | template |
| ptychography | M -> P -> D | M -> P -> D | PASS | Yes | photon | full |
| raman | M -> R -> D | M -> R -> D | PASS | Yes | photon | exotic |
| sem | M -> D | M -> D | PASS | Yes | electron | template |
| sim | M -> C -> D | M -> C -> D | PASS | Yes | photon | held_out |
| spc | M -> Sigma -> D | M -> D -> D | PASS | No | photon | full |
| spect | Pi -> S -> D | Pi -> S -> D | PASS | Yes | xray | template |
| sted | M -> C -> D | M -> C -> D | PASS | Yes | photon | template |
| tem | M -> C -> D | M -> C -> D | PASS | Yes | electron | template |
| thz_tds | C -> D | C -> D | PASS | Yes | photon | exotic |
| tirf | P -> C -> D | P -> C -> D | PASS | Yes | photon | template |
| ultrasound | P -> D | P -> D | PASS | Yes | acoustic | template |
