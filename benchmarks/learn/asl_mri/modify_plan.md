# Modify Plan -- asl_mri

## Algorithm Catalog Review

**Category:** medical | **Carrier:** Spin/RF | **Score key:** mri

| Algorithm | Type | Source |
|-----------|------|--------|
| Zero-Filled IFFT | Classical | Zbontar et al., arXiv 2018 |
| L1-Wavelet (ESPIRiT) | Compressed Sensing | Lustig et al., MRM 2007 |
| PnP-DnCNN | PnP | Ahmad et al., IEEE SPM 2020 |
| U-Net | Deep Learning | Zbontar et al., arXiv 2018 |
| E2E-VarNet | Deep Unrolling | Sriram et al., MICCAI 2020 |
| PromptMR | Deep Unrolling | Bai et al., ECCV 2024 |
| ReconFormer | Transformer | Guo et al., IEEE TMI 2024 |
| Score-MRI | Diffusion | Chung & Ye, Med. Image Anal. 2022 |

### Domain Appropriateness

**Excellent fit.** The carrier routing `("medical", "Spin/RF") -> "mri"` correctly sends ASL MRI to the dedicated MRI algorithm pool (the hand-crafted `mri` variant override). All 8 algorithms are real, published MRI reconstruction methods with correct citations:

- **Zero-Filled IFFT** -- Standard MRI baseline from fastMRI. Correct.
- **L1-Wavelet (ESPIRiT)** -- Lustig et al., MRM 2007 is the foundational compressed sensing MRI paper. Correct.
- **PnP-DnCNN** -- Ahmad et al., IEEE SPM 2020 is a real PnP MRI paper. Correct.
- **U-Net** -- Zbontar et al., arXiv 2018 (fastMRI baseline). Correct.
- **E2E-VarNet** -- Sriram et al., MICCAI 2020. Correct.
- **PromptMR** -- Bai et al., ECCV 2024. Correct.
- **ReconFormer** -- Guo et al., IEEE TMI 2024. Correct.
- **Score-MRI** -- Chung & Ye, Med. Image Anal. 2022. Correct.

All citations are real and verifiable.

**Minor note:** ASL MRI has unique perfusion quantification aspects (label/control subtraction, kinetic modeling) that go beyond standard MRI reconstruction. The algorithms here address the k-space undersampling reconstruction step, which is valid, but ASL-specific perfusion mapping methods (e.g., Buxton kinetic model fitting) are not represented.

### Learning Materials

`03_reconstruction_algorithms.md` lists "FBP" and "DL-Recon" as solvers, which are generic placeholders that do not match any of the 8 MRI algorithms on the leaderboard page. The learning materials need updating.

## Proposed Changes

1. **`03_reconstruction_algorithms.md`**: Replace "FBP" / "DL-Recon" with representative algorithms from the MRI pool (e.g., Zero-Filled IFFT, L1-Wavelet, E2E-VarNet).

No code changes needed in `_algorithm_catalog.py`. The carrier routing and MRI algorithm pool are correct and well-cited.

**Priority:** LOW -- only learning materials need sync.
