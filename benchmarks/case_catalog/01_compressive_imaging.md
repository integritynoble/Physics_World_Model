# Category 1: Compressive Imaging — Full Case Catalog

> 4 modalities, 28 system variants, 7,168 total test instances

---

## 1.1 CASSI — Coded Aperture Snapshot Spectral Imaging

### System Variants (9)

| # | Variant | DAG (11 Primitives) | Physical Elements | Key Difference |
|---|---------|---------------------|-------------------|----------------|
| 1 | **SD-CASSI** | Src(halogen) -> M(binary_mask) -> W(prism_SF11) -> Sigma(spectral_sum) -> D(CCD_12bit) | 1 coded aperture + 1 prism + relay lens + FPA | Single disperser after mask; simplest; most benchmarked |
| 2 | **DD-CASSI** | Src(halogen) -> W(prism1) -> M(binary_mask) -> W(prism2_inverse) -> D(CCD_12bit) | 2 prisms + 1 coded aperture + relay optics + FPA | Shear-code-unshear; best fidelity; complex alignment |
| 3 | **R-CASSI** | Src(halogen) -> M(reflective_mask) -> W(prism_folded) -> D(CCD_12bit) | 1 prism (folded path) + beam splitter + reflective mask + FPA | Compact folded DD-CASSI; Yu et al. Opt. Express 2022 |
| 4 | **SCCSI** | Src(halogen) -> W(prism) -> M(color_CFA) -> Sigma(spectral_sum) -> D(CCD_12bit) | 1 prism (before mask) + color-coded aperture + FPA | Reversed disperser order; +6 dB over SD-CASSI |
| 5 | **Multi-frame CASSI** | Src(halogen) -> M(binary_mask_piezo) -> W(prism) -> Sigma -> D(CCD_12bit) | SD-CASSI base + piezo translation stage (Newport) | N shifted snapshots (N=1,10,20,24) |
| 6 | **DCCHI (Dual-Camera)** | Src -> [M(mask)->W(prism)->Sigma->D1(CASSI); D2(PAN_camera)] | Beam splitter + SD-CASSI arm + panchromatic arm | Side information from PAN branch; +8 dB |
| 7 | **D-CASSI (Differential)** | Src -> M(complementary_mask_pair) -> W(prism) -> Sigma -> D | 2 complementary masks (+1/-1) + single disperser | Higher SNR from differential encoding |
| 8 | **SS-CASSI (Spatial-Spectral)** | Src -> [W -> M -> D] or [M -> W -> D] (flexible) | Flexible mask placement between spectral and sensor plane | Additional design degrees of freedom |
| 9 | **Zero-order CASSI** | Src -> M(binary_mask) -> W(grating) -> [D_0th + D_+1st + D_-1st] | Diffraction grating (not prism) + capture all orders | 0th order provides un-dispersed spatial info |

### Image Sizes (4)

| # | Label | Spatial (H x W) | Spectral Bands | Measurement Size | Note |
|---|-------|-----------------|----------------|------------------|------|
| 1 | Small | 128 x 128 | 28 | 128 x 182 | Quick validation |
| 2 | Standard | 256 x 256 | 28 | 256 x 310 | De facto benchmark (KAIST test) |
| 3 | Large | 512 x 512 | 28 | 512 x 566 | High-res evaluation |
| 4 | Real-hw | 660 x 660 | 28 | 660 x 714 | TSA-Net real hardware size |

### Compression Ratios (4)

| # | Snapshots | Ratio (shots/bands) | Effective Compression |
|---|-----------|---------------------|----------------------|
| 1 | 1 shot | 1/28 | ~23x (snapshot) |
| 2 | 2 shots | 2/28 | ~11.5x |
| 3 | 4 shots | 4/28 | ~5.8x |
| 4 | 8 shots | 8/28 | ~2.9x |

### Noise Levels (4)

| # | Level | Shot Noise (photons) | Read Noise (e-) | SNR (dB) |
|---|-------|---------------------|------------------|----------|
| 1 | Clean | inf | 0 | >60 |
| 2 | Low | 10000 | 2.0 | ~40 |
| 3 | Medium | 1000 | 5.0 | ~30 |
| 4 | High | 100 | 15.0 | ~20 |

### Mismatch Parameters (7)

| # | Parameter | Nominal | Range | Unit | Primitive Affected |
|---|-----------|---------|-------|------|--------------------|
| 1 | Mask shift dx | 0 | [-3.0, 3.0] | px | M (Modulate) |
| 2 | Mask shift dy | 0 | [-3.0, 3.0] | px | M (Modulate) |
| 3 | Mask rotation theta | 0 | [-2.0, 2.0] | deg | M (Modulate) |
| 4 | Dispersion slope a1 | 2.0 | [1.5, 2.5] | px/band | W (Disperse) |
| 5 | Dispersion offset alpha | 0 | [-0.5, 0.5] | px | W (Disperse) |
| 6 | Gain | 1.0 | [0.9, 1.1] | - | D (Detect) |
| 7 | Read noise | 5.0 | [1.0, 15.0] | e- | D (Detect) |

### B1 Cases: Design (Prompt -> Spec)

Each variant x prompt difficulty = 9 x 4 = **36 prompt cases**

| Difficulty | Example Prompt | Expected Spec Content |
|------------|---------------|----------------------|
| Easy | "Design a CASSI system for 28-band hyperspectral imaging at 256x256" | SD-CASSI, standard params |
| Medium | "Design a snapshot spectral imager with <-20 dB crosstalk, 450-650nm, binary mask" | Must choose SD vs DD vs R-CASSI |
| Hard | "Design a dual-disperser spectral imager that fits in 10cm optical path with reflective elements" | R-CASSI or folded DD-CASSI |
| Adversarial | "Design a CASSI with 100 spectral bands in single shot at 1024x1024 with SNR>40dB" | Must flag infeasibility / tradeoffs |

**B1 Total**: 9 variants x 4 difficulties = **36 cases**
**B1 with multi-round versions**: 36 x 3 rounds = **108 cases**

### B2 Cases: Forward + Reconstruct (Spec -> Reconstruction)

Dimensions: variants(9) x sizes(4) x ratios(4) x noise(4) x mismatch_level(5) = **2,880 cases**

| Mismatch Level | # Params Perturbed | Description |
|----------------|-------------------|-------------|
| M0 (nominal) | 0 | Perfect forward model |
| M1 (single) | 1 | One of 7 params perturbed |
| M2 (compound) | 3 | Three params simultaneously |
| M3 (real) | all | Real calibration errors (from TSA-Net data) |
| M4 (adversarial) | all | Worst-case injection optimized to max failure |

### B3 Cases: System Identification (Dataset + Prompt -> Spec)

Same combinatorial space as B2: **2,880 cases**

Each case has:
- **true-spec**: all 7 mismatch params with exact values
- **given-spec**: contestant knows only parameter ranges
- **dataset**: measurement Y + sensing matrix A + metadata

### B4 Cases: Correct + Diagnose (Dataset + Spec -> Correction)

Same combinatorial space: **2,880 cases**

Each case additionally has:
- **true-spec** with improvement suggestions
- Feedback on how to improve the physical system

### CASSI Data Sources

| Variant | Source | Label | Reference |
|---------|--------|-------|-----------|
| SD-CASSI (256x256x28) | KAIST 10 test scenes | `WEB` | Cai et al., MST, CVPR 2022 |
| SD-CASSI (256x256x28) | CAVE 32 training scenes | `WEB` | Columbia Univ. Multispectral DB |
| SD-CASSI (660x660x28) | TSA-Net real 5 scenes | `WEB` | Miao et al., ICCV 2019 |
| DD-CASSI | Gehm 2007 original data | `EXP` | Gehm et al., Opt. Express 2007 |
| R-CASSI | Yu et al. real data | `WEB` | Yu et al., Opt. Express 2022 |
| DCCHI | In2SET benchmark | `WEB` | arXiv 2312.13319 |
| SCCSI, D-CASSI, SS-CASSI | Generated from CAVE/KAIST + variant-specific forward models | `GEN` | Forward model code |
| Multi-frame | Generated with piezo shift simulation | `GEN` | Duke DISP framework |
| Zero-order | Generated with grating diffraction model | `GEN` | Custom code |

### CASSI Case Count Summary

| Benchmark | Formula | Count |
|-----------|---------|-------|
| B1 | 9 variants x 4 difficulties x 3 rounds | **108** |
| B2 | 9 variants x 4 sizes x 4 ratios x 4 noise x 5 mismatch | **2,880** |
| B3 | same as B2 | **2,880** |
| B4 | same as B2 | **2,880** |
| **CASSI Total** | | **8,748** |

---

## 1.2 SPC — Single-Pixel Camera

### System Variants (6)

| # | Variant | DAG | Physical Elements | Key Difference |
|---|---------|-----|-------------------|----------------|
| 1 | **Hadamard SPC** | Src(LED) -> M(DMD_Hadamard) -> Sigma(bucket_sum) -> D(photodiode) | DMD (TI DLP) + collection lens + single photodiode | Binary Hadamard patterns; noise-robust |
| 2 | **Fourier SPC** | Src(LED) -> M(DMD_sinusoidal) -> Sigma(bucket_sum) -> D(photodiode) | DMD with sinusoidal patterns + photodiode | Frequency-domain sampling |
| 3 | **Gaussian Random SPC** | Src(LED) -> M(DMD_random) -> Sigma(bucket_sum) -> D(photodiode) | Random Gaussian measurement matrix | CS-theory optimal; RIP guarantee |
| 4 | **Hyperspectral SPC** | Src(broadband) -> M(DMD_Hadamard) -> Sigma -> D(spectrometer) | DMD + spectrometer (not photodiode) | Spectral cube: 64x64x2048 |
| 5 | **Adaptive SPC** | Src -> M(DMD_wavelet_adaptive) -> Sigma -> D(photodiode) | Adaptive basis scan by wavelet prediction | Fewer measurements needed |
| 6 | **Compressive Ghost Imaging** | Src(thermal/SPDC) -> M(spatial_correlations) -> Sigma -> D(bucket) | Correlated photon pairs or pseudo-thermal source | Quantum/classical ghost imaging variant |

### Image Sizes (4)

| # | Spatial | Measurement Count Range | Note |
|---|---------|------------------------|------|
| 1 | 32 x 32 | 102-1024 | Quick test |
| 2 | 64 x 64 | 410-4096 | Standard SPC benchmark |
| 3 | 128 x 128 | 1638-16384 | Medium scale |
| 4 | 256 x 256 | 6554-65536 | Large scale |

### Compression Ratios (5)

| # | Sampling Rate | Measurements / Pixels |
|---|--------------|----------------------|
| 1 | 1% | 1/100 |
| 2 | 5% | 1/20 |
| 3 | 10% | 1/10 |
| 4 | 25% | 1/4 |
| 5 | 50% | 1/2 |

### Noise Levels (4)

| # | Level | Gain Drift | Measurement Noise sigma_y |
|---|-------|-----------|--------------------------|
| 1 | Clean | 1.0 (none) | 0.0 |
| 2 | Low | 0.98-1.02 | 0.01 |
| 3 | Medium | 0.90-1.10 | 0.05 |
| 4 | High | 0.80-1.20 | 0.10 |

### Mismatch Parameters (3)

| # | Parameter | Nominal | Range | Unit |
|---|-----------|---------|-------|------|
| 1 | Gain drift alpha | 1.0 | [0.8, 1.2] | - |
| 2 | Measurement noise sigma_y | 0.01 | [0, 0.1] | - |
| 3 | Pattern error (bit flips) | 0 | [0, 1%] | fraction |

### SPC Case Count

| Benchmark | Formula | Count |
|-----------|---------|-------|
| B1 | 6 variants x 4 difficulties x 3 rounds | **72** |
| B2 | 6 x 4 sizes x 5 ratios x 4 noise x 5 mismatch | **2,400** |
| B3 | same | **2,400** |
| B4 | same | **2,400** |
| **SPC Total** | | **7,272** |

### SPC Data Sources

| Source | Label | Reference |
|--------|-------|-----------|
| OpenSpyrit SPIHIM 64x64x2048 | `WEB` | https://github.com/openspyrit/spihim |
| STL-10 (96x96 natural images) | `WEB` | https://cs.stanford.edu/~acoates/stl10/ |
| Set11, BSD68, Urban100 | `SYN-WEB` | Standard test image sets |
| Generated Hadamard/Fourier/Random | `GEN` | measurement = Phi @ x + noise |

---

## 1.3 CACTI — Coded Aperture Compressive Temporal Imaging

### System Variants (5)

| # | Variant | DAG | Physical Elements | Key Difference |
|---|---------|-----|-------------------|----------------|
| 1 | **Grayscale CACTI** | Src(scene) -> M(shifting_binary_mask) -> Sigma(temporal_sum) -> D(CCD) | Binary mask on translation stage + CCD | N video frames compressed into 1 |
| 2 | **Color CACTI (Bayer)** | Src(scene) -> M(shifting_mask) -> Sigma(temporal_sum) -> D(Bayer_CCD) | Same + Bayer color filter array | Color video reconstruction |
| 3 | **Dual-mask CACTI** | Src -> M(mask1) -> M(mask2) -> Sigma -> D | Two masks at different planes | More spatial diversity |
| 4 | **Coded Exposure SCI** | Src -> M(electronic_shutter_pattern) -> Sigma -> D | Electronic shutter modulation (no moving parts) | Simpler; no mechanical mask |
| 5 | **Spectral-temporal CACTI** | Src -> M(mask) -> W(prism) -> Sigma -> D | Mask + disperser -> joint spectral-temporal | Hyperspectral video |

### Image Sizes (3)

| # | Spatial | Temporal Frames | Measurement Size |
|---|---------|----------------|------------------|
| 1 | 256 x 256 | 8 | 256 x 256 (single frame) |
| 2 | 512 x 512 | 8 | 512 x 512 |
| 3 | 1024 x 1024 | 8-32 | 1024 x 1024 |

### Compression Ratios (4)

| # | Frames Compressed | Ratio |
|---|------------------|-------|
| 1 | 8 | 8x |
| 2 | 16 | 16x |
| 3 | 24 | 24x |
| 4 | 32 | 32x |

### CACTI Case Count

| Benchmark | Formula | Count |
|-----------|---------|-------|
| B1 | 5 variants x 4 difficulties x 3 rounds | **60** |
| B2 | 5 x 3 sizes x 4 ratios x 4 noise x 5 mismatch | **1,200** |
| B3 | same | **1,200** |
| B4 | same | **1,200** |
| **CACTI Total** | | **3,660** |

### CACTI Data Sources

| Source | Label | Reference |
|--------|-------|-----------|
| Kobe, Runner, Drop, Traffic, Aerial, Vehicle (256x256x8) | `WEB` | https://github.com/liuyang12/DeSCI |
| Messi, Football (512x512x8) | `WEB` | https://github.com/liuyang12/PnP-SCI |
| Color CACTI benchmark | `WEB` | https://github.com/mcao92/EfficientSCI-plus-plus |
| Large-scale (1024) | `GEN` | Generated from DAVIS video frames |

---

## 1.4 Matrix — Generic Matrix Sensing

### System Variants (4)

| # | Variant | DAG | Physical Elements |
|---|---------|-----|-------------------|
| 1 | **Dense Gaussian** | Src -> M(dense_gaussian) -> D | Random dense measurement matrix |
| 2 | **Sparse Binary** | Src -> M(sparse_binary) -> D | Sparse 0/1 patterns |
| 3 | **Structured (DCT)** | Src -> M(partial_DCT) -> D | Partial discrete cosine transform |
| 4 | **Learned Matrix** | Src -> M(learned_matrix) -> D | End-to-end optimized measurement |

### Matrix Case Count

| Benchmark | Count |
|-----------|-------|
| B1 | 4 x 4 x 3 = **48** |
| B2 | 4 x 3 x 4 x 4 x 5 = **960** |
| B3 | **960** |
| B4 | **960** |
| **Matrix Total** | **2,928** |

### Matrix Data Sources

| Source | Label | Reference |
|--------|-------|-----------|
| MNIST, CIFAR-10 | `WEB` | Standard ML datasets |
| Set11, BSD68 | `SYN-WEB` | Standard test images |
| Generated random matrices | `GEN` | numpy.random |

---

## Category 1 Grand Total

| Modality | B1 | B2 | B3 | B4 | Total |
|----------|----|----|----|----|-------|
| CASSI | 108 | 2,880 | 2,880 | 2,880 | 8,748 |
| SPC | 72 | 2,400 | 2,400 | 2,400 | 7,272 |
| CACTI | 60 | 1,200 | 1,200 | 1,200 | 3,660 |
| Matrix | 48 | 960 | 960 | 960 | 2,928 |
| **Total** | **288** | **7,440** | **7,440** | **7,440** | **22,608** |
