
---

## Optical Microscopy & Super-Resolution — Modality Templates

---

### Confocal 3D Z-Stack (`confocal_3d`) Modality Template

#### Step 1: Verify Standard Dataset

For confocal 3D Z-Stack, what dataset do you use to verify? Is this dataset used for confocal 3D popular algorithms? Please ensure the standard dataset in `datasets/benchmark/confocal_3d/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original confocal 3D standard dataset.

**Popular datasets to consider:**
- **ISBI Deconvolution Grand Challenge (Vonesch et al., 2006; Sage et al., 2017)** — synthetic and real 3D confocal z-stacks with known ground truth PSFs; the canonical benchmark for 3D deconvolution algorithms
- **Confocal Fluorescence Microscopy Image Datasets (Broaddus et al., 2020)** — 3D z-stacks of various biological specimens (nuclei, membranes, microtubules) with paired high-SNR ground truth
- **BioImage Model Zoo / ZeroCostDL4Mic Datasets (von Chamier et al., 2021)** — community-curated 3D confocal stacks for denoising and restoration benchmarks
- **Cell Tracking Challenge 3D Datasets (Ulman et al., Nat Methods 2017)** — 3D confocal time-lapse z-stacks of fluorescent cells with ground truth segmentation and tracking

**Decision criteria:** The ISBI Deconvolution Grand Challenge dataset is the most widely used benchmark for 3D confocal deconvolution. Cell Tracking Challenge for segmentation-focused tasks. Use the dataset that appears in the largest number of confocal 3D reconstruction papers (2006–2026).

#### Step 2: List All Confocal 3D Algorithms

Please first ensure all the confocal 3D algorithms have been listed in `\Physics_World_Model\algorithm_base\confocal_3d\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/confocal_3d. Besides, you need to search all algorithms from 1950 to 2026. After listing all the confocal 3D solvers, please update the confocal 3D solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1970s–2005):_
- Nearest-neighbor deconvolution — subtract scaled adjacent slices to reduce out-of-focus blur (Agard, 1984)
- Inverse filter — naive Fourier-domain division by OTF (early 1970s); demonstrates ringing artifacts
- Wiener filter — regularized inverse filter balancing noise amplification (Wiener, 1949; applied to microscopy by Hiraoka et al., 1990)
- Tikhonov-regularized deconvolution — L2-penalized least-squares inversion for 3D PSF (Tikhonov, 1963; applied 1980s–1990s)
- Gold's ratio method — multiplicative iterative deconvolution (Gold, 1964)

_Iterative / Optimization (1972–2016):_
- Richardson-Lucy (RL) deconvolution — iterative ML for Poisson noise (Richardson, 1972; Lucy, 1974) — the most widely used 3D deconvolution algorithm
- Accelerated Richardson-Lucy — Biggs-Andrews acceleration of RL convergence (Biggs & Andrews, 1997)
- Maximum a Posteriori (MAP) with Gaussian prior — regularized RL with smoothness constraint (Conchello, 1998)
- Total Variation regularized deconvolution — TV-penalized 3D deconvolution (Dey et al., 2006)
- ISTA/FISTA for sparse 3D deconvolution — fast iterative shrinkage-thresholding with wavelet sparsity (Beck & Teboulle, 2009)
- ADMM-based 3D deconvolution — splitting methods for composite regularizers (Almeida & Figueiredo, 2013)
- Blind deconvolution — joint estimation of PSF and object (Holmes, 1992; Lam & Bhatt, 2000)
- Regularized blind deconvolution with sparsity — sparse prior on object and parametric PSF (Soulez et al., 2012)
- Good's roughness deconvolution (Intermittent deconvolution) — entropy-based regularization (Käseberg & Erdmann, 1990)
- Huygens deconvolution (SVI) — commercial maximum likelihood estimation with measured PSF (Scientific Volume Imaging, 1990s)

_Deep Learning (2017–2026):_
- CARE — Content-Aware Image Restoration for 3D confocal denoising and deconvolution (Weigert et al., Nat Methods 2018) — the seminal DL microscopy restoration paper
- CSBDeep / Noise2Void — self-supervised denoising without ground truth (Krull et al., CVPR 2019)
- DeconvNet 3D — 3D U-Net for learned deconvolution (2020)
- Richardson-Lucy Network (RLN) — physics-informed unrolled RL with learned regularization (Li et al., Nat Methods 2022)
- Self-Net — self-supervised blind deconvolution network (Chen et al., 2022)
- SwinIR-3D — 3D Swin Transformer for volumetric restoration (2023)
- Diffusion-based 3D deconvolution — score-based priors for confocal restoration (2024)
- Foundation model for fluorescence microscopy restoration (2025)

#### Step 3: Update Confocal 3D Solvers

After listing all confocal 3D solvers, update `algorithm_base/confocal_3d/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All confocal 3D solvers use the data format: `y` (D, H, W) 3D z-stack of fluorescence intensity, `psf` (D_psf, H_psf, W_psf) measured or estimated 3D point spread function. The `Confocal3DOperator` handles the forward model `y = PSF * x + noise` (3D convolution with Poisson–Gaussian noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for confocal 3D:**
- ISBI Deconvolution Challenge: Wiener ~25.0 dB, RL (50 iter) ~30.0 dB, TV-regularized ~31.5 dB, CARE ~34.0 dB, RLN ~35.5 dB
- 3D confocal denoising: raw SNR ~18 dB, Noise2Void ~28 dB, CARE ~30 dB
- All reference values from ISBI challenge leaderboard and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'confocal_3d' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/confocal_3d/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/confocal_3d/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/confocal_3d/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for confocal 3D. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/confocal_3d/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_3d/standard/`

---

### Confocal Laser Endomicroscopy (`confocal_endomicroscopy`) Modality Template

#### Step 1: Verify Standard Dataset

For confocal laser endomicroscopy (CLE), what dataset do you use to verify? Is this dataset used for CLE popular algorithms? Please ensure the standard dataset in `datasets/benchmark/confocal_endomicroscopy/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CLE standard dataset.

**Popular datasets to consider:**
- **MICCAI CLE Dataset (Aubreville et al., 2017; 2019)** — probe-based CLE (pCLE) images of oral squamous cell carcinoma with expert annotations; widely used for CLE classification benchmarks
- **ATLAS pCLE Dataset (Le Goualher et al., 2008; Mauna Kea Technologies)** — standardized pCLE video sequences from GI tract with diagnostic labels
- **GI Tract pCLE Dataset (Gora et al., 2013; Andre et al., 2011)** — pCLE images and video mosaics from Barrett's esophagus and colorectal polyps with histopathological ground truth
- **Brain Tumor CLE Dataset (Belykh et al., 2018; Martirosyan et al., 2016)** — intraoperative CLE of brain tumors with neuropathologist annotations

**Decision criteria:** The MICCAI CLE oral cancer dataset is the most widely benchmarked for classification. GI tract pCLE datasets for mosaicking and diagnostic tasks. Use the dataset that appears in the largest number of CLE papers (2008–2026).

#### Step 2: List All CLE Algorithms

Please first ensure all the CLE algorithms have been listed in `\Physics_World_Model\algorithm_base\confocal_endomicroscopy\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/confocal_endomicroscopy. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CLE solvers, please update the CLE solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2004–2012):_
- Fiber bundle pattern removal — interpolation and filtering to remove honeycomb fiber core pattern (Le Goualher et al., 2008)
- Flat-field correction for pCLE — intensity normalization across fiber bundle (2006)
- Adaptive median filtering for pCLE noise reduction (2008)
- Motion-compensated video mosaicking — feature-based registration for pCLE mosaics (Vercauteren et al., MICCAI 2005; TMI 2006) — the foundational pCLE mosaicking method
- Diffeomorphic registration for pCLE mosaicking (Vercauteren et al., NeuroImage 2009)

_Optimization (2010–2018):_
- Super-resolution from pCLE video — multi-frame super-resolution exploiting fiber displacement (Shao et al., 2009; Bria et al., 2013)
- Sparse reconstruction for pCLE — compressive sensing-based reconstruction from irregularly sampled fiber cores (Ravì et al., 2016)
- TV-regularized pCLE reconstruction — total variation denoising adapted for fiber bundle geometry (2014)
- Robust mosaicking with graph-cut optimization — globally consistent pCLE mosaic stitching (Rosa et al., 2012)
- Dictionary learning for pCLE denoising (2015)
- Non-local means adapted to pCLE fiber geometry (2013)
- Variational optical flow for pCLE motion compensation (2012)

_Deep Learning (2017–2026):_
- CNN for pCLE classification — tissue type classification from pCLE frames (Aubreville et al., 2017; Izadyyazdanabadi et al., 2018)
- GANs for fiber bundle artifact removal — image-to-image translation to remove honeycomb pattern (Ravì et al., J Biomed Opt 2019)
- U-Net for pCLE super-resolution — learned upsampling beyond fiber density limit (Shao et al., 2019)
- Deep mosaicking — learned feature matching and stitching for pCLE video (2020)
- Self-supervised denoising for pCLE — Noise2Self adapted to fiber bundle sampling (2021)
- Real-time CNN for intraoperative pCLE diagnosis (Izadyyazdanabadi et al., 2018; 2020)
- Transformer for pCLE video analysis and temporal coherence (2023)
- Vision foundation model fine-tuned for CLE tissue classification (2024)
- Diffusion model for pCLE artifact removal and super-resolution (2025)

#### Step 3: Update CLE Solvers

After listing all CLE solvers, update `algorithm_base/confocal_endomicroscopy/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CLE solvers use the data format: `y` (H, W) single pCLE frame or (T, H, W) pCLE video sequence, `fiber_mask` (H, W) binary mask of fiber core locations, `fiber_centers` (N_fibers, 2) coordinates of individual fiber cores. The `CLEOperator` handles the forward model (fiber bundle sampling * PSF convolution + Poisson noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CLE:**
- pCLE fiber pattern removal: interpolation ~28 dB, GAN-based ~33 dB PSNR on synthetic pairs
- pCLE classification (oral cancer): CNN accuracy >90%, AUC >0.95 (Aubreville et al., 2019)
- pCLE super-resolution: bilinear ~26 dB, U-Net ~31 dB on paired high-res/low-res data
- pCLE mosaicking: registration error <5 pixels on standard sequences (Vercauteren et al., 2006)
- All reference values from published papers and MICCAI challenge results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'confocal_endomicroscopy' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/confocal_endomicroscopy/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/confocal_endomicroscopy/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/confocal_endomicroscopy/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CLE. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/confocal_endomicroscopy/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_endomicroscopy/standard/`

---

### Confocal Live-Cell Microscopy (`confocal_livecell`) Modality Template

#### Step 1: Verify Standard Dataset

For confocal live-cell microscopy, what dataset do you use to verify? Is this dataset used for confocal live-cell popular algorithms? Please ensure the standard dataset in `datasets/benchmark/confocal_livecell/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original confocal live-cell standard dataset.

**Popular datasets to consider:**
- **LIVECell Dataset (Edlund et al., Nat Methods 2021)** — large-scale label-free and fluorescence live-cell images with instance segmentation ground truth; the primary benchmark for live-cell analysis
- **Cell Tracking Challenge Datasets (Ulman et al., Nat Methods 2017; Maska et al., Nat Methods 2023)** — time-lapse confocal sequences of live cells (Fluo-C2DL, Fluo-C3DL series) with tracking ground truth
- **Allen Institute Cell Collection (Spatio-temporal live-cell data, Viana et al., 2023)** — 3D confocal time-lapse of labeled organelles in live iPSC cells
- **ISBI Denoising Challenge Live-Cell Data (Zhang et al., 2019)** — low-SNR confocal live-cell data with paired high-SNR ground truth for denoising benchmarks

**Decision criteria:** Cell Tracking Challenge Fluo-C2DL/C3DL series is the most widely used for tracking and segmentation. LIVECell for instance segmentation. Use the dataset that appears in the largest number of live-cell confocal papers (2017–2026).

#### Step 2: List All Confocal Live-Cell Algorithms

Please first ensure all the confocal live-cell algorithms have been listed in `\Physics_World_Model\algorithm_base\confocal_livecell\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/confocal_livecell. Besides, you need to search all algorithms from 1950 to 2026. After listing all the confocal live-cell solvers, please update the confocal live-cell solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2010):_
- Photobleaching correction — exponential decay fitting and compensation for time-lapse intensity normalization (1990s)
- Background subtraction — rolling ball algorithm (Sternberg, 1983) adapted for live-cell fluorescence
- Temporal median filtering — separating static background from dynamic cell signals (2000s)
- Kalman filter for live-cell tracking — recursive state estimation of cell positions and velocities (1960; applied to cell tracking 1990s)
- Deconvolution with time-varying PSF — accounting for refractive index changes during live imaging (2005)

_Optimization (2006–2018):_
- Multi-frame denoising — exploiting temporal redundancy in time-lapse confocal (Boulanger et al., TMI 2010)
- Sparse + low-rank decomposition for live-cell — separating slowly varying structures from dynamic events (2012)
- Compressed sensing temporal super-resolution — recovering fast dynamics from undersampled confocal (2014)
- Joint denoising and deconvolution for live-cell confocal — coupled optimization exploiting temporal continuity (2013)
- Optical flow-based motion-compensated denoising (2011)
- BM4D — block-matching 4D collaborative filtering for 3D+t confocal data (Maggioni et al., TIP 2013)
- Variational segmentation with active contours for live-cell boundary detection (2008)
- Graph-cut based cell segmentation in confocal time-lapse (Al-Kofahi et al., 2010)

_Deep Learning (2017–2026):_
- CARE for live-cell — content-aware restoration reducing phototoxicity (Weigert et al., Nat Methods 2018)
- Noise2Noise / Noise2Void for live-cell — self-supervised denoising without clean ground truth (Krull et al., 2019; Batson & Royer, 2019)
- DenoiSeg — joint denoising and segmentation for live-cell (Buchholz et al., 2020)
- Cellpose — generalist cell segmentation model tested on confocal live-cell data (Stringer et al., Nat Methods 2021)
- StarDist — star-convex polygon detection for fluorescent nuclei (Schmidt et al., MICCAI 2018)
- EmbedTrack — embedding-based cell tracking from confocal time-lapse (Loffler & Schlafly, 2022)
- 3D U-Net for volumetric live-cell segmentation (Cicek et al., MICCAI 2016)
- pN2V — probabilistic Noise2Void for uncertainty quantification in live-cell denoising (2022)
- Diffusion model for live-cell temporal super-resolution — predicting intermediate frames (2024)
- Foundation model for cell segmentation across microscopy modalities (2025)

#### Step 3: Update Confocal Live-Cell Solvers

After listing all confocal live-cell solvers, update `algorithm_base/confocal_livecell/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All confocal live-cell solvers use the data format: `y` (T, H, W) or (T, D, H, W) time-lapse confocal sequence, `psf` (D_psf, H_psf, W_psf) or (H_psf, W_psf) point spread function, `bleach_curve` (T,) optional photobleaching decay profile. The `ConfocalLiveCellOperator` handles the forward model (PSF convolution * bleaching decay + Poisson–Gaussian noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for confocal live-cell:**
- Live-cell denoising: raw SNR ~15 dB, BM4D ~27 dB, Noise2Void ~29 dB, CARE ~32 dB
- Cell Tracking Challenge (Fluo-C2DL-MSC): tracking accuracy (TRA) >0.90 for top methods
- LIVECell segmentation: Cellpose AP@0.5 >0.75, StarDist AP@0.5 >0.70
- Temporal super-resolution: SSIM >0.90 for 4x frame interpolation
- All reference values from Cell Tracking Challenge leaderboard and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'confocal_livecell' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/confocal_livecell/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/confocal_livecell/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/confocal_livecell/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for confocal live-cell. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/confocal_livecell/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/confocal_livecell/standard/`

---

### Spinning Disk Confocal (`spinning_disk`) Modality Template

#### Step 1: Verify Standard Dataset

For spinning disk confocal microscopy, what dataset do you use to verify? Is this dataset used for spinning disk confocal popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spinning_disk/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original spinning disk confocal standard dataset.

**Popular datasets to consider:**
- **Cell Tracking Challenge Spinning Disk Datasets (Ulman et al., 2017)** — Fluo-N2DH-SIM+ and related spinning disk confocal time-lapse sequences with tracking ground truth
- **BioImage Model Zoo Spinning Disk Data (von Chamier et al., 2021)** — curated spinning disk confocal images for denoising and restoration benchmarks
- **Hagen et al. Spinning Disk Deconvolution Dataset (2021)** — paired low-exposure and high-exposure spinning disk confocal data for quantitative deconvolution benchmarks
- **Zenodo Spinning Disk Live-Cell Datasets (Various, 2019–2024)** — community-contributed spinning disk confocal time-lapse data with multiple fluorescent markers

**Decision criteria:** Cell Tracking Challenge spinning disk datasets are most widely benchmarked. BioImage Model Zoo datasets for restoration tasks. Use the dataset that appears in the largest number of spinning disk confocal papers (2017–2026).

#### Step 2: List All Spinning Disk Confocal Algorithms

Please first ensure all the spinning disk confocal algorithms have been listed in `\Physics_World_Model\algorithm_base\spinning_disk\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spinning_disk. Besides, you need to search all algorithms from 1950 to 2026. After listing all the spinning disk solvers, please update the spinning disk solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2010):_
- Pinhole crosstalk correction — correcting for inter-pinhole light leakage in Nipkow disk systems (Tanaami et al., 2002)
- Flat-field correction — compensating for non-uniform pinhole array illumination (2000s)
- Nearest-neighbor out-of-focus subtraction adapted for spinning disk (Müller & Bhatt, 2004)
- Wiener deconvolution with spinning disk PSF — accounting for pinhole-specific OTF (2005)
- Multi-point confocal background estimation and removal (2003)

_Optimization (2006–2018):_
- Richardson-Lucy deconvolution adapted for spinning disk PSF — iterative ML with disk-specific PSF model (2006)
- TV-regularized spinning disk deconvolution (2010)
- Multi-view spinning disk reconstruction — combining multiple focal planes for axial super-resolution (2012)
- Sparse deconvolution for spinning disk — L1-regularized reconstruction for punctate structures (2013)
- Joint denoising-deconvolution with Poisson noise model for spinning disk (2014)
- ADMM-based reconstruction for spinning disk confocal with background penalty (2015)
- Structured background removal via morphological filtering (2008)
- Non-local means denoising adapted for spinning disk confocal noise characteristics (2011)
- Stripe artifact removal in spinning disk confocal — frequency-domain filtering for disk rotation artifacts (2010)

_Deep Learning (2017–2026):_
- CARE for spinning disk — trained on paired low/high exposure spinning disk data (Weigert et al., 2018)
- Noise2Fast — fast self-supervised denoising for spinning disk confocal (Lequyer et al., 2022)
- CSBDeep spinning disk models — community-trained models for specific spinning disk microscopes (2019)
- 3D U-Net for spinning disk z-stack restoration (2020)
- Cellpose / StarDist for spinning disk cell segmentation (Stringer et al., 2021)
- DeepBacs — DL for bacterial imaging including spinning disk data (Spahn et al., 2022)
- Attention U-Net for multi-channel spinning disk deconvolution (2023)
- Self-supervised blind-spot networks for spinning disk denoising (2023)
- Diffusion model for spinning disk image restoration (2024)
- Cross-modality foundation model fine-tuned on spinning disk data (2025)

#### Step 3: Update Spinning Disk Solvers

After listing all spinning disk solvers, update `algorithm_base/spinning_disk/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All spinning disk solvers use the data format: `y` (D, H, W) or (T, D, H, W) spinning disk confocal z-stack or time-lapse, `psf` (D_psf, H_psf, W_psf) spinning disk-specific PSF accounting for pinhole geometry, `flat_field` (H, W) illumination non-uniformity map. The `SpinningDiskOperator` handles the forward model (pinhole array sampling * PSF convolution + background + Poisson noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for spinning disk:**
- Spinning disk deconvolution: Wiener ~26 dB, RL ~30 dB, TV-regularized ~31.5 dB, CARE ~34 dB
- Spinning disk denoising: raw SNR ~16 dB, Noise2Fast ~28 dB, CARE ~31 dB
- Cell segmentation on spinning disk data: Cellpose AP@0.5 >0.70
- All reference values from published papers and BioImage benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spinning_disk' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spinning_disk/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spinning_disk/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spinning_disk/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for spinning disk. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spinning_disk/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spinning_disk/standard/`

---

### Light-Sheet Fluorescence Microscopy (`lightsheet`) Modality Template

#### Step 1: Verify Standard Dataset

For light-sheet fluorescence microscopy (LSFM), what dataset do you use to verify? Is this dataset used for LSFM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/lightsheet/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original LSFM standard dataset.

**Popular datasets to consider:**
- **OpenSPIM Datasets (Pitrone et al., Nat Methods 2013)** — multi-view light-sheet data of Drosophila embryos and zebrafish with registration ground truth; canonical LSFM benchmark
- **Cell Tracking Challenge Light-Sheet Data (Ulman et al., 2017; Maska et al., 2023)** — light-sheet time-lapse of developing embryos with tracking and lineage ground truth
- **Zebrafish Light-Sheet Atlas (Amat et al., Nat Methods 2014; Keller et al., Science 2008)** — whole-embryo light-sheet recordings at cellular resolution
- **BigStitcher Benchmark Data (Horl et al., Nat Methods 2019)** — multi-tile multi-view light-sheet datasets for stitching and registration benchmarks

**Decision criteria:** OpenSPIM data and Cell Tracking Challenge light-sheet sets are the most widely benchmarked. BigStitcher datasets for registration tasks. Use the dataset that appears in the largest number of LSFM reconstruction papers (2008–2026).

#### Step 2: List All LSFM Algorithms

Please first ensure all the LSFM algorithms have been listed in `\Physics_World_Model\algorithm_base\lightsheet\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/lightsheet. Besides, you need to search all algorithms from 1950 to 2026. After listing all the LSFM solvers, please update the LSFM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2004–2012):_
- Maximum intensity projection (MIP) — basic z-projection for light-sheet overview (standard)
- Multi-view registration — rigid and affine alignment of opposing light-sheet views (Preibisch et al., Nat Methods 2010)
- Multi-view fusion by weighted averaging — content-based weighting of registered views (Preibisch et al., 2010)
- Stripe artifact removal — FFT-based filtering of illumination stripe artifacts in light-sheet data (Munch et al., Opt Express 2009)
- Light-sheet PSF modeling — Gaussian beam and Bessel beam theoretical PSF for LSFM (2008)

_Optimization (2010–2020):_
- Multi-view deconvolution — joint RL deconvolution of multiple views for isotropic resolution (Preibisch et al., Nat Methods 2014) — the standard multi-view LSFM reconstruction
- Content-based multi-view fusion with entropy weighting (Preibisch et al., 2014)
- Bayesian multi-view deconvolution — MAP estimation from multiple light-sheet views (2015)
- TV-regularized light-sheet deconvolution (2013)
- Blind deconvolution for light-sheet — estimating spatially varying PSF across the field of view (2016)
- Sparse deconvolution for light-sheet — L1 penalty for visualizing fine structures (2017)
- BigStitcher — tile-based registration and fusion of large multi-tile light-sheet data (Horl et al., Nat Methods 2019)
- Variational stripe removal — optimization-based stripe artifact correction (2015)
- Dual-view inverted SPIM (diSPIM) reconstruction — alternating-view deconvolution for isotropic resolution (Wu et al., Nat Biotechnol 2013)
- Block-face light-sheet deconvolution for cleared tissue (2018)

_Deep Learning (2018–2026):_
- CARE for light-sheet — denoising low-exposure light-sheet data with paired training (Weigert et al., 2018)
- Isotropic reconstruction with CARE — restoring axial resolution from single-view light-sheet (Weigert et al., Nat Methods 2018)
- DL multi-view fusion — learning optimal view combination from data (2021)
- DeStripe — DL-based stripe artifact removal for light-sheet (2021)
- FlowNet for light-sheet registration — learned optical flow for multi-view alignment (2020)
- Cellpose for light-sheet segmentation (Stringer et al., 2021)
- Self-supervised denoising for light-sheet (Noise2Void, Noise2Self) (2019–2020)
- 3D ResNet for light-sheet cell detection (2022)
- Transformer-based light-sheet volumetric reconstruction (2024)
- Foundation model for volumetric fluorescence microscopy (2025)

#### Step 3: Update LSFM Solvers

After listing all LSFM solvers, update `algorithm_base/lightsheet/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All LSFM solvers use the data format: `y` (N_views, D, H, W) multi-view light-sheet z-stacks, `psf` (N_views, D_psf, H_psf, W_psf) view-specific PSFs (anisotropic, elongated along detection axis), `transforms` (N_views, 4, 4) registration matrices between views. The `LightSheetOperator` handles the forward model (view-specific PSF convolution + Poisson noise) and adjoint operations including multi-view fusion.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for LSFM:**
- Multi-view fusion: weighted average ~28 dB, multi-view RL deconvolution ~33 dB, CARE isotropic ~35 dB
- Stripe removal: raw data with stripes, Munch FFT method ~30 dB, DL DeStripe ~33 dB
- Light-sheet denoising: raw SNR ~14 dB, CARE ~30 dB
- Cell Tracking Challenge (light-sheet embryo data): TRA >0.85 for top methods
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'lightsheet' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/lightsheet/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/lightsheet/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/lightsheet/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for LSFM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/lightsheet/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/lightsheet/standard/`

---

### Lattice Light-Sheet Microscopy (`lattice_lightsheet`) Modality Template

#### Step 1: Verify Standard Dataset

For lattice light-sheet microscopy (LLSM), what dataset do you use to verify? Is this dataset used for LLSM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/lattice_lightsheet/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original LLSM standard dataset.

**Popular datasets to consider:**
- **Janelia LLSM Datasets (Chen et al., Science 2014)** — the original lattice light-sheet datasets of live cells (T cells, fibroblasts, organelles) from the Betzig lab; canonical LLSM benchmark
- **LLSM Cell Biology Atlas (Liu et al., Science 2018)** — multi-color LLSM volumes of subcellular dynamics (mitochondria, ER, Golgi, plasma membrane)
- **Adaptive Optics LLSM Data (Liu et al., Science 2018)** — AO-corrected LLSM of cells in tissues and organoids
- **CZ BioHub LLSM Datasets (Ruan et al., 2023)** — Allen Institute / Chan Zuckerberg lattice light-sheet data of labeled organelles in hiPSC cells

**Decision criteria:** Janelia LLSM datasets are the founding benchmark. AO-LLSM datasets for aberration-corrected imaging. Use the dataset most widely referenced in LLSM analysis papers (2014–2026).

#### Step 2: List All LLSM Algorithms

Please first ensure all the LLSM algorithms have been listed in `\Physics_World_Model\algorithm_base\lattice_lightsheet\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/lattice_lightsheet. Besides, you need to search all algorithms from 1950 to 2026. After listing all the LLSM solvers, please update the LLSM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2014–2018):_
- Deskewing — affine transformation to correct for oblique light-sheet geometry (Chen et al., 2014) — mandatory preprocessing for all LLSM data
- Sample-scan to coverslip-scan coordinate transformation (2014)
- Maximum intensity projection after deskew — standard visualization (2014)
- Lattice pattern demodulation — extracting fluorescence from structured excitation pattern (2014)

_Optimization (2014–2020):_
- Richardson-Lucy deconvolution with lattice PSF — iterative deconvolution using experimentally measured LLSM PSF (Chen et al., 2014)
- Joint deskew-deconvolution — combined geometric correction and deconvolution (Ruan & Bhatt, 2019)
- TV-regularized LLSM deconvolution — total variation penalty for LLSM volumetric data (2016)
- Adaptive optics aberration correction — wavefront sensing and correction integrated with LLSM (Liu et al., Science 2018)
- Sparse deconvolution for LLSM — L1-regularized reconstruction for resolving organelle structures (2018)
- Multi-color LLSM unmixing — spectral unmixing of multi-channel LLSM data with crosstalk correction (2017)
- Anisotropic deconvolution — accounting for the anisotropic lattice light-sheet PSF (2016)
- ADMM-based LLSM reconstruction with positivity and sparsity constraints (2019)
- 3D Gaussian fitting for LLSM particle tracking (2016)
- Phase retrieval for lattice pattern optimization (2015)

_Deep Learning (2019–2026):_
- CARE for LLSM — denoising low-exposure lattice light-sheet data (Weigert et al., 2018; applied to LLSM 2019)
- DL deskew-deconvolution — end-to-end learned deskew and deconvolution (2021)
- Noise2Void for LLSM — self-supervised denoising for lattice light-sheet data (2020)
- 3D segmentation networks for LLSM organelle segmentation (Heinrich et al., Nature 2021)
- Instance segmentation of organelles from LLSM (CellPose3D adapted, 2022)
- GAN-based LLSM temporal super-resolution — predicting intermediate time points (2022)
- Physics-informed neural network for LLSM PSF estimation and deconvolution (2023)
- Transformer for LLSM volumetric tracking (2024)
- Foundation model for 4D lattice light-sheet analysis (2025)

#### Step 3: Update LLSM Solvers

After listing all LLSM solvers, update `algorithm_base/lattice_lightsheet/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All LLSM solvers use the data format: `y` (D, H, W) raw (skewed) LLSM volume or (T, D, H, W) time-lapse, `psf` (D_psf, H_psf, W_psf) experimentally measured lattice light-sheet PSF, `deskew_angle` (float) angle of the light-sheet relative to coverslip (typically 31.8 degrees). The `LLSMOperator` handles the forward model (deskew geometry * PSF convolution + Poisson noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for LLSM:**
- LLSM deconvolution: raw ~20 dB, RL (10 iter) ~28 dB, TV-regularized ~30 dB, CARE ~33 dB
- LLSM denoising: raw SNR ~16 dB, Noise2Void ~27 dB, CARE ~31 dB
- Organelle segmentation from LLSM: IoU >0.75 for mitochondria, >0.70 for ER (Heinrich et al., 2021)
- Deskew accuracy: <0.5 pixel registration error
- All reference values from published papers (Chen et al., 2014; Liu et al., 2018)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'lattice_lightsheet' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/lattice_lightsheet/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/lattice_lightsheet/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/lattice_lightsheet/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for LLSM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/lattice_lightsheet/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/lattice_lightsheet/standard/`

---

### Structured Illumination Microscopy (`sim`) Modality Template

#### Step 1: Verify Standard Dataset

For structured illumination microscopy (SIM), what dataset do you use to verify? Is this dataset used for SIM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sim/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SIM standard dataset.

**Popular datasets to consider:**
- **BioSR Dataset (Qiao et al., Nat Methods 2021)** — paired widefield and SIM images of biological structures (microtubules, F-actin, ER, mitochondria) with ground truth; the primary DL-SIM benchmark
- **fairSIM Test Data (Muller et al., Bioinformatics 2016)** — open-source SIM reconstruction test datasets with known ground truth patterns
- **OpenSIM Datasets (Lal et al., 2015; Lehmann et al., 2017)** — community SIM raw data with published reconstruction results for benchmarking
- **Hessian-SIM Dataset (Huang et al., Nat Biotechnol 2018)** — raw SIM data with Hessian-regularized reconstruction ground truth for live-cell super-resolution

**Decision criteria:** BioSR is the most widely used benchmark for DL-based SIM reconstruction (2021–2026). fairSIM for classical reconstruction validation. Use the dataset that appears in the largest number of SIM reconstruction papers (2015–2026).

#### Step 2: List All SIM Algorithms

Please first ensure all the SIM algorithms have been listed in `\Physics_World_Model\algorithm_base\sim\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sim. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SIM solvers, please update the SIM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2000–2012):_
- Gustafsson SIM reconstruction — frequency-domain separation, shifting, and recombination of structured illumination components (Gustafsson, J Microsc 2000; PNAS 2005) — the foundational SIM algorithm
- Wiener-filtered SIM — OTF-based Wiener filtering of separated frequency components with noise regularization (Gustafsson et al., Biophys J 2008)
- Linear SIM (2D-SIM) — three orientations, three phases per orientation for 2x lateral resolution (Gustafsson, 2000)
- 3D-SIM — five phases, three orientations for both lateral and axial super-resolution (Gustafsson et al., 2008)
- Parameter estimation for SIM — cross-correlation-based estimation of illumination pattern parameters (Wicker et al., Opt Express 2013)

_Optimization (2012–2020):_
- Iterative SIM (iSIM) reconstruction — iterative deconvolution of SIM data for improved resolution (Rego et al., PNAS 2012)
- MAP-SIM — maximum a posteriori SIM reconstruction with statistical noise model (Muller et al., Nat Commun 2016)
- Hessian-SIM — Hessian matrix-based regularization for high-fidelity SIM of live cells (Huang et al., Nat Biotechnol 2018)
- TV-regularized SIM reconstruction — total variation penalty for SIM with reduced artifacts (2016)
- Blind-SIM — joint estimation of illumination patterns and specimen (Jost et al., 2015; Mudry et al., Nat Photonics 2012)
- Nonlinear SIM (NL-SIM) / saturated SIM — exploiting fluorescence saturation for >2x resolution (Gustafsson, PNAS 2005; Rego et al., PNAS 2012)
- fairSIM — open-source Fiji plugin for SIM reconstruction (Muller et al., Bioinformatics 2016)
- SIMcheck — SIM data quality assessment tool (Ball et al., Sci Rep 2015)
- Rolling SIM — processing SIM with sliding window for time-lapse data (Lal et al., 2015)
- ADMM-based SIM with sparsity and non-negativity constraints (2018)

_Deep Learning (2018–2026):_
- Deep-SIM / ML-SIM — CNN-based SIM reconstruction from raw frames (Christensen et al., Biomed Opt Express 2021; Jin et al., 2020)
- DFCAN/DFGAN — deep Fourier channel attention for SIM reconstruction (Qiao et al., Nat Methods 2021) — state-of-the-art DL-SIM
- SIM reconstruction with fewer frames — DL reconstruction from reduced number of raw images (Ling et al., 2020)
- Physics-informed neural network for SIM — encoding forward model in loss function (2022)
- GAN-based SIM artifact reduction (2021)
- Transformer for SIM reconstruction (2023)
- Self-supervised SIM — reconstruction without paired ground truth (2023)
- Real-time DL-SIM for live-cell imaging (2022)
- Diffusion model for SIM super-resolution beyond 2x (2024)
- Foundation model for structured illumination across modalities (2025)

#### Step 3: Update SIM Solvers

After listing all SIM solvers, update `algorithm_base/sim/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SIM solvers use the data format: `y` (N_orientations, N_phases, H, W) raw SIM frames (typically 3 orientations x 3 or 5 phases), `otf` (H, W) optical transfer function of the detection path, `pattern_params` (N_orientations, 3) illumination pattern parameters (frequency, phase, modulation depth). The `SIMOperator` handles the forward model (structured illumination modulation * OTF convolution + Poisson noise) and adjoint operations including frequency unmixing.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SIM:**
- BioSR microtubules: widefield ~25 dB, Wiener-SIM ~30 dB, Hessian-SIM ~32 dB, DFCAN ~34.5 dB
- BioSR F-actin: Wiener-SIM ~29 dB, DFCAN ~33 dB
- SIM resolution: widefield ~250 nm, 2D-SIM ~120 nm, NL-SIM ~60 nm
- fairSIM test patterns: reconstruction NRMSE <5% for standard SIM
- All reference values from BioSR benchmark (Qiao et al., 2021) and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sim' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sim/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sim/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sim/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SIM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sim/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sim/standard/`

---

### Image Scanning Microscopy (`ism`) Modality Template

#### Step 1: Verify Standard Dataset

For image scanning microscopy (ISM), what dataset do you use to verify? Is this dataset used for ISM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ism/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ISM standard dataset.

**Popular datasets to consider:**
- **Zeiss Airyscan Datasets (Huff, 2015; Korobchevskaya et al., 2017)** — Airyscan (commercial ISM) images of standard biological specimens with conventional confocal comparisons; widely used for ISM resolution benchmarks
- **Muller & Bhatt ISM/Re-scan Confocal Datasets (Muller & Enderlein, PRL 2010; De Luca et al., Biomed Opt Express 2013)** — re-scan confocal microscopy data demonstrating sqrt(2) resolution improvement
- **Open-source ISM Datasets (Castello et al., Nat Methods 2019)** — raw ISM detector array data with pixel reassignment ground truth
- **SPAD-ISM Datasets (Zunino et al., Nat Commun 2023)** — single-photon avalanche diode array ISM data with photon-resolved detection

**Decision criteria:** Zeiss Airyscan data is the most widely available ISM benchmark (2015–2026). Castello et al. datasets for algorithmic benchmarking of pixel reassignment. Use the dataset that appears in the largest number of ISM papers (2010–2026).

#### Step 2: List All ISM Algorithms

Please first ensure all the ISM algorithms have been listed in `\Physics_World_Model\algorithm_base\ism\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ism. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ISM solvers, please update the ISM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2010–2016):_
- Pixel reassignment (PR) — shifting each detector element image by half the detector offset and summing (Muller & Enderlein, PRL 2010; Sheppard et al., 1988) — the foundational ISM algorithm, provides sqrt(2) resolution gain
- Sheppard sum — optimal weighted summation of reassigned detector images (Sheppard, Optik 1988)
- Optical pixel reassignment — hardware-based rescanning (De Luca et al., Biomed Opt Express 2013)
- Photon reassignment — photon-by-photon reassignment for optimal SNR (Roth et al., 2013)

_Optimization (2013–2020):_
- Multi-image deconvolution ISM — joint deconvolution of all detector element images exploiting the known shift-variant PSF (Muller & Enderlein, 2010; Schulz et al., 2013)
- Adaptive pixel reassignment — data-driven optimization of reassignment weights (Castello et al., Nat Methods 2019)
- Focus-ISM — extended depth-of-field from ISM detector array (Sheppard et al., 2020)
- Fourier reweighting ISM — frequency-domain reweighting for optimal resolution-SNR trade-off (Muller & Enderlein, 2010)
- ISM with structured detection — combining ISM with structured illumination principles (York et al., Nat Methods 2012; MSIM)
- ADMM-based ISM reconstruction with detector-specific PSFs (2018)
- Blind ISM deconvolution — estimating effective PSF from ISM data (2017)
- Sparse ISM reconstruction — L1-regularized deconvolution for punctate fluorophores (2019)
- Phase retrieval from ISM data — recovering pupil function from detector array measurements (2018)
- Maximum likelihood ISM — statistical estimation exploiting Poisson statistics per detector element (2016)

_Deep Learning (2020–2026):_
- DL pixel reassignment — learned reassignment weights via CNN (2021)
- ISM-Net — end-to-end neural network for ISM reconstruction from raw detector array data (2022)
- Noise2ISM — self-supervised denoising exploiting ISM detector redundancy (2022)
- DL-ISM deconvolution — unrolled optimization network for ISM (2023)
- GAN for ISM super-resolution beyond sqrt(2) improvement (2023)
- Transformer for ISM multi-detector fusion (2024)
- Physics-informed ISM network with PSF-aware layers (2024)
- Foundation model for scanning microscopy (ISM/confocal/STED) (2025)

#### Step 3: Update ISM Solvers

After listing all ISM solvers, update `algorithm_base/ism/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ISM solvers use the data format: `y` (N_det, H, W) ISM raw data from detector array (e.g., 32-element Airyscan or SPAD array), `psf` (N_det, H_psf, W_psf) detector-element-specific PSFs, `det_positions` (N_det, 2) detector element positions relative to optical axis. The `ISMOperator` handles the forward model (detector-position-dependent PSF convolution + Poisson noise per element) and adjoint operations including pixel reassignment.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for ISM:**
- Pixel reassignment: sqrt(2) lateral resolution improvement (~1.4x), ~170 nm resolution at 488 nm excitation
- Multi-image deconvolution ISM: ~2x resolution improvement over confocal, ~120 nm at 488 nm
- ISM vs confocal PSNR: pixel reassignment +3–5 dB over standard confocal, multi-image deconvolution +6–8 dB
- DL-ISM: ~130 nm resolution, PSNR improvement of ~4 dB over pixel reassignment
- All reference values from published papers (Castello et al., 2019; Muller & Enderlein, 2010)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ism' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ism/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ism/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ism/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ISM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ism/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ism/standard/`

---

### Fourier Ptychographic Microscopy (`fpm`) Modality Template

#### Step 1: Verify Standard Dataset

For Fourier ptychographic microscopy (FPM), what dataset do you use to verify? Is this dataset used for FPM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/fpm/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original FPM standard dataset.

**Popular datasets to consider:**
- **Zheng et al. FPM Datasets (Zheng et al., Nat Photonics 2013)** — the original FPM raw data (LED array illumination captures of USAF targets and blood smears) with reconstructed high-resolution and phase ground truth; canonical FPM benchmark
- **FPM USAF Resolution Target Data (Ou et al., Opt Lett 2013)** — standardized resolution target captures under varying LED illumination angles for quantitative resolution assessment
- **Pathology FPM Datasets (Horstmeyer et al., 2016; Chung et al., Biomed Opt Express 2016)** — FPM whole-slide images of tissue sections with comparisons to 40x objective ground truth
- **LED Array Microscope Open Data (Phillips et al., PLoS ONE 2017)** — open-source FPM datasets with calibration metadata for reproducible reconstruction

**Decision criteria:** Zheng et al. original FPM data is the most widely cited benchmark. Phillips et al. for open reproducible benchmarks. Use the dataset that appears in the largest number of FPM reconstruction papers (2013–2026).

#### Step 2: List All FPM Algorithms

Please first ensure all the FPM algorithms have been listed in `\Physics_World_Model\algorithm_base\fpm\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/fpm. Besides, you need to search all algorithms from 1950 to 2026. After listing all the FPM solvers, please update the FPM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2013–2015):_
- Alternating projection FPM — sequential phase retrieval by alternating between spatial and Fourier domain constraints for each LED (Zheng et al., Nat Photonics 2013) — the foundational FPM algorithm
- Gerchberg-Saxton for FPM — classical iterative phase retrieval adapted to FPM geometry (Gerchberg & Saxton, 1972; adapted 2013)
- Embedded pupil function recovery — joint estimation of specimen and pupil aberrations during FPM reconstruction (Ou et al., Opt Lett 2014)

_Optimization (2014–2020):_
- Wirtinger flow for FPM — gradient-based non-convex optimization for Fourier ptychographic phase retrieval (Candes et al., 2015; applied to FPM by Yeh et al., 2015)
- Regularized FPM reconstruction — TV and sparsity-penalized FPM phase retrieval (Bian et al., Opt Express 2015)
- State-space model FPM — Kalman-filter-based sequential update of FPM reconstruction (Bian et al., Opt Express 2015)
- Multiplexed FPM — simultaneous multi-LED illumination with coded patterns (Tian et al., Optica 2014)
- Adaptive step-size FPM — accelerated convergence with data-driven step size (Bian et al., 2016)
- Pupil recovery with Zernike decomposition — parametric aberration correction during FPM (Sun et al., 2016)
- ADMM for FPM — splitting methods for constrained FPM reconstruction (2017)
- Robust FPM with noise model — Poisson-Gaussian noise-aware FPM reconstruction (Yeh et al., Opt Express 2017)
- Motion-corrected FPM — compensating for sample drift during LED scanning (Bian et al., 2016)
- 3D FPM — extending FPM to volumetric reconstruction via multi-slice propagation (Tian & Waller, Optica 2015)
- Annular illumination FPM for darkfield phase contrast (Ou et al., Opt Lett 2013)

_Deep Learning (2018–2026):_
- DL-FPM — CNN-based FPM reconstruction from reduced LED captures (Jiang et al., Biomed Opt Express 2018; Nguyen et al., Opt Express 2018)
- Physics-informed neural network for FPM — encoding wave propagation in network architecture (Cheng et al., 2019)
- GAN for FPM super-resolution — adversarial training for enhanced FPM reconstruction (2020)
- Unrolled optimization for FPM — learned ADMM / ISTA for FPM phase retrieval (2021)
- Neural FPM aberration correction — learning spatially varying pupil from data (2022)
- Transformer for FPM — attention-based fusion of multi-angle captures (2023)
- Self-supervised FPM — reconstruction without paired ground truth (2023)
- Diffusion model for FPM phase retrieval — score-based prior for ambiguity reduction (2024)
- Foundation model for computational microscopy (FPM + CDI + ptychography) (2025)

#### Step 3: Update FPM Solvers

After listing all FPM solvers, update `algorithm_base/fpm/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All FPM solvers use the data format: `y` (N_leds, H, W) intensity images captured under different LED illumination angles, `led_positions` (N_leds, 3) LED coordinates in the array, `na_obj` (float) numerical aperture of the objective, `wavelength` (float) illumination wavelength. The `FPMOperator` handles the forward model (coherent imaging: pupil * shifted spectrum, intensity detection) and adjoint operations including pupil function estimation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for FPM:**
- FPM resolution improvement: 10x objective (NA 0.25) achieving NA ~0.5 synthetic aperture with full LED array
- USAF target reconstruction: alternating projection ~30 dB, Wirtinger flow ~32 dB, DL-FPM ~35 dB PSNR
- Phase reconstruction accuracy: RMSE <0.1 rad on calibration phantoms (Zheng et al., 2013)
- FPM with reduced captures: DL-FPM achieving comparable quality with 50% fewer LED captures
- All reference values from published papers (Zheng et al., 2013; Tian & Waller, 2015)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'fpm' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/fpm/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/fpm/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/fpm/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for FPM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/fpm/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/fpm/standard/`

---

### Optical Diffraction Tomography (`odt`) Modality Template

#### Step 1: Verify Standard Dataset

For optical diffraction tomography (ODT), what dataset do you use to verify? Is this dataset used for ODT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/odt/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ODT standard dataset.

**Popular datasets to consider:**
- **Tomocube ODT Datasets (Kim et al., various 2014–2023)** — 3D refractive index tomograms of live cells (RBCs, white blood cells, HeLa) from commercial holotomography; widely used ODT benchmark
- **KAIST ODT Open Data (Lim et al., Opt Express 2015; Park et al., Nat Photonics 2018)** — raw ODT interferograms with angular scanning and reconstructed 3D RI maps of cells and microspheres
- **Microsphere Phantom ODT Data (Muller et al., Optica 2015)** — polystyrene microspheres with known RI as quantitative calibration standard
- **RBC ODT Datasets (Park et al., various)** — red blood cell 3D RI tomograms, canonical for validating morphological measurements

**Decision criteria:** KAIST ODT datasets are the most widely cited for ODT algorithm validation. Microsphere phantoms for quantitative RI accuracy. Use the dataset that appears in the largest number of ODT reconstruction papers (2014–2026).

#### Step 2: List All ODT Algorithms

Please first ensure all the ODT algorithms have been listed in `\Physics_World_Model\algorithm_base\odt\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/odt. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ODT solvers, please update the ODT solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1969–2010):_
- Filtered back-projection for ODT — Wolf transform / diffraction tomography analogy to CT (Wolf, 1969; Devaney, 1981) — the foundational ODT reconstruction
- Rytov approximation ODT — linearized scattering model valid for weakly scattering objects (Rytov, 1937; applied by Muller et al., 2015)
- Born approximation ODT — first-order scattering approximation for thin/weak samples (Born, 1933; Devaney, 1981)
- Direct inversion in Fourier space — mapping angular spectra onto Ewald sphere (Lauer, 2002)
- Phase unwrapping for ODT — temporal or spatial unwrapping of interferometric phase maps (Goldstein et al., 1988)

_Optimization (2010–2020):_
- Iterative ODT with total variation — TV-regularized 3D RI reconstruction for missing-cone compensation (Sung et al., Opt Express 2009; Lim et al., 2015)
- ADMM-based ODT — splitting methods for constrained 3D RI tomography (Kamilov et al., Optica 2015)
- Beam propagation method (BPM) ODT — forward model using beam propagation for multiple scattering (Tian & Waller, Optica 2015)
- Learning-ADMM-ODT — ADMM with learned regularization parameters (2018)
- Compressive ODT — reconstructing from reduced angular measurements via sparsity (Cotte et al., Nat Photonics 2013)
- Tikhonov-regularized ODT with positivity constraint (2012)
- Multi-slice ODT — accounting for multiple scattering via layer-by-layer propagation (Chowdhury et al., Optica 2019)
- Missing cone extrapolation — iterative methods to fill the missing cone in ODT Fourier space (2016)
- Nonlinear ODT with FDTD forward model — full-wave forward model for strongly scattering objects (2018)
- Phase retrieval for ODT — recovering phase from intensity-only measurements (TIE-based, 2014)

_Deep Learning (2019–2026):_
- DL-ODT — CNN-based 3D RI reconstruction from limited angles (Goy et al., PRL 2018; Kamilov et al., 2019)
- Missing cone filling with neural network — learning to extrapolate missing Fourier information (2020)
- Physics-informed neural network for ODT — encoding wave equation in loss function (2021)
- 3D U-Net for ODT reconstruction and segmentation (2020)
- GAN for ODT super-resolution — enhancing RI tomogram resolution beyond diffraction limit (2022)
- Implicit neural representation for ODT — NeRF-style continuous 3D RI field (2023)
- Self-supervised ODT — reconstruction from unpaired data (2022)
- Transformer for ODT angular fusion (2024)
- Diffusion model for ODT missing cone completion (2024)
- Foundation model for quantitative phase and RI imaging (2025)

#### Step 3: Update ODT Solvers

After listing all ODT solvers, update `algorithm_base/odt/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ODT solvers use the data format: `y` (N_angles, H, W) complex-valued fields (amplitude and phase) at different illumination angles, `illumination_angles` (N_angles, 2) (theta, phi) illumination directions, `wavelength` (float) illumination wavelength, `n_medium` (float) background refractive index. The `ODTOperator` handles the forward model (Born/Rytov scattering + angular spectrum propagation) and adjoint operations including Fourier-space mapping onto the Ewald sphere.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for ODT:**
- Microsphere RI accuracy: known RI 1.588 (polystyrene), FBP error ~0.005, TV-regularized ~0.002, DL-ODT ~0.001
- 3D RI tomogram PSNR: FBP ~25 dB, TV-ADMM ~30 dB, multi-slice ~32 dB, DL-ODT ~35 dB
- Missing cone compensation: without filling ~22 dB (axial elongation), with TV filling ~28 dB, with DL ~32 dB
- All reference values from published papers (Lim et al., 2015; Kamilov et al., 2015)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'odt' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/odt/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/odt/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/odt/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ODT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/odt/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/odt/standard/`

---

### Differential Interference Contrast (`dic`) Modality Template

#### Step 1: Verify Standard Dataset

For differential interference contrast (DIC) microscopy, what dataset do you use to verify? Is this dataset used for DIC popular algorithms? Please ensure the standard dataset in `datasets/benchmark/dic/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original DIC standard dataset.

**Popular datasets to consider:**
- **DIC Phase Reconstruction Datasets (Kou et al., Opt Lett 2010; Mehta & Sheppard, J Opt Soc Am A 2008)** — DIC images of known phase objects (microspheres, etched glass) with quantitative phase ground truth
- **DIC Cell Imaging Datasets (Hsu et al., 2015; Yin et al., 2012)** — DIC images of live cells (HeLa, fibroblasts) with paired phase contrast or fluorescence ground truth for segmentation benchmarks
- **ISBI Cell Segmentation DIC Data (Various, 2014–2018)** — DIC time-lapse cell images with manual segmentation ground truth
- **Synthetic DIC Datasets (Preza, JOSA A 2000)** — computationally generated DIC images from known 3D RI distributions for validating phase retrieval algorithms

**Decision criteria:** Kou et al. and Mehta datasets are the most widely used for quantitative DIC phase reconstruction. ISBI cell segmentation for DIC analysis tasks. Use the dataset that appears in the largest number of DIC reconstruction papers (2000–2026).

#### Step 2: List All DIC Algorithms

Please first ensure all the DIC algorithms have been listed in `\Physics_World_Model\algorithm_base\dic\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/dic. Besides, you need to search all algorithms from 1950 to 2026. After listing all the DIC solvers, please update the DIC solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1950s–2005):_
- Nomarski DIC imaging — original differential interference contrast optics producing phase-gradient images (Nomarski, 1955) — foundational DIC technique
- DIC-to-phase integration — 1D integration along shear direction to recover phase from DIC phase gradient (Arnison et al., J Microsc 2004)
- Hilbert transform DIC phase recovery — analytic signal approach to extract phase gradient (2003)
- Bias retardation calibration — quantitative DIC requiring precise Wollaston prism calibration (Mehta & Sheppard, 2008)
- Two-image DIC phase retrieval — using complementary bias settings to solve for phase (Preza, 2000)

_Optimization (2005–2018):_
- Transport of intensity equation (TIE) for DIC — phase retrieval from defocused DIC images (2005)
- Spiral phase-based DIC phase recovery — using spiral phase contrast principles for isotropic DIC (Furhapter et al., Opt Lett 2005)
- Iterative DIC phase retrieval — non-linear optimization recovering 2D phase from multi-directional DIC (Kou et al., Opt Lett 2010)
- TV-regularized DIC phase reconstruction — total variation penalty for noise-robust DIC-to-phase conversion (2012)
- Tikhonov-regularized DIC deconvolution — regularized inversion of the DIC transfer function (2008)
- Rotational diversity DIC — phase retrieval from DIC images at multiple shear directions (Mehta & Sheppard, 2008)
- ADMM-based DIC phase retrieval with non-negativity constraints (2016)
- 3D DIC sectioning — recovering 3D RI from through-focus DIC stacks (Preza, JOSA A 2000)
- Phase-from-DIC via Poisson solver — casting DIC integration as Poisson equation (Arnison et al., 2004)
- Active contour segmentation for DIC cell images (Li et al., 2010)

_Deep Learning (2018–2026):_
- CNN for DIC-to-phase — learning direct mapping from DIC image to quantitative phase (Nguyen et al., Opt Express 2018)
- PhaseGAN — GAN-based DIC-to-phase conversion (2020)
- U-Net for DIC cell segmentation — semantic segmentation of cells in DIC images (Al-Kofahi et al., 2018)
- DL DIC denoising — removing DIC-specific halo artifacts (2021)
- Cross-modality translation: DIC to fluorescence (Ounkomol et al., Nat Methods 2018)
- Physics-informed network for DIC phase retrieval — encoding shear model (2022)
- CellPose adapted for DIC cell detection (Stringer et al., 2021)
- Transformer for DIC phase reconstruction (2024)
- Diffusion model for DIC-to-phase with uncertainty (2024)
- Foundation model for label-free microscopy (DIC + phase contrast + QPI) (2025)

#### Step 3: Update DIC Solvers

After listing all DIC solvers, update `algorithm_base/dic/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All DIC solvers use the data format: `y` (H, W) DIC intensity image or (N_bias, H, W) multi-bias DIC images, `shear_angle` (float) Wollaston prism shear direction in radians, `shear_amount` (float) lateral shear distance in pixels, `bias_retardation` (float or array) Wollaston prism bias. The `DICOperator` handles the forward model (interference of sheared wavefronts with bias retardation) and adjoint operations including phase gradient extraction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for DIC:**
- Microsphere phase reconstruction: integration method RMSE ~0.3 rad, iterative ~0.15 rad, CNN ~0.08 rad
- DIC-to-phase PSNR: integration ~22 dB, TV-regularized ~27 dB, CNN ~31 dB
- DIC cell segmentation: active contour IoU ~0.65, U-Net IoU ~0.80, Cellpose IoU ~0.82
- Cross-modality DIC-to-fluorescence: SSIM >0.75 (Ounkomol et al., 2018)
- All reference values from published papers (Kou et al., 2010; Mehta & Sheppard, 2008)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'dic' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/dic/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/dic/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/dic/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for DIC. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/dic/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/dic/standard/`

---

### Dark-Field Microscopy (`dark_field`) Modality Template

#### Step 1: Verify Standard Dataset

For dark-field microscopy, what dataset do you use to verify? Is this dataset used for dark-field popular algorithms? Please ensure the standard dataset in `datasets/benchmark/dark_field/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original dark-field standard dataset.

**Popular datasets to consider:**
- **Nanoparticle Dark-Field Datasets (Sonnichsen et al., Nat Biotechnol 2005; Ringe et al., 2012)** — dark-field scattering images and spectra of gold/silver nanoparticles with known sizes and plasmonic resonances
- **Dark-Field Blood Smear Datasets (Various, 2015–2023)** — dark-field images of blood smears for malaria parasite detection and cell characterization
- **Hyperspectral Dark-Field Datasets (CytoViva, 2012–2020)** — spectrally resolved dark-field images of nanoparticles in biological matrices
- **Dark-Field Phase Contrast Datasets (Mehta et al., 2013)** — combined dark-field and phase contrast images with quantitative phase ground truth

**Decision criteria:** Nanoparticle scattering datasets are the most widely used for dark-field spectral analysis. Blood smear datasets for biomedical applications. Use the dataset that appears in the largest number of dark-field microscopy papers (2005–2026).

#### Step 2: List All Dark-Field Algorithms

Please first ensure all the dark-field algorithms have been listed in `\Physics_World_Model\algorithm_base\dark_field\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/dark_field. Besides, you need to search all algorithms from 1950 to 2026. After listing all the dark-field solvers, please update the dark-field solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1950s–2005):_
- Dark-field condenser alignment and contrast optimization — oblique illumination geometry for scattering-only detection (Zsigmondy, 1903; standard technique)
- Background subtraction for dark-field — removing residual stray light (standard preprocessing)
- Mie theory fitting — fitting dark-field scattering spectra to Mie theory for nanoparticle sizing (Mie, 1908; applied to single-particle spectroscopy 2000s)
- Lorentzian peak fitting — extracting plasmonic resonance wavelength and linewidth from dark-field spectra (2003)
- Rayleigh scattering analysis — quantifying particle scattering cross-section from dark-field intensity (2000s)

_Optimization (2006–2018):_
- Spectral deconvolution for dark-field — unmixing overlapping nanoparticle scattering spectra (2008)
- FDTD-based inverse scattering — reconstructing nanoparticle geometry from dark-field scattering pattern via FDTD simulation (2010)
- Sparse spectral unmixing — L1-regularized decomposition of hyperspectral dark-field data (2014)
- Template matching for nanoparticle detection — correlation-based detection in dark-field images (2012)
- Multi-particle tracking in dark-field — linking detected scatterers across time-lapse frames (2010)
- TV-regularized dark-field image denoising (2013)
- Bayesian nanoparticle classification from dark-field spectra (2016)
- Total internal reflection dark-field — evanescent wave excitation for surface-selective scattering (2008)
- Dark-field tomography — angular scanning dark-field for 3D scatterer reconstruction (2015)
- Polarization-resolved dark-field — extracting nanoparticle orientation from polarimetric scattering (2014)

_Deep Learning (2018–2026):_
- CNN for nanoparticle classification from dark-field spectra (2018)
- DL nanoparticle sizing from dark-field images — predicting particle diameter from scattering pattern (2019)
- U-Net for dark-field cell segmentation (2020)
- GAN for dark-field image enhancement — improving contrast and removing artifacts (2021)
- YOLO for real-time nanoparticle detection in dark-field video (2021)
- Deep spectral unmixing for hyperspectral dark-field (2022)
- Physics-informed network for Mie scattering inversion (2023)
- Self-supervised dark-field denoising (2023)
- Transformer for dark-field spectral classification (2024)
- Foundation model for scattering microscopy (dark-field + iSCAT) (2025)

#### Step 3: Update Dark-Field Solvers

After listing all dark-field solvers, update `algorithm_base/dark_field/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All dark-field solvers use the data format: `y` (H, W) dark-field intensity image or (N_wavelengths, H, W) hyperspectral dark-field datacube, `illumination_na` (float, float) inner and outer NA of dark-field condenser, `wavelengths` (N_wavelengths,) spectral channels. The `DarkFieldOperator` handles the forward model (scattering cross-section * illumination + background) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for dark-field:**
- Nanoparticle sizing: Mie fitting error <5 nm for 50–150 nm gold nanoparticles, DL sizing error <3 nm
- Spectral peak detection: Lorentzian fit R-squared >0.95, DL peak prediction error <2 nm
- Nanoparticle detection: template matching precision >0.85, YOLO mAP >0.90
- Dark-field denoising: raw SNR ~12 dB, TV ~22 dB, DL ~26 dB
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'dark_field' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/dark_field/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/dark_field/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/dark_field/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for dark-field. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/dark_field/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/dark_field/standard/`

---

### Phase Contrast Microscopy (`phase_contrast`) Modality Template

#### Step 1: Verify Standard Dataset

For phase contrast microscopy, what dataset do you use to verify? Is this dataset used for phase contrast popular algorithms? Please ensure the standard dataset in `datasets/benchmark/phase_contrast/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original phase contrast standard dataset.

**Popular datasets to consider:**
- **ISBI Cell Tracking Challenge Phase Contrast Data (Ulman et al., 2017)** — PhC-C2DH and PhC-C2DL time-lapse phase contrast sequences with tracking and segmentation ground truth; the primary phase contrast benchmark
- **Sartorius Cell Instance Segmentation Dataset (2021)** — large-scale phase contrast cell images with instance segmentation labels from Kaggle competition
- **Phase Contrast Cell Counting Datasets (Arteta et al., MICCAI 2012; Lempitsky & Zisserman, NIPS 2010)** — phase contrast images with dot annotations for counting benchmarks
- **Label-Free Cell Datasets (Vicar et al., 2019)** — phase contrast images of multiple cell types with proliferation and morphology annotations

**Decision criteria:** ISBI Cell Tracking Challenge phase contrast data is the canonical benchmark for cell tracking and segmentation. Sartorius dataset for large-scale instance segmentation. Use the dataset that appears in the largest number of phase contrast analysis papers (2010–2026).

#### Step 2: List All Phase Contrast Algorithms

Please first ensure all the phase contrast algorithms have been listed in `\Physics_World_Model\algorithm_base\phase_contrast\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/phase_contrast. Besides, you need to search all algorithms from 1950 to 2026. After listing all the phase contrast solvers, please update the phase contrast solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1950s–2005):_
- Zernike phase contrast — original phase plate method converting phase to intensity (Zernike, 1942; Nobel Prize 1953) — foundational technique
- Phase contrast artifact correction — halo and shade-off artifact removal by background subtraction and flatfield correction (1980s)
- Phase contrast-to-phase conversion — recovering quantitative phase from Zernike phase contrast images (Nagata et al., 2003)
- Transport of intensity equation (TIE) — phase retrieval from through-focus intensity images (Teague, JOSA 1983; applied 1990s)
- Bright-field to phase contrast conversion — computational generation of phase contrast from bright-field stacks (2000s)

_Optimization (2005–2018):_
- TIE with Tikhonov regularization — regularized Poisson solver for phase retrieval from defocus series (Zuo et al., Opt Express 2013)
- Variational phase retrieval from phase contrast images — optimization-based quantitative phase recovery (Yin et al., 2012)
- Active contour for phase contrast cell segmentation — level-set and snake methods adapted for halo artifacts (Tsai et al., 2009; Bentabet et al., 2003)
- Watershed segmentation for phase contrast cell images (2006)
- Graph-cut segmentation adapted for phase contrast (Al-Kofahi et al., 2010)
- TV-regularized phase contrast denoising and artifact removal (2014)
- Dictionary-learning-based phase contrast image enhancement (2015)
- Phase contrast optical flow for cell motility estimation (2010)
- Multi-frequency TIE — using multiple defocus distances for robust phase recovery (2016)
- Sparse phase retrieval from phase contrast intensity (2017)

_Deep Learning (2017–2026):_
- U-Net for phase contrast cell segmentation (Falk et al., Nat Methods 2019) — widely used baseline
- Cellpose for phase contrast (Stringer et al., Nat Methods 2021)
- StarDist adapted for phase contrast nuclei detection (Schmidt et al., 2018)
- CNN for phase contrast-to-fluorescence prediction (Ounkomol et al., Nat Methods 2018; Christiansen et al., Cell 2018)
- DL phase retrieval from phase contrast images (2020)
- GAN for phase contrast halo artifact removal (2021)
- DeepCell — DL platform for phase contrast cell analysis (Bannon et al., Nat Biotechnol 2021)
- Self-supervised cell segmentation for phase contrast (2022)
- Segment Anything Model adapted for phase contrast cell segmentation (2023)
- Transformer for phase contrast cell tracking (2024)
- Foundation model for label-free cell microscopy (2025)

#### Step 3: Update Phase Contrast Solvers

After listing all phase contrast solvers, update `algorithm_base/phase_contrast/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All phase contrast solvers use the data format: `y` (H, W) phase contrast intensity image or (T, H, W) time-lapse, `phase_ring_params` (dict) phase ring geometry and retardation, `defocus_stack` (N_defocus, H, W) optional through-focus images for TIE-based phase retrieval. The `PhaseContrastOperator` handles the forward model (Zernike phase ring modulation of pupil plane + coherent imaging) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for phase contrast:**
- Cell Tracking Challenge (PhC-C2DH-U373): U-Net SEG >0.85, Cellpose SEG >0.88, TRA >0.95 for top methods
- Sartorius instance segmentation: Cellpose mAP@0.5 >0.70, SAM-adapted >0.75
- Phase retrieval from phase contrast: TIE RMSE <0.2 rad, DL phase recovery RMSE <0.1 rad
- Phase contrast-to-fluorescence: SSIM >0.70 (Ounkomol et al., 2018)
- All reference values from Cell Tracking Challenge leaderboard and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'phase_contrast' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/phase_contrast/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/phase_contrast/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/phase_contrast/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for phase contrast. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/phase_contrast/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_contrast/standard/`

---

### TIRF Microscopy (`tirf`) Modality Template

#### Step 1: Verify Standard Dataset

For total internal reflection fluorescence (TIRF) microscopy, what dataset do you use to verify? Is this dataset used for TIRF popular algorithms? Please ensure the standard dataset in `datasets/benchmark/tirf/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original TIRF standard dataset.

**Popular datasets to consider:**
- **ISBI Single-Molecule Localization Challenge (Sage et al., Nat Methods 2015; 2019)** — simulated and real TIRF-SMLM data with known ground truth molecule positions; the canonical TIRF/SMLM benchmark
- **TIRF Vesicle Fusion Datasets (Zenisek et al., 2000; Steyer et al., 2001)** — TIRF time-lapse of synaptic vesicle exocytosis and endocytosis events at the plasma membrane
- **TIRF Cytoskeleton Datasets (Risiberg et al., 2019)** — TIRF images of actin dynamics and focal adhesions near the coverslip surface
- **TIRF Single-Particle Tracking Datasets (Jaqaman et al., Nat Methods 2008)** — TIRF videos of fluorescent particles with tracking ground truth

**Decision criteria:** ISBI SMLM challenge data (TIRF-based) is the most widely benchmarked. Vesicle fusion datasets for temporal analysis. Use the dataset that appears in the largest number of TIRF analysis papers (2000–2026).

#### Step 2: List All TIRF Algorithms

Please first ensure all the TIRF algorithms have been listed in `\Physics_World_Model\algorithm_base\tirf\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/tirf. Besides, you need to search all algorithms from 1950 to 2026. After listing all the TIRF solvers, please update the TIRF solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1981–2005):_
- Evanescent wave penetration depth calculation — computing 1/e depth from incidence angle and refractive indices (Axelrod, J Cell Biol 1981) — foundational TIRF theory
- TIRF flat-field correction — correcting for non-uniform evanescent field illumination (2000s)
- Single-particle detection by local maximum finding — intensity thresholding and centroid estimation in TIRF images (2000s)
- Gaussian PSF fitting for single-molecule localization — 2D Gaussian fitting to diffraction-limited spots in TIRF (Thompson et al., Biophys J 2002)
- Nearest-neighbor particle linking for single-particle tracking (2003)

_Optimization (2006–2018):_
- Multi-angle TIRF (MA-TIRF) — varying incidence angle to reconstruct axial fluorophore distribution (Ruckstuhl et al., 2003; Boulanger et al., 2014)
- Variable-angle TIRF axial nanometry — computing z-position from multi-angle TIRF intensities (Leutenegger et al., 2012)
- Deconvolution of TIRF images — Richardson-Lucy with evanescent wave PSF model (2008)
- Single-molecule fitting with sCMOS noise model — MLE fitting accounting for pixel-dependent noise (Huang et al., 2013)
- Multiple hypothesis tracking (MHT) for dense TIRF particle tracking (Chetverikov & Verestoy, 1999; Jaqaman et al., 2008)
- u-track — comprehensive single-particle tracking software for TIRF data (Jaqaman et al., Nat Methods 2008)
- Bayesian multi-emitter fitting — resolving overlapping PSFs in dense TIRF images (Leutenegger et al., 2011)
- TV-regularized TIRF image denoising (2012)
- TIRF-SIM — combining TIRF with structured illumination for surface super-resolution (Li et al., Science 2015)
- Sparse localization algorithms for TIRF-SMLM (Zhu et al., 2012)

_Deep Learning (2018–2026):_
- DeepSTORM — CNN-based single-molecule localization from dense TIRF emitter images (Nehme et al., Optica 2018)
- DECODE — deep context-dependent single-molecule localization (Speiser et al., Nat Methods 2021)
- Deep-STORM3D — 3D localization from TIRF with engineered PSFs (Nehme et al., 2020)
- DL vesicle detection in TIRF — CNN for automated exocytosis event detection (2020)
- U-Net for TIRF image segmentation of membrane structures (2019)
- Graph neural network for TIRF particle tracking (2022)
- Self-supervised TIRF denoising — exploiting temporal redundancy (2021)
- Transformer for dense particle tracking in TIRF video (2023)
- Physics-informed localization network for TIRF-SMLM (2024)
- Foundation model for single-molecule microscopy (2025)

#### Step 3: Update TIRF Solvers

After listing all TIRF solvers, update `algorithm_base/tirf/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All TIRF solvers use the data format: `y` (T, H, W) TIRF time-lapse or (N_angles, H, W) multi-angle TIRF, `psf` (H_psf, W_psf) TIRF PSF (potentially engineered), `penetration_depth` (float) evanescent wave 1/e depth in nm, `pixel_size` (float) nm/pixel. The `TIRFOperator` handles the forward model (evanescent illumination * PSF convolution + Poisson noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for TIRF:**
- ISBI SMLM Challenge 2D: Gaussian fitting RMSE ~25 nm, DeepSTORM RMSE ~15 nm, DECODE RMSE ~12 nm at moderate density
- Single-particle tracking: u-track linking accuracy >0.90, DL tracking accuracy >0.93
- Multi-angle TIRF axial resolution: ~20 nm axial localization precision
- TIRF denoising: raw SNR ~10 dB, temporal averaging ~18 dB, DL denoising ~24 dB
- All reference values from ISBI SMLM challenge (Sage et al., 2019) and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'tirf' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/tirf/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/tirf/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/tirf/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for TIRF. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/tirf/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/tirf/standard/`

---

### Two-Photon Microscopy (`two_photon`) Modality Template

#### Step 1: Verify Standard Dataset

For two-photon microscopy (2PM), what dataset do you use to verify? Is this dataset used for 2PM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/two_photon/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original 2PM standard dataset.

**Popular datasets to consider:**
- **Allen Brain Observatory Two-Photon Datasets (de Vries et al., Nat Neurosci 2020)** — large-scale in vivo two-photon calcium imaging of mouse visual cortex with cell segmentation and activity ground truth; the primary 2PM benchmark
- **Neurofinder Challenge Data (CodeNeuro, 2016)** — two-photon calcium imaging datasets with manually annotated neuron ROIs for segmentation benchmarking
- **GENIE Project Two-Photon Data (Janelia, 2013–2020)** — in vivo two-photon data of GCaMP-expressing neurons with electrophysiology ground truth
- **CaImAn Benchmark Data (Giovannucci et al., eLife 2019)** — two-photon calcium imaging test datasets with manually annotated neurons and inferred spikes

**Decision criteria:** Allen Brain Observatory is the largest and most widely used 2PM benchmark (2020–2026). Neurofinder for segmentation evaluation. Use the dataset that appears in the largest number of two-photon analysis papers (2016–2026).

#### Step 2: List All 2PM Algorithms

Please first ensure all the 2PM algorithms have been listed in `\Physics_World_Model\algorithm_base\two_photon\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/two_photon. Besides, you need to search all algorithms from 1950 to 2026. After listing all the 2PM solvers, please update the 2PM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990–2008):_
- Two-photon excitation theory — nonlinear fluorescence excitation with femtosecond pulsed laser (Denk et al., Science 1990) — foundational 2PM paper
- Motion correction — rigid and non-rigid registration of two-photon time-lapse for brain motion compensation (Dombeck et al., 2007)
- Adaptive optics for two-photon — wavefront correction for deep tissue imaging (Ji et al., Nat Methods 2010)
- Deconvolution with two-photon PSF — 3D deconvolution using the nonlinear (squared) PSF (2000s)
- Background fluorescence subtraction — removing neuropil contamination from somatic signals (2005)

_Optimization (2008–2018):_
- CaImAn — Constrained Nonnegative Matrix Factorization (CNMF) for simultaneous source extraction and deconvolution of calcium signals (Pnevmatikakis et al., Neuron 2016; Giovannucci et al., eLife 2019) — the gold-standard 2PM analysis pipeline
- Suite2p — fast registration, cell detection, and spike deconvolution for two-photon data (Pachitariu et al., bioRxiv 2016; Stringer et al., 2021) — most widely used 2PM pipeline
- CNMF-E — CNMF for one-photon / endoscopic data with background model (Zhou et al., eLife 2018)
- ICA/PCA for ROI detection — independent component analysis for neuronal source separation (Mukamel et al., Neuron 2009)
- OASIS — fast online deconvolution of calcium transients (Friedrich et al., PLoS Comp Biol 2017)
- Non-rigid motion correction — piecewise affine and optical flow methods (Pnevmatikakis & Giovannucci, 2017)
- Neuropil contamination correction — regression-based subtraction of surrounding neuropil signal (2014)
- Compressed sensing for two-photon — accelerating acquisition via sparse sampling (2015)
- Blind deconvolution for deep two-photon imaging (2016)
- Adaptive optics with modal wavefront sensing — Zernike-based correction (Ji, 2017)

_Deep Learning (2018–2026):_
- DeepInterpolation — removing shot noise from two-photon data via temporal interpolation (Lecoq et al., Nat Methods 2021)
- Cellpose for two-photon neuron segmentation (Stringer et al., 2021)
- DL calcium signal denoising — CNN-based denoising of calcium traces (2020)
- DeepCAD — deep self-supervised calcium imaging denoising (Li et al., Nat Methods 2021)
- Cascade — deep learning for spike inference from calcium imaging (Rupprecht et al., Nat Neurosci 2021)
- VolPy — pipeline for voltage imaging analysis from two-photon data (Cai et al., 2021)
- 3D two-photon segmentation with 3D U-Net (2022)
- Transformer for two-photon calcium signal analysis (2023)
- Self-supervised motion correction for two-photon (2023)
- Foundation model for neural imaging (two-photon + widefield + fMRI) (2025)

#### Step 3: Update 2PM Solvers

After listing all 2PM solvers, update `algorithm_base/two_photon/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All 2PM solvers use the data format: `y` (T, H, W) two-photon calcium imaging time-lapse, `psf` (D_psf, H_psf, W_psf) two-photon PSF (squared excitation profile), `frame_rate` (float) imaging frame rate in Hz. The `TwoPhotonOperator` handles the forward model (nonlinear excitation * detection PSF + Poisson shot noise + readout noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for 2PM:**
- Neurofinder segmentation: Suite2p F1 ~0.70, CaImAn F1 ~0.72, Cellpose F1 ~0.78
- Allen Brain Observatory: Suite2p cell detection recall >0.85
- Spike inference: OASIS correlation ~0.60, Cascade correlation ~0.75 with ground truth electrophysiology
- Two-photon denoising: raw SNR ~8 dB, DeepInterpolation +7 dB improvement, DeepCAD +8 dB
- Motion correction: Suite2p registration error <1 pixel on standard data
- All reference values from published papers and Neurofinder leaderboard

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'two_photon' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/two_photon/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/two_photon/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/two_photon/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for 2PM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/two_photon/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/two_photon/standard/`

---

### Three-Photon Microscopy (`three_photon`) Modality Template

#### Step 1: Verify Standard Dataset

For three-photon microscopy (3PM), what dataset do you use to verify? Is this dataset used for 3PM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/three_photon/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original 3PM standard dataset.

**Popular datasets to consider:**
- **Horton et al. Three-Photon Datasets (Horton et al., Nat Photonics 2013)** — the original three-photon brain imaging data demonstrating deep tissue penetration (>1 mm) with GCaMP6 and Texas Red dextran
- **Wang et al. Deep Brain 3PM Data (Wang et al., Nat Methods 2018)** — three-photon calcium imaging at >1 mm depth in mouse hippocampus with neuronal activity ground truth
- **Ouzounov et al. 3PM Datasets (Ouzounov et al., Nat Methods 2017)** — three-photon functional imaging through the intact mouse skull
- **Three-Photon Vascular Imaging Data (2019–2023)** — three-photon angiography in deep cortical layers and hippocampus with structural ground truth

**Decision criteria:** Wang et al. hippocampus data is the most widely referenced 3PM benchmark. Ouzounov et al. for through-skull imaging. Use the dataset that appears in the largest number of 3PM papers (2013–2026).

#### Step 2: List All 3PM Algorithms

Please first ensure all the 3PM algorithms have been listed in `\Physics_World_Model\algorithm_base\three_photon\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/three_photon. Besides, you need to search all algorithms from 1950 to 2026. After listing all the 3PM solvers, please update the 3PM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2013–2018):_
- Three-photon excitation efficiency calculation — third-order nonlinear excitation cross-section estimation (Horton et al., 2013)
- Adaptive optics for 3PM — wavefront correction for imaging through scattering tissue at >1 mm depth (2017)
- 3PM PSF characterization — measuring the cubic (third-order) PSF of three-photon excitation (2014)
- Background fluorescence ratio (SBR) calculation — tissue-specific signal-to-background estimation for 3PM depth optimization (Wang et al., 2018)
- Motion correction adapted for deep 3PM — registration accounting for large tissue motion at depth (2018)

_Optimization (2016–2022):_
- Suite2p adapted for 3PM — extending Suite2p pipeline for low-SNR three-photon calcium imaging (2019)
- CaImAn for 3PM — CNMF adapted for the low frame rate and low SNR of 3PM data (2020)
- Aberration correction via indirect wavefront sensing for 3PM (Liu et al., 2019)
- Deconvolution with third-order PSF — RL deconvolution using measured 3PM PSF (2018)
- Pulse compression optimization — maximizing 3PM signal by compensating for group velocity dispersion (2016)
- Adaptive excitation for 3PM — power modulation to equalize signal across depth (2019)
- Motion-corrected temporal denoising for 3PM — exploiting slow frame rates with temporal filtering (2020)
- Sparse signal extraction for low-SNR 3PM data (2021)
- OASIS adapted for low-SNR 3PM calcium traces (2020)
- Joint registration and denoising for 3PM (2022)

_Deep Learning (2021–2026):_
- DeepInterpolation adapted for 3PM — temporal interpolation denoising for three-photon data (2021)
- DeepCAD for 3PM — self-supervised calcium imaging denoising at depth (2022)
- DL cell detection for low-SNR 3PM — CNN detecting neurons in noisy deep brain 3PM data (2022)
- 3PM-specific motion correction with DL optical flow (2023)
- Transfer learning from 2PM to 3PM — leveraging 2PM-trained models for 3PM analysis (2023)
- Physics-informed denoising for 3PM — encoding third-order PSF in network architecture (2024)
- Diffusion model for 3PM enhancement — generating high-SNR predictions from low-SNR 3PM (2024)
- Foundation model for multiphoton imaging (2PM + 3PM) (2025)

#### Step 3: Update 3PM Solvers

After listing all 3PM solvers, update `algorithm_base/three_photon/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All 3PM solvers use the data format: `y` (T, H, W) three-photon time-lapse or (T, D, H, W) volumetric 3PM data, `psf` (D_psf, H_psf, W_psf) third-order PSF (cubed excitation profile), `imaging_depth` (float) depth in tissue in micrometers, `frame_rate` (float) Hz. The `ThreePhotonOperator` handles the forward model (third-order nonlinear excitation * tissue scattering attenuation + Poisson shot noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for 3PM:**
- 3PM imaging depth: >1 mm in mouse cortex, >1.5 mm with adaptive optics
- 3PM denoising: raw SNR ~5 dB at depth, DeepInterpolation +5–7 dB, DeepCAD +6–8 dB
- Cell detection at depth: manual annotation recall baseline, DL detection F1 >0.60 at >1 mm depth
- 3PM calcium trace quality: spike inference correlation >0.50 at 1 mm depth (lower than 2PM due to SNR)
- All reference values from published papers (Wang et al., 2018; Ouzounov et al., 2017)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'three_photon' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/three_photon/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/three_photon/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/three_photon/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for 3PM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/three_photon/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/three_photon/standard/`

---

### STED Microscopy (`sted`) Modality Template

#### Step 1: Verify Standard Dataset

For stimulated emission depletion (STED) microscopy, what dataset do you use to verify? Is this dataset used for STED popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sted/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original STED standard dataset.

**Popular datasets to consider:**
- **Abberior STED Benchmark Datasets (Various, 2015–2023)** — STED images of standard biological structures (microtubules, nuclear pores, synapses) acquired on Abberior instruments with paired confocal images
- **STED Deconvolution Challenge Data (Leutenegger et al., 2018)** — STED images with known PSF and paired high-SNR ground truth for deconvolution benchmarking
- **STED Neuroscience Datasets (Berning et al., Science 2012; Urban et al., eLife 2011)** — in vivo and ex vivo STED images of synaptic structures and dendritic spines
- **BioImage Model Zoo STED Data (2021–2024)** — community-curated STED images for DL-based restoration and super-resolution benchmarks

**Decision criteria:** Abberior STED benchmark images are the most commonly used for resolution comparisons. Neuroscience STED for application-specific benchmarks. Use the dataset that appears in the largest number of STED analysis papers (2012–2026).

#### Step 2: List All STED Algorithms

Please first ensure all the STED algorithms have been listed in `\Physics_World_Model\algorithm_base\sted\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sted. Besides, you need to search all algorithms from 1950 to 2026. After listing all the STED solvers, please update the STED solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1994–2010):_
- STED PSF engineering — designing donut-shaped depletion beam for lateral super-resolution (Hell & Wichmann, Opt Lett 1994) — the foundational STED concept (Nobel Prize 2014)
- 3D STED — axial super-resolution with bottle-beam depletion (Klar et al., PNAS 2000)
- Wiener deconvolution with STED PSF — regularized inverse filtering using the effective STED PSF (2005)
- STED-FCS — fluorescence correlation spectroscopy at nanoscale observation volumes (Eggeling et al., Nature 2009)
- Linear unmixing for multi-color STED — spectral separation in multi-channel STED (2008)

_Optimization (2010–2020):_
- Richardson-Lucy deconvolution for STED — iterative deconvolution with measured STED PSF for improved resolution (Leutenegger et al., 2006; Vicidomini et al., Nat Methods 2018)
- STED with adaptive illumination — RESCue and DyMIN for reducing photobleaching (Staudt et al., 2011; Heine et al., Nat Methods 2017)
- Gated STED (gSTED) — time-gated detection for improved resolution with lower depletion power (Vicidomini et al., Nat Methods 2011)
- SPLIT-STED — separation of photons by lifetime tuning for resolution enhancement (Lanzano et al., Nat Methods 2015)
- TV-regularized STED deconvolution (2014)
- Sparse deconvolution for STED — L1-regularized reconstruction for sparse fluorophore distributions (2016)
- Blind STED deconvolution — estimating effective PSF from STED data (2015)
- STED-FLIM — combining STED with fluorescence lifetime for molecular environment sensing (2012)
- Multi-scale STED reconstruction — fusing STED and confocal data at different resolution scales (2017)
- ADMM-based STED deconvolution with non-negativity and sparsity constraints (2019)

_Deep Learning (2019–2026):_
- DL STED denoising — CNN-based denoising for low-power STED (Heine et al., 2017; Weigert et al., 2018)
- STED-to-confocal transfer — training on paired STED/confocal for cross-modality prediction (2020)
- Physics-informed STED deconvolution — unrolled RL with learned STED PSF parameters (2021)
- GAN for STED image restoration — enhancing low-dose STED to match high-dose quality (2021)
- Confocal-to-STED prediction — DL super-resolution from confocal to STED resolution (2022)
- Self-supervised STED denoising (Noise2Void adapted) (2022)
- 3D STED reconstruction with deep learning (2023)
- Transformer for STED image analysis and deconvolution (2024)
- Diffusion model for STED restoration with uncertainty quantification (2024)
- Foundation model for super-resolution microscopy (STED + SIM + SMLM) (2025)

#### Step 3: Update STED Solvers

After listing all STED solvers, update `algorithm_base/sted/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All STED solvers use the data format: `y` (H, W) STED image or (T, H, W) time-gated STED data, `psf_sted` (H_psf, W_psf) effective STED PSF (depends on depletion power), `psf_confocal` (H_psf, W_psf) confocal PSF for comparison, `depletion_power` (float) STED depletion beam power. The `STEDOperator` handles the forward model (effective STED PSF convolution accounting for depletion donut + Poisson noise + photobleaching) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for STED:**
- STED resolution: confocal ~250 nm, STED ~50 nm lateral, 3D-STED ~100 nm axial
- STED deconvolution: raw STED ~28 dB, RL deconvolution ~32 dB, DL denoising ~35 dB PSNR on paired data
- gSTED improvement: 1.5–2x resolution gain over standard STED at same depletion power
- Low-power STED restoration: DL achieving equivalent quality to 3x higher depletion power
- All reference values from published papers (Vicidomini et al., 2018; Heine et al., 2017)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sted' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sted/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sted/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sted/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for STED. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sted/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sted/standard/`

---

### PALM/STORM Single-Molecule Localization (`palm_storm`) Modality Template

#### Step 1: Verify Standard Dataset

For PALM/STORM single-molecule localization microscopy (SMLM), what dataset do you use to verify? Is this dataset used for PALM/STORM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/palm_storm/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original PALM/STORM standard dataset.

**Popular datasets to consider:**
- **ISBI SMLM Challenge Datasets (Sage et al., Nat Methods 2015; 2019)** — comprehensive simulated and real SMLM data with known ground truth molecule positions at varying densities; the undisputed gold-standard SMLM benchmark
- **EPFL SMLM Datasets (Sage et al., 2015)** — microtubule and mitochondria SMLM data with 2D and 3D ground truth localizations
- **STORM Tubulin Datasets (Rust et al., Nat Methods 2006; Huang et al., Science 2008)** — classic STORM images of immunolabeled microtubules demonstrating sub-diffraction resolution
- **PALM Live-Cell Datasets (Betzig et al., Science 2006; Hess et al., Biophys J 2006)** — founding PALM datasets of photoactivatable fluorescent proteins in live cells

**Decision criteria:** ISBI SMLM Challenge datasets are the universal benchmark for localization algorithm comparison (2015–2026). EPFL datasets for 3D SMLM. Use the dataset that appears in the largest number of SMLM papers (2006–2026).

#### Step 2: List All PALM/STORM Algorithms

Please first ensure all the PALM/STORM algorithms have been listed in `\Physics_World_Model\algorithm_base\palm_storm\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/palm_storm. Besides, you need to search all algorithms from 1950 to 2026. After listing all the PALM/STORM solvers, please update the PALM/STORM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2006–2012):_
- Gaussian PSF fitting — least-squares or MLE 2D Gaussian fitting for single-molecule localization (Thompson et al., Biophys J 2002) — the foundational SMLM localization method
- Centroid estimation — center-of-mass localization for isolated emitters (2006)
- 3D localization via astigmatism — cylindrical lens PSF engineering for z-encoding (Huang et al., Science 2008)
- 3D localization via biplane detection — dual focal plane imaging for axial position (Juette et al., Nat Methods 2008)
- Double-helix PSF for 3D SMLM — rotating PSF encoding axial position over extended range (Pavani et al., PNAS 2009)

_Optimization (2010–2020):_
- DAOSTORM — multi-emitter fitting for high-density SMLM (Holden et al., Nat Methods 2011)
- ThunderSTORM — comprehensive ImageJ plugin for SMLM analysis (Ovesny et al., Bioinformatics 2014) — most widely used SMLM software
- rapidSTORM — real-time SMLM localization engine (Wolter et al., Nat Methods 2012)
- CSSTORM — compressed sensing STORM for high-density localization (Zhu et al., Nat Methods 2012)
- 3B analysis — Bayesian analysis of blinking and bleaching for SMLM (Cox et al., Nat Methods 2012)
- SRRF — super-resolution radial fluctuations for live-cell SMLM-like imaging (Gustafsson et al., Nat Commun 2016)
- SOFI — super-resolution optical fluctuation imaging from blinking statistics (Dertinger et al., PNAS 2009)
- MLE with sCMOS noise model — maximum likelihood estimation accounting for pixel-dependent read noise (Huang et al., Nat Methods 2013)
- Multi-emitter fitting with model selection — BIC/AIC-based determination of emitter number per diffraction spot (2015)
- Drift correction for SMLM — fiducial-based and cross-correlation drift correction (2012)
- FALCON — fast algorithm for localization by convex optimization (Min et al., Sci Rep 2014)
- Sparsity-based SMLM — SPIDER and related L1-minimization approaches (2013)
- Tetrapod PSF for extended-range 3D SMLM (Shechtman et al., Nat Photonics 2015)
- Phasor-based localization for rapid SMLM (Reymond et al., 2019)

_Deep Learning (2018–2026):_
- DeepSTORM — CNN-based dense emitter localization (Nehme et al., Optica 2018)
- Deep-STORM3D — 3D single-molecule localization via deep learning (Nehme et al., Nat Methods 2020)
- DECODE — deep context-dependent SMLM with Bayesian uncertainty (Speiser et al., Nat Methods 2021) — state-of-the-art DL SMLM
- ANNA-PALM — artificial neural network accelerated PALM (Ouyang et al., Nat Biotechnol 2018)
- ZeroCostDL4Mic SMLM models (von Chamier et al., 2021)
- FD-DeepLoc — field-dependent deep learning localization (2021)
- DL drift correction for SMLM (2022)
- Transformer for SMLM localization (2023)
- Diffusion model for SMLM image reconstruction (2024)
- Physics-informed localization network with aberration-aware PSF model (2024)
- Foundation model for single-molecule imaging (2025)

#### Step 3: Update PALM/STORM Solvers

After listing all PALM/STORM solvers, update `algorithm_base/palm_storm/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All PALM/STORM solvers use the data format: `y` (T, H, W) SMLM image sequence (thousands of frames with sparse blinking emitters), `psf_model` (callable or array) PSF model (Gaussian, astigmatic, double-helix, or experimentally measured), `pixel_size` (float) nm/pixel, `camera_params` (dict) gain, offset, readout noise. The `SMLMOperator` handles the forward model (emitter positions * PSF + Poisson shot noise + camera noise) and adjoint operations producing localization lists.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for PALM/STORM:**
- ISBI SMLM 2D (low density): Gaussian MLE RMSE ~15 nm, ThunderSTORM ~14 nm, DECODE ~10 nm
- ISBI SMLM 2D (high density): DAOSTORM ~25 nm, DeepSTORM ~18 nm, DECODE ~15 nm
- ISBI SMLM 3D (astigmatism): 3D MLE lateral ~15 nm / axial ~40 nm, Deep-STORM3D lateral ~12 nm / axial ~30 nm
- Jaccard index at 50 nm threshold: low density >0.90, high density >0.70 for top methods
- All reference values from ISBI SMLM challenge (Sage et al., 2019)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'palm_storm' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/palm_storm/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/palm_storm/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/palm_storm/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for PALM/STORM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/palm_storm/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/palm_storm/standard/`

---

### DNA-PAINT Super-Resolution (`dna_paint`) Modality Template

#### Step 1: Verify Standard Dataset

For DNA-PAINT super-resolution microscopy, what dataset do you use to verify? Is this dataset used for DNA-PAINT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/dna_paint/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original DNA-PAINT standard dataset.

**Popular datasets to consider:**
- **Jungmann Lab DNA-PAINT Datasets (Jungmann et al., Nano Lett 2010; Nat Methods 2014)** — the original DNA-PAINT data including origami nanostructure calibration samples and cellular targets with known ground truth positions
- **DNA-PAINT Origami Nanoruler Data (Schnitzbauer et al., Nat Protoc 2017)** — DNA origami structures with precisely known inter-fluorophore distances for resolution quantification
- **Exchange-PAINT Multi-Target Data (Jungmann et al., Nat Methods 2014)** — sequential multi-color DNA-PAINT of different cellular targets using exchangeable imager strands
- **Quantitative DNA-PAINT (qPAINT) Datasets (Jungmann et al., Nat Methods 2016)** — DNA-PAINT data calibrated for molecular counting via binding kinetics

**Decision criteria:** DNA origami nanoruler datasets are the gold standard for DNA-PAINT resolution validation. Jungmann lab cellular datasets for biological benchmarks. Use the dataset that appears in the largest number of DNA-PAINT papers (2010–2026).

#### Step 2: List All DNA-PAINT Algorithms

Please first ensure all the DNA-PAINT algorithms have been listed in `\Physics_World_Model\algorithm_base\dna_paint\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/dna_paint. Besides, you need to search all algorithms from 1950 to 2026. After listing all the DNA-PAINT solvers, please update the DNA-PAINT solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2010–2015):_
- DNA-PAINT localization — standard Gaussian PSF fitting of transient binding events (Jungmann et al., Nano Lett 2010; Nat Methods 2014) — foundational DNA-PAINT analysis
- Binding kinetics analysis — extracting on-time (tau_on) and off-time (tau_off) from single-site binding traces (Jungmann et al., 2014)
- qPAINT counting — quantitative molecular counting from binding frequency (Jungmann et al., Nat Methods 2016)
- Drift correction via DNA origami fiducials — using origami structures as drift markers (2014)
- Multi-color Exchange-PAINT sequential imaging — buffer exchange protocol for multiplexed imaging (2014)

_Optimization (2015–2021):_
- Picasso — comprehensive DNA-PAINT analysis software with localization, rendering, and kinetic analysis (Schnitzbauer et al., Nat Protoc 2017) — the standard DNA-PAINT analysis tool
- MLE localization with DNA-PAINT-specific noise model — accounting for transient binding duration and photon budget (2017)
- Kinetic rate extraction via hidden Markov model — HMM-based analysis of binding/unbinding events (2018)
- Multi-emitter fitting for high-density DNA-PAINT — resolving simultaneously bound imagers (2019)
- 3D DNA-PAINT with engineered PSFs — astigmatic or double-helix PSF for axial localization (Dai et al., 2016)
- Drift correction via redundant cross-correlation — image-based drift correction exploiting temporal redundancy (2018)
- Clustering analysis for DNA-PAINT — DBSCAN and Bayesian clustering of localizations (2019)
- Speed-optimized DNA-PAINT — accelerated acquisition via optimized imager strand design (Strauss & Jungmann, Nat Methods 2020)
- RESI — Resolution Enhancement by Sequential Imaging for sub-nanometer precision (Reinhardt et al., Nature 2023)
- Bayesian grouping of localizations — probabilistic assignment of localizations to binding sites (2020)

_Deep Learning (2020–2026):_
- DL localization for DNA-PAINT — DECODE/DeepSTORM adapted for DNA-PAINT binding kinetics (2020)
- CNN for DNA-PAINT binding event classification — distinguishing specific from non-specific binding (2021)
- DL-accelerated DNA-PAINT — reducing acquisition time via dense frame analysis (2022)
- Neural network for qPAINT molecular counting (2022)
- GAN for DNA-PAINT image enhancement (2023)
- Self-supervised denoising for DNA-PAINT frames (2023)
- Transformer for DNA-PAINT kinetic analysis (2024)
- Physics-informed network for DNA-PAINT reaction-diffusion modeling (2024)
- Foundation model for SMLM (PALM + STORM + DNA-PAINT) (2025)

#### Step 3: Update DNA-PAINT Solvers

After listing all DNA-PAINT solvers, update `algorithm_base/dna_paint/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All DNA-PAINT solvers use the data format: `y` (T, H, W) DNA-PAINT image sequence (transient binding events), `psf_model` (callable or array) PSF model, `pixel_size` (float) nm/pixel, `imager_concentration` (float) nM concentration of imager strands, `camera_params` (dict) gain, offset, readout noise. The `DNAPAINTOperator` handles the forward model (binding kinetics * emitter PSF + Poisson noise + camera noise) and adjoint operations producing localization lists with kinetic parameters.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for DNA-PAINT:**
- DNA origami nanoruler: localization precision <5 nm, inter-site distance accuracy <2 nm (Schnitzbauer et al., 2017)
- qPAINT counting accuracy: <10% error for known copy numbers on origami standards
- RESI precision: <1 nm on DNA origami calibration targets (Reinhardt et al., 2023)
- DNA-PAINT vs STORM resolution: DNA-PAINT ~5 nm localization precision vs STORM ~15 nm (due to unlimited photon budget)
- All reference values from published papers (Jungmann et al., 2014; Schnitzbauer et al., 2017)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'dna_paint' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/dna_paint/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/dna_paint/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/dna_paint/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for DNA-PAINT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/dna_paint/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/dna_paint/standard/`

---

### MINFLUX Nanoscopy (`minflux`) Modality Template

#### Step 1: Verify Standard Dataset

For MINFLUX nanoscopy, what dataset do you use to verify? Is this dataset used for MINFLUX popular algorithms? Please ensure the standard dataset in `datasets/benchmark/minflux/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MINFLUX standard dataset.

**Popular datasets to consider:**
- **Balzarotti et al. MINFLUX Datasets (Balzarotti et al., Science 2017)** — the original MINFLUX data demonstrating 1 nm localization precision on fluorescent molecules
- **Abberior MINFLUX Application Data (Gwosch et al., Nat Methods 2020)** — 3D MINFLUX imaging of nuclear pore complexes and other cellular structures with molecular-scale resolution
- **MINFLUX Tracking Datasets (Eilers et al., Nat Methods 2018)** — MINFLUX single-molecule tracking data with sub-millisecond temporal resolution and nanometer precision
- **MINFLUX Multi-Color Datasets (Schmidt et al., 2021)** — multi-color MINFLUX imaging exploiting spectral or lifetime separation

**Decision criteria:** Balzarotti et al. original data is the founding MINFLUX benchmark. Gwosch et al. for biological application validation. Use the dataset that appears in the largest number of MINFLUX papers (2017–2026).

#### Step 2: List All MINFLUX Algorithms

Please first ensure all the MINFLUX algorithms have been listed in `\Physics_World_Model\algorithm_base\minflux\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/minflux. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MINFLUX solvers, please update the MINFLUX solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2017–2020):_
- MINFLUX position estimation — calculating molecule position from photon counts at donut beam positions using centroid-like estimator (Balzarotti et al., Science 2017) — the foundational MINFLUX algorithm
- Maximum likelihood MINFLUX localization — MLE of molecule position from Poisson photon statistics at excitation positions (2017)
- Cramer-Rao bound for MINFLUX — theoretical precision limit analysis (Balzarotti et al., 2017)
- Iterative MINFLUX — multi-round localization with decreasing beam diameter for progressively refined precision (Balzarotti et al., 2017)
- 3D MINFLUX — extending to 3D localization using 3D donut excitation patterns (Gwosch et al., 2020)

_Optimization (2018–2023):_
- Adaptive MINFLUX beam pattern — optimizing excitation positions based on current position estimate (2019)
- MINFLUX sequence optimization — optimizing the number and placement of excitation steps (2020)
- Bayesian MINFLUX localization — posterior probability estimation with prior on molecule position (2021)
- MINFLUX tracking with Kalman filter — recursive position estimation for fast single-molecule tracking (Eilers et al., 2018)
- p-MINFLUX — pulsed interleaved MINFLUX for multi-color imaging (2020)
- MINFLUX with fluorescence lifetime — adding FLIM dimension to MINFLUX for molecular identification (2021)
- Optimal experimental design for MINFLUX — information-theoretic optimization of beam parameters (2022)
- Hidden Markov model for MINFLUX state transitions — analyzing conformational dynamics from MINFLUX traces (2022)
- Background estimation and subtraction for MINFLUX — accounting for out-of-focus fluorescence (2021)
- Drift correction for MINFLUX — nanometer-precision drift correction via fiducials or autocorrelation (2022)

_Deep Learning (2022–2026):_
- DL MINFLUX localization — neural network for position estimation from photon count patterns (2022)
- CNN for MINFLUX event classification — distinguishing signal from background events (2023)
- Deep reinforcement learning for adaptive MINFLUX — learning optimal beam positioning strategy (2023)
- Neural network for MINFLUX tracking — improved trajectory analysis (2024)
- GAN for MINFLUX data augmentation (2024)
- Self-supervised MINFLUX trace denoising (2024)
- Physics-informed network encoding MINFLUX excitation geometry (2025)
- Foundation model for nanoscopy (MINFLUX + STED + SMLM) (2025)

#### Step 3: Update MINFLUX Solvers

After listing all MINFLUX solvers, update `algorithm_base/minflux/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MINFLUX solvers use the data format: `y` (N_localizations, N_beam_positions) photon counts at each excitation beam position per localization event, `beam_positions` (N_beam_positions, 2 or 3) coordinates of donut excitation beam centers, `beam_diameter` (float) donut beam diameter in nm, `background_rate` (float) background photon rate. The `MINFLUXOperator` handles the forward model (donut excitation profile * molecule position + background, Poisson photon detection) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MINFLUX:**
- MINFLUX 2D localization precision: ~1 nm with ~100 photons (Balzarotti et al., 2017)
- MINFLUX 3D precision: ~2–3 nm isotropic (Gwosch et al., 2020)
- MINFLUX tracking temporal resolution: ~100 microsecond with ~1 nm precision (Eilers et al., 2018)
- MINFLUX vs PALM/STORM: MINFLUX achieves ~1 nm with 100 photons vs PALM/STORM ~20 nm with 1000 photons
- Nuclear pore complex resolution: resolving 8-fold symmetry at ~5 nm (Gwosch et al., 2020)
- All reference values from published papers (Balzarotti et al., 2017; Gwosch et al., 2020)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'minflux' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/minflux/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/minflux/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/minflux/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MINFLUX. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/minflux/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/minflux/standard/`

---

### Expansion Microscopy (`expansion`) Modality Template

#### Step 1: Verify Standard Dataset

For expansion microscopy (ExM), what dataset do you use to verify? Is this dataset used for ExM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/expansion/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ExM standard dataset.

**Popular datasets to consider:**
- **Boyden Lab ExM Datasets (Chen et al., Science 2015; Tillberg et al., Nat Biotechnol 2016)** — the original expansion microscopy data of brain tissue and cultured cells with pre- and post-expansion paired images
- **ExM Connectomics Data (Gao et al., Science 2019)** — expansion microscopy of Drosophila brain circuits with nanoscale resolution on conventional microscopes
- **Ultrastructure ExM (U-ExM) Datasets (Gambarotto et al., Nat Methods 2019)** — ultrastructure expansion microscopy revealing organelle morphology with near-EM resolution
- **Pan-ExM Datasets (M'Saad & Bhatt, 2022)** — pan-expansion microscopy combining protein retention with total protein labeling

**Decision criteria:** Boyden lab original ExM data is the founding benchmark. U-ExM datasets for ultrastructural resolution validation. Use the dataset that appears in the largest number of ExM papers (2015–2026).

#### Step 2: List All ExM Algorithms

Please first ensure all the ExM algorithms have been listed in `\Physics_World_Model\algorithm_base\expansion\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/expansion. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ExM solvers, please update the ExM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2015–2018):_
- Expansion factor calibration — measuring actual expansion ratio from pre/post-expansion fiducial comparisons (Chen et al., 2015)
- Distortion mapping — quantifying non-uniform expansion by comparing pre- and post-expansion landmark positions (2016)
- Deconvolution of expanded samples — standard RL deconvolution applied to expanded specimen images (2016)
- Resolution calibration — measuring effective resolution (physical resolution / expansion factor) using nanoscale standards (2017)
- Chromatic aberration correction for multi-color ExM — correcting for wavelength-dependent aberrations in expanded gels (2017)

_Optimization (2016–2022):_
- Non-rigid registration for expansion distortion correction — B-spline or thin-plate spline registration between pre- and post-expansion images (2017)
- Iterative expansion — repeated expansion for higher expansion factors (Chang et al., Nat Methods 2017)
- ExM combined with SIM — structured illumination of expanded samples for further resolution gain (Gao et al., 2018)
- ExM combined with STED — STED imaging of expanded samples for sub-10 nm effective resolution (2019)
- TV-regularized deconvolution for expanded samples — accounting for altered PSF in gel medium (2018)
- Gel shrinkage compensation — correcting for partial re-contraction during imaging (2019)
- Magnification analysis through correlation — quantitative expansion factor mapping (Xu et al., 2019)
- Multi-round ExM registration — aligning multiple rounds of expansion for multiplexed imaging (2021)
- ADMM-based ExM image reconstruction with expansion model (2020)
- ExM-SMLM combination — PALM/STORM on expanded samples for molecular resolution (2019)

_Deep Learning (2020–2026):_
- DL distortion correction for ExM — CNN learning to predict and correct expansion non-uniformities (2021)
- U-Net for ExM segmentation — semantic segmentation of expanded tissue structures (2020)
- DL denoising for ExM — improving SNR of expanded (diluted fluorophore) images (2021)
- Virtual ExM — predicting expansion microscopy resolution from diffraction-limited input via deep learning (2022)
- GAN for ExM resolution enhancement — adversarial super-resolution of expanded images (2023)
- 3D ExM reconstruction with deep learning (2023)
- Cross-modality ExM-to-EM prediction — learning to predict EM-like images from ExM data (2024)
- Self-supervised ExM denoising (2023)
- Transformer for ExM multi-round image registration (2024)
- Foundation model for expansion microscopy analysis (2025)

#### Step 3: Update ExM Solvers

After listing all ExM solvers, update `algorithm_base/expansion/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ExM solvers use the data format: `y` (D, H, W) or (H, W) expanded sample image, `psf` (D_psf, H_psf, W_psf) PSF in gel medium (adjusted for refractive index of swollen hydrogel), `expansion_factor` (float) measured expansion ratio (typically 4–20x), `pre_expansion` (H_pre, W_pre) optional pre-expansion image for distortion correction. The `ExpansionOperator` handles the forward model (expansion geometry * PSF convolution in gel + fluorophore dilution + Poisson noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for ExM:**
- Expansion factor uniformity: <1% RMS distortion error on calibration grids (Chen et al., 2015)
- Effective resolution: ~70 nm for 4x ExM on confocal, ~25 nm for 4x ExM + SIM, ~15 nm for 10x iterative ExM
- ExM deconvolution: raw ~25 dB, RL ~30 dB, DL denoised ~33 dB PSNR
- Distortion correction: registration error <20 nm (effective) after non-rigid correction
- All reference values from published papers (Chen et al., 2015; Gambarotto et al., 2019)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'expansion' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/expansion/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/expansion/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/expansion/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ExM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/expansion/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/expansion/standard/`

---

### Low-Dose Widefield Microscopy (`widefield_lowdose`) Modality Template

#### Step 1: Verify Standard Dataset

For low-dose widefield microscopy, what dataset do you use to verify? Is this dataset used for low-dose widefield popular algorithms? Please ensure the standard dataset in `datasets/benchmark/widefield_lowdose/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original low-dose widefield standard dataset.

**Popular datasets to consider:**
- **W2S Dataset (Widefield-to-SIM, Qiao et al., 2021)** — paired low-SNR widefield images and high-quality SIM reconstructions for denoising and super-resolution benchmarks
- **BioImage Model Zoo Widefield Denoising Data (von Chamier et al., 2021)** — curated paired low-dose/high-dose widefield fluorescence images for denoising benchmarks
- **Lehtinen et al. Noise2Noise Microscopy Data (Lehtinen et al., ICML 2018)** — paired noisy widefield images for self-supervised denoising validation
- **FMD (Fluorescence Microscopy Denoising) Dataset (Zhang et al., 2019)** — widefield fluorescence images at multiple exposure levels with high-exposure ground truth

**Decision criteria:** FMD dataset is the most widely cited for microscopy denoising benchmarks. W2S for combined denoising + super-resolution. Use the dataset that appears in the largest number of widefield denoising papers (2018–2026).

#### Step 2: List All Low-Dose Widefield Algorithms

Please first ensure all the low-dose widefield algorithms have been listed in `\Physics_World_Model\algorithm_base\widefield_lowdose\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/widefield_lowdose. Besides, you need to search all algorithms from 1950 to 2026. After listing all the low-dose widefield solvers, please update the low-dose widefield solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1970s–2005):_
- Wiener filter denoising — frequency-domain denoising balancing noise reduction and detail preservation (Wiener, 1949; applied to microscopy)
- Gaussian smoothing — spatial-domain low-pass filtering for noise reduction (standard)
- Median filtering — non-linear edge-preserving denoising (1970s)
- Background subtraction — rolling ball and morphological top-hat for uneven illumination (Sternberg, 1983)
- Flat-field correction — dividing by illumination pattern to correct for non-uniform excitation (standard)

_Optimization (2005–2018):_
- Non-local means (NLM) denoising — patch-based self-similarity denoising (Buades et al., CVPR 2005)
- BM3D — block-matching and 3D collaborative filtering (Dabov et al., TIP 2007) — state-of-the-art classical denoiser
- TV denoising — total variation minimization for piecewise-smooth image recovery (Rudin et al., Physica D 1992)
- Poisson-Gaussian denoising — noise model specific to fluorescence microscopy (Luisier et al., TIP 2011)
- VST + BM3D — variance-stabilizing transform followed by Gaussian denoising (Makitalo & Foi, 2011)
- PURE-LET — Poisson unbiased risk estimate with linear expansion of thresholds (Luisier et al., 2010)
- Dictionary learning denoising — K-SVD and sparse representation for microscopy denoising (2012)
- Wiener deconvolution for widefield — deconvolution with widefield PSF to remove out-of-focus blur (2005)
- Richardson-Lucy deconvolution for widefield z-stacks (Agard, 1984)
- Structured illumination denoising — exploiting known illumination pattern for enhanced denoising (2015)

_Deep Learning (2017–2026):_
- CARE — content-aware image restoration (Weigert et al., Nat Methods 2018) — the seminal DL microscopy restoration paper
- Noise2Noise — training without clean targets using paired noisy images (Lehtinen et al., ICML 2018)
- Noise2Void — self-supervised denoising using blind-spot network (Krull et al., CVPR 2019)
- Noise2Self — self-supervised denoising via J-invariance (Batson & Royer, ICML 2019)
- DnCNN adapted for fluorescence microscopy (Zhang et al., TIP 2017; adapted 2019)
- RCAN for microscopy super-resolution — residual channel attention network (Chen et al., 2021)
- Noise2Fast — fast self-supervised denoising (Lequyer et al., 2022)
- HDN — hierarchical DivNoising with variational inference (Prakash et al., 2021)
- Structured denoising diffusion model for microscopy (2023)
- Self-supervised blind-spot networks with structured noise model (2023)
- Foundation model for fluorescence microscopy denoising (2025)

#### Step 3: Update Low-Dose Widefield Solvers

After listing all low-dose widefield solvers, update `algorithm_base/widefield_lowdose/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All low-dose widefield solvers use the data format: `y` (H, W) or (D, H, W) low-dose widefield fluorescence image or z-stack, `psf` (D_psf, H_psf, W_psf) widefield PSF (extended depth of field), `noise_params` (dict) estimated Poisson-Gaussian noise parameters (gain, offset, readout variance). The `WidefieldLowDoseOperator` handles the forward model (PSF convolution + Poisson shot noise + Gaussian readout noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for low-dose widefield:**
- FMD dataset denoising: raw ~20 dB, BM3D ~28 dB, VST+BM3D ~29 dB, CARE ~33 dB, Noise2Void ~31 dB
- W2S denoising + SR: widefield ~22 dB, CARE ~30 dB, RCAN ~32 dB
- Widefield deconvolution: raw ~18 dB, Wiener ~24 dB, RL ~27 dB
- Self-supervised vs supervised gap: Noise2Void typically 1–2 dB below CARE
- All reference values from published papers (Weigert et al., 2018; Krull et al., 2019)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'widefield_lowdose' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/widefield_lowdose/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for low-dose widefield. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/widefield_lowdose/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield_lowdose/standard/`

---

### Fluorescence Lifetime Imaging (`flim`) Modality Template

#### Step 1: Verify Standard Dataset

For fluorescence lifetime imaging microscopy (FLIM), what dataset do you use to verify? Is this dataset used for FLIM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/flim/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original FLIM standard dataset.

**Popular datasets to consider:**
- **FLIM-FRET Standard Datasets (Wallrabe & Bhatt, 2005; Elangovan et al., 2003)** — FLIM-FRET images of donor-acceptor pairs with known FRET efficiencies for lifetime analysis validation
- **Becker & Hickl FLIM Application Data (Various, 2005–2023)** — TCSPC-FLIM images from various biological applications with time-resolved photon histograms
- **PicoQuant FLIM Datasets (Various, 2010–2023)** — time-tagged FLIM data with reference lifetime values for calibration
- **Metabolic FLIM Datasets (NAD(P)H / FAD Lifetime, Skala et al., 2007; Walsh et al., 2021)** — autofluorescence FLIM of cellular metabolism with free/bound NAD(P)H lifetime ground truth

**Decision criteria:** Metabolic FLIM (NAD(P)H) datasets are the most widely benchmarked for biomedical FLIM analysis. FLIM-FRET for interaction studies. Use the dataset that appears in the largest number of FLIM analysis papers (2005–2026).

#### Step 2: List All FLIM Algorithms

Please first ensure all the FLIM algorithms have been listed in `\Physics_World_Model\algorithm_base\flim\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/flim. Besides, you need to search all algorithms from 1950 to 2026. After listing all the FLIM solvers, please update the FLIM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2008):_
- Single-exponential tail fitting — least-squares fitting of fluorescence decay tail to extract lifetime (standard, 1990s)
- Multi-exponential decay fitting — iterative reconvolution fitting with instrument response function (IRF) for bi/tri-exponential decays (Lakowicz, 1999)
- Phasor approach to FLIM — model-free analysis mapping each pixel to a point on the phasor plot (Digman et al., Biophys J 2008) — widely used for rapid FLIM analysis
- Rapid lifetime determination (RLD) — ratio-based fast lifetime estimation from two time gates (2003)
- Frequency-domain FLIM analysis — extracting lifetime from phase shift and modulation depth (Lakowicz, 1983)

_Optimization (2008–2020):_
- SPCImage analysis — commercial TCSPC FLIM fitting software with multi-exponential models and IRF deconvolution (Becker & Hickl)
- FLIMfit — open-source global analysis FLIM fitting tool (Warren et al., PLoS ONE 2013) — widely used open-source FLIM analysis
- Global analysis — simultaneous fitting of all pixels with shared lifetime components (Verveer et al., 2000)
- Bayesian FLIM — Bayesian inference for lifetime estimation with uncertainty quantification (Rowley et al., 2016)
- Compressed sensing FLIM — reconstructing FLIM from reduced time gates (2015)
- Maximum entropy method for FLIM — recovering lifetime distributions without assuming discrete components (2010)
- Laguerre deconvolution for FLIM — expanding decay in Laguerre basis for rapid analysis (Jo et al., 2004)
- TV-regularized spatial FLIM denoising — exploiting spatial smoothness of lifetime maps (2014)
- Stretched exponential fitting for heterogeneous FLIM decays (2012)
- Phasor-based segmentation of FLIM images (2015)

_Deep Learning (2019–2026):_
- DL FLIM fitting — CNN predicting lifetime parameters directly from photon histograms (Smith et al., 2019; Wu et al., Opt Express 2020)
- Net-FLICS — neural network for fluorescence lifetime imaging in compressed sensing (Yao et al., Biomed Opt Express 2019)
- 3D CNN for volumetric FLIM analysis (2021)
- Physics-informed neural network for FLIM — encoding exponential decay model in loss function (2022)
- DL phasor FLIM — learning phasor representations for rapid analysis (2022)
- Noise2Lifetime — self-supervised FLIM denoising exploiting photon statistics (2023)
- GAN for low-photon FLIM enhancement — generating high-count predictions from sparse decays (2023)
- Transformer for FLIM spectral analysis (2024)
- Diffusion model for FLIM decay reconstruction (2024)
- Foundation model for time-resolved fluorescence imaging (2025)

#### Step 3: Update FLIM Solvers

After listing all FLIM solvers, update `algorithm_base/flim/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All FLIM solvers use the data format: `y` (H, W, N_timebins) TCSPC FLIM data (photon count histogram per pixel), `irf` (N_timebins,) instrument response function, `time_axis` (N_timebins,) time bin centers in nanoseconds, `laser_period` (float) repetition period. The `FLIMOperator` handles the forward model (multi-exponential decay convolved with IRF + Poisson photon counting noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for FLIM:**
- NAD(P)H FLIM: free NAD(P)H tau ~0.4 ns, bound NAD(P)H tau ~2.5 ns; fitting error <0.1 ns for >100 photons/pixel
- FLIM-FRET efficiency: FRET efficiency error <5% for well-characterized donor-acceptor pairs
- Low-photon FLIM: at 50 photons/pixel, phasor CV ~20%, DL fitting CV ~10%
- DL FLIM speed: >100x faster than iterative reconvolution fitting with comparable accuracy
- All reference values from published papers (Digman et al., 2008; Warren et al., 2013)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'flim' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/flim/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/flim/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/flim/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for FLIM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/flim/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/flim/standard/`

---

### Second Harmonic Generation Microscopy (`shg`) Modality Template

#### Step 1: Verify Standard Dataset

For second harmonic generation (SHG) microscopy, what dataset do you use to verify? Is this dataset used for SHG popular algorithms? Please ensure the standard dataset in `datasets/benchmark/shg/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SHG standard dataset.

**Popular datasets to consider:**
- **Collagen SHG Datasets (Chen et al., 2012; Campagnola et al., various)** — SHG images of collagen fibers in tissue (skin, tendon, tumor stroma) with fiber orientation ground truth; the canonical SHG benchmark
- **Ovarian Cancer SHG Datasets (Nadiarnykh et al., BMC Cancer 2010)** — SHG images comparing normal and cancerous ovarian tissue collagen organization
- **Myosin SHG Datasets (Plotnikov et al., Biophys J 2006)** — SHG imaging of sarcomeric myosin in muscle tissue
- **CT-FIRE / CurveAlign Benchmark Data (Bredfeldt et al., J Biomed Opt 2014)** — SHG collagen images with manual fiber annotations for orientation and alignment analysis

**Decision criteria:** Collagen SHG fiber analysis datasets (CT-FIRE benchmarks) are the most widely used. Ovarian cancer SHG for diagnostic applications. Use the dataset that appears in the largest number of SHG analysis papers (2006–2026).

#### Step 2: List All SHG Algorithms

Please first ensure all the SHG algorithms have been listed in `\Physics_World_Model\algorithm_base\shg\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/shg. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SHG solvers, please update the SHG solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2000–2010):_
- SHG signal theory — second-order nonlinear susceptibility chi(2) imaging of non-centrosymmetric structures (Campagnola et al., Biophys J 2002) — foundational SHG microscopy for biology
- Forward/backward SHG ratio — F/B ratio analysis for collagen fibril diameter estimation (Williams et al., Biophys J 2005)
- Polarization-resolved SHG (P-SHG) — rotating input polarization to extract molecular orientation and chi(2) tensor components (Stoller et al., J Biomed Opt 2002)
- FFT-based orientation analysis — Fourier transform of SHG images for global fiber directionality (2005)
- Intensity thresholding and binarization for SHG fiber detection (standard)

_Optimization (2010–2020):_
- CT-FIRE — curvelet transform and fiber extraction algorithm for SHG collagen analysis (Bredfeldt et al., J Biomed Opt 2014) — the standard SHG collagen analysis tool
- CurveAlign — automated collagen alignment quantification relative to tumor boundaries (Bredfeldt et al., 2014)
- OrientationJ — structure tensor-based orientation analysis (Rezakhaniha et al., 2012; Fiji plugin)
- Wavelet-based SHG fiber analysis — multi-scale fiber detection and orientation (2012)
- Radon transform for SHG fiber orientation — projection-based angle estimation (2011)
- Grey-level co-occurrence matrix (GLCM) texture analysis for SHG collagen characterization (Cicchi et al., 2010)
- Hough transform for SHG fiber detection (2013)
- TV denoising for SHG images (2014)
- P-SHG fitting with Jones calculus — extracting chi(2) tensor from polarization-dependent SHG signals (Gusachenko et al., 2012)
- Machine learning classification of SHG texture features for tissue diagnosis (2017)

_Deep Learning (2018–2026):_
- CNN for SHG collagen fiber segmentation (2019)
- DL SHG-to-H&E virtual staining — predicting histology stain from SHG images (Rivenson et al., Nat Biomed Eng 2019)
- U-Net for SHG fiber detection and orientation mapping (2020)
- GAN for SHG image enhancement — improving SHG SNR and resolution (2021)
- DL collagen organization scoring from SHG for cancer prognosis (2021)
- ResNet for SHG tissue classification (normal vs tumor) (2022)
- Physics-informed network for P-SHG tensor recovery (2023)
- Self-supervised SHG denoising (2023)
- Transformer for SHG fiber tracking across 3D volumes (2024)
- Foundation model for nonlinear optical microscopy (SHG + THG + CARS) (2025)

#### Step 3: Update SHG Solvers

After listing all SHG solvers, update `algorithm_base/shg/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SHG solvers use the data format: `y` (H, W) SHG intensity image or (N_polarizations, H, W) polarization-resolved SHG data, `polarization_angles` (N_polarizations,) input polarization angles in radians, `excitation_wavelength` (float) fundamental wavelength (SHG at half-wavelength). The `SHGOperator` handles the forward model (chi(2) tensor * E-field squared * phase matching + Poisson noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SHG:**
- CT-FIRE collagen analysis: fiber detection recall >0.85, angle accuracy <5 degrees on annotated datasets
- CurveAlign alignment score: correlation >0.90 with expert manual scoring
- P-SHG orientation accuracy: <3 degree error on known collagen orientation
- SHG tissue classification: CNN accuracy >0.90 for normal vs tumor collagen (2022)
- SHG virtual staining: SSIM >0.70 compared to real H&E (Rivenson et al., 2019)
- All reference values from published papers (Bredfeldt et al., 2014; Campagnola et al., 2002)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'shg' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/shg/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/shg/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/shg/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SHG. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/shg/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/shg/standard/`

---

### Bioluminescence Tomography (`bioluminescence_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For bioluminescence tomography (BLT), what dataset do you use to verify? Is this dataset used for BLT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/bioluminescence_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original BLT standard dataset.

**Popular datasets to consider:**
- **IVIS Bioluminescence Imaging Datasets (Caliper/PerkinElmer, various)** — standard bioluminescence imaging data from IVIS systems of tumor-bearing mice with known source locations
- **Cong et al. BLT Phantom Data (Cong et al., Opt Express 2005; 2010)** — BLT phantom datasets with known embedded bioluminescent sources and multi-view surface measurements
- **MOBY Mouse Phantom BLT Data (Segars et al., 2004)** — digital mouse phantom with simulated bioluminescence data for reconstruction algorithm validation
- **Multi-Spectral BLT Datasets (Dehghani et al., Opt Lett 2006)** — spectrally resolved bioluminescence surface measurements enabling depth-resolved source reconstruction

**Decision criteria:** Cong et al. phantom datasets are the most widely used for BLT algorithm validation. MOBY phantom for standardized simulation benchmarks. Use the dataset that appears in the largest number of BLT reconstruction papers (2005–2026).

#### Step 2: List All BLT Algorithms

Please first ensure all the BLT algorithms have been listed in `\Physics_World_Model\algorithm_base\bioluminescence_tomo\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/bioluminescence_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the BLT solvers, please update the BLT solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (2003–2010):_
- Diffusion equation forward model — modeling photon propagation through tissue using the diffusion approximation to the radiative transfer equation (Wang et al., 2003) — foundational BLT forward model
- Born approximation BLT — linearized forward model for weakly scattering media (2004)
- Back-projection BLT — simple back-projection of surface radiance along lines of sight (2003)
- Multi-view surface radiance measurement — acquiring bioluminescence from multiple camera angles (2005)
- Spectral unmixing for multi-reporter BLT — separating multiple bioluminescent reporters by emission spectrum (2007)

_Optimization (2005–2020):_
- Tikhonov-regularized BLT — L2-regularized reconstruction of 3D bioluminescent source distribution (Cong et al., Opt Express 2005)
- L1-regularized BLT — sparse source reconstruction assuming localized bioluminescent sources (Han et al., 2006)
- Multi-spectral BLT — exploiting wavelength-dependent tissue absorption for depth resolution (Dehghani et al., 2006)
- Finite element method (FEM) BLT — FEM-based solution of diffusion equation for heterogeneous tissue (Cong et al., 2006)
- Adaptive FEM for BLT — mesh refinement guided by source estimate (2008)
- Bayesian BLT — posterior source estimation with anatomical priors from CT/MRI (2010)
- Total variation regularized BLT — TV penalty for piecewise constant source distributions (2012)
- Permissible source region BLT — constraining source to anatomically plausible regions using CT/MRI (Lv et al., 2006)
- ADMM-based BLT — splitting methods for constrained source reconstruction (2015)
- Sparse Bayesian learning for BLT (Gao et al., 2016)
- Radiative transfer equation BLT — full RTE forward model for improved accuracy in small animals (2014)
- Phase-space Monte Carlo BLT — Monte Carlo forward model for heterogeneous tissue (2016)

_Deep Learning (2019–2026):_
- DL-BLT — CNN-based 3D source reconstruction from surface radiance maps (2019)
- U-Net for BLT source localization (2020)
- Physics-informed neural network for BLT — encoding diffusion equation in loss function (2021)
- GAN for BLT reconstruction with anatomical prior (2022)
- 3D ResNet for BLT source distribution prediction (2022)
- DL multi-spectral BLT — learning spectral-depth relationship (2023)
- Self-supervised BLT from unpaired data (2023)
- Transformer for BLT multi-view fusion (2024)
- Diffusion model for BLT with uncertainty quantification (2024)
- Foundation model for optical/bioluminescence tomography (2025)

#### Step 3: Update BLT Solvers

After listing all BLT solvers, update `algorithm_base/bioluminescence_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All BLT solvers use the data format: `y` (N_views, H_surface, W_surface) or (N_wavelengths, N_views, H_surface, W_surface) surface radiance measurements, `tissue_mesh` (N_nodes, 3) FEM mesh of tissue volume from CT/MRI, `optical_properties` (N_elements, 2) absorption and scattering coefficients per tissue region, `surface_nodes` (N_surface,) indices of surface mesh nodes. The `BLTOperator` handles the forward model (diffusion equation photon propagation from internal source to surface) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for BLT:**
- BLT source localization: Tikhonov ~3 mm error, L1-sparse ~2 mm, multi-spectral ~1.5 mm, DL ~1 mm
- BLT source intensity: reconstruction error <20% for Tikhonov, <10% for multi-spectral
- BLT PSNR on phantom: Tikhonov ~20 dB, TV ~24 dB, DL ~28 dB
- Multi-spectral depth resolution: <1 mm depth discrimination with 4+ spectral bands
- All reference values from published papers (Cong et al., 2005; Dehghani et al., 2006)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'bioluminescence_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/bioluminescence_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/bioluminescence_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/bioluminescence_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for BLT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/bioluminescence_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/bioluminescence_tomo/standard/`

---

### Polarization Microscopy (`polarization`) Modality Template

#### Step 1: Verify Standard Dataset

For polarization microscopy, what dataset do you use to verify? Is this dataset used for polarization microscopy popular algorithms? Please ensure the standard dataset in `datasets/benchmark/polarization/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original polarization microscopy standard dataset.

**Popular datasets to consider:**
- **LC-PolScope Datasets (Oldenbourg, 2013; Mehta et al., J Opt 2013)** — liquid crystal polarization microscopy images of biological specimens (spindles, cytoskeleton, crystals) with quantitative retardance and orientation ground truth
- **Mueller Matrix Microscopy Datasets (He et al., J Biomed Opt 2015)** — full Mueller matrix polarimetric images of tissue sections with derived polarization parameters
- **Polarization-Sensitive OCT / Microscopy Datasets (Various, 2010–2023)** — polarization-resolved images of birefringent tissues (muscle, nerve, collagen)
- **Polychromatic Polarization Microscopy Data (Shribak & Inoue, 2006)** — quantitative birefringence images of known optical samples (mica, quartz wedge) for calibration

**Decision criteria:** LC-PolScope datasets are the most widely used for quantitative polarization microscopy. Mueller matrix data for comprehensive polarimetric analysis. Use the dataset that appears in the largest number of polarization microscopy papers (2006–2026).

#### Step 2: List All Polarization Microscopy Algorithms

Please first ensure all the polarization microscopy algorithms have been listed in `\Physics_World_Model\algorithm_base\polarization\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/polarization. Besides, you need to search all algorithms from 1950 to 2026. After listing all the polarization microscopy solvers, please update the polarization microscopy solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1950s–2005):_
- Crossed polarizer imaging — classical polarization microscopy between crossed polarizers for birefringence detection (standard, 1950s)
- Senarmont compensator method — quantitative retardance measurement using quarter-wave plate rotation (Senarmont, 1840; applied in microscopy)
- LC-PolScope — liquid crystal variable retarder-based quantitative polarization microscopy (Oldenbourg & Mei, J Microsc 1995) — the foundational quantitative polarization microscope
- Jones calculus for polarization microscopy — modeling polarization transformations through optical elements and sample (Jones, 1941)
- Mueller matrix polarimetry — full 4x4 Mueller matrix measurement for complete polarization characterization (Chipman, 1995)

_Optimization (2005–2018):_
- Four-frame algorithm for LC-PolScope — extracting retardance and slow-axis orientation from 4 or 5 intensity measurements (Shribak & Oldenbourg, Appl Opt 2003)
- Mueller matrix decomposition — Lu-Chipman polar decomposition into diattenuation, retardance, and depolarization (Lu & Chipman, JOSA A 1996)
- Stokes parameter imaging — computing Stokes vectors from intensity measurements at multiple analyzer angles (2006)
- TV-regularized polarization image denoising — exploiting spatial smoothness of retardance maps (2012)
- Birefringence tomography — reconstructing 3D birefringence distribution from through-focus polarization data (Mehta et al., Opt Lett 2013)
- Mueller matrix microscopy with structured illumination — combining polarimetry with SIM for enhanced resolution (2016)
- Depolarization index analysis — quantifying tissue scattering from Mueller matrix depolarization (2014)
- Differential Mueller matrix analysis — extracting differential polarization properties of thin layers (2010)
- ADMM-based birefringence reconstruction with smoothness constraints (2017)
- Phase-shifting interferometry for quantitative retardance mapping (2008)

_Deep Learning (2019–2026):_
- CNN for Mueller matrix tissue classification — classifying tissue types from polarimetric images (2019)
- DL retardance estimation — predicting quantitative retardance from fewer polarization measurements (2020)
- U-Net for polarization image segmentation — segmenting birefringent structures (collagen, nerve fibers) (2021)
- GAN for polarization image denoising and enhancement (2022)
- Physics-informed network for Mueller matrix decomposition (2022)
- DL birefringence tomography — 3D birefringence from reduced measurements (2023)
- Cross-modality prediction: polarization to H&E virtual staining (2023)
- Transformer for Mueller matrix analysis (2024)
- Self-supervised polarization microscopy denoising (2024)
- Foundation model for polarimetric imaging (microscopy + remote sensing) (2025)

#### Step 3: Update Polarization Microscopy Solvers

After listing all polarization microscopy solvers, update `algorithm_base/polarization/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All polarization microscopy solvers use the data format: `y` (N_measurements, H, W) intensity images at different polarization states (e.g., 4 or 5 LC-PolScope frames, or 16 Mueller matrix elements), `polarizer_states` (N_measurements, 4) input and output Stokes parameters for each measurement, `wavelength` (float) illumination wavelength. The `PolarizationOperator` handles the forward model (Mueller matrix * input Stokes vector -> output intensity + Poisson noise) and adjoint operations recovering retardance, orientation, diattenuation, and depolarization maps.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for polarization microscopy:**
- LC-PolScope retardance accuracy: <0.5 nm retardance error on calibrated mica samples (Oldenbourg, 2013)
- Orientation accuracy: <2 degree error for known crystal axes
- Mueller matrix decomposition: <5% error in diattenuation, retardance, depolarization on calibration standards
- Tissue classification from Mueller matrix: CNN accuracy >0.85 for collagen/nerve/tumor classification
- Polarization denoising: raw ~18 dB, TV ~26 dB, DL ~30 dB PSNR on retardance maps
- All reference values from published papers (Oldenbourg, 2013; Lu & Chipman, 1996)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'polarization' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/polarization/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/polarization/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/polarization/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for polarization microscopy. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/polarization/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/polarization/standard/`
