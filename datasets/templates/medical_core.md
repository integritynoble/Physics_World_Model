
---

## Medical Imaging Core — Modality Templates

---

### MRI (`mri`) Modality Template

#### Step 1: Verify Standard Dataset

For MRI, what dataset do you use to verify? Is this dataset used for MRI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MRI standard dataset.

**Popular datasets to consider:**
- **fastMRI (NYU Langone, Zbontar et al., 2018)** — the most widely used MRI reconstruction benchmark; multi-coil brain and knee k-space data with ground-truth fully-sampled images; 4x and 8x acceleration; used by virtually all deep learning MRI papers since 2019
- **Calgary-Campinas Multi-Coil Dataset (Souza et al., 2018)** — 12-channel brain MRI raw k-space; used for parallel imaging and compressed sensing benchmarks
- **IXI Dataset (Imperial College London)** — 600 healthy brain MRI scans (T1, T2, PD-weighted); widely used for training and evaluation
- **BrainWeb Phantom (Collins et al., 1998)** — synthetic MRI brain phantom with known ground truth; the canonical numerical benchmark for MRI reconstruction since the 1990s
- **HCP (Human Connectome Project)** — high-resolution multi-modal brain MRI; used for advanced reconstruction evaluation
- **Stanford 2D FSE Dataset (Epperson et al., 2013)** — knee MRI raw k-space data; used for compressed sensing validation
- **SKM-TEA (Stanford Knee MRI Multi-Task Evaluation, Desai et al., 2022)** — multi-task knee MRI benchmark with reconstruction and segmentation ground truth

**Decision criteria:** fastMRI is the undisputed gold standard for MRI reconstruction benchmarking (2019–2026); Calgary-Campinas for multi-coil parallel imaging. Use the dataset that appears in the largest number of MRI reconstruction papers.

#### Step 2: List All MRI Algorithms

Please first ensure all the MRI algorithms have been listed in `\Physics_World_Model\algorithm_base\mri\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MRI solvers, please update the MRI solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1970s–2009):_
- Zero-filled IFFT — baseline reconstruction from undersampled k-space (trivial, always reported)
- SENSE — SENSitivity Encoding for parallel MRI (Pruessmann et al., MRM 1999)
- GRAPPA — GeneRalized Autocalibrating Partially Parallel Acquisitions (Griswold et al., MRM 2002)
- Conjugate Gradient SENSE — iterative SENSE reconstruction (Pruessmann et al., MRM 2001)
- SPIRiT — iterative Self-consistent Parallel Imaging Reconstruction (Lustig & Pauly, MRM 2010; roots 2007)
- NUFFT-based reconstruction for non-Cartesian trajectories (Fessler & Sutton, IEEE TSP 2003)
- Sum-of-Squares coil combination (Roemer et al., MRM 1990)
- Homodyne detection for partial Fourier (Noll et al., IEEE TMI 1991)
- POCS — Projection Onto Convex Sets for partial Fourier (Haacke et al., 1991)

_Compressed Sensing & Optimization (2007–2016):_
- Sparse MRI / CS-MRI — Compressed Sensing MRI with wavelet + TV sparsity (Lustig et al., MRM 2007) — the foundational CS-MRI paper
- L1-ESPIRiT — joint parallel imaging + compressed sensing (Uecker et al., MRM 2014)
- Low-Rank + Sparse (L+S) decomposition for dynamic MRI (Otazo et al., MRM 2015)
- k-t FOCUSS — k-t space compressed sensing for dynamic MRI (Jung et al., MRM 2009)
- BART toolbox methods — Berkeley Advanced Reconstruction Toolbox (Uecker et al., 2015)
- Total Variation CS-MRI (Block et al., MRM 2007)
- Dictionary Learning MRI (Ravishankar & Bresler, TMI 2011)
- Bayesian CS-MRI (He & Carin, 2009)
- Joint multi-coil compressed sensing (Otazo et al., 2010)
- ADMM-based CS-MRI (Ramani & Fessler, 2011)
- ALOHA — Annihilating filter-based Low-Rank Hankel matrix (Jin et al., TIP 2016)

_Deep Learning (2017–2026):_
- AUTOMAP — Automated Transform by Manifold Approximation (Zhu et al., Nature 2018)
- Deep ADMM-Net for CS-MRI (Sun et al., NIPS 2016)
- D5C5 — Deep Cascade of CNNs for MRI Reconstruction (Schlemper et al., TMI 2018)
- KIKI-Net — cross-domain CNN (Eo et al., MRM 2018)
- MoDL — Model-Based Deep Learning for MRI (Aggarwal et al., TMI 2019)
- E2E-VarNet — End-to-End Variational Network (Sriram et al., fastMRI 2020) — fastMRI leaderboard top performer
- fastMRI U-Net baseline (Zbontar et al., 2018)
- HUMUS-Net — Hybrid Unrolled Multi-Scale Network (Fabian et al., 2022)
- PromptMR — Prompt-based learning for MRI reconstruction (Li et al., 2023)
- Score-based diffusion MRI reconstruction (Chung et al., MedIA 2022; Jalal et al., NeurIPS 2021)
- Implicit neural representation MRI — NeRP (Shen et al., 2022)
- k-space Transformer for MRI (Huang et al., 2022)
- SwinMR — Swin Transformer for MRI reconstruction (Huang et al., 2022)
- ReconFormer — Transformer for MRI reconstruction (Guo et al., 2023)
- vSHARP — variable Splitting Half-quadratic ADMM with learned Regularization and Priors (George et al., 2023) — fastMRI challenge winner
- Foundation model for multi-contrast MRI (2025)

#### Step 3: Update MRI Solvers

After listing all MRI solvers, update `algorithm_base/mri/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MRI solvers use the data format: `y` (num_coils, H, W) multi-coil k-space data, `sensitivity_maps` (num_coils, H, W) coil sensitivities, `mask` (H, W) or (1, W) undersampling mask. The `MRIOperator` handles forward `y = mask * F * S * x` and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MRI:**
- fastMRI multi-coil brain 4x: Zero-filled ~28.5 dB, GRAPPA ~33.0 dB, U-Net ~36.0 dB, E2E-VarNet ~38.5 dB, vSHARP ~39.2 dB
- fastMRI multi-coil brain 8x: Zero-filled ~25.0 dB, E2E-VarNet ~35.0 dB
- fastMRI multi-coil knee 4x: E2E-VarNet ~38.0 dB PSNR / 0.960 SSIM
- Calgary-Campinas: CS-MRI (Lustig) ~32.0 dB, MoDL ~36.5 dB
- All reference values from fastMRI leaderboard and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mri' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mri/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MRI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`

---

### CT (`ct`) Modality Template

#### Step 1: Verify Standard Dataset

For CT, what dataset do you use to verify? Is this dataset used for CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CT standard dataset.

**Popular datasets to consider:**
- **AAPM Mayo Clinic Low-Dose CT Grand Challenge (McCollough et al., 2016)** — the most widely used CT reconstruction benchmark; quarter-dose and full-dose paired data, 512x512, 10 patients; used by RED-CNN, FBPConvNet, LEARN, WGAN-VGG, CPCE, and virtually all low-dose CT papers
- **LoDoPaB-CT (Leuschner et al., Scientific Data 2021)** — large-scale low-dose parallel-beam CT benchmark from LIDC/IDRI; 362x362, fan-beam sparse-view (60 views); emerging standard for sparse-view CT
- **DeepLesion (Yan et al., JMRI 2018)** — 32K+ CT slices from NIH with lesion annotations; popular for detection but less for reconstruction
- **LIDC-IDRI (Armato et al., 2011)** — lung CT screening dataset with nodule annotations; raw projection data available for reconstruction research
- **TCIA Collections** — multiple clinical CT datasets for various organs and pathologies
- **Walnut CT Dataset (Der Sarkissian et al., 2019)** — high-quality industrial micro-CT benchmark with 2D fan-beam and 3D cone-beam projections

**Decision criteria:** The AAPM Mayo dataset is the gold standard for low-dose CT denoising/reconstruction. LoDoPaB-CT is the gold standard for sparse-view CT. Use the dataset that appears in the largest number of CT reconstruction papers (2017–2026).

#### Step 2: List All CT Algorithms

Please first ensure all the CT algorithms have been listed in `\Physics_World_Model\algorithm_base\ct\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CT solvers, please update the CT solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1960s–2000s):_
- FBP with Ram-Lak filter (Ramachandran & Lakshminarayanan, 1971)
- FBP with Shepp-Logan filter (Shepp & Logan, 1974)
- FBP with Hamming/Hann/Cosine windows
- ART — Algebraic Reconstruction Technique (Gordon et al., 1970)
- SIRT — Simultaneous Iterative Reconstruction Technique (Gilbert, 1972)
- SART — Simultaneous Algebraic Reconstruction Technique (Andersen & Kak, 1984)
- MLEM — Maximum Likelihood Expectation Maximization (Shepp & Vardi, 1982)
- OSEM — Ordered Subsets EM (Hudson & Larkin, 1994)
- FDK — Feldkamp-Davis-Kress for cone-beam CT (Feldkamp et al., 1984)

_Regularized / Optimization (2000s–2016):_
- TV regularization for CT (Sidky et al., PMB 2006; Sidky & Pan, 2008)
- TGV — Total Generalized Variation (Bredies et al., 2010)
- ADMM for CT (Boyd et al., 2011)
- FISTA for CT (Beck & Teboulle, 2009)
- Split Bregman for CT (Goldstein & Osher, 2009)
- Dictionary Learning CT (Xu et al., TMI 2012)
- PWLS — Penalized Weighted Least Squares (Fessler, 2000)
- Statistical iterative reconstruction — SIR (Thibault et al., 2007)
- PICCS — Prior Image Constrained Compressed Sensing (Chen et al., 2008)
- Chambolle-Pock / PDHG (Chambolle & Pock, 2011)
- MBIR — Model-Based Iterative Reconstruction (Thibault et al., Med Phys 2012)
- Low-Rank + Sparse CT (Gao et al., Med Phys 2012)
- PWLS-ULTRA (Zheng et al., 2016)
- ICD — Iterative Coordinate Descent (Bouman & Sauer, 2011)
- RED for CT — Regularization by Denoising (Romano et al., 2016)
- Interior tomography — DBP (Ye et al., 2013)

_Deep Learning (2016–2026):_
- RED-CNN — Residual Encoder-Decoder CNN (Chen et al., TMI 2017)
- FBPConvNet (Jin et al., TIP 2017)
- LEARN — Learned Experts' Assessment-based Reconstruction Network (Chen et al., TMI 2018)
- Learned Primal-Dual (Adler & Öktem, TMI 2018)
- WGAN-VGG for low-dose CT (Yang et al., TMI 2018)
- CPCE — Competitive Pathways CNN Ensemble (Shan et al., TMI 2019)
- DD-Net — Dense-Deconv Net (Zhang et al., 2018)
- iRadonMAP (He et al., 2019)
- DuDoNet — Dual Domain Network (Lin et al., TMI 2019)
- DuDoTrans (Wang et al., MICCAI 2022)
- Score-CT / DiffusionMBIR (Song et al., ICLR 2022; Chung et al., 2023)
- DOLCE — Diffusion Posterior Sampling for CT (Liu et al., 2023)
- InDuDoNet+ (Wang et al., MedIA 2023)
- CTformer — Transformer for low-dose CT (Wang et al., TMI 2023)
- FreeSeed (Chen et al., MICCAI 2024)
- DiffusionBlend (Gu et al., 2024)
- PnP-ADMM / PnP-HQS with BM3D/DnCNN for CT
- Neural Attenuation Fields for sparse-view CT (2023)

#### Step 3: Update CT Solvers

After listing all CT solvers, update `algorithm_base/ct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CT solvers use the data format: `y` (num_angles, num_detectors) sinogram data, `angles` array of projection angles. The `CTOperator` handles the forward model (Radon transform) and adjoint (backprojection) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CT:**
- AAPM Mayo quarter-dose: RED-CNN ~33.5 dB, WGAN-VGG ~34.0 dB, CTformer ~35.5 dB
- LoDoPaB-CT sparse-view (60 views): FBP ~21.0 dB, TV ~30.0 dB, Learned Primal-Dual ~35.5 dB, DiffusionMBIR ~37.0 dB
- LoDoPaB-CT challenge leaderboard top results
- Published PSNR/SSIM in the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ct/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`

---

### CT Fluorescence (`ct_fluorescence`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray Fluorescence CT, what dataset do you use to verify? Is this dataset used for XFCT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ct_fluorescence/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original XFCT standard dataset.

**Popular datasets to consider:**
- **Synchrotron XFCT Phantom Datasets (Cheong et al., PMB 2010)** — gold nanoparticle phantoms imaged at synchrotron beamlines; used for validating fluorescence CT reconstruction algorithms
- **Benchtop XFCT Phantoms (Jones et al., PMB 2012)** — laboratory-source XFCT phantom data with known element concentrations; used for benchtop XFCT algorithm development
- **K-edge Subtraction CT Datasets (Porra et al., 2010)** — synchrotron K-edge imaging data; related modality used for validation
- **Simulated XFCT Data (La Rivière et al., 2006)** — Monte Carlo-generated fluorescence CT datasets with known ground truth; standard for algorithm development
- **ANKA/ESRF Synchrotron XFCT Data (2015–2020)** — high-resolution synchrotron XRF-CT data; used in recent reconstruction papers

**Decision criteria:** Use simulated XFCT data with known ground truth for quantitative algorithm validation; supplement with synchrotron phantom data for real-data benchmarks.

#### Step 2: List All CT Fluorescence Algorithms

Please first ensure all the CT fluorescence algorithms have been listed in `\Physics_World_Model\algorithm_base\ct_fluorescence\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ct_fluorescence. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CT fluorescence solvers, please update the CT fluorescence solver.

**Key algorithms to cover (1950–2026):**

_Classical (1990s–2009):_
- FBP for XFCT — filtered back-projection applied to fluorescence sinograms (Hogan et al., 1991)
- MLEM for XFCT (La Rivière, IEEE TMI 2004)
- Simple line-integral model (neglecting attenuation) — baseline approach
- Attenuation-corrected FBP for XFCT (Rust & Weigelt, 1998)

_Optimization & Iterative (2010–2016):_
- Penalized likelihood XFCT with attenuation model (La Rivière & Bhatt, 2010)
- Monte Carlo forward model for XFCT (Bazalova-Carter et al., 2015)
- TV-regularized XFCT reconstruction (Ahmad et al., 2015)
- Compton scatter correction for XFCT (2012)
- Self-absorption correction algorithms (2013)
- Sparse-view XFCT reconstruction (2016)

_Deep Learning (2017–2026):_
- CNN-based XFCT image enhancement (2020)
- Deep learning XFCT reconstruction from sparse projections (2022)
- Physics-informed neural network for XFCT (2023)
- U-Net XFCT denoising (2021)
- Diffusion-prior XFCT (2025)

#### Step 3: Update CT Fluorescence Solvers

After listing all CT fluorescence solvers, update `algorithm_base/ct_fluorescence/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CT Fluorescence:**
- Synchrotron XFCT phantom: MLEM reconstruction element concentration accuracy within 10% of known values
- Benchtop XFCT: detection limit ~0.5% w/w gold nanoparticles (Jones et al., 2012)
- Published reconstruction PSNR/SSIM from recent deep learning papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ct_fluorescence' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ct_fluorescence/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CT fluorescence. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ct_fluorescence/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ct_fluorescence/standard/`

---

### PET-CT (`pet_ct`) Modality Template

#### Step 1: Verify Standard Dataset

For PET-CT, what dataset do you use to verify? Is this dataset used for PET-CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/pet_ct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original PET-CT standard dataset.

**Popular datasets to consider:**
- **AutoPET Challenge Dataset (Gatidis et al., 2022)** — whole-body FDG-PET/CT with lesion annotations; 900+ patients from University Hospital Tübingen; the primary benchmark for automated PET-CT analysis
- **TCIA PET-CT Collections** — multiple organ-specific PET-CT datasets (lung, head-neck, lymphoma)
- **NEMA IEC Body Phantom PET-CT Data** — standardized phantom with known sphere sizes and activity ratios; used for quantitative accuracy validation
- **SNMMI Clinical Trials Network Data** — standardized multi-site PET-CT phantom data for reproducibility assessment
- **QIN-HEADNECK (Clark et al., 2013)** — head-neck PET-CT with longitudinal imaging for treatment response evaluation

**Decision criteria:** AutoPET is the emerging community standard for whole-body PET-CT. NEMA phantom for quantitative accuracy validation. Use the dataset most widely referenced in PET-CT reconstruction and analysis papers (2010–2026).

#### Step 2: List All PET-CT Algorithms

Please first ensure all the PET-CT algorithms have been listed in `\Physics_World_Model\algorithm_base\pet_ct\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/pet_ct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the PET-CT solvers, please update the PET-CT solver.

**Key algorithms to cover (1950–2026):**

_Classical (1990s–2009):_
- Sequential PET + CT reconstruction — independent reconstruction of each modality
- CT-based attenuation correction for PET (Kinahan et al., Med Phys 1998)
- CT-derived scatter correction for PET (Watson, 2000)
- OSEM with CT attenuation map (standard clinical PET reconstruction)
- Registered PET-CT overlay — rigid registration of reconstructed volumes

_Optimization & Iterative (2010–2016):_
- Joint PET-CT reconstruction with structural prior (Bowsher et al., 2004; extended 2010+)
- Anatomy-guided PET reconstruction using CT edge information (2012)
- Kernel methods for PET with CT side information (Wang & Qi, TMI 2015)
- Total variation PET with CT anatomical penalty (Ehrhardt et al., 2015)
- Joint activity-attenuation estimation (2014)
- Motion-corrected PET-CT using CT-derived motion fields (2013)

_Deep Learning (2017–2026):_
- Deep learning PET attenuation correction from CT (Liu et al., 2018)
- CNN-based low-dose PET denoising with CT prior (Xiang et al., TMI 2017)
- Federated learning for PET-CT (Li et al., 2021)
- SubtlePET — deep learning PET enhancement (2020)
- Transformer-based PET-CT fusion (2023)
- AutoPET challenge methods — nnU-Net, UNETR for PET-CT segmentation (2022–2024)
- Diffusion-model PET-CT synthesis (2024)
- Multi-task PET-CT reconstruction + segmentation (2023)
- Deep learning scatter correction (2022)
- Foundation model for PET-CT analysis (2025)

#### Step 3: Update PET-CT Solvers

After listing all PET-CT solvers, update `algorithm_base/pet_ct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for PET-CT:**
- NEMA phantom: contrast recovery >90% for 37mm sphere, <10% background variability
- AutoPET challenge leaderboard (Dice, sensitivity, false positive volume)
- Published PET reconstruction PSNR/SSIM with CT prior vs. without

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'pet_ct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/pet_ct/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/pet_ct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/pet_ct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for PET-CT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/pet_ct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_ct/standard/`

---

### PET-MR (`pet_mr`) Modality Template

#### Step 1: Verify Standard Dataset

For PET-MR, what dataset do you use to verify? Is this dataset used for PET-MR popular algorithms? Please ensure the standard dataset in `datasets/benchmark/pet_mr/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original PET-MR standard dataset.

**Popular datasets to consider:**
- **Siemens mMR Benchmark Datasets** — simultaneous PET-MR brain and body data from Biograph mMR; used for attenuation correction and joint reconstruction validation
- **PETMR Attenuation Correction Challenge (Hofmann et al., 2011)** — brain PET-MR data with CT-derived reference attenuation maps; standard for MRAC algorithm validation
- **Ultra-high-field PET-MR Data (7T, 2018–2023)** — high-resolution brain PET-MR from research systems
- **IDB-Brain PET-MR Dataset** — multi-tracer brain PET-MR with anatomical MRI ground truth
- **Zubal Digital Phantom with PET-MR simulation** — synthetic PET-MR data with known activity and attenuation for quantitative validation

**Decision criteria:** Use PETMR AC Challenge data for attenuation correction benchmarks. Siemens mMR brain data for joint reconstruction. Use the dataset most widely referenced in PET-MR papers (2011–2026).

#### Step 2: List All PET-MR Algorithms

Please first ensure all the PET-MR algorithms have been listed in `\Physics_World_Model\algorithm_base\pet_mr\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/pet_mr. Besides, you need to search all algorithms from 1950 to 2026. After listing all the PET-MR solvers, please update the PET-MR solver.

**Key algorithms to cover (1950–2026):**

_Classical (2008–2009):_
- Dixon-based MR attenuation correction — MRAC from water/fat separation (Martinez-Möller et al., JNM 2009)
- Atlas-based attenuation correction — template registration for mu-map (Hofmann et al., JNM 2008)
- Segmentation-based MRAC — tissue classification (bone, air, soft tissue) from MRI
- UTE/ZTE-based attenuation maps — ultrashort/zero echo time for bone signal (Catana et al., 2010)

_Optimization & Iterative (2010–2016):_
- Joint PET-MR reconstruction with MR structural prior (Ehrhardt et al., 2015)
- Kernel method for PET with MR side information (Hutchcroft et al., 2016)
- Anatomy-guided PET reconstruction from MR segmentation (Bowsher et al., extended 2012)
- Simultaneous PET-MR motion correction (Catana et al., 2011)
- TOF-PET joint reconstruction with MR priors (2014)
- R2*-based MRAC correction (2015)
- Multi-atlas MRAC (Burgos et al., TMI 2014)

_Deep Learning (2017–2026):_
- Deep learning MRAC — CNN for synthetic CT from MRI (Han, MedIA 2017; Liu et al., 2018)
- CycleGAN for MR-to-CT attenuation map synthesis (Wolterink et al., 2017)
- Deep learning PET denoising with MR prior (Chen et al., 2019)
- MLAA-Net — joint activity-attenuation with deep learning (Hwang et al., 2019)
- Conditional GAN for synthetic CT from Dixon MRI (2020)
- Joint PET-MR deep reconstruction (Mehranian et al., 2020)
- Self-supervised PET-MR reconstruction (2022)
- Transformer-based MR-guided PET reconstruction (2024)
- Diffusion-model MRAC (2024)
- Foundation model for multi-modal PET-MR analysis (2025)

#### Step 3: Update PET-MR Solvers

After listing all PET-MR solvers, update `algorithm_base/pet_mr/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for PET-MR:**
- PETMR AC Challenge: bone-inclusive MRAC vs. CT reference — MAE <100 HU in bone regions
- Deep learning MRAC: MAE ~70-90 HU brain, ~120-150 HU pelvis (Liu et al., 2018)
- PET quantification accuracy: <5% bias in cortical regions with atlas-based MRAC
- Joint reconstruction: 1-3 dB PSNR improvement over sequential with MR prior

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'pet_mr' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/pet_mr/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/pet_mr/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/pet_mr/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for PET-MR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/pet_mr/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/pet_mr/standard/`

---

### SPECT-CT (`spect_ct`) Modality Template

#### Step 1: Verify Standard Dataset

For SPECT-CT, what dataset do you use to verify? Is this dataset used for SPECT-CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spect_ct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SPECT-CT standard dataset.

**Popular datasets to consider:**
- **SIMIND Monte Carlo Simulation (Ljungberg & Strand, 1989, updated 2020)** — the most widely used SPECT simulation tool; generates synthetic SPECT projections with known ground truth including attenuation, scatter, and collimator-detector response
- **XCAT Digital Phantom SPECT-CT (Segars et al., 2010)** — anatomically realistic digital phantom with cardiac and whole-body SPECT-CT simulation; standard for quantitative SPECT evaluation
- **Jaszczak Phantom Scans** — physical phantom with known sphere sizes for resolution/contrast assessment; used for scanner QC and algorithm validation
- **ACR SPECT Phantom Data** — standardized phantom scans for accreditation; provides reproducible benchmark
- **Open SPECT-CT Clinical Data (2020–2024)** — emerging open-access clinical SPECT-CT datasets for Tc-99m, Lu-177, I-131

**Decision criteria:** SIMIND + XCAT phantom is the standard for quantitative SPECT-CT algorithm development. Jaszczak phantom for hardware validation. Use the combination most widely referenced in SPECT reconstruction papers (2000–2026).

#### Step 2: List All SPECT-CT Algorithms

Please first ensure all the SPECT-CT algorithms have been listed in `\Physics_World_Model\algorithm_base\spect_ct\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spect_ct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SPECT-CT solvers, please update the SPECT-CT solver.

**Key algorithms to cover (1950–2026):**

_Classical (1980s–2009):_
- FBP for SPECT with Chang attenuation correction (Chang, JNM 1978)
- OSEM for SPECT (Hudson & Larkin, 1994)
- CT-based attenuation correction for SPECT (Hasegawa et al., 1993)
- Scatter correction using CT-derived scatter estimate (Frey & Tsui, 1993)
- Collimator-detector response (CDR) modeling in SPECT reconstruction
- Triple-energy window scatter correction (Ogawa et al., 1991)

_Optimization & Iterative (2010–2016):_
- MAP-OSEM with CT anatomical prior for SPECT (2010)
- 4D SPECT-CT with CT-derived respiratory motion model (2012)
- Quantitative SPECT/CT for dosimetry (Dewaraja et al., JNM 2012)
- Joint SPECT-CT reconstruction (Nuyts et al., 2013)
- Total variation regularized SPECT with CT prior (2014)
- Monte Carlo-based SPECT reconstruction with CT scatter model (2015)

_Deep Learning (2017–2026):_
- CNN-based SPECT denoising with CT prior (Ramon et al., 2020)
- Deep learning scatter correction for SPECT-CT (Xiang et al., 2020)
- Deep learning attenuation and scatter correction without CT (Shi et al., 2020)
- U-Net SPECT reconstruction (2021)
- DL-based quantitative SPECT for Lu-177 dosimetry (2022)
- Transformer-based SPECT-CT reconstruction (2024)
- Self-supervised SPECT denoising (2023)
- Physics-informed neural network for SPECT-CT (2024)
- Diffusion-model SPECT enhancement (2025)

#### Step 3: Update SPECT-CT Solvers

After listing all SPECT-CT solvers, update `algorithm_base/spect_ct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SPECT-CT:**
- XCAT phantom: OSEM+CT-AC quantification accuracy within 5% for Tc-99m myocardial perfusion
- Jaszczak phantom: contrast recovery >60% for 25mm sphere, <10% background variability
- Published quantitative SPECT accuracy from dosimetry papers (Dewaraja et al.)
- Deep learning SPECT: 2-4 dB improvement over OSEM on simulated data

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spect_ct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spect_ct/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spect_ct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spect_ct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SPECT-CT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spect_ct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spect_ct/standard/`

---

### US-MRI Fusion (`us_mri`) Modality Template

#### Step 1: Verify Standard Dataset

For Ultrasound-MRI Fusion, what dataset do you use to verify? Is this dataset used for US-MRI fusion popular algorithms? Please ensure the standard dataset in `datasets/benchmark/us_mri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original US-MRI fusion standard dataset.

**Popular datasets to consider:**
- **RESECT Dataset (Xiao et al., IJCARS 2017)** — intraoperative ultrasound + preoperative MRI brain registration benchmark; 23 cases with expert-annotated landmarks; the primary benchmark for neurosurgical US-MRI registration
- **CuRIOUS Challenge Dataset (2018, 2019)** — brain US-MRI registration challenge with ground-truth landmarks; community benchmark for deformable registration algorithms
- **Prostate US-MRI Fusion Biopsy Datasets (2015–2023)** — clinical prostate US-MRI registration data used for targeted biopsy guidance
- **Liver US-MRI Registration Data** — abdominal US-MRI pairs with deformable registration ground truth
- **MICCAI EASY-REG Challenge Data (2020)** — multi-organ US-MRI registration benchmark

**Decision criteria:** RESECT is the most widely used benchmark for US-MRI brain registration. Prostate US-MRI for clinical fusion. Use the dataset most widely referenced in US-MRI registration papers (2015–2026).

#### Step 2: List All US-MRI Algorithms

Please first ensure all the US-MRI algorithms have been listed in `\Physics_World_Model\algorithm_base\us_mri\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/us_mri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the US-MRI solvers, please update the US-MRI solver.

**Key algorithms to cover (1950–2026):**

_Classical (1990s–2009):_
- Rigid registration — point-based and intensity-based (ICP, mutual information)
- Landmark-based US-MRI alignment (manual fiducials, 2000)
- Block matching registration (Ourselin et al., 2000)
- Electromagnetic tracking-based US-MRI fusion (2005)

_Optimization & Model-Based (2010–2016):_
- Deformable registration — B-spline FFD (Rueckert et al., 1999; applied to US-MRI 2010+)
- Demons algorithm for US-MRI (Thirion, 1998; extended 2012)
- NiftyReg — GPU-accelerated B-spline registration (Modat et al., 2010)
- LC2 — Linear Correlation of Linear Combination similarity metric for US-MRI (Wein et al., 2008; refined 2013)
- Biomechanical model-based registration (Wittek et al., 2007; extended 2014)
- SyN — Symmetric Normalization in ANTs (Avants et al., 2008; applied 2015)
- Multi-modal image synthesis for US-MRI alignment (2016)

_Deep Learning (2017–2026):_
- VoxelMorph adapted for US-MRI (Balakrishnan et al., 2019; applied 2020)
- Label-driven weakly supervised US-MRI registration (Hu et al., MICCAI 2018)
- Deep learning US simulation from MRI for registration training (Prevost et al., 2017)
- KeyMorph — keypoint-based deformable registration (2022)
- TransMorph — Transformer registration (Chen et al., MedIA 2022; applied to US-MRI 2023)
- Self-supervised US-MRI registration (2021)
- Diffeomorphic registration networks for US-MRI (2022)
- Physics-informed registration with tissue biomechanics (2023)
- Foundation model for multi-modal medical image registration (2025)
- Real-time deep learning US-MRI fusion for interventional guidance (2024)

#### Step 3: Update US-MRI Solvers

After listing all US-MRI solvers, update `algorithm_base/us_mri/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for US-MRI Fusion:**
- RESECT dataset: mean target registration error (mTRE) — rigid ~5.0 mm, B-spline ~2.5 mm, VoxelMorph ~2.0 mm, best methods ~1.5 mm
- CuRIOUS challenge: mTRE leaderboard results
- Prostate US-MRI fusion: target registration accuracy <3 mm for targeted biopsy
- Published mTRE from VoxelMorph, TransMorph, biomechanical model papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'us_mri' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/us_mri/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/us_mri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/us_mri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for US-MRI fusion. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/us_mri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/us_mri/standard/`
