
---

## Flagship Modalities — Full 7-Step Templates

These 12 modalities are the **flagship** set validated in the PWM paper "Eleven Primitives and Three Gates" (Yang, Brady, Yuan). Each has full end-to-end **Scenario I–IV correction validation** across all five carrier families (Photon, Electron, Nuclear Spin, X-ray Photon, Acoustic Wave). They represent the most thoroughly benchmarked inverse-problem pipelines in the Physics World Model framework.

The 12 flagships span five carrier families:
- **Photon:** CASSI, CACTI, SPC, Lensless, Holography, Widefield Fluorescence
- **Electron:** Ptychography, Cryo-EM
- **Nuclear Spin:** MRI
- **X-ray Photon:** CT, CBCT
- **Acoustic Wave:** Ultrasound

---

### CASSI (`cassi`) Modality Template

Coded Aperture Snapshot Spectral Imaging. Carrier: Photon. DAG: M→W→Σ→D.

#### Step 1: Verify Standard Dataset

For CASSI, what dataset do you use to verify? Is this dataset used for CASSI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cassi/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CASSI standard dataset.

**Popular datasets to consider:**
- **KAIST Spectral Dataset (Choi et al., ICCV 2017)** — 28 spectral bands, 256×256 spatial resolution; the most widely used benchmark for coded aperture snapshot spectral imaging reconstruction; used by MST, CST, DAUHST, TSA-Net, HDNet, and virtually all CASSI papers since 2019
- **CAVE Multispectral Image Database (Yasuma et al., 2010)** — 31 bands (400–700 nm, 10 nm step), 512×512; indoor scenes of everyday objects; used for spectral reconstruction training and evaluation
- **Harvard Spectral Database (Chakrabarti & Zickler, CVPR 2011)** — 31 bands, 1392×1040; indoor and outdoor natural scenes; used as supplementary evaluation in many CASSI papers
- **ICVL Hyperspectral Dataset (Arad & Ben-Shahar, ECCV 2016)** — 31 bands, 1392×1300; 200+ natural scenes captured with a spectrograph; widely used for spectral super-resolution and CASSI training
- **ARAD_1K (Arad et al., CVPRW 2022)** — 1000 hyperspectral images, 31 bands, 482×512; the NTIRE spectral reconstruction challenge dataset
- **Meng et al. Real CASSI Data (2020)** — real hardware CASSI captures with physical coded aperture and dispersive element; used for real-system validation

**Decision criteria:** KAIST (28 bands, 256×256) is the undisputed gold standard for CASSI reconstruction benchmarking (2019–2026). CAVE and ICVL are the primary training datasets. Use KAIST as the standard evaluation set, consistent with the majority of published CASSI reconstruction papers.

#### Step 2: List All CASSI Algorithms

Please first ensure all the CASSI algorithms have been listed in `\Physics_World_Model\algorithm_base\cassi\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cassi. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CASSI solvers, please update the CASSI solver.

**Key algorithms to cover (2006–2026):**

_Classical / Analytic (2006–2012):_
- CASSI original reconstruction — two-step inversion via direct demultiplexing (Wagadarikar et al., Appl. Opt. 2008) — earliest coded aperture spectral imaging reconstruction
- TwIST — Two-step Iterative Shrinkage/Thresholding for CASSI (Bioucas-Dias & Figueiredo, IEEE TIP 2007) — one of the first optimization solvers applied to CASSI
- GPSR — Gradient Projection for Sparse Reconstruction (Figueiredo et al., JSTSP 2007) — applied to coded aperture spectral compressive sensing
- OMP for spectral CS — Orthogonal Matching Pursuit adapted for spectral compressive sensing (Tropp & Gilbert, IEEE TIT 2007)
- CoSaMP for CASSI (Needell & Tropp, Appl. Comput. Harmon. Anal. 2009)
- Dual-disperser CASSI — DD-CASSI with enhanced spectral multiplexing (Gehm et al., Opt. Express 2007)

_Optimization & Model-Based (2013–2018):_
- GAP-TV — Generalized Alternating Projection with Total Variation (Yuan, ICIP 2016) — the canonical optimization baseline for snapshot compressive imaging; PSNR ~33.0 dB on KAIST
- ADMM for CASSI — Alternating Direction Method of Multipliers for spectral SCI (Boyd et al., 2011; applied by Tan et al., 2015)
- DeSCI — Decompress Snapshot Compressive Imaging via weighted nuclear norm (Liu et al., IEEE TPAMI 2019) — rank minimization approach
- SeSCI — Self-supervised Snapshot Compressive Imaging (Yang et al., 2020)
- GAP-ADMM — Generalized Alternating Projection with ADMM inner solver (Yuan, 2020)
- PnP-CASSI — Plug-and-Play priors for CASSI with FFDNet/DnCNN denoiser (Yuan et al., 2020)
- Sparse Bayesian Learning for CASSI (Babacan et al., 2012)
- Dictionary Learning for CASSI spectral reconstruction (Lin et al., 2014)

_Deep Learning (2019–2026):_
- λ-Net — Learned spectral reconstruction network for CASSI (Miao et al., ICCV 2019) — first end-to-end deep learning for CASSI
- TSA-Net — Temporal-Spatial Attention Network for CASSI (Meng et al., ECCV 2020) — spatial-spectral attention mechanism; PSNR ~33.2 dB on KAIST
- HDNet — High-resolution Dual-domain Network for CASSI (Hu et al., CVPR 2022) — dual-domain learning; PSNR ~35.1 dB on KAIST
- DGSMP — Deep Gaussian Scale Mixture Prior for CASSI (Huang et al., CVPR 2021) — MAP-inspired deep unfolding; PSNR ~33.3 dB
- MST-S/MST-L — Mask-guided Spectral-wise Transformer (Cai et al., CVPR 2022) — spectral self-attention; MST-L PSNR ~35.5 dB on KAIST
- CST-L/CST-L+ — Coarse-to-fine Spectral Transformer (Cai et al., ECCV 2022) — progressive spectral reconstruction; CST-L PSNR ~36.1 dB on KAIST
- DAUHST — Degradation-Aware Unfolding Half-Shuffle Transformer (Cai et al., NeurIPS 2022) — unfolding + Transformer; PSNR ~37.2 dB on KAIST
- EfficientSCI — Efficient Snapshot Compressive Imaging (Wang et al., CVPR 2023) — lightweight architecture
- PADUT — Plug-and-Play Attention-based Deep Unfolding Transformer (Li et al., 2023)
- RDLUF-MixS2 — Residual Degradation Learning Unfolding Framework (Dong et al., CVPR 2023)
- BiSRNet — Bidirectional Spectral Reconstruction Network (Li et al., 2023)
- Diffusion-prior SCI — diffusion model priors for snapshot compressive imaging (Meng et al., 2023)
- SST — Spatial-Spectral Transformer for hyperspectral image reconstruction (Jiang et al., 2024)
- Foundation model for spectral SCI (2025)

#### Step 3: Update CASSI Solvers

After listing all CASSI solvers, update `algorithm_base/cassi/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CASSI solvers use the data format: `y` (H, W+λ-1) compressed 2D measurement (spatial-spectral sheared snapshot), `mask` (H, W, λ) coded aperture pattern across spectral bands, `shift_list` spectral dispersion offsets. The `CASSIOperator` handles forward `y = Σ_λ mask_λ · shift_λ(x_λ)` and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CASSI:**
- KAIST 28 bands, 256×256, 10 scenes: TwIST ~30.5 dB, GAP-TV ~33.0 dB / 0.917 SSIM, DeSCI ~33.3 dB, TSA-Net ~33.2 dB / 0.925 SSIM, DGSMP ~33.3 dB / 0.926 SSIM, λ-Net ~31.4 dB, HDNet ~35.1 dB / 0.952 SSIM, MST-L ~35.5 dB / 0.955 SSIM, CST-L ~36.1 dB / 0.960 SSIM, DAUHST ~37.2 dB / 0.967 SSIM
- CAVE 31 bands (simulated CASSI): similar ranking with slightly different absolute values
- All reference values from published papers and the MST/CST/DAUHST benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cassi' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cassi/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cassi/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CASSI. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cassi/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cassi/standard/`

---

### CACTI (`cacti`) Modality Template

Coded Aperture Compressive Temporal Imaging. Carrier: Photon. DAG: M→Σ→D.

#### Step 1: Verify Standard Dataset

For CACTI, what dataset do you use to verify? Is this dataset used for CACTI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cacti/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CACTI standard dataset.

**Popular datasets to consider:**
- **EfficientSCI Benchmark (Wang et al., CVPR 2023)** — 512×512 spatial, compression ratio (CR) 10; grayscale video snapshot compressive imaging with dynamic masks; the standard large-scale CACTI benchmark
- **Simulation 256×256×8 Dataset (Yuan et al.)** — 256×256 spatial, 8 temporal frames compressed into a single snapshot; the canonical small-scale CACTI evaluation set used by GAP-TV, PnP-FFDNet, BIRNAT, RevSCI, MetaSCI, and most temporal SCI papers
- **DAVIS Video Dataset (Pont-Tuset et al., arXiv 2017)** — high-quality video sequences used for generating simulated CACTI measurements; training set for many temporal SCI deep learning methods
- **Kobe/Traffic/Runner/Drop/Crash/Aerial 6-Scene Benchmark** — the classic 6-scene grayscale video test set (256×256×8 at CR=8) used by nearly all CACTI reconstruction papers since 2019
- **Real CACTI Hardware Data (Llull et al., Opt. Express 2013)** — real captures from the original CACTI prototype; used for real-system validation
- **Color Video SCI Benchmark (Yuan et al., 2021)** — color video SCI with Bayer-pattern coded apertures; extended benchmark for color temporal SCI

**Decision criteria:** The 6-scene benchmark (Kobe/Traffic/Runner/Drop/Crash/Aerial, 256×256×8, CR=8) is the undisputed gold standard for CACTI algorithm comparison (2019–2026). EfficientSCI benchmark (512×512, CR=10) is the emerging large-scale standard. Use the 6-scene set as the primary evaluation dataset, consistent with the majority of published papers.

#### Step 2: List All CACTI Algorithms

Please first ensure all the CACTI algorithms have been listed in `\Physics_World_Model\algorithm_base\cacti\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cacti. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CACTI solvers, please update the CACTI solver.

**Key algorithms to cover (2011–2026):**

_Classical / Optimization (2011–2018):_
- TwIST for CACTI — Two-step Iterative Shrinkage/Thresholding for video SCI (Bioucas-Dias & Figueiredo, TIP 2007; applied to CACTI by Llull et al., 2013)
- GAP-TV — Generalized Alternating Projection with Total Variation for video SCI (Yuan, ICIP 2016) — the canonical optimization baseline; PSNR ~27.0 dB on 6-scene benchmark
- GAP-wavelet — GAP with wavelet sparsity prior (Yuan, 2016)
- ADMM-TV for video SCI (Boyd et al., 2011; applied to CACTI)
- DeSCI — Decompress SCI via weighted nuclear norm minimization for video (Liu et al., TPAMI 2019) — rank-based approach exploiting temporal correlation; ~30.0 dB but very slow
- PnP-FFDNet — Plug-and-Play with FFDNet denoiser for video SCI (Yuan et al., 2020) — PnP framework with pre-trained denoiser; ~31.5 dB
- PnP-FastDVDnet — Plug-and-Play with FastDVDnet temporal denoiser (Yuan et al., 2021) — temporal-aware denoiser; ~32.0 dB
- GAP-FFDNet — GAP framework with FFDNet denoiser (2020)
- Sparse-3D for video SCI — 3D wavelet/DCT sparsity (2014)
- MMLE-GMM for CACTI — maximum marginal likelihood estimation with Gaussian mixture model (Yang et al., 2014)

_Deep Learning (2019–2026):_
- RevSCI — Reversible SCI Network (Cheng et al., CVPR 2021) — memory-efficient reversible architecture; ~33.0 dB on 6-scene benchmark
- BIRNAT — Bidirectional Recurrent Neural Architecture for Temporal SCI (Cheng et al., ECCV 2020) — bidirectional RNN for temporal SCI; ~32.7 dB
- MetaSCI — Meta-learning for Scalable SCI (Wang et al., CVPR 2021) — meta-learning framework for different compression ratios; ~31.8 dB
- STFormer — Spatial-Temporal Transformer for Video SCI (Wang et al., NeurIPS 2022) — Transformer-based architecture; ~34.5 dB on 6-scene benchmark
- EfficientSCI — Efficient Snapshot Compressive Imaging (Wang et al., CVPR 2023) — lightweight yet high-performance; ~33.0 dB on 6-scene, ~33.5 dB on 512×512 CR=10
- GAP-net — learned GAP unfolding for video SCI (Meng et al., 2020)
- ADMM-Net for video SCI — deep unfolding ADMM (Ma et al., 2019)
- SCI-3D — 3D CNN for video SCI reconstruction (2020)
- U-net baseline for video SCI (2019)
- Deep Equilibrium SCI — DEQ model for video SCI (2022)
- CTM-SCI — CNN-Transformer Mixed for video SCI (2023)
- Diffusion-prior video SCI (2024)
- Video SCI Foundation Model (2025)

#### Step 3: Update CACTI Solvers

After listing all CACTI solvers, update `algorithm_base/cacti/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CACTI solvers use the data format: `y` (H, W) single 2D compressed snapshot measurement, `mask` (H, W, T) binary coded aperture masks for T temporal frames. The `CACTIOperator` handles forward `y = Σ_t mask_t · x_t` (temporal multiplexing) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CACTI:**
- 6-scene benchmark (256×256×8, CR=8): GAP-TV ~27.0 dB / 0.839 SSIM, DeSCI ~30.0 dB / 0.907 SSIM, PnP-FFDNet ~31.5 dB / 0.926 SSIM, PnP-FastDVDnet ~32.0 dB / 0.933 SSIM, BIRNAT ~32.7 dB / 0.941 SSIM, RevSCI ~33.0 dB / 0.945 SSIM, STFormer ~34.5 dB / 0.960 SSIM, EfficientSCI ~33.0 dB / 0.948 SSIM
- EfficientSCI benchmark (512×512, CR=10): GAP-TV ~24.5 dB, EfficientSCI ~33.5 dB, STFormer ~34.0 dB
- All reference values from published papers and the STFormer/EfficientSCI benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cacti' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cacti/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cacti/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cacti/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cacti/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CACTI. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cacti/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cacti/standard/`

---

### SPC (`spc`) Modality Template

Single-Pixel Camera. Carrier: Photon. DAG: M→Σ→D.

#### Step 1: Verify Standard Dataset

For SPC, what dataset do you use to verify? Is this dataset used for SPC popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spc/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SPC standard dataset.

**Popular datasets to consider:**
- **Set11 (Kulkarni et al., CVPR 2016)** — 11 standard test images resized to 256×256 grayscale; the most widely used benchmark for single-pixel camera reconstruction; used by ReconNet, SPC-Net, HATNet, and virtually all deep learning SPC papers
- **BSD68 (Martin et al., ICCV 2001)** — 68 natural images from the Berkeley Segmentation Dataset; widely used as secondary evaluation for single-pixel imaging
- **DIV2K (Agustsson & Timofte, CVPRW 2017)** — 1000 high-resolution natural images; used for training deep learning SPC methods
- **Rice Single-Pixel Camera Experimental Data (Duarte et al., IEEE SPM 2008)** — real hardware SPC measurements using DMD-based random patterns and single photodetector; the original experimental SPC dataset
- **MNIST/CIFAR-10 (simulated SPC)** — standard classification datasets with simulated single-pixel measurements; used for low-resolution SPC algorithm validation
- **Set14 / Urban100 / Manga109** — extended natural image test sets used as supplementary SPC benchmarks

**Decision criteria:** Set11 (256×256 grayscale) is the undisputed gold standard for SPC reconstruction benchmarking at various compression ratios (1%, 4%, 10%, 25%, 50%). BSD68 for statistical evaluation. Use Set11 as the standard evaluation set, consistent with the majority of published SPC papers.

#### Step 2: List All SPC Algorithms

Please first ensure all the SPC algorithms have been listed in `\Physics_World_Model\algorithm_base\spc\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spc. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SPC solvers, please update the SPC solver.

**Key algorithms to cover (2006–2026):**

_Classical Sparse Recovery (2006–2012):_
- OMP — Orthogonal Matching Pursuit for SPC (Tropp & Gilbert, IEEE TIT 2007) — greedy sparse recovery baseline
- CoSaMP — Compressive Sampling Matching Pursuit (Needell & Tropp, Appl. Comput. Harmon. Anal. 2009)
- Basis Pursuit / L1 minimization (Candes & Tao, IEEE TIT 2005; Chen et al., SIAM J. Sci. Comput. 1998)
- LASSO for SPC (Tibshirani, JRSS-B 1996; applied to SPC)
- Bayesian Compressive Sensing (Ji et al., IEEE TSP 2008) — probabilistic sparse recovery
- Subspace Pursuit (Dai & Milenkovic, IEEE TIT 2009)
- Iterative Hard Thresholding — IHT (Blumensath & Davies, Appl. Comput. Harmon. Anal. 2009)
- Total Variation minimization for SPC (Chambolle, 2004; applied to SPC by Duarte et al., 2008)

_Optimization & Model-Based (2009–2018):_
- FISTA-TV — Fast Iterative Shrinkage-Thresholding with Total Variation (Beck & Teboulle, SIAM J. Imaging Sci. 2009) — PSNR ~26.0 dB at 25% sampling on Set11
- TVAL3 — TV minimization by Augmented Lagrangian and ALternating direction ALgorithms (Li et al., Rice TR 2009) — fast TV solver for CS; ~25.5 dB at 25% on Set11
- SPC-ADMM — ADMM for single-pixel imaging with sparsity + TV (2015)
- D-AMP — Denoising-based Approximate Message Passing (Metzler et al., IEEE TIT 2016) — state-of-the-art optimization SPC solver; ~28.0 dB at 25% on Set11
- LDAMP — Learned D-AMP (Metzler et al., NIPS 2017) — learned denoiser within AMP framework
- NLR-CS — Non-Local Regularization for CS (Dong et al., SIAM J. Imaging Sci. 2014)
- GSR — Group Sparse Representation for CS (Zhang et al., TIP 2014)
- Structured random patterns — Hadamard, Fourier, wavelet-based patterns (2012)

_Deep Learning (2016–2026):_
- ReconNet — CNN for real-time CS reconstruction (Kulkarni et al., CVPR 2016) — first end-to-end deep learning for SPC; ~24.5 dB at 25% on Set11
- DR2-Net — Deep Residual Reconstruction Network for CS (Yao et al., Neurocomputing 2019) — residual learning; ~26.5 dB at 25%
- SPC-Net / ISTA-Net — learned iterative shrinkage-thresholding for CS (Zhang & Ghanem, CVPR 2018) — deep unfolding for SPC; ~28.5 dB at 25%
- CSNet — Deep CS Network (Shi et al., CVPRW 2017) — joint sampling and reconstruction
- OPINE-Net — Optimization-Inspired Network for CS (Zhang et al., JSTSP 2020) — optimization-inspired deep architecture; ~29.0 dB at 25%
- AMP-Net — Denoising-based AMP Network (Zhang et al., TIP 2021) — deep unfolding AMP
- HATNet — Hybrid Attention Transformer for SPC (2023) — Transformer-based; ~29.0 dB at 25% on Set11
- TransCS — Transformer for Compressive Sensing (Shen et al., 2022)
- SCI-BDVP — Bayesian Deep Video Prior for SPC (2022)
- CASNet — Cross-Attention Sampling Network (2023)
- DPC — Deep Probabilistic CS (2022)
- Diffusion-CS — Diffusion model for CS reconstruction (2023)
- Adaptive sampling + reconstruction networks (2024)
- Foundation model for compressive sensing (2025)

#### Step 3: Update SPC Solvers

After listing all SPC solvers, update `algorithm_base/spc/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SPC solvers use the data format: `y` (M,) vector of M bucket-detector measurements (M < N for compression), `Phi` (M, N) measurement matrix (random Gaussian, binary, Hadamard, or learned), where N = H×W is the vectorized image dimension. The `SPCOperator` handles forward `y = Phi @ x` and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SPC:**
- Set11 256×256 grayscale at 25% sampling rate: FISTA-TV ~26.0 dB, TVAL3 ~25.5 dB, D-AMP ~28.0 dB, ReconNet ~24.5 dB, ISTA-Net ~28.5 dB, OPINE-Net ~29.0 dB, HATNet ~29.0 dB
- Set11 at 10% sampling rate: FISTA-TV ~22.0 dB, D-AMP ~24.5 dB, ISTA-Net ~25.5 dB, HATNet ~26.0 dB
- Set11 at 4% sampling rate: FISTA-TV ~19.0 dB, ReconNet ~20.5 dB, ISTA-Net ~22.0 dB
- Set11 at 50% sampling rate: FISTA-TV ~29.5 dB, D-AMP ~32.0 dB, ISTA-Net ~33.0 dB, HATNet ~33.5 dB
- All reference values from published papers and the ISTA-Net/OPINE-Net benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spc' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spc/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spc/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spc/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spc/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SPC. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spc/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spc/standard/`

---

### Lensless Imaging (`lensless`) Modality Template

Lensless Camera (Diffuser/Mask). Carrier: Photon. DAG: C→D.

#### Step 1: Verify Standard Dataset

For Lensless Imaging, what dataset do you use to verify? Is this dataset used for lensless imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/lensless/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original lensless imaging standard dataset.

**Popular datasets to consider:**
- **DiffuserCam Lensless Mirflickr Dataset (DLMD) (Monakhova et al., IEEE TCI 2019)** — 25,000 lensless images captured with a diffuser-based camera paired with lensed ground truth from Mirflickr-25K; the primary benchmark for lensless image reconstruction; used by FlatNet, U-Net lensless, and virtually all deep learning lensless papers
- **PhlatCam Dataset (Asif et al., Sci. Adv. 2017)** — lensless images from a phase-mask-based flat camera; used for PSF-based lensless reconstruction validation
- **FlatCam Dataset (Asif et al., IEEE TCI 2017)** — lensless images from an amplitude-mask flat camera with separable PSF; the original flat lensless camera benchmark
- **CelebA Lensless Simulation (2020)** — simulated lensless measurements from CelebA face dataset with measured PSFs; used for face reconstruction evaluation
- **LenslessPiCam Dataset (Bezzam et al., 2023)** — Raspberry Pi-based lensless camera dataset; low-cost lensless benchmark with measured PSFs
- **Multi-Spectral Lensless Dataset (2022)** — spectral lensless captures for color reconstruction evaluation

**Decision criteria:** DiffuserCam DLMD is the undisputed gold standard for lensless image reconstruction (2019–2026). FlatCam/PhlatCam for alternative lensless architectures. Use DiffuserCam DLMD as the standard evaluation set, consistent with the majority of published lensless imaging papers.

#### Step 2: List All Lensless Algorithms

Please first ensure all the lensless algorithms have been listed in `\Physics_World_Model\algorithm_base\lensless\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/lensless. Besides, you need to search all algorithms from 1950 to 2026. After listing all the lensless solvers, please update the lensless solver.

**Key algorithms to cover (2013–2026):**

_Classical / Analytic (2013–2016):_
- Wiener deconvolution for lensless imaging — frequency-domain inversion with PSF (2013)
- Tikhonov-regularized deconvolution — L2-penalized least-squares PSF inversion (applied to lensless by Asif et al., 2017)
- Matched filter — correlation-based reconstruction (2015)
- Naive inverse filtering — direct Fourier division by OTF (demonstrates noise amplification)
- Lucy-Richardson for lensless — iterative ML deconvolution with PSF (2014)

_Optimization & Model-Based (2016–2020):_
- FISTA-TV for lensless — Fast Iterative Shrinkage-Thresholding with Total Variation (Beck & Teboulle, 2009; applied to lensless reconstruction) — PSNR ~23.0 dB on DiffuserCam (PSF σ=2.0)
- ADMM for lensless imaging — Alternating Direction Method of Multipliers with sparsity + TV (Boyd et al., 2011; applied to lensless by Monakhova et al., 2019) — ~24.5 dB
- PnP-ADMM for lensless — Plug-and-Play with DnCNN/BM3D denoiser (Monakhova et al., 2019) — ~25.5 dB
- GD-TV — Gradient Descent with Total Variation for lensless (2017)
- Non-negative FISTA for lensless (enforcing physical non-negativity constraint, 2018)
- Separable PSF decomposition for FlatCam (Asif et al., 2017) — exploits rank-1 PSF structure for fast reconstruction
- Multi-Wiener for lensless — depth-dependent Wiener filter for 3D lensless (Adams et al., 2017)
- APGD — Accelerated Proximal Gradient Descent for lensless (2019)

_Deep Learning (2019–2026):_
- FlatNet — end-to-end U-Net for lensless image reconstruction (Khan et al., IEEE TCI 2020) — learned lensless reconstruction; PSNR ~28.0 dB on DiffuserCam
- Le-ADMM-U — Learned ADMM unrolled network for lensless imaging (Monakhova et al., Opt. Express 2021) — deep unfolding; ~27.5 dB
- LenslessNet — lightweight CNN for lensless reconstruction (2021)
- PhlatCam-DL — deep learning for phase-mask lensless camera (2020)
- U-Net lensless baseline (Monakhova et al., 2019) — U-Net directly mapping measurement to image; ~26.0 dB
- TrainInv — Training-free physics-informed lensless reconstruction (Zeng et al., 2021)
- Multi-depth lensless DL — depth-aware lensless reconstruction network (2022)
- MWDN — Multi-Wiener Deconvolution Network (2022) — learned multi-depth reconstruction
- Diffusion-prior lensless — diffusion model for lensless image reconstruction (Zeng et al., 2023)
- FlatDiffusion — diffusion-based lensless reconstruction (2024)
- LenslessGAN — GAN-based lensless image enhancement (2023)
- Physics-informed Transformer for lensless imaging (2024)
- Foundation model for computational imaging (2025)

#### Step 3: Update Lensless Solvers

After listing all lensless solvers, update `algorithm_base/lensless/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All lensless solvers use the data format: `y` (H, W) or (H, W, 3) raw sensor measurement (convolution of scene with PSF), `psf` (H, W) or (H, W, 3) measured point spread function of the lensless element (diffuser, mask, or phase plate). The `LenslessOperator` handles forward `y = PSF * x` (2D convolution) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Lensless Imaging:**
- DiffuserCam DLMD: Wiener ~20.0 dB, FISTA-TV ~23.0 dB / 0.65 SSIM, ADMM ~24.5 dB / 0.70 SSIM, PnP-ADMM ~25.5 dB / 0.75 SSIM, U-Net lensless ~26.0 dB / 0.78 SSIM, Le-ADMM-U ~27.5 dB / 0.83 SSIM, FlatNet ~28.0 dB / 0.85 SSIM
- FlatCam separable PSF: Tikhonov ~19.0 dB, Separable ADMM ~23.0 dB
- All reference values from published papers and the DiffuserCam/FlatNet benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'lensless' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/lensless/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/lensless/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/lensless/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/lensless/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for lensless imaging. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/lensless/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/lensless/standard/`

---

### Holography (`holography`) Modality Template

Digital Holographic Microscopy. Carrier: Photon. DAG: M→Σ→D.

#### Step 1: Verify Standard Dataset

For Holography, what dataset do you use to verify? Is this dataset used for holography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/holography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original holography standard dataset.

**Popular datasets to consider:**
- **USAF 1951 Resolution Target Hologram (standard)** — holographic recording of a USAF resolution target; the canonical benchmark for evaluating spatial resolution of holographic reconstruction algorithms; used universally across holography papers
- **Bead/Cell Hologram Datasets (Rivenson et al., Light: Science & Applications 2018)** — digital holographic microscopy recordings of microspheres and biological cells with known ground truth; used for validating phase retrieval and holographic reconstruction
- **DIH Datasets (Shao et al., 2020)** — digital in-line holography datasets with single and multiple particles; used for 3D particle tracking and reconstruction validation
- **HoloNet Training Data (Wu et al., Nat. Methods 2019)** — paired hologram and fully-focused ground truth for training deep learning holographic autofocusing and phase recovery
- **Fresnel Propagation Simulated Holograms** — synthetically generated holograms with known object amplitude and phase for quantitative evaluation of reconstruction accuracy
- **Off-Axis DHM Datasets (Colomb et al., Appl. Opt. 2006)** — off-axis digital holographic microscopy recordings with carrier frequency for spatial filtering-based reconstruction
- **NIST Holographic Microscopy Reference (2020)** — standardized holographic recording with calibrated phase objects

**Decision criteria:** USAF resolution target hologram is the universal baseline for spatial resolution evaluation. HoloNet paired data for deep learning evaluation. Bead/cell holograms for biological imaging applications. Use the dataset most widely referenced in holographic reconstruction papers (2006–2026).

#### Step 2: List All Holography Algorithms

Please first ensure all the holography algorithms have been listed in `\Physics_World_Model\algorithm_base\holography\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/holography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the holography solvers, please update the holography solver.

**Key algorithms to cover (1948–2026):**

_Analytic / Classical (1948–2005):_
- Angular Spectrum Method (ASM) — Fourier-domain free-space propagation (Goodman, Introduction to Fourier Optics, 1968) — the standard numerical propagation method; PSNR ~28.0 dB on typical holograms
- Fresnel Transform — paraxial approximation propagation (Schnars & Jüptner, Appl. Opt. 1994) — the canonical digital holographic reconstruction method
- Convolution Method — real-space propagation via PSF convolution (Kreis, Handbook of Holographic Interferometry, 2005)
- Off-axis spatial filtering — carrier frequency separation for off-axis holography (Cuche et al., Appl. Opt. 1999; Colomb et al., 2006)
- Phase-shifting interferometry — multi-frame phase retrieval with known reference phase shifts (Yamaguchi & Zhang, Opt. Lett. 1997)
- Double-exposure holographic interferometry (1965)
- Numerical refocusing — digital propagation to different focal planes (Ferraro et al., Opt. Lett. 2003)

_Phase Retrieval / Optimization (1972–2018):_
- Gerchberg-Saxton (GS) algorithm — alternating projections between object and Fourier planes (Gerchberg & Saxton, Optik 1972) — the foundational phase retrieval algorithm
- HIO — Hybrid Input-Output algorithm (Fienup, Appl. Opt. 1982) — improved convergence over GS with output constraint relaxation
- RAAR — Relaxed Averaged Alternating Reflections (Luke, Inverse Problems 2005) — relaxed version of HIO with convergence guarantees
- Error Reduction algorithm — strict constraint alternating projections (Fienup, 1982)
- Wirtinger Flow — gradient descent for phase retrieval (Candes et al., IEEE TIT 2015) — non-convex optimization with spectral initialization
- Phase Lift — semidefinite relaxation for phase retrieval (Candes et al., CPAM 2013) — convex relaxation approach
- Truncated Wirtinger Flow — TWF (Chen & Candes, 2017) — truncated gradient for robustness
- TIE — Transport of Intensity Equation phase retrieval (Teague, JOSA 1983) — deterministic phase retrieval from intensity derivatives
- Multi-height phase retrieval (Bao et al., Opt. Express 2015) — multiple defocus distances for robust phase recovery
- ADMM for holographic reconstruction (Chan et al., 2016)
- Compressive holography — CS-based reconstruction from subsampled holograms (Brady et al., Opt. Express 2009)

_Deep Learning (2017–2026):_
- PhaseNet — CNN for holographic phase retrieval (Sinha et al., Optica 2017) — first deep learning for holographic phase recovery; PSNR ~34.0 dB
- HoloNet — deep learning for holographic reconstruction and autofocusing (Wu et al., Nat. Methods 2019) — end-to-end holographic imaging
- prDeep — deep algorithm unrolling for phase retrieval (Metzler et al., NeurIPS 2018) — learned proximal gradient for phase retrieval
- Deep-DIH — deep learning for digital in-line holography (Rivenson et al., Light: Sci. Appl. 2018) — GAN-based holographic image enhancement
- eHoloNet — extended depth-of-field holographic reconstruction (Wu et al., 2020)
- U-PhaseNet — U-Net for holographic phase unwrapping (2020)
- DL-Phase — deep learning phase retrieval via optimization unrolling (Bostan et al., Optica 2020)
- Self-supervised holographic reconstruction (2021)
- Holographic diffusion model (2023)
- Physics-informed holographic reconstruction — PINN for holography (2022)
- Neural holography — differentiable holographic rendering (Peng et al., ACM TOG 2020)
- Transformer-based holographic reconstruction (2024)
- Foundation model for coherent imaging (2025)

#### Step 3: Update Holography Solvers

After listing all holography solvers, update `algorithm_base/holography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All holography solvers use the data format: `y` (H, W) intensity hologram (recorded interference pattern), `wavelength` illumination wavelength, `pixel_pitch` detector pixel size, `z` reconstruction distance. For off-axis: `y` contains carrier frequency. The `HolographyOperator` handles forward (Fresnel/Angular Spectrum propagation) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Holography:**
- USAF resolution target hologram: Angular Spectrum ~28.0 dB amplitude, Fresnel ~27.5 dB, GS ~30.0 dB (with support constraint), HIO ~31.0 dB, PhaseNet ~34.0 dB
- Bead/cell holograms: Angular Spectrum ~26.0 dB, prDeep ~32.0 dB, HoloNet ~33.5 dB
- Phase accuracy (rad): GS ~0.3 rad RMSE, HIO ~0.2 rad RMSE, PhaseNet ~0.1 rad RMSE
- All reference values from published papers and the PhaseNet/HoloNet benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'holography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/holography/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/holography/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/holography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/holography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for holography. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/holography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/holography/standard/`

---

### Ptychography (`ptychography`) Modality Template

Electron Ptychography. Carrier: Electron. DAG: M→P→D.

#### Step 1: Verify Standard Dataset

For Ptychography, what dataset do you use to verify? Is this dataset used for ptychography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ptychography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ptychography standard dataset.

**Popular datasets to consider:**
- **Zenodo 5113449 — SrTiO3 4D-STEM Ptychography Dataset (Jiang et al., 2021)** — 4D-STEM dataset of SrTiO3 crystal recorded at 300 kV, 128×128 scan positions with convergent-beam electron diffraction patterns; the primary open benchmark for electron ptychography; used for ePIE, WDD, PtychoNN validation
- **Ptycho.jl Simulation Benchmark (Odstrcil et al., 2016)** — simulated ptychographic datasets with known ground truth phase and amplitude; standard for algorithm development
- **Gold Nanoparticle Ptychography Dataset (Thibault et al., Science 2008)** — the original experimental X-ray ptychography dataset from the Diamond Light Source; used for PIE/ePIE validation
- **PtychoShelves Benchmark (Wakonig et al., J. Appl. Crystallogr. 2020)** — modular ptychography reconstruction benchmark with simulated and experimental data
- **cSAXS Ptychography Data (PSI, 2015–2023)** — X-ray ptychography datasets from the Swiss Light Source; widely used for benchmarking X-ray ptychography algorithms
- **NIST Electron Ptychography Standard (2022)** — standardized 4D-STEM dataset with calibrated specimen for quantitative phase evaluation

**Decision criteria:** Zenodo 5113449 (SrTiO3 4D-STEM) is the most widely used open benchmark for electron ptychography. Gold nanoparticle data for X-ray ptychography. Use the Zenodo SrTiO3 dataset as the standard evaluation set, consistent with the majority of recent electron ptychography papers.

#### Step 2: List All Ptychography Algorithms

Please first ensure all the ptychography algorithms have been listed in `\Physics_World_Model\algorithm_base\ptychography\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ptychography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ptychography solvers, please update the ptychography solver.

**Key algorithms to cover (1969–2026):**

_Classical / Iterative Phase Retrieval (1969–2010):_
- PIE — Ptychographic Iterative Engine (Rodenburg & Faulkner, Appl. Phys. Lett. 2004) — the foundational ptychographic reconstruction algorithm; single-probe constraint projection
- ePIE — extended Ptychographic Iterative Engine (Maiden & Rodenburg, Ultramicroscopy 2009) — simultaneously updates object and probe; the most widely used ptychography algorithm; ~30.0 dB phase reconstruction on SrTiO3
- rPIE — regularized PIE (Maiden et al., Ultramicroscopy 2017) — momentum and regularization for improved convergence
- DM — Difference Map for ptychography (Thibault et al., Science 2008) — constraint-based phase retrieval; first high-resolution X-ray ptychography
- WDD — Wigner Distribution Deconvolution (Rodenburg & Bates, Phil. Trans. R. Soc. A 1992) — direct (non-iterative) ptychographic reconstruction via 4D Fourier transform; fast but lower resolution
- ER — Error Reduction for ptychography (Fienup, 1982; adapted for ptychography)
- RAAR for ptychography — Relaxed Averaged Alternating Reflections (Luke, 2005; applied to ptychography)
- Oversampling smoothness — OSS (Rodriguez et al., J. Appl. Crystallogr. 2013)

_Optimization & Advanced Iterative (2010–2018):_
- ML-Ptychography — Maximum Likelihood ptychographic reconstruction (Thibault & Guizar-Sicairos, New J. Phys. 2012) — noise model-based optimization; handles Poisson statistics correctly
- LSQ-ML — Least-Squares Maximum Likelihood (Odstrčil et al., Opt. Express 2018) — GPU-accelerated ML ptychography
- Position correction ptychography — joint probe position and object reconstruction (Maiden et al., 2012; Beckers et al., 2013)
- Multi-slice ptychography — 3D object reconstruction from thick specimens (Maiden et al., JOSA A 2012)
- Blind ptychography — joint probe and object recovery without prior probe knowledge (Thibault et al., 2009)
- Mixed-state ptychography — partial coherence modeling (Thibault & Menzel, Nature 2013) — handles incoherent illumination
- ADMM for ptychography (Wen et al., 2012)
- Wirtinger Flow for ptychography (Xu et al., 2018)
- Automatic differentiation ptychography — AD-based gradient computation (2018)

_Deep Learning (2019–2026):_
- PtychoNN — Neural Network for ptychographic reconstruction (Cherukara et al., Appl. Phys. Lett. 2020) — direct CNN mapping from diffraction patterns to phase/amplitude; PSNR ~35.0 dB on SrTiO3; 100x faster than ePIE
- PtychoDV — Deep learning ptychography with variational inference (2021)
- AutoPhase — automated phase retrieval with deep learning (Nguyen et al., 2018)
- PtychoNet — U-Net for ptychographic phase retrieval (2020)
- Deep Ptychography — physics-informed deep learning for ptychography (Guzzi et al., 2022)
- Phaseless ptychography with generative models (2021)
- PtychoPINN — Physics-Informed Neural Network for ptychography (2023)
- Self-supervised ptychography — reconstruction without ground truth (2022)
- Diffusion-prior ptychography (2024)
- 4D-STEM deep learning reconstruction (Jiang et al., 2022)
- Transformer-based ptychography (2024)
- Foundation model for diffraction imaging (2025)

#### Step 3: Update Ptychography Solvers

After listing all ptychography solvers, update `algorithm_base/ptychography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ptychography solvers use the data format: `y` (N_pos, D, D) stack of N_pos diffraction patterns each of size D×D, `probe` (D, D) complex-valued illumination probe function, `positions` (N_pos, 2) scan positions. The `PtychographyOperator` handles forward `y_j = |F{P · O_j}|^2` (probe-object interaction + Fourier transform + intensity) and gradient operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Ptychography:**
- Zenodo SrTiO3 4D-STEM (128×128, 300 kV): WDD ~25.0 dB phase, ePIE ~30.0 dB phase / 0.92 SSIM, ML-ptychography ~32.0 dB, PtychoNN ~35.0 dB / 0.97 SSIM
- Gold nanoparticle X-ray ptychography: DM ~28.0 dB, ePIE ~31.0 dB, ML ~33.0 dB
- Phase RMSE (rad): WDD ~0.25 rad, ePIE ~0.12 rad, PtychoNN ~0.05 rad
- All reference values from published papers and the PtychoNN/ePIE benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ptychography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ptychography/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ptychography/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ptychography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ptychography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ptychography. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ptychography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ptychography/standard/`

---

### MRI (`mri`) Modality Template

Magnetic Resonance Imaging. Carrier: Nuclear Spin. DAG: M→F→S→D.

#### Step 1: Verify Standard Dataset

For MRI, what dataset do you use to verify? Is this dataset used for MRI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MRI standard dataset.

**Popular datasets to consider:**
- **fastMRI (NYU Langone, Zbontar et al., 2018)** — the most widely used MRI reconstruction benchmark; multi-coil brain and knee k-space data with ground-truth fully-sampled images; 4x and 8x acceleration; used by virtually all deep learning MRI papers since 2019
- **Calgary-Campinas Multi-Coil Dataset (Souza et al., 2018)** — 12-channel brain MRI raw k-space; used for parallel imaging and compressed sensing benchmarks
- **IXI Dataset (Imperial College London)** — 600 healthy brain MRI scans (T1, T2, PD-weighted); widely used for training and evaluation
- **BrainWeb Phantom (Collins et al., 1998)** — synthetic MRI brain phantom with known ground truth; the canonical numerical benchmark for MRI reconstruction since the 1990s
- **HCP (Human Connectome Project)** — high-resolution multi-modal brain MRI; used for advanced reconstruction evaluation
- **M4Raw (Lyu et al., NeurIPS 2023)** — multi-channel multi-contrast raw k-space dataset; emerging multi-contrast MRI benchmark
- **SKM-TEA (Stanford Knee MRI Multi-Task Evaluation, Desai et al., 2022)** — multi-task knee MRI benchmark with reconstruction and segmentation ground truth

**Decision criteria:** fastMRI is the undisputed gold standard for MRI reconstruction benchmarking (2019–2026); Calgary-Campinas for multi-coil parallel imaging. Use the dataset that appears in the largest number of MRI reconstruction papers.

#### Step 2: List All MRI Algorithms

Please first ensure all the MRI algorithms have been listed in `\Physics_World_Model\algorithm_base\mri\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MRI solvers, please update the MRI solver.

**Key algorithms to cover (1970–2026):**

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
- ESPIRiT — Eigenvalue-based sensitivity estimation for parallel imaging (Uecker et al., MRM 2014)
- L1-ESPIRiT — joint parallel imaging + compressed sensing (Uecker et al., MRM 2014)
- Low-Rank + Sparse (L+S) decomposition for dynamic MRI (Otazo et al., MRM 2015)
- k-t FOCUSS — k-t space compressed sensing for dynamic MRI (Jung et al., MRM 2009)
- BART toolbox methods — Berkeley Advanced Reconstruction Toolbox (Uecker et al., 2015)
- Total Variation CS-MRI (Block et al., MRM 2007)
- Dictionary Learning MRI (Ravishankar & Bresler, TMI 2011)
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

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mri/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MRI. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mri/standard/`

---

### CT (`ct`) Modality Template

X-ray Computed Tomography. Carrier: X-ray Photon. DAG: Pi→D.

#### Step 1: Verify Standard Dataset

For CT, what dataset do you use to verify? Is this dataset used for CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CT standard dataset.

**Popular datasets to consider:**
- **AAPM Mayo Clinic Low-Dose CT Grand Challenge (McCollough et al., 2016)** — the most widely used CT reconstruction benchmark; quarter-dose and full-dose paired data, 512×512, 10 patients; used by RED-CNN, FBPConvNet, LEARN, WGAN-VGG, CPCE, and virtually all low-dose CT papers
- **LoDoPaB-CT (Leuschner et al., Scientific Data 2021)** — large-scale low-dose parallel-beam CT benchmark from LIDC/IDRI; 362×362, 60-view sparse-view; emerging standard for sparse-view CT
- **FIPS Walnut micro-CT (Der Sarkissian et al., 2019)** — high-quality walnut micro-CT with 2D fan-beam and 3D cone-beam projections; industrial/scientific CT benchmark
- **HTC 2022 — Helsinki Tomography Challenge (Bubba et al., 2022)** — limited-angle CT challenge with real experimental data; benchmark for limited-angle reconstruction
- **LIDC-IDRI (Armato et al., 2011)** — lung CT screening dataset; raw projection data available for reconstruction
- **DeepLesion (Yan et al., JMRI 2018)** — 32K+ CT slices from NIH with lesion annotations

**Decision criteria:** The AAPM Mayo dataset is the gold standard for low-dose CT denoising/reconstruction. LoDoPaB-CT is the gold standard for sparse-view CT. Use the dataset that appears in the largest number of CT reconstruction papers (2017–2026).

#### Step 2: List All CT Algorithms

Please first ensure all the CT algorithms have been listed in `\Physics_World_Model\algorithm_base\ct\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CT solvers, please update the CT solver.

**Key algorithms to cover (1960–2026):**

_Analytic / Classical (1960s–2000s):_
- FBP with Ram-Lak filter (Ramachandran & Lakshminarayanan, 1971) — the foundational CT reconstruction algorithm
- FBP with Shepp-Logan filter (Shepp & Logan, 1974)
- FBP with Hamming/Hann/Cosine windows — smoothed FBP variants
- ART — Algebraic Reconstruction Technique (Gordon et al., 1970)
- SIRT — Simultaneous Iterative Reconstruction Technique (Gilbert, 1972)
- SART — Simultaneous Algebraic Reconstruction Technique (Andersen & Kak, 1984)
- MLEM — Maximum Likelihood Expectation Maximization (Shepp & Vardi, 1982)
- OSEM — Ordered Subsets EM (Hudson & Larkin, 1994)
- FDK — Feldkamp-Davis-Kress for cone-beam CT (Feldkamp et al., 1984)

_Regularized / Optimization (2000s–2016):_
- TV regularization for CT (Sidky et al., PMB 2006; Sidky & Pan, 2008) — total variation minimization for sparse-view CT; ~30.0 dB on LoDoPaB 60-view
- TGV — Total Generalized Variation (Bredies et al., 2010)
- ADMM for CT (Boyd et al., 2011)
- FISTA for CT (Beck & Teboulle, 2009)
- Split Bregman for CT (Goldstein & Osher, 2009)
- Dictionary Learning CT (Xu et al., TMI 2012)
- PWLS — Penalized Weighted Least Squares (Fessler, 2000)
- MBIR — Model-Based Iterative Reconstruction (Thibault et al., Med Phys 2012)
- Chambolle-Pock / PDHG (Chambolle & Pock, 2011)
- PICCS — Prior Image Constrained Compressed Sensing (Chen et al., 2008)
- RED for CT — Regularization by Denoising (Romano et al., 2016)

_Deep Learning (2016–2026):_
- RED-CNN — Residual Encoder-Decoder CNN (Chen et al., TMI 2017)
- FBPConvNet — FBP followed by CNN post-processing (Jin et al., TIP 2017)
- LEARN — Learned Experts' Assessment-based Reconstruction Network (Chen et al., TMI 2018)
- Learned Primal-Dual (Adler & Oktem, TMI 2018) — deep unfolding of primal-dual optimization; ~35.5 dB on LoDoPaB 60-view
- WGAN-VGG for low-dose CT (Yang et al., TMI 2018)
- CPCE — Competitive Pathways CNN Ensemble (Shan et al., TMI 2019)
- DuDoNet — Dual Domain Network (Lin et al., TMI 2019)
- DuDoTrans — Dual Domain Transformer (Wang et al., MICCAI 2022)
- Score-CT / DiffusionMBIR (Song et al., ICLR 2022; Chung et al., 2023) — diffusion-based CT reconstruction; ~37.0 dB on LoDoPaB 60-view
- DOLCE — Diffusion posterior sampling for CT (Liu et al., 2023)
- CTformer — Transformer for low-dose CT (Wang et al., TMI 2023)
- PnP-ADMM / PnP-HQS with BM3D/DnCNN for CT
- Neural Attenuation Fields for sparse-view CT (2023)
- FreeSeed (Chen et al., MICCAI 2024)
- Foundation model for CT reconstruction (2025)

#### Step 3: Update CT Solvers

After listing all CT solvers, update `algorithm_base/ct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CT solvers use the data format: `y` (num_angles, num_detectors) sinogram data, `angles` array of projection angles in radians. The `CTOperator` handles the forward model (Radon transform) and adjoint (filtered backprojection) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CT:**
- AAPM Mayo quarter-dose: RED-CNN ~33.5 dB, WGAN-VGG ~34.0 dB, CTformer ~35.5 dB
- LoDoPaB-CT sparse-view (60 views): FBP ~21.0 dB / 0.45 SSIM, TV ~30.0 dB / 0.87 SSIM, Learned Primal-Dual ~35.5 dB / 0.95 SSIM, DiffusionMBIR ~37.0 dB / 0.97 SSIM
- LoDoPaB-CT challenge leaderboard top results
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ct/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ct/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CT. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ct/standard/`

---

### CBCT (`cbct`) Modality Template

Cone-Beam CT. Carrier: X-ray Photon. DAG: Pi→D (cone-beam geometry).

#### Step 1: Verify Standard Dataset

For CBCT, what dataset do you use to verify? Is this dataset used for CBCT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cbct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CBCT standard dataset.

**Popular datasets to consider:**
- **LoDoPaB-CT (resized for cone-beam, Leuschner et al., Scientific Data 2021)** — adapted from the parallel-beam LoDoPaB for cone-beam geometry evaluation; provides large-scale benchmark with ground truth
- **FIPS Walnut micro-CT (Der Sarkissian et al., 2019)** — walnut cone-beam micro-CT with full 3D projection data and 3D ground-truth volume; the primary open CBCT benchmark for algorithm development
- **HTC 2022 — Helsinki Tomography Challenge (Bubba et al., 2022)** — limited-angle CT challenge with real cone-beam data; benchmark for limited-angle CBCT
- **AAPM CBCT Challenge Data (2020)** — clinical cone-beam CT data for image-guided radiation therapy with paired planning CT ground truth
- **CatPhan Phantom CBCT Scans** — standardized phantom scans for CBCT quality assurance; known geometry and contrast inserts
- **TIGRE Simulated CBCT (Biguri et al., 2016)** — GPU-accelerated CBCT simulation toolkit with configurable geometry; widely used for algorithm development

**Decision criteria:** FIPS walnut is the most widely used open CBCT benchmark with full 3D data. AAPM CBCT for clinical applications. TIGRE simulated data for controlled algorithm evaluation. Use the dataset most widely referenced in CBCT reconstruction papers (2016–2026).

#### Step 2: List All CBCT Algorithms

Please first ensure all the CBCT algorithms have been listed in `\Physics_World_Model\algorithm_base\cbct\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cbct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CBCT solvers, please update the CBCT solver.

**Key algorithms to cover (1984–2026):**

_Analytic / Classical (1984–2005):_
- FDK — Feldkamp-Davis-Kress algorithm (Feldkamp et al., JOSA A 1984) — the foundational cone-beam CT reconstruction; approximate filtered backprojection for circular trajectory; PSNR ~22.0 dB on sparse-view CBCT
- Katsevich exact algorithm — exact reconstruction for helical cone-beam (Katsevich, SIAM J. Appl. Math. 2002)
- Grangeat formula — cone-beam to parallel-beam conversion (Grangeat, 1991)
- FBP for cone-beam with Parker weighting (Parker, Med. Phys. 1982)
- Wang's half-scan FDK — short-scan cone-beam reconstruction (Wang, 1993)
- Approximate Katsevich for circular trajectory (2005)

_Iterative / Optimization (2006–2018):_
- SART for CBCT — Simultaneous Algebraic Reconstruction Technique for cone-beam (Andersen & Kak, 1984; cone-beam extension) — PSNR ~28.0 dB on sparse-view
- OSEM for CBCT — Ordered Subsets EM for 3D cone-beam (Hudson & Larkin, 1994; 3D extension)
- TV-CBCT — Total Variation regularized CBCT reconstruction (Sidky et al., PMB 2006; applied to cone-beam)
- TIGRE toolkit algorithms — GPU-accelerated iterative CBCT (Biguri et al., Biomed. Phys. Eng. Express 2016) — includes FDK, SART, OS-SART, ASD-POCS, CGLS, FISTA
- ASTRA Toolbox algorithms — GPU-based CT/CBCT reconstruction (van Aarle et al., Opt. Express 2016) — includes FBP, SIRT, CGLS for cone-beam geometry
- ASD-POCS — Adaptive Steepest Descent POCS for CBCT (Sidky et al., 2008)
- CGLS — Conjugate Gradient Least Squares for CBCT (Hestenes & Stiefel, 1952; applied to CBCT)
- PnP-CBCT — Plug-and-Play priors for CBCT (2018)
- Scatter correction for CBCT — Monte Carlo and kernel-based methods (Siewerdsen & Jaffray, 2001; Niu et al., 2010)
- MBIR for CBCT — Model-Based Iterative Reconstruction (2012)

_Deep Learning (2018–2026):_
- DL-CBCT — deep learning CBCT reconstruction (Shan et al., 2019) — CNN-based; PSNR ~33.0 dB on sparse-view
- FDKConvNet — FDK followed by CNN post-processing for CBCT (Wurfl et al., MICCAI 2018)
- AUTOMAP-CBCT — learned reconstruction for cone-beam geometry (2019)
- Learned Primal-Dual for CBCT (Adler & Oktem, 2018; extended to 3D cone-beam)
- NAF — Neural Attenuation Fields for CBCT (Zha et al., 2022) — NeRF-inspired volumetric CBCT reconstruction
- DiffusionCBCT — diffusion-based CBCT reconstruction (2023)
- CBCT-Net — 3D U-Net for CBCT artifact correction (2020)
- Scatter-Net — deep learning scatter correction for CBCT (Maier et al., Med. Phys. 2019)
- IntraTomo — self-supervised CBCT from single scan (2022)
- CBCT-NeRF — Neural Radiance Fields for CBCT (2023)
- Transformer-based CBCT reconstruction (2024)
- Foundation model for 3D CT/CBCT (2025)

#### Step 3: Update CBCT Solvers

After listing all CBCT solvers, update `algorithm_base/cbct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CBCT solvers use the data format: `y` (num_angles, num_rows, num_cols) 2D projection images at each angle (cone-beam geometry), `angles` array of projection angles, `geometry` dict containing source-to-detector distance, source-to-object distance, detector pixel size. The `CBCTOperator` handles the forward model (cone-beam projection) and adjoint (FDK-type backprojection) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CBCT:**
- FIPS walnut cone-beam: FDK ~22.0 dB / 0.55 SSIM, SART ~28.0 dB / 0.82 SSIM, TV-CBCT ~30.5 dB / 0.88 SSIM, DL-CBCT ~33.0 dB / 0.93 SSIM
- TIGRE simulated sparse-view (60 angles): FDK ~20.0 dB, SART ~26.0 dB, TIGRE-FISTA ~29.0 dB
- Clinical CBCT: FDK with scatter correction ~25.0 dB, DL-CBCT ~31.0 dB
- All reference values from published papers and the TIGRE/ASTRA benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cbct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cbct/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cbct/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cbct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cbct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CBCT. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cbct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cbct/standard/`

---

### Ultrasound (`ultrasound`) Modality Template

B-mode Ultrasound Imaging. Carrier: Acoustic Wave. DAG: M→F→S→D.

#### Step 1: Verify Standard Dataset

For Ultrasound, what dataset do you use to verify? Is this dataset used for ultrasound popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ultrasound/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ultrasound standard dataset.

**Popular datasets to consider:**
- **PICMUS — Plane-wave Imaging Challenge in Medical UltraSound (Liebgott et al., IEEE IUS 2016)** — the primary benchmark for ultrasound beamforming algorithms; includes resolution phantom, contrast phantom, carotid cross-section, and carotid longitudinal datasets; Scenario I (simulation) and Scenario II (experimental); used by DAS, MV, DMAS, ADMIRE, and virtually all beamforming papers since 2016
- **DeepUS CIRS-040GSE (Luijten et al., 2020)** — deep learning ultrasound dataset captured with a CIRS 040GSE multipurpose phantom and Verasonics Vantage 256; paired plane-wave data with reference compound images; used for DL beamforming benchmarks
- **CUBDL — Challenge in Ultrasound Beamforming with Deep Learning (Bell et al., IEEE TMI 2020)** — standardized deep learning beamforming benchmark with simulated and in-vivo data
- **IUS Simulation Benchmark (Jensen, Field II)** — Field II-simulated point targets, cyst phantoms, and tissue-mimicking phantoms; the canonical simulation benchmark for ultrasound beamforming
- **CIRS Phantom Experimental Data** — CIRS tissue-mimicking phantom scans from multiple research groups; standard for quantitative evaluation
- **Plane Wave Ultrasound Datasets (Rindal et al., IEEE TUFFC 2019)** — multi-angle plane-wave data for coherent compounding benchmarks

**Decision criteria:** PICMUS is the undisputed gold standard for ultrasound beamforming algorithm evaluation (2016–2026). CUBDL for deep learning methods. Use PICMUS as the standard evaluation set, consistent with the majority of published ultrasound beamforming papers.

#### Step 2: List All Ultrasound Algorithms

Please first ensure all the ultrasound algorithms have been listed in `\Physics_World_Model\algorithm_base\ultrasound\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ultrasound. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ultrasound solvers, please update the ultrasound solver.

**Key algorithms to cover (1960–2026):**

_Classical Beamforming (1960–2005):_
- DAS — Delay-and-Sum beamforming (the foundational ultrasound beamforming algorithm; standard in all clinical scanners)
- Dynamic Receive Focusing — depth-dependent receive delay profiles (1970s–1980s)
- Coherent Plane-Wave DAS — plane-wave compounding with delay-and-sum (Montaldo et al., IEEE TUFFC 2009; roots in 1990s) — baseline for plane-wave imaging
- SA-DAS — Synthetic Aperture DAS (Jensen et al., 1992)
- Phased Array Beamforming — focused transmit/receive (1990s)

_Adaptive & Advanced Beamforming (2005–2018):_
- MV — Minimum Variance (Capon) beamforming for ultrasound (Synnevag et al., IEEE TUFFC 2007; Capon, 1969) — adaptive beamforming with improved resolution
- DMAS — Delay-Multiply-and-Sum (Matrone et al., IEEE TMI 2015) — multiplicative beamforming for improved contrast
- GCF — Generalized Coherence Factor weighted DAS (Li & Li, IEEE TUFFC 2003)
- CF — Coherence Factor beamforming (Mallart & Fink, JASA 1994)
- PCF — Phase Coherence Factor (Camacho et al., IEEE TUFFC 2009)
- ADMIRE — Aperture Domain Model Image Reconstruction (Byram et al., IEEE TUFFC 2015) — model-based beamforming; PSNR ~36.0 dB on PICMUS Scenario I
- iMAP — iterative MAP beamforming (Rindal et al., 2017)
- Eigenspace-based MV beamforming (Asl & Mahloojifar, Ultrasonics 2012)
- Coherent Flow Power Doppler (2010)
- Filtered-delay multiply and sum — F-DMAS (Prieur et al., IEEE TUFFC 2018)
- Short-Lag Spatial Coherence imaging — SLSC (Lediju et al., IEEE TUFFC 2011)

_Deep Learning (2018–2026):_
- DL-Beamforming — deep learning-based beamforming (Luijten et al., IEEE TUFFC 2020) — learned end-to-end beamforming from raw channel data
- ABLE — Adaptive Beamforming using Deep Learning (Luchies & Byram, IEEE TMI 2018) — CNN-based aperture weighting
- DeepUS — deep neural network for ultrafast ultrasound imaging (2020)
- IQ-Net — deep learning for IQ data to B-mode (2019)
- UBF-Net — Ultrasound Beamforming Network (Khan et al., 2020)
- Plane-wave deep learning compounding (Gasse et al., IEEE TUFFC 2017)
- DNN-based clutter suppression for ultrasound (2019)
- US-diffusion — diffusion model for ultrasound image enhancement (2023)
- Transformer-based ultrasound beamforming (2024)
- Self-supervised ultrasound reconstruction (2022)
- Physics-informed neural beamforming (2023)
- Foundation model for ultrasound imaging (2025)

#### Step 3: Update Ultrasound Solvers

After listing all ultrasound solvers, update `algorithm_base/ultrasound/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ultrasound solvers use the data format: `y` (num_elements, num_samples, num_angles) raw RF/IQ channel data from plane-wave transmissions, `sound_speed` (m/s), `probe_geometry` dict with element positions, pitch, and frequency. The `UltrasoundOperator` handles forward (wave propagation + receive) and adjoint (DAS-type backpropagation) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Ultrasound:**
- PICMUS Scenario I (simulation, resolution phantom): PW-DAS (1 angle) ~28.0 dB, PW-DAS (75 angles) ~33.0 dB, MV ~34.5 dB, DMAS ~34.0 dB, ADMIRE ~36.0 dB, DL-Beamforming ~36.5 dB
- PICMUS Scenario I contrast phantom: PW-DAS CNR ~15 dB, MV CNR ~18 dB, ADMIRE CNR ~22 dB
- PICMUS Scenario II (experimental, carotid): qualitative comparison + CNR/resolution metrics
- CUBDL challenge leaderboard
- All reference values from published papers and PICMUS/CUBDL benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ultrasound' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ultrasound/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ultrasound/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ultrasound/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ultrasound/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ultrasound. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ultrasound/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasound/standard/`

---

### Cryo-EM (`cryo_em`) Modality Template

Cryo-Electron Microscopy (Single Particle Analysis). Carrier: Electron. DAG: M→P→D.

#### Step 1: Verify Standard Dataset

For Cryo-EM, what dataset do you use to verify? Is this dataset used for cryo-EM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cryo_em/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original cryo-EM standard dataset.

**Popular datasets to consider:**
- **EMDB Benchmark Entries** — canonical Electron Microscopy Data Bank entries used universally for method validation:
  - **EMD-5778 (beta-galactosidase, Campbell et al., 2015)** — 2.2 A resolution; the primary cryo-EM benchmark structure
  - **EMD-2984 (beta-galactosidase, Bartesaghi et al., 2015)** — 2.2 A; alternative reconstruction of the same protein
  - **EMD-6287 (TRPV1, Liao et al., Nature 2013)** — 3.4 A resolution; the landmark cryo-EM structure that launched the resolution revolution
  - **EMD-11103 (apoferritin, Yip et al., Nature 2020)** — 1.2 A resolution; ultra-high resolution benchmark
  - **EMD-21375 (aldolase, Herzik et al., Nat. Methods 2019)** — 1.8 A; small protein benchmark
- **EMPIAR Benchmark Datasets (Iudin et al., 2016)** — Electron Microscopy Public Image Archive; raw micrograph data for several EMDB entries; provides raw data for end-to-end reconstruction benchmarking
- **Synthetic Cryo-EM Datasets (Gupta et al., 2020)** — simulated projections with known 3D ground truth, controlled SNR, and CTF parameters; standard for algorithm development

**Decision criteria:** EMDB entries (EMD-5778, EMD-6287, EMD-11103) are universally used benchmarks for cryo-EM reconstruction. EMPIAR raw data enables end-to-end evaluation. Use the datasets most widely referenced in cryo-EM reconstruction papers (2013–2026).

#### Step 2: List All Cryo-EM Algorithms

Please first ensure all the cryo-EM algorithms have been listed in `\Physics_World_Model\algorithm_base\cryo_em\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cryo_em. Besides, you need to search all algorithms from 1950 to 2026. After listing all the cryo-EM solvers, please update the cryo-EM solver.

**Key algorithms to cover (1968–2026):**

_Classical / Fourier Methods (1968–2010):_
- Wiener filter for cryo-EM — CTF-corrected Fourier inversion (the baseline cryo-EM reconstruction; always reported); PSNR ~18.5 dB on typical datasets
- Common-lines method — angular assignment from projection common lines (Crowther, 1970; Van Heel, 1987) — the foundational 3D reconstruction approach
- Weighted back-projection (WBP) — Fourier-weighted backprojection for 3D reconstruction (Radermacher, 1988)
- Direct Fourier inversion — gridding reconstruction in 3D Fourier space (1988)
- Random conical tilt — two-tilt angular assignment (Radermacher et al., 1987)
- CTFFIND — CTF estimation from micrograph power spectra (Mindell & Grigorieff, J. Struct. Biol. 2003) — the standard CTF estimation tool
- GCTF — GPU-accelerated CTF estimation (Zhang, J. Struct. Biol. 2016)

_Iterative / Maximum Likelihood (2005–2018):_
- RELION — REgularized LIkelihood OptimizatioN (Scheres, J. Struct. Biol. 2012) — the most widely used cryo-EM reconstruction software; Bayesian approach with regularized likelihood; RELION 3.0 ~25.0 dB
- cryoSPARC — cryo-EM Single Particle Ab initio Reconstruction and Classification (Punjani et al., Nat. Methods 2017) — stochastic gradient descent-based; fast ab initio and refinement; ~27.0 dB
- FREALIGN / cisTEM — Fourier REconstruction and ALIGNment / computational imaging system for TEM (Grigorieff, 2007; Grant et al., eLife 2018) — template matching and refinement
- EMAN2 — comprehensive cryo-EM processing suite (Tang et al., J. Struct. Biol. 2007)
- SPIDER — System for Processing of Image Data from Electron microscopy and Related fields (Frank et al., 1996)
- SIMPLE — iterative projection matching refinement (Elmlund & Elmlund, 2012)
- 3D classification — heterogeneity analysis via 3D classification (Scheres, 2012)
- Multibody refinement — flexible fitting of conformational states (Nakane et al., eLife 2018)
- Bayesian polishing — per-particle motion correction (Zivanov et al., eLife 2019)
- Ewald sphere correction for high-resolution reconstruction (DeRosier, 2000; Russo & Henderson, 2018)

_Deep Learning (2019–2026):_
- CryoDRGN — Deep Reconstructing Generative Networks for cryo-EM (Zhong et al., ICLR 2020) — variational autoencoder for continuous heterogeneity; landmark deep learning cryo-EM paper
- DeepEMhancer — deep learning map sharpening and denoising (Sanchez-Garcia et al., Commun. Biol. 2021) — post-processing enhancement of cryo-EM maps
- Topaz — deep learning particle picking (Bepler et al., Nat. Methods 2019) — CNN-based positive-unlabeled particle detection
- crYOLO — You Only Look Once for cryo-EM (Wagner et al., Commun. Biol. 2019) — YOLO-based real-time particle picking
- CryoAI — amortized inference for cryo-EM pose estimation (Levy et al., 2022)
- CryoPoseNet — CNN for orientation estimation (Nashed et al., 2021)
- E2GMM — Gaussian Mixture Model for cryo-EM heterogeneity (Chen et al., 2022)
- DynaMight — dynamic cryo-EM reconstruction (Schwab et al., 2023)
- 3DFlex — 3D flexible refinement in RELION (Punjani & Fleet, 2023)
- ModelAngelo — automated atomic model building (Jamali et al., Nature 2024)
- AlphaFold-guided cryo-EM refinement (2023)
- Diffusion-based cryo-EM denoising (2024)
- Foundation model for cryo-EM (2025)

#### Step 3: Update Cryo-EM Solvers

After listing all cryo-EM solvers, update `algorithm_base/cryo_em/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All cryo-EM solvers use the data format: `y` (N, D, D) stack of N 2D projection images (particle images) each of size D×D, `ctf_params` (N, K) per-particle CTF parameters (defocus, astigmatism, B-factor), `orientations` (N, 3) Euler angles (if known; estimated otherwise). The `CryoEMOperator` handles forward (3D projection + CTF modulation + noise) and adjoint (backprojection) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Cryo-EM:**
- EMD-5778 (beta-galactosidase): Wiener ~18.5 dB, RELION 3.0 ~25.0 dB / 2.2 A resolution, cryoSPARC ~27.0 dB / 2.0 A resolution
- EMD-11103 (apoferritin): RELION 4.0 ~28.0 dB / 1.2 A, cryoSPARC ~29.0 dB
- EMD-6287 (TRPV1): RELION ~23.0 dB / 3.4 A, cryoSPARC ~24.5 dB / 3.0 A
- Resolution metrics: FSC 0.143 criterion for gold-standard resolution
- DeepEMhancer: 2–4 dB improvement over raw maps
- All reference values from EMDB deposited maps and published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cryo_em' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cryo_em/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cryo_em/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for cryo-EM. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cryo_em/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`

---

### Widefield Fluorescence (`widefield`) Modality Template

Widefield Fluorescence Microscopy Deconvolution. Carrier: Photon. DAG: M→Sigma→D.

#### Step 1: Verify Standard Dataset

For Widefield Fluorescence, what dataset do you use to verify? Is this dataset used for widefield fluorescence popular algorithms? Please ensure the standard dataset in `datasets/benchmark/widefield/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original widefield fluorescence standard dataset.

**Popular datasets to consider:**
- **Synthetic Phantoms (puncta, filaments, nuclei, membranes, mixed) — 64×64** — simulated widefield fluorescence microscopy images with known ground truth; configurable PSF, noise level, and structure type; the primary quantitative benchmark for widefield deconvolution and denoising algorithms
- **ISBI Deconvolution Grand Challenge (Sage et al., 2017)** — 2D and 3D fluorescence microscopy images with known PSFs and ground truth; used for benchmarking deconvolution algorithms since 2013
- **Fluorescence Microscopy Denoising Dataset (FMDD, Zhang et al., 2019)** — paired noisy/clean widefield fluorescence microscopy images of various structures; used for denoising benchmarks
- **BioImage Model Zoo / ZeroCostDL4Mic Datasets (von Chamier et al., 2021)** — community-curated widefield microscopy images for restoration; paired low-SNR and high-SNR images for training and evaluation
- **Hagen et al. Widefield Benchmark (2021)** — widefield fluorescence datasets with calibrated PSFs and multiple noise levels for systematic evaluation
- **W2S — Widefield-to-SIM Dataset (Qiao et al., Nat. Methods 2021)** — paired widefield and structured illumination microscopy images for super-resolution benchmarking
- **Fluorescent Bead Datasets** — standardized fluorescent microsphere samples for PSF calibration and deconvolution validation

**Decision criteria:** Synthetic phantoms with known ground truth are the gold standard for quantitative evaluation of widefield deconvolution. ISBI Deconvolution Challenge for community comparison. FMDD for deep learning denoising. Use the dataset that appears in the largest number of widefield fluorescence restoration papers (2013–2026).

#### Step 2: List All Widefield Fluorescence Algorithms

Please first ensure all the widefield fluorescence algorithms have been listed in `\Physics_World_Model\algorithm_base\widefield\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/widefield. Besides, you need to search all algorithms from 1950 to 2026. After listing all the widefield fluorescence solvers, please update the widefield fluorescence solver.

**Key algorithms to cover (1949–2026):**

_Classical / Analytic (1949–2005):_
- Wiener filter — regularized inverse filter for fluorescence microscopy (Wiener, 1949; applied to microscopy by Hiraoka et al., 1990) — baseline; PSNR ~22.0 dB on synthetic phantoms
- Inverse filter — naive Fourier-domain OTF division (demonstrates noise amplification and ringing)
- Tikhonov-regularized deconvolution — L2-penalized PSF inversion (1980s)
- Gold's ratio method — multiplicative iterative deconvolution (Gold, 1964)
- Nearest-neighbor deconvolution — subtract scaled adjacent planes for out-of-focus removal (Agard, 1984)

_Iterative / Optimization (1972–2018):_
- Richardson-Lucy (RL) deconvolution — iterative ML for Poisson noise (Richardson, 1972; Lucy, 1974) — the most widely used fluorescence deconvolution algorithm; PSNR ~27.0 dB on synthetic phantoms
- Accelerated RL — Biggs-Andrews acceleration (Biggs & Andrews, J. Opt. Soc. Am. A 1997)
- ADMM-TV — Alternating Direction Method of Multipliers with Total Variation for fluorescence deconvolution (Boyd et al., 2011; applied to microscopy) — ~25.0 dB
- FISTA for fluorescence — Fast Iterative Shrinkage-Thresholding with wavelet sparsity (Beck & Teboulle, 2009)
- BM3D for fluorescence denoising — Block-Matching and 3D filtering (Dabov et al., TIP 2007) — the gold standard non-local denoising; ~26.0 dB
- PnP-PGD — Plug-and-Play Proximal Gradient Descent with BM3D/DnCNN denoiser for fluorescence (Romano et al., 2016; applied to microscopy) — ~27.5 dB
- MAP with Gaussian/Poisson-Gaussian noise model (2010)
- Blind deconvolution for fluorescence — joint PSF and object estimation (Lam & Bhatt, 2000)
- Sparse deconvolution — L1-penalized deconvolution for punctate structures (Mukamel et al., 2009)
- Regularized RL with TV penalty (Dey et al., 2006)
- Half-quadratic splitting for fluorescence deconvolution (2015)
- Total Generalized Variation deconvolution (Bredies et al., 2010; applied to microscopy 2015)

_Deep Learning (2017–2026):_
- CARE — Content-Aware Image Restoration (Weigert et al., Nat. Methods 2018) — the seminal deep learning microscopy restoration paper; U-Net trained on paired low/high SNR images; PSNR ~33.0 dB on synthetic phantoms
- Noise2Void — self-supervised denoising without clean ground truth (Krull et al., CVPR 2019) — blind-spot training for fluorescence; ~29.0 dB
- Noise2Self — self-supervised denoising via J-invariance (Batson & Royer, ICML 2019) — ~28.5 dB
- DnCNN for fluorescence — Denoising CNN (Zhang et al., TIP 2017; applied to microscopy)
- Noise2Noise — learning from noisy pairs (Lehtinen et al., ICML 2018)
- CSBDeep — deep learning toolbox for fluorescence microscopy (Weigert et al., 2018)
- DeepCAD — deep self-supervised learning for calcium imaging denoising (Li et al., Nat. Methods 2021)
- RCAN — Residual Channel Attention Network for microscopy (Chen et al., Nat. Methods 2021)
- Probabilistic Noise2Void — pN2V (Krull et al., Front. Comput. Sci. 2020)
- DivNoising — diversity-promoting denoising (Prakash et al., ICML 2021)
- HDN — Hierarchical Disentangled Network for microscopy denoising (Prakash et al., 2022)
- Structured N2V — Noise2Void with structured noise model (Broaddus et al., 2020)
- Diffusion-based fluorescence denoising (2023)
- ViT-based microscopy restoration — Vision Transformer for fluorescence (2024)
- Foundation model for microscopy image restoration (2025)

#### Step 3: Update Widefield Fluorescence Solvers

After listing all widefield fluorescence solvers, update `algorithm_base/widefield/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All widefield fluorescence solvers use the data format: `y` (H, W) or (H, W, C) noisy blurred widefield fluorescence image, `psf` (H, W) point spread function (Gaussian or measured Airy disk pattern), `noise_params` dict with Poisson and Gaussian noise parameters. The `WidefieldOperator` handles forward `y = Poisson(PSF * x) + Gaussian_noise` and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Widefield Fluorescence:**
- Synthetic phantoms (64×64, mixed structures, SNR ~10): Wiener ~22.0 dB / 0.60 SSIM, RL ~27.0 dB / 0.78 SSIM, ADMM-TV ~25.0 dB / 0.72 SSIM, BM3D ~26.0 dB / 0.76 SSIM, PnP-PGD ~27.5 dB / 0.80 SSIM, Noise2Void ~29.0 dB / 0.85 SSIM, CARE ~33.0 dB / 0.93 SSIM
- Synthetic puncta phantom: Wiener ~24.0 dB, RL ~30.0 dB, CARE ~36.0 dB
- Synthetic filament phantom: Wiener ~20.0 dB, RL ~25.0 dB, CARE ~31.0 dB
- ISBI Deconvolution Challenge: published leaderboard results
- All reference values from published papers and the CARE/Noise2Void benchmark tables

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'widefield' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/widefield/`.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/widefield/`

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/widefield/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/widefield/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for widefield fluorescence. You keep the most popular dataset for local and GCS.

**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/standard/`

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/widefield/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/widefield/standard/`
