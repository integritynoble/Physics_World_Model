
---

## Electron Microscopy (Additional) — Modality Templates

---

### Cryo-EM (`cryo_em`) Modality Template

#### Step 1: Verify Standard Dataset

For Cryo-EM, what dataset do you use to verify? Is this dataset used for Cryo-EM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cryo_em/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Cryo-EM standard dataset.

**Popular datasets to consider:**
- **EMPIAR Archive (Iudin et al., 2016, updated continuously)** — the Electron Microscopy Public Image Archive; the central repository for raw cryo-EM micrographs and particle stacks; hosts benchmark entries for virtually all single-particle reconstruction methods
- **EMDB (Electron Microscopy Data Bank)** — the primary repository for cryo-EM 3D density maps; provides reference reconstructions for resolution validation
- **RELION Benchmark Datasets** — apoferritin (EMPIAR-10146), beta-galactosidase (EMPIAR-10061), TRPV1 (EMPIAR-10005); the canonical test cases for single-particle cryo-EM pipelines since 2012; used to validate RELION, CryoSPARC, and virtually all new methods
- **Benchmark Initiative Datasets (Henderson et al., 2019)** — community-curated benchmark with well-characterized specimens at known resolutions; designed for systematic comparison of cryo-EM processing workflows
- **T20S Proteasome (EMPIAR-10025)** — high-symmetry test case (D7) widely used for benchmarking CTF estimation and high-resolution refinement
- **Spliceosome (EMPIAR-10180)** — conformationally heterogeneous complex; standard test for 3D classification and variability analysis

**Decision criteria:** RELION benchmark datasets (apoferritin, beta-galactosidase, TRPV1) are the undisputed gold standards for single-particle cryo-EM benchmarking (2012-2026). Apoferritin for high-resolution validation, TRPV1 for general workflow validation. Use the dataset that appears in the largest number of cryo-EM reconstruction papers.

#### Step 2: List All Cryo-EM Algorithms

Please first ensure all the Cryo-EM algorithms have been listed in `\pwm\public\algorithm_base\cryo_em\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cryo_em. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Cryo-EM solvers, please update the Cryo-EM solver.

**Key algorithms to cover (1968-2026):**

_CTF Estimation & Correction (1968-2020):_
- Thon ring fitting — original CTF estimation from power spectra (Thon, 1968)
- CTFFIND — automated CTF estimation from micrographs (Mindell & Grigorieff, JSB 2003; CTFFIND4, Rohou & Grigorieff, JSB 2015) — the most widely used CTF tool
- Gctf — GPU-accelerated CTF estimation with per-particle refinement (Zhang, JSB 2016)
- EMAN2 e2ctf — CTF fitting integrated with EMAN2 pipeline (Tang et al., JSB 2007)
- CTF phase-flip correction, Wiener filtering, and amplitude correction methods

_Particle Picking (2004-2026):_
- Template matching — cross-correlation-based particle detection (Roseman, JSB 2004)
- DoG picker — Difference-of-Gaussians particle picking (Voss et al., JSB 2009)
- Topaz — deep learning particle picking with positive-unlabeled learning (Bepler et al., Nat Methods 2019) — state-of-the-art for challenging datasets
- crYOLO — YOLO-based automated particle picking (Wagner et al., Commun Biol 2019)
- CryoSPARC blob picker — template-free blob detection (Punjani et al., 2017)
- SPHIRE EMAN2 Gauss picker — Gaussian-based particle detection
- DeepPicker — CNN-based particle picking (Wang et al., JSB 2016)
- CryoTransformer — transformer-based particle picking (2024)

_2D Classification (2003-2026):_
- ISAC — Iterative Stable Alignment and Clustering (Yang et al., Structure 2012) — reference-free 2D classification with reproducibility guarantee
- RELION 2D classification — Bayesian 2D classification (Scheres, JSB 2012)
- CryoSPARC 2D classification — stochastic gradient descent 2D averaging (Punjani et al., Nat Methods 2017)
- EMAN2 e2refine2d — iterative 2D refinement
- XMIPP CL2D — clustering-based 2D classification (Sorzano et al., 2010)
- Topaz-Denoise for 2D class enhancement (Bepler et al., 2020)

_3D Reconstruction — Ab Initio (1987-2026):_
- Common lines / angular reconstitution (Van Heel, 1987; Penczek et al., 1996)
- Random conical tilt (Radermacher et al., JSB 1987)
- CryoSPARC ab initio — stochastic gradient descent initial volume estimation (Punjani et al., 2017)
- RELION 3D initial model — gradient-driven ab initio (Scheres, 2016)
- EMAN2 initial model — e2initialmodel (Tang et al., 2007)
- SIMPLE Prime — probabilistic initial model estimation (Elmlund et al., 2013)

_3D Refinement — Bayesian & Iterative (1998-2026):_
- RELION auto-refine — Bayesian gold-standard refinement with Fourier shell correlation (Scheres, JSB 2012; Zivanov et al., eLife 2018) — the dominant cryo-EM refinement engine 2012-present
- CryoSPARC homogeneous refinement — branch-and-bound optimization with non-uniform refinement (Punjani et al., Nat Methods 2017; Punjani et al., Nat Methods 2020) — fastest single-particle refinement
- CryoSPARC non-uniform refinement — cross-validation-based local resolution optimization (Punjani et al., 2020) — best resolution for many datasets
- FREALIGN — Fourier-space refinement (Grigorieff, JSB 2007; cisTEM implementation, Grant et al., eLife 2018)
- EMAN2 e2refine_easy / gold-standard pipeline (Tang et al., 2007)
- SPIDER — single-particle iterative reconstruction (Frank et al., Ultramicroscopy 1981; updated through 2020)
- Particle polishing / Bayesian polishing — beam-induced motion and radiation damage correction (Zivanov et al., eLife 2019)
- CTF refinement / per-particle CTF (Zivanov et al., eLife 2018)
- Ewald sphere correction for high-resolution (DeRosier, 2000; Wolf et al., eLife 2020)

_3D Variability & Heterogeneity (2007-2026):_
- Multi-body refinement — rigid-body decomposition for flexible complexes (Nakane et al., eLife 2018)
- CryoSPARC 3D variability analysis — principal subspace estimation of continuous heterogeneity (Punjani & Fleet, JSB 2021)
- CryoSPARC 3D classification — heterogeneous refinement (Punjani et al., 2017)
- RELION multi-class 3D classification — Bayesian discrete classification (Scheres, 2012)
- CryoDRGN — deep generative model for continuous cryo-EM heterogeneity (Zhong et al., Nat Methods 2021) — transformative approach using variational autoencoders
- CryoDRGN2 — amortized inference for cryo-EM heterogeneity (Zhong et al., ICLR 2021)
- e2gmm — Gaussian mixture model-based heterogeneity (Chen & Bhatt, Nat Methods 2023)
- ManifoldEM — manifold embedding for energy landscapes (Dashti et al., PNAS 2014; extended 2020)
- 3D-FLEX — 3D flexible refinement (Punjani & Fleet, 2023)
- RECOVAR — covariance-regularized heterogeneity (Gilles & Singer, 2024)
- DynaMight — dynamics from cryo-EM (Schwab et al., 2024)

_Post-Processing & Sharpening (2013-2026):_
- B-factor sharpening — Rosenthal-Henderson (Rosenthal & Henderson, JMB 2003)
- LocRes — local resolution estimation (Kucukelbir et al., 2014; Cardone et al., JSB 2013)
- DeepEMhancer — deep learning map sharpening and denoising (Sanchez-Garcia et al., Commun Biol 2021) — widely adopted for post-processing
- EMReady — deep learning map improvement (He et al., 2023)
- Model-based local sharpening (Jakobi et al., eLife 2017)
- Phenix auto_sharpen (Terwilliger et al., 2018)
- ModelAngelo — automated model building (Jamali et al., Nature 2024)

_Deep Learning & AI-Guided (2017-2026):_
- AlphaFold-guided fitting — AlphaFold2/3 structure prediction docked into cryo-EM maps (Jumper et al., Nature 2021; applied to cryo-EM 2021-2026)
- DeepPose — deep learning pose estimation for cryo-EM particles (2022)
- CryoAI — amortized inference for cryo-EM reconstruction (Levy et al., 2022)
- End-to-end differentiable cryo-EM reconstruction (Rosenbaum et al., 2021)
- Equivariant neural networks for cryo-EM (Levy et al., ICML 2022)
- Diffusion-prior cryo-EM reconstruction (2024)
- Foundation model for cryo-EM (2025-2026)

#### Step 3: Update Cryo-EM Solvers

After listing all Cryo-EM solvers, update `algorithm_base/cryo_em/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Cryo-EM solvers use the data format: `y` (N, H, W) stack of 2D particle images (cryo-EM micrograph crops), `ctf_params` per-particle CTF parameters (defocus, astigmatism, phase shift), `orientations` Euler angles (phi, theta, psi) and shifts (tx, ty). The `CryoEMOperator` handles the forward model (3D volume -> project -> CTF-corrupt -> noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Cryo-EM:**
- Apoferritin (EMPIAR-10146): RELION ~1.8 A, CryoSPARC ~1.7 A resolution (FSC 0.143)
- Beta-galactosidase (EMPIAR-10061): RELION ~2.2 A, CryoSPARC ~2.1 A
- TRPV1 (EMPIAR-10005): RELION ~3.4 A, CryoSPARC ~3.2 A (original 3.4 A by Liao et al., 2013)
- CryoDRGN heterogeneity: reproduces known conformational states on spliceosome data
- DeepEMhancer: FSC improvement of 0.1-0.5 A over standard B-factor sharpening
- All reference values from EMDB depositions and published papers

**Verification criteria:**
- `done` — PWM achieves resolution within 0.5 A of reference (FSC 0.143)
- `partial` — 0.5-2.0 A shortfall
- `gap` — >2.0 A shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged or reconstruction failed

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cryo_em' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cryo_em/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Cryo-EM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cryo_em/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_em/standard/`

---

### Cathodoluminescence (`cathodoluminescence`) Modality Template

#### Step 1: Verify Standard Dataset

For Cathodoluminescence, what dataset do you use to verify? Is this dataset used for CL popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cathodoluminescence/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CL standard dataset.

**Popular datasets to consider:**
- **CL Spectral Imaging of Semiconductors (Christen et al., 1990s-2020)** — hyperspectral CL datasets from III-V and II-VI semiconductors (GaN, InGaN quantum wells, ZnO); the most widely used benchmarks for CL spectral deconvolution and defect analysis
- **Geological CL Mineral Datasets (Marshall, 1988; Gotze et al., 2001-2020)** — CL spectra and images of quartz, feldspar, zircon, and carbonates; standard for geological provenance studies and mineral characterization
- **Nanostructure CL Datasets (Kociak & Zagonel, 2014-2023)** — CL from plasmonic nanoparticles, quantum dots, and nanowires measured in STEM-CL; used for validating spatial deconvolution and mode mapping algorithms
- **Photonic Crystal CL Data (Sapienza et al., 2012-2020)** — CL spectral maps of photonic nanostructures; used for local density of optical states (LDOS) mapping validation
- **Cathodoluminescence Benchmark Round-Robin (2018-2022)** — inter-laboratory comparison datasets with standardized acquisition parameters

**Decision criteria:** Semiconductor CL hyperspectral datasets (GaN/InGaN) are the most widely used for algorithm validation. Nanostructure STEM-CL for spatial deconvolution. Use the dataset most widely referenced in CL analysis papers (2000-2026).

#### Step 2: List All Cathodoluminescence Algorithms

Please first ensure all the Cathodoluminescence algorithms have been listed in `\pwm\public\algorithm_base\cathodoluminescence\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cathodoluminescence. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CL solvers, please update the CL solver.

**Key algorithms to cover (1965-2026):**

_Classical Spectral Analysis (1965-2009):_
- Peak fitting with Gaussian/Lorentzian profiles — standard CL spectral analysis (Yacobi & Holt, 1990)
- Background subtraction — polynomial and Shirley-type background removal for CL spectra
- Spectral deconvolution — multi-peak fitting with Voigt profiles for overlapping CL bands (Reshchikov & Morkoc, JAP 2005)
- Temperature-dependent CL analysis — Arrhenius activation energy from CL thermal quenching (Bebb & Williams, 1972)
- Chromaticity mapping — CIE color space representation of panchromatic CL (Marshall & Mariano, 1988)
- Depth-resolved CL — Monte Carlo electron trajectory simulation for depth profiling (Hovington et al., Scanning 1997)

_Spatial Deconvolution & Hyperspectral (2005-2016):_
- Spatial deconvolution of CL — electron beam broadening correction using Monte Carlo-simulated generation volume (Diener et al., 2010)
- Hyperspectral unmixing for CL — Non-negative Matrix Factorization (NMF) applied to CL spectral images (Kociak & Zagonel, Ultramicroscopy 2014)
- Vertex Component Analysis (VCA) for CL spectral unmixing (Nascimento & Bioucas-Dias, 2005; applied to CL 2012)
- Principal Component Analysis (PCA) for CL dimensionality reduction and noise filtering
- Independent Component Analysis (ICA) for CL spectral separation (2008)
- Maximum likelihood spectral fitting with physics-based CL models (Christen et al., 2011)
- LDOS reconstruction from angle-resolved CL (Sapienza et al., Nat Mater 2012)
- Kramers-Kronig analysis of CL for optical property extraction (Garcia de Abajo, Rev Mod Phys 2010)

_Advanced Processing & Machine Learning (2017-2026):_
- Deep learning CL spectral deconvolution — CNN-based peak identification and fitting (2023)
- Autoencoder-based CL hyperspectral unmixing (2022)
- Physics-informed neural network for CL carrier diffusion modeling (2024)
- GAN-based CL image super-resolution (2023)
- Transfer learning for CL defect classification (2021)
- Bayesian spectral decomposition for CL (2019)
- Real-time CL spectral classification with random forest/SVM (2020)
- Variational autoencoder for CL latent space analysis (2024)
- Transformer-based CL hyperspectral analysis (2025)
- Foundation model for electron microscopy spectroscopy including CL (2025-2026)

#### Step 3: Update Cathodoluminescence Solvers

After listing all CL solvers, update `algorithm_base/cathodoluminescence/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CL solvers use the data format: `y` (H, W, num_wavelengths) hyperspectral CL datacube, `wavelengths` array of spectral channel centers, `beam_params` (accelerating voltage, beam current, spot size). The `CLOperator` handles the forward model (material emission spectrum * generation volume * collection efficiency -> detector signal) and spectral deconvolution operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Cathodoluminescence:**
- Semiconductor CL: peak position accuracy <1 nm, FWHM accuracy <5% for GaN near-band-edge emission
- Hyperspectral unmixing: NMF recovers known spectral components with >95% cosine similarity on synthetic data
- Spatial deconvolution: resolves quantum well features down to ~20 nm (vs. ~50 nm raw beam broadening)
- Published spectral fitting R-squared and residual metrics from CL literature

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cathodoluminescence' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cathodoluminescence/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Cathodoluminescence. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cathodoluminescence/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cathodoluminescence/standard/`

---

### CLEM (`clem`) Modality Template

#### Step 1: Verify Standard Dataset

For Correlative Light-Electron Microscopy, what dataset do you use to verify? Is this dataset used for CLEM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/clem/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CLEM standard dataset.

**Popular datasets to consider:**
- **CLEM Challenge Dataset (ISBI 2014, 2015)** — paired fluorescence and electron microscopy images with ground-truth landmark correspondences; the primary community benchmark for CLEM registration algorithms
- **Cell Biology CLEM Datasets (Müller et al., 2012-2020)** — correlative datasets from cell biology applications (endosomes, mitochondria, viral entry); widely used for demonstrating registration workflows
- **Vascular CLEM Datasets (Bhatt et al., 2017-2022)** — 3D CLEM datasets of vasculature with fluorescent labels and serial-section EM; used for volume CLEM registration benchmarks
- **FIB-SEM CLEM Datasets (Heymann et al., 2006; Narayan et al., 2015)** — combined confocal fluorescence + focused ion beam SEM; used for 3D correlative volume registration
- **Cryo-CLEM Datasets (Schorb & Briggs, 2014; Arnold et al., 2016)** — cryo-fluorescence paired with cryo-EM; standard for cryo-CLEM workflow validation
- **EMBL CLEM Benchmark (Kukulski et al., 2011)** — yeast cell CLEM with fiducial markers; widely cited protocol benchmark

**Decision criteria:** ISBI CLEM Challenge dataset is the most widely used benchmark for CLEM registration algorithm comparison. Cell biology CLEM datasets for practical workflow validation. Use the dataset most widely referenced in CLEM registration papers (2012-2026).

#### Step 2: List All CLEM Algorithms

Please first ensure all the CLEM algorithms have been listed in `\pwm\public\algorithm_base\clem\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/clem. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CLEM solvers, please update the CLEM solver.

**Key algorithms to cover (1990-2026):**

_Classical Registration (1990-2009):_
- Landmark-based registration — manual fiducial point matching with affine/polynomial transform (Mori et al., 2006)
- Fiducial bead alignment — gold or fluorescent bead-based coordinate transfer between LM and EM (Kukulski et al., JCB 2011)
- Rigid/affine registration — ICP and intensity-based alignment for 2D CLEM (2005)
- Grid-based coordinate mapping — finder grid and relocation strategies (Müller & Bhatt, 2007)
- Thin-plate spline warping — landmark-based non-rigid registration for CLEM (Bookstein, 1989; applied 2008)

_Deformable Registration & Feature-Based (2010-2016):_
- Deformable registration for CLEM — B-spline FFD applied to fluorescence-EM pairs (Rueckert et al., 1999; adapted 2012)
- ec-CLEM — easy correlative light-electron microscopy plugin (Paul-Gilloteaux et al., Nat Methods 2017; roots 2014) — the most widely used CLEM registration software
- Feature-based registration — SIFT/SURF feature matching for CLEM (2012)
- Warping — elastic deformation correction for section distortion in EM (Saalfeld et al., Nat Methods 2012; applied to CLEM 2014)
- Multi-modal mutual information registration — MI-based alignment of fluorescence and EM (Maes et al., 1997; adapted for CLEM 2013)
- Fiducial-free registration using cellular landmarks (Vazi et al., 2015)
- 3D CLEM registration — serial-section alignment of confocal + SEM volumes (Narayan et al., J Struct Biol 2015)
- BigWarp — landmark-based deformable registration in Fiji (Bogovic et al., 2016)

_Deep Learning & Advanced (2017-2026):_
- Deep learning CLEM registration — CNN for predicting deformation fields between fluorescence and EM (2021)
- Virtual staining — predicting fluorescence from EM using conditional GANs/U-Net (Ounkomol et al., Nat Methods 2018; extended 2023) — transformative approach eliminating physical fluorescence labeling
- CycleGAN for EM-to-fluorescence translation (2020)
- Self-supervised CLEM registration using contrastive learning (2022)
- Attention-based multi-modal CLEM alignment (2023)
- DeepCLEM — end-to-end deep learning CLEM pipeline (2023)
- Transformer-based CLEM registration (2024)
- 3D CLEM volume registration with deep learning (2022)
- Implicit neural representation for CLEM coordinate transfer (2024)
- Foundation model for correlative multi-modal microscopy registration (2025)

#### Step 3: Update CLEM Solvers

After listing all CLEM solvers, update `algorithm_base/clem/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CLEM solvers use the data format: `y_fluorescence` (H_f, W_f) or (H_f, W_f, C) fluorescence image(s), `y_em` (H_e, W_e) electron microscopy image, `landmarks` (N, 2) optional fiducial coordinates. The `CLEMOperator` handles the forward model (coordinate transform, resampling, intensity mapping between modalities) and registration error evaluation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CLEM:**
- ISBI CLEM Challenge: target registration error (TRE) — rigid ~500 nm, affine ~200 nm, deformable (ec-CLEM) ~100 nm, DL methods ~50-80 nm
- Fiducial-based CLEM: registration accuracy ~50-100 nm with gold beads
- Virtual staining: SSIM ~0.7-0.85 for EM-to-fluorescence prediction on cell organelles
- Published TRE and Dice coefficients from CLEM registration papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'clem' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/clem/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/clem/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/clem/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CLEM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/clem/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/clem/standard/`

---

## Nuclear & Particle Techniques — Modality Templates

---

### Atom Probe (`atom_probe`) Modality Template

#### Step 1: Verify Standard Dataset

For Atom Probe Tomography, what dataset do you use to verify? Is this dataset used for APT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/atom_probe/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original APT standard dataset.

**Popular datasets to consider:**
- **Cameca LEAP Benchmark Datasets (2005-2023)** — standard reference tips (pure Al, Si, stainless steel, Ni-based superalloys) acquired on LEAP 3000/4000/5000 instruments; the canonical benchmarks for APT reconstruction and mass spectrum analysis
- **Simulated APT Tips (Vurpillot et al., 2004; Larson et al., 2013)** — Monte Carlo field evaporation simulations with known atomic positions; gold standard for reconstruction algorithm validation since ground truth is exactly known
- **IVAS/AP Suite Reference Datasets (Cameca, 2010-2023)** — vendor-provided calibration datasets with known compositions for quantitative analysis validation
- **APT Open Database (Peng et al., 2019-2023)** — emerging community repository for shared APT datasets with metadata
- **Semiconductor APT Data (Inoue et al., 2009-2020)** — Si/SiGe/InGaAs device tips; used for validating spatial resolution and compositional accuracy in multilayer structures
- **TAPSim Simulated Data (Oberdorfer & Schmitz, 2011)** — field evaporation simulation with tip shape evolution; widely used for reconstruction algorithm development

**Decision criteria:** Simulated APT tips with known ground truth (TAPSim, Vurpillot) are essential for quantitative reconstruction validation. Cameca LEAP benchmarks for real-data workflow testing. Use the combination most widely referenced in APT reconstruction papers (2000-2026).

#### Step 2: List All Atom Probe Algorithms

Please first ensure all the Atom Probe algorithms have been listed in `\pwm\public\algorithm_base\atom_probe\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/atom_probe. Besides, you need to search all algorithms from 1950 to 2026. After listing all the APT solvers, please update the APT solver.

**Key algorithms to cover (1968-2026):**

_Classical Reconstruction (1968-2005):_
- Point projection reconstruction — original Müller field ion microscopy reconstruction (Müller et al., 1968)
- Bas et al. reconstruction — standard reverse point projection for atom probe tomography (Bas et al., Appl Surf Sci 1995) — the foundational reconstruction algorithm used in all modern APT software
- Voltage curve correction — evaporation field calibration from voltage history (Miller, 2000)
- Laser pulse correction — thermal tail correction for pulsed laser APT (Bunton et al., 2007)
- Flight path correction — bowl correction for curved detector geometry (2003)
- Mass spectrum analysis — peak identification, ranging, and deconvolution of mass-to-charge spectra (Miller & Smith, 1989)
- Multi-hit detection and correction (Saxey, Ultramicroscopy 2011; roots 2005)

_Refined Reconstruction & Quantitative (2006-2016):_
- Geiser refined reconstruction — improved tip shape evolution model with shank angle correction (Geiser et al., Microsc Microanal 2007) — the standard in IVAS/AP Suite
- Spatial Distribution Map (SDM) analysis — 3D pair correlation function for APT data (Geiser et al., 2007; Moody et al., 2007) — standard for spatial statistics
- Ion-by-ion reconstruction — sequential back-projection with per-ion tip model update (Gault et al., JAP 2010)
- Density-corrected reconstruction — enforcing known lattice density in reconstruction (Larson et al., 2013)
- Vurpillot field evaporation model — Monte Carlo simulation of field evaporation sequence (Vurpillot et al., JPhys D 2004; extended 2015)
- Cluster analysis — maximum separation method, DBSCAN, isosurface for precipitate analysis (Marquis & Hyde, 2010; Vaumousse et al., 2003)
- Proximity histogram (proxigram) analysis — compositional profiling across interfaces (Hellman et al., 2000)
- Local electrode reconstruction geometry models (Larson et al., 2004)
- Tip shape fitting and reconstruction parameter optimization (Haley et al., 2015)

_Machine Learning & Deep Learning (2017-2026):_
- ML atom identification — random forest / SVM for peak overlap deconvolution in mass spectra (Wei et al., Microsc Microanal 2019)
- Deep learning mass spectrum ranging — CNN-based automatic peak identification and decomposition (2021)
- Neural network reconstruction — learned projection model replacing analytic back-projection (2023)
- GAN-based APT data augmentation and noise reduction (2022)
- Graph neural network for APT point cloud segmentation and clustering (2024)
- Physics-informed neural network for field evaporation modeling (2023)
- Deep learning APT reconstruction — end-to-end detector-to-3D reconstruction (2024)
- Transformer-based mass spectrum analysis (2025)
- Self-supervised learning for APT artifact correction (2024)
- Foundation model for atom probe data analysis (2025-2026)

#### Step 3: Update Atom Probe Solvers

After listing all Atom Probe solvers, update `algorithm_base/atom_probe/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All APT solvers use the data format: `y` (N, 4) detector hits with (x_det, y_det, time_of_flight, sequence_number), `voltage_curve` array of evaporation voltages, `detector_params` (efficiency, multi-hit deadtime, bowl correction). The `APTOperator` handles the forward model (3D atomic position -> field evaporation -> flight -> detector hit) and inverse reconstruction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Atom Probe:**
- Simulated tips: Bas reconstruction spatial resolution ~0.3 nm lateral, ~0.1 nm depth; Geiser refined ~0.2 nm lateral
- Compositional accuracy: <2 at.% error for binary alloys (Al-Cu, Ni-Cr) with known composition
- SDM analysis: recovers lattice planes in pure metals (Al, W) with known interplanar spacings
- Cluster analysis: detects precipitates >1 nm radius in age-hardened alloys (Al-Cu-Mg)
- Published reconstruction metrics from APT community round-robin studies

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'atom_probe' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/atom_probe/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/atom_probe/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/atom_probe/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Atom Probe. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/atom_probe/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/atom_probe/standard/`

---

### Muon Tomography (`muon_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For Muon Tomography, what dataset do you use to verify? Is this dataset used for muon tomography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/muon_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original muon tomography standard dataset.

**Popular datasets to consider:**
- **LANL Muon Tomography Data (Borozdin et al., 2003; Morris et al., 2008)** — cosmic-ray muon scattering data from Los Alamos drift tube detectors; the original benchmark for muon tomography reconstruction algorithms; includes measurements of uranium, lead, iron, and aluminum test objects
- **CERN Muon Scattering Data (Anghel et al., 2015-2020)** — controlled muon beam scattering experiments with known target materials; used for validating scattering angle reconstruction
- **ScanPyramids Cosmic-Ray Muon Data (Morishima et al., Nature 2017)** — cosmic-ray muon transmission data from the Great Pyramid of Giza (Khufu); high-profile application dataset demonstrating muon tomography for archaeology
- **Simulated Muon Tomography Data (Schultz et al., NIM-A 2004; Thomay et al., 2016)** — GEANT4-based Monte Carlo simulations of cosmic-ray muon interactions with known material distributions; gold standard for algorithm development
- **Decision Sciences/Silverside Muon Data (2012-2020)** — commercial muon tomography scanner data with cargo/vehicle phantoms
- **MuTe Muon Telescope Data (Vesga-Ramirez et al., 2020)** — volcano muography datasets for geological applications

**Decision criteria:** LANL muon data with known test objects is the foundational benchmark for muon scattering tomography. GEANT4 simulations with known ground truth for quantitative algorithm validation. Use the combination most widely referenced in muon tomography papers (2003-2026).

#### Step 2: List All Muon Tomography Algorithms

Please first ensure all the Muon Tomography algorithms have been listed in `\pwm\public\algorithm_base\muon_tomo\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/muon_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the muon tomography solvers, please update the muon tomography solver.

**Key algorithms to cover (1955-2026):**

_Classical & Transmission (1955-2005):_
- Muon absorption/transmission imaging — counting-based attenuation measurement (George, 1955; Alvarez et al., Science 1970 — pyramid imaging)
- Simple scattering angle histogram — material discrimination by scattering angle distribution (Borozdin et al., Nature 2003)
- Projection-based muon imaging — straight-line path assumption for transmission muography (Nagamine et al., NIM-A 1995)

_Point-of-Closest-Approach & Scattering (2003-2012):_
- PoCA — Point of Closest Approach reconstruction (Schultz et al., NIM-A 2004) — the baseline scattering tomography algorithm; assumes single scatter point at the midpoint of closest approach between incoming/outgoing muon tracks
- MLP — Most Likely Path reconstruction (Schultz et al., NIM-A 2007; Wang et al., 2009) — improved over PoCA by computing the most probable muon trajectory through the volume accounting for multiple Coulomb scattering
- Binned scattering density — voxelized scattering angle statistics (Gilboy et al., 2007)
- Metric tomography — scattering angle variance per voxel (Durham et al., 2008)
- Angle statistics method — median/RMS scattering angles per voxel (2009)

_Iterative & Statistical (2010-2018):_
- MLEM for muon tomography — maximum likelihood expectation maximization adapted for muon scattering data (Wang et al., JINST 2009; Schultz, IEEE TNS 2010) — iterative refinement of scattering density from muon track pairs
- Bayesian reconstruction — probabilistic framework incorporating prior knowledge of scattering distributions (Riggi et al., NIM-A 2014) — improved material discrimination via posterior probability
- Filtered back-projection adapted for muon scattering angles (2011)
- Algebraic reconstruction technique for muon tomography (2012)
- Penalized likelihood muon tomography with total variation prior (2015)
- Multi-group scattering reconstruction — energy-dependent scattering analysis (2013)
- Momentum-dependent reconstruction — using muon momentum information (2016)
- Profile likelihood method for muon scattering (2017)

_Deep Learning & Modern (2019-2026):_
- CNN-based muon image classification — material identification from scattering images (Vanini et al., JINST 2019)
- Deep learning muon imaging — CNN/U-Net for scattering density reconstruction (Miryala & Shankar, 2022)
- Graph neural network for muon track reconstruction (2023)
- 3D CNN for volumetric muon tomography (2023)
- Physics-informed neural network for muon scattering inversion (2024)
- Transformer-based muon tomography reconstruction (2024)
- Generative model for muon tomography with limited statistics (2025)
- Real-time deep learning muon screening for cargo inspection (2024)
- Diffusion-prior muon reconstruction (2025)
- Hybrid MLP + DL muon tomography (2025-2026)

#### Step 3: Update Muon Tomography Solvers

After listing all muon tomography solvers, update `algorithm_base/muon_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All muon tomography solvers use the data format: `y` (N, 12) array of muon track pairs — each row contains (x_in, y_in, z_in, dx_in, dy_in, dz_in, x_out, y_out, z_out, dx_out, dy_out, dz_out) for incoming and outgoing muon tracks, `volume_bounds` (3, 2) inspection volume bounds. The `MuonTomoOperator` handles the forward model (scattering density volume -> Coulomb scattering angles -> track deflections) and inverse reconstruction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Muon Tomography:**
- LANL test objects: PoCA detects 10x10x10 cm uranium/lead blocks within ~1 min exposure; MLP improves spatial resolution by ~30% over PoCA
- Simulated data (GEANT4): MLEM achieves ~2 cm spatial resolution for high-Z materials with 10-min exposure
- Material discrimination: correctly classify high-Z (U, Pu, Pb) vs. medium-Z (Fe, Cu) vs. low-Z (Al, C) with >95% accuracy in 1-min scan
- Bayesian method: 2-5 dB improvement in scattering density contrast over MLEM
- Published ROC curves and detection rates from LANL/Decision Sciences papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'muon_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/muon_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Muon Tomography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/muon_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/muon_tomo/standard/`

---

### Neutron Tomography (`neutron_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For Neutron Tomography, what dataset do you use to verify? Is this dataset used for neutron tomography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/neutron_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original neutron tomography standard dataset.

**Popular datasets to consider:**
- **NIST Neutron Imaging Facility Data (Hussey et al., 2005-2023)** — neutron CT datasets from the NIST Center for Neutron Research; calibration phantoms and materials science specimens; widely used for beam hardening correction and reconstruction validation
- **PSI NEUTRA/ICON Beamline Data (Lehmann et al., 2001-2023)** — neutron radiography and tomography from the Swiss Spallation Neutron Source (SINQ); the most prolific neutron imaging facility providing benchmark datasets for the community
- **ISIS Neutron Imaging Data (Kockelmann et al., 2007-2023)** — pulsed-source energy-resolved neutron imaging from ISIS; used for Bragg-edge and energy-selective tomography benchmarks
- **IAEA Neutron Imaging Standards (2005-2020)** — standardized reference objects (step wedges, resolution targets) for neutron imaging facility qualification
- **J-PARC Neutron Data (Shinohara et al., 2011-2023)** — high-intensity pulsed neutron source data for energy-resolved imaging
- **HZB CONRAD/BER-II Data (Kardjilov et al., 2011-2020)** — neutron tomography from Helmholtz-Zentrum Berlin; used for magnetic field imaging and cultural heritage studies

**Decision criteria:** PSI NEUTRA/ICON data is the most widely referenced in neutron imaging publications. NIST data for standardized calibration. Use the dataset most widely referenced in neutron tomography reconstruction papers (2000-2026).

#### Step 2: List All Neutron Tomography Algorithms

Please first ensure all the Neutron Tomography algorithms have been listed in `\pwm\public\algorithm_base\neutron_tomo\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/neutron_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the neutron tomography solvers, please update the neutron tomography solver.

**Key algorithms to cover (1975-2026):**

_Classical & Analytic (1975-2005):_
- FBP for neutron CT — filtered back-projection applied to neutron transmission sinograms (Schillinger et al., 2000) — the baseline neutron tomography reconstruction
- Parallel-beam FBP with Ram-Lak/Shepp-Logan filters for neutron CT
- ART for neutron CT — algebraic reconstruction technique (2001)
- SIRT for neutron CT — simultaneous iterative reconstruction (2003)
- SART for neutron CT — simultaneous algebraic reconstruction technique (Andersen & Kak, 1984; applied to neutron CT 2005)
- Neutron beam hardening correction — polychromatic spectrum correction for thermal/cold neutron beams (Hassanein et al., NIM-A 2005)
- Scatter correction for neutron CT — Monte Carlo and empirical scatter removal (Kardjilov et al., NIM-A 2005)
- Normalization and flat-field correction for neutron radiographs

_Iterative & Energy-Resolved (2006-2016):_
- MLEM for neutron CT (2006) — maximum likelihood reconstruction for low-flux neutron data
- TV-regularized neutron CT — total variation for sparse-angle and noisy neutron data (2010)
- Bragg-edge strain imaging — neutron wavelength-resolved transmission for crystallographic strain mapping (Santisteban et al., JAC 2001; tomographic extension 2012) — unique to neutron imaging
- Energy-resolved neutron tomography — wavelength-selective reconstruction exploiting resonance absorption and Bragg edges (Tremsin et al., NIM-A 2012; Kardjilov et al., 2012)
- Neutron phase-contrast tomography — grating-based differential phase contrast with neutrons (Pfeiffer et al., PRL 2006; tomographic extension 2008)
- Neutron dark-field imaging — ultra-small-angle scattering contrast (Strobl, Sci Rep 2015)
- Time-of-flight resolved neutron tomography — pulsed source energy discrimination (Kockelmann et al., 2007)
- Penalized weighted least squares for neutron CT (2014)
- Joint reconstruction of attenuation and scattering from neutron data (2016)

_Deep Learning & Modern (2017-2026):_
- CNN-based neutron CT denoising — U-Net/ResNet for low-dose neutron image enhancement (2021)
- Deep learning sparse-angle neutron CT — learned reconstruction from limited projections (2022)
- Physics-informed neural network for neutron transport (2023)
- Generative model for neutron CT artifact removal (2023)
- Deep learning Bragg-edge analysis — automated strain mapping from TOF neutron data (2022)
- Neural network beam hardening correction (2023)
- Transformer-based neutron tomography (2024)
- Self-supervised denoising for neutron radiographs (2023)
- Diffusion-prior neutron CT reconstruction (2025)
- Foundation model for neutron imaging (2025-2026)

#### Step 3: Update Neutron Tomography Solvers

After listing all neutron tomography solvers, update `algorithm_base/neutron_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All neutron tomography solvers use the data format: `y` (num_angles, H, W) neutron transmission radiographs (or sinograms), `angles` array of projection angles, `flat_field` (H, W) open beam normalization, `dark_field` (H, W) detector dark current. The `NeutronTomoOperator` handles the forward model (Beer-Lambert attenuation through 3D volume -> transmission projections) and adjoint (backprojection) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Neutron Tomography:**
- PSI NEUTRA phantoms: FBP achieves ~50 um spatial resolution at full angular sampling (>300 projections)
- Sparse-angle (30-60 projections): FBP heavily degraded, TV ~25 dB, DL methods ~30 dB PSNR
- Bragg-edge strain accuracy: ~100 microstrain precision with TOF neutron data
- Beam hardening correction: <5% attenuation coefficient error for 1 cm water/metal phantoms
- Published metrics from PSI, NIST, and ISIS neutron imaging papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'neutron_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/neutron_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Neutron Tomography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/neutron_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_tomo/standard/`

---

### Neutron Diffraction (`neutron_diffraction`) Modality Template

#### Step 1: Verify Standard Dataset

For Neutron Diffraction, what dataset do you use to verify? Is this dataset used for neutron diffraction popular algorithms? Please ensure the standard dataset in `datasets/benchmark/neutron_diffraction/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original neutron diffraction standard dataset.

**Popular datasets to consider:**
- **COD Neutron Entries (Crystallography Open Database, Grazulis et al., 2009-2023)** — crystallographic structures determined by neutron diffraction; provides reference structures for Rietveld refinement validation
- **ISIS/SNS Powder Diffraction Archives (2000-2023)** — powder neutron diffraction patterns from HRPD (ISIS), POWGEN (SNS), and other high-resolution powder diffractometers; the primary benchmark datasets for powder refinement algorithms
- **ORNL TOPAZ Single-Crystal Data (Schultz et al., 2014-2023)** — Laue single-crystal neutron diffraction from the TOPAZ beamline at SNS; used for single-crystal structure solution benchmarks
- **ILL D2B/D20 Powder Data (Hewat, 1975-2023)** — powder diffraction from Institut Laue-Langevin; the most prolific neutron diffraction facility, providing canonical test datasets
- **NIST Standard Reference Powder Data (2005-2020)** — certified reference materials (LaB6, Si, Al2O3) for instrument calibration and peak shape validation
- **IUCr Commission on Neutron Scattering Round-Robin Data (1982-2015)** — inter-laboratory comparison datasets for validating refinement methods

**Decision criteria:** ISIS/SNS powder archives with well-characterized reference materials are the most widely used for algorithm validation. ILL D2B data for high-resolution powder diffraction. Use the dataset most widely referenced in neutron diffraction analysis papers (1990-2026).

#### Step 2: List All Neutron Diffraction Algorithms

Please first ensure all the Neutron Diffraction algorithms have been listed in `\pwm\public\algorithm_base\neutron_diffraction\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/neutron_diffraction. Besides, you need to search all algorithms from 1950 to 2026. After listing all the neutron diffraction solvers, please update the neutron diffraction solver.

**Key algorithms to cover (1964-2026):**

_Classical Structure Solution (1964-1990):_
- Patterson methods for neutron diffraction — heavy-atom location from Patterson maps (Patterson, 1934; applied to neutron data extensively post-1964)
- Fourier difference maps — iterative structure completion from partial models with neutron data
- Direct methods adapted for neutron data — statistical phasing from neutron structure factors (Hauptman & Karle, 1953; neutron-specific considerations for negative scattering lengths)
- Nuclear density mapping — exploiting neutron sensitivity to hydrogen/deuterium positions (1970s)

_Rietveld & Profile Refinement (1969-2009):_
- Rietveld refinement — whole-pattern powder diffraction refinement (Rietveld, JAC 1969) — the foundational algorithm for neutron powder diffraction; fits crystal structure parameters to full diffraction pattern
- Le Bail extraction — pattern decomposition without structural model (Le Bail et al., Mater Res Bull 1988) — model-free intensity extraction for indexing and space group determination
- Pawley refinement — constrained pattern decomposition with unit cell (Pawley, JAC 1981) — individual peak intensities refined within unit cell constraints
- GSAS / GSAS-II — General Structure Analysis System for neutron/X-ray refinement (Larson & Von Dreele, 1994; Toby & Von Dreele, JAC 2013) — the most widely used neutron refinement software
- FullProf — Rietveld refinement suite with neutron-specific features (Rodriguez-Carvajal, 1993)
- JANA — crystallographic computing system for modulated/composite structures (Petricek et al., 2014)
- Absorption correction for neutron diffraction — cylindrical, spherical, and arbitrary shape corrections (Rouse & Cooper, 1970)
- TOF neutron powder profile functions — back-to-back exponential convolved with Gaussian/Lorentzian (Von Dreele et al., 1982)
- Preferred orientation correction — March-Dollase, spherical harmonics (Dollase, JAC 1986)
- Magnetic structure refinement — Rietveld with magnetic symmetry (Bertaut, 1963; Shull & Smart, 1949; representational analysis)

_Advanced Analysis (2005-2016):_
- Maximum Entropy Method (MEM) for neutron Fourier maps — model-free nuclear density reconstruction from structure factors (Sakata & Sato, Acta Cryst 1990; widely used for neutron data 2005+)
- Pair Distribution Function (PDF) analysis — total scattering analysis for local structure from neutron data (Egami & Billinge, 2003; PDFgui, Farrow et al., 2007) — exploits neutron advantages for light elements
- Reverse Monte Carlo (RMC) for neutron total scattering — atomic configuration modeling (McGreevy & Pusztai, 1988; RMCProfile, Tucker et al., 2007)
- Magnetic PDF — local magnetic correlations from neutron total scattering (Frandsen et al., 2014)
- DIFFaX — simulation of diffraction from faulted crystals (Treacy et al., 1991)
- TOPAS — general-purpose diffraction analysis including neutron (Coelho, 2018)
- Bayesian refinement — posterior probability structural analysis for neutron data (Sivia & David, 2001)
- Combined neutron + X-ray refinement — joint refinement for complementary sensitivity (2008)
- Texture analysis from neutron diffraction — ODF determination via MAUD, MTEX (2006)

_Machine Learning & Deep Learning (2017-2026):_
- ML crystal structure prediction — accelerated structure solution from neutron powder patterns (Ziletti et al., Nat Commun 2018; adapted to neutron 2020)
- Random forest / neural network for automated phase identification from neutron powder data (2020)
- Deep learning peak fitting — CNN-based profile decomposition replacing manual fitting (2022)
- Generative model for crystal structure from diffraction — variational autoencoder approach (2023)
- Neural network for magnetic structure determination from neutron data (2023)
- Graph neural network for crystal structure prediction from diffraction (2024)
- Self-supervised representation learning for diffraction patterns (2024)
- DiffCSP — diffusion-based crystal structure prediction (Jiao et al., ICLR 2024; applied 2025)
- Foundation model for crystallographic analysis (2025)
- Automated Rietveld refinement with reinforcement learning (2025-2026)

#### Step 3: Update Neutron Diffraction Solvers

After listing all neutron diffraction solvers, update `algorithm_base/neutron_diffraction/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All neutron diffraction solvers use the data format: `y` (num_channels,) or (num_channels, num_detectors) diffraction intensity pattern, `two_theta` or `d_spacing` or `tof` arrays for the diffraction axis, `instrument_params` (wavelength, resolution function, detector geometry). The `NeutronDiffractionOperator` handles the forward model (crystal structure -> structure factors -> powder pattern with profile functions) and refinement operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Neutron Diffraction:**
- NIST LaB6 reference: Rietveld Rwp <5% for instrument calibration
- Standard structures (Y2O3, Al2O3): lattice parameters within 0.001 A of reference, thermal parameters within 10%
- Hydrogen positions: neutron refinement locates H/D atoms to within 0.01 A of known positions (advantage over X-ray)
- PDF analysis: G(r) peak positions within 0.005 A, widths within 5% for crystalline standards
- Published R-factors and goodness-of-fit from IUCr standards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'neutron_diffraction' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/neutron_diffraction/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Neutron Diffraction. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/neutron_diffraction/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/neutron_diffraction/standard/`

---

### Proton Radiography (`proton_radiography`) Modality Template

#### Step 1: Verify Standard Dataset

For Proton Radiography, what dataset do you use to verify? Is this dataset used for proton radiography/CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/proton_radiography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original proton radiography standard dataset.

**Popular datasets to consider:**
- **LANL pRad Data (Morris et al., 2006-2023)** — proton radiography datasets from the Los Alamos Neutron Science Center (LANSCE) 800 MeV proton beam; the canonical benchmark for proton radiography image formation and reconstruction; used for studying dynamic materials and explosively driven experiments
- **GSI Proton CT Phantoms (Schulte et al., 2005; Rinaldi et al., 2013)** — proton CT datasets from GSI Helmholtzzentrum using therapeutic-energy proton beams (100-250 MeV); used for medical proton CT reconstruction algorithm validation
- **PRIMA Proton Imaging Data (PRoton IMAging, Johnson et al., 2017-2022)** — proton CT prototype scanner data with calibration phantoms (Gammex, CTP404); used for water-equivalent path length calibration and relative stopping power reconstruction
- **Simulated Proton Radiography/CT Data (Penfold et al., 2009; Schulte et al., 2008)** — GEANT4/TOPAS Monte Carlo simulations with known phantom geometry; gold standard for algorithm development with exact ground truth
- **Loma Linda pCT Prototype Data (Bashkirov et al., 2016)** — phase-II proton CT scanner data with head phantom; used for medical pCT reconstruction
- **Bergen pCT Prototype Data (Pettersen et al., 2017-2023)** — ALPIDE-based digital tracking calorimeter proton CT data

**Decision criteria:** GEANT4/TOPAS simulated data with known ground truth is essential for quantitative algorithm validation. LANL pRad for high-energy proton radiography. GSI/PRIMA for medical proton CT. Use the combination most widely referenced in proton imaging papers (2005-2026).

#### Step 2: List All Proton Radiography Algorithms

Please first ensure all the Proton Radiography algorithms have been listed in `\pwm\public\algorithm_base\proton_radiography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/proton_radiography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the proton radiography solvers, please update the proton radiography solver.

**Key algorithms to cover (1963-2026):**

_Classical Radiography (1963-2000):_
- Proton transmission radiography — attenuation-based imaging with high-energy protons (Koehler, Science 1968)
- Nuclear scattering radiography — multiple Coulomb scattering imaging (West & Sherwood, Nature 1972)
- Magnetic lens imaging for pRad — quadrupole lens systems for proton imaging at LANL (Morris et al., 1998)
- Areal density measurement from proton energy loss (Bichsel, 1972)
- Range radiography — residual range-based imaging (Romero et al., 1995)

_Proton CT Reconstruction — Analytic (2000-2012):_
- FBP for proton CT — filtered back-projection of water-equivalent path length sinograms (Zygmanski et al., PMB 2000) — baseline proton CT reconstruction
- MLP reconstruction — most likely path estimation for individual proton trajectories replacing straight-line assumption (Schulte et al., Med Phys 2008) — the key algorithmic advancement for proton CT; accounts for multiple Coulomb scattering to determine the most probable curved path through tissue
- Space-angle reconstruction — using both spatial and angular information from entrance/exit proton tracks (Li et al., PMB 2006)
- WEPL calibration — water-equivalent path length calibration from calorimeter/range detector measurements (Hurley et al., Med Phys 2012) — essential preprocessing for quantitative proton CT
- Straight-line path approximation — simplified reconstruction ignoring MCS (baseline comparison)
- Cubic spline path approximation (Williams, PMB 2004)

_Iterative & Optimization (2010-2020):_
- Iterative proton CT reconstruction — algebraic/statistical methods incorporating MLP (Penfold et al., PMB 2009; Schulte et al., Med Phys 2008) — SART and MLEM adapted for proton CT with individual track data
- TV-regularized proton CT — total variation for sparse-angle proton CT (Rit et al., Med Phys 2013)
- DROP — diagonally relaxed orthogonal projections for proton CT (Penfold et al., PMB 2010)
- Superiorization for proton CT — perturbation of iterative algorithms toward TV minimization (Penfold et al., 2015)
- Block-iterative projection algorithms for proton CT (Censor et al., 2012)
- Density reconstruction from energy loss and scattering — combined WEPL + scattering angle inversion (2014)
- Binning-based proton CT — grouped proton path reconstruction (2016)
- Pencil beam proton CT — intensity-modulated proton CT (2018)
- Regularized proton CT with scattering prior (2019)

_Deep Learning & Modern (2020-2026):_
- Deep learning proton CT — CNN-based image reconstruction from proton projections (DeJongh et al., PMB 2021)
- Neural network MLP — learned most likely path estimation replacing analytic formula (2022)
- U-Net artifact removal for proton CT (2022)
- Physics-informed neural network for proton transport (2023)
- Deep unfolding for iterative proton CT (2023)
- Transformer-based proton CT reconstruction (2024)
- Diffusion-prior proton CT (2025)
- Graph neural network for proton track reconstruction (2024)
- End-to-end differentiable proton CT pipeline (2025)
- Real-time deep learning proton radiography for adaptive therapy (2025-2026)

#### Step 3: Update Proton Radiography Solvers

After listing all proton radiography solvers, update `algorithm_base/proton_radiography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All proton radiography solvers use the data format: `y` (N, 8+) individual proton track data — each row contains (x_in, theta_in, y_in, phi_in, x_out, theta_out, y_out, phi_out, WEPL) for entrance/exit position, angle, and energy loss, `scanner_geometry` (source-detector distances, tracker positions). The `ProtonRadiographyOperator` handles the forward model (RSP volume -> MLP trajectories -> energy loss -> detector signal) and inverse reconstruction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Proton Radiography:**
- Simulated Gammex phantom: FBP-SLP ~3% RSP error, FBP-MLP ~1.5% RSP error, iterative MLP ~1% RSP error
- Spatial resolution: MLP achieves ~1 mm for 200 MeV protons (vs. ~3 mm for straight-line path)
- CTP404 phantom: RSP accuracy within 1% for tissue-equivalent inserts with iterative MLP
- Imaging dose: proton CT achieves comparable image quality to X-ray CT at ~1-5 mGy
- Published RSP accuracy and spatial resolution from Loma Linda, PRIMA, Bergen prototype papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'proton_radiography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/proton_radiography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/proton_radiography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/proton_radiography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Proton Radiography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/proton_radiography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_radiography/standard/`

---

## X-ray Techniques — Modality Templates

---

### SAXS (`saxs`) Modality Template

#### Step 1: Verify Standard Dataset

For Small-Angle X-ray Scattering, what dataset do you use to verify? Is this dataset used for SAXS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/saxs/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SAXS standard dataset.

**Popular datasets to consider:**
- **SASBDB (Small Angle Scattering Biological Data Bank, Valentini et al., 2015-2023)** — the primary curated repository of SAXS/SANS experimental data and derived models; provides benchmark profiles with associated atomic structures for validation of analysis algorithms
- **SAS Portal (Doucet et al., 2019-2023)** — NIST/ORNL small-angle scattering data portal with standardized formats and metadata; emerging community resource for reproducible SAS analysis
- **IUCr Commission on SAS Round-Robin Data (Jacques et al., 2012; Trewhella et al., 2017)** — inter-laboratory SAXS comparison on standard proteins (BSA, lysozyme, glucose isomerase); the gold standard for validating data processing and analysis pipelines
- **ATSAS Test Datasets (Franke et al., 2017)** — bundled with ATSAS suite; standard test proteins for ab initio shape reconstruction and rigid-body modeling validation
- **BioISIS (Hura et al., 2009)** — biological SAXS database with well-characterized protein solutions; widely used for P(r) and modeling benchmarks
- **Simulated SAXS Data with Known Ground Truth** — synthetic scattering profiles computed from PDB structures; essential for quantitative algorithm validation

**Decision criteria:** SASBDB entries with associated crystal structures are the gold standard for SAXS analysis algorithm validation. IUCr round-robin data for reproducibility assessment. Use the dataset most widely referenced in SAXS analysis papers (2005-2026).

#### Step 2: List All SAXS Algorithms

Please first ensure all the SAXS algorithms have been listed in `\pwm\public\algorithm_base\saxs\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/saxs. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SAXS solvers, please update the SAXS solver.

**Key algorithms to cover (1938-2026):**

_Classical Analysis (1938-1990):_
- Guinier analysis — radius of gyration and I(0) from low-q region (Guinier, 1938) — the foundational SAXS analysis method
- Kratky plot analysis — molecular fold assessment from q^2*I(q) vs q (Kratky & Porod, 1949)
- Porod analysis — surface area from high-q power-law decay (Porod, 1951)
- Debye formula — scattering calculation from atomic coordinates (Debye, 1915; applied to macromolecules)
- Zimm plot — molecular weight determination from multi-concentration SAXS
- Unified fit — Guinier + Porod combined for hierarchical structures (Beaucage, JAC 1995; roots 1990)

_Indirect Fourier Transform & P(r) (1977-2010):_
- Indirect Fourier Transform / GNOM — model-free pair distance distribution P(r) determination (Svergun, JAC 1992; Moore, JAC 1980) — the standard method for real-space analysis of SAXS data; regularized inversion of scattering profile to obtain P(r) function
- Moore P(r) — Shannon sampling-based indirect transform (Moore, JAC 1980)
- GIFT — Generalized Indirect Fourier Transform for interacting systems (Glatter, JAC 1977; Fritz et al., 2000)
- BayesApp — Bayesian indirect Fourier transform with automated regularization (Hansen, JAC 2000)
- BIFT — Bayesian Indirect Fourier Transform (Vestergaard & Hansen, 2006)
- Tikhonov regularization for SAXS P(r) — L-curve and cross-validation approaches (2005)

_Ab Initio Shape Reconstruction (1999-2016):_
- DAMMIN — ab initio shape reconstruction using dummy atom model with simulated annealing (Svergun, Biophys J 1999) — the foundational SAXS shape reconstruction tool
- DAMMIF — fast ab initio shape reconstruction (Franke & Svergun, JAC 2009) — faster implementation of DAMMIN, widely used
- GASBOR — ab initio reconstruction at residue level using dummy residues (Svergun et al., 2001)
- DENSS — density from solution scattering (Grant, Nat Methods 2018) — electron density reconstruction from SAXS using iterative phase retrieval
- MONSA — multi-phase ab initio reconstruction for complexes (Svergun, 1999)
- DAMAVER / DAMCLUST — averaging and clustering of multiple ab initio models (Volkov & Svergun, 2003)

_Rigid-Body & Hybrid Modeling (2001-2016):_
- SASREF — rigid-body modeling of multi-domain complexes against SAXS data (Petoukhov & Svergun, Biophys J 2005)
- CORAL — rigid body + linker modeling (Petoukhov et al., 2012)
- BUNCH — combined rigid-body and ab initio modeling (Petoukhov & Svergun, 2005)
- FoXS — fast SAXS profile calculation from atomic structure (Schneidman-Duhovny et al., NAR 2010)
- CRYSOL — SAXS profile calculation from crystal structure with hydration shell (Svergun et al., JAC 1995)
- EOM — Ensemble Optimization Method for flexible systems (Bernado et al., JACS 2007; Tria et al., 2015)
- MultiFoXS — multi-state modeling (Schneidman-Duhovny et al., 2016)

_Bayesian & Advanced (2014-2020):_
- Bayesian ensemble modeling for SAXS — posterior sampling of conformational ensembles (Pelikan et al., Gen Physiol Biophys 2009; extended 2015)
- BES — Bayesian Ensemble SAXS (Antonov et al., 2016)
- ATSAS suite — integrated SAXS analysis pipeline (Franke et al., JAC 2017) — the most comprehensive SAXS analysis software
- SEC-SAXS analysis — size-exclusion chromatography coupled SAXS deconvolution (2014)
- Evolving factor analysis for SEC-SAXS (Meisburger et al., JACS 2016)
- Singular value decomposition for time-resolved SAXS (2015)

_Deep Learning (2020-2026):_
- Deep learning SAXS classification — CNN for shape/topology prediction from I(q) (Franke et al., 2018; extended 2022)
- Neural network P(r) estimation — direct prediction of pair distance distribution (2022)
- Autoencoder for SAXS latent space and anomaly detection (2023)
- Physics-informed neural network for SAXS inverse problem (2024)
- GAN-based SAXS data augmentation and denoising (2023)
- Transformer-based SAXS analysis (2024)
- Diffusion model for SAXS 3D shape reconstruction (2025)
- Deep learning ensemble modeling for intrinsically disordered proteins (2024)
- Foundation model for scattering data analysis (2025-2026)

#### Step 3: Update SAXS Solvers

After listing all SAXS solvers, update `algorithm_base/saxs/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SAXS solvers use the data format: `y` (num_q_points,) or (num_q_points, 3) scattering intensity I(q) with optional q and sigma columns, `q` array of scattering vector magnitudes (A^-1), `concentration` sample concentration (mg/mL). The `SAXSOperator` handles the forward model (3D shape/structure -> orientation-averaged scattering profile I(q)) and inverse analysis operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SAXS:**
- SASBDB lysozyme: Guinier Rg within 0.5 A of crystallographic value (~15.3 A), chi-squared <1.5 for CRYSOL fit
- BSA round-robin: GNOM P(r) Dmax within 2 A of consensus (~80 A), Rg within 0.3 A (~28 A)
- Ab initio shape (DAMMIF): NSD <1.0 between independent reconstructions; recovers correct symmetry
- CRYSOL: chi-squared <2.0 for crystal structure fit to experimental SAXS profile
- Published chi-squared, NSD, and Rg metrics from ATSAS validation papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'saxs' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/saxs/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/saxs/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/saxs/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SAXS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/saxs/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/saxs/standard/`

---

### WAXS (`waxs`) Modality Template

#### Step 1: Verify Standard Dataset

For Wide-Angle X-ray Scattering, what dataset do you use to verify? Is this dataset used for WAXS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/waxs/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original WAXS standard dataset.

**Popular datasets to consider:**
- **ICDD Powder Diffraction File (PDF-4+, 2023)** — the most comprehensive database of powder diffraction patterns; >1,000,000 entries; the canonical reference for phase identification from WAXS/XRD patterns
- **COD (Crystallography Open Database, Grazulis et al., 2009-2023)** — open-access crystal structure database with computed powder patterns; widely used for phase identification algorithm validation
- **2D WAXS Detector Calibration Datasets (Ashiotis et al., 2015; pyFAI)** — LaB6, CeO2, silver behenate calibration images from 2D area detectors; standard for geometry calibration and azimuthal integration validation
- **NIST Standard Reference Materials (SRM 640e Si, SRM 660c LaB6)** — certified powder diffraction standards for peak position and profile calibration
- **RRUFF Mineral Database (Lafuente et al., 2015)** — XRD patterns of minerals with known structures; used for phase ID algorithm benchmarking

**Decision criteria:** ICDD PDF is the gold standard for phase identification from WAXS. COD for open-access validation. NIST SRMs for instrument calibration and peak shape analysis. Use the dataset most widely referenced in WAXS/XRD analysis papers (1990-2026).

#### Step 2: List All WAXS Algorithms

Please first ensure all the WAXS algorithms have been listed in `\pwm\public\algorithm_base\waxs\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/waxs. Besides, you need to search all algorithms from 1950 to 2026. After listing all the WAXS solvers, please update the WAXS solver.

**Key algorithms to cover (1912-2026):**

_Classical Analysis (1912-1990):_
- Bragg peak indexing — unit cell determination from peak positions (Bragg, 1913)
- Peak fitting with Gaussian/Lorentzian/Voigt profiles — single and multi-peak fitting for line profile analysis (1960s-present)
- Pseudo-Voigt profile fitting — empirical convolution approximation widely used for XRD/WAXS (Thompson, Cox, & Hastings, JAC 1987)
- Scherrer equation — crystallite size from peak broadening (Scherrer, 1918)
- Williamson-Hall analysis — size-strain separation from peak broadening (Williamson & Hall, 1953)
- Warren-Averbach analysis — Fourier analysis of line profiles for size and strain distributions (Warren & Averbach, JAP 1950)
- Degree of crystallinity — amorphous/crystalline decomposition of WAXS patterns (1960s)

_Rietveld & Whole-Pattern (1969-2010):_
- Rietveld refinement — whole-pattern crystal structure refinement (Rietveld, JAC 1969) — the dominant method for quantitative WAXS/XRD analysis
- Le Bail whole-pattern decomposition — model-free intensity extraction (Le Bail et al., 1988)
- Pawley refinement — constrained whole-pattern decomposition (Pawley, 1981)
- Whole pattern fitting / total pattern analysis — quantitative phase analysis (Bish & Howard, JAC 1988)
- TOPAS — fundamental parameters approach for profile modeling (Cheary & Coelho, 1992; Coelho, JAC 2018)
- GSAS-II — comprehensive diffraction analysis (Toby & Von Dreele, JAC 2013)
- FullProf / WinPLOTR (Rodriguez-Carvajal, 1993)
- MAUD — materials analysis using diffraction, including texture (Lutterotti et al., 1999)

_Texture & Orientation Analysis (1980-2016):_
- Texture / Orientation Distribution Function (ODF) — pole figure analysis and ODF calculation from 2D WAXS (Bunge, 1982; MTEX, Bachmann et al., 2010)
- 2D WAXS azimuthal integration — cake integration for fiber/film texture (pyFAI, Ashiotis et al., JAC 2015)
- WAXS pole figure reconstruction — from 2D detector images (2005)
- Hermans orientation parameter — quantification of polymer chain alignment (Hermans et al., 1946; from WAXS)
- Fiber diffraction analysis — helical diffraction theory (Cochran et al., 1952; applied to polymer WAXS)

_Nanoparticle & Special Methods (1990-2016):_
- Debye function analysis (DFA) — total scattering from nanoparticles using atomic pair sums (Debye, 1915; Cervellino et al., JAC 2003) — the reference method for nanoparticle WAXS analysis
- PDF from WAXS — pair distribution function analysis of wide-angle data (Egami & Billinge, 2003; PDFgetX3, Juhas et al., JAC 2013)
- DSE — Debye Scattering Equation for nanocrystals with size/shape/strain (Cervellino et al., 2015)
- Whole nanoparticle modeling — fitting atomistic models to WAXS data (Banerjee et al., 2018)
- Amorphous materials analysis — radial distribution function from WAXS (Wright, 1974)

_Machine Learning & Deep Learning (2017-2026):_
- ML phase identification from WAXS/XRD — random forest/CNN for automated phase matching against ICDD/COD (Vecsei et al., 2019; Oviedo et al., npj Comp Mater 2019)
- Deep learning peak fitting — neural network profile decomposition (2021)
- CNN crystallographic symmetry classification from powder patterns (Park et al., IUCrJ 2017; extended 2020)
- Transfer learning for WAXS phase identification with limited training data (2022)
- GAN-based synthetic WAXS data generation for training (2023)
- Neural network Rietveld — learned refinement acceleration (2024)
- Automated WAXS analysis pipeline with deep learning (2023)
- Transformer-based WAXS pattern interpretation (2025)
- Foundation model for diffraction pattern analysis (2025)
- Self-supervised representation learning for WAXS (2025-2026)

#### Step 3: Update WAXS Solvers

After listing all WAXS solvers, update `algorithm_base/waxs/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All WAXS solvers use the data format: `y` (num_channels,) 1D integrated intensity pattern or (H, W) 2D detector image, `two_theta` or `q` array for diffraction axis, `wavelength` X-ray wavelength (A), `detector_geometry` (sample-detector distance, beam center, tilt). The `WAXSOperator` handles the forward model (crystal structure + texture -> diffraction pattern with instrumental broadening) and analysis operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for WAXS:**
- NIST SRM 640e Si: Rietveld Rwp <3%, peak positions within 0.001 degrees 2theta
- Quantitative phase analysis: accuracy within 1 wt.% for binary mixtures (IUCr round-robin, Madsen et al., 2001)
- ML phase ID: >95% top-1 accuracy on ICDD test set for single-phase patterns, >85% for multi-phase
- Debye function: nanoparticle size within 5% of TEM reference
- Published R-factors, Rwp, and phase quantification from crystallographic standards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'waxs' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/waxs/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/waxs/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/waxs/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for WAXS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/waxs/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/waxs/standard/`

---

### XFEL-SFX (`xfel_sfx`) Modality Template

#### Step 1: Verify Standard Dataset

For Serial Femtosecond Crystallography, what dataset do you use to verify? Is this dataset used for XFEL-SFX popular algorithms? Please ensure the standard dataset in `datasets/benchmark/xfel_sfx/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original XFEL-SFX standard dataset.

**Popular datasets to consider:**
- **CXIDB (Coherent X-ray Imaging Data Bank, Maia, 2012-2023)** — the primary repository for XFEL and coherent X-ray imaging datasets; hosts SFX diffraction patterns from lysozyme, photosystem II, rhodopsin, and other proteins; the canonical benchmark for SFX processing algorithms
- **LCLS SFX Datasets (Boutet et al., Science 2012; Barty et al., 2014)** — serial femtosecond crystallography data from the Linac Coherent Light Source; includes the first SFX structure determinations; widely used for benchmarking CrystFEL and related tools
- **SACLA SFX Data (Tono et al., 2015-2023)** — SFX data from SPring-8 Angstrom Compact Free Electron Laser; complementary pulse structure to LCLS
- **European XFEL SFX Data (Mancuso et al., 2019-2023)** — MHz-rate SFX data exploiting the high repetition rate of European XFEL; used for benchmarking high-throughput processing
- **Simulated SFX Data (Kirian et al., 2010; Ginn et al., 2015)** — Monte Carlo-simulated SFX diffraction patterns with known structure factors; essential for algorithm validation with ground truth
- **SFX Lysozyme Standard (Barends et al., 2014)** — hen egg-white lysozyme SFX data; the canonical test protein for SFX pipeline validation due to known structure

**Decision criteria:** CXIDB lysozyme datasets are the gold standard for SFX processing pipeline validation. LCLS data for real-XFEL benchmarks. Use the dataset most widely referenced in SFX processing papers (2012-2026).

#### Step 2: List All XFEL-SFX Algorithms

Please first ensure all the XFEL-SFX algorithms have been listed in `\pwm\public\algorithm_base\xfel_sfx\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/xfel_sfx. Besides, you need to search all algorithms from 1950 to 2026. After listing all the XFEL-SFX solvers, please update the XFEL-SFX solver.

**Key algorithms to cover (2006-2026):**

_Hit Finding & Data Reduction (2006-2016):_
- Cheetah hit finding — real-time diffraction pattern classification from XFEL data stream (Barty et al., JAC 2014) — the standard hit finder for SFX experiments
- Lit-pixel counting — simple threshold-based hit detection (2010)
- Radial profile-based hit finding (2012)
- Background subtraction for SFX — per-pixel and radial background models (2013)
- Detector geometry optimization — sub-pixel panel alignment from powder rings (2014)
- Multi-crystal hit sorting — distinguishing single vs. multi-crystal diffraction patterns (2015)

_Indexing & Integration (2010-2020):_
- CrystFEL indexing + integration — the primary SFX data processing suite; auto-indexing of individual snapshot diffraction patterns and Monte Carlo integration (White et al., JAC 2012; White et al., JAC 2016) — used in the majority of published SFX structures
- CrystFEL indexamajig — wrapper for multiple indexing algorithms (MOSFLM, DirAx, XDS, XGANDALF, TakeTwo, pinkIndexer) applied to still-shot data
- Monte Carlo integration — partial reflection intensity estimation by averaging over many crystals with random orientations (Kirian et al., Opt Express 2010; Kirian et al., Acta Cryst 2011) — the foundational integration method for SFX exploiting crystal-to-crystal averaging
- Post-refinement / partiality correction — per-crystal refinement of partiality, scale, and B-factor (White, JAC 2014; Ginn et al., Acta Cryst 2015)
- cctbx.xfel — SFX processing within the CCTBX framework (Sauter et al., Acta Cryst 2013; Brewster et al., 2018)
- DIALS for SFX — Diffraction Integration for Advanced Light Sources (Winter et al., 2018; applied to stills 2019)
- XGANDALF — gradient-descent auto-indexing (Gevorkov et al., Acta Cryst 2019)
- TakeTwo — reference-based indexing for challenging lattices (Ginn et al., 2016)
- pinkIndexer — indexing for broad-bandwidth/pink-beam SFX (Gevorkov et al., 2020)

_Merging & Scaling (2012-2020):_
- partialator — CrystFEL's scaling and merging (White et al., 2016)
- PRIME — post-refinement and merging (Uervirojnangkoorn et al., eLife 2015)
- cxi.merge — CCTBX merging for SFX (Sauter, 2015)
- Bayesian scaling — probabilistic scale factor estimation for SFX data (2018)
- EMC-based merging — expectation-maximization for compression of still-shot data (2016)

_Phase Retrieval & Structure Determination (2012-2020):_
- Molecular replacement for SFX — standard phasing using known homologous structures (McCoy et al., 2007; applied to SFX 2012)
- SAD/MAD phasing from SFX — anomalous phasing using heavy atoms with serial data (Barends et al., Nature 2014)
- Native SAD-SFX — anomalous signal from native sulfur/phosphorus in serial data (Nass et al., IUCrJ 2016)
- XFEL time-resolved structure determination — pump-probe SFX for reaction intermediates (Tenboer et al., Science 2014)
- De novo phasing from SFX — direct methods and charge flipping applied to serial data (2018)

_Single-Particle & Coherent Imaging (2011-2020):_
- EMC — Expand-Maximize-Compress for single-particle XFEL imaging (Loh & Elser, PRE 2009) — orientation recovery and 3D reconstruction from single-particle XFEL diffraction patterns without crystallization
- Multi-tiered iterative phasing for single-particle CDI (2016)
- XFEL phase retrieval — oversampling-based phase retrieval for coherent diffractive imaging (Miao et al., 1999; applied to XFEL 2011)
- Cryptotomography — orientation-free single-particle reconstruction (2017)

_Deep Learning & Modern (2018-2026):_
- Deep learning hit finding — CNN-based classification of XFEL diffraction patterns (Ke et al., JAC 2018; extended 2020) — automated Bragg spot vs. blank/junk classification
- Neural network indexing — learned auto-indexing for challenging lattices (2022)
- Deep learning background subtraction for SFX (2021)
- GAN-based diffraction pattern denoising (2023)
- Transformer-based orientation determination for single-particle XFEL (2024)
- Diffusion model for structure determination from SFX data (2024)
- Physics-informed neural network for partiality modeling (2023)
- AlphaFold-guided molecular replacement for SFX (2022)
- End-to-end deep learning SFX pipeline (2025)
- Foundation model for X-ray diffraction analysis (2025-2026)

#### Step 3: Update XFEL-SFX Solvers

After listing all XFEL-SFX solvers, update `algorithm_base/xfel_sfx/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All XFEL-SFX solvers use the data format: `y` (N, H, W) stack of 2D diffraction patterns (individual XFEL shots), `detector_geometry` (panel positions, pixel size, beam center), `beam_params` (wavelength, bandwidth, pulse energy). The `XFELSFXOperator` handles the forward model (crystal structure + random orientation -> partial Bragg reflections -> detector pattern) and processing operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for XFEL-SFX:**
- Lysozyme SFX: CrystFEL indexing rate >60%, merged data Rsplit ~8%, CC1/2 >0.99, resolution ~1.9 A
- Photosystem II SFX: resolution ~2.0 A (native data), ~3.0 A (time-resolved intermediates)
- Cheetah hit finding: >95% true positive rate, <5% false positive rate on standard datasets
- Monte Carlo integration: converges to correct structure factors with ~10,000+ indexed patterns
- Published R-factors, CC1/2, and resolution from CXIDB depositions and Nature/Science SFX papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'xfel_sfx' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/xfel_sfx/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for XFEL-SFX. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/xfel_sfx/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/xfel_sfx/standard/`

---

### X-ray Crystallography (`xray_crystallography`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray Crystallography, what dataset do you use to verify? Is this dataset used for X-ray crystallography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/xray_crystallography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original X-ray crystallography standard dataset.

**Popular datasets to consider:**
- **PDB (Protein Data Bank, Berman et al., 2000)** — the central repository for macromolecular crystal structures; >200,000 entries; provides deposited structure factors and coordinates for validation of refinement and phasing algorithms
- **CSD (Cambridge Structural Database, Groom et al., 2016)** — >1,200,000 small-molecule crystal structures; the primary reference for small-molecule crystallography algorithm validation
- **COD (Crystallography Open Database, Grazulis et al., 2009)** — open-access crystal structure database; widely used for inorganic/mineral structure validation
- **PDB-REDO (Joosten et al., 2014-2023)** — re-refined PDB entries with optimized refinement parameters; the standard for crystallographic refinement algorithm benchmarking
- **Phenix Regression Test Data** — curated set of challenging crystallographic problems for testing phasing and refinement; includes twinned crystals, pseudo-symmetry, and low-resolution cases
- **IUCr Validation Test Structures** — community-curated problematic/interesting structures for validation software testing

**Decision criteria:** PDB with deposited structure factors is the gold standard for macromolecular crystallography. CSD for small molecules. PDB-REDO for refinement benchmarking. Use the dataset most widely referenced in crystallographic methods papers (1970-2026).

#### Step 2: List All X-ray Crystallography Algorithms

Please first ensure all the X-ray Crystallography algorithms have been listed in `\pwm\public\algorithm_base\xray_crystallography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/xray_crystallography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the X-ray crystallography solvers, please update the X-ray crystallography solver.

**Key algorithms to cover (1913-2026):**

_Patterson & Heavy Atom Methods (1934-1970):_
- Patterson methods — interatomic vector map from squared structure factor magnitudes (Patterson, Phys Rev 1934) — the first general method for phase-free structure solution; used for heavy-atom location and molecular replacement
- Patterson superposition — Buerger minimum function for deconvolution (Buerger, 1959)
- Isomorphous replacement — SIR/MIR phasing using heavy-atom derivatives (Green et al., 1954; Harker, 1956)
- Anomalous scattering — Bijvoet difference for absolute configuration and phasing (Bijvoet, 1951)

_Direct Methods (1953-1990):_
- Direct methods — statistical phase determination from structure factor magnitudes (Hauptman & Karle, 1953; Nobel Prize 1985) — the dominant method for small-molecule structure solution; uses probability relationships among phases (Sayre equation, tangent formula)
- SHELXS — direct methods implementation (Sheldrick, Acta Cryst 1990; 2008) — the most widely used small-molecule structure solution program
- SIR — semi-invariants and direct methods (Burla et al., 1989-2014)
- Shake-and-Bake — dual-space direct methods for larger structures (Weeks et al., Acta Cryst 1994)
- Charge flipping — iterative phase retrieval without atomicity (Oszlanyi & Suto, Acta Cryst 2004)
- SUPERFLIP — charge flipping implementation (Palatinus & Chapuis, JAC 2007)

_Molecular Replacement (1962-2010):_
- Molecular replacement — phasing using homologous structure as search model (Rossmann & Blow, Acta Cryst 1962) — the most commonly used macromolecular phasing method; rotation and translation function search
- AMoRe — automated molecular replacement (Navaza, Acta Cryst 1994)
- Phaser — maximum likelihood molecular replacement (McCoy et al., JAC 2007) — current state-of-the-art MR program; uses log-likelihood targets for rotation and translation
- MOLREP — automated MR (Vagin & Teplyakov, 1997)
- MR-SAD — combined molecular replacement and anomalous phasing (2008)

_Experimental Phasing — MAD/SAD (1980-2015):_
- MAD phasing — Multi-wavelength Anomalous Dispersion (Hendrickson, Science 1991) — phasing using anomalous signal at multiple wavelengths near absorption edge
- SAD phasing — Single-wavelength Anomalous Dispersion (Wang, Meth Enzymol 1985; dominant since 2000) — phasing from anomalous signal at single wavelength with density modification
- SHELXC/D/E — integrated SAD/MAD phasing pipeline (Sheldrick, Acta Cryst 2010) — substructure determination (D) and phase extension (E)
- HySS — hybrid substructure search in Phenix (Grosse-Kunstleve & Adams, 2003)
- Auto-Rickshaw — automated experimental phasing pipeline (Panjikar et al., 2005)
- S-SAD — sulfur SAD for native phasing without heavy atoms (Dauter et al., 2002)

_Density Modification (1957-2010):_
- Solvent flattening — constraining solvent region to flat density (Wang, Meth Enzymol 1985)
- Histogram matching — constraining protein electron density histogram (Zhang & Main, 1990)
- NCS averaging — non-crystallographic symmetry averaging for phase improvement (Bricogne, 1976)
- DM / RESOLVE — density modification programs (Cowtan, 1994; Terwilliger, 2000)
- SHELXE — density modification and autotracing (Sheldrick, 2002; 2010)
- Iterative model building + refinement for density modification (2005)

_Refinement (1971-2020):_
- Least-squares refinement — minimization of sum(w * (Fo-Fc)^2) (Konnert & Hendrickson, Acta Cryst 1980)
- SHELXL — small-molecule and macromolecular refinement (Sheldrick, Acta Cryst 2008; 2015) — the standard for small-molecule crystallography
- ML refinement / REFMAC — maximum likelihood refinement for macromolecules (Murshudov et al., Acta Cryst 1997; REFMAC5, Murshudov et al., 2011) — widely used CCP4 refinement program
- phenix.refine — ML refinement in Phenix (Afonine et al., Acta Cryst 2012) — modern refinement engine with automated optimization
- CNS — Crystallography and NMR System, simulated annealing refinement (Brunger et al., Acta Cryst 1998)
- BUSTER — maximum likelihood refinement with local structure similarity restraints (Bricogne et al., 2011)
- Anisotropic B-factor refinement (TLS, individual anisotropic) (Winn et al., 2001)
- Occupancy refinement, riding hydrogen, disorder modeling
- PDB-REDO — automated re-refinement with optimized parameters (Joosten et al., IUCrJ 2014)
- Quantum-mechanical refinement — QM/MM for crystallographic refinement (Ryde, 2003; extended 2015)

_Automated Pipeline (2000-2020):_
- AutoBuild — automated model building into electron density (Terwilliger et al., Acta Cryst 2008)
- ARP/wARP — automated building and refinement (Langer et al., 2008)
- Buccaneer — statistical model building (Cowtan, Acta Cryst 2006)
- AutoSol — automated structure solution pipeline in Phenix (Terwilliger et al., 2009)
- CRANK2 — combined experimental phasing pipeline (Skubak & Pannu, 2013)
- XDS — X-ray Detector Software for data processing (Kabsch, Acta Cryst 2010)
- DIALS — modern data integration software (Winter et al., 2018)

_Deep Learning & AI-Guided (2019-2026):_
- AlphaFold-guided molecular replacement — using AlphaFold2/3 predicted structures as MR search models (Jumper et al., Nature 2021; applied to crystallography 2021-2026) — transformative impact on MR success rate, enabling solution of previously unsolvable structures
- ModelAngelo — deep learning automated model building (Jamali et al., Nature 2024)
- Deep learning phase prediction — direct phase estimation from magnitudes (2023)
- Neural network refinement — learned energy functions replacing classical restraints (2024)
- GNN for crystal structure prediction from diffraction (2023)
- Diffusion model for electron density (2024)
- DL-based structure validation (MolProbity + ML, 2022)
- Foundation model for crystallographic structure solution (2025)
- End-to-end deep learning crystallography pipeline (2025-2026)

#### Step 3: Update X-ray Crystallography Solvers

After listing all X-ray crystallography solvers, update `algorithm_base/xray_crystallography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All X-ray crystallography solvers use the data format: `y` (num_reflections, 3+) reflection list with (h, k, l, |F|, sigma_F) or (h, k, l, I, sigma_I), `cell_params` (a, b, c, alpha, beta, gamma), `space_group` Hermann-Mauguin or number, `wavelength` X-ray wavelength (A). The `XrayCrystOperator` handles the forward model (atomic model -> structure factors |F_calc|) and phasing/refinement operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for X-ray Crystallography:**
- PDB-REDO test set: REFMAC5/phenix.refine Rfree typically 20-30% depending on resolution; PDB-REDO improves Rfree by ~1% on average
- Direct methods (SHELXS): solves structures up to ~2000 atoms in asymmetric unit for small molecules
- Molecular replacement (Phaser): success rate >80% with homologous model (>30% sequence identity, <2 A RMSD)
- AlphaFold MR: enables solution with <20% sequence identity in many cases
- Published R-factors, Rfree, and resolution from PDB depositions and IUCr journals

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'xray_crystallography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/xray_crystallography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/xray_crystallography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/xray_crystallography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for X-ray Crystallography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/xray_crystallography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_crystallography/standard/`

---

### XRF Tomography (`xrf_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray Fluorescence Tomography, what dataset do you use to verify? Is this dataset used for XRF-CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/xrf_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original XRF-CT standard dataset.

**Popular datasets to consider:**
- **Synchrotron XRF-CT Phantom Datasets (De Jonge et al., 2010; Boisseau et al., 1986-2020)** — XRF-CT data from synchrotron microprobes (APS, ESRF, PETRA-III) with known multi-element phantom compositions; the primary benchmark for XRF-CT reconstruction algorithms
- **APS XRF Imaging Data (Vogt et al., 2003-2023)** — X-ray fluorescence microscopy and tomography datasets from the Advanced Photon Source; includes biological, environmental, and materials science specimens with well-characterized elemental compositions
- **ID16B ESRF Nano-XRF Data (Martinez-Criado et al., 2016-2023)** — high-resolution nano-XRF tomography data from the European Synchrotron; used for nanoscale elemental mapping validation
- **Simulated XRF-CT Data (La Riviere et al., 2006; Schroer, 2001)** — Monte Carlo-generated XRF-CT datasets with known element distributions and self-absorption effects; essential for algorithm development with exact ground truth
- **ANKA/DESY XRF-CT Data (2010-2020)** — multi-element XRF tomography from German synchrotrons; used for self-absorption correction validation

**Decision criteria:** Synchrotron XRF-CT phantom data with known elemental compositions is the gold standard for reconstruction validation. Simulated data with self-absorption for quantitative algorithm assessment. Use the dataset most widely referenced in XRF-CT reconstruction papers (2000-2026).

#### Step 2: List All XRF Tomography Algorithms

Please first ensure all the XRF Tomography algorithms have been listed in `\pwm\public\algorithm_base\xrf_tomo\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/xrf_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the XRF-CT solvers, please update the XRF-CT solver.

**Key algorithms to cover (1986-2026):**

_Classical & Analytic (1986-2005):_
- FBP for XRF-CT — filtered back-projection applied to XRF sinograms assuming negligible self-absorption (Boisseau & Grodzins, Hyperfine Interact 1986) — the baseline XRF-CT reconstruction; simple but neglects attenuation of incident and fluorescent X-rays
- Simple line-integral model — XRF signal proportional to integrated element concentration along beam path (1990s)
- Attenuation-corrected FBP for XRF-CT — incorporating incident beam and fluorescence attenuation (Hogan et al., IEEE TMI 1991)
- MLEM for XRF-CT — maximum likelihood expectation maximization with fluorescence forward model (La Riviere, IEEE TMI 2004; Schroer, Appl Phys Lett 2001)

_Self-Absorption Correction & Iterative (2005-2016):_
- Self-absorption correction algorithms — iterative correction for attenuation of both incident and fluorescent X-rays within the sample (De Jonge & Vogt, PNAS 2010; McNear et al., 2005) — critical for quantitative XRF-CT of dense/high-Z samples
- Absorption-corrected MLEM — iterative reconstruction with explicit self-absorption forward model (La Riviere & Vargas, IEEE TMI 2006)
- Monte Carlo forward model for XRF-CT — GEANT4/Penelope simulation of full XRF physics including scattering and fluorescence cascade (2010)
- TV-regularized XRF-CT reconstruction — total variation penalty for sparse-element and sparse-angle XRF-CT (Gürsoy et al., J Synchrotron Rad 2015)
- Simultaneous XRF + transmission CT reconstruction — joint inversion of fluorescence and attenuation data (Vekemans et al., 2004; extended 2012)
- Sparse-angle XRF-CT — compressed sensing for reducing scan time in XRF tomography (2014)
- 3D confocal XRF — depth-resolved XRF imaging without rotation (Kanngiesser et al., 2003; tomographic extension 2010)
- Compton scatter correction for XRF-CT (2013)
- Multi-element simultaneous XRF-CT reconstruction (2016)

_Deep Learning & Modern (2017-2026):_
- CNN-based XRF-CT artifact removal — U-Net denoising/artifact correction for XRF-CT images (2022)
- Deep learning sparse-angle XRF-CT — learned reconstruction from limited projections (2023)
- Physics-informed neural network for XRF-CT with self-absorption (2024)
- GAN-based XRF-CT super-resolution — enhancing spatial resolution of elemental maps (2023)
- Deep unfolding for iterative XRF-CT (2024)
- Autoencoder for XRF spectral decomposition in XRF-CT (2022)
- Transfer learning from transmission CT to XRF-CT (2023)
- Transformer-based multi-element XRF-CT reconstruction (2025)
- Diffusion-prior XRF-CT reconstruction (2025)
- Foundation model for synchrotron X-ray imaging including XRF-CT (2025-2026)

#### Step 3: Update XRF Tomography Solvers

After listing all XRF-CT solvers, update `algorithm_base/xrf_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All XRF-CT solvers use the data format: `y` (num_angles, num_positions, num_elements) XRF sinogram data — fluorescence intensity for each element at each scan position and angle, `angles` array of rotation angles, `incident_energy` (keV), `element_lines` list of fluorescence line energies. The `XRFTomoOperator` handles the forward model (element concentration map -> self-absorbed fluorescence signal -> detector counts) and inverse reconstruction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for XRF Tomography:**
- Synchrotron XRF-CT phantom: FBP without self-absorption correction shows ~20-50% concentration error for high-Z elements; with correction <10%
- MLEM with self-absorption: element concentration accuracy within 5% for known phantoms
- Sparse-angle (30-60 angles): FBP severely degraded, TV-regularized ~5 dB improvement
- Multi-element reconstruction: simultaneous recovery of 5+ elements with known concentration ratios
- Published concentration accuracy and spatial resolution from APS, ESRF, and PETRA-III papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'xrf_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/xrf_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for XRF Tomography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/xrf_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_tomo/standard/`
