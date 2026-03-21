
---

## Medical Imaging Extended — Modality Templates

---

### PET (`pet`) Modality Template

#### Step 1: Verify Standard Dataset

For PET, what dataset do you use to verify? Is this dataset used for PET popular algorithms? Please ensure the standard dataset in `datasets/benchmark/pet/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original PET standard dataset.

**Popular datasets to consider:**
- **Ultra-low-dose PET Imaging Challenge (Sanaat et al., 2022)** — paired full-dose / low-dose brain PET; canonical benchmark for PET denoising and reconstruction
- **ADNI PET (Alzheimer's Disease Neuroimaging Initiative)** — FDG and amyloid PET brain scans with clinical labels; widely used for quantitative PET evaluation
- **Siemens Biograph Vision Phantom Dataset** — NEMA IEC body phantom acquisitions with known activity concentrations; used for resolution and quantitative accuracy benchmarks
- **TCIA Head-and-Neck PET/CT (Vallières et al., 2017)** — multi-center PET/CT with tumor segmentation ground truth
- **Helsinki Ultra-Low-Dose PET Dataset (Mehranian et al., 2020)** — 1% and 5% count-reduced PET sinograms with full-dose reference

**Decision criteria:** The Ultra-low-dose PET Challenge dataset is the current gold standard for PET reconstruction benchmarking (2022–2026). ADNI PET is the most widely cited clinical PET dataset. Use the dataset that appears in the largest number of PET reconstruction papers.

#### Step 2: List All PET Algorithms

Please first ensure all the PET algorithms have been listed in `\pwm\public\algorithm_base\pet\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/pet. Besides, you need to search all algorithms from 1950 to 2026. After listing all the PET solvers, please update the PET solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1960s–2005):_
- Filtered Back-Projection (FBP) — analytic PET reconstruction baseline (Brooks & Di Chiro, 1976)
- FBP-3DRP — 3D Reprojection Algorithm (Kinahan & Rogers, 1989)
- Single-Slice Rebinning (SSRB) — 3D to 2D sinogram rebinning (Daube-Witherspoon & Muehllehner, 1987)
- Fourier Rebinning (FORE) — frequency-domain rebinning for 3D PET (Defrise et al., 1997)
- MLEM — Maximum Likelihood Expectation Maximization (Shepp & Vardi, 1982)
- OSEM — Ordered Subsets EM (Hudson & Larkin, 1994) — the dominant clinical PET reconstruction algorithm

_Optimization / Model-Based (2000s–2016):_
- MAP-EM — Maximum A Posteriori EM with Gaussian MRF prior (Levitan & Herman, 1987; Green, 1990)
- PSF-OSEM — Resolution modeling / point-spread-function OSEM (Panin et al., 2006; Reader et al., 2003)
- BSREM — Block Sequential Regularized EM (De Pierro & Yamagishi, 2001) — used in GE Q.Clear
- TOF-OSEM — Time-of-Flight OSEM (Conti, 2006; Surti et al., 2007)
- Kernel EM — Kernel-based PET reconstruction using MR side information (Wang & Qi, 2015)
- Total Variation regularized PET (Sawatzky et al., 2008)
- Joint PET-MR reconstruction with anatomical priors (Bowsher et al., 2004; Ehrhardt et al., 2015)
- Penalized Weighted Least Squares (PWLS) for PET (Fessler, 1994)

_Deep Learning (2017–2026):_
- DeepPET — CNN-based PET image denoising (Häggström et al., 2019)
- DIP-PET — Deep Image Prior for PET (Gong et al., 2019)
- FBSEM-Net — learned regularization for PET (Mehranian & Reader, 2020)
- Unrolled MLEM-Net for PET reconstruction (Lim et al., 2020)
- TransPET — Transformer for PET denoising (Zhang et al., 2022)
- Score-based diffusion for PET denoising (Xie & Li, 2022)
- MAPEM-Net — unrolled MAP-EM network (Xiang et al., 2021)
- Federated learning for multi-site low-dose PET (Guo et al., 2023)
- Foundation model for PET reconstruction (2025)

#### Step 3: Update PET Solvers

After listing all PET solvers, update `algorithm_base/pet/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All PET solvers use the data format: `y` (num_angles, num_detectors) sinogram data or (num_LORs,) list-mode data. The `PETOperator` handles the system matrix forward projection `y = A * x` with attenuation, normalization, scatter, and randoms corrections.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for PET:**
- NEMA IEC phantom: FBP ~25.0 dB, OSEM ~32.0 dB, PSF-OSEM ~34.0 dB, BSREM ~35.5 dB
- Ultra-low-dose brain (5% counts): OSEM ~28.0 dB, DeepPET ~34.0 dB, MAPEM-Net ~36.0 dB
- Ultra-low-dose brain (1% counts): OSEM ~22.0 dB, DIP-PET ~30.0 dB, Score-based ~33.0 dB
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'pet' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/pet/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/pet/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/pet/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for PET. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/pet/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/pet/standard/`

---

### SPECT (`spect`) Modality Template

#### Step 1: Verify Standard Dataset

For SPECT, what dataset do you use to verify? Is this dataset used for SPECT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spect/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SPECT standard dataset.

**Popular datasets to consider:**
- **Jaszczak Phantom Dataset (Data Spectrum Corp.)** — standardized SPECT phantom with known hot-rod and cold-sphere inserts; canonical quality control benchmark
- **SIMIND Monte Carlo SPECT (Ljungberg & Strand, 1989)** — simulated SPECT projection data with exact ground truth; widely used for reconstruction validation
- **Multi-Centre SPECT Calibration Dataset (Dewaraja et al., 2012)** — NIST-traceable SPECT/CT quantitative imaging phantom data
- **DaTSCAN Parkinson SPECT (PPMI, Marek et al., 2011)** — dopamine transporter SPECT brain imaging; the largest clinical SPECT dataset
- **Cardiac SPECT Dataset (Slomka et al., 2017)** — myocardial perfusion SPECT with stress/rest protocols

**Decision criteria:** SIMIND Monte Carlo phantoms provide exact ground truth for reconstruction algorithm comparison. DaTSCAN/PPMI is the most widely used clinical SPECT dataset.

#### Step 2: List All SPECT Algorithms

Please first ensure all the SPECT algorithms have been listed in `\pwm\public\algorithm_base\spect\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spect. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SPECT solvers, please update the SPECT solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1960s–2000):_
- FBP with Ramp filter — standard analytic SPECT reconstruction (Shepp & Logan, 1974)
- Chang attenuation correction — first-order uniform attenuation correction (Chang, 1978)
- MLEM for SPECT — iterative ML reconstruction (Shepp & Vardi, 1982; Lange & Carson, 1984)
- OSEM for SPECT — ordered subsets EM (Hudson & Larkin, 1994)
- Frequency-Distance Relation (FDR) — depth-dependent resolution compensation in FBP (Edholm et al., 1986)

_Optimization / Model-Based (2000s–2016):_
- OSEM with CDR — collimator-detector response modeling (Zeng et al., 1991; Frey & Tsui, 1996)
- MAP-OSEM with anatomical priors — MRI/CT-guided SPECT (Bowsher et al., 1996)
- Dual-matrix OSEM — separate system matrices for forward/back projection (Kamphuis et al., 1998; Zeng & Gullberg, 2000)
- RBI-EM — Rescaled Block-Iterative EM (Byrne, 1998)
- TV-regularized SPECT reconstruction (Bruyant, 2002)
- SAGE — Space-Alternating Generalized EM for SPECT (Fessler & Hero, 1994)
- Monte Carlo–based system matrix OSEM (Beekman et al., 2002)
- OSEM with scatter correction — Triple Energy Window (TEW) and ESSE methods (Ogawa et al., 1991; Frey & Tsui, 1996)

_Deep Learning (2017–2026):_
- U-Net denoising for low-count SPECT (Shiri et al., 2020)
- DuDoNet-SPECT — dual-domain SPECT denoising (Xiang et al., 2021)
- Learned primal-dual for SPECT (Adler & Öktem, 2018, applied to SPECT)
- Physics-informed DL-SPECT reconstruction (Ryden et al., 2021)
- Unrolled OSEM network for SPECT (Lim et al., 2020)
- Score-based generative models for SPECT (2023)
- CycleGAN for SPECT-to-PET translation (Pan et al., 2022)
- Foundation model for SPECT reconstruction (2025)

#### Step 3: Update SPECT Solvers

After listing all SPECT solvers, update `algorithm_base/spect/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SPECT solvers use the data format: `y` (num_angles, H, W) projection data with collimator geometry. The `SPECTOperator` handles the system matrix forward projection `y = C * A * R * x` incorporating collimator response C, attenuation A, and resolution R modeling.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SPECT:**
- Jaszczak phantom: FBP ~24.0 dB, OSEM ~30.0 dB, OSEM+CDR ~33.0 dB, MAP-OSEM ~34.5 dB
- DaTSCAN brain: FBP ~22.0 dB, OSEM ~28.0 dB, DL-denoised ~33.0 dB
- Cardiac perfusion (half-count): OSEM ~26.0 dB, DuDoNet ~32.0 dB
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spect' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spect/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spect/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spect/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SPECT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spect/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spect/standard/`

---

### Spectral CT (`spectral_ct`) Modality Template

#### Step 1: Verify Standard Dataset

For Spectral CT, what dataset do you use to verify? Is this dataset used for Spectral CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spectral_ct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Spectral CT standard dataset.

**Popular datasets to consider:**
- **Spectral CT Simulation Toolkit (Mechlem et al., 2018)** — physics-based photon-counting CT simulation with exact material ground truth; widely used for material decomposition benchmarking
- **AAPM Spectral CT Grand Challenge (2022)** — standardized dual-energy and photon-counting CT reconstruction tasks with reference images
- **Siemens NAEOTOM Alpha Clinical Dataset** — first FDA-cleared photon-counting CT data; multi-energy bins with virtual monoenergetic and material maps
- **FORBILD Thorax Phantom (multi-energy extension)** — digital phantom extended for spectral CT with multiple material basis functions

**Decision criteria:** The AAPM Spectral CT Challenge dataset or Mechlem simulation toolkit are the standard benchmarks for photon-counting spectral CT. Use datasets with known material composition ground truth.

#### Step 2: List All Spectral CT Algorithms

Please first ensure all the Spectral CT algorithms have been listed in `\pwm\public\algorithm_base\spectral_ct\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spectral_ct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Spectral CT solvers, please update the Spectral CT solver.

**Key algorithms to cover (1950–2026):**

_Classical / Pre-Processing (1970s–2005):_
- Dual-energy decomposition — Alvarez & Macovski two-basis material decomposition (Alvarez & Macovski, 1976)
- Image-domain material decomposition — post-reconstruction basis material mapping (Kalender et al., 1986)
- Projection-domain material decomposition — pre-reconstruction sinogram decomposition (Engler & Friedman, 1990)
- Filtered back-projection per energy bin — independent FBP of each spectral channel

_Optimization / Model-Based (2005–2016):_
- Joint statistical iterative material decomposition (Long & Fessler, 2014)
- One-step spectral CT (Barber et al., 2016) — simultaneous reconstruction and decomposition
- PWLS spectral CT with edge-preserving regularization (Niu et al., 2014)
- Empirical dual-energy calibration (Stenner et al., 2007)
- Multi-material decomposition (Mendonça et al., 2014)
- Tensor dictionary learning for spectral CT (Semerci et al., 2014)
- Total nuclear variation for joint spectral CT (Rigie & La Rivière, 2015)
- ADMM-based spectral CT decomposition (Mechlem et al., 2018)
- Cramér-Rao lower bound analysis for spectral CT (Roessl & Proksa, 2007)

_Deep Learning (2017–2026):_
- Butterfly-Net for spectral CT material decomposition (Zhang et al., 2019)
- LEARN — Learned Experts' Assessment-based Reconstruction Network for spectral CT (Chen et al., 2018)
- DuDo-SS — dual-domain self-supervised spectral CT (Wu et al., 2021)
- DOLCE — diffusion-based spectral CT reconstruction (Liu et al., 2023)
- Material-decomposition U-Net (Clark et al., 2020)
- Unrolled one-step spectral CT network (Abascal et al., 2021)
- Implicit neural representation for spectral CT (Reed et al., 2023)
- Foundation model for multi-energy CT (2025)

#### Step 3: Update Spectral CT Solvers

After listing all Spectral CT solvers, update `algorithm_base/spectral_ct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Spectral CT solvers use the data format: `y` (num_energies, num_angles, num_detectors) multi-energy sinogram data. The `SpectralCTOperator` handles the energy-dependent forward model `y_e = integral{ S_e(E) * exp(-sum_m mu_m(E) * A_m * x_m) dE }` with spectral response functions S_e and mass attenuation coefficients mu_m.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Spectral CT:**
- AAPM Challenge phantom: FBP per bin ~26.0 dB, Joint PWLS ~32.0 dB, One-step ~34.0 dB, DOLCE ~37.0 dB
- Material decomposition RMSE (mg/mL): Classical ~5.0, Joint statistical ~2.5, DL-based ~1.2
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spectral_ct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spectral_ct/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spectral_ct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spectral_ct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Spectral CT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spectral_ct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spectral_ct/standard/`

---

### Functional MRI (`fmri`) Modality Template

#### Step 1: Verify Standard Dataset

For fMRI, what dataset do you use to verify? Is this dataset used for fMRI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/fmri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original fMRI standard dataset.

**Popular datasets to consider:**
- **HCP fMRI (Human Connectome Project, Van Essen et al., 2013)** — high-resolution resting-state and task fMRI with dense temporal sampling; the gold standard for fMRI analysis
- **OpenNeuro / OpenfMRI (Poldrack & Gorgolewski, 2017)** — large collection of shared fMRI datasets in BIDS format; widely used for reproducibility studies
- **UK Biobank Brain Imaging (Miller et al., 2016)** — 100,000+ subject brain MRI/fMRI dataset; largest population fMRI study
- **ABCD Study (Casey et al., 2018)** — adolescent brain cognitive development fMRI dataset
- **fastMRI+ Brain fMRI (NYU, 2022)** — accelerated fMRI k-space data with temporal dynamics

**Decision criteria:** HCP is the undisputed gold standard for fMRI reconstruction and analysis benchmarking. OpenNeuro provides the broadest collection of task-based fMRI paradigms.

#### Step 2: List All fMRI Algorithms

Please first ensure all the fMRI algorithms have been listed in `\pwm\public\algorithm_base\fmri\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/fmri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the fMRI solvers, please update the fMRI solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2005):_
- Sliding window reconstruction — temporal interpolation for dynamic MRI (Riederer et al., 1988)
- UNFOLD — Unaliasing by Fourier-encoding the Overlaps Using the temporal Dimension (Madore et al., 1999)
- k-t BLAST / k-t SENSE — k-t space acceleration for dynamic MRI (Tsao et al., 2003)
- Keyhole imaging — central k-space update for temporal resolution (van Vaals et al., 1993)
- GRAPPA-based fMRI acceleration (Griswold et al., 2002; Polimeni et al., 2006)

_Optimization / Model-Based (2005–2016):_
- k-t FOCUSS — k-t compressed sensing for dynamic MRI (Jung et al., 2009)
- k-t SPARSE-SENSE — joint parallel imaging + CS for fMRI (Otazo et al., 2010)
- Low-Rank + Sparse (L+S) for dynamic fMRI (Otazo et al., 2015)
- PROUD — PaRallel imaging and cOmpressed sensing Using Dictionaries (Doneva et al., 2010)
- PS-Sparse — partially separable model for fMRI (Zhao et al., 2012)
- MB-SENSE — Multi-Band SENSE for simultaneous multi-slice fMRI (Setsompop et al., 2012)
- MICA — Multi-scale Image Constraint for Accelerated fMRI (Chiew et al., 2015)
- Structured low-rank matrix completion for fMRI (Shin et al., 2014)

_Deep Learning (2017–2026):_
- MANTIS — Multi-scale Accelerated Neural-net for Temporal Imaging Sequences (Wang et al., 2020)
- CIRCUS — Cascaded Iteratively Refined U-net for fMRI (Zeng et al., 2021)
- Deep-J-SENSE for accelerated fMRI (Heckel et al., 2020)
- fMRI-Transformer — spatiotemporal transformer for fMRI reconstruction (2023)
- Score-based diffusion for dynamic MRI (Chung et al., 2022)
- Implicit neural representation for fMRI time series (Shen et al., 2022)
- SSL-fMRI — self-supervised learning for fMRI reconstruction (Yaman et al., 2020)
- Foundation model for dynamic MRI (2025)

#### Step 3: Update fMRI Solvers

After listing all fMRI solvers, update `algorithm_base/fmri/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All fMRI solvers use the data format: `y` (num_coils, num_frames, H, W) time-series multi-coil k-space data, with temporal undersampling masks. The `fMRIOperator` handles the spatiotemporal forward model `y_t = mask_t * F * S * x_t` with frame-dependent undersampling patterns and multi-band excitation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for fMRI:**
- HCP fMRI R=4: Zero-filled ~26.0 dB, GRAPPA ~31.0 dB, L+S ~34.0 dB, MANTIS ~36.5 dB
- HCP fMRI R=8: Zero-filled ~23.0 dB, k-t SPARSE-SENSE ~30.0 dB, DL-based ~34.0 dB
- Temporal SNR preservation: GRAPPA >85%, L+S >90%, DL-based >93%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'fmri' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/fmri/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/fmri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/fmri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for fMRI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/fmri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/fmri/standard/`

---

### Diffusion MRI (`diffusion_mri`) Modality Template

#### Step 1: Verify Standard Dataset

For Diffusion MRI, what dataset do you use to verify? Is this dataset used for Diffusion MRI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/diffusion_mri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Diffusion MRI standard dataset.

**Popular datasets to consider:**
- **HCP Diffusion MRI (Sotiropoulos et al., 2013)** — multi-shell (b=1000/2000/3000) diffusion data with 270 directions; gold standard for diffusion reconstruction and tractography
- **ISMRM Tractography Challenge (Maier-Hein et al., 2017)** — synthetic DWI phantom with known ground-truth fiber bundles; canonical tractography benchmark
- **Fibercup Phantom (Poupon et al., 2008)** — physical phantom with known fiber crossings for diffusion reconstruction validation
- **dHCP (developing Human Connectome Project)** — neonatal diffusion MRI with multi-shell acquisitions
- **MASSIVE Dataset (Froeling et al., 2017)** — 1260-direction diffusion dataset of a single subject for angular super-resolution benchmarking

**Decision criteria:** HCP diffusion data is the gold standard for diffusion MRI reconstruction. ISMRM Tractography Challenge for fiber tracking evaluation.

#### Step 2: List All Diffusion MRI Algorithms

Please first ensure all the Diffusion MRI algorithms have been listed in `\pwm\public\algorithm_base\diffusion_mri\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/diffusion_mri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Diffusion MRI solvers, please update the Diffusion MRI solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2005):_
- DTI — Diffusion Tensor Imaging, linear least-squares fit (Basser et al., 1994)
- ADC mapping — Apparent Diffusion Coefficient from Stejskal-Tanner equation (Stejskal & Tanner, 1965)
- IVIM — Intra-Voxel Incoherent Motion bi-exponential model (Le Bihan et al., 1988)
- DKI — Diffusion Kurtosis Imaging, quadratic extension of DTI (Jensen et al., 2005)
- Spherical Harmonics / Q-Ball ODF estimation (Tuch, 2004; Descoteaux et al., 2007)

_Model-Based / Optimization (2005–2016):_
- CSD — Constrained Spherical Deconvolution for fiber ODF estimation (Tournier et al., 2007)
- DSI — Diffusion Spectrum Imaging, Fourier transform of q-space (Wedeen et al., 2005)
- SHORE — Simple Harmonic Oscillator-based Reconstruction and Estimation (Özarslan et al., 2009)
- MAP-MRI — Mean Apparent Propagator MRI (Özarslan et al., 2013)
- NODDI — Neurite Orientation Dispersion and Density Imaging (Zhang et al., 2012)
- DIAMOND — Distribution of 3D Anisotropic Microstructural environments (Scherrer et al., 2016)
- SMT — Spherical Mean Technique for microstructure (Kaden et al., 2016)
- Joint DWI super-resolution and denoising (Ye et al., 2016)
- MPPCA denoising for DWI (Veraart et al., 2016)

_Deep Learning (2017–2026):_
- q-DL — deep learning for q-space undersampled diffusion MRI (Golkov et al., 2016)
- DeepDTI — deep learning DTI from sparse data (Tian et al., 2020)
- MESC-Net — multi-scale encoder for diffusion microstructure (Ye et al., 2020)
- Patch2Self — self-supervised DWI denoising (Fadnavis et al., 2020)
- ESD — Equivariant Spherical Deconvolution (Elaldi et al., 2021)
- TractGeoNet — geometric deep learning for tractography (2022)
- Diffusion-transformer for angular super-resolution (2023)
- Foundation model for diffusion MRI (2025)

#### Step 3: Update Diffusion MRI Solvers

After listing all Diffusion MRI solvers, update `algorithm_base/diffusion_mri/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Diffusion MRI solvers use the data format: `y` (num_directions, H, W, D) diffusion-weighted volumes with b-values and b-vectors. The `DiffusionMRIOperator` handles the signal model `y_i = S0 * exp(-b_i * g_i^T D g_i)` for DTI or the multi-compartment forward models for advanced microstructure estimation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Diffusion MRI:**
- HCP DTI (6-dir from 90-dir): Linear LS ~30.0 dB, DeepDTI ~36.0 dB
- ISMRM Tractography: CSD valid-bundles ~55%, NODDI microstructure RMSE ~0.05
- Angular super-resolution (15-dir to 90-dir): SH interpolation ~28.0 dB, q-DL ~34.0 dB, Transformer ~36.0 dB
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'diffusion_mri' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/diffusion_mri/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Diffusion MRI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/diffusion_mri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/diffusion_mri/standard/`

---

### ASL MRI (`asl_mri`) Modality Template

#### Step 1: Verify Standard Dataset

For ASL MRI, what dataset do you use to verify? Is this dataset used for ASL MRI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/asl_mri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ASL MRI standard dataset.

**Popular datasets to consider:**
- **ASL Digital Reference Object (DRO, Lorenz et al., 2018)** — standardized simulation phantom with known perfusion ground truth; recommended by ISMRM Perfusion Study Group
- **OASIS ASL Dataset** — multi-site ASL brain perfusion data with demographic metadata
- **HCP ASL Pilot Data (Fan et al., 2017)** — high-resolution multi-PLD pCASL from the Human Connectome Project
- **ExploreASL Test Dataset (Mutsaerts et al., 2020)** — multi-site ASL data bundled with the ExploreASL toolbox for reproducibility validation

**Decision criteria:** The ASL DRO is the standard benchmark for perfusion quantification algorithms with exact ground truth. HCP ASL provides the highest quality in-vivo ASL data.

#### Step 2: List All ASL MRI Algorithms

Please first ensure all the ASL MRI algorithms have been listed in `\pwm\public\algorithm_base\asl_mri\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/asl_mri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ASL MRI solvers, please update the ASL MRI solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2005):_
- Pairwise subtraction — basic control-label difference for perfusion estimation (Williams et al., 1992)
- Single-compartment kinetic model — Buxton general kinetic model (Buxton et al., 1998)
- QUIPSS / QUIPSS II — quantitative imaging of perfusion using a single subtraction (Wong et al., 1998)
- Multi-PLD fitting — multi-delay ASL for bolus arrival time estimation (Dai et al., 2012)
- Surround subtraction — sinc interpolation between control/label pairs (Lu et al., 2006)

_Optimization / Model-Based (2005–2016):_
- Bayesian inference for ASL — BASIL / FABBER framework (Chappell et al., 2009)
- Partial volume correction for ASL — linear regression and kernel methods (Asllani et al., 2008)
- Multi-compartment ASL models — two-compartment exchange model (Parkes & Tofts, 2002)
- Vessel-encoded ASL — territorial perfusion mapping (Wong, 2007)
- Spatial regularization of CBF maps — adaptive spatial smoothing (Groves et al., 2009)
- Hadamard-encoded multi-PLD ASL (Teeuwisse et al., 2014)
- Model-free deconvolution ASL (Petersen et al., 2006)
- Background suppression optimization (Garcia et al., 2005)

_Deep Learning (2017–2026):_
- DeepASL — CNN for perfusion quantification from ASL (Ulas et al., 2018)
- U-Net denoising for ASL CBF maps (Xie et al., 2020)
- DL-based partial volume correction for ASL (Chen et al., 2021)
- Physics-informed neural network for ASL kinetic modeling (2022)
- Self-supervised ASL denoising (Kim et al., 2022)
- ASL super-resolution using diffusion models (2023)
- Foundation model for perfusion MRI (2025)

#### Step 3: Update ASL MRI Solvers

After listing all ASL MRI solvers, update `algorithm_base/asl_mri/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ASL MRI solvers use the data format: `y` (num_PLDs, num_averages, H, W) control-label difference images or raw ASL time-series. The `ASLOperator` handles the kinetic forward model `deltaM(t) = 2 * M0 * f * alpha * c(t)` with Buxton kinetic curve c(t) incorporating bolus arrival time, labeling duration, and T1 relaxation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for ASL MRI:**
- ASL DRO (CBF estimation RMSE in mL/100g/min): Pairwise subtraction ~12.0, BASIL ~6.0, DeepASL ~3.5
- ASL DRO (ATT estimation RMSE in ms): Single-PLD N/A, Multi-PLD ~200, BASIL ~120, DL-based ~80
- HCP ASL SNR: Raw ~2.0, Averaged ~5.0, DL-denoised ~10.0
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'asl_mri' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/asl_mri/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/asl_mri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/asl_mri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ASL MRI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/asl_mri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/asl_mri/standard/`

---

### CEST MRI (`cest_mri`) Modality Template

#### Step 1: Verify Standard Dataset

For CEST MRI, what dataset do you use to verify? Is this dataset used for CEST MRI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cest_mri/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CEST MRI standard dataset.

**Popular datasets to consider:**
- **CEST Phantom Dataset (Zaiss et al., 2018)** — standardized egg-white and BSA phantoms with known exchange parameters; used for Lorentzian fitting validation
- **ISMRM CEST Challenge Data (Herz et al., 2019)** — multi-site, multi-vendor CEST data for reproducibility benchmarking
- **Bloch-McConnell Simulation Toolkit (Zaiss & Bachert, 2013)** — numerical simulations with exact ground truth Z-spectra for CEST quantification
- **Clinical GBM APT-CEST Dataset (Togao et al., 2014)** — amide proton transfer CEST data for glioblastoma grading

**Decision criteria:** The ISMRM CEST Challenge dataset is the community standard for CEST quantification benchmarking. Bloch-McConnell simulations provide exact ground truth.

#### Step 2: List All CEST MRI Algorithms

Please first ensure all the CEST MRI algorithms have been listed in `\pwm\public\algorithm_base\cest_mri\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cest_mri. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CEST MRI solvers, please update the CEST MRI solver.

**Key algorithms to cover (1950–2026):**

_Classical / Asymmetry-Based (1990s–2010):_
- MTR asymmetry — Magnetization Transfer Ratio asymmetry analysis (Zhou et al., 2003)
- APT-weighted imaging — Amide Proton Transfer (Zhou et al., 2003)
- B0 correction with WASSR — Water Saturation Shift Referencing (Kim et al., 2009)
- Saturation time and power optimization (Sun et al., 2005)

_Model-Based / Fitting (2010–2020):_
- Multi-pool Lorentzian fitting — decomposition of Z-spectrum into Lorentzian peaks (Zaiss et al., 2014)
- Bloch-McConnell equation fitting — full numerical solution for exchange parameters (Zaiss & Bachert, 2013)
- AREX — Apparent Exchange-dependent Relaxation (Zaiss et al., 2014) — spillover-corrected CEST metric
- EMR — Extrapolated MTR reference for isolation of CEST effect (Heo et al., 2016)
- QUESP/QUEST — quantification of exchange using saturation power/time (McMahon et al., 2006)
- Polynomial and spline-based Z-spectrum fitting (Windschuh et al., 2015)
- Multi-echo CEST — MEGA-CEST for improved sensitivity (Xu et al., 2016)
- Inverse Z-spectrum analysis (Zaiss et al., 2018)
- CERT — Chemical Exchange Rotation Transfer (Zu et al., 2012)

_Deep Learning (2017–2026):_
- DeepCEST — neural network for rapid CEST quantification (Zaiss et al., 2019)
- CNN-based CEST denoising and B0 correction (Kang et al., 2021)
- CEST fingerprinting with deep learning (Cohen et al., 2018)
- Physics-informed CEST network (Glang et al., 2020)
- Self-supervised CEST reconstruction (2022)
- Transformer-based Z-spectrum decomposition (2023)
- Foundation model for CEST quantification (2025)

#### Step 3: Update CEST MRI Solvers

After listing all CEST MRI solvers, update `algorithm_base/cest_mri/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CEST MRI solvers use the data format: `y` (num_offsets, H, W) Z-spectrum images at different saturation frequency offsets. The `CESTOperator` handles the Bloch-McConnell equations for multi-pool chemical exchange: `dM/dt = A*M + C` with exchange matrix coupling water and solute pools.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CEST MRI:**
- ISMRM Challenge (APT quantification RMSE %): MTR asym ~25%, Lorentzian ~10%, AREX ~7%, DeepCEST ~3%
- Bloch-McConnell phantom (exchange rate RMSE): Lorentzian ~30%, Bloch fit ~10%, DL-based ~5%
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cest_mri' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cest_mri/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cest_mri/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cest_mri/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CEST MRI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cest_mri/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cest_mri/standard/`

---

### Susceptibility-Weighted Imaging (`swi`) Modality Template

#### Step 1: Verify Standard Dataset

For SWI, what dataset do you use to verify? Is this dataset used for SWI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/swi/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SWI standard dataset.

**Popular datasets to consider:**
- **QSM Challenge Dataset (Langkammer et al., 2018)** — multi-echo gradient-echo brain MRI with COSMOS ground truth; the canonical QSM/SWI benchmark
- **QSM Challenge 2.0 (Bilgic et al., 2021)** — updated challenge with calcification and hemorrhage phantoms
- **Cornell QSM Brain Dataset (Wang & Liu, 2015)** — in-vivo brain phase data with COSMOS reference maps
- **Simulated QSM Phantom (Marques & Bowtell, 2005)** — numerical phantom with known susceptibility distribution

**Decision criteria:** The QSM Challenge datasets are the community gold standard for susceptibility mapping evaluation. Use datasets with COSMOS reference maps for quantitative validation.

#### Step 2: List All SWI Algorithms

Please first ensure all the SWI algorithms have been listed in `\pwm\public\algorithm_base\swi\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/swi. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SWI solvers, please update the SWI solver.

**Key algorithms to cover (1950–2026):**

_Classical / Phase Processing (1990s–2010):_
- SWI phase mask multiplication — original SWI processing (Haacke et al., 2004; Reichenbach et al., 1997)
- Homodyne high-pass phase filtering (Noll et al., 1991; Haacke et al., 2004)
- Phase unwrapping — Laplacian-based unwrapping (Schofield & Zhu, 2003)
- SHARP — Sophisticated Harmonic Artifact Reduction for Phase data (Schweser et al., 2011)
- PDF — Projection onto Dipole Fields for background removal (Liu et al., 2011)

_Model-Based / QSM (2010–2020):_
- TKD — Truncated K-space Division for QSM (Shmueli et al., 2009; Wharton et al., 2010)
- COSMOS — Calculation Of Susceptibility through Multiple Orientation Sampling (Liu et al., 2009)
- MEDI — Morphology Enabled Dipole Inversion (Liu et al., 2012)
- iLSQR — iterative LSQR for QSM (Li et al., 2011)
- TV-regularized QSM — Total Variation dipole inversion (Bilgic et al., 2014)
- STI-Suite — Susceptibility Tensor Imaging (Li et al., 2012)
- HEIDI — Homogeneity Enabled Incremental Dipole Inversion (Schweser et al., 2012)
- FANSI — Fast Algorithm for Nonlinear Susceptibility Inversion (Milovic et al., 2018)
- R2* mapping from multi-echo GRE (Fernández-Seara & Wehrli, 2000)

_Deep Learning (2017–2026):_
- QSMnet — deep learning QSM from single-orientation (Yoon et al., 2018)
- DeepQSM — U-Net for dipole inversion (Bollmann et al., 2019)
- xQSM — explainable QSM network (Gao et al., 2021)
- LPCNN — Laplacian pyramid CNN for QSM (Wei et al., 2020)
- NeXtQSM — next-generation QSM with physics (Cognolato et al., 2023)
- Diffusion-model QSM (2023)
- Foundation model for susceptibility mapping (2025)

#### Step 3: Update SWI Solvers

After listing all SWI solvers, update `algorithm_base/swi/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SWI solvers use the data format: `y` (num_echoes, H, W, D) complex-valued multi-echo gradient-echo data. The `SWIOperator` handles the dipole forward model `phi = F^{-1} { D(k) * F{chi} }` where D(k) is the dipole kernel `(1/3 - kz^2/k^2)` relating susceptibility chi to field perturbation phi.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SWI:**
- QSM Challenge 2.0: TKD ~24.0 dB, MEDI ~30.0 dB, FANSI ~32.0 dB, QSMnet ~34.0 dB, xQSM ~35.5 dB
- QSM RMSE (ppb): TKD ~70, MEDI ~45, DL-based ~30
- COSMOS reference: MEDI NRMSE ~15%, DL-based ~8%
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'swi' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/swi/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/swi/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/swi/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SWI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/swi/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/swi/standard/`

---

### MR Elastography (`mr_elastography`) Modality Template

#### Step 1: Verify Standard Dataset

For MR Elastography, what dataset do you use to verify? Is this dataset used for MR Elastography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mr_elastography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MR Elastography standard dataset.

**Popular datasets to consider:**
- **QIBA MRE Phantom Dataset (Ehman et al., 2014)** — RSNA QIBA standardized viscoelastic gel phantoms with known shear moduli; canonical MRE calibration benchmark
- **Mayo Clinic Liver MRE Dataset (Yin et al., 2007)** — clinical liver stiffness measurements with biopsy-confirmed fibrosis staging
- **Charite Brain MRE Dataset (Sack et al., 2008)** — multifrequency brain MRE wave data with regional stiffness maps
- **BIOQIC MRE Simulation Data (Papazoglou et al., 2012)** — finite-element simulated wave fields in heterogeneous media with exact stiffness ground truth

**Decision criteria:** QIBA MRE phantom data provides traceable ground truth for inversion algorithm validation. BIOQIC simulations provide exact heterogeneous stiffness maps. Mayo Clinic liver data is the most clinically cited.

#### Step 2: List All MR Elastography Algorithms

Please first ensure all the MR Elastography algorithms have been listed in `\pwm\public\algorithm_base\mr_elastography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mr_elastography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MR Elastography solvers, please update the MR Elastography solver.

**Key algorithms to cover (1950–2026):**

_Classical / Direct Inversion (1990s–2005):_
- LFE — Local Frequency Estimation for shear wave speed (Knutsson et al., 1994; Manduca et al., 2001)
- Phase gradient — direct inversion via spatial wavelength estimation (Muthupillai et al., 1995)
- Algebraic Helmholtz Inversion — direct solution of Helmholtz equation (Oliphant et al., 2001)
- Curl operator preprocessing — removing compressional waves from displacement fields (Sinkus et al., 2005)
- Multi-frequency dual elasto-visco inversion (MDEV) (Papazoglou et al., 2012)

_Optimization / Model-Based (2005–2016):_
- FEM-based iterative inversion — finite element method for heterogeneous media (Van Houten et al., 2001)
- Nonlinear inversion (NLI) for MRE — iterative stiffness reconstruction (McGarry et al., 2012)
- Subzone-based inversion — local homogeneity assumption for stability (Manduca et al., 2003)
- Heterogeneous multifrequency direct inversion (HMDI) (Dittmann et al., 2017)
- Variational MRE inversion with TV regularization (Honarvar et al., 2013)
- Tomoelastography — multi-frequency inversion for high-resolution stiffness maps (Tzschätzsch et al., 2016)
- Poroelastic MRE inversion (Perrinez et al., 2010)
- k-MDEV — k-space-based MDEV for improved resolution (Dittmann et al., 2017)

_Deep Learning (2017–2026):_
- CNN-MRE — convolutional neural network for stiffness estimation (Murphy et al., 2019)
- U-Net MRE inversion — end-to-end wave-to-stiffness network (Scott et al., 2020)
- Physics-informed neural network for Helmholtz inversion (Chen et al., 2022)
- NLI-Net — learned nonlinear inversion (McGarry et al., 2023)
- Diffusion model for MRE noise reduction (2023)
- Self-supervised MRE inversion (2024)
- Foundation model for elastography (2025)

#### Step 3: Update MR Elastography Solvers

After listing all MR Elastography solvers, update `algorithm_base/mr_elastography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MR Elastography solvers use the data format: `y` (num_frequencies, 3, H, W, D) complex displacement fields (3 motion-encoding directions) at multiple drive frequencies. The `MREOperator` handles the viscoelastic Helmholtz equation `rho * omega^2 * u + nabla . (G* nabla u) = 0` relating complex shear modulus G* to displacement u.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MR Elastography:**
- QIBA phantom (stiffness RMSE in kPa): LFE ~0.8, Algebraic ~0.5, NLI ~0.3, CNN-MRE ~0.15
- BIOQIC simulation: LFE ~26.0 dB, FEM inversion ~32.0 dB, DL-based ~35.0 dB
- Liver fibrosis staging AUC: Clinical MRE ~0.94, DL-enhanced ~0.96
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mr_elastography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mr_elastography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mr_elastography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mr_elastography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MR Elastography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mr_elastography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_elastography/standard/`

---

### MR Fingerprinting (`mr_fingerprinting`) Modality Template

#### Step 1: Verify Standard Dataset

For MR Fingerprinting, what dataset do you use to verify? Is this dataset used for MR Fingerprinting popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mr_fingerprinting/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MR Fingerprinting standard dataset.

**Popular datasets to consider:**
- **ISMRM MRF Challenge Dataset (2015)** — standardized MRF acquisition data with reference T1/T2 maps from gold-standard sequences
- **NIST/ISMRM System Phantom MRF Data (Jiang et al., 2017)** — MRF scans of NIST phantom with traceable T1/T2 values; canonical quantitative validation benchmark
- **BrainWeb MRF Simulation (Ma et al., 2013)** — Bloch-simulated MRF signal evolutions with exact tissue parameter ground truth
- **EUROSPIN Phantom MRF Dataset (Buonincontri & Sawiak, 2016)** — multi-tube phantom with known relaxation times

**Decision criteria:** The NIST phantom MRF dataset provides traceable ground truth for quantitative accuracy. BrainWeb simulations provide exact parameter maps for reconstruction algorithm comparison.

#### Step 2: List All MR Fingerprinting Algorithms

Please first ensure all the MR Fingerprinting algorithms have been listed in `\pwm\public\algorithm_base\mr_fingerprinting\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mr_fingerprinting. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MR Fingerprinting solvers, please update the MR Fingerprinting solver.

**Key algorithms to cover (1950–2026):**

_Classical / Dictionary-Based (2013–2016):_
- Original MRF — dictionary matching via inner product (Ma et al., 2013)
- SVD compression of MRF dictionary — low-rank temporal subspace (McGivney et al., 2014)
- Group matching for MRF — accelerated dictionary search (Cauley et al., 2015)
- FISP-MRF — Fast Imaging with Steady-state Precession MRF (Jiang et al., 2015)

_Optimization / Model-Based (2014–2020):_
- ADMM-MRF — alternating direction for joint reconstruction and matching (Assländer et al., 2018)
- Low-rank alternating direction method for MRF (Zhao et al., 2018)
- Subspace reconstruction for MRF — temporal subspace modeling (Zhao et al., 2018)
- Multi-resolution MRF dictionary (Cline et al., 2017)
- Bayesian MRF — probabilistic tissue parameter estimation (McGivney et al., 2018)
- Model-based MRF reconstruction (Asslander et al., 2018)
- k-SVD dictionary learning for MRF (Mazor et al., 2018)
- Iterative MRF with spatial regularization (Doneva et al., 2017)
- BLIP-MRF — balanced and linearized phase-constrained MRF (Assländer et al., 2019)

_Deep Learning (2017–2026):_
- DRONE — deep learning for rapid MRF (Cohen et al., 2018)
- SCQ — Self-Consistent Quantification network for MRF (Fang et al., 2019)
- DeepMRF — neural network replacing dictionary matching (Virtue et al., 2017)
- MRF-ResNet — residual network for MRF reconstruction (Balsiger et al., 2019)
- Spatiotemporal MRF network — joint spatial and temporal processing (Hamilton et al., 2020)
- Physics-informed DL-MRF (Chen et al., 2022)
- Transformer-based MRF tissue mapping (2023)
- Foundation model for quantitative MRI (2025)

#### Step 3: Update MR Fingerprinting Solvers

After listing all MR Fingerprinting solvers, update `algorithm_base/mr_fingerprinting/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MR Fingerprinting solvers use the data format: `y` (num_timepoints, num_coils, H, W) highly undersampled k-space time series with varying flip angles and TRs. The `MRFOperator` handles the Bloch-equation forward model mapping tissue parameters (T1, T2, PD, B0, B1) to temporal signal evolutions via Bloch simulation, combined with spatial encoding `y_t = mask_t * F * S * x_t`.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MR Fingerprinting:**
- NIST phantom T1 RMSE (ms): Dictionary matching ~25, SVD-MRF ~20, DRONE ~12, DL-based ~8
- NIST phantom T2 RMSE (ms): Dictionary matching ~8, SVD-MRF ~6, DRONE ~3, DL-based ~2
- Brain T1 map NRMSE: Dictionary ~5%, Subspace ~3.5%, DRONE ~2.5%, Transformer ~1.8%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mr_fingerprinting' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mr_fingerprinting/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mr_fingerprinting/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mr_fingerprinting/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MR Fingerprinting. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mr_fingerprinting/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mr_fingerprinting/standard/`

---

### MR Angiography (`mra`) Modality Template

#### Step 1: Verify Standard Dataset

For MR Angiography, what dataset do you use to verify? Is this dataset used for MRA popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mra/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MRA standard dataset.

**Popular datasets to consider:**
- **IXI MRA Dataset (Imperial College London)** — 3D time-of-flight MRA of healthy brains; widely used for vessel segmentation and reconstruction
- **ADAM Challenge Dataset (2020)** — Amsterdam intracranial artery assessment; MRA with aneurysm annotation for detection and segmentation
- **TubeTK Vessel Dataset (Aylward & Bullitt, 2002)** — MRA with centerline ground truth for vascular tree extraction
- **OASIS-3 MRA (LaMontagne et al., 2019)** — longitudinal brain MRA dataset from the OASIS cohort
- **CASILab MRA Dataset (UNC)** — cerebral MRA with manual vessel segmentation ground truth

**Decision criteria:** IXI MRA is the most widely used MRA reconstruction dataset. ADAM Challenge for clinical applications with vessel segmentation ground truth.

#### Step 2: List All MRA Algorithms

Please first ensure all the MRA algorithms have been listed in `\pwm\public\algorithm_base\mra\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mra. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MRA solvers, please update the MRA solver.

**Key algorithms to cover (1950–2026):**

_Classical / Acquisition-Based (1980s–2005):_
- TOF-MRA — Time-of-Flight MRA with flow-related enhancement (Laub, 1990; Keller et al., 1989)
- PC-MRA — Phase-Contrast MRA for velocity encoding (Dumoulin & Hart, 1986)
- CE-MRA — Contrast-Enhanced MRA with gadolinium bolus (Prince et al., 1993)
- MIP — Maximum Intensity Projection rendering for angiographic display (Laub & Kaiser, 1988)
- MOTSA — Multiple Overlapping Thin Slab Acquisition for TOF (Parker et al., 1991)

_Optimization / Model-Based (2005–2016):_
- HYPR — HighlY constrained back PRojection for CE-MRA (Mistretta et al., 2006)
- k-t GRAPPA for dynamic CE-MRA (Tsao et al., 2005)
- Compressed sensing MRA with spatiotemporal sparsity (Lustig et al., 2008)
- GRASP — Golden-angle RAdial SParse parallel MRI for MRA (Feng et al., 2014)
- 4D Flow MRI reconstruction — divergence-free constraint (Santelli et al., 2016)
- Parallel imaging MRA with SENSE (Sodickson & Manning, 1997)
- Non-contrast-enhanced MRA — flow-sensitive dephasing (Miyazaki & Lee, 2008)
- Background suppression for non-CE-MRA (Edelman et al., 2010)

_Deep Learning (2017–2026):_
- DL-MRA denoising — CNN for low-dose CE-MRA enhancement (Gong et al., 2018)
- MRA vessel segmentation networks — U-Net, nnU-Net (Livne et al., 2019)
- Zero-shot MRA reconstruction — DIP-based (2020)
- 4D Flow super-resolution with deep learning (Ferdian et al., 2020)
- Transformer for MRA reconstruction (2022)
- Score-based MRA reconstruction (2023)
- Foundation model for vascular imaging (2025)

#### Step 3: Update MRA Solvers

After listing all MRA solvers, update `algorithm_base/mra/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MRA solvers use the data format: `y` (num_coils, H, W, D) 3D k-space data (TOF) or (num_coils, num_frames, H, W, D) for dynamic CE-MRA. The `MRAOperator` handles the flow-enhanced signal model incorporating inflow effects, velocity encoding gradients for PC-MRA, or contrast kinetic models for CE-MRA.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MRA:**
- IXI TOF-MRA (accelerated 4x): Zero-filled ~27.0 dB, CS-MRA ~33.0 dB, DL-based ~36.0 dB
- CE-MRA temporal resolution: HYPR ~34.0 dB, GRASP ~35.5 dB, DL-based ~37.0 dB
- Vessel segmentation Dice: U-Net ~0.85, nnU-Net ~0.88
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mra' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mra/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mra/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mra/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MRA. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mra/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mra/standard/`

---

### MR Spectroscopy (`mrs`) Modality Template

#### Step 1: Verify Standard Dataset

For MR Spectroscopy, what dataset do you use to verify? Is this dataset used for MRS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mrs/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MRS standard dataset.

**Popular datasets to consider:**
- **ISMRM MRS Fitting Challenge (Near et al., 2021)** — standardized simulated and in-vivo MRS datasets for fitting algorithm comparison; the community benchmark
- **Big GABA Dataset (Mikkelsen et al., 2017)** — multi-site GABA-edited MRS data for reproducibility evaluation
- **TARQUIN/LCModel Test Spectra** — reference spectra with known metabolite concentrations for fitting validation
- **Simulated MRS basis sets (Simpson et al., 2017)** — FID-A simulated spectra with exact ground truth concentrations

**Decision criteria:** The ISMRM MRS Fitting Challenge is the community standard for spectral fitting algorithm comparison. Simulated spectra provide exact ground truth for quantitative validation.

#### Step 2: List All MRS Algorithms

Please first ensure all the MRS algorithms have been listed in `\pwm\public\algorithm_base\mrs\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mrs. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MRS solvers, please update the MRS solver.

**Key algorithms to cover (1950–2026):**

_Classical / Time-Domain (1970s–2000):_
- FFT of FID — basic spectral analysis via Fourier transform
- HLSVD — Hankel Lanczos Singular Value Decomposition for water removal (Pijnappel et al., 1992)
- AMARES — Advanced Method for Accurate, Robust, and Efficient Spectral fitting (Vanhamme et al., 1997)
- AQSES — Automated Quantitation of Short Echo-time MRS (Ratiney et al., 2005)
- Eddy current correction — reference scan phase deconvolution (Klose, 1990)

_Model-Based / Frequency-Domain (2000–2016):_
- LCModel — Linear Combination of Model spectra (Provencher, 1993, 2001) — the gold standard MRS fitting tool
- TARQUIN — Totally Automatic Robust Quantitation In NMR (Wilson et al., 2011)
- jMRUI — Java-based MRS processing suite (Stefan et al., 2009)
- QUEST — QUantification based on quantum ESTimation (Ratiney et al., 2005)
- ProFit — 2D Prior-knowledge Fitting for MRSI (Schulte & Boesiger, 2006)
- Bayesian MRS fitting (Bretthorst, 1990; Albert et al., 2017)
- Spectral registration for frequency/phase correction (Near et al., 2015)
- MEGA-PRESS processing for J-difference editing (Mescher et al., 1998)
- Multi-voxel MRSI with Hamming filtering and zero-filling

_Deep Learning (2017–2026):_
- DeepMRS — CNN for metabolite quantification (Hatami et al., 2019)
- DL-Spectroscopy — deep learning spectral fitting (Lee & Kim, 2020)
- Learned spectral denoising (Chen et al., 2021)
- Physics-informed MRS quantification network (Rizzo et al., 2022)
- Self-supervised MRS fitting (2023)
- Transformer-based spectral analysis (2023)
- Foundation model for spectroscopy (2025)

#### Step 3: Update MRS Solvers

After listing all MRS solvers, update `algorithm_base/mrs/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MRS solvers use the data format: `y` (num_points,) complex FID time-domain signal or (num_voxels, num_points) for MRSI. The `MRSOperator` handles the spectral forward model `y(t) = sum_m a_m * phi_m(t) * exp(-t/T2_m) * exp(i*2*pi*f_m*t)` where a_m, T2_m, f_m are concentration, transverse relaxation, and frequency for each metabolite m, and phi_m is the basis spectrum.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MRS:**
- ISMRM Challenge (NAA concentration CRLB %): LCModel ~3%, TARQUIN ~5%, DeepMRS ~4%, DL-based ~2.5%
- Metabolite quantification MAPE: AMARES ~12%, LCModel ~6%, DL-based ~4%
- GABA quantification CV: MEGA-PRESS + LCModel ~10%, DL-enhanced ~6%
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mrs' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mrs/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mrs/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mrs/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MRS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mrs/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mrs/standard/`

---

### Digital Breast Tomosynthesis (`digital_breast_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For Digital Breast Tomosynthesis, what dataset do you use to verify? Is this dataset used for DBT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/digital_breast_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original DBT standard dataset.

**Popular datasets to consider:**
- **VICTRE (FDA Virtual Imaging Clinical Trial, Badano et al., 2018)** — simulated DBT projections with known lesion locations and breast anatomy; FDA-endorsed benchmark for reconstruction algorithm evaluation
- **DBTex Dataset (Buda et al., 2021)** — clinical DBT volumes with biopsy-proven lesion annotations from Duke University
- **OPTIMAM DBT Dataset (Halling-Brown et al., 2021)** — large-scale UK clinical DBT imaging database with cancer annotations
- **ACR DBT Phantom Dataset** — standardized accreditation phantom scans for quality control

**Decision criteria:** VICTRE provides exact ground truth for reconstruction algorithm comparison and is the FDA-endorsed simulation benchmark. DBTex is the most widely used clinical DBT dataset.

#### Step 2: List All Digital Breast Tomosynthesis Algorithms

Please first ensure all the Digital Breast Tomosynthesis algorithms have been listed in `\pwm\public\algorithm_base\digital_breast_tomo\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/digital_breast_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the DBT solvers, please update the DBT solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1970s–2005):_
- Shift-and-add — simple back-projection for limited-angle tomography (Grant, 1972)
- FBP for DBT — filtered back-projection adapted for limited angular range (Mertelmeier et al., 2006)
- Tuned-Aperture Computed Tomography (TACT) — weighted shift-and-add (Webber et al., 1997)
- SAA with iterative deblurring — shift-and-add with post-processing (Wu et al., 2004)

_Optimization / Model-Based (2005–2016):_
- SART — Simultaneous Algebraic Reconstruction Technique for DBT (Andersen & Kak, 1984; Zhang et al., 2006)
- MLEM for DBT — maximum likelihood for limited-angle problem (Sidky et al., 2009)
- Total Variation regularized DBT — TV minimization for limited-angle artifacts (Sidky et al., 2009)
- ADMM-TV for DBT reconstruction (Ramirez-Giraldo et al., 2011)
- Model-based iterative reconstruction (MBIR) for DBT — Hologic InSight (Garrett et al., 2015)
- Non-local means regularized DBT (Borges et al., 2017)
- Dictionary learning for DBT (Xu et al., 2012)
- Bilateral filter regularized iterative DBT (Zheng et al., 2018)
- Compressed sensing DBT with directional wavelets (Piccolomini & Loli, 2015)

_Deep Learning (2017–2026):_
- DBToR — Deep Breast Tomosynthesis Reconstruction (Teuwen et al., 2021)
- FBPConvNet for DBT — post-processing CNN after FBP (Jin et al., 2017, applied to DBT)
- Learned primal-dual for DBT (Adler & Öktem, 2018, applied to DBT)
- GAN-based artifact removal for DBT (2020)
- DL super-resolution for DBT slice enhancement (Li et al., 2021)
- Score-based diffusion for limited-angle DBT (2023)
- Physics-informed DBT reconstruction network (2023)
- Foundation model for breast imaging (2025)

#### Step 3: Update Digital Breast Tomosynthesis Solvers

After listing all DBT solvers, update `algorithm_base/digital_breast_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All DBT solvers use the data format: `y` (num_angles, H_det, W_det) limited-angle projection images (typically 9-25 projections over ±25° arc). The `DBTOperator` handles the cone-beam forward projection with limited angular range, modeling the X-ray source trajectory and flat-panel detector geometry.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Digital Breast Tomosynthesis:**
- VICTRE phantom: FBP ~24.0 dB, SART ~28.0 dB, TV-regularized ~31.0 dB, DBToR ~34.0 dB
- Lesion detection AUC: FBP ~0.82, MBIR ~0.88, DL-based ~0.92
- In-plane resolution (LP/mm): FBP ~4.0, iterative ~5.5, DL-based ~6.5
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'digital_breast_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/digital_breast_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for DBT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/digital_breast_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/digital_breast_tomo/standard/`

---

### DEXA (`dexa`) Modality Template

#### Step 1: Verify Standard Dataset

For DEXA, what dataset do you use to verify? Is this dataset used for DEXA popular algorithms? Please ensure the standard dataset in `datasets/benchmark/dexa/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original DEXA standard dataset.

**Popular datasets to consider:**
- **ESP European Spine Phantom Dataset (Kalender, 1992)** — standardized semi-anthropomorphic spine phantom with known BMD values; the canonical DEXA calibration benchmark
- **Hologic/Lunar Cross-Calibration Dataset (Shepherd et al., 2006)** — multi-vendor DEXA phantom data for standardization
- **NHANES DEXA Dataset (CDC)** — population-level whole-body DEXA scans with body composition data; the largest public DEXA repository
- **Zurich Longitudinal DEXA Phantom Dataset (Lamy et al., 2007)** — long-term precision monitoring data for DEXA systems

**Decision criteria:** ESP phantom dataset provides traceable BMD ground truth for algorithm validation. NHANES is the most widely used clinical DEXA dataset.

#### Step 2: List All DEXA Algorithms

Please first ensure all the DEXA algorithms have been listed in `\pwm\public\algorithm_base\dexa\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/dexa. Besides, you need to search all algorithms from 1950 to 2026. After listing all the DEXA solvers, please update the DEXA solver.

**Key algorithms to cover (1950–2026):**

_Classical / Dual-Energy (1960s–2000):_
- SPA/DPA — Single/Dual Photon Absorptiometry baseline (Cameron & Sorenson, 1963)
- Dual-energy decomposition — basis material decomposition for bone and soft tissue (Mazess et al., 1990)
- K-edge subtraction — iodine K-edge for BMD calibration (Ruegsegger et al., 1976)
- Fan-beam DEXA geometry — improved spatial resolution (Hologic QDR, 1990s)
- Pencil-beam DEXA — original rectilinear scanning geometry (Lunar DPX, 1987)

_Optimization / Segmentation (2000–2016):_
- Auto-segmentation of ROIs — automated vertebral and femoral neck detection (Faulkner et al., 1993)
- Sub-regional BMD analysis — Ward's triangle, trochanter partitioning (WHO, 1994)
- Body composition analysis — three-compartment model for fat, lean, bone (Pietrobelli et al., 1996)
- Cross-calibration algorithms — standardized BMD (sBMD) conversion (Genant et al., 1994)
- Trabecular Bone Score (TBS) — texture analysis of DEXA images (Pothuaud et al., 2009)
- FRAX integration — fracture risk assessment from DEXA BMD (Kanis et al., 2008)
- Vertebral fracture assessment (VFA) from lateral DEXA (Rea et al., 2000)
- Hip structural analysis (HSA) from DEXA (Beck et al., 1990)
- Finite element modeling from DEXA — 2D FEM for strength estimation (Langton et al., 2009)

_Deep Learning (2017–2026):_
- CNN for automated DEXA ROI detection (Burns et al., 2019)
- DXA-Net — deep learning body composition from DEXA (2020)
- Opportunistic CT-to-DEXA BMD prediction (Pan et al., 2020)
- U-Net vertebral segmentation for DEXA (Valentinitsch et al., 2019)
- DL-enhanced TBS estimation (2022)
- Federated learning for multi-site DEXA standardization (2023)
- Foundation model for bone densitometry (2025)

#### Step 3: Update DEXA Solvers

After listing all DEXA solvers, update `algorithm_base/dexa/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All DEXA solvers use the data format: `y` (2, H, W) dual-energy projection images at low and high kVp. The `DEXAOperator` handles the dual-energy decomposition model `y_lo = I0_lo * exp(-(mu_b_lo * t_b + mu_s_lo * t_s))` and `y_hi = I0_hi * exp(-(mu_b_hi * t_b + mu_s_hi * t_s))` solving for bone thickness t_b and soft tissue thickness t_s.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for DEXA:**
- ESP phantom BMD accuracy (g/cm²): Standard DEXA ~2% error, Cross-calibrated ~1%, DL-enhanced ~0.8%
- Body composition (fat mass RMSE in kg): Standard ~0.5, DL-based ~0.3
- Vertebral fracture detection sensitivity: VFA ~85%, DL-based ~92%
- All reference values from published papers and ISCD guidelines

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'dexa' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/dexa/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/dexa/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/dexa/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for DEXA. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/dexa/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/dexa/standard/`

---

### Mammography (`mammography`) Modality Template

#### Step 1: Verify Standard Dataset

For Mammography, what dataset do you use to verify? Is this dataset used for Mammography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mammography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Mammography standard dataset.

**Popular datasets to consider:**
- **DDSM (Digital Database for Screening Mammography, Heath et al., 2000)** — 2,620 cases with biopsy-proven ground truth; the canonical mammography benchmark dataset
- **INbreast (Moreira et al., 2012)** — full-field digital mammography with precise ROI annotations; widely used for detection and segmentation
- **VinDr-Mammo (Nguyen et al., 2023)** — 5,000 exams with BI-RADS assessment and bounding boxes from two institutions
- **CBIS-DDSM (Lee et al., 2017)** — curated version of DDSM with updated ROI segmentations and pathology-confirmed labels
- **RSNA Screening Mammography Challenge (2023)** — large-scale Kaggle competition dataset for cancer detection

**Decision criteria:** DDSM/CBIS-DDSM is the most widely used mammography benchmark. INbreast for high-quality annotations. RSNA Challenge for large-scale evaluation.

#### Step 2: List All Mammography Algorithms

Please first ensure all the Mammography algorithms have been listed in `\pwm\public\algorithm_base\mammography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mammography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Mammography solvers, please update the Mammography solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1970s–2005):_
- Unsharp masking — contrast enhancement for mammography (Rosenfeld & Kak, 1982)
- CLAHE — Contrast Limited Adaptive Histogram Equalization (Pizer et al., 1987)
- Multiscale wavelet enhancement — Laplacian pyramid contrast enhancement (Laine et al., 1994)
- Anti-scatter grid correction — scatter removal and flat-field correction
- Logarithmic subtraction for contrast-enhanced mammography (Lewin et al., 2003)

_Optimization / Model-Based (2005–2016):_
- Scatter correction using Monte Carlo modeling (Sechopoulos et al., 2007)
- Dual-energy contrast-enhanced mammography decomposition (Dromain et al., 2011)
- Iterative artifact reduction for digital mammography (Salvagnini et al., 2012)
- Microcalcification enhancement with morphological filtering (Gavrielides et al., 2002)
- Breast density estimation — Cumulus and automated BI-RADS (Boyd et al., 2007)
- CADe — Computer-Aided Detection for mammography (Birdwell et al., 2001)
- Multi-view feature correlation for lesion detection (Wei et al., 2011)
- Gabor filter–based texture analysis (Rangayyan et al., 2007)
- Penalized maximum likelihood for dose-reduced mammography (2012)

_Deep Learning (2017–2026):_
- End-to-end cancer detection — NYU whole-image classifier (Wu et al., 2020)
- GMIC — Globally-aware Multiple Instance Classifier for mammography (Shen et al., 2021)
- DMV-CNN — Dual Multi-View CNN for mammography (Carneiro et al., 2017)
- DL breast density estimation (Lehman et al., 2019)
- nnDetection for mammographic mass detection (2021)
- Transformer for mammography classification (2022)
- Score-based denoising for low-dose mammography (2023)
- Foundation model for breast imaging (2025)

#### Step 3: Update Mammography Solvers

After listing all Mammography solvers, update `algorithm_base/mammography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Mammography solvers use the data format: `y` (H, W) raw detector count image (DICOM "For Processing"). The `MammographyOperator` handles the X-ray transmission model `y = I0 * exp(-integral mu(x,E) dx)` with polyenergetic spectrum, scatter, and detector response, plus flat-field and gain corrections.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Mammography:**
- DDSM/CBIS-DDSM cancer detection AUC: CADe ~0.80, CNN ~0.87, GMIC ~0.91, NYU ~0.93
- INbreast mass segmentation Dice: Classical ~0.65, U-Net ~0.82, DL-based ~0.88
- Low-dose mammography (50% dose): CLAHE ~28.0 dB, DL-denoised ~34.0 dB
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mammography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mammography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mammography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mammography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Mammography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mammography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mammography/standard/`

---

### X-ray Angiography (`angiography`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray Angiography, what dataset do you use to verify? Is this dataset used for angiography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/angiography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original angiography standard dataset.

**Popular datasets to consider:**
- **XCAD (X-ray Coronary Artery Disease) Dataset (Ma et al., 2022)** — coronary angiography sequences with vessel segmentation ground truth; widely used for vessel enhancement and segmentation
- **DCA1 (Digital Coronary Angiography, 2018)** — 134 coronary angiograms with manual vessel annotations
- **ARCADE Challenge Dataset (2023)** — multi-center coronary angiography with anatomical labeling and stenosis annotations
- **3DRA Cerebral Aneurysm Dataset (Bogunović et al., 2011)** — 3D rotational angiography with aneurysm segmentation ground truth

**Decision criteria:** XCAD is the current standard benchmark for coronary angiography vessel enhancement and segmentation. 3DRA datasets for 3D reconstruction tasks.

#### Step 2: List All Angiography Algorithms

Please first ensure all the angiography algorithms have been listed in `\pwm\public\algorithm_base\angiography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/angiography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the angiography solvers, please update the angiography solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1950s–2000):_
- DSA — Digital Subtraction Angiography, logarithmic mask subtraction (Kruger et al., 1977; Mistretta et al., 1981)
- TID — Time Interval Differencing for motion compensation (Buzug et al., 1998)
- Vessel enhancement filtering — Hessian-based / Frangi vesselness filter (Frangi et al., 1998)
- Top-hat morphological filtering for vessel extraction (Eiho & Qian, 1997)
- Pixel-shifting for cardiac motion compensation in DSA (Meijering et al., 1999)

_Optimization / Model-Based (2000–2016):_
- Layer decomposition for DSA — ICA/PCA-based background separation (Aach et al., 2001)
- Robust PCA for angiographic background subtraction (Otazo et al., 2015)
- Elastic registration for motion-compensated DSA (Meijering et al., 1999)
- 3DRA — 3D Rotational Angiography reconstruction with FDK (Feldkamp et al., 1984; Fahrig et al., 1997)
- Model-based iterative 3DRA reconstruction (Defined et al., 2006)
- Temporal MIP and integration for angiographic display
- Non-rigid motion compensation using B-splines (Ganguly et al., 2013)
- Compressed sensing for undersampled 3DRA (Chen et al., 2012)
- Low-rank + sparse decomposition for angiography (Otazo et al., 2015)

_Deep Learning (2017–2026):_
- CNN vessel segmentation for coronary angiography (Nasr-Esfahani et al., 2018)
- DL-DSA — deep learning digital subtraction (Gao et al., 2019)
- RPCA-Net — learned robust PCA for angiographic background removal (2021)
- U-Net for vessel segmentation in angiography (Shin et al., 2019)
- GAN-based motion-compensated DSA (2021)
- Transformer for temporal angiographic analysis (2023)
- Self-supervised vessel enhancement (2023)
- Foundation model for vascular imaging (2025)

#### Step 3: Update Angiography Solvers

After listing all angiography solvers, update `algorithm_base/angiography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All angiography solvers use the data format: `y` (num_frames, H, W) temporal X-ray fluoroscopic sequence with contrast injection, plus `mask` (H, W) pre-contrast reference frame. The `AngiographyOperator` handles the DSA forward model `y_dsa = log(I_mask/I_contrast)` and for 3DRA the cone-beam projection `y = integral I0 * exp(-mu(x) dl)` over rotational trajectory.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Angiography:**
- XCAD vessel segmentation Dice: Frangi ~0.55, U-Net ~0.78, DL-based ~0.83
- DSA background removal PSNR: Standard DSA ~30.0 dB, RPCA ~34.0 dB, DL-DSA ~37.0 dB
- 3DRA reconstruction: FDK ~28.0 dB, Iterative ~33.0 dB, DL-based ~36.0 dB
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'angiography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/angiography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/angiography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/angiography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Angiography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/angiography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/angiography/standard/`

---

### Fluoroscopy (`fluoroscopy`) Modality Template

#### Step 1: Verify Standard Dataset

For Fluoroscopy, what dataset do you use to verify? Is this dataset used for Fluoroscopy popular algorithms? Please ensure the standard dataset in `datasets/benchmark/fluoroscopy/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Fluoroscopy standard dataset.

**Popular datasets to consider:**
- **DeepFluoro Dataset (Grupp et al., 2020)** — intraoperative fluoroscopy of hip arthroplasty with 3D CT registration ground truth; widely used for 2D/3D registration
- **RANZCR CLiP Dataset (2021)** — chest fluoroscopy/radiography with catheter and line position annotations
- **Fluoroscopy Dose Reduction Phantom Dataset (AAPM TG-201)** — standardized phantom acquisitions at varying dose levels for noise evaluation
- **Cardiac Fluoroscopy Dataset (Ambrosini et al., 2017)** — cardiac catheterization fluoroscopy sequences with device tracking annotations

**Decision criteria:** DeepFluoro is the standard for 2D/3D registration benchmarking. AAPM phantom data for dose reduction algorithm evaluation.

#### Step 2: List All Fluoroscopy Algorithms

Please first ensure all the Fluoroscopy algorithms have been listed in `\pwm\public\algorithm_base\fluoroscopy\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/fluoroscopy. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Fluoroscopy solvers, please update the Fluoroscopy solver.

**Key algorithms to cover (1950–2026):**

_Classical / Analog-to-Digital (1950s–2000):_
- Analog fluoroscopy — image intensifier with TV camera (Sturm & Morgan, 1949)
- Recursive temporal filtering — exponential averaging for noise reduction (IEC 62220, 1990s)
- Last-image-hold (LIH) — dose reduction by displaying last captured frame
- Pulsed fluoroscopy — reduced frame rate for dose reduction (Aufrichtig et al., 1994)
- Flat-panel detector fluoroscopy — transition from II to FPD (Granfors & Aufrichtig, 2000)

_Optimization / Model-Based (2000–2016):_
- Temporal recursive noise filtering with motion detection (Defined & Aufrichtig, 2003)
- Multi-resolution temporal noise reduction (Defined et al., 2008)
- 2D/3D registration — intensity-based registration of fluoroscopy to CT (Markelj et al., 2012)
- Digital variance angiography (DVA) — parametric imaging from fluoroscopy (Kiss et al., 2014)
- Bilateral temporal filtering for fluoroscopy (Metz et al., 2013)
- Scatter correction for cone-beam CT from fluoroscopy (Siewerdsen et al., 2001)
- Motion-compensated temporal filtering (Defined et al., 2010)
- Adaptive ROI fluoroscopy (Defined et al., 2012)
- Compressed sensing for dynamic fluoroscopy (Chen et al., 2008)

_Deep Learning (2017–2026):_
- CNN denoising for low-dose fluoroscopy (Leuliet et al., 2020)
- DL 2D/3D registration — learning-based pose estimation (Miao et al., 2018)
- Noise2Noise for fluoroscopy denoising (Lehtinen et al., 2018, applied to fluoroscopy)
- LSTM temporal denoising for fluoroscopy (2020)
- GAN-based dose reduction (2021)
- Real-time DL tracking for interventional fluoroscopy (2022)
- Foundation model for interventional imaging (2025)

#### Step 3: Update Fluoroscopy Solvers

After listing all Fluoroscopy solvers, update `algorithm_base/fluoroscopy/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Fluoroscopy solvers use the data format: `y` (num_frames, H, W) temporal fluoroscopic image sequence. The `FluoroscopyOperator` handles the real-time X-ray projection model with flat-panel detector response, temporal noise characteristics, and motion blur modeling.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Fluoroscopy:**
- AAPM phantom (dose reduction 75%): Recursive filter ~26.0 dB, Bilateral ~29.0 dB, CNN ~33.0 dB, DL-based ~35.0 dB
- DeepFluoro registration TRE (mm): Intensity-based ~3.0, DL-based ~1.5
- Real-time processing FPS: Recursive >30, DL-based >15
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'fluoroscopy' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/fluoroscopy/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Fluoroscopy. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/fluoroscopy/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/fluoroscopy/standard/`

---

### Fiber Bundle Endoscopy (`endoscopy`) Modality Template

#### Step 1: Verify Standard Dataset

For Fiber Bundle Endoscopy, what dataset do you use to verify? Is this dataset used for endoscopy popular algorithms? Please ensure the standard dataset in `datasets/benchmark/endoscopy/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original endoscopy standard dataset.

**Popular datasets to consider:**
- **EndoSLAM Dataset (Ozyoruk et al., 2021)** — endoscopic video with ground-truth depth, pose, and 3D reconstruction; widely used for endoscopic reconstruction
- **Hamlyn Centre Endoscopy Dataset (Mountney et al., 2010)** — stereo endoscopy with 3D surface ground truth; canonical reconstruction benchmark
- **SCARED Dataset (Allan et al., 2021)** — Stereo Correspondence and Reconstruction of Endoscopic Data; MICCAI challenge benchmark
- **Kvasir-SEG (Jha et al., 2020)** — gastrointestinal polyp segmentation dataset with ground-truth masks
- **CholecSeg8k (Hong et al., 2020)** — cholecystectomy video frames with semantic segmentation

**Decision criteria:** Hamlyn Centre / SCARED datasets are the standard for endoscopic 3D reconstruction. Kvasir-SEG for segmentation. EndoSLAM for simultaneous localization and mapping.

#### Step 2: List All Endoscopy Algorithms

Please first ensure all the Endoscopy algorithms have been listed in `\pwm\public\algorithm_base\endoscopy\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/endoscopy. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Endoscopy solvers, please update the Endoscopy solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1960s–2005):_
- Fiber bundle pattern removal — honeycomb artifact suppression (Lee & Bhargava, 2001)
- Mosaicking — image stitching for expanded field of view (Behrens et al., 2001)
- Structure from Motion (SfM) for endoscopic 3D (Mountney & Yang, 2010)
- Stereo reconstruction — disparity-based depth from stereo endoscopes (Stoyanov et al., 2005)
- Color correction for fiber bundle endoscopy (Elter et al., 2006)

_Optimization / Model-Based (2005–2016):_
- SLAM for endoscopy — Simultaneous Localization and Mapping (Mountney et al., 2006)
- Dense 3D reconstruction from monocular endoscopy (Stoyanov et al., 2010)
- Shape-from-Shading for endoscopic surfaces (Okatani & Deguchi, 1997; Collins & Bartoli, 2012)
- Deformable surface tracking for endoscopy (Collins & Bartoli, 2012)
- NBI / chromoendoscopy image enhancement (Gono et al., 2004)
- Specular reflection removal (Stehle et al., 2006)
- Light-field endoscopy reconstruction (Orth et al., 2015)
- Super-resolution for fiber bundle imaging (Han et al., 2013)
- Photometric stereo for endoscopic surfaces (2013)

_Deep Learning (2017–2026):_
- Endo-SfMLearner — self-supervised depth and pose for endoscopy (Ozyoruk et al., 2021)
- MonoDepth for endoscopy — monocular depth estimation (Liu et al., 2019)
- EndoNeRF — neural radiance fields for deformable tissue reconstruction (Wang et al., 2022)
- DL fiber bundle artifact removal (Shao et al., 2019)
- U-Net polyp segmentation (Jha et al., 2019)
- TransUNet for endoscopic segmentation (Chen et al., 2021)
- Foundation model for endoscopic vision (2025)

#### Step 3: Update Endoscopy Solvers

After listing all Endoscopy solvers, update `algorithm_base/endoscopy/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Endoscopy solvers use the data format: `y` (num_frames, H, W, 3) RGB video frames from fiber bundle endoscope. The `EndoscopyOperator` handles the fiber bundle forward model including honeycomb sampling pattern, inter-fiber crosstalk, chromatic aberration, and the projection model for 3D surface reconstruction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Endoscopy:**
- SCARED depth estimation (RMSE mm): SfM ~5.0, MonoDepth ~3.0, Endo-SfMLearner ~2.5, EndoNeRF ~1.8
- Fiber bundle super-resolution: Bilinear ~26.0 dB, DL-based ~32.0 dB
- Kvasir-SEG polyp Dice: U-Net ~0.82, TransUNet ~0.88
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'endoscopy' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/endoscopy/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/endoscopy/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/endoscopy/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Endoscopy. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/endoscopy/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/endoscopy/standard/`

---

### Fundus Camera (`fundus`) Modality Template

#### Step 1: Verify Standard Dataset

For Fundus Camera, what dataset do you use to verify? Is this dataset used for fundus popular algorithms? Please ensure the standard dataset in `datasets/benchmark/fundus/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original fundus standard dataset.

**Popular datasets to consider:**
- **DRIVE (Digital Retinal Images for Vessel Extraction, Staal et al., 2004)** — 40 fundus images with manual vessel segmentation; the canonical vessel segmentation benchmark
- **STARE (Structured Analysis of the Retina, Hoover et al., 2000)** — 20 fundus images with vessel annotations and pathology labels
- **EyePACS / Kaggle Diabetic Retinopathy Dataset (2015)** — 88,000+ fundus images with DR severity grading; the largest fundus classification dataset
- **CHASE_DB1 (Fraz et al., 2012)** — retinal vessel segmentation dataset from child health screening
- **IDRiD (Indian Diabetic Retinopathy Image Dataset, Porwal et al., 2018)** — fundus images with lesion segmentation, DR grading, and optic disc/fovea localization

**Decision criteria:** DRIVE is the canonical vessel segmentation benchmark. EyePACS for DR classification. IDRiD for comprehensive multi-task evaluation.

#### Step 2: List All Fundus Algorithms

Please first ensure all the Fundus algorithms have been listed in `\pwm\public\algorithm_base\fundus\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/fundus. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Fundus solvers, please update the Fundus solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1970s–2005):_
- Green channel extraction — vessel enhancement in green channel (Chaudhuri et al., 1989)
- Matched filter for vessel detection (Chaudhuri et al., 1989)
- Morphological vessel extraction — top-hat and opening operations (Zana & Klein, 2001)
- Illumination normalization — shade correction for fundus images (Foracchia et al., 2005)
- Optic disc detection — Hough transform and template matching (Sinthanayothin et al., 1999)

_Optimization / Model-Based (2005–2016):_
- Frangi vesselness filter for retinal vessels (Frangi et al., 1998; applied to fundus)
- Supervised pixel classification for vessels — feature-based classifiers (Staal et al., 2004)
- Active contour / level set for vessel segmentation (Sum & Cheung, 2008)
- Multi-scale retinal vessel segmentation (Martinez-Perez et al., 2007)
- Graph-cut based optic disc segmentation (Xu et al., 2007)
- Lesion detection using random forests (Antal & Hajdu, 2014)
- Sparse representation for fundus image quality assessment (Remeseiro et al., 2017)
- Image registration for longitudinal fundus analysis (Stewart et al., 2003)
- Wavelet-based microaneurysm detection (Quellec et al., 2008)

_Deep Learning (2017–2026):_
- InceptionV3 for DR grading — Kaggle competition winner (2015, refined 2017)
- U-Net for retinal vessel segmentation (Ronneberger et al., 2015; applied to fundus)
- DeepDR — deep learning diabetic retinopathy detection (Gargeya & Leng, 2017)
- Attention U-Net for retinal vessel segmentation (2019)
- GAN for fundus image synthesis (Costa et al., 2018)
- Self-supervised fundus representation learning (2022)
- DL optic disc/cup segmentation for glaucoma (Fu et al., 2018)
- Foundation model for ophthalmology (RETFound, Zhou et al., 2023)

#### Step 3: Update Fundus Solvers

After listing all Fundus solvers, update `algorithm_base/fundus/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Fundus solvers use the data format: `y` (H, W, 3) RGB fundus photograph. The `FundusOperator` handles the retinal imaging forward model including illumination optics, chromatic aberration, media opacity effects, and the mapping from retinal reflectance to detector signal.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Fundus:**
- DRIVE vessel segmentation AUC: Matched filter ~0.93, U-Net ~0.97, Attention U-Net ~0.98
- DRIVE vessel Dice: Matched filter ~0.70, U-Net ~0.80, DL-based ~0.83
- EyePACS DR detection AUC: InceptionV3 ~0.95, DL-based ~0.97
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'fundus' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/fundus/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/fundus/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/fundus/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Fundus. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/fundus/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/fundus/standard/`

---

### Intravascular Ultrasound (`ivus`) Modality Template

#### Step 1: Verify Standard Dataset

For IVUS, what dataset do you use to verify? Is this dataset used for IVUS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ivus/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original IVUS standard dataset.

**Popular datasets to consider:**
- **IVUS Segmentation Challenge Dataset (Balocco et al., 2014)** — IVUS pullback sequences with expert-annotated lumen and media borders; the canonical IVUS benchmark
- **IntrA Dataset (Yang et al., 2020)** — intracranial artery 3D IVUS/MRA with vessel segmentation ground truth
- **IVUS 20 MHz Dataset (Katouzian et al., 2012)** — 20 MHz IVUS frames with tissue characterization ground truth
- **iVUS Dataset (El-Zehiry et al., 2009)** — IVUS with plaque composition annotations

**Decision criteria:** The IVUS Segmentation Challenge dataset is the gold standard for IVUS lumen/media border detection benchmarking. Use datasets with expert-annotated contours.

#### Step 2: List All IVUS Algorithms

Please first ensure all the IVUS algorithms have been listed in `\pwm\public\algorithm_base\ivus\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ivus. Besides, you need to search all algorithms from 1950 to 2026. After listing all the IVUS solvers, please update the IVUS solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1990s–2005):_
- Polar-to-Cartesian conversion — standard IVUS display transformation
- Log compression and envelope detection — RF to B-mode processing (Mintz et al., 2001)
- Edge detection for lumen border — gradient and Canny-based (Sonka et al., 1995)
- Texture analysis for plaque characterization — echogenicity classification (Prati et al., 2001)
- Gating and motion compensation — ECG-gated IVUS (Bruining et al., 1998)

_Optimization / Model-Based (2005–2016):_
- Active contour / snake for IVUS border detection (Kovalski et al., 2000)
- Graph search for lumen/media segmentation (Sonka et al., 1995; Cardinal et al., 2006)
- Random forest for IVUS tissue classification (Defined et al., 2014)
- 3D IVUS reconstruction from pullback sequences (Wahle et al., 1999)
- Virtual Histology IVUS — spectral analysis for plaque characterization (Nair et al., 2002)
- iMAP — intravascular tissue characterization (Sathyanarayana et al., 2009)
- Deconvolution for IVUS axial resolution enhancement (Katouzian et al., 2008)
- Hidden Markov model for temporal IVUS segmentation (Defined et al., 2012)
- IVUS-OCT co-registration (Li et al., 2014)

_Deep Learning (2017–2026):_
- CNN for IVUS lumen segmentation (Yang et al., 2018)
- IVUS-Net — U-Net for border detection (Yang et al., 2019)
- GAN-based IVUS despeckling (Bargsten et al., 2020)
- DL plaque burden estimation (Sofian et al., 2020)
- Attention-based IVUS segmentation (2021)
- Temporal LSTM for IVUS pullback analysis (2022)
- Foundation model for intravascular imaging (2025)

#### Step 3: Update IVUS Solvers

After listing all IVUS solvers, update `algorithm_base/ivus/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All IVUS solvers use the data format: `y` (num_frames, num_lines, num_samples) RF data or (num_frames, H, W) B-mode polar images from rotational transducer pullback. The `IVUSOperator` handles the ultrasound forward model with rotational scanning geometry, including beam profile, attenuation, and backscatter for intravascular tissue.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for IVUS:**
- Challenge 2014 lumen Dice: Active contour ~0.82, Graph search ~0.88, IVUS-Net ~0.93
- Challenge 2014 media Dice: Active contour ~0.75, Graph search ~0.83, DL-based ~0.90
- Plaque burden MAE (%): VH-IVUS ~8%, DL-based ~4%
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ivus' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ivus/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ivus/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ivus/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for IVUS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ivus/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ivus/standard/`

---

### Portal Imaging (`portal_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For Portal Imaging (EPID), what dataset do you use to verify? Is this dataset used for portal imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/portal_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original portal imaging standard dataset.

**Popular datasets to consider:**
- **AAPM TG-58 EPID Dataset** — standardized portal imaging quality assurance data with known dosimetric ground truth
- **EPID Dosimetry Commissioning Dataset (van Elmpt et al., 2008)** — EPID transit dosimetry data with ion chamber reference measurements
- **Varian aSi-EPID Calibration Dataset** — factory calibration data for amorphous silicon flat-panel EPID systems
- **Pre-treatment IMRT QA EPID Dataset (Miri et al., 2016)** — portal images for IMRT plan verification with planned dose comparison

**Decision criteria:** The AAPM TG-58 EPID dataset provides standardized portal imaging data. EPID dosimetry commissioning data provides dosimetric ground truth.

#### Step 2: List All Portal Imaging Algorithms

Please first ensure all the Portal Imaging algorithms have been listed in `\pwm\public\algorithm_base\portal_imaging\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/portal_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Portal Imaging solvers, please update the Portal Imaging solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1980s–2005):_
- Portal film digitization — optical densitometry of film-based portal images (Boyer et al., 1992)
- Flood-field and dark-field correction for EPID — pixel gain and offset calibration (Antonuk et al., 1998)
- Portal image enhancement — contrast stretching and histogram equalization (Leszczynski et al., 1993)
- Template matching for patient setup verification (Gilhuijs et al., 1993)
- DRR — Digitally Reconstructed Radiograph generation from CT for comparison (Sherouse et al., 1990)

_Optimization / Model-Based (2005–2016):_
- EPID dosimetry — back-projection for in-vivo dose reconstruction (Nijsten et al., 2007)
- Kernel-based scatter correction for EPID (Swindell & Evans, 1996; Rowshanfarzad et al., 2010)
- Monte Carlo EPID dose prediction (Siebers et al., 2004)
- Portal dose image prediction (PDIP) — forward calculation of expected EPID signal (van Elmpt et al., 2008)
- 2D gamma analysis for EPID QA (Low et al., 1998)
- MLC leaf position detection from EPID (Baker et al., 2005)
- CBCT reconstruction from portal images — EPID-based CBCT (Mao et al., 2008)
- Anatomical landmark matching for EPID registration (Bijhold et al., 1991)
- 3D dose reconstruction from transit EPID (van Elmpt et al., 2006)

_Deep Learning (2017–2026):_
- CNN for EPID-based dose prediction (Yousefian et al., 2021)
- DL portal image enhancement — denoising and super-resolution (2020)
- U-Net for MLC error detection from EPID (Carlson et al., 2019)
- AutoQA — automated EPID quality assurance with deep learning (2021)
- DL transit dosimetry — learned dose back-projection (2022)
- Real-time anatomy detection from portal images (2023)
- Foundation model for radiation therapy imaging (2025)

#### Step 3: Update Portal Imaging Solvers

After listing all Portal Imaging solvers, update `algorithm_base/portal_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Portal Imaging solvers use the data format: `y` (H, W) megavoltage X-ray portal image from EPID or (num_frames, H, W) cine EPID acquisition. The `PortalImagingOperator` handles the MV X-ray transmission model with EPID detector response `y = conv(K_scatter, I0 * exp(-integral mu_MV dx)) + I_scatter` including MV beam spectrum, patient scatter, and detector optical spread.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Portal Imaging:**
- EPID dosimetry gamma pass rate (3%/3mm): Classical ~92%, MC-based ~96%, DL-based ~98%
- Portal image enhancement PSNR: Raw ~22.0 dB, Enhanced ~28.0 dB, DL-based ~32.0 dB
- Setup error detection accuracy: Template ~85%, DL-based ~94%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'portal_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/portal_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/portal_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/portal_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Portal Imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/portal_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/portal_imaging/standard/`

---

### Brachytherapy Imaging (`brachytherapy_img`) Modality Template

#### Step 1: Verify Standard Dataset

For Brachytherapy Imaging, what dataset do you use to verify? Is this dataset used for brachytherapy imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/brachytherapy_img/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original brachytherapy imaging standard dataset.

**Popular datasets to consider:**
- **AAPM TG-43 Validation Dataset** — Monte Carlo–calculated dose distributions around brachytherapy sources with known geometry; canonical dosimetric benchmark
- **GEC-ESTRO Cervix Brachytherapy Dataset (Pötter et al., 2006)** — MRI-guided brachytherapy planning images with contoured target volumes
- **ABS/AAPM Prostate Brachytherapy Dataset** — CT and US-guided implant images with seed localization ground truth
- **Brachytherapy Source Localization Challenge (Zaffino et al., 2020)** — CT images with manually identified brachytherapy source positions

**Decision criteria:** AAPM TG-43 provides exact dosimetric ground truth. GEC-ESTRO cervix data is the most widely used clinical brachytherapy imaging dataset.

#### Step 2: List All Brachytherapy Imaging Algorithms

Please first ensure all the Brachytherapy Imaging algorithms have been listed in `\pwm\public\algorithm_base\brachytherapy_img\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/brachytherapy_img. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Brachytherapy Imaging solvers, please update the Brachytherapy Imaging solver.

**Key algorithms to cover (1950–2026):**

_Classical / Localization (1960s–2005):_
- Orthogonal film reconstruction — source localization from AP/lateral radiographs (Anderson et al., 1976)
- Isodose curve calculation — TG-43 point-source dose formalism (Nath et al., 1995)
- Template-based seed matching — correspondence between planned and implanted seeds (Tubic et al., 2001)
- CT-based source localization — thresholding and centroid detection from CT (Peschel et al., 1999)
- Fluoroscopy-based catheter tracking — real-time C-arm guidance (Siewerdsen et al., 2000)

_Optimization / Model-Based (2005–2016):_
- Monte Carlo dose calculation for brachytherapy — TG-186 full transport (Beaulieu et al., 2012)
- TRUS-based prostate seed localization — 3D ultrasound seed segmentation (Orio et al., 2007)
- Iterative seed localization from CT — artifact reduction and clustering (Lam et al., 2004)
- MRI-based brachytherapy planning — MR-only needle/source reconstruction (Haack et al., 2009)
- Deformable registration for brachytherapy dose accumulation (Kim et al., 2013)
- Electromagnetic tracking for catheter reconstruction (Zhou et al., 2013)
- Dose volume histogram optimization for brachytherapy (Lessard & Pouliot, 2001)
- Atlas-based auto-segmentation for brachytherapy planning (Dubois et al., 2015)
- GPU-accelerated Monte Carlo for real-time dose (Hissoiny et al., 2011)

_Deep Learning (2017–2026):_
- CNN for automatic seed detection in CT (Mehta et al., 2020)
- DL catheter segmentation in MRI (Zaffino et al., 2020)
- U-Net for brachytherapy target segmentation (Mohammadi et al., 2021)
- DL dose prediction for brachytherapy (Nguyen et al., 2019)
- GAN-based artifact reduction around seeds (2022)
- Transformer for brachytherapy treatment planning (2023)
- Foundation model for radiation therapy (2025)

#### Step 3: Update Brachytherapy Imaging Solvers

After listing all Brachytherapy Imaging solvers, update `algorithm_base/brachytherapy_img/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Brachytherapy Imaging solvers use the data format: `y` (H, W, D) CT volume with implanted sources, or (H, W) radiographic images, or (H, W, D) TRUS volume. The `BrachytherapyOperator` handles source localization (seed/catheter geometry extraction) and the TG-43/TG-186 dose forward model `D(r,theta) = S_k * Lambda * G(r,theta)/G(r0,theta0) * g(r) * F(r,theta)` with geometry function G, radial dose function g, and anisotropy function F.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Brachytherapy Imaging:**
- Seed localization error (mm): CT threshold ~1.5, Iterative ~0.8, DL-based ~0.5
- Dose calculation accuracy (% within 2%): TG-43 ~95% (homogeneous), MC ~99%, DL-predicted ~97%
- Target segmentation Dice: Atlas ~0.75, U-Net ~0.85, DL-based ~0.89
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'brachytherapy_img' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/brachytherapy_img/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Brachytherapy Imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/brachytherapy_img/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/brachytherapy_img/standard/`

---

### Proton Therapy Imaging (`proton_therapy_img`) Modality Template

#### Step 1: Verify Standard Dataset

For Proton Therapy Imaging, what dataset do you use to verify? Is this dataset used for proton therapy imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/proton_therapy_img/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original proton therapy imaging standard dataset.

**Popular datasets to consider:**
- **OpenPCT (Open Proton CT Dataset, Johnson et al., 2017)** — proton CT projection data with known phantom geometry; the standard pCT reconstruction benchmark
- **Geant4 Proton CT Simulation (Penfold et al., 2009)** — Monte Carlo simulated proton tracking data with exact RSP ground truth
- **ProtonVDA Dataset (Collins-Fekete et al., 2017)** — proton radiography and CT data with Most Likely Path (MLP) tracking
- **CIRS Electron Density Phantom pCT Data** — proton CT of tissue-equivalent phantom with known RSP values

**Decision criteria:** OpenPCT and Geant4 simulations provide exact RSP ground truth for proton CT reconstruction. CIRS phantom data for clinical-calibration validation.

#### Step 2: List All Proton Therapy Imaging Algorithms

Please first ensure all the Proton Therapy Imaging algorithms have been listed in `\pwm\public\algorithm_base\proton_therapy_img\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/proton_therapy_img. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Proton Therapy Imaging solvers, please update the Proton Therapy Imaging solver.

**Key algorithms to cover (1950–2026):**

_Classical / Analytic (1960s–2005):_
- Proton radiography — integral RSP measurement from energy loss (Cormack, 1963; Koehler, 1968)
- CT number to RSP conversion — stoichiometric calibration (Schneider et al., 1996)
- FBP for proton CT — filtered back-projection with curved path correction (Hanson, 1979)
- Range probe measurement — single-projection range verification (Mumot et al., 2010)
- Prompt gamma imaging — real-time range verification from nuclear reactions (Min et al., 2006)

_Optimization / Model-Based (2005–2016):_
- Most Likely Path (MLP) — Bayesian proton path estimation in pCT (Williams, 2004; Schulte et al., 2008)
- Algebraic reconstruction (ART/DROP) for proton CT (Penfold et al., 2010)
- Total Variation regularized proton CT (Rit et al., 2013)
- Dual-energy CT for RSP estimation (Yang et al., 2010)
- PG Compton imaging — gamma camera for range verification (Hueso-González et al., 2015)
- PET range verification — positron emission from nuclear fragmentation (Parodi et al., 2007)
- Ionoacoustic range verification — acoustic signal from Bragg peak (Assmann et al., 2015)
- List-mode proton CT reconstruction (Schulte et al., 2005)
- WEPL calibration — water-equivalent path length from multi-stage detector (Hurley et al., 2012)

_Deep Learning (2017–2026):_
- CNN for CT-to-RSP conversion — replacing stoichiometric calibration (Han et al., 2019)
- DL proton CT reconstruction (DeJongh et al., 2021)
- U-Net for prompt gamma range prediction (Gueth et al., 2022)
- Physics-informed neural network for proton tracking (2022)
- Diffusion model for proton CT denoising (2023)
- DL dose verification from EPID (2023)
- Foundation model for proton therapy (2025)

#### Step 3: Update Proton Therapy Imaging Solvers

After listing all Proton Therapy Imaging solvers, update `algorithm_base/proton_therapy_img/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Proton Therapy Imaging solvers use the data format: `y` (num_protons, 6) list-mode proton tracking data [x_in, y_in, angle_in, x_out, y_out, angle_out, WEPL] or (num_angles, H, W) integrated WEPL sinograms. The `ProtonCTOperator` handles the proton transport forward model incorporating Multiple Coulomb Scattering (MCS), energy loss via Bethe-Bloch equation, and Most Likely Path estimation through the object.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Proton Therapy Imaging:**
- Catphan phantom RSP accuracy (%): CT calibration ~1.5%, FBP-pCT ~1.0%, TV-pCT ~0.6%, DL-pCT ~0.4%
- Head phantom RSP RMSE: FBP ~0.03, Iterative ~0.015, DL-based ~0.008
- Range verification accuracy (mm): PET ~3.0, PG imaging ~2.0, DL-based ~1.5
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'proton_therapy_img' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/proton_therapy_img/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/proton_therapy_img/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/proton_therapy_img/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Proton Therapy Imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/proton_therapy_img/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/proton_therapy_img/standard/`

---

### Doppler Ultrasound (`doppler_ultrasound`) Modality Template

#### Step 1: Verify Standard Dataset

For Doppler Ultrasound, what dataset do you use to verify? Is this dataset used for Doppler Ultrasound popular algorithms? Please ensure the standard dataset in `datasets/benchmark/doppler_ultrasound/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Doppler Ultrasound standard dataset.

**Popular datasets to consider:**
- **PICMUS (Plane-wave Imaging Challenge in Medical UltraSound, Liebgott et al., 2016)** — standardized flow phantom and in-vivo carotid Doppler data; canonical ultrafast Doppler benchmark
- **Field II Doppler Simulation (Jensen, 1996)** — simulated pulsed-wave and color Doppler data with exact velocity ground truth
- **ABI Flow Phantom Dataset (Hoskins et al., 2010)** — commercial flow phantom with calibrated steady and pulsatile flow profiles
- **Carotid Duplex Dataset (AIUM, 2012)** — clinical carotid Doppler examinations with stenosis grading

**Decision criteria:** PICMUS provides standardized Doppler benchmark data. Field II simulations provide exact velocity ground truth. Use datasets with calibrated flow phantoms.

#### Step 2: List All Doppler Ultrasound Algorithms

Please first ensure all the Doppler Ultrasound algorithms have been listed in `\pwm\public\algorithm_base\doppler_ultrasound\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/doppler_ultrasound. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Doppler Ultrasound solvers, please update the Doppler Ultrasound solver.

**Key algorithms to cover (1950–2026):**

_Classical / Signal Processing (1960s–2000):_
- Continuous Wave (CW) Doppler — frequency shift detection for velocity measurement (Satomura, 1957)
- Pulsed Wave (PW) Doppler — range-gated velocity estimation (Baker, 1970)
- Autocorrelation velocity estimator — Kasai estimator for color flow (Kasai et al., 1985)
- Spectral Doppler — FFT-based spectral analysis of Doppler signal (Atkinson & Woodcock, 1982)
- Wall filter — high-pass filtering to remove clutter (Bjaerum et al., 2002)

_Optimization / Model-Based (2000–2016):_
- Ultrafast Doppler — plane-wave compound Doppler imaging (Bercoff et al., 2011; Tanter & Fink, 2014)
- SVD clutter filter — singular value decomposition for tissue clutter removal (Demené et al., 2015)
- Adaptive wall filtering — eigen-based clutter rejection (Yu & Lovstakken, 2010)
- Vector Doppler — multi-angle velocity estimation (Dunmire et al., 2000; Jensen & Munk, 1998)
- Transverse oscillation method — lateral velocity estimation (Jensen & Munk, 1998)
- Speckle tracking — 2D velocity estimation from B-mode displacement (Bohs & Trahey, 1991)
- Power Doppler — amplitude-based flow detection (Rubin et al., 1994)
- Functional ultrasound (fUS) — ultrafast Doppler for brain hemodynamics (Macé et al., 2011)
- Spatiotemporal clutter filtering for micro-vessel imaging (Song et al., 2017)

_Deep Learning (2017–2026):_
- DL clutter rejection for Doppler — learned SVD replacement (Solomon et al., 2019)
- CNN velocity estimation from IQ data (Tehrani et al., 2020)
- U-Net for power Doppler denoising (2020)
- DeepDoppler — end-to-end flow estimation (Youn et al., 2020)
- Physics-informed flow estimation network (2022)
- Transformer for temporal Doppler analysis (2023)
- Foundation model for vascular ultrasound (2025)

#### Step 3: Update Doppler Ultrasound Solvers

After listing all Doppler Ultrasound solvers, update `algorithm_base/doppler_ultrasound/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Doppler Ultrasound solvers use the data format: `y` (num_frames, num_elements, num_samples) raw IQ data from plane-wave or focused acquisitions with an ensemble of transmissions. The `DopplerOperator` handles the Doppler forward model `y_slow(t) = A * v(x) * exp(i*2*pi*f_d*t) + clutter` where f_d = 2*v*cos(theta)*f0/c is the Doppler frequency shift, with beamforming and clutter modeling.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Doppler Ultrasound:**
- PICMUS flow phantom velocity RMSE (cm/s): Kasai ~3.0, SVD filter ~1.5, DL-based ~0.8
- Micro-vessel detection sensitivity: Power Doppler ~60%, SVD-filtered ~85%, DL-enhanced ~92%
- Spectral Doppler peak velocity error (%): Standard ~8%, DL-based ~3%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'doppler_ultrasound' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/doppler_ultrasound/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Doppler Ultrasound. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/doppler_ultrasound/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/doppler_ultrasound/standard/`

---

### Contrast-Enhanced Ultrasound (`ceus`) Modality Template

#### Step 1: Verify Standard Dataset

For CEUS, what dataset do you use to verify? Is this dataset used for CEUS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ceus/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CEUS standard dataset.

**Popular datasets to consider:**
- **CEUS Liver Lesion Dataset (Claudon et al., 2013)** — contrast-enhanced liver ultrasound cine loops with lesion characterization ground truth; follows EFSUMB guidelines
- **ULM Simulation Dataset (Couture et al., 2018)** — simulated microbubble signals for Ultrasound Localization Microscopy with known microvasculature ground truth
- **SonoVue Perfusion Phantom Dataset (Averkiou et al., 2010)** — flow phantom with contrast agent bolus kinetics; calibrated perfusion curves
- **CEUS Brain Perfusion Dataset (Eyding et al., 2006)** — transcranial CEUS with perfusion time-intensity curves

**Decision criteria:** ULM simulation data provides exact micro-vascular ground truth for super-resolution algorithms. Liver lesion CEUS datasets are the most clinically cited.

#### Step 2: List All CEUS Algorithms

Please first ensure all the CEUS algorithms have been listed in `\pwm\public\algorithm_base\ceus\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ceus. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CEUS solvers, please update the CEUS solver.

**Key algorithms to cover (1950–2026):**

_Classical / Nonlinear Imaging (1990s–2005):_
- Harmonic imaging — second harmonic detection from microbubble nonlinear oscillation (Burns et al., 1992)
- Pulse inversion — subtraction of inverted-phase pulse pairs for harmonic extraction (Simpson et al., 1999)
- Power modulation — amplitude-modulated multi-pulse technique (Brock-Fisher et al., 1996)
- Contrast pulse sequencing (CPS) — combined phase/amplitude modulation (Phillips, 2001)
- Time-intensity curve (TIC) analysis — bolus kinetics for perfusion quantification (Cosgrove, 2006)

_Optimization / Model-Based (2005–2016):_
- Lognormal perfusion model fitting — parametric TIC analysis (Strouthos et al., 2010)
- Destruction-replenishment kinetics — flash-replenishment for flow rate estimation (Wei et al., 1998)
- SVD-based microbubble separation — singular value decomposition for tissue/bubble separation (Errico et al., 2015)
- Ultrasound Localization Microscopy (ULM) — super-resolution from individual microbubble tracking (Errico et al., 2015; Christensen-Jeffries et al., 2015)
- CEUS Maximum Intensity Projection — temporal MIP for vascular mapping
- Motion compensation for CEUS — rigid and deformable registration (Leen et al., 2012)
- Linearization of contrast signal — log compression inversion (Tang et al., 2008)
- Spectral analysis of microbubble signals (Renaud et al., 2012)
- RPCA for CEUS tissue/bubble separation (Solomon et al., 2019)

_Deep Learning (2017–2026):_
- Deep-ULM — deep learning for microbubble localization (van Sloun et al., 2021)
- mSPCN — microbubble signal processing CNN (Liu et al., 2020)
- DL CEUS liver lesion classification (Guo et al., 2021)
- U-Net for CEUS perfusion map estimation (2021)
- Physics-informed bubble dynamics network (2022)
- Transformer for CEUS temporal analysis (2023)
- Foundation model for contrast ultrasound (2025)

#### Step 3: Update CEUS Solvers

After listing all CEUS solvers, update `algorithm_base/ceus/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CEUS solvers use the data format: `y` (num_frames, num_elements, num_samples) raw RF/IQ data from multi-pulse contrast sequences (pulse inversion pairs or CPS triads). The `CEUSOperator` handles the nonlinear microbubble forward model based on Rayleigh-Plesset bubble dynamics, combined with multi-pulse extraction schemes and beamforming.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CEUS:**
- ULM simulation (localization RMSE in um): Centroid ~30, Radial symmetry ~15, Deep-ULM ~8
- ULM vessel diameter accuracy (%): Classical ~20%, SVD-filtered ~10%, DL-based ~5%
- Liver lesion classification accuracy: TIC analysis ~75%, DL-based ~88%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ceus' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ceus/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ceus/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ceus/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CEUS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ceus/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ceus/standard/`

---

### Shear-Wave Elastography (`elastography`) Modality Template

#### Step 1: Verify Standard Dataset

For Shear-Wave Elastography, what dataset do you use to verify? Is this dataset used for elastography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/elastography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original elastography standard dataset.

**Popular datasets to consider:**
- **QIBA Ultrasound SWE Phantom Dataset (Hall et al., 2013)** — RSNA QIBA standardized elasticity phantoms with known Young's modulus values; canonical SWE benchmark
- **CIRS Elastography Phantom Dataset** — tissue-mimicking phantoms with calibrated elastic inclusions for shear-wave speed validation
- **k-Wave SWE Simulation (Treeby & Cox, 2010)** — simulated shear-wave propagation with exact elasticity ground truth
- **Breast SWE Clinical Dataset (Berg et al., 2012)** — shear-wave elastography of breast lesions with biopsy correlation

**Decision criteria:** QIBA SWE phantom data provides traceable elasticity ground truth. k-Wave simulations provide exact shear modulus maps for algorithm evaluation.

#### Step 2: List All Shear-Wave Elastography Algorithms

Please first ensure all the Shear-Wave Elastography algorithms have been listed in `\pwm\public\algorithm_base\elastography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/elastography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Elastography solvers, please update the Elastography solver.

**Key algorithms to cover (1950–2026):**

_Classical / Static Elastography (1990s–2005):_
- Strain imaging — quasi-static compression with displacement tracking (Ophir et al., 1991)
- Cross-correlation displacement estimation — time-delay estimation between pre/post compression RF (Ophir et al., 1991)
- Sonoelastography — low-frequency vibration with Doppler detection (Lerner et al., 1990)
- ARFI — Acoustic Radiation Force Impulse for local tissue displacement (Nightingale et al., 2001)
- Shear wave speed estimation from ARFI push (Sarvazyan et al., 1998)

_Optimization / Model-Based (2005–2016):_
- SSI — Supersonic Shear Imaging using Mach cone (Bercoff et al., 2004)
- Time-of-flight shear wave speed estimation — directional filtering and arrival time (Palmeri et al., 2008)
- 2D shear wave speed recovery — local frequency estimation from shear wave movies (Deffieux et al., 2009)
- Algebraic Helmholtz inversion for SWE (Manduca et al., 2001, applied to US)
- Robust shear wave speed estimation — RANSAC and Radon transform (Wang et al., 2010)
- Viscoelastic characterization — complex shear modulus from dispersion (Chen et al., 2009)
- Crawling wave sonoelastography (Wu et al., 2006)
- Comb-push ultrasound shear elastography (CUSE) (Song et al., 2012)
- Reverberant shear wave elastography (Ormachea et al., 2019)

_Deep Learning (2017–2026):_
- CNN for shear wave speed estimation — direct image-to-elasticity mapping (Kibria & Rivaz, 2018)
- DL displacement tracking — learned RF correlation (Tehrani & Rivaz, 2020)
- U-Net for SWE artifact removal (2020)
- Physics-informed neural network for wave equation inversion (2022)
- GAN-based SWE quality enhancement (2022)
- Transformer for temporal SWE analysis (2023)
- Foundation model for elastography (2025)

#### Step 3: Update Shear-Wave Elastography Solvers

After listing all Elastography solvers, update `algorithm_base/elastography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Elastography solvers use the data format: `y` (num_frames, num_elements, num_samples) ultrafast IQ data capturing shear wave propagation after acoustic radiation force push. The `ElastographyOperator` handles the shear wave equation `rho * d^2u/dt^2 = mu * nabla^2 u` relating tissue displacement u to shear modulus mu, with beamformed particle velocity movies as intermediate representation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Shear-Wave Elastography:**
- QIBA phantom SWS bias (%): TOF ~5%, Algebraic ~3%, DL-based ~1.5%
- k-Wave simulation (SWS RMSE m/s): TOF ~0.15, Helmholtz ~0.08, DL-based ~0.04
- Breast lesion classification AUC: SWE ~0.93, DL-enhanced ~0.96
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'elastography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/elastography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/elastography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/elastography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Shear-Wave Elastography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/elastography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/elastography/standard/`

---

### Optical Coherence Tomography (`oct`) Modality Template

#### Step 1: Verify Standard Dataset

For OCT, what dataset do you use to verify? Is this dataset used for OCT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/oct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original OCT standard dataset.

**Popular datasets to consider:**
- **Duke DME OCT Dataset (Chiu et al., 2015)** — spectral-domain OCT B-scans with manual retinal layer segmentation; widely used layer segmentation benchmark
- **Kermany OCT Dataset (Kermany et al., 2018)** — 84,000 retinal OCT images with disease classification labels (CNV, DME, drusen, normal); largest OCT classification dataset
- **RETOUCH Challenge (Bogunović et al., 2019)** — multi-vendor retinal OCT with fluid segmentation ground truth; the canonical OCT segmentation challenge
- **ROSE (Retinal OCT SEgmentation, 2020)** — retinal OCT with layer and pathology annotations
- **OCTA-500 (Li et al., 2020)** — combined OCT/OCTA dataset with 500 subjects and multi-layer annotations

**Decision criteria:** Kermany dataset is the most widely used OCT classification benchmark. RETOUCH for multi-vendor segmentation. Duke DME for layer segmentation.

#### Step 2: List All OCT Algorithms

Please first ensure all the OCT algorithms have been listed in `\pwm\public\algorithm_base\oct\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/oct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the OCT solvers, please update the OCT solver.

**Key algorithms to cover (1950–2026):**

_Classical / Signal Processing (1990s–2005):_
- Time-domain OCT reconstruction — low-coherence interferometry with reference mirror scanning (Huang et al., 1991)
- Fourier-domain OCT — spectral interferogram to depth profile via FFT (Fercher et al., 1995; Wojtkowski et al., 2002)
- Swept-source OCT — wavelength-swept laser with balanced detection (Chinn et al., 1997; Yun et al., 2003)
- Dispersion compensation — numerical correction of chromatic dispersion mismatch (Wojtkowski et al., 2004)
- k-space resampling / recalibration — mapping from wavelength to wavenumber (Huber et al., 2005)

_Optimization / Model-Based (2005–2016):_
- Graph-cut retinal layer segmentation (Chiu et al., 2010)
- Sparse OCT reconstruction — compressed sensing for accelerated volumetric OCT (Liu & Kang, 2010)
- BM3D denoising for OCT speckle (Dabov et al., 2007, applied to OCT; Fang et al., 2012)
- Dynamic programming for layer boundary detection (Yazdanpanah et al., 2011)
- Random forest–based layer segmentation (Lang et al., 2013)
- Averaging-based speckle reduction — multi-frame registration and averaging (Jørgensen et al., 2007)
- Complex OCT — full-range imaging via phase shifting (Wojtkowski et al., 2002)
- Non-local means denoising for OCT (Coupe et al., 2008)
- Optical coherence elastography — phase-sensitive displacement detection (Kennedy et al., 2015)

_Deep Learning (2017–2026):_
- ReLayNet — retinal layer segmentation network (Roy et al., 2017)
- CNN for OCT disease classification — transfer learning from ImageNet (Kermany et al., 2018)
- DL-OCT denoising — Noise2Void for speckle reduction (Krull et al., 2019, applied to OCT)
- U-Net for retinal fluid segmentation (Schlegl et al., 2018)
- OCT super-resolution with SRGAN (Das et al., 2020)
- Self-supervised OCT denoising (2022)
- Transformer for volumetric OCT analysis (2023)
- Foundation model for ophthalmic OCT (2025)

#### Step 3: Update OCT Solvers

After listing all OCT solvers, update `algorithm_base/oct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All OCT solvers use the data format: `y` (num_Ascans, num_spectral_points) raw spectral interferograms, or (num_Bscans, H, W) processed B-scan volumes. The `OCTOperator` handles the spectral-domain OCT forward model `I(k) = |E_r + E_s|^2 = S(k) * [1 + 2*Re{r(z)*exp(i*2*k*z)}]` where S(k) is source spectrum, r(z) is sample reflectivity profile, mapping spectral interferograms to depth-resolved reflectivity via inverse Fourier transform.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for OCT:**
- Duke DME layer segmentation (mean boundary error um): Graph-cut ~4.0, ReLayNet ~2.5, DL-based ~1.8
- Kermany classification accuracy: Transfer CNN ~96%, ViT ~98%
- RETOUCH fluid segmentation Dice: Random forest ~0.65, U-Net ~0.78, DL-based ~0.84
- OCT denoising PSNR: BM3D ~30.0 dB, Noise2Void ~33.0 dB, DL-based ~35.0 dB
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'oct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/oct/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/oct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/oct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for OCT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/oct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/oct/standard/`

---

### OCT Angiography (`octa`) Modality Template

#### Step 1: Verify Standard Dataset

For OCTA, what dataset do you use to verify? Is this dataset used for OCTA popular algorithms? Please ensure the standard dataset in `datasets/benchmark/octa/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original OCTA standard dataset.

**Popular datasets to consider:**
- **OCTA-500 (Li et al., 2020)** — 500 subjects with 3mm and 6mm OCTA volumes, FAZ annotations, and retinal layer segmentation; the canonical OCTA benchmark
- **ROSE (Retinal OCT-Angiography vessel SEgmentation, Ma et al., 2021)** — OCTA en-face images with manual vessel annotations for segmentation benchmarking
- **FAZID (FAZ in Diabetics, Diaz et al., 2019)** — OCTA images with foveal avascular zone segmentation in diabetic patients
- **DRAC Challenge (2022)** — diabetic retinopathy analysis from ultra-widefield OCTA

**Decision criteria:** OCTA-500 is the largest and most comprehensive OCTA benchmark. ROSE for vessel segmentation. DRAC for clinical grading tasks.

#### Step 2: List All OCTA Algorithms

Please first ensure all the OCTA algorithms have been listed in `\pwm\public\algorithm_base\octa\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/octa. Besides, you need to search all algorithms from 1950 to 2026. After listing all the OCTA solvers, please update the OCTA solver.

**Key algorithms to cover (1950–2026):**

_Classical / Motion Contrast (2000–2012):_
- SSADA — Split-Spectrum Amplitude-Decorrelation Angiography (Jia et al., 2012)
- OMAG — Optical Microangiography, differential phase/amplitude detection (Wang et al., 2007)
- Phase-variance OCT — inter-B-scan phase difference for flow detection (Fingler et al., 2007)
- Speckle-variance OCT — inter-frame speckle decorrelation (Mariampillai et al., 2008)
- Doppler OCT — phase-sensitive velocity mapping (Chen et al., 1997; Leitgeb et al., 2003)

_Optimization / Model-Based (2012–2020):_
- Eigen-decomposition OCTA — SVD-based separation of static/moving tissue (Zhang et al., 2016)
- Complex-differential-variance OCTA — combined amplitude and phase contrast (Nam & Bhatt, 2014)
- Bulk motion correction — registration-based artifact suppression (Kraus et al., 2012)
- Projection artifact removal — slab subtraction and normalization (Zhang et al., 2016)
- FAZ segmentation — morphometric analysis of foveal avascular zone (Díaz et al., 2019)
- Layer-by-layer en-face projection — superficial/deep/outer retina vascular maps
- Motion-corrected averaging for OCTA quality improvement (Camino et al., 2016)
- Frangi vesselness for OCTA vessel enhancement (Frangi et al., 1998, applied to OCTA)
- Hessian-based OCTA vessel segmentation (Mou et al., 2019)

_Deep Learning (2017–2026):_
- CNN for OCTA vessel segmentation (Mou et al., 2019)
- GAN for OCTA image synthesis from structural OCT (Lee et al., 2020)
- U-Net for FAZ segmentation (Guo et al., 2019)
- DL projection artifact removal (Hormel et al., 2021)
- Self-supervised OCTA super-resolution (2022)
- OCTA quality enhancement with diffusion models (2023)
- Transformer for OCTA-based disease grading (2023)
- Foundation model for OCTA (2025)

#### Step 3: Update OCTA Solvers

After listing all OCTA solvers, update `algorithm_base/octa/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All OCTA solvers use the data format: `y` (num_repeats, num_Bscans, num_Ascans, num_spectral_points) repeated OCT spectral data at same location, or (num_repeats, H, W) complex-valued B-scans. The `OCTAOperator` handles the angiographic forward model computing inter-scan decorrelation metrics (amplitude, phase, or complex) to separate static tissue from moving blood cells.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for OCTA:**
- ROSE vessel segmentation Dice: Frangi ~0.65, CNN ~0.78, DL-based ~0.83
- OCTA-500 FAZ segmentation Dice: Thresholding ~0.88, U-Net ~0.94, DL-based ~0.96
- OCTA-500 vessel density MAE (%): Manual ~3.0%, DL-based ~1.5%
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'octa' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/octa/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/octa/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/octa/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for OCTA. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/octa/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/octa/standard/`

---

### Photoacoustic Imaging (`photoacoustic`) Modality Template

#### Step 1: Verify Standard Dataset

For Photoacoustic Imaging, what dataset do you use to verify? Is this dataset used for photoacoustic popular algorithms? Please ensure the standard dataset in `datasets/benchmark/photoacoustic/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original photoacoustic standard dataset.

**Popular datasets to consider:**
- **IPASC Photoacoustic Simulation Dataset (Gröhl et al., 2022)** — International Photoacoustic Standardization Consortium; standardized simulated PA data with known absorption maps
- **k-Wave PA Simulation (Treeby & Cox, 2010)** — simulated PA pressure data with exact initial pressure ground truth; the most widely used PA simulation tool
- **OADAT (Optoacoustic Data Analysis Toolbox, 2023)** — multi-target PA imaging dataset with absorption ground truth
- **Photoacoustic Breast Phantom (Xia et al., 2014)** — tissue-mimicking phantom with known vasculature and optical properties

**Decision criteria:** k-Wave simulations and IPASC datasets provide exact initial pressure ground truth for reconstruction algorithm comparison. Use datasets with known absorption distributions.

#### Step 2: List All Photoacoustic Algorithms

Please first ensure all the Photoacoustic algorithms have been listed in `\pwm\public\algorithm_base\photoacoustic\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/photoacoustic. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Photoacoustic solvers, please update the Photoacoustic solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1990s–2005):_
- Universal back-projection — time-reversal analytic reconstruction (Xu & Wang, 2005)
- Delay-and-sum beamforming — simple PA image formation from time-domain signals (Hoelen et al., 1998)
- Fourier-domain reconstruction — wavenumber domain PA inversion (Köstli et al., 2001)
- Filtered back-projection for PA — adapted from CT FBP (Finch et al., 2004)
- Spherical Radon transform inversion (Kunyansky, 2007)

_Optimization / Model-Based (2005–2016):_
- Time-reversal reconstruction — boundary value re-propagation (Burgholzer et al., 2007; Treeby et al., 2010)
- Model-based iterative reconstruction for PA (Rosenthal et al., 2010; Huang et al., 2013)
- Total Variation regularized PA (Arridge et al., 2016)
- Compressed sensing PA — sparse recovery from limited sensors (Provost & Bhatt, 2009)
- Multi-spectral PA — spectral unmixing for chromophore mapping (Laufer et al., 2007)
- Quantitative PA — light fluence correction for absorption recovery (Cox et al., 2006)
- Acoustic attenuation compensation (La Rivière et al., 2006)
- Limited-view artifact mitigation (Xu et al., 2004)
- Joint optical-acoustic inversion (Bal & Ren, 2011)

_Deep Learning (2017–2026):_
- CNN for PA image reconstruction from limited data (Antholzer et al., 2019)
- Learned iterative PA reconstruction (Hauptmann et al., 2018)
- U-Net for PA artifact removal (Allman et al., 2018)
- DL spectral unmixing for PA (Gröhl et al., 2021)
- Physics-informed neural network for PA inversion (2022)
- Diffusion model for PA image enhancement (2023)
- Transformer for volumetric PA reconstruction (2023)
- Foundation model for photoacoustic imaging (2025)

#### Step 3: Update Photoacoustic Solvers

After listing all Photoacoustic solvers, update `algorithm_base/photoacoustic/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Photoacoustic solvers use the data format: `y` (num_sensors, num_timepoints) time-resolved acoustic pressure signals recorded by ultrasound transducer array. The `PhotoacousticOperator` handles the PA wave equation forward model `nabla^2 p - (1/c^2) d^2p/dt^2 = -(beta/(Cp)) dH/dt` with initial pressure `p0 = Gamma * mu_a * Phi` relating absorbed optical energy (absorption coefficient mu_a, fluence Phi) to initial pressure via Grüneisen parameter Gamma.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Photoacoustic:**
- k-Wave 2D phantom: DAS ~24.0 dB, UBP ~28.0 dB, Time-reversal ~32.0 dB, Model-based ~35.0 dB, DL-based ~37.0 dB
- Limited-view (90° arc): UBP ~20.0 dB, TV-regularized ~26.0 dB, DL-based ~31.0 dB
- Spectral unmixing SO2 RMSE (%): Linear ~12%, DL-based ~5%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'photoacoustic' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/photoacoustic/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Photoacoustic Imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/photoacoustic/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/photoacoustic/standard/`

---

### Diffuse Optical Tomography (`dot`) Modality Template

#### Step 1: Verify Standard Dataset

For Diffuse Optical Tomography, what dataset do you use to verify? Is this dataset used for DOT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/dot/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original DOT standard dataset.

**Popular datasets to consider:**
- **TOAST++ Simulation Dataset (Schweiger & Arridge, 2014)** — finite element DOT simulation with known absorption and scattering maps; the canonical DOT reconstruction benchmark
- **PMI Toolbox Phantoms (Dehghani et al., 2009)** — NIRFAST simulation toolkit phantoms with exact optical property ground truth
- **Multi-modality Breast DOT Dataset (Brooksby et al., 2006)** — clinical DOT breast data with MRI structural prior
- **CW-DOT Phantom Dataset (Yamada et al., 2009)** — continuous-wave DOT measurements on tissue-mimicking phantoms with known inclusions

**Decision criteria:** TOAST++ and NIRFAST simulations provide exact optical property ground truth. Use simulation datasets for reconstruction algorithm comparison.

#### Step 2: List All DOT Algorithms

Please first ensure all the DOT algorithms have been listed in `\pwm\public\algorithm_base\dot\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/dot. Besides, you need to search all algorithms from 1950 to 2026. After listing all the DOT solvers, please update the DOT solver.

**Key algorithms to cover (1950–2026):**

_Classical / Linear (1980s–2005):_
- Modified Beer-Lambert Law (MBLL) — linear approximation for absorption changes (Delpy et al., 1988)
- Back-projection for DOT — simple tomographic reconstruction from boundary measurements (Arridge & Schweiger, 1993)
- Diffusion equation Green's function — analytic forward model for homogeneous media (Patterson et al., 1989)
- Perturbation-based linearized reconstruction — Born/Rytov approximation (O'Leary et al., 1995)
- Singular Value Decomposition (SVD) reconstruction for DOT (Arridge, 1999)

_Optimization / Model-Based (2005–2016):_
- Nonlinear iterative reconstruction — Gauss-Newton / Levenberg-Marquardt for DOT (Arridge & Schweiger, 1998)
- FEM-based DOT forward model — finite element solution of diffusion equation (Schweiger et al., 1995)
- Tikhonov-regularized DOT (Arridge, 1999)
- L1/TV-regularized DOT — sparse and edge-preserving priors (Correia et al., 2011)
- Time-domain DOT — temporal point spread function analysis (Ntziachristos et al., 2001)
- Frequency-domain DOT — amplitude and phase measurements (Fantini et al., 1995)
- Structural prior DOT — MRI/CT-guided soft priors (Brooksby et al., 2006)
- Multi-spectral DOT — spectral constraints for chromophore estimation (Corlu et al., 2005)
- Transport-equation–based DOT — RTE solver replacing diffusion approximation (Klose & Hielscher, 1999)

_Deep Learning (2017–2026):_
- CNN for DOT image reconstruction (Yoo et al., 2020)
- DL-DOT — end-to-end learned optical tomography (Ben Yedder et al., 2021)
- Physics-informed neural network for DOT (2022)
- U-Net for DOT artifact removal (2021)
- Implicit neural representation for DOT (2023)
- Score-based diffusion for DOT reconstruction (2023)
- Foundation model for optical tomography (2025)

#### Step 3: Update DOT Solvers

After listing all DOT solvers, update `algorithm_base/dot/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All DOT solvers use the data format: `y` (num_sources, num_detectors) boundary intensity measurements (CW-DOT), or (num_sources, num_detectors, num_timepoints) temporal point spread functions (TD-DOT), or (num_sources, num_detectors, 2) amplitude and phase (FD-DOT). The `DOTOperator` handles the photon diffusion equation forward model `-nabla . (D(r) nabla Phi(r)) + mu_a(r) Phi(r) = S(r)` with diffusion coefficient D = 1/(3*(mu_a + mu_s')) mapping absorption mu_a and reduced scattering mu_s' to boundary photon density Phi.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for DOT:**
- TOAST++ phantom (absorption RMSE %): Linear ~25%, Gauss-Newton ~12%, TV-regularized ~8%, DL-based ~4%
- Inclusion detection CNR: Back-projection ~3.0, Nonlinear ~8.0, DL-based ~12.0
- Spatial resolution (mm): Linear ~15, Nonlinear ~10, DL-based ~7
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'dot' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/dot/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/dot/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/dot/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for DOT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/dot/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/dot/standard/`

---

### Functional Near-Infrared Spectroscopy (`nirs_brain`) Modality Template

#### Step 1: Verify Standard Dataset

For fNIRS, what dataset do you use to verify? Is this dataset used for fNIRS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/nirs_brain/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original fNIRS standard dataset.

**Popular datasets to consider:**
- **fNIRS2MW (Shin et al., 2018)** — multi-wavelength fNIRS dataset with motor imagery and mental arithmetic tasks; widely used for BCI benchmarking
- **Open Access fNIRS Dataset (Bak et al., 2019)** — multi-session fNIRS with finger tapping tasks; canonical fNIRS signal processing benchmark
- **ICBM fNIRS Atlas Dataset (2020)** — co-registered fNIRS-MRI data for spatial validation
- **TU Berlin fNIRS Dataset (Shin et al., 2017)** — simultaneous fNIRS-EEG for multi-modal BCI evaluation

**Decision criteria:** fNIRS2MW is the most widely used fNIRS processing benchmark. Open Access fNIRS for basic hemodynamic response function evaluation.

#### Step 2: List All fNIRS Algorithms

Please first ensure all the fNIRS algorithms have been listed in `\pwm\public\algorithm_base\nirs_brain\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/nirs_brain. Besides, you need to search all algorithms from 1950 to 2026. After listing all the fNIRS solvers, please update the fNIRS solver.

**Key algorithms to cover (1950–2026):**

_Classical / Signal Processing (1990s–2005):_
- Modified Beer-Lambert Law (MBLL) — continuous-wave intensity-to-concentration conversion (Cope & Delpy, 1988)
- Band-pass filtering — hemodynamic band extraction (0.01–0.2 Hz) (Hoshi & Tamura, 1993)
- OD to concentration conversion — differential pathlength factor (DPF) estimation (Duncan et al., 1995)
- Motion artifact detection and rejection — threshold-based channel rejection (Cooper et al., 2012)
- Block averaging — trial-averaged HRF estimation (Huppert et al., 2006)

_Optimization / Model-Based (2005–2016):_
- GLM for fNIRS — General Linear Model adapted from fMRI for HRF estimation (Ye et al., 2009; Huppert et al., 2009)
- Short-separation regression — systemic physiology removal (Saager & Berger, 2005)
- Wavelet-based motion correction — wavelet de-spiking for fNIRS (Molavi & Dumont, 2012)
- PCA/ICA for fNIRS artifact removal — principal/independent component analysis (Zhang et al., 2005)
- Temporal derivative distribution repair (TDDR) — robust motion correction (Fishburn et al., 2019)
- Bayesian fNIRS — hierarchical Bayesian HRF estimation (Tak & Ye, 2014)
- DOT-based fNIRS reconstruction — tomographic image reconstruction from fNIRS (Boas et al., 2004)
- Adaptive filter for physiological noise removal (Bauernfeind et al., 2014)
- Granger causality and functional connectivity analysis (Tak & Ye, 2014)

_Deep Learning (2017–2026):_
- CNN for fNIRS-based BCI classification (Trakoolwilaiwan et al., 2018)
- LSTM for temporal fNIRS decoding (Asghar et al., 2020)
- DL motion artifact correction for fNIRS (Yang et al., 2022)
- Transformer for fNIRS brain-computer interface (2022)
- Self-supervised fNIRS representation learning (2023)
- GAN-based fNIRS signal enhancement (2023)
- Foundation model for functional neuroimaging (2025)

#### Step 3: Update fNIRS Solvers

After listing all fNIRS solvers, update `algorithm_base/nirs_brain/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All fNIRS solvers use the data format: `y` (num_channels, num_timepoints, num_wavelengths) raw optical density time series at 2+ wavelengths (typically 760nm and 850nm). The `fNIRSOperator` handles the Modified Beer-Lambert Law `delta_OD(lambda,t) = epsilon_HbO(lambda) * delta[HbO](t) * DPF(lambda) * d + epsilon_HbR(lambda) * delta[HbR](t) * DPF(lambda) * d` mapping concentration changes in oxy/deoxy-hemoglobin to optical density changes via molar extinction coefficients epsilon and source-detector distance d.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for fNIRS:**
- fNIRS2MW BCI accuracy: MBLL+LDA ~72%, CSP+SVM ~78%, CNN ~83%, Transformer ~86%
- HRF estimation RMSE (uM): Block avg ~0.8, GLM ~0.5, Bayesian ~0.3
- Motion artifact correction (residual artifact %): Wavelet ~15%, TDDR ~8%, DL-based ~4%
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'nirs_brain' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/nirs_brain/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for fNIRS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/nirs_brain/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/nirs_brain/standard/`

---

### X-ray Radiography (`xray_radiography`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray Radiography, what dataset do you use to verify? Is this dataset used for X-ray radiography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/xray_radiography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original X-ray radiography standard dataset.

**Popular datasets to consider:**
- **CheXpert (Irvin et al., 2019)** — 224,316 chest X-rays with multi-label pathology annotations from Stanford; one of the largest CXR benchmarks
- **MIMIC-CXR (Johnson et al., 2019)** — 377,110 chest X-rays with free-text radiology reports from BIDMC; the largest public CXR dataset
- **NIH ChestX-ray14 (Wang et al., 2017)** — 112,120 frontal chest X-rays with 14 disease labels; the original large-scale CXR benchmark
- **VinDr-CXR (Nguyen et al., 2022)** — 18,000 chest X-rays with bounding box annotations for 22 findings
- **JSRT (Japanese Society of Radiological Technology, Shiraishi et al., 2000)** — 247 chest X-rays with lung nodule annotations; canonical lung segmentation benchmark

**Decision criteria:** CheXpert and MIMIC-CXR are the gold standard for chest X-ray classification. NIH ChestX-ray14 for broad multi-label evaluation. JSRT for segmentation tasks.

#### Step 2: List All X-ray Radiography Algorithms

Please first ensure all the X-ray Radiography algorithms have been listed in `\pwm\public\algorithm_base\xray_radiography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/xray_radiography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the X-ray Radiography solvers, please update the X-ray Radiography solver.

**Key algorithms to cover (1950–2026):**

_Classical / Image Processing (1950s–2005):_
- Analog-to-digital radiography — computed radiography (CR) with photostimulable phosphor (Sonoda et al., 1983)
- Flat-field correction — gain and offset calibration for digital radiography (Seibert et al., 1998)
- Unsharp masking — edge enhancement for digital radiography (Prokop et al., 2003)
- Histogram equalization / CLAHE for X-ray contrast enhancement (Pizer et al., 1987)
- Anti-scatter grid correction — scatter rejection for radiography (Neitzel, 1992)

_Optimization / Model-Based (2005–2016):_
- Dual-energy subtraction radiography — tissue/bone decomposition (Kelcz et al., 1994)
- Scatter correction using Monte Carlo (Kyriakou et al., 2006)
- CADe for chest X-ray — computer-aided detection for lung nodules (Li et al., 2003)
- Bone suppression — temporal subtraction and dual-energy for rib removal (Suzuki et al., 2006)
- Contrast enhancement with multi-scale decomposition (Stahl et al., 2000)
- Noise reduction with structure-adaptive filtering (Defined et al., 2010)
- Image stitching for full-spine/full-leg radiography (2010)
- Exposure index standardization (IEC 62494)
- Tomosynthesis from radiographic projections (Dobbins & Godfrey, 2003)

_Deep Learning (2017–2026):_
- CheXNet — 121-layer DenseNet for chest X-ray pathology (Rajpurkar et al., 2017)
- DL bone suppression — learned tissue decomposition (Yang et al., 2017)
- U-Net for lung segmentation from CXR (Souza et al., 2019)
- GAN-based scatter correction for DR (Maier et al., 2019)
- CXR-Foundation — pre-trained models for chest X-ray (Sellergren et al., 2022)
- DL dose reduction for radiography (2021)
- Vision-language models for CXR report generation (2023)
- Foundation model for radiography (CheXZero, Tiu et al., 2022)

#### Step 3: Update X-ray Radiography Solvers

After listing all X-ray Radiography solvers, update `algorithm_base/xray_radiography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All X-ray Radiography solvers use the data format: `y` (H, W) raw detector image (DICOM "For Processing"), or (2, H, W) dual-energy image pair. The `XrayRadiographyOperator` handles the X-ray projection model `y = integral I0(E) * exp(-integral mu(x,E) dl) * D(E) dE` with polyenergetic spectrum I0(E), linear attenuation mu, detector response D(E), and scatter contribution.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for X-ray Radiography:**
- CheXpert 5-class AUC: CheXNet ~0.89, DenseNet ~0.90, CheXZero ~0.93
- JSRT lung segmentation Dice: Atlas ~0.92, U-Net ~0.97
- Dose-reduced CXR (25% dose) PSNR: Raw ~26.0 dB, BM3D ~31.0 dB, DL-denoised ~35.0 dB
- Bone suppression PSNR: Dual-energy ~32.0 dB, DL-based ~36.0 dB
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'xray_radiography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/xray_radiography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/xray_radiography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/xray_radiography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for X-ray Radiography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/xray_radiography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_radiography/standard/`

---

### Magnetic Particle Imaging (`magnetic_particle`) Modality Template

#### Step 1: Verify Standard Dataset

For Magnetic Particle Imaging, what dataset do you use to verify? Is this dataset used for MPI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/magnetic_particle/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MPI standard dataset.

**Popular datasets to consider:**
- **Open MPI Data (Knopp et al., 2020)** — open-access MPI dataset with system matrix measurements and phantom data; the canonical MPI reconstruction benchmark
- **MPI Simulation Toolkit (Knopp & Buzug, 2012)** — simulated MPI signals with known nanoparticle concentrations; exact ground truth for reconstruction validation
- **Bruker MPI Preclinical Dataset** — preclinical MPI scans of mouse phantoms with known tracer distributions
- **Berlin MPI Phantom Dataset (Weizenecker et al., 2009)** — early MPI system data demonstrating real-time imaging capability

**Decision criteria:** Open MPI Data is the community standard for MPI reconstruction benchmarking with system matrix and x-space datasets. Simulation toolkits provide exact concentration ground truth.

#### Step 2: List All MPI Algorithms

Please first ensure all the MPI algorithms have been listed in `\pwm\public\algorithm_base\magnetic_particle\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/magnetic_particle. Besides, you need to search all algorithms from 1950 to 2026. After listing all the MPI solvers, please update the MPI solver.

**Key algorithms to cover (1950–2026):**

_Classical / System Matrix (2005–2012):_
- System matrix reconstruction — Kaczmarz / ART iterative solver (Gleich & Weizenecker, 2005; Knopp et al., 2010)
- X-space MPI — direct Fourier velocity–based reconstruction (Goodwill & Conolly, 2010)
- Singular Value Decomposition (SVD) reconstruction for MPI (Knopp et al., 2010)
- Frequency-domain MPI — harmonic analysis of nanoparticle response (Gleich & Weizenecker, 2005)
- Deconvolution-based MPI — PSF deconvolution in x-space (Goodwill & Conolly, 2011)

_Optimization / Model-Based (2012–2020):_
- Regularized Kaczmarz for MPI — Tikhonov and TV regularization (Knopp et al., 2010; Storath et al., 2017)
- Model-based system matrix — Langevin model for tracer response (Knopp et al., 2011; Rahmer et al., 2012)
- Multi-patch MPI reconstruction — field-free-point scanning (Knopp et al., 2015)
- Joint multi-contrast MPI — multi-color nanoparticle separation (Rahmer et al., 2015)
- Compressed sensing MPI — sparse recovery from undersampled system matrix (Ilbey et al., 2017)
- Background correction for MPI — direct feedthrough removal (Them et al., 2016)
- Rowspace-based MPI — efficient system matrix compression (Knopp et al., 2019)
- 3D Lissajous trajectory MPI reconstruction (Weizenecker et al., 2009)
- Joint estimation of nanoparticle core-size distribution (Weizenecker et al., 2012)

_Deep Learning (2017–2026):_
- DL system matrix calibration — learned system function from sparse measurements (Askin et al., 2022)
- CNN for MPI image denoising (Gungor et al., 2021)
- ADMM-Net for MPI reconstruction (Dittmer et al., 2020)
- U-Net for MPI artifact removal (2021)
- Physics-informed MPI reconstruction (2022)
- Score-based diffusion for MPI (2023)
- Foundation model for particle imaging (2025)

#### Step 3: Update MPI Solvers

After listing all MPI solvers, update `algorithm_base/magnetic_particle/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MPI solvers use the data format: `y` (num_receive_channels, num_timepoints) voltage signals induced in receive coils, or (num_frequencies, num_channels) frequency-domain harmonic amplitudes. The `MPIOperator` handles the system matrix forward model `u(t) = -dPhi/dt = -mu_0 * integral p(r) * dM(H(r,t))/dt * S(r) dr` where M(H) is the nonlinear nanoparticle magnetization (Langevin function), H(r,t) is the time-varying applied field (drive + selection), p(r) is particle concentration, and S(r) is receive coil sensitivity.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MPI:**
- Open MPI phantom: Kaczmarz ~28.0 dB, Regularized Kaczmarz ~32.0 dB, TV-regularized ~34.0 dB, DL-based ~36.0 dB
- X-space 1D: Deconvolution ~30.0 dB, DL-enhanced ~34.0 dB
- Spatial resolution (mm): System matrix ~1.5, Regularized ~1.0, DL-based ~0.7
- All reference values from published papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'magnetic_particle' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/magnetic_particle/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MPI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/magnetic_particle/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/magnetic_particle/standard/`
