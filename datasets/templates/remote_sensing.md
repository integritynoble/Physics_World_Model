
---

## Remote Sensing & Radar — Modality Templates

---

### SAR (`sar`) Modality Template

#### Step 1: Verify Standard Dataset

For SAR, what dataset do you use to verify? Is this dataset used for SAR popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SAR standard dataset.

**Popular datasets to consider:**
- **MSTAR (Moving and Stationary Target Acquisition and Recognition, AFRL, 1995–1998)** — the most widely used SAR automatic target recognition benchmark; 10 military vehicle classes at multiple depression angles; X-band spotlight SAR chips; used by virtually all SAR ATR papers since 1998
- **SEN1-2 (Schmitt et al., 2018)** — 282,384 paired Sentinel-1 SAR and Sentinel-2 optical image patches; used for SAR-optical fusion, SAR image translation, and SAR despeckling benchmarks
- **SpaceNet SAR (SpaceNet 6, Shermeyer et al., 2020)** — multi-sensor SAR and EO building footprint dataset over Rotterdam; 120k building labels; used for SAR building extraction benchmarks
- **TerraSAR-X Public Datasets (DLR)** — high-resolution X-band spotlight and stripmap SAR data; used for image formation, autofocus, and change detection validation
- **SAMPLE (Synthetic and Measured Paired and Labeled Experiment, Lewis et al., 2019)** — paired synthetic and measured SAR target chips; used for domain adaptation SAR ATR
- **OpenSARShip (Huang et al., 2017)** — Sentinel-1 SAR ship detection dataset; used for SAR ship classification benchmarks
- **AIR-SARShip (Sun et al., 2020)** — high-resolution SAR ship detection dataset from GaoFen-3; used for SAR object detection

**Decision criteria:** MSTAR is the undisputed gold standard for SAR ATR benchmarking (1998–2026). SEN1-2 for SAR-to-optical translation and despeckling. SpaceNet SAR for building extraction. For image formation algorithm validation, use TerraSAR-X raw data with known phase history. Use the dataset that appears in the largest number of SAR reconstruction/recognition papers.

#### Step 2: List All SAR Algorithms

Please first ensure all the SAR algorithms have been listed in `\pwm\public\algorithm_base\sar\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SAR solvers, please update the SAR solver.

**Key algorithms to cover (1950–2026):**

_Image Formation / Classical (1970s–2009):_
- Range-Doppler Algorithm — RDA, the canonical SAR image formation method (Wu et al., 1976; Cumming & Wong, 2005); azimuth FFT-based processing with range cell migration correction (RCMC)
- Omega-K Algorithm (Stolt migration) — wavenumber domain SAR focusing (Cafforio et al., IEEE TGRS 1991); exact 2D transfer function; handles range-dependent migration without interpolation
- Chirp Scaling Algorithm — CSA (Raney et al., IEEE TGRS 1994); avoids interpolation-based RCMC by phase multiplication; efficient for stripmap SAR
- Polar Format Algorithm — PFA (Walker, IEEE TAES 1980); maps phase history to polar grid then interpolates to Cartesian; standard for spotlight SAR
- Back-Projection Algorithm — BPA / time-domain correlation (Ulander et al., 2003); exact but computationally expensive; used for ultra-wideband and bistatic SAR
- Extended Chirp Scaling — ECS (Moreira et al., IEEE TGRS 1996); handles squinted acquisition geometries
- SPECAN — Spectral Analysis algorithm for ScanSAR (Scanlan, 1989)

_Autofocus & Phase Correction (1988–2015):_
- Phase Gradient Autofocus — PGA (Wahl et al., IEEE TAES 1994); iterative dominant-scatterer phase estimation; the standard SAR autofocus technique
- Mapdrift Autofocus (Calloway, 1988) — subaperture registration-based phase estimation
- Phase Difference Autofocus (Jakowatz et al., 1996)
- Minimum Entropy Autofocus (Xi et al., IEEE TGRS 1999) — optimizes image sharpness metric
- FAST-PGA (Ash, 2012) — accelerated phase gradient autofocus

_Speckle Filtering (1980s–2015):_
- Lee Filter (Lee, IEEE TPAMI 1980) — local statistics adaptive filter; foundational SAR speckle filter
- Frost Filter (Frost et al., IEEE TPAMI 1982) — exponentially weighted adaptive filter
- Kuan Filter (Kuan et al., IEEE TASSP 1985) — minimum mean square error filter
- Gamma-MAP Filter (Lopes et al., 1993) — MAP estimation under multiplicative speckle model
- Enhanced Lee / Enhanced Frost (Lopes et al., 1990) — improved edge-preserving variants
- IDAN — Intensity-Driven Adaptive-Neighborhood speckle filter (Vasile et al., IEEE TGRS 2006)
- NL-SAR — Non-Local SAR despeckling (Deledalle et al., IEEE TGRS 2015); patch-based non-local means adapted for SAR
- SAR-BM3D — block matching 3D for SAR speckle (Parrilli et al., IEEE TGRS 2012)
- PPB — Probabilistic Patch-Based SAR despeckling (Deledalle et al., 2009)

_Compressed Sensing SAR (2010–2016):_
- CS-SAR — Compressed Sensing SAR imaging (Patel et al., ICIP 2010; Ender, 2010); sparse signal recovery from undersampled phase history
- Sparse SAR autofocusing (Kelly et al., IEEE TAES 2014)
- TV-regularized SAR imaging (Cetin & Karl, IEEE TIP 2001)
- Bayesian CS-SAR (Potter et al., 2010)

_Deep Learning (2018–2026):_
- SAR-CNN despeckling (Chierchia et al., IEEE GRSL 2017) — first CNN for SAR speckle reduction
- SAR-DRN — Deep Residual Network for SAR despeckling (Zhang et al., 2018)
- Deep learning SAR autofocus (Pu et al., 2021) — CNN-based phase error estimation
- SAR2SAR — self-supervised despeckling (Dalsasso et al., IEEE TGRS 2021); no clean-image supervision
- MERLIN — Multi-temporal self-supervised SAR despeckling (Dalsasso et al., IEEE TGRS 2022)
- Speckle2Void (Molini et al., IEEE TGRS 2022) — blind-spot self-supervised SAR despeckling
- DL-based SAR image formation / learned imaging (Mason et al., 2017)
- Complex-valued CNN for SAR (Zhang et al., 2020)
- Transformer-based SAR despeckling (Perera et al., 2023)
- Diffusion-model SAR despeckling (Perera et al., 2024)
- Foundation model for SAR understanding (2025)

#### Step 3: Update SAR Solvers

After listing all SAR solvers, update `algorithm_base/sar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SAR solvers use the data format: `y` (num_range_bins, num_azimuth_samples) raw phase history or complex SAR image data, `platform_params` dict containing pulse parameters, geometry, and PRF. The `SAROperator` handles forward (phase history simulation) and adjoint (matched-filter image formation) operations. For despeckling: `y` (H, W) single-look complex or detected SAR image, output is despeckled image.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SAR:**
- MSTAR SOC (Standard Operating Conditions) 10-class ATR: SVM ~92%, CNN ~97%, complex-valued CNN ~99% recognition accuracy
- SEN1-2 despeckling: Lee filter ~24.0 dB, SAR-BM3D ~28.5 dB, SAR2SAR ~30.0 dB, Speckle2Void ~29.5 dB
- TerraSAR-X autofocus: PGA residual phase error <0.1 rad RMS
- Image formation: impulse response resolution within 1.2x theoretical limit; PSLR <-13 dB, ISLR <-10 dB
- Published PSNR/SSIM/ENL from the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SAR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sar/standard/`

---

### PolSAR (`polsar`) Modality Template

#### Step 1: Verify Standard Dataset

For PolSAR, what dataset do you use to verify? Is this dataset used for PolSAR popular algorithms? Please ensure the standard dataset in `datasets/benchmark/polsar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original PolSAR standard dataset.

**Popular datasets to consider:**
- **AIRSAR San Francisco (NASA/JPL)** — the most widely used PolSAR classification benchmark; L-band fully polarimetric data over San Francisco Bay with urban, ocean, and vegetation classes; used by virtually all PolSAR classification papers since 2000
- **UAVSAR Datasets (NASA/JPL)** — L-band fully polarimetric data from multiple campaigns; high-resolution repeat-pass; used for PolSAR decomposition and change detection
- **RADARSAT-2 Fine Quad-Pol Datasets** — C-band fully polarimetric data; Flevoland (Netherlands) and San Francisco scenes widely used for classification benchmarks
- **ALOS-2 PALSAR-2 Quad-Pol Data (JAXA)** — L-band fully polarimetric wide-swath data; used for forest/agriculture monitoring and PolSAR algorithm validation
- **PolSARpro Sample Datasets (ESA)** — curated PolSAR datasets distributed with PolSARpro software; ESAR, EMISAR, and RADARSAT-2 scenes with ground truth; standard for PolSAR education and algorithm development
- **Flevoland AIRSAR Dataset** — fully polarimetric L-band data with 15 crop classes; the canonical PolSAR classification test scene
- **E-SAR/F-SAR Oberpfaffenhofen (DLR)** — multi-frequency fully polarimetric data with detailed ground truth

**Decision criteria:** AIRSAR San Francisco and Flevoland are the gold standards for PolSAR classification benchmarking (2000–2026). PolSARpro sample datasets for decomposition validation. RADARSAT-2 Flevoland for C-band benchmarks. Use the dataset that appears in the largest number of PolSAR decomposition and classification papers.

#### Step 2: List All PolSAR Algorithms

Please first ensure all the PolSAR algorithms have been listed in `\pwm\public\algorithm_base\polsar\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/polsar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the PolSAR solvers, please update the PolSAR solver.

**Key algorithms to cover (1950–2026):**

_Polarimetric Decomposition / Classical (1985–2009):_
- Cloude-Pottier H/A/alpha Decomposition (Cloude & Pottier, IEEE TGRS 1997) — eigenvalue-based decomposition of coherency matrix into entropy (H), anisotropy (A), and mean alpha angle; the foundational unsupervised PolSAR analysis tool
- Freeman-Durden 3-Component Decomposition (Freeman & Durden, IEEE TGRS 1998) — model-based decomposition into surface, double-bounce, and volume scattering components; the most widely cited PolSAR decomposition
- Yamaguchi 4-Component Decomposition (Yamaguchi et al., IEEE TGRS 2005) — extends Freeman-Durden with helix scattering component for urban areas
- Pauli Decomposition — coherent decomposition into single-bounce, double-bounce, and 45-degree components (Pauli basis)
- Krogager Decomposition (Krogager, 1990) — sphere-diplane-helix coherent decomposition
- Cameron Decomposition (Cameron et al., 1996) — coherent target decomposition
- Huynen Decomposition (Huynen, 1970) — target fork decomposition
- van Zyl Decomposition (van Zyl, 1992) — model-based 3-component with non-negative constraint
- Arii 3-Component Decomposition (Arii et al., IEEE TGRS 2010) — volume model refinement

_Speckle Filtering for PolSAR (1998–2015):_
- Polarimetric Lee Filter (Lee et al., IEEE TGRS 1999) — adapted Lee filter using full polarimetric covariance matrix
- IDAN for PolSAR — Intensity-Driven Adaptive-Neighborhood (Vasile et al., IEEE TGRS 2006); data-driven region growing for PolSAR speckle
- Refined Lee Filter (Lee et al., 2006) — edge-aligned directional filter
- NL-SAR for PolSAR (Deledalle et al., IEEE TGRS 2015) — non-local means extended to polarimetric covariance matrices
- Pretest NL-InSAR/PolSAR (Deledalle et al., 2011)
- SARBM3D-Pol (Deledalle et al., 2014) — block matching 3D for PolSAR

_Classification (1999–2016):_
- Wishart Classification (Lee et al., IEEE TGRS 1999) — ML classification using complex Wishart distribution of covariance matrix; standard supervised PolSAR classifier
- H/alpha-Wishart Unsupervised Classification (Lee et al., IEEE TGRS 1999) — combines Cloude-Pottier decomposition with Wishart clustering; the canonical unsupervised PolSAR method
- Freeman-Wishart Classification (2004) — decomposition-guided Wishart classification
- SVM for PolSAR (Fukuda & Hirosawa, 2001) — support vector machine with polarimetric features
- Random Forest for PolSAR (2010)
- CRF/MRF spatial regularization for PolSAR classification (2012)
- Sparse representation-based PolSAR classification (2014)

_Deep Learning (2019–2026):_
- Complex-valued CNN for PolSAR (Zhang et al., IEEE TGRS 2017) — first deep learning approach preserving complex-valued polarimetric data
- CV-CNN — Complex-Valued CNN for PolSAR classification (Zhang et al., 2017)
- PolSAR-CNN (Zhou et al., 2018) — real-valued CNN on polarimetric features
- Graph Neural Network for PolSAR (Ren et al., 2021)
- Self-supervised PolSAR representation learning (2022)
- Transformer for PolSAR classification (Dong et al., IEEE TGRS 2023)
- Contrastive learning for PolSAR (2023)
- DL PolSAR decomposition (Xiang et al., 2024)
- Foundation model for PolSAR understanding (2025)

#### Step 3: Update PolSAR Solvers

After listing all PolSAR solvers, update `algorithm_base/polsar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All PolSAR solvers use the data format: `y` (H, W, 3, 3) complex coherency/covariance matrix T3 or C3, or `y` (H, W, 4) scattering vector in lexicographic or Pauli basis. The `PolSAROperator` handles coherency/covariance matrix formation, decomposition, and classification operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for PolSAR:**
- AIRSAR San Francisco classification: Wishart ~85% OA, SVM ~90% OA, CNN ~95% OA, Transformer ~97% OA
- AIRSAR Flevoland 15-class: H/alpha-Wishart ~75% OA, Freeman-Wishart ~80% OA, CV-CNN ~96% OA
- Decomposition validation: Freeman-Durden power components within 5% of published values on PolSARpro reference scenes
- Speckle filtering: ENL improvement >5x over original, edge preservation index >0.7
- Published OA/Kappa from the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'polsar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/polsar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/polsar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/polsar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for PolSAR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/polsar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/polsar/standard/`

---

### InSAR (`insar`) Modality Template

#### Step 1: Verify Standard Dataset

For InSAR, what dataset do you use to verify? Is this dataset used for InSAR popular algorithms? Please ensure the standard dataset in `datasets/benchmark/insar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original InSAR standard dataset.

**Popular datasets to consider:**
- **Sentinel-1 InSAR Pairs (ESA Copernicus)** — the most widely used free InSAR data source; C-band IW mode SLC pairs; global coverage; used by virtually all modern InSAR processing papers since 2015
- **ALOS/ALOS-2 InSAR Pairs (JAXA)** — L-band InSAR data with superior coherence in vegetated areas; used for deformation monitoring and phase unwrapping validation
- **LiCSAR Products (COMET, Lazecky et al., 2020)** — pre-processed Sentinel-1 interferograms and coherence maps; the primary community resource for large-scale InSAR time-series analysis
- **COMET InSAR Data (Centre for Observation and Modelling of Earthquakes, Volcanoes and Tectonics)** — curated InSAR datasets for tectonic and volcanic deformation studies with GPS validation
- **UAVSAR Repeat-Pass InSAR (NASA/JPL)** — L-band airborne InSAR with precisely controlled baselines; used for algorithm development and validation
- **TOPS Sentinel-1 InSAR Benchmarks (ESA)** — standardized test datasets for TOPS interferometric processing
- **Simulated InSAR Data with Known Deformation** — synthetic interferograms with known ground truth for phase unwrapping algorithm validation (e.g., Zebker & Lu, JOSA A 1998)

**Decision criteria:** Sentinel-1 is the undisputed standard for modern InSAR processing (2015–2026). UAVSAR for high-coherence algorithm validation. Simulated data with known deformation field for quantitative phase unwrapping benchmarks. Use the dataset that appears in the largest number of InSAR processing papers.

#### Step 2: List All InSAR Algorithms

Please first ensure all the InSAR algorithms have been listed in `\pwm\public\algorithm_base\insar\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/insar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the InSAR solvers, please update the InSAR solver.

**Key algorithms to cover (1950–2026):**

_Classical InSAR Processing (1988–2005):_
- Two-Pass DInSAR — Differential InSAR using external DEM (Gabriel et al., JGR 1989) — the foundational InSAR deformation measurement technique
- Three-Pass DInSAR — uses three SAR acquisitions to remove topographic phase (Zebker & Rosen, JGR 1994)
- InSAR coregistration — sub-pixel cross-correlation and spectral diversity methods
- Flat-earth phase removal and topographic phase subtraction
- Goldstein phase filter (Goldstein & Werner, GRL 1998) — adaptive spectral filter for interferogram noise reduction; standard InSAR filter
- Multilook processing for coherence estimation

_Phase Unwrapping (1988–2015):_
- Branch-cut phase unwrapping (Goldstein et al., Radio Science 1988) — residue identification and branch-cut connection; the classical method
- SNAPHU — Statistical-cost Network-flow Algorithm for Phase Unwrapping (Chen & Zebker, JOSA A 2001) — maximum a posteriori estimation via network flow; the most widely used 2D phase unwrapping software
- Minimum Cost Flow (MCF) phase unwrapping (Costantini, IEEE TGRS 1998) — integer programming formulation
- Least-squares phase unwrapping (Ghiglia & Romero, 1994)
- Region-growing phase unwrapping (Xu & Cumming, 1999)
- Multi-baseline phase unwrapping (Ferretti et al., 2001)

_Time-Series InSAR (2001–2016):_
- PSInSAR — Permanent/Persistent Scatterer InSAR (Ferretti et al., IEEE TGRS 2001) — the breakthrough time-series InSAR technique; identifies stable scatterers for mm-level deformation monitoring
- SBAS — Small Baseline Subset (Berardino et al., IEEE TGRS 2002) — uses short-baseline interferograms to maximize spatial coherence; SVD inversion for deformation time-series
- StaMPS — Stanford Method for Persistent Scatterers (Hooper et al., JGR 2004) — amplitude dispersion and spatial correlation for PS selection
- SqueeSAR (Ferretti et al., IEEE TGRS 2011) — statistically homogeneous pixel selection for distributed scatterers
- CAESAR — Component extraction from InSAR time-series (Ebmeier, JGR 2016)
- NSBAS — New SBAS (Lopez-Quiroz et al., 2009) — improved SBAS with temporal constraints
- MInTS — Multiscale InSAR Time Series (Hetland et al., JGR 2012) — wavelet-based temporal decomposition
- ISCE InSAR processing framework (Rosen et al., 2012)

_Deep Learning (2020–2026):_
- PhaseNet — deep learning phase unwrapping (Spoorthi et al., IEEE SPL 2019; Wu et al., IEEE TGRS 2021) — CNN for phase unwrapping classification
- DL-InSAR deformation estimation (Anantrasirichai et al., JGR 2021) — CNN for volcanic deformation detection from wrapped interferograms
- Deep learning InSAR noise filtering (Sica et al., IEEE TGRS 2021)
- U-Net phase unwrapping (Zhou et al., 2020) — semantic segmentation approach to phase unwrapping
- InSAR-Transformer for deformation time-series (2023)
- Self-supervised InSAR phase unwrapping (2023)
- Diffusion-model InSAR denoising (2024)
- Foundation model for InSAR analysis (2025)
- DL atmospheric phase screen removal (2022)
- Physics-informed neural network for InSAR inversion (2024)

#### Step 3: Update InSAR Solvers

After listing all InSAR solvers, update `algorithm_base/insar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All InSAR solvers use the data format: `y` (H, W) complex interferogram or (num_ifg, H, W) interferogram stack, `coherence` (H, W) coherence map, `baseline_info` dict containing perpendicular baselines and temporal baselines. The `InSAROperator` handles interferogram formation, phase simulation, and deformation model operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for InSAR:**
- Simulated interferogram phase unwrapping: Goldstein branch-cut ~85% correct pixels, MCF ~92%, SNAPHU ~97%, DL unwrapping ~98% on moderate-noise scenarios
- Sentinel-1 deformation: PSInSAR velocity accuracy ~1 mm/yr (validated against GPS), SBAS ~2 mm/yr
- LiCSAR products: interferogram coherence >0.3 for valid pixels; deformation correlation >0.95 with GPS
- Published RMSE/correlation from the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'insar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/insar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/insar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/insar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for InSAR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/insar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/insar/standard/`

---

### GPR (`gpr`) Modality Template

#### Step 1: Verify Standard Dataset

For GPR, what dataset do you use to verify? Is this dataset used for GPR popular algorithms? Please ensure the standard dataset in `datasets/benchmark/gpr/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original GPR standard dataset.

**Popular datasets to consider:**
- **GSSI Benchmark B-Scan Data (Geophysical Survey Systems Inc.)** — standard GPR B-scan profiles from controlled test sites with known buried targets; used for migration and detection algorithm validation
- **Synthetic B-Scan Datasets (gprMax, Warren et al., 2016)** — FDTD-generated synthetic GPR data with known ground truth; gprMax is the most widely used open-source GPR simulation tool; standard for algorithm development
- **SHRP2 Highway Subsurface GPR Data (FHWA)** — large-scale GPR surveys of highway pavements; used for pavement condition assessment and rebar detection benchmarks
- **DIGISOIL GPR Dataset (2010)** — multi-frequency GPR data for soil characterization with ground truth
- **Open GPR Datasets (Ishitsuka et al., 2018)** — curated GPR datasets for utility detection with annotated hyperbola locations
- **GprMax Simulation Benchmarks** — standardized simulation scenarios for validating forward modeling and migration algorithms
- **COST Action TU1208 GPR Datasets (2013–2017)** — European GPR benchmark datasets for civil engineering applications

**Decision criteria:** gprMax synthetic data is the standard for quantitative algorithm validation (known ground truth). GSSI real-data benchmarks for practical validation. SHRP2 for pavement GPR. Use the dataset that appears in the largest number of GPR imaging/detection papers (2000–2026).

#### Step 2: List All GPR Algorithms

Please first ensure all the GPR algorithms have been listed in `\pwm\public\algorithm_base\gpr\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/gpr. Besides, you need to search all algorithms from 1950 to 2026. After listing all the GPR solvers, please update the GPR solver.

**Key algorithms to cover (1950–2026):**

_Forward Modeling (1966–2010):_
- FDTD Forward Modeling — Finite-Difference Time-Domain simulation of GPR wave propagation (Yee, 1966; applied to GPR: Bergmann et al., 1998); the standard GPR forward model
- gprMax FDTD Simulator (Warren et al., Computer Physics Communications 2016) — the most widely used open-source GPR FDTD tool
- Ray-tracing GPR forward model (Goodman, 1994)
- Born approximation forward model for GPR (2005)

_Migration / Imaging (1980s–2015):_
- Kirchhoff Migration — diffraction summation migration for GPR (Stolt, 1978; applied to GPR: Fisher et al., 1992); time-domain summation along diffraction hyperbolas; the most commonly used GPR migration method
- FK Migration (Stolt Migration for GPR) — frequency-wavenumber domain migration (Stolt, Geophysics 1978; applied to GPR: Bitri & Grandjean, 1998); efficient Fourier-domain focusing for constant-velocity media
- Reverse Time Migration for GPR — RTM (Baysal et al., 1983; applied to GPR: Bradford et al., 2006); full-wavefield migration using time-reversed wave propagation; handles complex velocity models
- Phase-Shift Migration (Gazdag, 1978; applied to GPR: Bitri et al., 1998)
- Back-Propagation Migration for GPR (Mast & Johansson, 1994)
- Pre-stack Depth Migration for GPR (2005)

_Detection & Estimation (1990s–2016):_
- Hyperbola Fitting — parametric detection of buried objects by fitting diffraction hyperbolae in B-scans (Shihab & Al-Nuaimy, 2005); standard target localization technique
- CFAR Detection for GPR (1998) — constant false alarm rate detector for GPR target detection
- Hidden Markov Model for GPR (Gader et al., 2001)
- Template Matching for GPR hyperbola detection (Al-Nuaimy et al., 2000)
- Background Subtraction (mean/median trace removal) — standard GPR clutter reduction
- Velocity analysis (semblance, NMO) for GPR (2000)
- Time-frequency analysis for GPR (2005)

_Deep Learning (2019–2026):_
- CNN for GPR hyperbola detection (Besaw & Stimac, 2015; Lameri et al., 2017)
- YOLO-GPR — real-time object detection adapted for B-scans (Pham & Lefevre, 2018)
- U-Net for GPR migration (Wang et al., 2020) — learned imaging replacing classical migration
- DL-based GPR velocity estimation (Feng et al., 2021)
- Physics-informed neural network for GPR inversion (Li et al., 2022)
- GANs for GPR data augmentation and simulation (Lei et al., 2019)
- Transformer-based GPR interpretation (2023)
- Self-supervised GPR feature learning (2023)
- Diffusion-model GPR denoising and super-resolution (2024)
- Foundation model for subsurface imaging (2025)

#### Step 3: Update GPR Solvers

After listing all GPR solvers, update `algorithm_base/gpr/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All GPR solvers use the data format: `y` (num_traces, num_time_samples) B-scan radargram data, `scan_params` dict containing antenna positions, time window, and sampling interval. The `GPROperator` handles forward modeling (FDTD/Born), migration (adjoint), and velocity model operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for GPR:**
- gprMax synthetic point targets: Kirchhoff migration lateral resolution within 1.5x theoretical, FK migration within 1.2x theoretical
- SHRP2 rebar detection: template matching ~80% detection rate, CNN ~92%, YOLO ~95% at <5% false alarm
- Synthetic B-scan with known objects: migration position error <lambda/4, velocity estimation error <5%
- DL migration: SSIM >0.9 vs. RTM reference on synthetic data
- Published detection rate/RMSE from the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'gpr' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/gpr/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/gpr/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/gpr/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for GPR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/gpr/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/gpr/standard/`

---

### Hyperspectral Remote Sensing (`hyperspectral_remote`) Modality Template

#### Step 1: Verify Standard Dataset

For Hyperspectral Remote Sensing, what dataset do you use to verify? Is this dataset used for hyperspectral remote sensing popular algorithms? Please ensure the standard dataset in `datasets/benchmark/hyperspectral_remote/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original hyperspectral remote sensing standard dataset.

**Popular datasets to consider:**
- **Indian Pines (AVIRIS, Purdue University, 1992)** — the most widely used hyperspectral classification benchmark; 145x145 pixels, 220 bands, 16 vegetation/agriculture classes; used by virtually all hyperspectral classification papers since 1992
- **Pavia University (ROSIS sensor, 2002)** — 610x340 pixels, 103 bands, 9 urban classes; the second most popular hyperspectral benchmark; widely used for urban classification
- **Salinas Valley (AVIRIS, Purdue)** — 512x217 pixels, 224 bands, 16 crop classes; standard benchmark for agricultural hyperspectral classification
- **Houston 2013 (IEEE GRSS Data Fusion Contest)** — 349x1905 pixels, 144 bands, 15 classes including urban and vegetation; used for the 2013 GRSS DFC
- **Houston 2018 (IEEE GRSS Data Fusion Contest)** — larger coverage with LiDAR fusion; 20 classes; used for the 2018 GRSS DFC
- **Chikusei (Yokoya & Iwasaki, 2016)** — 2517x2335 pixels, 128 bands, 19 classes; large-area hyperspectral benchmark from Chikusei, Japan
- **EnMAP Data (DLR, 2022–)** — operational spaceborne hyperspectral imagery; 30m resolution, 244 bands; emerging benchmark for next-generation hyperspectral algorithms
- **ICVL (Arad & Ben-Shahar, 2016)** — hyperspectral image dataset for spectral reconstruction from RGB
- **CAVE (Columbia, Yasuma et al., 2010)** — indoor hyperspectral scenes for spectral imaging research
- **Washington DC Mall (HYDICE)** — 1208x307 pixels, 191 bands; urban hyperspectral scene

**Decision criteria:** Indian Pines is the undisputed gold standard for hyperspectral classification benchmarking; Pavia University is the second most cited. Salinas for crop classification. Houston 2013/2018 for data fusion challenges. Use the dataset that appears in the largest number of hyperspectral classification and processing papers (1992–2026).

#### Step 2: List All Hyperspectral Remote Sensing Algorithms

Please first ensure all the hyperspectral remote sensing algorithms have been listed in `\pwm\public\algorithm_base\hyperspectral_remote\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/hyperspectral_remote. Besides, you need to search all algorithms from 1950 to 2026. After listing all the hyperspectral remote sensing solvers, please update the hyperspectral remote sensing solver.

**Key algorithms to cover (1950–2026):**

_Dimensionality Reduction & Feature Extraction (1988–2010):_
- PCA — Principal Component Analysis for hyperspectral data (Jolliffe, 1986; applied to HSI: Richards & Jia, 1999); linear decorrelation and dimensionality reduction; baseline for all HSI analysis
- MNF — Minimum Noise Fraction transform (Green et al., IEEE TGRS 1988); noise-adjusted PCA that maximizes SNR; the standard HSI dimensionality reduction method
- ICA — Independent Component Analysis for hyperspectral unmixing (Nascimento & Bioucas-Dias, TGRS 2005)
- Kernel PCA for nonlinear HSI feature extraction (Scholkopf et al., 1998; applied 2005)
- LDA — Linear Discriminant Analysis for HSI (Bandos et al., IEEE TGRS 2009)

_Spectral Unmixing (1990s–2016):_
- VCA — Vertex Component Analysis (Nascimento & Bioucas-Dias, IEEE TGRS 2005) — pure pixel endmember extraction; the most widely used endmember extraction algorithm
- SUnSAL — Sparse Unmixing by variable Splitting and Augmented Lagrangian (Iordache et al., IEEE TGRS 2011) — sparse regression-based spectral unmixing using spectral library
- N-FINDR endmember extraction (Winter, 1999) — maximum volume simplex endmember identification
- FCLS — Fully Constrained Least Squares abundance estimation (Heinz & Chang, 2001)
- MESMA — Multiple Endmember Spectral Mixture Analysis (Roberts et al., 1998)
- NMF — Non-negative Matrix Factorization for unmixing (2006)
- Bayesian unmixing (Dobigeon et al., IEEE TSP 2009)

_Classification (2001–2016):_
- SVM for Hyperspectral (Melgani & Bruzzone, IEEE TGRS 2004) — the dominant classical HSI classifier; kernel SVM with spectral features
- Random Forest for HSI (Ham et al., IEEE TGRS 2005)
- Morphological Profiles — Extended Morphological Profiles for spatial-spectral classification (Benediktsson et al., IEEE TGRS 2005)
- Composite Kernel SVM (Camps-Valls et al., IEEE TGRS 2006) — joint spectral-spatial kernel
- Sparse Representation Classifier for HSI (Chen et al., IEEE TGRS 2011)
- Gabor features for HSI (Jia et al., 2015)
- Markov Random Field spatial regularization (Tarabalka et al., 2010)

_Denoising (2005–2016):_
- BM4D — Block Matching 4D for hyperspectral denoising (Maggioni et al., IEEE TIP 2013) — extends BM3D to volumetric data; the standard non-DL HSI denoiser
- LRMR — Low-Rank Matrix Recovery for HSI denoising (Zhang et al., IEEE TGRS 2014)
- TV-regularized HSI denoising (Yuan et al., 2012)
- Wavelet-based HSI denoising (2008)
- PARAFAC tensor decomposition for HSI (2014)

_Deep Learning (2017–2026):_
- 3D-CNN for HSI Classification (Li et al., Remote Sensing 2017) — the first 3D convolutional approach exploiting joint spectral-spatial features; widely cited and replicated
- HybridSN — Hybrid Spectral-Spatial 3D-2D CNN (Roy et al., IEEE GRSL 2020) — combines 3D and 2D convolutions; popular benchmark model
- SSRN — Spectral-Spatial Residual Network (Zhong et al., IEEE TGRS 2018)
- SpectralFormer (Hong et al., IEEE TGRS 2022) — the first Vision Transformer for hyperspectral classification; cross-layer adaptive fusion of spectral tokens
- MaskSST — Masked Spectral-Spatial Transformer (Li et al., IEEE TGRS 2024) — self-supervised pre-training with spectral-spatial masking; state-of-the-art HSI classification
- DBDA — Dual-Branch Dual-Attention CNN (Li et al., 2020)
- A2S2K-ResNet — Adaptive Spectral-Spatial Kernel (Roy et al., IEEE JSTARS 2021)
- HSI-BERT — self-supervised spectral representation learning (2022)
- GAN-based HSI super-resolution (2020)
- Deep unfolding for HSI denoising (ADMM-Net, 2020)
- DL spectral unmixing (EndNet, Ozkan et al., IEEE TGRS 2019)
- AutoML for HSI classification (2021)
- Diffusion-model HSI generation and restoration (2024)
- Foundation model for hyperspectral understanding (SpectralGPT, 2024)

#### Step 3: Update Hyperspectral Remote Sensing Solvers

After listing all hyperspectral remote sensing solvers, update `algorithm_base/hyperspectral_remote/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All hyperspectral remote sensing solvers use the data format: `y` (H, W, B) hyperspectral image cube with B spectral bands, `wavelengths` array of band center wavelengths. The `HyperspectralRemoteOperator` handles spectral unmixing forward model (endmember mixing), dimensionality reduction, and classification operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Hyperspectral Remote Sensing:**
- Indian Pines classification: SVM ~85% OA, 3D-CNN ~92% OA, HybridSN ~95% OA, SpectralFormer ~96% OA, MaskSST ~97.5% OA (with standard train/test split)
- Pavia University: SVM ~90% OA, 3D-CNN ~95% OA, SpectralFormer ~97% OA
- Salinas: SVM ~92% OA, HybridSN ~97% OA
- Denoising (simulated Gaussian + stripe noise): BM4D ~32 dB, LRMR ~34 dB, DL denoising ~37 dB
- Unmixing (synthetic mixtures): VCA endmember SAD <5 degrees, SUnSAL abundance RMSE <0.05
- Published OA/AA/Kappa from the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'hyperspectral_remote' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/hyperspectral_remote/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for hyperspectral remote sensing. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/hyperspectral_remote/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/hyperspectral_remote/standard/`

---

### Multispectral Satellite (`multispectral_sat`) Modality Template

#### Step 1: Verify Standard Dataset

For Multispectral Satellite, what dataset do you use to verify? Is this dataset used for multispectral satellite popular algorithms? Please ensure the standard dataset in `datasets/benchmark/multispectral_sat/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original multispectral satellite standard dataset.

**Popular datasets to consider:**
- **SpaceNet (CosmiQ Works / Maxar, 2017–2020)** — the most widely used satellite imagery benchmark series; multi-sensor (WorldView, GeoEye) multispectral + PAN imagery with building footprint, road, and flood labels; SpaceNet 1–7 covering multiple cities worldwide
- **UC Merced Land Use Dataset (Yang & Newsam, 2010)** — 2100 aerial images, 256x256 pixels, 21 land-use classes; the foundational remote sensing scene classification benchmark
- **DOTA — Dataset for Object Detection in Aerial Images (Xia et al., CVPR 2018)** — large-scale oriented object detection benchmark; 2806 images with 188,282 instances of 15 object categories; standard for satellite/aerial object detection
- **EuroSAT (Helber et al., IEEE JSTARS 2019)** — 27,000 Sentinel-2 image patches, 10 LULC classes, 13 spectral bands; widely used for land cover classification
- **BigEarthNet (Sumbul et al., IEEE TGRS 2019)** — 590,326 Sentinel-2 patches with multi-label land cover annotations; the largest annotated multispectral satellite dataset; used for multi-label classification benchmarks
- **GaoFen Image Dataset (Cheng et al., 2017)** — high-resolution Chinese satellite multispectral imagery for scene classification and object detection
- **RESISC-45 (Cheng et al., 2017)** — 31,500 images, 45 scene classes, 256x256 pixels; large-scale remote sensing scene classification
- **AID — Aerial Image Dataset (Xia et al., IEEE TGRS 2017)** — 10,000 images, 30 aerial scene classes
- **ReducedWorldView (Loncan et al., 2015)** — standard pansharpening benchmark with WorldView-2 8-band imagery
- **GaoFen-2 Pansharpening Benchmark** — Chinese high-resolution satellite pansharpening test data

**Decision criteria:** SpaceNet is the undisputed gold standard for satellite building/road extraction (2017–2026). EuroSAT/BigEarthNet for land cover classification. UC Merced/RESISC-45 for scene classification. Wald protocol datasets for pansharpening. Use the dataset that appears in the largest number of multispectral satellite processing papers.

#### Step 2: List All Multispectral Satellite Algorithms

Please first ensure all the multispectral satellite algorithms have been listed in `\pwm\public\algorithm_base\multispectral_sat\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/multispectral_sat. Besides, you need to search all algorithms from 1950 to 2026. After listing all the multispectral satellite solvers, please update the multispectral satellite solver.

**Key algorithms to cover (1950–2026):**

_Pansharpening (1991–2015):_
- PCA Pansharpening (Shettigara, 1992) — replaces first principal component of multispectral with high-resolution PAN; the simplest component substitution method
- IHS Pansharpening — Intensity-Hue-Saturation fusion (Carper et al., 1990) — replaces intensity component; the most widely used traditional pansharpening method
- Brovey Transform (Gillespie et al., 1987) — ratio-based spectral normalization with PAN injection
- GS — Gram-Schmidt pansharpening (Laben & Brower, 2000) — orthogonal component substitution; widely used in commercial software
- SFIM — Smoothing Filter-based Intensity Modulation (Liu, 2000) — multiplicative injection with low-pass PAN
- MTF-GLP — Generalized Laplacian Pyramid with Modulation Transfer Function (Aiazzi et al., IEEE TGRS 2006) — multi-resolution analysis matching sensor MTF; the gold standard traditional pansharpening method
- ATWT — A Trous Wavelet Transform pansharpening (Nunez et al., 1999)
- HPF — High-Pass Filtering pansharpening (Schowengerdt, 1980)
- Variational pansharpening (Ballester et al., IJCV 2006)

_Super-Resolution (2010–2016):_
- Sparse representation super-resolution for satellite (Yang et al., 2010; applied to RS 2014)
- Multi-frame satellite super-resolution (Farsiu et al., 2004; applied to RS)
- POCS super-resolution for satellite imagery (2008)
- Total variation super-resolution (2012)

_Classification (1990s–2016):_
- OBIA — Object-Based Image Analysis (Blaschke, 2010) — segmentation-then-classification paradigm; standard for high-resolution satellite imagery analysis
- Random Forest for satellite classification (Belgiu & Dragut, ISPRS 2016) — ensemble decision tree classifier; the most popular ML classifier for satellite LULC
- SVM for satellite classification (Mountrakis et al., ISPRS 2011) — kernel-based classifier widely used for multispectral land cover mapping
- Maximum Likelihood Classifier (Richards & Jia, 1999) — the classical statistical pixel-based classifier
- k-NN classifier for satellite imagery (2005)
- Boosting methods (AdaBoost, Gradient Boosting) for RS (2008)
- Spectral indices (NDVI, NDWI, NDBI) — band ratio features for thematic mapping

_Deep Learning (2016–2026):_
- CNN for satellite scene classification (Penatti et al., 2015; Castelluccio et al., 2015) — first deep learning for satellite scene understanding
- PNN — PanNet Pansharpening (Masi et al., Remote Sensing 2016) — first CNN for pansharpening
- MSDCNN — Multi-Scale and Detail CNN (Wei et al., 2017) — deep pansharpening with multi-scale features
- PanGAN — GAN-based pansharpening (Ma et al., 2020)
- ViT for Remote Sensing (Bazi et al., IEEE GRSL 2021) — Vision Transformer adapted for satellite image classification
- SatMAE — Masked Autoencoder for satellite (Cong et al., NeurIPS 2022) — self-supervised pre-training for multispectral imagery
- ResNet/DenseNet for satellite classification (2017–2019)
- U-Net for satellite segmentation (2017)
- DeepLab for satellite semantic segmentation (2018)
- DINO-MC — self-supervised multispectral features (2022)
- Pansharpening Transformer (2023)
- DiffPan — diffusion-based pansharpening (2024)
- Foundation model for satellite understanding (SatCLIP, GFM, 2024–2025)
- Segment Anything Model adapted for satellite (2024)

#### Step 3: Update Multispectral Satellite Solvers

After listing all multispectral satellite solvers, update `algorithm_base/multispectral_sat/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All multispectral satellite solvers use the data format: `y` (H, W, B) multispectral image with B bands, `pan` (H_pan, W_pan) panchromatic image for pansharpening tasks. The `MultispectralSatOperator` handles spectral degradation (MTF filtering), spatial degradation, pansharpening forward model, and classification operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Multispectral Satellite:**
- WorldView-2 pansharpening (Wald protocol reduced resolution): IHS ~28 dB, GS ~30 dB, MTF-GLP ~33 dB, PanNet ~35 dB, DiffPan ~37 dB (PSNR on MS bands)
- EuroSAT classification: Random Forest ~88% OA, ResNet-50 ~94% OA, ViT ~96% OA, SatMAE ~97% OA
- BigEarthNet multi-label: ResNet ~80% mAP, SatMAE ~85% mAP
- UC Merced scene classification: SVM ~85% OA, VGG-16 ~95% OA, ViT ~98% OA
- SpaceNet building extraction: U-Net IoU ~0.65, winning methods IoU ~0.75
- Published PSNR/SSIM/OA from the original papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'multispectral_sat' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/multispectral_sat/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for multispectral satellite. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/multispectral_sat/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/multispectral_sat/standard/`

---

### Ocean Color (`ocean_color`) Modality Template

#### Step 1: Verify Standard Dataset

For Ocean Color, what dataset do you use to verify? Is this dataset used for ocean color popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ocean_color/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ocean color standard dataset.

**Popular datasets to consider:**
- **NASA MODIS Ocean Color (NASA OBPG, 2002–present)** — the most widely used ocean color satellite data; Aqua/Terra MODIS Level 1B/2/3 products; global coverage; 36 bands (9 ocean color bands); used by virtually all ocean color algorithm papers since 2002
- **SeaWiFS (NASA, 1997–2010)** — the foundational ocean color mission; 8 bands optimized for ocean color; the standard reference for atmospheric correction and bio-optical algorithm validation
- **VIIRS Ocean Color (NOAA/NASA, 2012–present)** — operational ocean color from SNPP/NOAA-20; 22 bands; continuation of MODIS and SeaWiFS missions
- **Copernicus OLCI (ESA Sentinel-3, 2016–present)** — 21 bands with improved spectral coverage for ocean color; 300m resolution; the European operational ocean color sensor
- **MERIS (ESA Envisat, 2002–2012)** — 15 bands with programmable spectral configuration; used for European coastal and ocean color research; precursor to OLCI
- **IOCCG Simulated Dataset (IOCCG Report 5, Lee et al., 2006)** — synthetic ocean color reflectance dataset with known IOPs (inherent optical properties); the standard benchmark for bio-optical algorithm intercomparison
- **NOMAD (NASA bio-Optical Marine Algorithm Dataset, Werdell & Bailey, 2005)** — in situ bio-optical measurements matched with satellite data; the primary validation dataset for ocean color algorithms
- **SeaBASS (SeaWiFS Bio-optical Archive and Storage System)** — comprehensive archive of in situ oceanographic and atmospheric data for satellite calibration/validation
- **AERONET-OC (Zibordi et al., 2009)** — above-water radiometry network for ocean color satellite validation

**Decision criteria:** IOCCG simulated dataset is the gold standard for bio-optical algorithm intercomparison (known ground truth IOPs). NOMAD/SeaBASS for in situ validation. NASA MODIS for operational algorithm benchmarks. Use the dataset that appears in the largest number of ocean color retrieval papers (1997–2026).

#### Step 2: List All Ocean Color Algorithms

Please first ensure all the ocean color algorithms have been listed in `\pwm\public\algorithm_base\ocean_color\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ocean_color. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ocean color solvers, please update the ocean color solver.

**Key algorithms to cover (1950–2026):**

_Atmospheric Correction (1980s–2010):_
- Gordon & Wang Atmospheric Correction (Gordon & Wang, Applied Optics 1994) — the foundational satellite ocean color atmospheric correction algorithm; uses NIR bands to estimate aerosol radiance and remove atmospheric signal; used by all NASA ocean color processing since SeaWiFS
- MUMM Atmospheric Correction (Ruddick et al., 2000) — modified NIR-based atmospheric correction for turbid coastal waters
- SWIR-based Atmospheric Correction (Wang & Shi, 2007) — uses shortwave infrared bands for turbid water atmospheric correction
- Bayesian atmospheric correction (Fan et al., 2017)
- 6S Radiative Transfer Code (Vermote et al., 1997) — vector radiative transfer for atmospheric correction
- ACOLITE (Vanhellemont & Ruddick, 2018) — atmospheric correction for coastal and inland waters

_Bio-Optical Models & Retrieval (1977–2015):_
- Morel & Prieur Bio-Optical Model (Morel & Prieur, Limnology & Oceanography 1977) — the foundational classification of ocean waters (Case 1 and Case 2) based on optical properties; basis for all subsequent bio-optical models
- OC4v6 / OCx Band-Ratio Chlorophyll Algorithm (O'Reilly et al., JGR 1998) — the standard operational chlorophyll-a retrieval algorithm using blue-green band ratios; used by NASA for MODIS/SeaWiFS operational products
- GSM — Garver-Siegel-Maritorena Semi-Analytical Model (Maritorena et al., Applied Optics 2002) — semi-analytical retrieval of chlorophyll, CDOM, and particle backscattering from remote sensing reflectance
- QAA — Quasi-Analytical Algorithm (Lee et al., Applied Optics 2002) — physics-based inversion for inherent optical properties (absorption and backscattering) from Rrs; widely used for multi-parameter retrieval
- GIOP — Generalized IOP Algorithm (Werdell et al., Applied Optics 2013) — flexible framework combining multiple semi-analytical approaches for IOP retrieval
- Carder Model (Carder et al., 1999) — semi-analytical chlorophyll retrieval
- HOPE Model (Lee et al., 1999) — Hyperspectral Optimization Process Exemplar for IOP inversion
- Aiken Fluorescence Line Height (FLH) — chlorophyll fluorescence-based retrieval
- Nechad TSM Algorithm (Nechad et al., 2010) — single-band total suspended matter retrieval for turbid waters
- CDOM absorption retrieval (Mannino et al., 2008)

_Statistical & Machine Learning (2000–2016):_
- NN Ocean Color — Neural Network for ocean color retrieval (IOCCG Report 5, 2006; Doerffer & Schiller, 2007) — the standard ML approach for complex water retrieval; C2RCC processor for Sentinel-3
- Empirical band-ratio algorithms (nFLH, CI, MCI) for specific products (Hu et al., 2012)
- Color Index (CI) Algorithm (Hu et al., JGR 2012) — improved chlorophyll retrieval for oligotrophic waters
- Bayesian ocean color retrieval (2010)
- Random Forest for water quality (2014)
- Multivariate regression for ocean color (2008)

_Deep Learning (2020–2026):_
- DL Chlorophyll Retrieval (Pahlevan et al., Remote Sensing of Environment 2020) — multi-layer perceptron for chlorophyll-a retrieval; outperforms traditional band-ratio and semi-analytical methods
- CNN for ocean color atmospheric correction (Fan et al., 2021)
- Mixture Density Network for ocean color (Pahlevan et al., RSE 2022) — probabilistic DL retrieval with uncertainty estimation
- Physics-informed neural network for ocean color (2022)
- Transformer for ocean color time-series (2023)
- Transfer learning for inland water quality from ocean color (2021)
- Self-supervised ocean color representation learning (2023)
- Diffusion-model ocean color super-resolution (2024)
- Foundation model for aquatic remote sensing (2025)

#### Step 3: Update Ocean Color Solvers

After listing all ocean color solvers, update `algorithm_base/ocean_color/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ocean color solvers use the data format: `y` (H, W, B) top-of-atmosphere or remote sensing reflectance with B spectral bands, `wavelengths` array of band center wavelengths, `geometry` dict containing solar/viewing zenith and azimuth angles. The `OceanColorOperator` handles atmospheric correction (forward radiative transfer), bio-optical forward model, and IOP retrieval operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Ocean Color:**
- IOCCG synthetic dataset: OC4v6 chlorophyll RMSE ~30% for Case 1 waters, QAA absorption RMSE <15%, GSM 3-parameter RMSE <20%
- NOMAD matchups: OC4v6 chlorophyll R2 ~0.85, NN retrieval R2 ~0.90, DL retrieval R2 ~0.93
- Atmospheric correction: Rrs residual <5% in open ocean, <15% in coastal waters (Gordon & Wang standard)
- Published retrieval accuracy from IOCCG intercomparison reports and original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ocean_color' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ocean_color/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ocean_color/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ocean_color/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ocean color. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ocean_color/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_color/standard/`

---

### Passive Microwave (`passive_microwave`) Modality Template

#### Step 1: Verify Standard Dataset

For Passive Microwave, what dataset do you use to verify? Is this dataset used for passive microwave popular algorithms? Please ensure the standard dataset in `datasets/benchmark/passive_microwave/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original passive microwave standard dataset.

**Popular datasets to consider:**
- **AMSR-E / AMSR2 (JAXA, 2002–present)** — the most widely used passive microwave radiometer dataset; 6.9–89 GHz dual-polarization brightness temperatures; used for sea surface temperature, soil moisture, sea ice, precipitation, and water vapor retrieval; AMSR-E (Aqua, 2002–2011) and AMSR2 (GCOM-W1, 2012–present)
- **SMAP (NASA Soil Moisture Active Passive, 2015–present)** — L-band (1.4 GHz) microwave radiometer optimized for soil moisture; the gold standard for soil moisture retrieval algorithm benchmarks
- **SSM/I (Defense Meteorological Satellite Program, 1987–present)** — Special Sensor Microwave/Imager; 19–85 GHz; the foundational passive microwave dataset used for climate data records; decades of continuous observations
- **MIRS (Microwave Integrated Retrieval System, NOAA)** — operational retrieval products from multiple passive microwave sensors (ATMS, AMSU, MHS); used for temperature/moisture profiling and precipitation
- **GPM Microwave Imager (GMI, 2014–present)** — multi-frequency (10–183 GHz) conical-scanning radiometer; the primary passive microwave sensor for global precipitation measurement
- **SMOS (ESA, 2009–present)** — L-band interferometric radiometer for soil moisture and ocean salinity
- **ERA5 Reanalysis (ECMWF)** — used as reference/truth for passive microwave retrieval validation
- **In Situ Soil Moisture Networks (ISMN, Dorigo et al., 2011)** — ground-truth soil moisture for passive microwave retrieval validation

**Decision criteria:** AMSR-E/AMSR2 is the most widely used passive microwave dataset for multi-parameter retrieval. SMAP for soil moisture. SSM/I for climate data records. ISMN for soil moisture validation. Use the dataset that appears in the largest number of passive microwave retrieval papers (1987–2026).

#### Step 2: List All Passive Microwave Algorithms

Please first ensure all the passive microwave algorithms have been listed in `\pwm\public\algorithm_base\passive_microwave\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/passive_microwave. Besides, you need to search all algorithms from 1950 to 2026. After listing all the passive microwave solvers, please update the passive microwave solver.

**Key algorithms to cover (1950–2026):**

_Radiative Transfer & Forward Models (1960s–2010):_
- Radiative Transfer Inversion — microwave radiative transfer equation inversion for geophysical parameter retrieval; the foundational physics-based approach; includes surface emissivity models (Wegmuller & Matzler, 1999) and atmospheric absorption models (Rosenkranz, 1998)
- Community Microwave Emission Model — CMEM (de Rosnay et al., 2009; Holmes et al., 2009) — coupled land surface-atmosphere microwave emission model; used for SMOS/SMAP calibration and validation
- Microwave Land Emission Model — MLEM (Drusch et al., 2001)
- L-band Microwave Emission of the Biosphere — L-MEB (Wigneron et al., RSE 2007) — the standard L-band soil moisture forward model used by SMOS and SMAP
- Wentz Ocean Radiative Transfer Model (Wentz, 1997) — ocean surface emissivity and atmospheric transmission for SST retrieval
- Tau-Omega Model — vegetation optical depth model for soil moisture retrieval (Jackson & Schmugge, 1991)

_Statistical Regression (1987–2015):_
- Statistical Regression Retrieval — linear/nonlinear regression between brightness temperatures and geophysical parameters; widely used operational approach (Alishouse et al., 1990)
- Wentz SST Algorithm (Wentz & Meissner, 2000) — operational sea surface temperature retrieval from AMSR-E/AMSR2; statistical regression with physical constraints
- NASA Team Sea Ice Algorithm (Cavalieri et al., 1984) — the standard sea ice concentration retrieval using polarization/frequency ratios
- Bootstrap Sea Ice Algorithm (Comiso, 1986) — alternative sea ice concentration retrieval
- Kummerow Precipitation Algorithm (Kummerow et al., JAOT 1996) — physically-based precipitation retrieval from passive microwave (Goddard Profiling Algorithm)
- GPROF — Goddard Profiling Algorithm for precipitation (Kummerow et al., 2015) — operational GPM precipitation retrieval
- Land Parameter Retrieval Model — LPRM (Owe et al., JGR 2008) — soil moisture and vegetation optical depth retrieval from AMSR-E
- Delta-Index soil moisture (Koike et al., 2004)

_Bayesian & Optimal Estimation (2005–2016):_
- Bayesian Retrieval for passive microwave (Kummerow et al., 2011; Evans et al., 2012) — probabilistic retrieval using a priori information and Bayesian inversion framework
- Optimal Estimation / OE (Rodgers, 2000; applied to microwave: Boukabara et al., 2011) — maximum a posteriori retrieval with full error characterization; used in MIRS operational system
- 1DVAR — One-Dimensional Variational Retrieval (Garand et al., 2001; English, 2008) — variational assimilation framework for simultaneous temperature, humidity, and surface parameter retrieval from microwave radiances; standard in NWP data assimilation
- Ensemble-based retrieval (2014)
- Multi-sensor Bayesian fusion (2013)
- MCMC retrieval for uncertainty quantification (2015)

_Deep Learning (2020–2026):_
- DL Retrieval for passive microwave — deep neural network for geophysical parameter retrieval from brightness temperatures (Blackwell, 2005 early NN; modern DL 2020+); outperforms traditional regression for complex multi-parameter retrieval
- CNN for precipitation estimation from passive microwave (Sadeghi et al., RSE 2021)
- Random Forest / Gradient Boosting for soil moisture (2018; used as baseline for DL)
- LSTM for microwave time-series retrieval (2020)
- U-Net for passive microwave spatial downscaling (2021)
- Physics-constrained neural network for microwave retrieval (2022)
- Transfer learning across microwave sensors (2022)
- Transformer for multi-frequency microwave retrieval (2023)
- Self-supervised microwave representation learning (2024)
- Foundation model for passive microwave remote sensing (2025)

#### Step 3: Update Passive Microwave Solvers

After listing all passive microwave solvers, update `algorithm_base/passive_microwave/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All passive microwave solvers use the data format: `y` (H, W, C) brightness temperature array with C channels (frequency-polarization combinations), `frequency_ghz` array of channel frequencies, `polarization` array of polarization types (V/H). The `PassiveMicrowaveOperator` handles forward radiative transfer (emission model), Jacobian computation, and retrieval inversion operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Passive Microwave:**
- SMAP soil moisture: L-MEB retrieval ubRMSE ~0.04 m3/m3, statistical regression ~0.05 m3/m3, DL retrieval ~0.035 m3/m3 (validated against ISMN)
- AMSR2 SST: Wentz algorithm RMSE ~0.5 K, NN retrieval ~0.4 K (validated against buoys)
- Sea ice concentration: NASA Team RMSE ~5%, Bootstrap ~4%, DL ~3% (validated against ASI)
- GPM precipitation: GPROF correlation ~0.7, DL ~0.8 (validated against ground radar)
- Published retrieval accuracy from algorithm theoretical basis documents (ATBDs)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'passive_microwave' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/passive_microwave/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for passive microwave. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/passive_microwave/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/passive_microwave/standard/`

---

### Weather Radar (`weather_radar`) Modality Template

#### Step 1: Verify Standard Dataset

For Weather Radar, what dataset do you use to verify? Is this dataset used for weather radar popular algorithms? Please ensure the standard dataset in `datasets/benchmark/weather_radar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original weather radar standard dataset.

**Popular datasets to consider:**
- **NEXRAD Level-II (NOAA, 1988–present)** — the most widely used weather radar dataset; WSR-88D S-band dual-polarization Doppler radar; base reflectivity, velocity, spectrum width, Zdr, Kdp, CC; 160 radar sites across the US; the primary benchmark for all weather radar algorithms
- **DWD Radar Composites (Deutscher Wetterdienst)** — high-resolution C-band radar composites over Germany; 5-minute temporal resolution; used for European precipitation nowcasting research
- **OPERA (EUMETNET Operational Programme for the Exchange of Weather Radar Information)** — European-wide radar composite dataset; quality-controlled reflectivity and precipitation products; used for continental-scale nowcasting benchmarks
- **ERA5 Reanalysis (ECMWF)** — used as reference/truth for radar-based precipitation estimation and nowcasting verification
- **MRMS (Multi-Radar Multi-Sensor, NOAA)** — quality-controlled national radar mosaic; used as ground truth for precipitation algorithms
- **RYDL (Radar-based convective storm Dataset, DWD)** — curated convective storm events for deep learning benchmarks
- **SEVIR (Storm EVent ImagRy Dataset, Veillette et al., NeurIPS 2020)** — large-scale multi-sensor weather dataset including NEXRAD MESH and VIL; standard for deep learning weather nowcasting

**Decision criteria:** NEXRAD Level-II is the undisputed standard for weather radar algorithm development and validation (1988–2026). SEVIR for deep learning nowcasting benchmarks. DWD/OPERA for European research. Use the dataset that appears in the largest number of weather radar processing and nowcasting papers.

#### Step 2: List All Weather Radar Algorithms

Please first ensure all the weather radar algorithms have been listed in `\pwm\public\algorithm_base\weather_radar\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/weather_radar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the weather radar solvers, please update the weather radar solver.

**Key algorithms to cover (1950–2026):**

_Radar Meteorology Fundamentals (1948–1990):_
- Z-R Relationship — Marshall-Palmer reflectivity-rainfall relationship (Marshall & Palmer, Journal of Meteorology 1948) — Z = aR^b; the foundational radar precipitation estimation; Z=200R^1.6 standard; used operationally worldwide
- Doppler Processing — radial velocity estimation from pulse-pair or FFT spectral processing; mean velocity, spectrum width, and spectral moment estimation
- WSR-88D signal processing chain — pulse compression, clutter filtering, range dealiasing, velocity dealiasing
- VPR Correction — Vertical Profile of Reflectivity correction (Joss & Lee, 1995) — accounts for beam broadening and bright band effects

_Dual-Polarization (2000–2016):_
- Kdp-based Rainfall Estimation — specific differential phase for precipitation rate (Ryzhkov & Zrnic, JAOT 2005) — R(Kdp) relations; more robust than Z-R for heavy rain
- Zdr-based Rainfall — differential reflectivity for raindrop size discrimination (Seliga & Bringi, 1976)
- Hydrometeor Classification Algorithm — HCA (Park et al., JAOT 2009) — fuzzy-logic classifier using Zh, Zdr, Kdp, rhohv for hydrometeor type identification
- QPE using dual-pol — composite algorithms using R(Z), R(Kdp), R(Z,Zdr) (Ryzhkov et al., 2005)
- Attenuation correction using dual-pol (Bringi et al., 2001)
- Drop Size Distribution retrieval from dual-pol (Brandes et al., 2004)
- Self-consistency calibration (Gorgucci et al., 1999)

_Quantitative Precipitation Estimation / QPE (1990–2016):_
- Gauge-adjusted radar QPE (Seo et al., 1999) — bias correction using rain gauge networks
- NEXRAD Precipitation Processing System — PPS (Fulton et al., 1998) — operational QPE algorithm
- Multisensor Precipitation Estimator — MPE (Seo, 1998) — combines radar with rain gauges
- Probabilistic QPE (Ciach et al., 2007)
- Bayesian QPE (Kirstetter et al., 2015)

_Nowcasting — Classical (1995–2016):_
- Optical Flow Nowcasting — semi-Lagrangian advection of radar fields using optical flow (Bowler et al., QJRMS 2006); computes motion field then advects echoes; widely used baseline for 0–2 hour forecasts
- STEPS — Short-Term Ensemble Prediction System (Bowler et al., QJRMS 2006) — probabilistic nowcasting combining optical flow advection with stochastic perturbations and NWP blending; the standard ensemble nowcasting method
- TITAN — Thunderstorm Identification, Tracking, Analysis and Nowcasting (Dixon & Wiener, 1993) — cell tracking and extrapolation
- TREC — Tracking Radar Echoes by Correlation (Rinehart & Garvey, 1978) — cross-correlation motion estimation
- Spectral Prognosis — S-PROG (Seed, 2003) — scale-dependent nowcasting using Fourier decomposition
- MAPLE — McGill Algorithm for Precipitation Lagrangian Extrapolation (Germann & Zawadzki, 2002)
- pySTEPS (Pulkkinen et al., GMD 2019) — open-source Python nowcasting library implementing STEPS/S-PROG/Lagrangian methods

_Deep Learning Nowcasting (2020–2026):_
- RainNet (Ayzel et al., GMD 2020) — U-Net architecture for radar precipitation nowcasting; trained on DWD composites; first major DL nowcasting paper in geoscience
- MetNet (Sonderby et al., 2020) — large-scale neural weather model for precipitation up to 8 hours; uses axial attention
- NowcastNet (Zhang et al., Nature 2023) — physics-informed deep learning for extreme precipitation nowcasting; combines advection physics with generative modeling; state-of-the-art for heavy rain events
- GenCast (Price et al., Nature 2024) — generative diffusion model for ensemble weather forecasting up to 15 days; outperforms ENS for medium-range forecasts with uncertainty calibration
- DGMR — Deep Generative Model of Radar (Ravuri et al., Nature 2021) — conditional GAN for realistic radar nowcasting; DeepMind; expert-preferred over operational methods
- Pangu-Weather (Bi et al., Nature 2023) — 3D Earth-specific Transformer for global weather forecasting
- GraphCast (Lam et al., Science 2023) — graph neural network for global weather forecasting
- SwinRDM — Swin Transformer radar diffusion model (2023)
- ConvLSTM for radar nowcasting (Shi et al., NeurIPS 2015) — the foundational DL sequence-to-sequence model for radar prediction
- PredRNN (Wang et al., NeurIPS 2017) — spatiotemporal LSTM for radar prediction
- DL attenuation correction for radar (2022)
- Foundation model for weather prediction (FourCastNet, Pathak et al., 2022; Aurora, 2024)

#### Step 3: Update Weather Radar Solvers

After listing all weather radar solvers, update `algorithm_base/weather_radar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All weather radar solvers use the data format: `y` (num_elevations, num_azimuths, num_range_gates) polar volume scan data or (H, W, T) Cartesian radar composite time-series, `radar_params` dict containing wavelength, PRF, beamwidth, and scan strategy. The `WeatherRadarOperator` handles radar equation forward model, Doppler processing, dual-pol variable estimation, and advection operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Weather Radar:**
- NEXRAD QPE: Z-R (Marshall-Palmer) RMSE ~5 mm/h, dual-pol QPE ~3 mm/h, DL QPE ~2.5 mm/h (validated against gauge networks)
- Nowcasting (1-hour lead time, SEVIR): optical flow CSI ~0.35, ConvLSTM ~0.40, DGMR ~0.45, NowcastNet ~0.50
- Nowcasting (6-hour lead time): STEPS CSI ~0.15, GenCast CRPS improvement >10% over ENS
- Hydrometeor classification: HCA accuracy ~85% (validated against disdrometer observations)
- Published CSI/FSS/CRPS from the original papers and operational verification

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'weather_radar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/weather_radar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/weather_radar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/weather_radar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for weather radar. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/weather_radar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/weather_radar/standard/`

---

### Sonar (`sonar`) Modality Template

#### Step 1: Verify Standard Dataset

For Sonar, what dataset do you use to verify? Is this dataset used for sonar popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sonar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original sonar standard dataset.

**Popular datasets to consider:**
- **NSWC Mine-Like Objects Dataset (US Naval Surface Warfare Center)** — the most widely referenced sonar classification benchmark; mine-like objects vs. clutter in sidescan sonar imagery; used for sonar ATR algorithm development since the 1990s
- **Forward-Looking Sonar (FLS) Benchmark Datasets** — acoustic camera data (e.g., DIDSON, ARIS, BlueView) with annotated targets; used for underwater object detection and tracking
- **MBES Bathymetry Datasets (IHO standards)** — multibeam echosounder bathymetric survey data with known seabed topography; used for beamforming and depth estimation algorithm validation
- **SSAS Datasets (Synthetic and Structured Aperture Sonar)** — high-resolution SAS imagery with autofocus validation data; used for SAS image formation algorithm benchmarks
- **Seabed Objects using Synthetic Aperture Sonar (SOSAS, 2020)** — curated SAS dataset for object detection and classification
- **UCI Sonar Dataset (Gorman & Sejnowski, 1988)** — 208 sonar returns (mines vs. rocks); classical ML benchmark; small but historically significant
- **MARIS Dataset (Marine Autonomous Systems, 2018)** — forward-looking sonar images for marine robotics
- **Klipsch Sonar Image Benchmark** — annotated sidescan sonar images for target detection
- **NURC NATO SAS Datasets** — multi-look SAS data from NATO research center for mine countermeasures

**Decision criteria:** NSWC mine-like objects dataset is the standard for sonar ATR benchmarking. SAS datasets for image formation algorithm validation. MBES for bathymetric algorithm benchmarks. FLS benchmarks for real-time underwater detection. Use the dataset that appears in the largest number of sonar processing and classification papers (1988–2026).

#### Step 2: List All Sonar Algorithms

Please first ensure all the sonar algorithms have been listed in `\pwm\public\algorithm_base\sonar\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sonar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the sonar solvers, please update the sonar solver.

**Key algorithms to cover (1950–2026):**

_Beamforming (1960s–2010):_
- DAS Beamforming — Delay-And-Sum (conventional beamforming); the foundational sonar signal processing technique; sums time-delayed signals from array elements to steer and focus the beam; used in all sonar systems
- MVDR Beamforming — Minimum Variance Distortionless Response / Capon beamformer (Capon, 1969) — adaptive beamformer that minimizes output power while maintaining unity response in look direction; superior resolution to DAS
- MUSIC — Multiple Signal Classification (Schmidt, 1986) — subspace-based high-resolution direction of arrival estimation; used for sonar source localization
- ESPRIT — Estimation of Signal Parameters via Rotational Invariance Techniques (Roy & Kailath, 1989)
- Adaptive Beamforming (Van Trees, 2002) — broad class of data-dependent beam steering methods
- Frequency-domain beamforming (Etter, 2013)
- Broadband beamforming for wideband sonar (2005)

_Matched Filtering & Detection (1950s–2015):_
- Matched Filter — optimal SNR filter correlating received signal with transmitted waveform replica; the fundamental sonar detection technique (Van Trees, 1971)
- CFAR Detection — Constant False Alarm Rate (Finn & Johnson, 1968; applied to sonar) — adaptive threshold setting based on local noise statistics; used for sonar target detection in clutter
- Cell-Averaging CFAR (CA-CFAR) for sonar (1973)
- OS-CFAR — Order Statistics CFAR for non-homogeneous clutter (1985)
- Generalized Likelihood Ratio Test for sonar detection (1990)
- Replica correlation for active sonar (1960s)

_SAS Image Formation & Autofocus (1990s–2015):_
- SAS Autofocus — Synthetic Aperture Sonar phase correction (Bonifant et al., 2000); micronavigation and autofocus algorithms analogous to SAR PGA; compensates for platform motion errors in SAS
- Phase Gradient Autofocus for SAS (Fortune et al., 2001) — PGA adapted for sonar geometry and motion characteristics
- Displaced Phase Center Antenna (DPCA) micronavigation (Bellettini & Pinto, 2002) — uses along-track array overlap for motion estimation
- Omega-K SAS image formation (Hawkins, 1996) — wavenumber domain SAS focusing
- Time-domain backprojection for SAS (Hunter et al., 2003)
- Multi-aspect SAS image formation (2008)
- Wideband SAS autofocus (2010)

_Sidescan & Seabed Processing (1980s–2015):_
- Sidescan Sonar Mosaicking — geocorrection and mosaicking of sidescan sonar imagery (Reed & Hussong, 1989); slant-range to ground-range conversion, radiometric normalization, and spatial registration
- Bottom Detection for MBES (Lurton, 2002) — amplitude and phase detection for seabed depth extraction from multibeam sonar
- Backscatter Correction and Angular Response Compensation (de Moustier, 1993)
- Acoustic seafloor classification (Preston et al., 2004)
- Texture-based seabed segmentation (Reed & Hussong, 1989)
- MBES water column processing (2010)

_Deep Learning (2019–2026):_
- CNN for Sonar ATR (Williams, 2016; Zhu et al., 2019) — convolutional neural networks for mine-like object classification in sonar imagery; outperforms traditional feature-based classifiers
- DL Beamforming (Luo et al., 2020) — learned beamforming replacing or augmenting conventional beamformers
- YOLO for real-time sonar object detection (Lee et al., 2020)
- GAN for sonar image generation and data augmentation (Song et al., 2019)
- Transfer learning from optical to sonar domain (2020)
- U-Net for sonar image segmentation (2020)
- Self-supervised sonar representation learning (2022)
- SAS autofocus with deep learning (2022)
- Transformer for sonar target classification (2023)
- Diffusion-model sonar image enhancement (2024)
- Foundation model for underwater acoustic sensing (2025)

#### Step 3: Update Sonar Solvers

After listing all sonar solvers, update `algorithm_base/sonar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All sonar solvers use the data format: `y` (num_elements, num_time_samples) raw hydrophone array data or (num_pings, num_range_samples) sonar imagery, `array_params` dict containing element positions, sampling rate, and speed of sound. The `SonarOperator` handles beamforming (forward/adjoint), matched filtering, SAS image formation, and detection operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Sonar:**
- NSWC mine-like objects: template matching ~75% Pd at 1% FAR, SVM ~85%, CNN ~93%, Transformer ~95%
- SAS autofocus: DPCA+PGA residual phase error <0.1 rad RMS; image contrast improvement >3 dB
- MBES bathymetry: DAS depth accuracy IHO Order 1 (~0.5m), MVDR improved seabed resolution >20%
- UCI sonar benchmark: logistic regression ~75%, neural network ~85%, deep CNN ~90% accuracy
- Published Pd/FAR/contrast from the original papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sonar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sonar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sonar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sonar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for sonar. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sonar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sonar/standard/`
