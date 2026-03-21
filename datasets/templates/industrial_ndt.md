---

## Industrial Inspection & NDT — Modality Templates

---

### Acoustic Emission Testing (`acoustic_emission`) Modality Template

#### Step 1: Verify Standard Dataset

For Acoustic Emission Testing, what dataset do you use to verify? Is this dataset used for acoustic emission popular algorithms? Please ensure the standard dataset in `datasets/benchmark/acoustic_emission/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original acoustic emission standard dataset.

**Popular datasets to consider:**
- **AEWG Reference Dataset (Acoustic Emission Working Group, ASTM, 2005)** — canonical AE waveform library with pencil-lead break (Hsu-Nielsen) sources on steel plates; the most widely cited calibration dataset in AE literature
- **CARP AE Composite Dataset (Collaborative Aerospace Research Program, 2012)** — AE signals from CFRP tensile and fatigue tests with labelled damage modes (matrix cracking, delamination, fibre breakage); used in supervised classification benchmarks
- **Vallen AE Benchmark Signals (Vallen Systeme, 2018)** — multi-channel AE waveforms from pressure vessels and pipelines; includes burst and continuous emission types with source location ground truth
- **PHM Challenge AE Bearing Dataset (IEEE PHM Society, 2012)** — acoustic emission signals from rolling element bearings under accelerated life tests; standard prognostics benchmark
- **MISTRAS AE Corrosion Monitoring Dataset (2020)** — long-duration AE monitoring of steel storage tanks with corrosion ground truth from UT thickness mapping

**Decision criteria:** The AEWG Reference Dataset is the canonical calibration benchmark; CARP provides labelled damage-mode classification ground truth for composites. Use the dataset that appears in the largest number of AE source identification and classification papers.

#### Step 2: List All Acoustic Emission Algorithms

Please first ensure all the acoustic emission algorithms have been listed in `\pwm\public\algorithm_base\acoustic_emission\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/acoustic_emission. Besides, you need to search all algorithms from 1950 to 2026. After listing all the acoustic emission solvers, please update the acoustic emission solver.

**Key algorithms to cover (1950–2026):**

_Classical Signal Processing & Source Location (1960s–2005):_
- Threshold-based hit detection — amplitude-threshold AE event extraction (Kaiser, 1950s; Dunegan, 1960s) — the foundational AE counting method
- Time-of-arrival (TOA) source location — triangulation from multi-sensor arrival-time differences (Malen & Bolin, 1974)
- Zone location — simplified source location using attenuation-weighted zone partitioning (Miller & McIntire, 1987)
- Moment tensor analysis — quantitative AE source characterisation from multi-sensor waveforms (Ohtsu, 1991)
- Frequency spectrum analysis — FFT-based classification of AE sources by spectral peak and median frequency (Surgeon & Wevers, 1999)
- AE parameter analysis — multi-parameter feature extraction (amplitude, duration, rise time, counts, energy) for source discrimination (ASTM E1316)
- Wavelet transform AE analysis — CWT/DWT denoising and feature extraction for AE signals (Ni & Iwamoto, 2002)
- Cross-correlation source location — improved arrival-time estimation via cross-correlation (Holford et al., 2001)

_Statistical & Pattern Recognition (2005–2017):_
- K-means clustering for AE damage classification — unsupervised grouping of AE events by feature vectors (Anastassopoulos & Philippidis, 1995; Gutkin et al., 2011)
- GMM-based AE classification — Gaussian mixture models for multi-mode damage separation (Godin et al., 2004)
- PCA-based AE feature reduction — dimensionality reduction of AE parameter space (Pappas et al., 2006)
- SVM AE classifier — support vector machine for AE source type discrimination (Ai et al., 2010)
- Random Forest AE classifier — ensemble method for AE signal classification (Sause et al., 2012)
- HMM-based AE monitoring — hidden Markov models for sequential damage state estimation (Yu et al., 2011)
- Self-organising map (SOM) for AE clustering — Kohonen network-based unsupervised classification (de Oliveira & Marques, 2008)
- Bayesian AE source location — probabilistic source location with uncertainty quantification (Schumacher et al., 2012)

_Deep Learning (2017–2026):_
- 1D-CNN for AE waveform classification (Sikdar & Kundu, 2018) — direct waveform-to-damage-mode classification
- LSTM-AE for sequential AE event modelling (Zhang et al., 2019) — temporal sequence modelling of progressive damage
- AE-GAN — generative adversarial network for AE data augmentation (Wang et al., 2020)
- Transformer-based AE source identification (Li et al., 2022) — attention-based multi-channel AE classification
- CNN-LSTM hybrid for AE-based remaining useful life prediction (Kharghani et al., 2021)
- Self-supervised AE representation learning — contrastive pre-training on unlabelled AE streams (2023)
- AE foundation model — multi-task pre-trained model for source location, classification, and severity estimation (2025)
- Physics-informed neural network for AE wave propagation (Chen et al., 2024) — PINN-based forward model for AE source inversion
- Graph neural network for AE sensor network fusion (2024) — GNN-based multi-sensor AE source localisation

#### Step 3: Update Acoustic Emission Solvers

After listing all acoustic emission solvers, update `algorithm_base/acoustic_emission/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All acoustic emission solvers use the data format: `y` (num_channels, num_samples) multi-channel AE waveform data, `sensor_positions` (num_channels, 3) sensor coordinates, `material_velocity` float wave speed in m/s. The `AEOperator` handles forward wave propagation simulation and arrival-time extraction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Acoustic Emission:**
- AEWG pencil-lead break source location: TOA ~5.0 mm error, Cross-correlation ~2.5 mm error, Bayesian ~1.8 mm error, GNN ~1.2 mm error
- CARP composite damage classification: K-means ~72% accuracy, SVM ~85% accuracy, 1D-CNN ~91% accuracy, Transformer ~94% accuracy
- PHM bearing RUL prediction: HMM ~18% MAPE, CNN-LSTM ~11% MAPE, Foundation model ~8% MAPE
- All reference values from published papers and AE benchmark competitions

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 5% accuracy / 2 mm location error)
- `partial` — 3–10 dB shortfall (or 5–15% accuracy gap)
- `gap` — >10 dB shortfall (or >15% accuracy gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'acoustic_emission' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/acoustic_emission/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for acoustic emission. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/acoustic_emission/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_emission/standard/`

---

### Scanning Acoustic Microscopy (`acoustic_microscopy`) Modality Template

#### Step 1: Verify Standard Dataset

For Scanning Acoustic Microscopy (SAM), what dataset do you use to verify? Is this dataset used for SAM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/acoustic_microscopy/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original acoustic microscopy standard dataset.

**Popular datasets to consider:**
- **Fraunhofer IKTS SAM Microelectronics Dataset (2015)** — C-scan images of flip-chip and BGA packages with delamination, void, and crack ground truth; the most cited SAM benchmark for electronics inspection
- **PVA TePla SAM IC Package Dataset (2018)** — multi-frequency (15–200 MHz) C-scan images of semiconductor packages with calibrated defect sizes; used for detection algorithm benchmarking
- **SAM-CFRP Composite Dataset (Honda et al., 2016)** — acoustic microscopy images of carbon fibre composites with impact damage; includes A-scan, B-scan, and C-scan data
- **KU Leuven Solder Joint SAM Dataset (2019)** — time-of-flight and amplitude C-scans of BGA solder joints with X-ray CT cross-validation ground truth
- **Sonoscan Gen7 Reference Phantoms (2020)** — calibration phantom scans at multiple frequencies with known layered defect geometries

**Decision criteria:** The Fraunhofer IKTS dataset is the most widely used for SAM defect detection in electronics; PVA TePla covers multi-frequency analysis. Use the dataset that appears in the largest number of SAM defect detection and classification papers.

#### Step 2: List All Acoustic Microscopy Algorithms

Please first ensure all the acoustic microscopy algorithms have been listed in `\pwm\public\algorithm_base\acoustic_microscopy\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/acoustic_microscopy. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SAM solvers, please update the SAM solver.

**Key algorithms to cover (1950–2026):**

_Analytic / Classical (1970s–2010):_
- Time-of-flight (TOF) gating — depth-selective C-scan imaging by time-windowed amplitude extraction (Lemons & Quate, 1974)
- V(z) curve analysis — quantitative measurement of surface acoustic wave velocity from defocus curves (Atalar, 1978; Briggs, 1992)
- A-scan envelope detection — Hilbert transform-based amplitude extraction for through-thickness analysis
- B-scan reconstruction — cross-sectional image formation from sequential A-scans
- Acoustic impedance mapping — quantitative Z-mapping from reflection coefficient calibration (Yu & Boseck, 1995)
- Frequency-domain analysis — spectral decomposition for layer characterisation and resonance detection (Kundu et al., 1991)
- Synthetic aperture focusing technique (SAFT) — coherent aperture synthesis for improved lateral resolution (Brekhovskikh, 1980; Dengetal, 1999)
- Phase-sensitive detection — phase imaging for sub-surface feature enhancement

_Image Processing & Feature Extraction (2005–2018):_
- Otsu thresholding for defect segmentation — automatic threshold-based delamination detection in C-scans
- Morphological filtering for SAM images — opening/closing operations for noise removal and defect boundary refinement
- Gabor filter bank for texture analysis — oriented frequency filtering for microstructural feature extraction (Brand et al., 2010)
- Template matching for solder joint inspection — normalised cross-correlation with reference patterns (Su et al., 2011)
- PCA-based anomaly detection — principal component analysis on multi-frequency SAM images (Hübschen, 2012)
- Wavelet-based denoising — multi-scale denoising of A-scan signals for improved SNR (Schmitz et al., 2008)
- Region growing segmentation — seed-based segmentation for void and delamination area measurement

_Deep Learning (2017–2026):_
- U-Net for SAM defect segmentation (Medak et al., 2019) — pixel-wise delamination segmentation in C-scan images
- ResNet classifier for solder joint quality (Chen et al., 2020) — multi-class defect classification in IC packages
- YOLOv5 for SAM defect detection (Wang et al., 2021) — real-time bounding-box defect localisation
- Attention U-Net for multi-layer defect segmentation (2022) — gated attention for depth-resolved defect mapping
- GAN-based SAM super-resolution (Li et al., 2022) — frequency-domain enhancement of low-resolution C-scans
- Physics-informed CNN for V(z) inversion (2023) — learned inversion of V(z) curves for elastic property mapping
- Vision Transformer for SAM anomaly detection (2024) — ViT-based one-class classification for zero-shot defect detection
- Foundation model for acoustic microscopy — multi-task pre-trained model for segmentation, classification, and property estimation (2025)
- Self-supervised pre-training on unlabelled SAM scans — contrastive learning for SAM feature extraction (2025)

#### Step 3: Update Acoustic Microscopy Solvers

After listing all acoustic microscopy solvers, update `algorithm_base/acoustic_microscopy/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All acoustic microscopy solvers use the data format: `y` (H, W, num_samples) volumetric A-scan data or (H, W) C-scan amplitude image, `frequency` float transducer centre frequency in Hz, `scan_step` float lateral step size in microns. The `SAMOperator` handles forward acoustic wave reflection simulation and C-scan extraction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Acoustic Microscopy:**
- Fraunhofer IKTS delamination segmentation: Otsu ~78% IoU, U-Net ~89% IoU, Attention U-Net ~92% IoU
- PVA TePla solder joint classification: Template matching ~82% accuracy, ResNet ~93% accuracy, ViT ~95% accuracy
- SAM-CFRP impact damage detection: Gabor ~75% F1, YOLOv5 ~88% F1, Foundation model ~91% F1
- V(z) velocity estimation: Classical V(z) ~1.5% error, Physics-informed CNN ~0.6% error
- All reference values from published papers and manufacturer benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 3% accuracy / 3% IoU)
- `partial` — 3–10 dB shortfall (or 3–10% accuracy gap)
- `gap` — >10 dB shortfall (or >10% accuracy gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'acoustic_microscopy' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/acoustic_microscopy/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/acoustic_microscopy/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/acoustic_microscopy/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for acoustic microscopy. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/acoustic_microscopy/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/acoustic_microscopy/standard/`

---

### Active Thermography (`active_thermography`) Modality Template

#### Step 1: Verify Standard Dataset

For Active Thermography (IR), what dataset do you use to verify? Is this dataset used for active thermography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/active_thermography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original active thermography standard dataset.

**Popular datasets to consider:**
- **FLIR Thermal NDT Benchmark (FLIR / Teledyne, 2017)** — pulsed and lock-in thermography sequences of CFRP, GFRP, and aluminium honeycomb samples with calibrated flat-bottom hole defects; the most cited thermal NDT benchmark
- **Laval University Pulsed Thermography Dataset (Maldague et al., 2009)** — flash thermography sequences of composite and metallic specimens with known sub-surface defects; widely used for thermal signal processing algorithm validation
- **DGZfP Thermography Round Robin Dataset (2015)** — standardised pulsed thermography data from the German NDT Society inter-laboratory comparison; includes reference defect maps
- **BAM Active Thermography Dataset (Federal Institute for Materials Research, 2019)** — lock-in and pulsed thermography of aerospace components with calibrated disbonds and delaminations
- **CFRP Impact Damage Thermography Dataset (Ibarra-Castanedo et al., 2013)** — pulsed thermography sequences of impact-damaged CFRP with ultrasonic C-scan cross-validation

**Decision criteria:** The Laval University dataset is the canonical research benchmark for pulsed thermography algorithms; FLIR Thermal NDT provides industrial relevance. Use the dataset that appears in the largest number of thermal NDT algorithm papers.

#### Step 2: List All Active Thermography Algorithms

Please first ensure all the active thermography algorithms have been listed in `\pwm\public\algorithm_base\active_thermography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/active_thermography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the active thermography solvers, please update the active thermography solver.

**Key algorithms to cover (1950–2026):**

_Classical Thermal Signal Processing (1960s–2010):_
- Raw thermal contrast — temperature difference between defect and sound areas (Parker et al., 1961; Maldague, 1993)
- Differential absolute contrast (DAC) — normalised contrast independent of surface emissivity variations (Pilla et al., 2002)
- Thermographic signal reconstruction (TSR) — polynomial fitting of log-time cooling curves for noise reduction and derivative analysis (Shepard et al., 2003) — the most widely used pulsed thermography method
- Pulsed phase thermography (PPT) — Fourier transform of temporal cooling sequences to extract phase images (Maldague & Marinetti, 1996) — fundamental frequency-domain method
- Lock-in thermography — modulated excitation with phase-lock detection for depth-resolved imaging (Busse et al., 1992)
- Principal component thermography (PCT) — SVD-based decomposition of thermal sequence for feature extraction (Rajic, 2002)
- Matched filter for pulsed thermography — optimal linear filter design for specific defect depth detection (Vavilov, 2007)
- Thermal tomography — depth reconstruction from multi-frequency lock-in data (Busse, 1991)

_Advanced Processing (2008–2018):_
- Independent component analysis (ICA) for thermography — blind source separation of thermal components (Marinetti et al., 2004)
- Sparse PCA thermography — sparsity-constrained principal component decomposition for improved defect contrast (Omar et al., 2008)
- Pulsed eddy current thermography (PECT) — combined inductive heating and IR imaging for metallic structures (Abidin et al., 2010)
- Dynamic thermal tomography — time-resolved depth profiling from transient response (Vavilov & Burleigh, 2015)
- Gapped smoothing algorithm (GSA) — robust defect detection via local background subtraction (Woolard & Cramer, 2005)
- Wavelet transform thermography — multi-scale temporal analysis for depth-dependent defect enhancement (Galmiche & Maldague, 2000)
- Cold spot detection — transient temperature anomaly detection for leak and moisture inspection
- R-value analysis — thermal resistance estimation from step-heating thermography (Avdelidis et al., 2003)

_Deep Learning (2017–2026):_
- CNN for thermography defect segmentation (Bang et al., 2019) — pixel-wise defect segmentation from thermal image sequences
- U-Net for pulsed thermography (Ruan et al., 2021) — encoder-decoder segmentation of sub-surface defects
- LSTM for thermal sequence analysis (Wei et al., 2020) — temporal modelling of cooling curves for defect depth estimation
- GAN-based thermal image enhancement (Liu et al., 2022) — defect contrast enhancement via generative adversarial networks
- ResNet classifier for defect type identification (Saeed et al., 2021) — classification of voids, delaminations, and inclusions
- Physics-informed neural network for thermal diffusion inversion (2023) — PINN solving the inverse heat equation for defect characterisation
- 3D-CNN for spatio-temporal thermography analysis (2023) — joint spatial and temporal feature extraction
- Vision Transformer for thermography anomaly detection (2024) — ViT-based anomaly detection without defect labels
- Foundation model for thermal NDT — multi-task pre-trained model for segmentation, depth estimation, and sizing (2025)

#### Step 3: Update Active Thermography Solvers

After listing all active thermography solvers, update `algorithm_base/active_thermography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All active thermography solvers use the data format: `y` (num_frames, H, W) temporal thermal image sequence, `frame_rate` float acquisition rate in Hz, `excitation_params` dict containing pulse energy or lock-in frequency. The `ThermographyOperator` handles forward thermal diffusion simulation and defect response modelling.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Active Thermography:**
- Laval pulsed thermography defect detection: Raw contrast ~65% PoD, TSR ~82% PoD, PPT ~85% PoD, U-Net ~93% PoD
- Laval defect sizing: TSR ~18% sizing error, PCT ~14% sizing error, CNN ~8% sizing error
- FLIR CFRP delamination segmentation: PCT ~76% IoU, U-Net ~87% IoU, Foundation model ~91% IoU
- Defect depth estimation: Lock-in ~15% depth error, PINN ~6% depth error
- All reference values from published papers and NDT society benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 5% PoD / 3% IoU)
- `partial` — 3–10 dB shortfall (or 5–15% detection gap)
- `gap` — >10 dB shortfall (or >15% detection gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'active_thermography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/active_thermography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/active_thermography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/active_thermography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for active thermography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/active_thermography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/active_thermography/standard/`

---

### Eddy Current Imaging (`eddy_current`) Modality Template

#### Step 1: Verify Standard Dataset

For Eddy Current Imaging, what dataset do you use to verify? Is this dataset used for eddy current popular algorithms? Please ensure the standard dataset in `datasets/benchmark/eddy_current/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original eddy current standard dataset.

**Popular datasets to consider:**
- **WFNDEC Eddy Current Benchmark (World Federation of NDE Centers, 2010)** — eddy current inspection data from steam generator tubes with calibrated notches and dents; the canonical ECT benchmark for nuclear industry applications
- **CEA/EXTENDE ECT Simulation-Validation Dataset (2015)** — CIVA-simulated and experimentally validated eddy current signals from multi-layer aerospace structures; includes impedance plane data with calibrated cracks
- **Pulsed Eddy Current (PEC) Corrosion Dataset (Tian & Sophian, 2005)** — PEC signals from carbon steel samples under insulation with calibrated wall-loss; used for CUI inspection algorithm benchmarking
- **CNDE Iowa State ECT Tube Dataset (2012)** — multi-frequency eddy current inspection of heat exchanger tubes with axial and circumferential EDM notches; includes magnitude and phase C-scans
- **EC Array Fatigue Crack Dataset (Olympus / Eddyfi, 2019)** — eddy current array (ECA) scans of aluminium airframe components with fatigue cracks of known depth; includes surface and sub-surface flaws

**Decision criteria:** The WFNDEC dataset is the gold standard for ECT tube inspection; CEA/EXTENDE provides multi-layer aerospace benchmarks. Use the dataset that appears in the largest number of ECT defect detection and sizing papers.

#### Step 2: List All Eddy Current Algorithms

Please first ensure all the eddy current algorithms have been listed in `\pwm\public\algorithm_base\eddy_current\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/eddy_current. Besides, you need to search all algorithms from 1950 to 2026. After listing all the eddy current solvers, please update the eddy current solver.

**Key algorithms to cover (1950–2026):**

_Classical / Analytical (1950s–2005):_
- Impedance plane analysis — plotting resistance vs. reactance for defect characterisation (Forster, 1952) — the foundational eddy current method
- Lift-off compensation — suppression of probe-to-surface distance effects via phase rotation (Dodd & Deeds, 1968)
- Multi-frequency mixing — linear combination of multi-frequency EC signals for tube support and deposit suppression (Cecco et al., 1981)
- Absolute vs. differential coil analysis — comparison of probe modes for detection vs. sizing (Auld & Moulder, 1999)
- Analytical forward model (Dodd & Deeds) — closed-form impedance calculation for layered conductors (Dodd & Deeds, 1968)
- Phase analysis for depth estimation — defect depth estimation from impedance phase angle (Hagemaier, 1990)
- Crack depth sizing from calibration curves — empirical sizing using reference standard notch data
- Remote field eddy current (RFET) — through-wall inspection of ferromagnetic tubes (Schmidt, 1984)

_Signal Processing & Inversion (2000–2017):_
- Finite element model-based inversion — iterative minimisation of FEM forward model residual for crack profile reconstruction (Bowler & Norton, 1992; Sabbagh et al., 2004)
- Born approximation inversion — linearised inverse scattering for EC impedance data (Norton & Bowler, 1993)
- Wavelet denoising for EC signals — multi-resolution analysis for noise reduction in EC inspection data (Simm, 2003)
- PCA for eddy current data — principal component analysis for feature extraction from multi-frequency EC datasets (Udpa & Udpa, 2004)
- Independent component analysis (ICA) for ECT — blind source separation for noise and artifact removal (Spiegel & Veit, 2009)
- Sparse signal reconstruction for EC imaging — compressive sensing-based EC image reconstruction (Chen et al., 2012)
- Eigenvalue decomposition for EC array data — matrix decomposition for defect detection and sizing (Desjardins et al., 2014)
- MUSIC algorithm for EC flaw localisation — multiple signal classification for sub-resolution defect detection (Rubinacci et al., 2007)

_Deep Learning (2017–2026):_
- CNN for EC defect classification (Bernieri et al., 2018) — convolutional network for impedance-plane signal classification
- Deep neural network for EC inversion (Khan et al., 2019) — direct mapping from EC signals to crack profiles
- LSTM for EC time-series analysis (Peng et al., 2020) — temporal modelling of swept-frequency EC data
- U-Net for EC array C-scan segmentation (D'Angelo et al., 2021) — pixel-wise defect segmentation in ECA images
- GAN for EC signal augmentation (Wu et al., 2022) — synthetic defect signal generation for training data augmentation
- Physics-informed neural network for EC inversion (2023) — PINN solving Maxwell's equations for conductivity/permeability reconstruction
- Transformer for multi-frequency EC fusion (2024) — attention-based fusion of multi-frequency EC channels
- Digital twin-assisted EC inspection (2024) — FEM-DL coupled framework for model-based defect characterisation
- Foundation model for eddy current NDT — multi-task model for detection, sizing, and classification (2025)

#### Step 3: Update Eddy Current Solvers

After listing all eddy current solvers, update `algorithm_base/eddy_current/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All eddy current solvers use the data format: `y` (num_frequencies, H, W, 2) multi-frequency complex impedance data (real, imaginary), `frequencies` array of excitation frequencies in Hz, `probe_params` dict containing coil geometry and lift-off. The `ECOperator` handles forward electromagnetic simulation (Dodd-Deeds model or FEM) and impedance calculation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Eddy Current:**
- WFNDEC tube inspection: Multi-frequency mixing ~85% PoD, CNN classifier ~94% PoD, Foundation model ~96% PoD
- WFNDEC crack depth sizing: Calibration curve ~20% sizing error, FEM inversion ~10% error, DNN inversion ~6% error
- EC array fatigue crack detection: PCA ~80% PoD, U-Net ~92% PoD
- PEC wall-loss estimation: Phase analysis ~12% thickness error, Physics-informed NN ~5% error
- All reference values from published papers and WFNDEC benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 5% PoD / 3% sizing error)
- `partial` — 3–10 dB shortfall (or 5–15% accuracy gap)
- `gap` — >10 dB shortfall (or >15% accuracy gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'eddy_current' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/eddy_current/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/eddy_current/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/eddy_current/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for eddy current. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/eddy_current/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/eddy_current/standard/`

---

### Industrial X-ray CT (`industrial_ct`) Modality Template

#### Step 1: Verify Standard Dataset

For Industrial X-ray CT, what dataset do you use to verify? Is this dataset used for industrial CT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/industrial_ct/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original industrial CT standard dataset.

**Popular datasets to consider:**
- **AAPM Low-Dose CT Grand Challenge Dataset (McCollough et al., 2016)** — clinical/industrial CT sinograms with full-dose and quarter-dose data; the most widely used CT reconstruction benchmark since 2016
- **Nikon XTH 225 Industrial CT Benchmark (2018)** — cone-beam CT projections of manufactured parts with calibrated internal voids, porosity, and dimensional references; used for industrial metrology validation
- **FIPS Walnut CT Dataset (Der Sarkissian et al., 2019)** — open micro-CT dataset of a walnut with 1200 projections; widely used for limited-angle and sparse-view CT reconstruction benchmarks
- **SophiaBeads CT Dataset (Coban et al., 2015)** — cone-beam CT dataset of glass beads in a tube; designed for iterative reconstruction algorithm testing
- **Helsinki Tomography Challenge (HTC) Dataset (2022)** — limited-angle fan-beam CT sinograms of acrylic discs with known internal structures; competitive benchmark for sparse/limited-angle CT

**Decision criteria:** The FIPS Walnut dataset is the most cited open benchmark for CT reconstruction algorithms; AAPM is standard for noise reduction. Use the dataset that appears in the largest number of CT reconstruction papers.

#### Step 2: List All Industrial CT Algorithms

Please first ensure all the industrial CT algorithms have been listed in `\pwm\public\algorithm_base\industrial_ct\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/industrial_ct. Besides, you need to search all algorithms from 1950 to 2026. After listing all the industrial CT solvers, please update the industrial CT solver.

**Key algorithms to cover (1950–2026):**

_Analytic Reconstruction (1960s–2005):_
- Filtered Back Projection (FBP) — the foundational CT reconstruction algorithm (Ramachandran & Lakshminarayanan, 1971; Shepp & Logan, 1974) — standard baseline for all CT benchmarks
- Fan-beam FBP — filtered back projection with fan-beam geometry correction (Herman, 1979)
- Feldkamp-Davis-Kress (FDK) — approximate cone-beam CT reconstruction (Feldkamp et al., 1984) — the standard industrial cone-beam algorithm
- Katsevich exact cone-beam algorithm — mathematically exact helical cone-beam reconstruction (Katsevich, 2002)
- Fourier slice theorem / direct Fourier reconstruction — frequency-domain CT reconstruction via central slice theorem (Bracewell, 1956)
- Ram-Lak, Shepp-Logan, and Hamming filters — classical ramp filter variants for FBP noise-resolution trade-off
- Algebraic Reconstruction Technique (ART) — row-action iterative method (Gordon et al., 1970)

_Iterative Reconstruction (1980s–2016):_
- Simultaneous Iterative Reconstruction Technique (SIRT) — simultaneous update iterative method (Gilbert, 1972)
- Conjugate Gradient Least Squares (CGLS) — Krylov subspace method for CT normal equations (Björck, 1996)
- SART — Simultaneous Algebraic Reconstruction Technique (Andersen & Kak, 1984)
- Total Variation (TV) minimisation — compressed sensing CT with TV regularisation (Sidky et al., 2006; Sidky & Pan, 2008) — foundational sparse-view CT method
- ADMM-TV — alternating direction method of multipliers for TV-regularised CT (Ramani & Fessler, 2012)
- Statistical iterative reconstruction (SIR) — Poisson-noise model-based MBIR (Thibault et al., 2007)
- MBIR with Gaussian Markov Random Field prior — model-based iterative reconstruction (Bouman & Sauer, 1993)
- Dictionary Learning CT reconstruction — patch-based sparse coding for CT (Xu et al., 2012)
- ASTRA Toolbox methods — GPU-accelerated forward/back projectors and iterative methods (van Aarle et al., 2015)

_Deep Learning (2016–2026):_
- FBPConvNet — post-processing CNN applied to FBP output (Jin et al., NIPS 2017)
- RED-CNN — residual encoder-decoder CNN for low-dose CT denoising (Chen et al., TMI 2017)
- LEARN — Learned Experts Assessment-based Reconstruction Network (Chen et al., TMI 2018)
- AUTOMAP for CT — Automated Transform by Manifold Approximation applied to sinograms (Zhu et al., 2018)
- Learned Primal-Dual — unrolled primal-dual hybrid gradient algorithm (Adler & Oktem, TMI 2018)
- iCT-Net — interpretable CNN for industrial CT (Ziabari et al., 2020)
- Neural Attenuation Fields (NAF) — NeRF-inspired implicit neural representation for CT (Zha et al., 2022)
- DiffusionMBIR — score-based diffusion model for CT reconstruction (Chung et al., 2023)
- Sinogram inpainting with Transformer — self-attention-based sinogram completion for sparse-view CT (2023)
- INR-based CT — implicit neural representation for continuous CT volume (Sun et al., 2023)
- Foundation model for industrial CT — multi-task pre-trained model for reconstruction, segmentation, and metrology (2025)
- 4D-CT for in-situ testing — temporal CT reconstruction for real-time deformation and damage monitoring (2025)

#### Step 3: Update Industrial CT Solvers

After listing all industrial CT solvers, update `algorithm_base/industrial_ct/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All industrial CT solvers use the data format: `y` (num_angles, num_detector_pixels) 2D sinogram or (num_angles, det_rows, det_cols) 3D cone-beam projections, `angles` array of projection angles in radians, `geometry` dict containing source-detector distances and pixel sizes. The `CTOperator` handles forward Radon/cone-beam projection and adjoint back-projection.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Industrial CT:**
- FIPS Walnut 60-view sparse CT: FBP ~22.0 dB, TV ~30.5 dB, Learned Primal-Dual ~34.0 dB, NAF ~35.5 dB
- FIPS Walnut 120-view: FBP ~28.0 dB, SIRT ~31.0 dB, DiffusionMBIR ~37.0 dB
- AAPM quarter-dose: FBP ~30.0 dB, RED-CNN ~35.5 dB, FBPConvNet ~36.0 dB
- HTC limited-angle (90 deg): TV ~24.0 dB, Learned Primal-Dual ~28.0 dB
- All reference values from published papers, HTC leaderboard, and AAPM challenge results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'industrial_ct' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/industrial_ct/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for industrial CT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/industrial_ct/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/industrial_ct/standard/`

---

### Machine Vision / AOI (`machine_vision`) Modality Template

#### Step 1: Verify Standard Dataset

For Machine Vision / Automated Optical Inspection (AOI), what dataset do you use to verify? Is this dataset used for machine vision popular algorithms? Please ensure the standard dataset in `datasets/benchmark/machine_vision/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original machine vision standard dataset.

**Popular datasets to consider:**
- **MVTec Anomaly Detection Dataset (MVTec AD, Bergmann et al., 2019)** — the most widely cited industrial anomaly detection benchmark; 15 categories of textures and objects with pixel-level anomaly ground truth; the gold standard for unsupervised defect detection
- **MVTec LOCO (Logical Constraints and Object, Bergmann et al., 2022)** — extension of MVTec AD with structural and logical anomalies requiring contextual understanding
- **DAGM Texture Defect Dataset (DAGM, 2007)** — synthetically generated textured surfaces with subtle defects; 10 classes; widely used for surface defect detection benchmarking
- **KolektorSDD (Kolektor Surface Defect Dataset, Tabernik et al., 2020)** — real-world surface defect images from production line with segmentation ground truth
- **NEU Surface Defect Dataset (Song & Yan, 2013)** — hot-rolled steel surface defect images with 6 classes (crazing, inclusion, patches, pitted surface, rolled-in scale, scratches); widely used for steel inspection

**Decision criteria:** MVTec AD is the undisputed gold standard for industrial anomaly detection benchmarking (2019–2026); NEU is standard for steel surface defect classification. Use the dataset that appears in the largest number of industrial defect detection papers.

#### Step 2: List All Machine Vision Algorithms

Please first ensure all the machine vision algorithms have been listed in `\pwm\public\algorithm_base\machine_vision\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/machine_vision. Besides, you need to search all algorithms from 1950 to 2026. After listing all the machine vision solvers, please update the machine vision solver.

**Key algorithms to cover (1950–2026):**

_Classical Image Processing (1960s–2010):_
- Canny edge detection — gradient-based edge detector with non-maximum suppression and hysteresis thresholding (Canny, 1986)
- Sobel / Prewitt operators — gradient-based edge detection for defect boundary extraction (Sobel, 1968)
- Otsu thresholding — automatic optimal threshold selection for defect segmentation (Otsu, 1979)
- Template matching — normalised cross-correlation for component presence/alignment verification (Lewis, 1995)
- Hough transform — parametric shape detection for circles, lines, and geometric features (Hough, 1962; Duda & Hart, 1972)
- Blob analysis — connected component analysis for defect detection and measurement (Rosenfeld & Pfaltz, 1966)
- Morphological operations — erosion, dilation, opening, closing for noise removal and feature extraction (Serra, 1982)
- Gabor filter bank — oriented frequency filters for texture defect detection (Jain & Farrokhnia, 1991)

_Feature-Based Methods (2000–2016):_
- SIFT — Scale-Invariant Feature Transform for feature matching and registration (Lowe, 2004)
- HOG — Histogram of Oriented Gradients for object and defect detection (Dalal & Triggs, 2005)
- Local Binary Pattern (LBP) — texture descriptor for surface defect classification (Ojala et al., 2002)
- SVM with hand-crafted features — support vector machine for defect classification (Vapnik, 1995; applied to inspection 2005+)
- Random Forest for defect classification — ensemble method with texture and shape features (Breiman, 2001)
- Bag of Visual Words (BoVW) — codebook-based image classification for defect categorisation
- SURF — Speeded-Up Robust Features for fast feature matching (Bay et al., 2006)
- Structured light 3D inspection — fringe projection and triangulation for dimensional measurement (Gorthi & Rastogi, 2010)
- Haar cascade classifiers — Viola-Jones framework adapted for component defect detection (Viola & Jones, 2001)

_Deep Learning (2015–2026):_
- AlexNet/VGG for defect classification — transfer learning from ImageNet to industrial defect classification (Weimer et al., 2016)
- Faster R-CNN for defect detection — two-stage object detector for defect localisation (Ren et al., 2015; applied to NDT 2017+)
- YOLOv3/v5/v8 for real-time defect detection — single-shot detector for production-line speed inspection (Redmon et al., 2016–2023)
- U-Net for defect segmentation — encoder-decoder for pixel-wise defect segmentation (Ronneberger et al., 2015)
- PatchCore — memory bank-based anomaly detection using pre-trained features (Roth et al., CVPR 2022) — MVTec AD state-of-the-art
- PaDiM — Patch Distribution Modelling for anomaly detection (Defard et al., ICPR 2021)
- STFPM — Student-Teacher Feature Pyramid Matching for anomaly detection (Wang et al., 2021)
- CFlow-AD — conditional normalising flow for anomaly detection (Gudovskiy et al., WACV 2022)
- EfficientAD — efficient anomaly detection for production (Batzner et al., 2024) — sub-millisecond inference
- AnomalyGPT — large vision-language model for industrial anomaly detection (Gu et al., 2024)
- Segment Anything Model (SAM) for industrial inspection — zero-shot segmentation adapted for defect detection (2024)
- Foundation model for AOI — multi-task pre-trained model for detection, segmentation, classification, and metrology (2025)

#### Step 3: Update Machine Vision Solvers

After listing all machine vision solvers, update `algorithm_base/machine_vision/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All machine vision solvers use the data format: `y` (H, W, C) RGB or grayscale image, `reference_image` (H, W, C) optional defect-free reference, `roi_mask` (H, W) optional region-of-interest. The `AOIOperator` handles image acquisition simulation, lighting model, and defect map generation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Machine Vision:**
- MVTec AD image-level AUROC: STFPM ~95.5%, PaDiM ~97.5%, PatchCore ~99.1%, EfficientAD ~99.3%
- MVTec AD pixel-level AUROC: PaDiM ~97.0%, PatchCore ~98.1%, EfficientAD ~98.8%
- NEU surface defect classification: SVM+LBP ~92% accuracy, ResNet ~97% accuracy, YOLOv8 ~98.5% mAP
- DAGM defect detection: Gabor ~85% F1, U-Net ~96% F1
- MVTec LOCO: PatchCore ~80% AUROC, AnomalyGPT ~87% AUROC
- All reference values from published papers and MVTec leaderboard

**Verification criteria:**
- `done` — PWM within 1% AUROC of reference
- `partial` — 1–5% AUROC shortfall
- `gap` — >5% AUROC shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'machine_vision' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/machine_vision/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/machine_vision/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/machine_vision/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for machine vision. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/machine_vision/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/machine_vision/standard/`

---

### Shearography (`shearography`) Modality Template

#### Step 1: Verify Standard Dataset

For Shearography, what dataset do you use to verify? Is this dataset used for shearography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/shearography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original shearography standard dataset.

**Popular datasets to consider:**
- **Steinbichler / Dantec Shearography Reference Dataset (2010)** — digital shearography fringe patterns from CFRP, honeycomb, and bonded structures with calibrated disbonds and delaminations; the most widely used shearography benchmark
- **DLR Composite Shearography Dataset (German Aerospace Center, 2014)** — thermal and vacuum-loaded shearography images of aerospace composite panels with known impact damage and inserts
- **Fraunhofer LBF Tyre Shearography Dataset (2016)** — shearography inspection of vehicle tyres with known internal defects (separations, bubbles); used for automotive NDT validation
- **ISI Shearography Phase Map Dataset (2019)** — unwrapped phase maps from digital shearography of aluminium and composite structures with calibrated flat-bottom holes and teflon inserts
- **NASA Shearography Composite Dataset (2012)** — shearography inspection data from composite overwrapped pressure vessels (COPVs) with induced disbonds

**Decision criteria:** The Steinbichler/Dantec dataset is the most widely cited for shearography algorithm validation; DLR provides aerospace composite benchmarks. Use the dataset that appears in the largest number of shearography defect detection and phase analysis papers.

#### Step 2: List All Shearography Algorithms

Please first ensure all the shearography algorithms have been listed in `\pwm\public\algorithm_base\shearography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/shearography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the shearography solvers, please update the shearography solver.

**Key algorithms to cover (1950–2026):**

_Classical Interferometric Processing (1970s–2005):_
- Temporal phase stepping (TPS) — N-step phase shift algorithm for quantitative phase extraction (Creath, 1985; Steinchen & Yang, 2003)
- Spatial phase shifting — single-frame phase extraction using carrier fringes (Takeda et al., 1982; adapted for shearography)
- Fourier transform method (FTM) — frequency-domain phase extraction from single fringe pattern (Takeda et al., 1982)
- Phase unwrapping — Goldstein branch-cut algorithm for 2-pi ambiguity removal (Goldstein et al., 1988)
- Quality-guided phase unwrapping — reliability-driven path-following unwrapping (Bone, 1991)
- Least-squares phase unwrapping — global optimisation for robust unwrapping (Ghiglia & Romero, 1994)
- Stacking / temporal averaging — multi-frame averaging for speckle noise reduction (Bhaduri et al., 2007)
- Derivative-to-displacement integration — numerical integration of shearographic phase derivatives to obtain displacement fields (Hung, 1982)

_Advanced Processing (2005–2017):_
- Windowed Fourier transform (WFT) — localised frequency analysis for fringe pattern analysis (Kemao, 2004)
- Wavelet transform phase extraction — multi-scale fringe analysis for shearography (Watkins et al., 2005)
- Empirical mode decomposition (EMD) — adaptive decomposition of fringe patterns (Bernini et al., 2009)
- Hilbert transform for single-frame phase — analytic signal approach to phase extraction (Larkin et al., 2001)
- Regularised phase tracking (RPT) — demodulation with regularisation for complex fringe patterns (Servin et al., 1999)
- Digital image correlation (DIC) enhanced shearography — hybrid displacement-strain measurement (Francis et al., 2010)
- Phase derivative variance (PDV) — automated defect detection from shearographic phase noise statistics (Groves et al., 2006)
- Speckle noise filtering — Lee, Frost, and median filtering adapted for shearographic speckle (Lee, 1980; Groves et al., 2004)

_Deep Learning (2017–2026):_
- CNN for shearography defect detection (Kurita et al., 2019) — classification of defective vs. sound regions from phase maps
- U-Net for shearography phase map segmentation (De Angelis et al., 2021) — pixel-wise defect segmentation from unwrapped phase maps
- Deep learning phase unwrapping — PhaseNet for single-step 2D phase unwrapping (Spoorthi et al., 2019) — eliminates classical unwrapping errors
- GAN for speckle denoising in shearography (Zhou et al., 2021) — generative adversarial network for speckle suppression
- CNN-based automated excitation optimisation (2022) — learned optimal thermal/vacuum loading parameters
- Physics-informed neural network for strain field reconstruction (2023) — PINN for displacement field recovery from shearographic phase derivatives
- Vision Transformer for defect classification in shearography (2024) — ViT-based multi-class defect identification
- Self-supervised shearography feature learning (2024) — contrastive pre-training on unlabelled shearographic fringe patterns
- Foundation model for shearography — multi-task pre-trained model for phase unwrapping, defect detection, and sizing (2025)

#### Step 3: Update Shearography Solvers

After listing all shearography solvers, update `algorithm_base/shearography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All shearography solvers use the data format: `y` (num_frames, H, W) phase-stepped interferogram sequence or (H, W) single fringe pattern, `shear_amount` float lateral shearing distance in pixels, `wavelength` float laser wavelength in nm. The `ShearographyOperator` handles forward speckle interferometry simulation, phase derivative calculation, and fringe pattern generation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Shearography:**
- Steinbichler disbond detection: PDV ~78% PoD, CNN ~90% PoD, U-Net ~94% PoD
- Phase unwrapping accuracy: Goldstein ~92% correct pixels, Quality-guided ~96%, PhaseNet ~99.2%
- DLR composite defect sizing: Classical integration ~20% sizing error, PINN ~8% sizing error
- Fraunhofer tyre defect detection: Manual inspection ~85% PoD, ViT ~95% PoD
- All reference values from published papers and NDT round-robin results

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 5% PoD / 3% phase accuracy)
- `partial` — 3–10 dB shortfall (or 5–15% accuracy gap)
- `gap` — >10 dB shortfall (or >15% accuracy gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'shearography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/shearography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/shearography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/shearography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for shearography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/shearography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/shearography/standard/`

---

### Terahertz Imaging (`terahertz`) Modality Template

#### Step 1: Verify Standard Dataset

For Terahertz Imaging, what dataset do you use to verify? Is this dataset used for terahertz popular algorithms? Please ensure the standard dataset in `datasets/benchmark/terahertz/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original terahertz standard dataset.

**Popular datasets to consider:**
- **Terahertz NDT Composite Dataset (Ospald et al., 2015)** — THz time-domain imaging of GFRP and foam-core sandwich panels with calibrated impact damage and disbonds; the most cited THz-NDT benchmark
- **THz Spectral Database (Naftaly et al., THz-Bridge, 2005)** — reference THz time-domain spectroscopy data for common materials with extracted optical constants; standard for THz material characterisation
- **DTU THz Imaging Dataset (Technical University of Denmark, 2018)** — raster-scanned THz images of pharmaceutical tablets and polymer samples with known internal structures; used for THz image reconstruction benchmarks
- **Fraunhofer HHI THz Security Imaging Dataset (2016)** — THz reflection images of concealed objects; used for THz image reconstruction and threat detection
- **Hamamatsu THz Painting Inspection Dataset (2019)** — THz reflection imaging of layered paint and coating samples with calibrated layer thicknesses; cultural heritage and automotive applications

**Decision criteria:** The Ospald composite dataset is the canonical THz-NDT benchmark; DTU provides algorithmic reconstruction validation. Use the dataset that appears in the largest number of THz imaging and reconstruction papers.

#### Step 2: List All Terahertz Algorithms

Please first ensure all the terahertz algorithms have been listed in `\pwm\public\algorithm_base\terahertz\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/terahertz. Besides, you need to search all algorithms from 1950 to 2026. After listing all the terahertz solvers, please update the terahertz solver.

**Key algorithms to cover (1950–2026):**

_Classical THz Signal Processing (1990s–2010):_
- THz time-domain analysis — direct pulse amplitude and time-of-flight extraction from THz-TDS waveforms (Hu & Nuss, 1995)
- THz spectral extraction — FFT-based extraction of amplitude and phase spectra from THz time-domain signals (Grischkowsky et al., 1990)
- Transfer function method — material parameter extraction (refractive index, absorption coefficient) from transmission THz-TDS (Duvillaret et al., 1996)
- Deconvolution for THz pulse shaping — reference-based deconvolution of THz time-domain waveforms for improved temporal resolution (Withayachumnankul & Naftaly, 2014)
- THz-TDS tomography — time-of-flight based cross-sectional imaging (Mittleman et al., 1997)
- Fresnel coefficient analysis — layer thickness and refractive index estimation from THz reflection data (Jepsen et al., 2007)
- Kramers-Kronig analysis — extraction of complex dielectric properties from THz reflection spectroscopy
- Binary lens / diffractive THz imaging — computational focusing for THz beam-scanned systems

_Advanced Processing (2008–2018):_
- Sparse deconvolution for THz depth profiling — L1-regularised deconvolution for resolving closely-spaced layers (Chen et al., 2010)
- Wavelet-based THz signal denoising — multi-scale denoising for low-SNR THz signals (Ferguson & Abbott, 2001)
- PCA for THz spectral imaging — dimensionality reduction for multi-spectral THz image classification (Shen, 2011)
- Matched filter for THz defect detection — optimal linear filter for specific defect signature detection in THz images
- Compressed sensing THz imaging — sub-Nyquist sampling with sparsity-based reconstruction for accelerated THz imaging (Chan et al., 2008)
- CLEAN algorithm for THz deconvolution — iterative deconvolution borrowed from radio astronomy for THz pulse analysis (Naftaly, 2015)
- Gaussian mixture model for THz material classification — probabilistic clustering of THz spectral features
- Support vector machine for THz-based material identification — SVM classification using THz spectral features (Kawase et al., 2003)

_Deep Learning (2017–2026):_
- CNN for THz image classification (Chen et al., 2019) — convolutional network for defect/material classification from THz C-scans
- U-Net for THz image segmentation (Valdes et al., 2021) — pixel-wise defect segmentation in THz images
- 1D-CNN for THz spectral identification (Liu et al., 2020) — waveform-level material classification
- Autoencoder for THz anomaly detection (Fan et al., 2021) — unsupervised defect detection via reconstruction error
- GAN for THz image super-resolution (2022) — generative model for enhancing spatial resolution of raster-scanned THz images
- Recurrent neural network for THz time-series (2021) — LSTM/GRU for THz pulse classification and depth estimation
- Physics-informed neural network for THz material inversion (2023) — PINN for simultaneous refractive index and thickness estimation
- Diffusion model for THz image reconstruction (2024) — score-based generative model for compressed sensing THz imaging
- Transformer for multi-spectral THz fusion (2024) — attention-based fusion across THz frequency channels
- Foundation model for THz imaging — multi-task model for material identification, defect detection, and depth profiling (2025)

#### Step 3: Update Terahertz Solvers

After listing all terahertz solvers, update `algorithm_base/terahertz/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All terahertz solvers use the data format: `y` (H, W, num_time_samples) THz time-domain raster-scan data or (H, W) single-frequency THz image, `time_axis` array of time delays in ps, `reference_pulse` (num_time_samples,) reference THz pulse for deconvolution. The `THzOperator` handles forward THz pulse propagation simulation, material transfer function application, and spectral extraction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Terahertz:**
- Ospald GFRP defect detection: TOF amplitude ~72% PoD, PCA ~83% PoD, U-Net ~91% PoD
- DTU layer thickness estimation: Transfer function ~8% error, Sparse deconvolution ~4% error, PINN ~2% error
- THz material classification (THz-Bridge): SVM ~88% accuracy, 1D-CNN ~94% accuracy, Transformer ~96% accuracy
- CS-THz 10x acceleration: CS ~28 dB PSNR, Diffusion model ~33 dB PSNR
- All reference values from published papers and THz imaging benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 5% accuracy / 3% thickness error)
- `partial` — 3–10 dB shortfall (or 5–15% accuracy gap)
- `gap` — >10 dB shortfall (or >15% accuracy gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'terahertz' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/terahertz/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/terahertz/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/terahertz/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for terahertz. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/terahertz/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/terahertz/standard/`

---

### Ultrasonic Phased Array — TFM/FMC (`ultrasonic_phased_array`) Modality Template

#### Step 1: Verify Standard Dataset

For Ultrasonic Phased Array (TFM/FMC), what dataset do you use to verify? Is this dataset used for phased array popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ultrasonic_phased_array/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ultrasonic phased array standard dataset.

**Popular datasets to consider:**
- **BRAIN (Bristol Research in Array Imaging and NDT) FMC Dataset (2015)** — full matrix capture (FMC) data from calibration blocks with side-drilled holes (SDH) and EDM notches; the most widely cited TFM/FMC benchmark; includes steel and aluminium specimens
- **CEA CIVA Simulation-Validation FMC Dataset (2018)** — CIVA-simulated and experimentally validated FMC data for phased array calibration; includes multi-mode (L-L, L-T, T-T) datasets
- **TWI Weld Inspection FMC Dataset (The Welding Institute, 2017)** — FMC data from austenitic stainless steel and carbon steel weld specimens with calibrated lack-of-fusion, porosity, and crack defects
- **Olympus OmniScan Reference FMC Dataset (2020)** — multi-element (64-element) phased array FMC data from NDT calibration blocks with reference reflectors
- **UKAEA Nuclear Inspection Phased Array Dataset (2019)** — FMC data from thick-section nuclear pressure vessel steels with calibrated flaws; includes anisotropic austenitic weld materials

**Decision criteria:** The BRAIN dataset is the gold standard for TFM algorithm benchmarking; CEA CIVA provides multi-mode validation. Use the dataset that appears in the largest number of TFM/FMC reconstruction papers.

#### Step 2: List All Ultrasonic Phased Array Algorithms

Please first ensure all the ultrasonic phased array algorithms have been listed in `\pwm\public\algorithm_base\ultrasonic_phased_array\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ultrasonic_phased_array. Besides, you need to search all algorithms from 1950 to 2026. After listing all the phased array solvers, please update the phased array solver.

**Key algorithms to cover (1950–2026):**

_Classical Beamforming (1960s–2005):_
- Delay-and-Sum (DAS) beamforming — basic phased array focusing by time-delay compensation (von Ramm & Smith, 1983) — the foundational ultrasonic beamforming algorithm
- Total Focusing Method (TFM) — post-processing of FMC data with all element-pair focusing (Holmes et al., 2005) — the gold standard for FMC imaging
- Synthetic Aperture Focusing Technique (SAFT) — coherent aperture synthesis for single-element scanning (Seydel, 1982; Doctor et al., 1986)
- Phased array sector/linear scan — real-time electronic beam steering and focusing (McNab & Campbell, 1987)
- Half-skip / full-skip TFM — multi-mode imaging using mode-converted paths (L-L, L-T, T-T, T-L) (Brath et al., 2017; roots in Holmes et al., 2005)
- Time reversal focusing — adaptive focusing via time-reversal invariance of the wave equation (Fink, 1992)
- Born approximation imaging — linearised inverse scattering for ultrasonic imaging (Langenberg, 1987)

_Advanced Imaging & Inversion (2005–2018):_
- Plane Wave Imaging (PWI) — fast acquisition using unfocused plane wave transmissions with coherent compounding (Montaldo et al., 2009)
- Coherence-based beamforming — phase coherence factor (PCF) and sign coherence factor (SCF) for sidelobe and noise suppression (Camacho et al., 2009)
- Minimum Variance (MV) beamforming — adaptive beamforming for improved resolution (Synnevag et al., 2007; adapted for NDT 2012+)
- DORT — Decomposition of the Time Reversal Operator for defect detection and characterisation (Prada & Fink, 1994)
- Wavenumber algorithm — frequency-wavenumber domain imaging for fast TFM computation (Hunter et al., 2008)
- Phase shift migration — Stolt/Gazdag migration adapted from seismic to ultrasonic NDT (Skjelvareid et al., 2011)
- Reverse Time Migration (RTM) — full-waveform-based imaging for complex geometries (McKee et al., 2014)
- MUSIC for ultrasonic imaging — multiple signal classification for super-resolution defect localisation (Fan et al., 2014)
- Sparse array design — optimised element placement for grating-lobe suppression (Chio & Schaubert, 2000; NDT 2014+)
- Autoregressive spectral extrapolation — bandwidth extension for improved axial resolution

_Deep Learning (2017–2026):_
- CNN for TFM image defect classification (Medak et al., 2021) — classification of flaw types from TFM images
- U-Net for TFM image segmentation (Cantero-Chinchilla et al., 2022) — pixel-wise defect segmentation in TFM images
- Deep learning beamforming — learned DAS weights for improved image quality (Luijten et al., 2020; adapted for NDT 2022+)
- Physics-informed neural network for FWI — PINN-based full waveform inversion for ultrasonic imaging (Rao et al., 2023)
- GAN for FMC data augmentation (2022) — synthetic FMC data generation for training data expansion
- Compressed sensing FMC — sub-sampled FMC acquisition with sparsity-based TFM reconstruction (Bai et al., 2018)
- Transformer for multi-mode TFM fusion (2024) — attention-based fusion of half-skip TFM modes
- Neural operator for ultrasonic wave propagation (2024) — Fourier neural operator for fast forward simulation
- Foundation model for ultrasonic phased array — multi-task model for imaging, defect detection, sizing, and characterisation (2025)
- Self-supervised FMC representation learning — contrastive pre-training on unlabelled FMC datasets (2025)

#### Step 3: Update Ultrasonic Phased Array Solvers

After listing all ultrasonic phased array solvers, update `algorithm_base/ultrasonic_phased_array/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ultrasonic phased array solvers use the data format: `y` (num_tx, num_rx, num_time_samples) full matrix capture (FMC) data, `element_positions` (num_elements, 2) array element coordinates, `velocity` float longitudinal wave speed in m/s, `dt` float sampling interval in seconds. The `FMCOperator` handles forward ultrasonic wave propagation simulation, FMC data synthesis, and TFM image formation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Ultrasonic Phased Array:**
- BRAIN SDH TFM imaging: DAS ~18.0 dB SNR, TFM ~24.0 dB SNR, MV-TFM ~28.0 dB SNR, DL beamforming ~30.0 dB SNR
- BRAIN lateral resolution: TFM ~0.7 mm, PCF-TFM ~0.4 mm, MV-TFM ~0.3 mm
- TWI weld defect detection: TFM ~82% PoD, Multi-mode TFM ~90% PoD, U-Net ~95% PoD
- TWI defect sizing: TFM 6-dB drop ~25% error, RTM ~12% error, PINN-FWI ~7% error
- Plane wave imaging: PWI (11 angles) ~22.0 dB SNR vs. TFM ~24.0 dB SNR at 10x faster acquisition
- All reference values from published papers and BRAIN benchmark results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ultrasonic_phased_array' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ultrasonic_phased_array/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ultrasonic phased array. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ultrasonic_phased_array/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ultrasonic_phased_array/standard/`

---

### X-ray NDT Radiography (`xray_ndt`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray NDT Radiography, what dataset do you use to verify? Is this dataset used for X-ray NDT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/xray_ndt/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original X-ray NDT standard dataset.

**Popular datasets to consider:**
- **GDXray (GRIMA X-ray, Mery et al., 2015)** — the most widely cited X-ray NDT benchmark; 19,407 radiographic images in 5 categories (castings, welds, baggage, nature, settings) with ground-truth defect annotations; used by virtually all DL-based X-ray NDT papers
- **RIPI Weld Radiograph Dataset (Research Institute of Petroleum Industry, 2019)** — digital radiographs of pipeline welds with 6 defect classes (crack, lack of penetration, lack of fusion, porosity, slag inclusion, undercut); used for weld defect classification benchmarks
- **BAM Reference Radiographs (Federal Institute for Materials Research, 2015)** — calibrated digital radiographs of steel and aluminium test specimens with IQI (Image Quality Indicator) references; used for image quality and sensitivity evaluation
- **Casting X-ray Defect Dataset (Kaggle/MVTec, 2020)** — automotive aluminium casting radiographs with defect/no-defect labels; 7,348 images; widely used for binary classification benchmarks
- **CNDDE Weld Radiography Dataset (Chinese NDT Database, 2021)** — large-scale pipeline weld radiograph dataset with multi-class defect annotations; used for Chinese standard GB/T 3323 compliance

**Decision criteria:** GDXray is the undisputed gold standard for X-ray NDT algorithm benchmarking (2015–2026); RIPI provides multi-class weld defect evaluation. Use the dataset that appears in the largest number of radiographic defect detection papers.

#### Step 2: List All X-ray NDT Algorithms

Please first ensure all the X-ray NDT algorithms have been listed in `\pwm\public\algorithm_base\xray_ndt\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/xray_ndt. Besides, you need to search all algorithms from 1950 to 2026. After listing all the X-ray NDT solvers, please update the X-ray NDT solver.

**Key algorithms to cover (1950–2026):**

_Classical Radiographic Processing (1950s–2005):_
- Film density analysis — optical density measurement and characteristic curve (H&D curve) analysis (Hurter & Driffield, 1890; applied to NDT radiography)
- Contrast enhancement — histogram equalisation and CLAHE for radiographic contrast improvement (Pizer et al., 1987)
- Unsharp masking — high-pass frequency enhancement for radiographic detail visibility (Rosenfeld & Kak, 1982)
- Spatial filtering — low-pass and band-pass filtering for noise reduction in digital radiographs
- Image Quality Indicator (IQI) assessment — wire-type and hole-type IQI sensitivity measurement per ASTM E1025 / EN 462
- Digital radiography calibration — flat-field and dark-field correction, gain normalisation for DR systems
- Geometric magnification correction — magnification and distortion correction for source-object-detector geometry
- Dual-energy radiography — material decomposition using two X-ray energies (Alvarez & Macovski, 1976)

_Feature-Based Detection (2000–2016):_
- Edge detection for crack identification — Canny/Sobel-based crack detection in radiographs (Mery & Arteta, 2002)
- Texture analysis for porosity detection — GLCM and LBP-based texture features for porosity clustering (Liao et al., 2006)
- Region growing for defect segmentation — seed-based segmentation of voids and inclusions
- Hough transform for weld seam detection — automated weld centreline detection in radiographs
- SIFT/SURF-based defect matching — feature matching for defect detection by comparison with reference images
- Random Forest for radiograph classification — ensemble classifier with texture and shape features for multi-class defect identification (Mery & Arteta, 2005)
- Active contour / snake for defect boundary extraction — deformable models for precise defect contour delineation (Kass et al., 1988)
- Bag of Visual Words for radiograph classification — codebook-based classification of casting and weld defects

_Deep Learning (2016–2026):_
- CNN for casting defect classification (Ferguson et al., 2017) — transfer learning from ImageNet for radiographic defect classification
- Faster R-CNN for weld defect detection (Yang et al., 2019) — two-stage detector for multi-class weld defect localisation
- YOLOv3/v5/v8 for radiograph defect detection — single-shot real-time defect detection in industrial radiographs (applied 2019–2024)
- U-Net for defect segmentation in radiographs (Du et al., 2020) — pixel-wise defect segmentation
- Mask R-CNN for instance-level defect segmentation (Mery & Arteta, 2021) — instance segmentation of multiple defects
- ResNet/EfficientNet for defect classification (2020) — deep classification of radiographic defect types
- GAN for defect data augmentation (Niu et al., 2021) — synthetic defect generation for imbalanced dataset handling
- CycleGAN for film-to-digital radiograph translation (2022) — style transfer between analog film and digital radiography
- Anomaly detection for radiograph inspection — PatchCore / STFPM adapted for radiographic anomaly detection (2023)
- Vision Transformer for radiograph analysis (2023) — ViT-based multi-class defect detection
- Foundation model for X-ray NDT — multi-task model for detection, segmentation, classification, and accept/reject decision (2025)
- Multimodal LLM for radiograph interpretation — VLM-based natural-language defect reporting from radiographs (2025)

#### Step 3: Update X-ray NDT Solvers

After listing all X-ray NDT solvers, update `algorithm_base/xray_ndt/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All X-ray NDT solvers use the data format: `y` (H, W) grayscale digital radiograph or (H, W, num_energies) multi-energy radiograph, `exposure_params` dict containing kV, mA, and exposure time, `geometry` dict containing source-object and object-detector distances. The `RadiographyOperator` handles forward X-ray attenuation simulation (Beer-Lambert law) and image formation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for X-ray NDT:**
- GDXray castings defect detection: Edge detection ~65% mAP, Faster R-CNN ~82% mAP, YOLOv8 ~88% mAP
- GDXray welds defect detection: Texture analysis ~70% mAP, U-Net ~85% mAP, Mask R-CNN ~89% mAP
- RIPI weld classification (6-class): Random Forest ~78% accuracy, ResNet ~91% accuracy, ViT ~94% accuracy
- Casting defect binary classification: SVM ~85% accuracy, EfficientNet ~97% accuracy
- All reference values from published papers and GDXray benchmark results

**Verification criteria:**
- `done` — PWM within 3% mAP / accuracy of reference
- `partial` — 3–10% shortfall
- `gap` — >10% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'xray_ndt' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/xray_ndt/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for X-ray NDT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/xray_ndt/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/xray_ndt/standard/`

---

### X-ray Fluorescence Imaging (`xrf_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For X-ray Fluorescence (XRF) Imaging, what dataset do you use to verify? Is this dataset used for XRF popular algorithms? Please ensure the standard dataset in `datasets/benchmark/xrf_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original XRF imaging standard dataset.

**Popular datasets to consider:**
- **NIST XRF Reference Spectra Database (SRD 14, 2009)** — calibrated XRF spectra of pure elements and standard reference materials; the canonical spectral reference for XRF quantification algorithms
- **Bruker M6 Jetstream XRF Scanning Dataset (2016)** — macro-XRF (MA-XRF) elemental maps of paintings and cultural heritage objects; widely used for XRF imaging algorithm benchmarking
- **IAEA XRF Proficiency Test Dataset (2015)** — inter-laboratory comparison XRF spectra from certified reference materials with known elemental concentrations; used for quantification algorithm validation
- **Synchrotron XRF Microprobe Dataset (ESRF ID21, 2018)** — micro-XRF maps from synchrotron beamlines with sub-micron resolution; includes elemental distribution maps with fluorescence yield ground truth
- **Portable XRF (pXRF) Soil Survey Dataset (US EPA, 2020)** — field pXRF measurements of contaminated soil samples with ICP-MS cross-validation; used for quantification and matrix correction algorithm benchmarking

**Decision criteria:** The NIST SRD 14 is the gold standard for spectral reference and quantification validation; Bruker M6 provides imaging-mode benchmarks. Use the dataset that appears in the largest number of XRF quantification and imaging papers.

#### Step 2: List All XRF Imaging Algorithms

Please first ensure all the XRF imaging algorithms have been listed in `\pwm\public\algorithm_base\xrf_imaging\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/xrf_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the XRF imaging solvers, please update the XRF imaging solver.

**Key algorithms to cover (1950–2026):**

_Classical XRF Spectral Processing (1960s–2005):_
- Gaussian peak fitting — least-squares fitting of Gaussian functions to XRF spectral peaks for elemental identification (Markowicz, 1993)
- Background subtraction — SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) algorithm for spectral background removal (Ryan et al., 1988)
- Fundamental Parameters (FP) method — physics-based quantification using theoretical fluorescence yields and attenuation coefficients (Sherman, 1955; Shiraiwa & Fujino, 1966)
- Empirical calibration / influence coefficient method — matrix-corrected quantification using calibration standards (de Jongh, 1973)
- Net peak area extraction — trapezoidal or peak fitting-based extraction of characteristic X-ray line intensities
- Matrix correction (ZAF) — corrections for atomic number, absorption, and fluorescence effects (Philibert, 1963)
- Escape peak and sum peak correction — correction for detector artefacts in Si(Li) and SDD detectors
- Deconvolution of overlapping peaks — spectral deconvolution for closely-spaced fluorescence lines (Van Espen et al., 1977)

_Advanced Processing & Imaging (2000–2018):_
- PyMCA spectral fitting — multi-peak fitting and quantification framework (Sole et al., 2007) — the most widely used open-source XRF processing software
- AXIL — Analysis of X-ray spectra by Iterative Least-squares fitting (Van Espen et al., 1991)
- PCA for XRF spectral imaging — principal component analysis of multi-element XRF maps for component separation (Alfeld & Janssens, 2015)
- Non-negative matrix factorisation (NMF) for XRF — spectral unmixing of XRF image data cubes (de Viguerie et al., 2018)
- Monte Carlo simulation for XRF — stochastic simulation of X-ray interactions for quantification (Vincze et al., 1993; XMI-MSIM)
- Maximum likelihood spectral fitting — Poisson-noise-aware spectral fitting (Scholze & Procop, 2009)
- Spectral deconvolution with detector response function — full detector model-based spectral analysis
- Dynamic analysis for MA-XRF — real-time elemental map generation during scanning (Alfeld et al., 2013)
- Region of Interest (ROI) mapping — energy-windowed integration for fast elemental mapping

_Deep Learning (2018–2026):_
- CNN for XRF spectrum classification (Panchuk et al., 2018) — convolutional network for material/alloy identification from XRF spectra
- 1D-CNN for XRF peak identification (Figueroa et al., 2021) — automated element identification in complex spectra
- U-Net for XRF map super-resolution (2022) — spatial resolution enhancement of MA-XRF elemental maps
- Autoencoder for XRF spectral denoising (Chen et al., 2021) — learned denoising for low-count XRF spectra
- GAN for XRF spectral augmentation (2022) — synthetic spectral generation for training data expansion
- Neural network quantification — DNN replacing FP method for matrix-independent quantification (2022)
- Physics-informed neural network for XRF self-absorption correction (2023) — PINN for attenuation correction in thick samples
- Transformer for multi-element spectral analysis (2024) — attention-based simultaneous multi-peak fitting
- Foundation model for XRF — multi-task model for element identification, quantification, and spatial mapping (2025)

#### Step 3: Update XRF Imaging Solvers

After listing all XRF imaging solvers, update `algorithm_base/xrf_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All XRF imaging solvers use the data format: `y` (H, W, num_channels) hyperspectral XRF data cube or (num_channels,) single-point XRF spectrum, `energy_axis` array of energy bin centres in keV, `excitation_params` dict containing tube voltage, current, and filter. The `XRFOperator` handles forward fluorescence simulation (fundamental parameters model), spectral generation, and elemental map extraction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for XRF Imaging:**
- NIST SRM quantification (major elements): FP method ~5% relative error, PyMCA ~3% error, DNN quantification ~1.5% error
- NIST SRM quantification (trace elements): FP method ~15% error, Monte Carlo ~8% error, PINN ~5% error
- Bruker M6 elemental map segmentation: ROI mapping ~70% IoU, NMF ~82% IoU, U-Net ~90% IoU
- XRF spectral element identification: Gaussian fitting ~90% recall, CNN ~96% recall, Transformer ~98% recall
- All reference values from published papers and IAEA proficiency test results

**Verification criteria:**
- `done` — PWM within 3% relative quantification error of reference
- `partial` — 3–10% quantification gap
- `gap` — >10% quantification gap
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'xrf_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/xrf_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for XRF imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/xrf_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/xrf_imaging/standard/`

---

### Electrical Impedance Tomography (`impedance_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For Electrical Impedance Tomography (EIT), what dataset do you use to verify? Is this dataset used for EIT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/impedance_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original EIT standard dataset.

**Popular datasets to consider:**
- **EIDORS Benchmark Problems (Adler et al., 2009)** — the canonical EIT simulation and reconstruction benchmark; includes 2D and 3D forward models with standardised test phantoms; used by the majority of EIT reconstruction papers
- **KIT4 Tank Experiment Dataset (Hauptmann et al., 2017)** — real EIT measurement data from a cylindrical saline tank with known conductive and resistive inclusions; the most widely used experimental EIT benchmark
- **ACT3/ACT4 Clinical EIT Dataset (Rensselaer Polytechnic Institute, 2002)** — multi-frequency EIT data from human thorax with simultaneous CT ground truth; used for clinical EIT algorithm validation
- **Finland EIT Phantom Dataset (Kuopio, Vauhkonen et al., 2013)** — high-precision EIT tank measurements with calibrated circular and irregular inclusions; used for absolute imaging algorithm benchmarks
- **GREIT Evaluation Dataset (Adler et al., 2009)** — standardised test images for evaluating EIT reconstruction quality using GREIT figures of merit (amplitude response, position error, ringing, resolution)

**Decision criteria:** The EIDORS benchmark problems are the canonical numerical benchmark; KIT4 is the gold standard for experimental validation. Use the dataset that appears in the largest number of EIT reconstruction papers.

#### Step 2: List All EIT Algorithms

Please first ensure all the EIT algorithms have been listed in `\pwm\public\algorithm_base\impedance_tomo\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/impedance_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the EIT solvers, please update the EIT solver.

**Key algorithms to cover (1950–2026):**

_Classical / Linear Methods (1970s–2005):_
- Linear Back Projection (LBP) — simple weighted back projection of voltage measurements (Barber & Brown, 1984) — the foundational EIT reconstruction method
- Sheffield Back Projection — filtered back projection using sensitivity maps (Barber & Seagar, 1987)
- NOSER — Newton's One-Step Error Reconstructor (Cheney et al., 1990) — single linearisation step with Jacobian
- Tikhonov regularised linear reconstruction — J^T(JJ^T + lambda*I)^-1 delta_V (Vauhkonen et al., 1998) — the most common EIT reconstruction baseline
- GREIT — Graz consensus Reconstruction algorithm for EIT (Adler et al., 2009) — standardised linear reconstruction with controlled spatial response
- Sensitivity coefficient method — linearised reconstruction using element sensitivity distributions
- Adjacent / opposite drive patterns — current injection strategies affecting reconstruction quality (Seagar et al., 1987)
- Complete electrode model (CEM) — accurate forward model including electrode geometry and contact impedance (Somersalo et al., 1992)

_Iterative / Nonlinear Methods (1990s–2016):_
- Gauss-Newton iterative method — nonlinear least-squares with Jacobian update for absolute imaging (Yorkey et al., 1987)
- Modified Newton-Raphson — regularised iterative nonlinear reconstruction (Woo et al., 1993)
- Total Variation regularised EIT — TV penalty for preserving sharp conductivity boundaries (Borsic et al., 2010)
- L1 regularised EIT — sparsity-promoting reconstruction for localised inclusions (Gehre et al., 2012)
- D-bar method — direct nonlinear reconstruction via scattering transform (Siltanen et al., 2000; Mueller & Siltanen, 2012)
- Monotonicity-based reconstruction — shape reconstruction using monotonicity of Neumann-to-Dirichlet map (Harrach & Ullrich, 2013)
- Level set method for EIT — implicit interface tracking for shape-based reconstruction (Dorn & Lesselier, 2006)
- Bayesian EIT — statistical inversion with prior models for uncertainty quantification (Kaipio et al., 2000)
- Kalman filter EIT — temporal regularisation for real-time dynamic EIT (Adler et al., 2011)
- ADMM for constrained EIT — alternating direction method for inequality-constrained conductivity recovery

_Deep Learning (2017–2026):_
- CNN for EIT image reconstruction (Michalikova et al., 2014; Hamilton & Hauptmann, 2018) — direct voltage-to-image mapping
- U-Net for EIT reconstruction (Li et al., 2020) — encoder-decoder architecture for EIT image formation
- Learned D-bar — deep learning-enhanced D-bar reconstruction (Hamilton & Hauptmann, 2018) — hybrid model-DL approach
- Conditional GAN for EIT (Chen et al., 2020) — cGAN for high-resolution EIT reconstruction from sparse measurements
- Physics-informed neural network for EIT (2022) — PINN solving the conductivity equation for absolute EIT imaging
- Neural operator for EIT — Fourier neural operator mapping voltages to conductivity fields (2023)
- Unrolled Gauss-Newton network for EIT (2023) — algorithm unrolling with learned regularisation
- Transformer for multi-frequency EIT fusion (2024) — attention-based reconstruction from multi-frequency impedance data
- Foundation model for EIT — multi-task model for reconstruction, segmentation, and process monitoring (2025)
- Real-time EIT with edge AI — lightweight neural network for embedded real-time EIT reconstruction (2025)

#### Step 3: Update EIT Solvers

After listing all EIT solvers, update `algorithm_base/impedance_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All EIT solvers use the data format: `y` (num_measurements,) voltage measurements or (num_measurements, num_frequencies) multi-frequency data, `electrode_positions` (num_electrodes, 2) or (num_electrodes, 3) electrode coordinates, `drive_pattern` (num_measurements, num_electrodes) current injection matrix. The `EITOperator` handles forward FEM-based voltage computation using the complete electrode model and Jacobian calculation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for EIT:**
- EIDORS 2D circular phantom: LBP ~12.0 dB, Tikhonov ~18.0 dB, Gauss-Newton ~22.0 dB, U-Net ~26.0 dB
- KIT4 single inclusion: GREIT AR ~0.85, Gauss-Newton AR ~0.92, Learned D-bar AR ~0.95
- KIT4 position error: GREIT ~5.0% diameter, TV-EIT ~3.2%, CNN ~2.1%
- Dynamic EIT (ventilation monitoring): Kalman filter ~15 dB, Real-time neural ~20 dB
- D-bar reconstruction: classical ~16.0 dB, Learned D-bar ~22.0 dB
- All reference values from published papers and EIDORS/GREIT benchmark results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'impedance_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/impedance_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for EIT. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/impedance_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/impedance_tomo/standard/`

---

### Generic Matrix Sensing (`matrix`) Modality Template

#### Step 1: Verify Standard Dataset

For Generic Matrix Sensing, what dataset do you use to verify? Is this dataset used for matrix sensing popular algorithms? Please ensure the standard dataset in `datasets/benchmark/matrix/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original matrix standard dataset.

**Popular datasets to consider:**
- **MovieLens 100K/1M/10M (GroupLens, Harper & Konstan, 2015)** — the canonical matrix completion benchmark; user-item rating matrices of various sizes; used by virtually all matrix completion and recommender system papers
- **Netflix Prize Dataset (Netflix, 2006)** — 100M ratings from 480K users on 17K movies; the most famous large-scale matrix completion challenge dataset
- **Jester Online Joke Dataset (Goldberg et al., 2001)** — dense user-joke rating matrix (73K users, 100 jokes); used for matrix completion algorithms requiring high fill rates
- **Matrix Completion Synthetic Benchmark (Candes & Recht, 2009)** — synthetic low-rank matrices with random Gaussian entries and known rank; the standard theoretical benchmark for nuclear norm minimisation
- **ImageNet Image Matrix Dataset (Dong et al., 2014)** — images represented as low-rank matrices for compressive sensing and matrix recovery benchmarks

**Decision criteria:** MovieLens is the undisputed gold standard for matrix completion benchmarking; the Candes-Recht synthetic benchmark provides controlled theoretical evaluation. Use the dataset that appears in the largest number of matrix completion and sensing papers.

#### Step 2: List All Matrix Sensing Algorithms

Please first ensure all the matrix sensing algorithms have been listed in `\pwm\public\algorithm_base\matrix\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/matrix. Besides, you need to search all algorithms from 1950 to 2026. After listing all the matrix sensing solvers, please update the matrix sensing solver.

**Key algorithms to cover (1950–2026):**

_Classical Linear Algebra (1950s–2000):_
- Singular Value Decomposition (SVD) — exact low-rank decomposition (Golub & Kahan, 1965) — the foundational matrix factorisation method
- Truncated SVD — rank-k approximation by keeping top-k singular values (Eckart & Young, 1936)
- QR decomposition — orthogonal factorisation for least-squares problems (Householder, 1958)
- LU decomposition — triangular factorisation for linear system solving (Crout, 1941)
- Pseudoinverse (Moore-Penrose) — minimum-norm least-squares solution (Moore, 1920; Penrose, 1955)
- Power iteration / Lanczos — iterative methods for dominant singular value computation (Lanczos, 1950)
- Randomised SVD — random projection-based fast approximate SVD (Halko, Martinsson & Tropp, 2011; roots in 2006)
- Nyström approximation — low-rank kernel matrix approximation via column sampling (Nyström, 1930; Williams & Seeger, 2001)

_Matrix Completion & Recovery (2006–2016):_
- Nuclear Norm Minimisation — convex relaxation for rank minimisation (Candes & Recht, 2009; Recht, Fazel & Parrilo, 2010) — the foundational matrix completion theory
- Singular Value Thresholding (SVT) — proximal gradient for nuclear norm minimisation (Cai & Osher, 2010)
- Alternating Least Squares (ALS) — bilinear factorisation-based matrix completion (Koren et al., 2009) — standard for recommender systems
- Augmented Lagrange Multiplier (ALM) / ADMM for matrix recovery — robust PCA via L+S decomposition (Wright et al., 2009; Lin et al., 2010)
- Robust PCA — decomposition into low-rank + sparse components via nuclear + L1 norm (Candes et al., 2011)
- OptSpace — Riemannian optimisation on Grassmannian for matrix completion (Keshavan et al., 2010)
- GROUSE — Grassmannian Rank-One Update Subspace Estimation for streaming matrix completion (Balzano et al., 2010)
- Soft-Impute — iterative soft-thresholded SVD for matrix completion (Mazumder et al., 2010)
- Matrix factorisation with side information — inductive matrix completion using feature matrices (Jain & Dhillon, 2013)
- Bayesian matrix factorisation — probabilistic matrix completion with uncertainty (Salakhutdinov & Mnih, 2008)

_Deep Learning & Neural Methods (2016–2026):_
- Autoencoder for matrix completion — deep autoencoder for collaborative filtering (Sedhain et al., AutoRec, 2015)
- Deep Matrix Factorisation (DMF) — multi-layer neural network factorisation (Trigeorgis et al., 2017)
- Graph Neural Network for matrix completion — GNN exploiting row/column graph structure (van den Berg et al., 2017)
- Neural Collaborative Filtering (NCF) — neural network replacing inner product in matrix factorisation (He et al., WWW 2017)
- Variational Autoencoder for matrix completion — VAE with Gaussian likelihood for rating prediction (Liang et al., 2018)
- Deep unrolled ADMM for robust PCA — algorithm unrolling for L+S decomposition (Solomon et al., 2019)
- Implicit neural representation for matrix — INR-based continuous matrix completion (2022)
- Transformer for matrix completion — self-attention for capturing long-range row-column dependencies (2023)
- Diffusion model for matrix recovery — score-based generative model for matrix inpainting (2024)
- Foundation model for matrix sensing — multi-task model for completion, denoising, and decomposition (2025)

#### Step 3: Update Matrix Sensing Solvers

After listing all matrix sensing solvers, update `algorithm_base/matrix/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All matrix sensing solvers use the data format: `y` (M, N) partially observed matrix with NaN for missing entries or (num_observations,) vectorised observed values, `mask` (M, N) boolean observation mask, `rank` int estimated rank (if known). The `MatrixOperator` handles forward observation (masking/linear measurement) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Matrix Sensing:**
- MovieLens 1M (RMSE): ALS ~0.860, SVT ~0.850, NCF ~0.830, Transformer ~0.810
- Candes-Recht synthetic (rank-10, 50% observed): SVT ~1e-6 relative error, Nuclear Norm ~1e-7, Deep unrolled ADMM ~1e-8
- Netflix Prize: ALS ~0.900 RMSE, Bayesian MF ~0.880, GNN ~0.860
- Robust PCA (synthetic L+S): ALM ~30 dB, Deep unrolled ADMM ~36 dB
- Image matrix completion (30% observed): Soft-Impute ~28 dB PSNR, INR ~33 dB PSNR, Diffusion ~35 dB PSNR
- All reference values from published papers and benchmark challenge results

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 0.02 RMSE)
- `partial` — 3–10 dB shortfall (or 0.02–0.05 RMSE gap)
- `gap` — >10 dB shortfall (or >0.05 RMSE gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'matrix' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/matrix/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/matrix/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/matrix/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for matrix sensing. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/matrix/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/matrix/standard/`

---
