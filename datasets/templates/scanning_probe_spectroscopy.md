
---

## Scanning Probe Microscopy -- Modality Templates

---

### AFM (`afm`) Modality Template

#### Step 1: Verify Standard Dataset

For AFM, what dataset do you use to verify? Is this dataset used for AFM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/afm/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original AFM standard dataset.

**Popular datasets to consider:**
- **NIST AFM Reference Standards (Vorburger et al., 2005)** -- calibrated step height and pitch standards (SRM 2059, SRM 2090); the primary metrological reference for AFM dimensional accuracy; used universally for tip calibration and scanner linearization
- **Gwyddion Sample Files (Necas & Klapetek, 2012)** -- bundled SPM data in multiple formats (Bruker, Asylum, Park, Nanoscope); widely used for testing image processing algorithms; includes calibration gratings and biological samples
- **Bruker Application Note Datasets** -- application-specific AFM images (polymers, semiconductors, biological membranes) with known feature dimensions; used for algorithm demonstrations
- **Silicon Calibration Grating Datasets (TGZ series, NT-MDT)** -- step height and pitch gratings with NIST-traceable calibration; used for z-calibration and tip shape estimation benchmarks
- **AFM Biological Test Images (Muller et al., 2009)** -- membrane protein arrays (bacteriorhodopsin, aquaporin) with known lattice constants; standard for biological AFM resolution benchmarks
- **Open-source AFM Datasets (Sokolov et al., 2020)** -- force-distance curve datasets on calibrated polymer samples; used for nanomechanical property extraction benchmarks
- **Sparse AFM Benchmarks (Belianinov et al., 2016)** -- datasets with known ground truth for evaluating compressed sensing and GP-regression AFM approaches

**Decision criteria:** NIST reference standards are the canonical metrology benchmark. Gwyddion samples for cross-format compatibility. Sparse AFM benchmarks for evaluating modern reconstruction algorithms. Use the dataset that appears in the largest number of AFM image processing and reconstruction papers.

#### Step 2: List All AFM Algorithms

Please first ensure all the AFM algorithms have been listed in `\Physics_World_Model\algorithm_base\afm\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/afm. Besides, you need to search all algorithms from 1986 to 2026. After listing all the AFM solvers, please update the AFM solver.

**Key algorithms to cover (1986--2026):**

_Classical Image Processing (1986--2009):_
- Plane leveling -- first-order polynomial subtraction to remove sample tilt (standard in all SPM software since the 1980s)
- Polynomial background subtraction -- second and third-order polynomial fitting for bow and curvature removal
- Line-by-line correction (median/mean alignment) -- corrects scan-line artifacts from feedback drift; standard preprocessing step in every AFM package
- Three-point plane fit -- determines sample tilt from three user-selected points (Horcas et al., RSI 2007)
- Row alignment by matching histograms (Necas & Klapetek, Gwyddion 2012)
- Scar removal / line defect correction -- removes horizontal streaks from tip contamination or feedback errors
- Median filtering and Gaussian smoothing -- noise reduction for topography images
- FFT-based filtering -- low-pass, high-pass, band-pass for periodic noise removal (e.g., electrical interference)
- Step detection (edge detection) for calibration gratings (ISO 5436-1)
- Roughness analysis (Ra, Rq, Rz) per ISO 4287 -- standard quantitative metric for AFM surfaces

_Tip Characterization & Deconvolution (1993--2010):_
- Blind tip estimation -- reconstruct tip shape from image of unknown sample (Villarrubia, JRNIST 1997) -- the foundational tip deconvolution algorithm
- Tip deconvolution / erosion -- remove tip convolution artifacts using known or estimated tip geometry (Villarrubia, Surf Sci 1994)
- Dilation-erosion morphological operations for tip-sample interaction (Williams et al., 1996)
- Tip characterization using calibration gratings (TipCheck, Bykov et al., 2000)

_Advanced Classical & Optimization (2010--2016):_
- Drift correction via correlation -- correct thermal/piezo drift between scan lines using cross-correlation (Rahe et al., 2010)
- Creep and hysteresis correction -- compensate nonlinear piezo scanner behavior (Croft et al., 2001; improved 2012)
- Lateral force calibration -- wedge method for friction force microscopy (Ogletree et al., 1996; refined 2013)
- Force curve analysis -- contact point detection, Young's modulus extraction via Hertz/DMT/JKR models (Butt et al., Surf Sci Rep 2005)
- QNM (Quantitative Nanomechanical Mapping) analysis (Derjaguin et al. model, Pittenger et al., 2012)
- Gaussian Process regression for sparse AFM (Belianinov et al., ACS Nano 2016) -- reconstruct full image from sparse scan lines using GP prior; key sparse-sampling paper

_Deep Learning (2017--2026):_
- CNN-based AFM image denoising (Alldritt et al., Science Advances 2020) -- deep learning denoising of AFM topography; trained on simulated + real AFM data
- Deep learning tip artifact removal (2020)
- Super-resolution AFM via deep learning (Rashidi & Wolkow, Mach Learn Sci Technol 2022) -- reconstruct high-resolution AFM from low-resolution scans
- Neural network for automated force curve classification (2019)
- DeepSPM -- reinforcement learning for automated STM/AFM operation (Krull et al., Commun Phys 2020)
- GAN-based AFM image enhancement (2021)
- Physics-informed neural network for AFM cantilever dynamics (2023)
- Diffusion-model AFM denoising and super-resolution (2024)
- Transformer-based AFM image segmentation and feature detection (2025)
- Foundation model for SPM image analysis (2025--2026)

#### Step 3: Update AFM Solvers

After listing all AFM solvers, update `algorithm_base/afm/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All AFM solvers use the data format: `y` (H, W) or (N_lines, W) topography image or sparse scan data, `tip_shape` (Ht, Wt) estimated tip geometry for deconvolution. The `AFMOperator` handles forward model (dilation with tip) and adjoint (erosion) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for AFM:**
- NIST calibration grating: step height accuracy within 1% of certified value after plane leveling + line-by-line correction
- Tip deconvolution: feature width reduction >30% on sharp-edge calibration samples (Villarrubia, 1997)
- GP sparse AFM (25% sampling): PSNR ~30 dB on periodic surface, ~25 dB on heterogeneous surface (Belianinov et al., 2016)
- DL denoising: 3--5 dB PSNR improvement over median filtering (Alldritt et al., 2020)
- Super-resolution AFM (4x): PSNR ~28 dB, SSIM ~0.85 on silicon grating images
- Published PSNR/SSIM and dimensional accuracy from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'afm' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/afm/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/afm/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/afm/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for AFM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/afm/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/afm/standard/`

---

### STM (`stm`) Modality Template

#### Step 1: Verify Standard Dataset

For STM, what dataset do you use to verify? Is this dataset used for STM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/stm/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original STM standard dataset.

**Popular datasets to consider:**
- **NIST STM Image Gallery** -- curated STM images of well-characterized surfaces (Si, Au, graphite) with known atomic structure; reference for calibration and processing validation
- **Si(111) 7x7 Reconstruction Datasets** -- the canonical STM benchmark surface; 7x7 unit cell with known adatom positions (Binnig et al., PRL 1983); used universally for STM resolution and calibration verification
- **HOPG (Highly Oriented Pyrolytic Graphite) Datasets** -- atomic-resolution graphite STM images with known 0.246 nm lattice constant; the most common STM calibration sample
- **QPI (Quasiparticle Interference) Datasets (Hoffman et al., Science 2002)** -- Fourier-transform STS maps on cuprate superconductors; standard for QPI analysis algorithm benchmarks
- **Au(111) Herringbone Reconstruction** -- well-known surface reconstruction with 22x sqrt(3) periodicity; used for drift correction and calibration
- **Cu(111) Surface State Datasets** -- standing wave patterns near step edges; Friedel oscillation benchmarks for LDOS mapping
- **Createc/Omicron Open STM Datasets (2015--2023)** -- open-access STM/STS data from various research groups; used for ML training

**Decision criteria:** Si(111) 7x7 is the undisputed gold standard for STM topographic imaging. HOPG for atomic calibration. QPI datasets for spectroscopic analysis. Use the dataset that appears in the largest number of STM image analysis papers.

#### Step 2: List All STM Algorithms

Please first ensure all the STM algorithms have been listed in `\Physics_World_Model\algorithm_base\stm\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/stm. Besides, you need to search all algorithms from 1981 to 2026. After listing all the STM solvers, please update the STM solver.

**Key algorithms to cover (1981--2026):**

_Classical Image Processing (1981--2009):_
- Plane subtraction -- first-order tilt removal for STM topography (standard since the first STM images)
- Polynomial background subtraction -- second/third-order for scanner bow correction
- Line-by-line correction (median difference) -- corrects feedback-loop noise between scan lines
- FFT filtering -- remove periodic electrical noise (50/60 Hz harmonics, mechanical vibrations)
- Scar/streak removal -- correct line defects from tip changes or contamination
- Gaussian and median filtering for noise reduction
- Drift correction via lattice calibration -- use known atomic lattice to correct x/y drift (Horcas et al., RSI 2007)
- Fourier transform lattice analysis -- extract lattice vectors from FFT peaks for crystallographic analysis

_Spectroscopic Analysis (1985--2015):_
- STS dI/dV mapping -- lock-in measurement of differential conductance for local density of states (LDOS) imaging (Stroscio et al., PRL 1986)
- Numerical dI/dV from I-V curves -- finite-difference differentiation of tunneling current spectra
- Normalization of dI/dV spectra -- (dI/dV)/(I/V) normalization for LDOS extraction (Feenstra, Surf Sci 1994)
- QPI analysis -- Fourier transform of dI/dV maps for quasiparticle dispersion (Hoffman et al., Science 2002; Crommie et al., Nature 1993)
- Lawler-Fujita ratio map algorithm for nematic order detection in QPI (Lawler et al., Nature 2010)
- Multi-pass lock-in for inelastic tunneling spectroscopy (IETS)

_Drift Correction & Advanced Processing (2005--2016):_
- Cross-correlation drift correction between up and down scans (Lapshin, RSI 2004)
- Creep compensation via forward/backward scan comparison (2007)
- Atom tracking for thermal drift measurement and correction (2010)
- Tip artifact identification and removal using autocorrelation (2009)
- Lattice-averaging / unit cell averaging for atomic-resolution enhancement (2008)

_Machine Learning & Deep Learning (2017--2026):_
- ML atom finding -- automated atom position detection using random forests and feature engineering (Ziatdinov et al., ACS Nano 2017) -- first ML-based atom detection for STM
- CNN for atom classification in STM images (Ziatdinov et al., 2018)
- Deep learning STM image denoising (Gordon et al., Mach Learn Sci Technol 2021)
- AtomAI framework for STM/AFM analysis (Ziatdinov et al., Nat Mach Intell 2022)
- GAN-based STM image enhancement and super-resolution (2021)
- Automated STS analysis with neural networks (2020)
- Variational autoencoder for STM feature extraction (2022)
- Reinforcement learning for automated STM tip conditioning (2023)
- Transformer-based STM defect detection and classification (2024)
- Diffusion-model STM denoising and inpainting (2024)
- Foundation model for scanning probe microscopy (2025--2026)

#### Step 3: Update STM Solvers

After listing all STM solvers, update `algorithm_base/stm/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All STM solvers use the data format: `y` (H, W) topography image or (N_energies, H, W) dI/dV spectroscopic map, `bias_voltages` (N_energies,) array of tunneling biases for STS data. The `STMOperator` handles the forward model (tunneling current calculation from LDOS) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for STM:**
- Si(111) 7x7: adatom position accuracy <0.02 nm after drift correction + lattice calibration
- HOPG: lattice constant extracted within 2% of known value (0.246 nm) after FFT analysis
- QPI analysis: dispersion relation extraction matching published band structure within 10% energy accuracy
- ML atom finding: detection F1-score >0.95 on Si(111) 7x7 (Ziatdinov et al., 2017)
- DL denoising: 3--5 dB PSNR improvement over Gaussian filtering (Gordon et al., 2021)
- Published PSNR/SSIM and positional accuracy from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'stm' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/stm/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/stm/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/stm/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for STM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/stm/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/stm/standard/`

---

### MFM (`mfm`) Modality Template

#### Step 1: Verify Standard Dataset

For MFM, what dataset do you use to verify? Is this dataset used for MFM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/mfm/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MFM standard dataset.

**Popular datasets to consider:**
- **NIST Magnetic Reference Samples (Porthun et al., 1998; NIST SRM 2171)** -- calibrated magnetic thin-film samples with known domain structure and stray field; the primary metrology reference for MFM quantification
- **MFM Calibration Samples (Kebe & Carl, JAP 2004)** -- patterned Permalloy thin films with known magnetization for tip calibration; used for quantitative MFM benchmarks
- **Hard Disk Drive Media Benchmarks** -- bit-patterned media and longitudinal/perpendicular recording tracks with known written bit patterns; widely used for MFM resolution and sensitivity benchmarks
- **Magnetic Skyrmion Datasets (Yu et al., Nature 2010; 2017--2023)** -- MFM images of skyrmion lattices in chiral magnets; used for evaluating skyrmion detection algorithms
- **NIST/PTB Quantitative MFM Round-Robin Data (Porthun et al., 2002)** -- multi-lab comparison datasets for MFM reproducibility assessment
- **Simulated MFM Reference Images (Rave et al., 1998)** -- micromagnetically simulated stray-field images with known ground truth; standard for algorithm validation

**Decision criteria:** NIST magnetic reference samples for metrology validation. Hard disk benchmarks for resolution assessment. Simulated references with known ground truth for quantitative algorithm validation. Use the dataset that appears in the largest number of MFM quantification papers.

#### Step 2: List All MFM Algorithms

Please first ensure all the MFM algorithms have been listed in `\Physics_World_Model\algorithm_base\mfm\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/mfm. Besides, you need to search all algorithms from 1987 to 2026. After listing all the MFM solvers, please update the MFM solver.

**Key algorithms to cover (1987--2026):**

_Classical & Lift-Mode (1987--2009):_
- Lift-mode MFM -- two-pass technique: first pass for topography, second pass at lift height for magnetic signal (Martin & Wickramasinghe, APL 1987)
- Phase detection MFM -- measure cantilever phase shift proportional to force gradient (standard since early 1990s)
- Frequency modulation MFM (FM-MFM) -- measure frequency shift for improved quantitative sensitivity (Albrecht et al., JAP 1991)
- Plane leveling and line-by-line correction for MFM phase/frequency images
- Topographic cross-talk subtraction -- remove residual topographic signal from lift-mode data

_Tip Deconvolution & Quantification (1996--2015):_
- Tip transfer function calibration -- calibrate MFM tip response using known reference samples (Hug et al., JAP 1998)
- Tip deconvolution in Fourier space -- deconvolve tip point spread function for stray-field quantification (Porthun et al., JAP 1998)
- Point-probe approximation -- model tip as magnetic monopole or dipole for quantitative interpretation (Hartmann, 1999)
- Extended charge model for tip (Lohau et al., JAP 1999)
- Quantitative MFM via calibrated tip (van Schendel et al., JAP 2000) -- the foundational quantitative MFM method
- Tip moment calibration using known current loops or patterned structures (Kebe & Carl, JAP 2004)

_Stray-Field Simulation & Inverse Problems (2000--2016):_
- Stray field simulation from known magnetization -- forward model using Green's function / Fourier-space convolution (Hubert & Schafer, 1998)
- Inverse magnetization from MFM -- recover surface charge/magnetization distribution from measured stray field (Rawlings & Durkan, Nanotechnology 2012)
- Micromagnetic simulation comparison -- OOMMF/MuMax3 computed stray fields compared with MFM data (Donahue & Porter, 1999)
- Transfer function approach for quantitative MFM (Engel-Herbert et al., JAP 2005)
- Constrained deconvolution with Tikhonov regularization for stray field inversion (2010)
- Kelvin probe force microscopy (KPFM) subtraction for electrostatic artifact removal (2012)

_Deep Learning (2017--2026):_
- CNN-based MFM image denoising and enhancement (2022)
- Deep learning domain wall detection and classification from MFM images (2022)
- Physics-informed neural network for MFM stray-field inversion (2023)
- U-Net for MFM tip deconvolution (2023)
- GAN-based MFM super-resolution (2024)
- Automated magnetic domain segmentation via deep learning (2024)
- Diffusion-model MFM reconstruction from sparse scans (2025)
- Transformer-based skyrmion detection and tracking in MFM time series (2025)

#### Step 3: Update MFM Solvers

After listing all MFM solvers, update `algorithm_base/mfm/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MFM solvers use the data format: `y` (H, W) phase/frequency shift image at lift height, `tip_tf` (H, W) tip transfer function in Fourier space, `lift_height` scalar in nm. The `MFMOperator` handles the forward model (magnetization -> stray field -> phase shift via tip transfer function) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MFM:**
- NIST reference sample: stray field magnitude within 15% of calibrated value after tip deconvolution (van Schendel et al., 2000)
- Hard disk benchmark: bit pattern correctly resolved at written bit pitch; domain wall width within 20% of known value
- Simulated reference: inverse magnetization recovery PSNR >25 dB for simple domain patterns
- DL denoising: 2--4 dB PSNR improvement over standard processing
- Published quantitative accuracy and resolution metrics from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'mfm' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/mfm/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/mfm/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/mfm/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MFM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/mfm/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/mfm/standard/`

---

### NSOM (`nsom`) Modality Template

#### Step 1: Verify Standard Dataset

For NSOM, what dataset do you use to verify? Is this dataset used for NSOM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/nsom/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original NSOM standard dataset.

**Popular datasets to consider:**
- **Sub-wavelength Test Pattern Datasets (Betzig & Trautman, Science 1992)** -- nano-fabricated fluorescent patterns with known geometry below the diffraction limit; the original NSOM resolution demonstration; used for validating near-field imaging resolution claims
- **Nano-antenna Near-field Datasets (Novotny & van Hulst, Nat Photonics 2011)** -- near-field maps of plasmonic nanoantennas with known resonance and field distributions; standard for near-field enhancement characterization
- **s-SNOM Reference Datasets (Keilmann & Hillenbrand, Phil Trans R Soc A 2004)** -- scattering-type SNOM images of semiconductors and polymers with known optical contrast; widely used for s-SNOM algorithm validation
- **Nano-FTIR Reference Spectra (Huth et al., Nano Lett 2012)** -- broadband near-field spectroscopic data on known polymer blends; benchmark for nano-FTIR analysis algorithms
- **Plasmonic Waveguide Near-field Maps (2015--2023)** -- measured near-field distributions on photonic structures with FDTD/FEM simulation ground truth
- **Near-field Phonon Polariton Datasets (Dai et al., Science 2014)** -- s-SNOM images of phonon polaritons on h-BN and SiC; used for polariton propagation analysis benchmarks

**Decision criteria:** Sub-wavelength fluorescent patterns for aperture NSOM resolution. s-SNOM semiconductor references for scattering-NSOM. Nano-FTIR polymer spectra for spectroscopic analysis. Use the dataset most widely referenced in NSOM/s-SNOM image analysis papers.

#### Step 2: List All NSOM Algorithms

Please first ensure all the NSOM algorithms have been listed in `\Physics_World_Model\algorithm_base\nsom\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/nsom. Besides, you need to search all algorithms from 1984 to 2026. After listing all the NSOM solvers, please update the NSOM solver.

**Key algorithms to cover (1984--2026):**

_Aperture NSOM (1984--2009):_
- Aperture NSOM imaging -- metal-coated fiber tip with sub-wavelength aperture; collection/illumination/reflection modes (Pohl et al., APL 1984; Betzig & Trautman, Science 1992)
- Topographic artifact subtraction -- remove z-dependent coupling from near-field optical signal (Hecht et al., 1997)
- Shear-force distance regulation -- maintain constant tip-sample gap during NSOM scanning (Betzig et al., 1992)
- Near-field fluorescence lifetime imaging (NSOM-FLIM) (2000)
- Polarization-resolved NSOM (2003)

_Scattering NSOM (s-SNOM) & Signal Extraction (2000--2016):_
- s-SNOM (scattering-type NSOM) -- detect elastically scattered near-field signal from sharp AFM tip (Keilmann & Hillenbrand, Phil Trans R Soc A 2004)
- Pseudo-heterodyne detection -- interferometric demodulation to extract near-field amplitude and phase at harmonics of tapping frequency (Ocelic et al., APL 2006) -- the standard s-SNOM signal extraction method
- Higher-harmonic demodulation -- lock-in detection at n-th harmonic (n >= 2) for background suppression (Knoll & Keilmann, Opt Commun 2000)
- Nano-FTIR spectroscopy -- broadband near-field spectroscopy using Fourier-transform interferometry at the nanoscale (Huth et al., Nano Lett 2012)
- Self-homodyne vs. pseudo-heterodyne signal processing comparison (2010)
- Tip-model deconvolution for s-SNOM -- finite-dipole model and point-dipole model for quantitative near-field extraction (Cvitkovic et al., Opt Express 2007)

_Near-to-Far-Field & Inverse Problems (2005--2016):_
- Near-to-far-field transformation -- compute far-field radiation pattern from measured near-field distribution (Balanis, antenna theory, applied to NSOM 2005)
- Near-field deconvolution -- remove instrumental PSF from aperture NSOM images (Aigouy et al., 2003)
- Inverse dipole model for quantitative permittivity extraction from s-SNOM (Govyadinov et al., Phys Rev B 2013)
- Multilayer substrate model for s-SNOM contrast interpretation (2014)
- Spectral self-referencing for nano-FTIR normalization (2015)

_Deep Learning (2017--2026):_
- CNN-based near-field image denoising for s-SNOM (2023)
- Deep learning for automated polariton wavelength extraction from s-SNOM images (2023)
- Neural network for quantitative permittivity inversion from s-SNOM data (2024)
- Physics-informed neural network for near-field simulation and inversion (2024)
- GAN-based NSOM super-resolution beyond aperture limit (2024)
- Automated phase extraction and unwrapping for s-SNOM (2025)
- Transformer-based nano-FTIR spectral classification (2025)
- Diffusion-model NSOM image reconstruction from sparse scans (2025)

#### Step 3: Update NSOM Solvers

After listing all NSOM solvers, update `algorithm_base/nsom/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All NSOM solvers use the data format: `y` (H, W) near-field optical amplitude/phase image or (N_harmonics, H, W) multi-harmonic demodulated s-SNOM data, `wavelength` excitation wavelength, `tip_model` finite-dipole or point-dipole parameters. The `NSOMOperator` handles the forward model (tip-sample near-field interaction) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for NSOM:**
- Sub-wavelength test patterns: optical resolution <100 nm demonstrated (lambda/5 for visible light)
- s-SNOM on Si/SiO2: near-field contrast ratio matching published values within 15% (Keilmann & Hillenbrand, 2004)
- Nano-FTIR on PMMA/PS blends: spectral peak positions within 5 cm^-1 of FTIR library values
- Polariton wavelength: measured propagation length within 10% of FDTD prediction
- DL denoising: 2--4 dB PSNR improvement over harmonic demodulation alone
- Published near-field contrast, spectral accuracy, and spatial resolution from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'nsom' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/nsom/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/nsom/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/nsom/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for NSOM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/nsom/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/nsom/standard/`

---

## Spectroscopic Imaging -- Modality Templates

---

### Raman Imaging (`raman_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For Raman Imaging, what dataset do you use to verify? Is this dataset used for Raman Imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/raman_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Raman Imaging standard dataset.

**Popular datasets to consider:**
- **RRUFF Database (Lafuente et al., 2015)** -- the most comprehensive mineral Raman spectral library; >3800 minerals with high-quality Raman spectra; universally used as reference for Raman mineral identification and classification algorithms
- **Raman Open Database (ROD, 2019)** -- open-access database of Raman spectra for organic and inorganic compounds; growing community resource for spectral matching
- **SERS Benchmark Datasets (Bell et al., 2005; Langer et al., Chem Rev 2020)** -- surface-enhanced Raman scattering spectra on standardized substrates; used for SERS quantification and reproducibility benchmarks
- **Tissue Raman Maps (Movasaghi et al., Appl Spectrosc Rev 2007; 2015--2023)** -- hyperspectral Raman maps of tissue sections with histopathology ground truth; benchmark for biomedical Raman imaging
- **Pharmaceutical Raman Tablets (ASTM E2529)** -- multi-component pharmaceutical tablets with known composition; standard for Raman mapping quantification
- **SCAMP Spectral Dataset (2020)** -- community Raman spectral dataset for machine learning training; used in several DL classification papers

**Decision criteria:** RRUFF is the gold standard for Raman spectral library benchmarks. Tissue Raman maps for biomedical imaging. Pharmaceutical tablets for quantitative unmixing. Use the dataset that appears in the largest number of Raman imaging analysis papers.

#### Step 2: List All Raman Imaging Algorithms

Please first ensure all the Raman Imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\raman_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/raman_imaging. Besides, you need to search all algorithms from 1960 to 2026. After listing all the Raman Imaging solvers, please update the Raman Imaging solver.

**Key algorithms to cover (1960--2026):**

_Preprocessing & Baseline Correction (1970s--2010):_
- Polynomial baseline correction -- fit and subtract polynomial (degree 3--7) from Raman spectra to remove fluorescence background (Lieber & Mahadevan-Jansen, Appl Spectrosc 2003)
- SNIP (Statistics-sensitive Non-linear Iterative Peak-clipping) -- iterative baseline estimation algorithm (Ryan et al., 1988; Morhavc et al., NIM A 1997)
- airPLS (adaptive iteratively reweighted Penalized Least Squares) -- asymmetric least squares baseline correction (Zhang et al., Analyst 2010) -- widely used modern baseline method
- Asymmetric least squares (AsLS) baseline (Eilers, 2005)
- Rubberband baseline correction (2000)
- Cosmic ray removal -- spike detection and interpolation for CCD artifacts (standard in all Raman software)
- Wavenumber calibration -- silicon 520.7 cm^-1 peak calibration (ASTM E1840)
- Intensity calibration using NIST SRM 2241/2242/2243 fluorescence standards
- Spectral smoothing -- Savitzky-Golay filtering (Savitzky & Golay, Anal Chem 1964)

_Spectral Unmixing & Decomposition (1990s--2016):_
- CLS (Classical Least Squares) -- linear unmixing using known reference spectra (Haaland & Thomas, Anal Chem 1988)
- PCA (Principal Component Analysis) -- unsupervised spectral decomposition for dimensionality reduction (Pelletier, Appl Spectrosc 2003)
- MCR-ALS (Multivariate Curve Resolution-Alternating Least Squares) -- self-modeling mixture analysis with non-negativity constraints (Tauler, Chemom Intell Lab Syst 1995; de Juan & Tauler, 2006) -- the standard unmixing method for Raman imaging
- NMF (Non-negative Matrix Factorization) -- sparse non-negative decomposition for Raman maps (Lee & Seung, Nature 1999; applied to Raman 2010+)
- VCA (Vertex Component Analysis) -- endmember extraction for hyperspectral Raman (Nascimento & Bioucas-Dias, 2005; applied 2012)
- HCA (Hierarchical Cluster Analysis) -- unsupervised spatial segmentation of Raman maps (2005)
- k-means clustering for Raman spatial segmentation (2008)
- Independent Component Analysis (ICA) for Raman unmixing (2010)
- Band target entropy minimization (BTEM) (2004)

_Deep Learning (2017--2026):_
- DL Raman classification -- CNN for rapid Raman spectral identification (Ho et al., Nat Commun 2019; Liu et al., Analyst 2017)
- 1D-CNN for bacterial Raman identification (2018)
- ResNet for Raman spectra classification (Luo et al., Analyst 2019)
- Transfer learning for Raman spectral identification (2020)
- Super-resolution Raman imaging via deep learning (Manifold et al., Nat Mach Intell 2021; 2022) -- reconstruct high-spatial-resolution Raman maps from sparse measurements
- Deep learning baseline correction (2020)
- Autoencoder-based denoising for low-SNR Raman (2021)
- GAN-based Raman spectral augmentation (2022)
- Self-supervised Raman feature learning (2023)
- Transformer-based Raman spectral classification (2024)
- Diffusion-model Raman denoising (2025)
- Foundation model for vibrational spectroscopy (2025--2026)

#### Step 3: Update Raman Imaging Solvers

After listing all Raman Imaging solvers, update `algorithm_base/raman_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Raman Imaging solvers use the data format: `y` (H, W, N_wavenumbers) hyperspectral Raman datacube, `wavenumbers` (N_wavenumbers,) array of Raman shifts in cm^-1, `reference_spectra` (N_components, N_wavenumbers) optional known reference spectra for CLS. The `RamanOperator` handles the forward model (component spectra x concentrations + baseline) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Raman Imaging:**
- RRUFF mineral identification: classification accuracy >95% top-1 for clean spectra, >85% for noisy/fluorescent spectra (Ho et al., 2019)
- Pharmaceutical tablet unmixing (MCR-ALS): component concentration RMSE <5% for 3-component mixtures
- Tissue Raman classification: accuracy >90% for cancer vs. normal on validated datasets
- Super-resolution Raman (4x spatial): PSNR ~28 dB, SSIM ~0.85 on tissue maps (Manifold et al., 2021)
- airPLS baseline: residual fluorescence <2% of original intensity on standardized test spectra
- Published classification accuracy, unmixing RMSE, and PSNR/SSIM from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'raman_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/raman_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Raman Imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/raman_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/raman_imaging/standard/`

---

### SRS (`srs`) Modality Template

#### Step 1: Verify Standard Dataset

For SRS, what dataset do you use to verify? Is this dataset used for SRS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/srs/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SRS standard dataset.

**Popular datasets to consider:**
- **SRS Cell Imaging Datasets (Freudiger et al., Science 2008; 2015--2023)** -- SRS images of cells and tissues at CH2/CH3 stretch (lipid/protein) channels; the canonical SRS benchmark for demonstrating chemical-specific contrast
- **Lipid/Protein Two-Channel SRS Data (Ji et al., 2013)** -- dual-channel SRS images separating lipid and protein contributions; standard for unmixing algorithm validation
- **Brain Tissue SRS Datasets (Ji et al., Sci Transl Med 2013; Orringer et al., Nat Biomed Eng 2017)** -- fresh brain tissue SRS for intraoperative tumor detection; used for virtual H&E benchmarks
- **Hyperspectral SRS (hSRS) Datasets (Zhang et al., Nat Biotechnol 2019)** -- spectrally-resolved SRS datacubes in fingerprint region; used for metabolic imaging and unmixing benchmarks
- **SRS-FRAME Datasets (Liao et al., Nat Commun 2021)** -- high-speed SRS with Fourier-transform approach; used for video-rate SRS benchmarks
- **DO-SRS (Deuterium oxide SRS) Metabolic Datasets (Shi et al., 2020)** -- SRS imaging with deuterium labeling for metabolic tracing

**Decision criteria:** Brain tissue SRS is the most widely used benchmark for demonstrating clinical utility. Hyperspectral SRS for unmixing algorithms. Cell imaging SRS for fundamental chemical contrast. Use the dataset most widely referenced in SRS analysis papers (2008--2026).

#### Step 2: List All SRS Algorithms

Please first ensure all the SRS algorithms have been listed in `\Physics_World_Model\algorithm_base\srs\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/srs. Besides, you need to search all algorithms from 2008 to 2026. After listing all the SRS solvers, please update the SRS solver.

**Key algorithms to cover (2008--2026):**

_Signal Detection & Extraction (2008--2015):_
- Lock-in detection for SRS -- high-frequency modulation transfer detection to extract stimulated Raman gain/loss signal (Freudiger et al., Science 2008) -- the foundational SRS detection method
- Balanced detection for SRS -- common-mode laser noise rejection using balanced photodetector (2010)
- Spectral focusing -- chirped pulse approach for narrowband SRS excitation from broadband lasers (Hellerer et al., APL 2004; Fu et al., J Phys Chem B 2013) -- standard technique for hyperspectral SRS
- Frequency-modulation SRS (FM-SRS) for background-free imaging (Zhang et al., 2012)
- Time-lens SRS for high-speed spectral acquisition (Ozeki et al., Nat Photonics 2012)

_Hyperspectral Unmixing & Analysis (2013--2020):_
- MCR-ALS for hyperspectral SRS -- multivariate curve resolution for decomposing SRS datacubes into chemical components (Zhang et al., Sci Adv 2017)
- Linear unmixing of multi-channel SRS -- least-squares decomposition using reference SRS spectra (Ji et al., 2013)
- PCA for hSRS dimensionality reduction (2015)
- Phasor approach for SRS spectral analysis (Fu et al., JACS 2012)
- Spectral phasor for rapid SRS component mapping (2016)
- NMF for SRS hyperspectral unmixing (2018)
- Independent component analysis (ICA) for SRS (2017)

_Deep Learning (2017--2026):_
- DL SRS denoising -- CNN-based denoising for low-power SRS imaging (Manifold et al., Biomed Opt Express 2019; 2020) -- enables reduced laser power and faster acquisition
- Virtual H&E from SRS -- deep learning to synthesize H&E-stained histology images from two-channel SRS (Orringer et al., Nat Biomed Eng 2017; Hollon et al., Nat Med 2020)
- CNN for SRS-based intraoperative brain tumor classification (Hollon et al., 2020)
- U-Net for SRS image segmentation (2021)
- Self-supervised SRS denoising (Noise2Noise-style) (2022)
- GAN-based SRS virtual staining beyond H&E (2022)
- Transformer-based hyperspectral SRS classification (2023)
- Super-resolution SRS via deep learning (2023)
- Diffusion-model SRS image synthesis and augmentation (2024)
- Foundation model for label-free tissue imaging (SRS + CARS + fluorescence) (2025)

#### Step 3: Update SRS Solvers

After listing all SRS solvers, update `algorithm_base/srs/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SRS solvers use the data format: `y` (H, W) single-channel SRS image or (N_wavenumbers, H, W) hyperspectral SRS datacube, `wavenumbers` (N_wavenumbers,) array of Raman shifts, `reference_spectra` (N_components, N_wavenumbers) optional reference SRS spectra for unmixing. The `SRSOperator` handles the forward model (Raman cross-section * concentration + non-resonant background) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SRS:**
- Brain tissue SRS virtual H&E: pathologist agreement >90% with real H&E (Hollon et al., 2020)
- Two-channel SRS unmixing: lipid/protein concentration error <10% on known phantoms
- hSRS MCR-ALS unmixing: spectral correlation >0.95 with reference Raman spectra for major tissue components
- DL SRS denoising: 4--6 dB PSNR improvement at 10x reduced laser power (Manifold et al., 2019)
- SRS tumor classification: accuracy >92% on brain tumor test set (Hollon et al., 2020)
- Published PSNR/SSIM, classification accuracy, and unmixing correlation from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'srs' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/srs/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/srs/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/srs/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SRS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/srs/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/srs/standard/`

---

### CARS (`cars`) Modality Template

#### Step 1: Verify Standard Dataset

For CARS, what dataset do you use to verify? Is this dataset used for CARS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cars/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CARS standard dataset.

**Popular datasets to consider:**
- **CARS Cell Imaging Datasets (Cheng et al., Biophys J 2002; Zumbusch et al., PRL 1999)** -- CARS images of cells showing lipid droplets, membranes, and organelles; the canonical CARS biological imaging benchmark
- **Multiplex CARS Spectral Datasets (Muller & Zumbusch, Chem Phys Lett 2007; Kano & Hamaguchi, 2005)** -- broadband CARS spectra with non-resonant background; standard for testing background removal algorithms
- **Tissue CARS Datasets (Evans et al., PNAS 2005; 2010--2023)** -- CARS images of tissue sections (atherosclerotic plaque, brain white matter, adipose tissue); used for biomedical CARS benchmarks
- **CARS Polymer Blend Standards (Kee & Cicerone, Opt Lett 2004)** -- known polymer mixtures with well-characterized CARS spectra; quantitative unmixing benchmark
- **Broadband CARS Reference Spectra (Camp & Cicerone, Nat Photonics 2015)** -- high-spectral-resolution CARS reference data with known Raman-equivalent spectra; used for KK/MEM algorithm validation
- **CARS Microspectroscopy Datasets (Okuno et al., Opt Lett 2010)** -- multiplex CARS data with simultaneous spontaneous Raman for cross-validation

**Decision criteria:** CARS cell imaging for biological CARS benchmarks. Multiplex CARS spectra for background removal algorithm validation. Tissue CARS for biomedical applications. Use the dataset most widely referenced in CARS image analysis papers (1999--2026).

#### Step 2: List All CARS Algorithms

Please first ensure all the CARS algorithms have been listed in `\Physics_World_Model\algorithm_base\cars\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cars. Besides, you need to search all algorithms from 1999 to 2026. After listing all the CARS solvers, please update the CARS solver.

**Key algorithms to cover (1999--2026):**

_Non-resonant Background Suppression (1999--2015):_
- Maximum Entropy Method (MEM) for CARS -- extract resonant chi^(3) from CARS spectrum by maximizing spectral entropy subject to data constraint (Vartiainen et al., JOSA B 1992; applied to microscopy, Rinia et al., JOSA B 2007) -- the most widely used CARS background removal algorithm
- Kramers-Kronig (KK) transform for CARS -- retrieve imaginary part of chi^(3) (Raman-equivalent spectrum) from CARS intensity via KK relations (Liu et al., Opt Lett 2009; Camp et al., Nat Photonics 2015) -- fast and widely adopted
- Time-domain CARS background suppression -- exploit temporal delay between resonant and non-resonant signals (Volkmer et al., PRL 2001)
- Epi-detected CARS (E-CARS) for background rejection (Cheng et al., Opt Lett 2001)
- Frequency-modulated CARS (FM-CARS) for background-free imaging (Ganikhanov et al., Opt Lett 2006)
- Interferometric CARS -- heterodyne detection for phase-sensitive CARS and background suppression (Potma et al., Opt Lett 2006)
- Polarization CARS (P-CARS) -- suppress non-resonant background using polarization selection (Cheng et al., PRL 2001)
- Phase retrieval for CARS spectra -- iterative algorithms to extract complex susceptibility (2010)

_Spectral Analysis & Unmixing (2005--2020):_
- Spectral focusing for CARS -- chirped broadband pulses for narrowband spectral resolution (Hellerer et al., APL 2004)
- MCR-ALS for multiplex CARS unmixing (2012)
- PCA/HCA for CARS hyperspectral segmentation (2008)
- Singular value decomposition (SVD) denoising of CARS spectra (2009)
- CLS fitting of CARS spectra using MEM/KK-corrected reference library (2011)
- Spectral phasor analysis for CARS (2016)

_Deep Learning (2017--2026):_
- DL CARS background removal -- CNN to directly remove non-resonant background from CARS spectra (Valensise et al., APL Photonics 2020; 2021)
- Deep learning for rapid KK retrieval (2022)
- CNN-based CARS image denoising (2021)
- GAN-based CARS spectral enhancement (2022)
- Virtual H&E from CARS images (2023)
- U-Net for CARS tissue segmentation (2022)
- Physics-informed neural network for CARS spectral inversion (2024)
- Transformer-based multiplex CARS classification (2024)
- Diffusion-model CARS denoising and spectral recovery (2025)
- Foundation model for coherent Raman spectroscopy (CARS + SRS) (2025--2026)

#### Step 3: Update CARS Solvers

After listing all CARS solvers, update `algorithm_base/cars/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CARS solvers use the data format: `y` (H, W) single-channel CARS intensity image or (N_wavenumbers, H, W) multiplex/broadband CARS datacube, `wavenumbers` (N_wavenumbers,) array of Raman shifts, `non_resonant_ref` (N_wavenumbers,) non-resonant background reference spectrum. The `CARSOperator` handles the forward model (|chi^(3)_R + chi^(3)_NR|^2 intensity) and phase retrieval operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CARS:**
- MEM on known polymer blend: retrieved Raman peak positions within 3 cm^-1 of spontaneous Raman reference; spectral correlation >0.95
- KK transform: identical spectral fidelity to MEM but 100x faster processing; spectral correlation >0.94 (Camp et al., 2015)
- Multiplex CARS unmixing: component concentration RMSE <8% for 3-component mixtures
- DL background removal: spectral correlation >0.93 with 10x faster processing than MEM (Valensise et al., 2020)
- Tissue CARS classification: accuracy >88% for lipid-rich vs. protein-rich regions
- Published spectral correlation, RMSE, and processing speed from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cars' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cars/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cars/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cars/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CARS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cars/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cars/standard/`

---

### Brillouin (`brillouin`) Modality Template

#### Step 1: Verify Standard Dataset

For Brillouin Microscopy, what dataset do you use to verify? Is this dataset used for Brillouin popular algorithms? Please ensure the standard dataset in `datasets/benchmark/brillouin/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Brillouin standard dataset.

**Popular datasets to consider:**
- **Phantom Hydrogel Datasets (Scarcelli & Yun, Nat Photonics 2008; 2012--2023)** -- polyacrylamide hydrogels with known stiffness (elastic modulus); the primary calibration standard for Brillouin microscopy; used for validating frequency shift to modulus conversion
- **Brillouin Cell/Tissue Data (Scarcelli et al., Nat Methods 2015)** -- Brillouin shift maps of cells and tissue sections with correlative AFM/rheology ground truth; benchmark for biological Brillouin microscopy
- **In-vivo Cornea Brillouin Datasets (Scarcelli et al., IOVS 2012; Shao et al., 2019)** -- Brillouin maps of corneal stiffness in-vivo; the most clinically relevant Brillouin benchmark
- **Intralipid/Glycerol Phantom Series (Antonacci et al., 2013)** -- liquid phantoms with known viscosity and refractive index; used for Brillouin spectrometer calibration
- **Water/Methanol Reference Spectra** -- known Brillouin shift (7.46 GHz for water at 532 nm excitation, 25 C); universal spectrometer calibration standard
- **3D Brillouin Elasticity Maps (Prevedel et al., Nat Methods 2019; 2020--2023)** -- volumetric Brillouin maps of organoids and zebrafish embryos; used for 3D mechanical imaging benchmarks

**Decision criteria:** Hydrogel phantoms with known stiffness for quantitative calibration. In-vivo cornea for clinical benchmarks. Cell/tissue data for biological imaging. Use the dataset most widely referenced in Brillouin microscopy papers (2008--2026).

#### Step 2: List All Brillouin Algorithms

Please first ensure all the Brillouin algorithms have been listed in `\Physics_World_Model\algorithm_base\brillouin\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/brillouin. Besides, you need to search all algorithms from 2008 to 2026. After listing all the Brillouin solvers, please update the Brillouin solver.

**Key algorithms to cover (2008--2026):**

_VIPA Spectrometer Analysis & Signal Processing (2008--2015):_
- VIPA spectrometer analysis -- extract Brillouin spectrum from virtually imaged phased array (VIPA) etalon output (Scarcelli & Yun, Opt Express 2011) -- the foundational Brillouin spectrometer signal processing method
- Lorentzian fitting -- fit Brillouin peaks with Lorentzian lineshape to extract frequency shift and linewidth (standard since first Brillouin microscopy papers, 2008)
- Voigt profile fitting -- combined Lorentzian (intrinsic) and Gaussian (instrumental) contributions for more accurate linewidth extraction (2012)
- Multi-order VIPA spectrum extraction -- handling multiple free spectral range orders in VIPA output (2012)
- Elastic scattering rejection -- spectral notch filtering to suppress Rayleigh peak (2010)
- Spectral calibration using known reference materials (water, methanol) (2009)

_Advanced Fitting & Statistical Methods (2013--2020):_
- Bayesian fitting for Brillouin spectra -- probabilistic peak fitting with uncertainty quantification (Caponi et al., Biophys J 2020)
- Multi-peak analysis -- fitting multiple Brillouin peaks for heterogeneous or anisotropic samples (Mattana et al., 2017)
- Maximum likelihood estimation for Brillouin shift (2018)
- Spectral moment analysis -- extract mean shift and width from spectral moments without explicit peak fitting (Nikolov et al., 2016)
- Temperature and hydration correction for Brillouin shift (Wu et al., 2018)
- Refractive index correction -- separate Brillouin shift changes from refractive index vs. mechanical modulus contributions (Scarcelli et al., 2015)
- Brillouin-Raman combined analysis for simultaneous mechanical and chemical mapping (2017)

_Deep Learning (2017--2026):_
- DL Brillouin shift extraction -- CNN to rapidly extract Brillouin shift from raw VIPA spectrograms without iterative fitting (Mattana et al., APL 2021; 2022)
- Neural network for Brillouin spectral denoising (2022)
- Deep learning for rapid 3D Brillouin map reconstruction (2023)
- Physics-informed neural network for Brillouin spectrum inversion (2023)
- U-Net for Brillouin image segmentation (mechanical phenotyping) (2024)
- GAN-based Brillouin image enhancement from sparse measurements (2024)
- Transfer learning for Brillouin modulus prediction across tissue types (2024)
- Transformer-based Brillouin-Raman joint analysis (2025)
- Diffusion-model Brillouin spectral denoising (2025)
- Foundation model for mechanical microscopy (Brillouin + AFM) (2025--2026)

#### Step 3: Update Brillouin Solvers

After listing all Brillouin solvers, update `algorithm_base/brillouin/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Brillouin solvers use the data format: `y` (H, W) Brillouin shift map or (H, W, N_spectral) raw VIPA spectrograms, `calibration` dict containing spectrometer parameters (FSR, dispersion, reference shift). The `BrillouinOperator` handles the forward model (elastic modulus -> Brillouin shift via sound velocity and refractive index) and fitting operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Brillouin:**
- Water reference: Brillouin shift extraction within 10 MHz of known value (7.46 GHz at 532 nm, 25 C)
- Hydrogel phantoms: extracted elastic modulus within 15% of AFM/rheology reference value (Scarcelli et al., 2015)
- In-vivo cornea: Brillouin shift spatial variation consistent with known anterior-posterior stiffness gradient
- DL shift extraction: 5x faster than iterative Lorentzian fitting with <20 MHz accuracy degradation (Mattana et al., 2022)
- Bayesian fitting: uncertainty estimates consistent with repeated measurements
- Published Brillouin shift accuracy and modulus correlation from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'brillouin' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/brillouin/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/brillouin/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/brillouin/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Brillouin. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/brillouin/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/brillouin/standard/`

---

### FTIR Imaging (`ftir_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For FTIR Imaging, what dataset do you use to verify? Is this dataset used for FTIR Imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ftir_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original FTIR Imaging standard dataset.

**Popular datasets to consider:**
- **NIST/EPA Gas-Phase IR Library** -- comprehensive gas-phase infrared spectral library; reference standard for spectral identification algorithms
- **SDBS (Spectral Database for Organic Compounds, AIST)** -- large collection of standard IR spectra for organic compounds; widely used for spectral matching benchmarks
- **FTIR Tissue Microarray Datasets (Fernandez et al., Nat Biotechnol 2005; 2010--2023)** -- FTIR hyperspectral images of tissue microarrays with histopathology ground truth; the primary benchmark for FTIR tissue imaging classification
- **Pharmaceutical FTIR Imaging Benchmarks (Kazarian & Andrew, Chem Soc Rev 2006)** -- drug tablet and formulation FTIR maps with known component distribution; used for quantitative imaging validation
- **FTIR Microplastics Datasets (Primpke et al., Anal Bioanal Chem 2018; 2020--2023)** -- FTIR images and spectra of environmental microplastic particles; growing benchmark for automated identification
- **RMieS-EMSC Correction Benchmark Data (Bassan et al., Analyst 2010)** -- FTIR tissue spectra with known Mie scattering artifacts; standard for scatter correction algorithm validation
- **OPUS/KnowItAll Spectral Libraries** -- commercial but widely referenced spectral databases for algorithm benchmarking

**Decision criteria:** FTIR tissue microarrays with histopathology ground truth for biomedical imaging. RMieS-EMSC benchmark for scatter correction. SDBS for spectral identification. Use the dataset most widely referenced in FTIR imaging analysis papers (2005--2026).

#### Step 2: List All FTIR Imaging Algorithms

Please first ensure all the FTIR Imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\ftir_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ftir_imaging. Besides, you need to search all algorithms from 1960 to 2026. After listing all the FTIR Imaging solvers, please update the FTIR Imaging solver.

**Key algorithms to cover (1960--2026):**

_Preprocessing & Scatter Correction (1980s--2015):_
- Atmospheric correction -- subtract water vapor and CO2 spectral contributions from FTIR spectra (standard in all FTIR software)
- Baseline correction -- rubberband, polynomial, and derivative-based baseline correction for FTIR absorbance spectra
- RMieS-EMSC (Resonant Mie Scattering Extended Multiplicative Signal Correction) -- iterative correction for Mie scattering artifacts in FTIR tissue spectra (Bassan et al., Analyst 2009, 2010) -- the standard scatter correction for FTIR microscopy
- EMSC (Extended Multiplicative Signal Correction) -- model-based spectral preprocessing (Martens & Stark, JNIRS 1991; Kohler et al., Appl Spectrosc 2005)
- Kramers-Kronig correction for ATR-FTIR spectra (2005)
- Derivative spectroscopy (1st and 2nd derivative) for spectral resolution enhancement (Savitzky-Golay)
- SNV (Standard Normal Variate) normalization (Barnes et al., 1989)
- MC (Mean Centering) and Pareto scaling (standard chemometric preprocessing)

_Multivariate Analysis & Classification (1990s--2016):_
- PCA (Principal Component Analysis) for FTIR hyperspectral dimensionality reduction (Diem et al., Appl Spectrosc 1999)
- HCA (Hierarchical Cluster Analysis) for FTIR spatial segmentation (Lasch et al., 2002)
- MCR-ALS for FTIR spectral unmixing -- extract pure component spectra and concentration maps (Tauler, 2005; applied to FTIR imaging 2008)
- PLS-DA (Partial Least Squares Discriminant Analysis) for FTIR tissue classification (Fernandez et al., Nat Biotechnol 2005)
- LDA (Linear Discriminant Analysis) for FTIR histopathology (Lasch, Chemom Intell Lab Syst 2012)
- Random Forest (RF) classification for FTIR tissue typing (Kallenbach-Thieltges et al., JBIR 2013)
- SVM for FTIR spectral classification (2010)
- k-means and fuzzy c-means clustering for FTIR maps (2005)
- CLS for quantitative FTIR analysis using reference spectra (2000)

_Deep Learning (2017--2026):_
- DL FTIR tissue classification -- CNN for automated FTIR histopathology (Kuepper et al., Sci Rep 2018; Raulf et al., Analyst 2019)
- 1D-CNN for FTIR spectral classification (2019)
- ResNet for FTIR tissue typing (2020)
- Transfer learning from visible histopathology to FTIR (2020)
- Super-resolution FTIR imaging via deep learning (Wafai et al., ACS Photonics 2023) -- reconstruct high-spatial-resolution FTIR maps from low-resolution measurements
- Autoencoder-based FTIR denoising and dimensionality reduction (2021)
- GAN-based FTIR spectral augmentation for rare tissue types (2022)
- U-Net for FTIR tissue segmentation (2022)
- Physics-informed DL for scatter-corrected FTIR (2023)
- Transformer-based FTIR spectral classification (2024)
- Diffusion-model FTIR denoising (2025)
- Foundation model for vibrational spectroscopy imaging (FTIR + Raman) (2025--2026)

#### Step 3: Update FTIR Imaging Solvers

After listing all FTIR Imaging solvers, update `algorithm_base/ftir_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All FTIR Imaging solvers use the data format: `y` (H, W, N_wavenumbers) FTIR absorbance datacube, `wavenumbers` (N_wavenumbers,) array of wavenumbers in cm^-1, `reference_spectra` (N_components, N_wavenumbers) optional pure component spectra for CLS/MCR-ALS. The `FTIROperator` handles the forward model (Beer-Lambert absorption with scatter correction) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for FTIR Imaging:**
- FTIR tissue microarray classification: accuracy >90% for major tissue types (epithelium, stroma, necrosis) with RF/SVM (Kallenbach-Thieltges et al., 2013)
- RMieS-EMSC scatter correction: spectral distortion reduced by >80% on scattering tissue spectra (Bassan et al., 2010)
- MCR-ALS unmixing: component recovery correlation >0.95 on known pharmaceutical mixtures
- DL FTIR classification: accuracy >93% on tissue microarray test set (Kuepper et al., 2018)
- Super-resolution FTIR (4x): PSNR ~27 dB, SSIM ~0.82 on tissue maps
- Published classification accuracy, spectral fidelity, and PSNR/SSIM from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ftir_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ftir_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ftir_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ftir_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for FTIR Imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ftir_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ftir_imaging/standard/`

---

### LIBS (`libs`) Modality Template

#### Step 1: Verify Standard Dataset

For LIBS, what dataset do you use to verify? Is this dataset used for LIBS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/libs/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original LIBS standard dataset.

**Popular datasets to consider:**
- **NIST Atomic Spectra Database (Kramida et al., continuously updated)** -- the definitive reference for atomic emission line wavelengths and transition probabilities; universally used for LIBS line identification and CF-LIBS analysis
- **ChemCam Mars LIBS Database (Wiens et al., Space Sci Rev 2012; Clegg et al., SAB 2017)** -- LIBS spectra from the Curiosity rover on Mars with laboratory reference standards; the most prominent LIBS dataset; includes >6000 laboratory reference spectra on geological standards
- **Steel Certification Reference Standards (NIST SRMs, BAM CRMs)** -- certified reference materials with known elemental composition; standard for LIBS quantification accuracy assessment
- **LIBS Soil Database (Hussain et al., 2020; Hark & Harmon, 2014)** -- LIBS spectra of soils with ICP-AES reference compositions; used for environmental LIBS benchmarks
- **NELIBS Benchmark Dataset (De Giacomo et al., 2014)** -- nanoparticle-enhanced LIBS spectra with enhanced sensitivity; used for NELIBS algorithm validation
- **IAEA LIBS Database** -- reference LIBS spectra for nuclear materials analysis; used for specialized LIBS applications

**Decision criteria:** NIST ASD is the universal line identification reference. ChemCam database for multivariate classification/quantification. Steel CRMs for quantitative accuracy. Use the dataset most widely referenced in LIBS analysis papers (1999--2026).

#### Step 2: List All LIBS Algorithms

Please first ensure all the LIBS algorithms have been listed in `\Physics_World_Model\algorithm_base\libs\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/libs. Besides, you need to search all algorithms from 1960 to 2026. After listing all the LIBS solvers, please update the LIBS solver.

**Key algorithms to cover (1960--2026):**

_Peak Identification & Classical Quantification (1960s--2009):_
- Peak identification -- match observed emission lines to NIST ASD wavelength database for elemental assignment (standard since the 1960s)
- Calibration curve method -- plot line intensity vs. known concentration for univariate quantification (the oldest LIBS quantification approach)
- Internal standard method -- normalize analyte line by reference element line for matrix-independent quantification (1970s)
- CF-LIBS (Calibration-Free LIBS) -- calculate elemental composition from spectral line intensities using Boltzmann distribution and Saha equation without calibration standards (Ciucci et al., Appl Spectrosc 1999) -- the foundational calibration-free method
- Self-absorption correction for CF-LIBS (Bulajic et al., SAB 2002)
- Continuum background subtraction -- remove Bremsstrahlung and recombination radiation (standard preprocessing)
- Spectral deconvolution -- fit overlapping lines with Lorentzian/Voigt profiles (2000)
- Stark broadening analysis for electron density measurement (2005)

_Multivariate Calibration & Chemometrics (2005--2016):_
- PLS (Partial Least Squares) regression for LIBS quantification (Sirven et al., Anal Chem 2006) -- the most widely used multivariate calibration for LIBS
- PCR (Principal Component Regression) for LIBS (2005)
- PCA for LIBS spectral dimensionality reduction and sample classification (2004)
- Artificial Neural Network (ANN) for LIBS classification and quantification (2008)
- SVM for LIBS spectral classification (2010)
- LIBS mapping -- spatially-resolved elemental distribution from rastered LIBS spectra (Motto-Ros et al., SAB 2012)
- Random Forest for LIBS classification (2014)
- ML classification for geological sample ID (Harmon et al., Geochem Explor Environ Anal 2006; expanded 2016) -- LIBS + machine learning for geological classification

_Deep Learning (2017--2026):_
- DL LIBS quantification -- CNN/MLP for direct elemental quantification from LIBS spectra without feature selection (Castorena et al., SAB 2021; Yang et al., 2020)
- 1D-CNN for LIBS spectral classification (El Haddad et al., SAB 2019; 2020)
- Transfer learning for LIBS across different matrices (2021)
- LSTM for LIBS time-resolved spectral analysis (2020)
- Autoencoder-based LIBS spectral denoising (2021)
- GAN-based LIBS spectral augmentation for small training sets (2022)
- Deep learning CF-LIBS -- neural network replacing Boltzmann plot analysis (2023)
- Transformer-based LIBS spectral classification (2024)
- Multi-task DL for simultaneous LIBS classification and quantification (2024)
- Diffusion-model LIBS spectral denoising (2025)
- Foundation model for atomic emission spectroscopy (LIBS + ICP-OES) (2025--2026)

#### Step 3: Update LIBS Solvers

After listing all LIBS solvers, update `algorithm_base/libs/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All LIBS solvers use the data format: `y` (N_pixels,) or (H, W, N_wavelengths) LIBS spectrum or spectral map, `wavelengths` (N_wavelengths,) array of emission wavelengths in nm, `nist_lines` dict of reference line positions and transition probabilities. The `LIBSOperator` handles the forward model (plasma emission with self-absorption) and spectral fitting operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for LIBS:**
- Steel CRM quantification: CF-LIBS accuracy within 15-20% relative error for major elements (Ciucci et al., 1999); PLS within 5-10% (Sirven et al., 2006)
- ChemCam rock classification: ML classification accuracy >85% for major rock types (Harmon et al., 2016; Clegg et al., 2017)
- DL quantification: RMSE <2 wt% for major elements in geological samples (Yang et al., 2020)
- LIBS mapping: spatial resolution ~50-100 um with elemental sensitivity ~10-100 ppm
- Published classification accuracy, quantification RMSE, and detection limits from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'libs' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/libs/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/libs/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/libs/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for LIBS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/libs/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/libs/standard/`

---

### MALDI-MSI (`maldi_msi`) Modality Template

#### Step 1: Verify Standard Dataset

For MALDI-MSI, what dataset do you use to verify? Is this dataset used for MALDI-MSI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/maldi_msi/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original MALDI-MSI standard dataset.

**Popular datasets to consider:**
- **METASPACE (Alexandrov et al., Nat Methods 2020)** -- the largest community repository for mass spectrometry imaging data with metabolite annotations; >10,000 public datasets; the primary benchmark and resource for MSI algorithm development
- **Human Protein Atlas Mass Spectrometry (Uhlen et al., 2015; 2020--2023)** -- MALDI-MSI datasets of human tissue sections with protein/peptide annotations; used for spatial proteomics benchmarks
- **SCiLS Benchmark Datasets (Bruker)** -- standardized MALDI-MSI datasets from commercial software; used for algorithm comparison and validation
- **MALDI-MSI Tissue Microarray Datasets (Balluff et al., Histochem Cell Biol 2011)** -- MALDI-MSI on tissue microarrays with clinical outcome data; benchmark for biomarker discovery algorithms
- **3D MALDI-MSI Datasets (Oetjen et al., GigaScience 2015)** -- serial section MALDI-MSI with 3D reconstruction; used for volumetric MSI analysis benchmarks
- **MALDI-MSI Lipid Atlas (Berry et al., Nat Rev Chem 2022)** -- comprehensive MALDI lipid imaging datasets across tissues; reference for lipid annotation algorithms
- **MSI Benchmark Peptide/Protein Standards** -- spotted peptide/protein standard arrays with known m/z and concentration; used for mass accuracy and quantification validation

**Decision criteria:** METASPACE is the undisputed community standard for MALDI-MSI data sharing and benchmarking. Tissue microarrays for classification. Lipid atlas for lipidomics. Use the dataset most widely referenced in MALDI-MSI analysis papers (2010--2026).

#### Step 2: List All MALDI-MSI Algorithms

Please first ensure all the MALDI-MSI algorithms have been listed in `\Physics_World_Model\algorithm_base\maldi_msi\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/maldi_msi. Besides, you need to search all algorithms from 1997 to 2026. After listing all the MALDI-MSI solvers, please update the MALDI-MSI solver.

**Key algorithms to cover (1997--2026):**

_Preprocessing & Signal Processing (1997--2010):_
- Baseline subtraction -- remove matrix-related chemical noise from MALDI mass spectra (standard since first MALDI-MSI papers, Caprioli et al., Anal Chem 1997)
- Peak picking -- detect and centroid mass spectral peaks above noise threshold (TIC normalization, median filtering)
- Peak alignment / mass recalibration -- correct m/z drift across pixels using lock masses or internal calibrants
- TIC (Total Ion Current) normalization -- normalize spectra by total ion current for pixel-to-pixel comparison
- Median normalization and RMS normalization -- alternative intensity normalization strategies
- Spectral smoothing -- Gaussian or Savitzky-Golay smoothing for noise reduction
- Matrix cluster removal -- filter out matrix-derived peaks from analyte spectra

_Spatial Analysis & Unsupervised Methods (2005--2016):_
- Spatial segmentation by k-means clustering -- unsupervised partitioning of MSI data based on spectral similarity (Alexandrov et al., Anal Chem 2010)
- t-SNE visualization for MSI -- dimensionality reduction for visualizing spectral heterogeneity in MSI datasets (Abdelmoula et al., PNAS 2016)
- UMAP for MSI dimensionality reduction (2019, foundations 2016)
- PCA for MSI dimensionality reduction and denoising (2005)
- HCA (Hierarchical Cluster Analysis) for MSI tissue segmentation (2008)
- NMF for MSI spectral unmixing (2012)
- Probabilistic latent semantic analysis (pLSA) for MSI (Hanselmann et al., Anal Chem 2008)
- Spatial-aware segmentation (Bayesian spatial model, Alexandrov et al., 2013)
- Ion image colocalization analysis -- measure spatial correlation between ion images (2014)

_Supervised Classification & Quantification (2010--2020):_
- Random Forest for MSI pixel classification (Inglese et al., Analyst 2017)
- SVM for MSI tissue classification (2012)
- PLS-DA for MSI biomarker analysis (2013)
- Linear discriminant analysis for MSI (2011)
- Ion image denoising -- spatial smoothing and edge-preserving denoising for noisy ion images (2015)

_Deep Learning (2017--2026):_
- DL MSI classification -- CNN for automated tissue typing from MSI spectra (Behrmann et al., Bioinformatics 2018; 2019)
- Autoencoder for MSI dimensionality reduction and denoising (2019)
- U-Net for MSI tissue segmentation (2020)
- Super-resolution MSI via deep learning -- reconstruct high-spatial-resolution ion images from low-resolution MALDI-MSI (Zhang et al., Anal Chem 2023) -- enables enhanced spatial detail
- GAN-based MSI data augmentation (2021)
- Self-supervised learning for MSI feature extraction (2022)
- Deep learning metabolite annotation for MSI (2022)
- Transformer-based MSI spectral analysis (2024)
- Graph neural network for MSI spatial analysis (2023)
- Diffusion-model MSI denoising and inpainting (2025)
- Foundation model for mass spectrometry imaging (2025--2026)

#### Step 3: Update MALDI-MSI Solvers

After listing all MALDI-MSI solvers, update `algorithm_base/maldi_msi/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All MALDI-MSI solvers use the data format: `y` (H, W, N_mz) mass spectrometry imaging datacube, `mz_values` (N_mz,) array of m/z values, `pixel_size` spatial resolution in micrometers. The `MALDIMSIOperator` handles the forward model (ionization + desorption + mass analysis) and spectral processing operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for MALDI-MSI:**
- METASPACE metabolite annotation: false discovery rate <10% at score threshold (Alexandrov et al., 2020)
- k-means/t-SNE spatial segmentation: Adjusted Rand Index >0.7 vs. histopathology annotation on tissue sections
- DL tissue classification: accuracy >90% on tissue microarray test set (Behrmann et al., 2019)
- Super-resolution MSI (4x): ion image PSNR ~26 dB, SSIM ~0.80 (Zhang et al., 2023)
- NMF unmixing: component recovery correlation >0.90 on known lipid mixtures
- Published ARI, classification accuracy, PSNR/SSIM from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'maldi_msi' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/maldi_msi/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for MALDI-MSI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/maldi_msi/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/maldi_msi/standard/`

---

### DESI (`desi`) Modality Template

#### Step 1: Verify Standard Dataset

For DESI, what dataset do you use to verify? Is this dataset used for DESI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/desi/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original DESI standard dataset.

**Popular datasets to consider:**
- **DESI Tissue Benchmark Datasets (Wiseman et al., PNAS 2006; Eberlin et al., 2013)** -- DESI mass spectrometry images of tissue sections (brain, liver, kidney) with histopathology ground truth; the original and most widely cited DESI imaging benchmarks
- **Lipid Atlas DESI Data (Eberlin et al., Angew Chem 2010; 2015--2023)** -- comprehensive DESI lipid imaging across tissue types; standard for DESI lipidomics validation
- **Intraoperative DESI Datasets (Balog et al., Sci Transl Med 2013; Takats et al.)** -- iKnife/DESI data from intraoperative tissue classification; the key clinical DESI benchmark; includes hundreds of tissue samples with surgical pathology labels
- **DESI Cancer vs. Normal Tissue Datasets (Dill et al., Anal Chem 2011; 2015--2023)** -- DESI spectra/images from matched cancer and normal tissue pairs across multiple organs
- **3D DESI-MSI Datasets (2018--2023)** -- serial section DESI imaging with 3D reconstruction; used for volumetric DESI analysis
- **DESI Drug Distribution Datasets (Wiseman et al., 2008)** -- DESI images showing drug distribution in tissue sections with LC-MS/MS quantitative validation

**Decision criteria:** Intraoperative DESI (Balog/Takats) is the most impactful clinical DESI dataset. Tissue benchmark for imaging. Lipid atlas for lipidomics. Use the dataset most widely referenced in DESI analysis papers (2006--2026).

#### Step 2: List All DESI Algorithms

Please first ensure all the DESI algorithms have been listed in `\Physics_World_Model\algorithm_base\desi\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/desi. Besides, you need to search all algorithms from 2004 to 2026. After listing all the DESI solvers, please update the DESI solver.

**Key algorithms to cover (2004--2026):**

_Preprocessing & Signal Processing (2004--2012):_
- Background subtraction -- remove solvent and substrate signal contributions from DESI spectra (standard since Takats et al., Science 2004)
- m/z calibration -- lock-mass calibration to correct mass drift across DESI experiments
- TIC normalization -- normalize spectra by total ion current for pixel comparison
- Median normalization and reference peak normalization -- alternative normalization strategies
- Peak picking and centroiding -- detect and centroid mass spectral peaks
- Spectral alignment -- correct mass shifts between scan lines and experiments
- Ion suppression correction -- compensate for ion suppression effects from matrix/tissue composition (2008)
- Solvent-related artifact removal -- filter out electrospray solvent clusters and adducts

_Multivariate Analysis & Classification (2008--2020):_
- PCA for DESI spectral analysis -- dimensionality reduction and exploratory analysis (Eberlin et al., 2010)
- PLS-DA (Partial Least Squares Discriminant Analysis) -- supervised tissue classification from DESI spectra (Dill et al., Anal Chem 2011) -- the most widely used DESI classification method
- Lasso/Elastic Net for DESI biomarker selection (2012)
- Random Forest for DESI tissue classification (2015)
- SVM for DESI cancer classification (2013)
- LDA for DESI tissue typing (Balog et al., Sci Transl Med 2013)
- k-means and HCA for DESI spatial segmentation (2012)
- Bayesian classifier for intraoperative DESI (2015)
- NMF for DESI spectral unmixing (2016)

_Deep Learning (2017--2026):_
- CNN for DESI tissue classification (Inglese et al., Chem Sci 2017; 2020)
- DL for rapid intraoperative DESI classification (2020)
- Autoencoder for DESI feature learning and denoising (2019)
- U-Net for DESI tissue segmentation (2021)
- Transfer learning from MALDI-MSI to DESI (2022)
- GAN-based DESI data augmentation (2022)
- Self-supervised DESI spectral representation learning (2023)
- Transformer-based DESI spectral classification (2024)
- Graph neural network for DESI spatial analysis (2024)
- Diffusion-model DESI image denoising (2025)
- Foundation model for ambient ionization mass spectrometry (DESI + REIMS) (2025--2026)

#### Step 3: Update DESI Solvers

After listing all DESI solvers, update `algorithm_base/desi/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All DESI solvers use the data format: `y` (H, W, N_mz) DESI mass spectrometry imaging datacube or (N_scans, N_mz) line scan spectra, `mz_values` (N_mz,) array of m/z values, `pixel_size` spatial resolution in micrometers. The `DESIOperator` handles the forward model (desorption electrospray ionization + mass analysis) and spectral processing operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for DESI:**
- Intraoperative DESI tissue classification: sensitivity >92%, specificity >95% for cancer vs. normal using PLS-DA/LDA (Balog et al., 2013)
- DESI brain tissue segmentation: classification accuracy >90% for white/gray matter and tumor regions
- Lipid identification: >80% of major phospholipid species correctly assigned with <5 ppm mass accuracy
- DL DESI classification: accuracy comparable to or exceeding PLS-DA (>93%) with faster inference (2020)
- Published sensitivity, specificity, and classification accuracy from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'desi' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/desi/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/desi/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/desi/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for DESI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/desi/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/desi/standard/`

---

### SIMS (`sims`) Modality Template

#### Step 1: Verify Standard Dataset

For SIMS, what dataset do you use to verify? Is this dataset used for SIMS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sims/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SIMS standard dataset.

**Popular datasets to consider:**
- **NIST SIMS Reference Materials (Wilson et al., 1989; SRM series)** -- implant dose standards and relative sensitivity factors (RSFs) for quantitative SIMS; the primary metrological reference for SIMS depth profiling
- **NanoSIMS Isotope Standards (Hoppe et al., Appl Surf Sci 2013)** -- isotopically labeled reference materials for NanoSIMS calibration; standard for isotope ratio imaging benchmarks
- **ToF-SIMS Spectral Libraries (Belu et al., Anal Chem 2003; SurfaceSpectra, Vickerman)** -- reference ToF-SIMS mass spectra for polymer, organic, and inorganic surfaces; standard for spectral identification
- **VAMAS ToF-SIMS Round-Robin Data (Gilmore et al., Surf Interface Anal 2007)** -- multi-laboratory ToF-SIMS comparison data for reproducibility assessment; community benchmark
- **NanoSIMS Biological Imaging Standards (Steinhauser et al., 2012)** -- isotope-labeled cell datasets with known isotope enrichment; used for biological NanoSIMS algorithm validation
- **Dynamic SIMS Depth Profile Standards (ASTM E2091, E1438)** -- standardized depth profiling data on implanted semiconductors; used for depth resolution and quantification benchmarks
- **ToF-SIMS 3D Datasets (Breitenstein et al., 2007; 2015--2023)** -- 3D sputter depth profiling data with known layered structures

**Decision criteria:** NIST SIMS reference materials for quantitative depth profiling. VAMAS round-robin for ToF-SIMS spectral analysis. NanoSIMS isotope standards for isotope ratio imaging. Use the dataset most widely referenced in SIMS analysis papers (2000--2026).

#### Step 2: List All SIMS Algorithms

Please first ensure all the SIMS algorithms have been listed in `\Physics_World_Model\algorithm_base\sims\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sims. Besides, you need to search all algorithms from 1960 to 2026. After listing all the SIMS solvers, please update the SIMS solver.

**Key algorithms to cover (1960--2026):**

_Signal Correction & Quantification (1960s--2009):_
- Dead-time correction -- correct detector saturation at high count rates using Poisson model (standard since earliest SIMS instruments)
- Mass calibration -- polynomial mass calibration for ToF-SIMS (m/z = a(t - t0)^2; standard in all ToF-SIMS software)
- RSF-based quantification -- quantify elemental concentrations using relative sensitivity factors from implant standards (Wilson et al., 1989)
- Depth calibration -- convert sputter time to depth using known crater depth (profilometry) or known layer thickness
- Matrix effect correction -- compensate for secondary ion yield variations with matrix composition (2000)
- Charge compensation correction for insulating samples (2005)
- Isotope ratio calculation with statistical uncertainty (Poisson counting statistics)

_Depth Profiling & 3D Analysis (1990s--2016):_
- Depth profiling -- plot secondary ion intensity vs. sputter time/depth for layered structures (standard since the 1970s)
- Depth resolution deconvolution -- extract true concentration profile from broadened depth profile using response function (Hofmann, Surf Interface Anal 1998)
- MRI (Mixing-Roughness-Information depth) model for depth profile analysis (Hofmann, 1998)
- 3D visualization -- reconstruct volumetric elemental maps from sequential 2D ion images during sputtering (Breitenstein et al., 2007)
- Sputter rate correction for 3D SIMS -- correct for differential sputter rates across heterogeneous samples (2010)
- 3D alignment and drift correction for NanoSIMS image stacks (2012)

_Multivariate Analysis for ToF-SIMS (2000--2016):_
- PCA for ToF-SIMS -- principal component analysis for spectral interpretation and surface classification (Wagner et al., Surf Interface Anal 2002) -- the most widely used multivariate method for ToF-SIMS
- MCR for ToF-SIMS spectral unmixing (Tyler et al., Surf Interface Anal 2007)
- MAF (Maximum Autocorrelation Factor) analysis for ToF-SIMS imaging (Keenan et al., Surf Interface Anal 2004)
- Non-negative matrix factorization (NMF) for ToF-SIMS (Lee et al., 2012)
- Gentle-SIMCA for ToF-SIMS classification (Graham et al., 2006)
- k-means clustering for ToF-SIMS spatial segmentation (2010)
- Random Forest for ToF-SIMS surface classification (2015)
- G-SIMS (Gentle SIMS) spectral simplification (Gilmore & Seah, Appl Surf Sci 2000)

_Deep Learning (2017--2026):_
- CNN for ToF-SIMS spectral classification (Matsuda et al., Anal Chem 2020; 2022)
- Autoencoder for ToF-SIMS dimensionality reduction and denoising (2021)
- Deep learning for NanoSIMS image denoising (2022)
- U-Net for SIMS image segmentation (2023)
- GAN-based ToF-SIMS data augmentation for rare materials (2023)
- Self-supervised learning for ToF-SIMS feature extraction (2024)
- Physics-informed neural network for SIMS depth profile analysis (2024)
- Transformer-based ToF-SIMS spectral classification (2025)
- Deep learning 3D SIMS reconstruction from sparse sputter slices (2025)
- Foundation model for mass spectrometry surface analysis (ToF-SIMS + XPS) (2025--2026)

#### Step 3: Update SIMS Solvers

After listing all SIMS solvers, update `algorithm_base/sims/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SIMS solvers use the data format: `y` (H, W, N_masses) ToF-SIMS imaging datacube or (N_depth, N_masses) depth profiling data, `mass_list` (N_masses,) array of m/z values, `sputter_time` (N_depth,) array of sputter times for depth profiling. The `SIMSOperator` handles the forward model (sputtering + ionization + mass analysis) and spectral/depth processing operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SIMS:**
- NIST implant standards: RSF-quantified concentration within 10% of certified dose for B, As, P in Si
- Depth profile resolution: interface width <2 nm for sharp SiO2/Si interface (state-of-art dynamic SIMS)
- ToF-SIMS PCA: correct classification >90% on VAMAS round-robin polymer samples (Gilmore et al., 2007)
- NanoSIMS isotope ratio: precision <1% for 13C/12C on biological samples
- DL ToF-SIMS classification: accuracy >92% on surface identification benchmarks (Matsuda et al., 2022)
- Published classification accuracy, quantification error, and depth resolution from original papers

**Verification criteria:**
- `done` -- PWM within 3 dB of reference
- `partial` -- 3--10 dB shortfall
- `gap` -- >10 dB shortfall
- `no_ckpt` -- Algorithm documented but pretrained weights not available
- `fail` -- Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sims' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sims/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sims/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sims/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SIMS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sims/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sims/standard/`
