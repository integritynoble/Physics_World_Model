
---

## Astronomy, Geophysics & Experimental Science — Modality Templates

---

### Event Horizon Telescope Imaging (`eht_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For EHT imaging, what dataset do you use to verify? Is this dataset used for EHT popular algorithms? Please ensure the standard dataset in `datasets/benchmark/eht_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original EHT imaging standard dataset.

**Popular datasets to consider:**
- **EHT M87* 2017 Campaign (Event Horizon Telescope Collaboration, 2019)** — the iconic first black hole image dataset; calibrated VLBI visibilities at 230 GHz from eight stations; used by all EHT imaging papers and challenge entries
- **EHT Sgr A* 2017 Campaign (Event Horizon Telescope Collaboration, 2022)** — Galactic Center black hole observations; time-variable source requiring snapshot imaging; the second major EHT science target
- **EHT 2017 Imaging Challenge Synthetic Data (EHT Collaboration, 2019)** — synthetic VLBI datasets with known ground-truth images used to validate imaging pipelines; includes geometric models, GRMHD simulations, and crescent morphologies
- **VLBA Calibrator Survey (Beasley et al., 2002; Petrov et al., 2008)** — compact AGN calibrator sources observed with the Very Long Baseline Array; standard VLBI imaging benchmarks
- **ngEHT Reference Simulations (Doeleman et al., 2023)** — next-generation EHT synthetic datasets with expanded array coverage; used for evaluating future imaging performance

**Decision criteria:** EHT M87* 2017 is the gold standard for VLBI imaging benchmarks (2019-2026); the EHT Imaging Challenge synthetic data provides known ground truth for quantitative evaluation. Use the dataset that appears in the largest number of EHT/VLBI imaging papers.

#### Step 2: List All EHT Imaging Algorithms

Please first ensure all the EHT imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\eht_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/eht_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the EHT imaging solvers, please update the EHT imaging solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1958-2005):_
- Dirty Image (inverse Fourier transform of sampled visibilities) — trivial baseline, always reported
- CLEAN — Hogbom CLEAN deconvolution (Hogbom, A&AS 1974) — the foundational radio interferometric imaging algorithm
- Clark CLEAN — major/minor cycle CLEAN (Clark, A&A 1980)
- Cotton-Schwab CLEAN — wide-field CLEAN with W-projection (Schwab, 1984)
- Multi-Scale CLEAN — extended emission deconvolution (Cornwell, 2008; roots in Wakker & Schwarz 1988)
- Maximum Entropy Method (MEM) — image reconstruction maximizing entropy subject to chi-squared constraint (Cornwell & Evans, A&A 1985; Narayan & Nityananda, ARA&A 1986)
- Self-Calibration (Selfcal) — iterative antenna gain calibration and imaging (Readhead & Wilkinson, 1978; Pearson & Readhead, 1984)
- Bispectrum / Closure Phase Imaging — imaging robust to station-based phase errors (Rogers et al., 1974; Jennison, 1958)

_Regularized Inverse Methods (2014-2020):_
- eht-imaging (ehtim) / RML — Regularized Maximum Likelihood with multiple regularizers (Chael et al., ApJ 2016, 2018) — primary EHT pipeline
- SMILI — Sparse Modeling Imaging Library for Interferometry (Akiyama et al., ApJ 2017) — L1+TSV regularization for EHT
- THEMIS — Bayesian parameter estimation and model comparison for EHT (Broderick et al., ApJ 2020)
- DMC — Dynamic Measurement-set Characterization; Bayesian imaging framework (Pesce, AJ 2021)
- Stochastic Optics — scattering mitigation for Sgr A* (Johnson, ApJ 2016)
- Polarimetric RML imaging — Stokes I/Q/U/V simultaneous reconstruction (Chael et al., ApJ 2016)
- StarWarps — Bayesian time-variable imaging (Bouman et al., 2018)
- Multi-frequency synthesis imaging for EHT (Chael et al., 2023)
- PRIMO — Principal-component Interferometric Modeling (Medeiros et al., ApJL 2023)

_Deep Learning (2019-2026):_
- Deep Horizon — CNN-based EHT image reconstruction (Sun et al., 2020)
- Bayesian Deep Learning for EHT (Sun & Bouman, 2021)
- Score-based diffusion models for VLBI imaging (Feng et al., 2023)
- Variational Image Reconstruction for EHT (Muller & Lobanov, 2023)
- Neural closure phase imaging (Levis et al., 2022)
- Physics-informed neural network for VLBI imaging (2024)
- Foundation model for radio interferometric imaging (2025)

#### Step 3: Update EHT Imaging Solvers

After listing all EHT imaging solvers, update `algorithm_base/eht_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All EHT imaging solvers use the data format: `y` (num_baselines, num_time, num_freq) complex visibilities, `uv` (num_baselines, num_time, 2) baseline coordinates in wavelengths, `sigma` (num_baselines, num_time) thermal noise standard deviations. The `VLBIOperator` handles forward `y = F_nufft * x` and adjoint operations including closure quantities.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for EHT imaging:**
- EHT M87* 2017: Dirty Image ~15 dB, CLEAN ~22 dB, MEM ~25 dB, ehtim/RML ~28 dB, SMILI ~27 dB
- EHT Imaging Challenge (synthetic crescent): CLEAN ~24 dB, ehtim ~32 dB, PRIMO ~33 dB
- EHT Sgr A* 2017 (time-averaged): ehtim ~26 dB, THEMIS ~25 dB
- Metrics: NXCORR (normalized cross-correlation), NRMSE, SSIM on log-brightness images
- All reference values from EHT Collaboration papers and imaging challenge results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'eht_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/eht_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for EHT imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/eht_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/eht_imaging/standard/`

---

### Solar EUV/X-ray Imaging (`solar_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For solar imaging, what dataset do you use to verify? Is this dataset used for solar imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/solar_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original solar imaging standard dataset.

**Popular datasets to consider:**
- **SDO/AIA Level 1.5 Data (Lemen et al., Solar Physics 2012)** — Solar Dynamics Observatory Atmospheric Imaging Assembly; full-disk EUV images in 7 wavelength channels (94-335 A); the most widely used solar imaging dataset since 2010
- **SOHO/EIT Synoptic Archive (Delaboudiniere et al., 1995)** — 26 years of full-disk EUV images at 171, 195, 284, 304 A; the canonical long-baseline solar EUV dataset
- **Hinode/XRT Level 1 Data (Golub et al., 2007)** — high-resolution soft X-ray images of the solar corona; used for DEM reconstruction and coronal loop studies
- **STEREO/EUVI Data (Wuelser et al., 2004)** — dual-viewpoint EUV images enabling 3D reconstruction of coronal structures
- **RHESSI Hard X-ray Visibilities (Lin et al., 2002)** — Reuven Ramaty High Energy Solar Spectroscopic Imager; Fourier-based imaging of solar flares in hard X-rays

**Decision criteria:** SDO/AIA is the undisputed gold standard for solar EUV imaging (2010-2026); RHESSI for hard X-ray image reconstruction from visibilities. Use the dataset that appears in the largest number of solar image reconstruction and enhancement papers.

#### Step 2: List All Solar Imaging Algorithms

Please first ensure all the solar imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\solar_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/solar_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the solar imaging solvers, please update the solar imaging solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1970s-2009):_
- Back-Projection for RHESSI rotating modulation collimator (Hurford et al., Solar Physics 2002)
- CLEAN for RHESSI visibilities (Hurford et al., 2002)
- MEM-NJIT / Maximum Entropy for RHESSI (Schmahl et al., Solar Physics 2007)
- Pixon Method — adaptive pixelization for RHESSI imaging (Metcalf et al., ApJ 1996)
- Filter-ratio DEM reconstruction — temperature maps from multi-channel EUV ratios (Narukage et al., 2011)
- Richardson-Lucy deconvolution for AIA PSF correction (Poduval et al., ApJ 2013)
- Van Cittert deconvolution for EUV image enhancement (DeForest et al., 2009)

_Regularized Inverse Methods (2010-2020):_
- Sparse DEM inversion — L1-regularized Differential Emission Measure reconstruction from multi-channel EUV (Cheung et al., ApJ 2015)
- Regularized DEM inversion — Tikhonov/GSVD-based DEM (Hannah & Kontar, A&A 2012) — widely used DEM code
- xrt_dem_iterative2 — iterative DEM from Hinode/XRT and AIA (Weber et al., 2004; updated Cheng et al.)
- CLEAN + forward-fitting for RHESSI (Dennis & Pernak, 2009)
- VIS_FWDFIT — visibility forward-fitting for RHESSI (Massa et al., 2020)
- Compressed Sensing for RHESSI (Felix et al., 2017)
- Basis Pursuit DEM reconstruction (Plowman et al., ApJ 2013)
- Total Variation regularized solar EUV deconvolution (Schuh et al., 2014)

_Deep Learning (2018-2026):_
- DeepEM — deep learning DEM inversion from SDO/AIA (Su et al., ApJ 2018)
- SolarNet — CNN for solar feature detection and enhancement (Park et al., 2019)
- Image-to-image translation for solar far-side imaging (Kim et al., Nature Astronomy 2019)
- Super-resolution for SDO/AIA (Salvatelli et al., 2022; Diaz Baso et al., 2019)
- Neural network EUV irradiance prediction (Szenicer et al., 2019)
- Physics-informed DEM inversion networks (Wright et al., 2024)
- Diffusion model for solar image super-resolution (2025)
- Virtual observatory — AI-generated missing-channel EUV synthesis (Jarolim et al., 2023)
- STIX imaging with learned regularization (Perracchione et al., 2023)
- Foundation model for multi-wavelength solar imaging (2025)

#### Step 3: Update Solar Imaging Solvers

After listing all solar imaging solvers, update `algorithm_base/solar_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All solar imaging solvers use the data format: `y` (num_channels, H, W) multi-wavelength EUV/X-ray images or (num_detectors, num_time) modulated count rates for RHESSI/STIX, `response_matrix` (num_channels, num_temperature_bins) instrument temperature response functions for DEM problems. The `SolarImagingOperator` handles forward modeling of EUV emission from DEM distributions and RHESSI/STIX visibility transforms.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for solar imaging:**
- SDO/AIA DEM reconstruction (6-channel): Filter ratio ~18 dB, Hannah-Kontar ~24 dB, Cheung Sparse ~26 dB, DeepEM ~28 dB
- RHESSI 25-50 keV flare imaging: Back-Projection ~18 dB, CLEAN ~24 dB, Pixon ~28 dB, MEM-NJIT ~27 dB
- AIA PSF deconvolution: Richardson-Lucy ~30 dB, DL super-resolution ~34 dB
- Metrics: PSNR, SSIM, chi-squared of DEM fit to observed channel intensities, photometric accuracy
- All reference values from published solar physics papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'solar_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/solar_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for solar imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/solar_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/solar_imaging/standard/`

---

### Radio Aperture Synthesis (`radio_astronomy`) Modality Template

#### Step 1: Verify Standard Dataset

For radio aperture synthesis, what dataset do you use to verify? Is this dataset used for radio astronomy popular algorithms? Please ensure the standard dataset in `datasets/benchmark/radio_astronomy/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original radio astronomy standard dataset.

**Popular datasets to consider:**
- **VLA Sky Survey (VLASS, Lacy et al., 2020)** — the most comprehensive modern radio survey; 2-4 GHz continuum; widely used for imaging algorithm benchmarking
- **ALMA Science Verification Data (ALMA Partnership, 2015)** — calibrated visibility data from ALMA commissioning; includes HL Tau protoplanetary disk and other targets; standard sub-mm imaging benchmarks
- **SKA Data Challenge (SKA Organisation, 2019-2024)** — synthetic SKA-scale visibility datasets with known sky models; designed for testing scalable imaging algorithms
- **3C Sources (VLA Archive)** — classic bright radio sources (3C273, 3C84, Cygnus A); canonical imaging targets since the 1960s used by every CLEAN implementation
- **LOFAR Surveys (Shimwell et al., 2017)** — low-frequency (120-168 MHz) survey data requiring direction-dependent calibration; used for modern wide-field imaging benchmarks

**Decision criteria:** ALMA Science Verification data is the most widely used for sub-mm/mm imaging benchmarks; VLA 3C sources are the canonical radio imaging test cases. Use the dataset that appears in the largest number of radio imaging papers.

#### Step 2: List All Radio Astronomy Algorithms

Please first ensure all the radio astronomy algorithms have been listed in `\Physics_World_Model\algorithm_base\radio_astronomy\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/radio_astronomy. Besides, you need to search all algorithms from 1950 to 2026. After listing all the radio astronomy solvers, please update the radio astronomy solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1962-2000):_
- Dirty Image — inverse Fourier transform of gridded visibilities (Ryle & Hewish, 1960)
- Hogbom CLEAN — iterative point-source subtraction (Hogbom, A&AS 1974)
- Clark CLEAN — major/minor cycle acceleration (Clark, A&A 1980)
- Cotton-Schwab CLEAN — image-plane + uv-plane hybrid CLEAN (Schwab, 1984)
- Multi-Frequency Synthesis (MFS) — bandwidth smearing correction and spectral imaging (Conway et al., 1990; Sault & Wieringa, 1994)
- Maximum Entropy Method (MEM/VTESS) — entropy-regularized imaging (Cornwell & Evans, A&A 1985)
- Self-Calibration — iterative gain calibration and imaging (Pearson & Readhead, ARA&A 1984)
- W-Projection — wide-field imaging correcting non-coplanar baselines (Cornwell et al., IEEE JSTSP 2008; roots 2003)

_Advanced Methods (2005-2018):_
- Multi-Scale CLEAN — MS-CLEAN for extended emission (Cornwell, 2008)
- Multi-Scale Multi-Frequency Synthesis (MS-MFS / MT-MFS) (Rau & Cornwell, A&A 2011)
- A-Projection — direction-dependent effects in imaging (Bhatnagar et al., ApJ 2008)
- CASA tclean — the standard radio imaging task (McMullin et al., 2007; CASA team)
- WSClean — fast wide-field imager with w-stacking (Offringa et al., MNRAS 2014) — the most widely used stand-alone radio imager
- MORESANE — compressive sampling for radio imaging (Dabbech et al., MNRAS 2015)
- SARA / PURIFY — sparse regularization via ADMM for radio imaging (Carrillo et al., MNRAS 2012, 2014)
- Faceting — wide-field imaging via faceted sky planes (Cornwell & Perley, 1992)
- uv-Shapelet decomposition (Refregier & Bacon, 2003)

_Deep Learning & Scalable Methods (2019-2026):_
- R2D2 — Residual-to-Residual DNN for radio imaging (Terris et al., MNRAS 2022)
- AIRI — AI for Regularization in Imaging; learned proximal operator for radio imaging (Terris et al., 2023)
- Rascil — Radio Astronomy Simulation, Calibration, and Imaging Library (SKA, 2022)
- DDFacet — direction-dependent faceting imager (Tasse et al., A&A 2018)
- killMS — direction-dependent calibration (Tasse, 2014; Smirnov & Tasse, 2015)
- Deep learning deconvolution for radio interferometry (Connor et al., 2022)
- Score-based diffusion for radio imaging (Drozdova et al., 2024)
- Scalable distributed imaging for SKA (Pratley et al., 2019; Thouvenin et al., 2022)
- Foundation model for radio continuum imaging (2025)

#### Step 3: Update Radio Astronomy Solvers

After listing all radio astronomy solvers, update `algorithm_base/radio_astronomy/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All radio astronomy solvers use the data format: `y` (num_baselines, num_channels, num_polarizations) complex visibilities, `uvw` (num_baselines, 3) baseline coordinates in wavelengths, `weights` (num_baselines, num_channels) visibility weights/flags. The `RadioInterferometerOperator` handles forward `y = DFT * x` or `y = G * W * F * x` (with gridding) and adjoint operations including W-projection and A-projection.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for radio astronomy:**
- ALMA HL Tau continuum: Dirty Image ~18 dB, Hogbom CLEAN ~26 dB, MS-MFS ~30 dB, WSClean ~31 dB, SARA ~33 dB
- VLA Cygnus A: CLEAN ~28 dB, MS-CLEAN ~31 dB, MEM ~30 dB, R2D2 ~35 dB
- SKA Data Challenge 1: WSClean ~28 dB, PURIFY ~31 dB, R2D2 ~34 dB
- Metrics: PSNR, SSIM, dynamic range, flux recovery accuracy, source detection completeness
- All reference values from published radio astronomy papers and SKA challenge results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'radio_astronomy' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/radio_astronomy/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for radio astronomy. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/radio_astronomy/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_astronomy/standard/`

---

### Seismic Tomography (`seismic_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For seismic tomography, what dataset do you use to verify? Is this dataset used for seismic tomography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/seismic_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original seismic tomography standard dataset.

**Popular datasets to consider:**
- **Marmousi2 Model (Martin et al., Geophysics 2006; original Versteeg, 1994)** — the most widely used synthetic seismic velocity model benchmark; complex geological structure with faults and thin layers; standard test for tomographic inversion algorithms
- **SEG/EAGE Salt Model (Aminzadeh et al., 1997)** — 3D salt-body velocity model; canonical benchmark for migration and tomographic methods dealing with strong velocity contrasts
- **SEG/EAGE Overthrust Model (Aminzadeh et al., 1997)** — 3D model with complex geology including overthrust structures; widely used for large-scale tomography testing
- **BP 2004 Velocity Benchmark (Billette & Brandsberg-Dahl, 2004)** — sub-salt imaging benchmark with realistic salt geometry; standard for evaluating velocity model building
- **OpenFWI Datasets (Deng et al., NeurIPS 2022)** — large-scale open benchmarks for seismic FWI with multiple geological settings (FlatVel, CurveVel, Style, FlatFault, CurveFault); designed for deep learning evaluation

**Decision criteria:** Marmousi2 is the undisputed gold standard for seismic velocity inversion benchmarking (1994-2026); SEG/EAGE Salt for salt-body imaging. Use the dataset that appears in the largest number of seismic tomography papers.

#### Step 2: List All Seismic Tomography Algorithms

Please first ensure all the seismic tomography algorithms have been listed in `\Physics_World_Model\algorithm_base\seismic_tomo\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/seismic_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the seismic tomography solvers, please update the seismic tomography solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1970s-2000):_
- Straight-Ray Tomography — linearized traveltime inversion assuming straight rays (Aki & Lee, JGR 1976)
- SIRT — Simultaneous Iterative Reconstruction Technique for seismic traveltime tomography (Trampert & Leveque, JGR 1990)
- ART — Algebraic Reconstruction Technique applied to seismic rays (Gordon et al., 1970; adapted for geophysics)
- Backprojection Tomography — filtered backprojection along raypaths (Humphreys & Clayton, JGR 1988)
- Curved-Ray Tomography — traveltime tomography with raytracing through heterogeneous media (Benz et al., JGR 1996)
- Fresnel Volume Tomography — finite-frequency traveltime tomography (Dahlen et al., GJI 2000; Montelli et al., 2004)
- Surface-Wave Tomography — group/phase velocity dispersion inversion (Ritzwoller & Levshin, JGR 1998)

_Regularized Inverse Methods (2000-2018):_
- Adjoint-State Traveltime Tomography — gradient computation via adjoint methods (Tromp et al., GJI 2005)
- Finite-Frequency Tomography — banana-doughnut kernels for body-wave tomography (Dahlen et al., GJI 2000; Montelli et al., Science 2004)
- Double-Difference Tomography — tomoDD for local earthquake tomography (Zhang & Thurber, BSSA 2003)
- FMTOMO — Fast Marching Tomography (Rawlinson & Sambridge, GJI 2004)
- Ambient Noise Tomography — cross-correlation-based surface-wave tomography (Shapiro et al., Science 2005)
- Bayesian Seismic Tomography — transdimensional McMC inversion (Bodin & Sambridge, JGR 2009; Galetti et al., 2015)
- Multi-scale tomography with adaptive parameterization (Sambridge & Gudmundsson, 1998)
- Eikonal Tomography — traveltime gradient-based surface-wave tomography (Lin et al., GJI 2009)

_Deep Learning (2019-2026):_
- InversionNet — encoder-decoder CNN for seismic velocity inversion (Wu & Lin, Geophysics 2019)
- VelocityGAN — GAN-based seismic velocity model building (Zhang & Alkhalifah, 2020)
- Physics-informed neural network (PINN) for traveltime tomography (Smith et al., 2020; Waheed et al., JGR 2021)
- SeisFlowNet — flow-based generative model for seismic inversion (Zhang et al., 2021)
- Encoder-decoder with uncertainty quantification for tomography (Zhu et al., 2022)
- Neural operator for seismic tomography (Yang et al., 2023)
- Diffusion model prior for seismic velocity estimation (Wang et al., 2024)
- Foundation model for geophysical inversion (2025)

#### Step 3: Update Seismic Tomography Solvers

After listing all seismic tomography solvers, update `algorithm_base/seismic_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All seismic tomography solvers use the data format: `y` (num_sources, num_receivers) traveltime picks or (num_sources, num_receivers, num_time_samples) seismograms, `source_positions` (num_sources, 3) source coordinates, `receiver_positions` (num_receivers, 3) receiver coordinates. The `SeismicTomoOperator` handles forward traveltime computation via raytracing or eikonal solvers and adjoint sensitivity kernel computation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for seismic tomography:**
- Marmousi2 traveltime tomography: Straight-Ray ~15 dB, Curved-Ray ~20 dB, FMTOMO ~24 dB, Adjoint-State ~26 dB
- Marmousi2 velocity recovery: SIRT ~18 dB, Double-Difference ~22 dB, InversionNet ~28 dB
- SEG/EAGE Salt: Straight-Ray ~12 dB, Finite-Frequency ~20 dB, DL methods ~26 dB
- Metrics: PSNR of recovered velocity model, RMSE of velocity, traveltime misfit, structural similarity
- All reference values from published geophysics papers and OpenFWI benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'seismic_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/seismic_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for seismic tomography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/seismic_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/seismic_tomo/standard/`

---

### Full-Waveform Inversion (`fwi`) Modality Template

#### Step 1: Verify Standard Dataset

For full-waveform inversion, what dataset do you use to verify? Is this dataset used for FWI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/fwi/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original FWI standard dataset.

**Popular datasets to consider:**
- **Marmousi2 Model (Martin et al., Geophysics 2006)** — the canonical FWI benchmark; includes complex velocity, density, and attenuation; used by virtually all FWI papers since 1990s
- **OpenFWI Benchmark (Deng et al., NeurIPS 2022)** — large-scale multi-geology synthetic FWI datasets with paired seismograms and velocity models; 12 sub-datasets covering Flat, Curve, Fault, and Style classes
- **BP 2004 Velocity Benchmark (Billette & Brandsberg-Dahl, 2004)** — realistic sub-salt model; standard for evaluating multi-scale FWI strategies
- **Overthrust Model (Aminzadeh et al., 1997)** — 3D complex geological model; used for large-scale 3D FWI benchmarking
- **SEAM Phase I (Fehler & Larner, The Leading Edge 2008)** — Society of Exploration Geophysicists Advanced Modeling; comprehensive 3D earth model with salt, sub-salt, and deepwater geology

**Decision criteria:** Marmousi2 is the undisputed gold standard for FWI benchmarking (1990s-2026); OpenFWI for data-driven/deep learning FWI methods. Use the dataset that appears in the largest number of FWI papers.

#### Step 2: List All FWI Algorithms

Please first ensure all the FWI algorithms have been listed in `\Physics_World_Model\algorithm_base\fwi\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/fwi. Besides, you need to search all algorithms from 1950 to 2026. After listing all the FWI solvers, please update the FWI solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1983-2000):_
- Time-Domain FWI — original formulation via adjoint-state gradient (Tarantola, Geophysics 1984; Gauthier et al., Geophysics 1986) — the foundational FWI paper
- Frequency-Domain FWI — mono-frequency sequential inversion (Pratt et al., Geophysics 1998; Pratt, Geophysics 1999) — breakthrough enabling practical FWI
- Laplace-Domain FWI — damped wavefield inversion for long-wavelength recovery (Shin & Cha, Geophysics 2008)
- Laplace-Fourier Domain FWI — combined Laplace and Fourier domain (Shin & Cha, GJI 2009)
- Gauss-Newton FWI — second-order optimization for FWI (Pratt et al., Geophysics 1998)

_Multi-scale & Regularized Methods (2005-2018):_
- Multi-scale FWI — hierarchical frequency/time progression (Bunks et al., Geophysics 1995; Sirgue & Pratt, Geophysics 2004)
- Envelope-based FWI — low-frequency recovery from envelope misfit (Wu et al., Geophysics 2014)
- Optimal Transport FWI — Wasserstein distance misfit for cycle-skipping mitigation (Engquist & Froese, 2014; Metivier et al., Geophysics 2016)
- Normalized Integration FWI — misfit via integration to reduce cycle skipping (Choi & Alkhalifah, Geophysics 2012)
- Adaptive Waveform Inversion (AWI) — amplitude-free phase inversion (Warner & Guasch, Geophysics 2016)
- Extended/Subsurface-Offset FWI — relaxing the source-consistency constraint (Symes, 2008; van Leeuwen & Herrmann, GJI 2013)
- Wavefield Reconstruction Inversion (WRI) — penalty method relaxing wave equation (van Leeuwen & Herrmann, GJI 2013)
- Total Variation regularized FWI (Anagaw & Sacchi, 2012)
- Elastic FWI — multi-parameter inversion for P-wave, S-wave, density (Virieux & Operto, Geophysics 2009)
- Attenuation/Visco-acoustic FWI — Q-factor inversion (Kamei & Pratt, GJI 2013)

_Deep Learning (2018-2026):_
- InversionNet — end-to-end CNN for FWI (Wu & Lin, Geophysics 2019)
- VelocityGAN — conditional GAN for velocity model building (Zhang & Alkhalifah, GRL 2020)
- UPFWI — unsupervised physics-guided FWI (Jin et al., 2021)
- Physics-Informed Neural Network (PINN) FWI — embedding wave equation in loss (Rasht-Behesht et al., 2022)
- Neural Operator FWI — Fourier Neural Operator for wave propagation (Li et al., 2021; Yang et al., 2023)
- Learned regularization for FWI — plug-and-play priors (Sun & Alkhalifah, 2021)
- Score-based diffusion prior for FWI — generative model regularization (Wang et al., 2024)
- WISE — Waveform Inversion via Spectral Extension (Fang et al., Geophysics 2020)
- Foundation model for seismic inversion (2025)

#### Step 3: Update FWI Solvers

After listing all FWI solvers, update `algorithm_base/fwi/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All FWI solvers use the data format: `y` (num_sources, num_receivers, num_time_samples) seismic shot gathers or (num_sources, num_receivers, num_frequencies) frequency-domain data, `source_wavelet` (num_time_samples,) source signature, `source_positions` (num_sources, 2/3) source coordinates, `receiver_positions` (num_receivers, 2/3) receiver coordinates. The `FWIOperator` handles forward wave simulation (finite-difference, spectral-element) and adjoint wavefield computation for gradient calculation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for FWI:**
- Marmousi2 acoustic FWI (5 Hz start): Time-Domain FWI ~24 dB, Frequency-Domain FWI ~26 dB, Multi-scale FWI ~30 dB, Optimal Transport FWI ~32 dB
- Marmousi2 acoustic FWI (starting from 1D gradient): Multi-scale ~28 dB, AWI ~30 dB, WRI ~31 dB
- OpenFWI FlatVel-A: InversionNet ~32 dB, VelocityGAN ~30 dB, UPFWI ~34 dB
- BP 2004 sub-salt: Multi-scale FWI ~22 dB, Optimal Transport FWI ~26 dB
- Metrics: PSNR of recovered velocity model, RMSE velocity error, structural similarity (SSIM), data misfit reduction
- All reference values from published geophysics papers and OpenFWI leaderboard

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'fwi' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/fwi/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/fwi/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/fwi/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for FWI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/fwi/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/fwi/standard/`

---

### Ocean Acoustic Tomography (`ocean_acoustic_tomo`) Modality Template

#### Step 1: Verify Standard Dataset

For ocean acoustic tomography, what dataset do you use to verify? Is this dataset used for ocean acoustic tomography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ocean_acoustic_tomo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ocean acoustic tomography standard dataset.

**Popular datasets to consider:**
- **ATOC / AMODE Data (Worcester et al., JGR 1999; Dushaw et al., JPO 2009)** — Acoustic Thermometry of Ocean Climate; trans-Pacific and trans-Arctic traveltime data; the canonical large-scale ocean acoustic tomography dataset
- **CANAPE Dataset (Dzieciuch et al., 2017)** — Canada Basin Acoustic Propagation Experiment; Arctic ocean acoustic data with ice cover; used for modern inversion benchmarks
- **Philippine Sea Experiment (Worcester et al., JASA 2013)** — deep-water acoustic propagation and tomography data; benchmark for ray-based and mode-based inversions
- **Heard Island Feasibility Test (Munk et al., JASA 1994)** — first global-scale acoustic thermometry experiment; historical benchmark dataset
- **SWellEx-96 (Murray et al., JASA 2001)** — shallow water evaluation cell experiment; widely used for matched-field processing and geoacoustic inversion benchmarks

**Decision criteria:** ATOC/AMODE is the gold standard for basin-scale ocean acoustic tomography (1990s-2026); SWellEx-96 for shallow-water matched-field inversions. Use the dataset that appears in the largest number of ocean acoustic tomography papers.

#### Step 2: List All Ocean Acoustic Tomography Algorithms

Please first ensure all the ocean acoustic tomography algorithms have been listed in `\Physics_World_Model\algorithm_base\ocean_acoustic_tomo\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ocean_acoustic_tomo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ocean acoustic tomography solvers, please update the ocean acoustic tomography solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1979-2000):_
- Ray-Based Travel-Time Tomography — linearized inversion of ray traveltimes for sound-speed perturbations (Munk & Wunsch, Deep-Sea Research 1979) — the foundational ocean acoustic tomography paper
- Modal Travel-Time Inversion — normal-mode-based traveltime tomography (Shang, JASA 1989)
- Matched-Field Processing (MFP) — ambiguity surface search over environment parameters (Baggeroer et al., IEEE JOE 1993)
- Matched-Field Tomography — joint source localization and environmental inversion (Tolstoy, JASA 1993)
- Born/Rytov Approximation Tomography — diffraction tomography for ocean sound speed (Munk & Wunsch, 1979; Skarsoulis & Cornuelle, JASA 2004)
- SOFAR Channel Tomography — sound fixing and ranging channel inversions (Spiesberger & Metzger, 1991)

_Regularized Inverse Methods (2000-2018):_
- Stochastic Inverse / Gauss-Markov Estimator — optimal estimation with prior covariance (Cornuelle et al., JPO 1985; Munk et al., 1995)
- Sequential Assimilation — Kalman filter-based ocean acoustic tomography (Elisseeff et al., JASA 2002)
- Bayesian Geoacoustic Inversion — McMC sampling for seabed parameters from acoustic data (Dosso et al., JASA 2002)
- Adjoint-Method Ocean Acoustic Tomography — gradient via adjoint parabolic equation (Hursky et al., JASA 2004)
- Sparse Reconstruction for Ocean Tomography — L1-regularized traveltime inversion (Bianco & Gerstoft, JASA 2016)
- Compressive Ocean Acoustic Tomography — undersampled measurements with CS (Bianco et al., JASA 2018)
- Range-Dependent Inversion — adiabatic mode + PE-based inversion (Shang & Wang, 1999)
- Trans-Dimensional Bayesian Inversion for ocean sound speed (Dosso et al., JASA 2014)

_Deep Learning (2019-2026):_
- Deep Learning for Sound Speed Profile Inversion (Bianco et al., JASA 2019) — CNN-based inversion of acoustic data
- Neural Network Matched-Field Processing (Niu et al., JASA 2017; Huang et al., 2018)
- Physics-Informed Neural Network for Ocean Acoustic Inversion (Caldwell et al., 2021)
- LSTM-based Ocean Sound Speed Prediction (Sun et al., 2023)
- Variational Autoencoder for Ocean Acoustic Tomography (2023)
- Encoder-decoder for range-dependent sound speed estimation (2024)
- Foundation model for underwater acoustic inversion (2025)

#### Step 3: Update Ocean Acoustic Tomography Solvers

After listing all ocean acoustic tomography solvers, update `algorithm_base/ocean_acoustic_tomo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ocean acoustic tomography solvers use the data format: `y` (num_source_receiver_pairs, num_arrivals) acoustic traveltimes or (num_receivers, num_frequencies) complex pressure fields for matched-field methods, `source_positions` (num_sources, 3) source coordinates, `receiver_positions` (num_receivers, 3) receiver/hydrophone array coordinates, `bathymetry` (num_range_points,) ocean depth profile. The `OceanAcousticOperator` handles forward acoustic propagation via ray tracing, normal modes, or parabolic equation (PE) solvers and adjoint sensitivity computation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for ocean acoustic tomography:**
- ATOC basin-scale tomography: Ray-Based ~18 dB, Stochastic Inverse ~22 dB, Sequential Kalman ~24 dB
- SWellEx-96 matched-field inversion: MFP (Bartlett) ~15 dB, MFP (MV) ~20 dB, Bayesian McMC ~25 dB, DL-MFP ~27 dB
- Philippine Sea: Modal inversion ~20 dB, Adjoint PE ~24 dB
- Metrics: RMSE of recovered sound speed profile (m/s), traveltime residual RMS (ms), Bartlett power, source localization error
- All reference values from published JASA and IEEE JOE papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ocean_acoustic_tomo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ocean_acoustic_tomo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ocean acoustic tomography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ocean_acoustic_tomo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ocean_acoustic_tomo/standard/`

---

### Gravitational Wave Detection (`gravitational_wave`) Modality Template

#### Step 1: Verify Standard Dataset

For gravitational wave detection, what dataset do you use to verify? Is this dataset used for gravitational wave popular algorithms? Please ensure the standard dataset in `datasets/benchmark/gravitational_wave/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original gravitational wave standard dataset.

**Popular datasets to consider:**
- **LIGO Open Science Center (LOSC/GWOSC) Strain Data (Abbott et al., 2021)** — calibrated strain time series from LIGO Hanford, LIGO Livingston, and Virgo; includes all confirmed detections (GW150914 onwards); the primary dataset for all GW analysis papers
- **MLGWSC-1 (Schafer et al., PRD 2023)** — Machine Learning Gravitational-Wave Search Challenge; standardized dataset for benchmarking ML-based GW detection algorithms with injected signals in colored Gaussian noise
- **LIGO-Virgo O3 Data Release (GWTC-3, Abbott et al., 2023)** — third observing run catalog; ~90 confident detections; the most comprehensive GW event catalog
- **PyCBC/GstLAL Injection Sets (Nitz et al., 2021)** — software injection campaigns with known waveform parameters; standard for detection pipeline sensitivity evaluation
- **Einstein Telescope Mock Data Challenge (Regimbau et al., 2012; ET Collaboration, 2024)** — next-generation detector synthetic data; used for evaluating future detection algorithms

**Decision criteria:** GWOSC O3 strain data is the gold standard for GW detection benchmarking (2019-2026); MLGWSC-1 for ML-based detection methods. Use the dataset that appears in the largest number of gravitational wave detection/parameter estimation papers.

#### Step 2: List All Gravitational Wave Algorithms

Please first ensure all the gravitational wave algorithms have been listed in `\Physics_World_Model\algorithm_base\gravitational_wave\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/gravitational_wave. Besides, you need to search all algorithms from 1950 to 2026. After listing all the gravitational wave solvers, please update the gravitational wave solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1960s-2005):_
- Matched Filtering — optimal detection via cross-correlation with template bank (Wainstein & Zubakov, 1962; Sathyaprakash & Dhurandhar, PRD 1991) — the foundational GW detection method
- Template Bank Construction — geometric placement of templates in parameter space (Owen, PRD 1996; Owen & Sathyaprakash, PRD 1999)
- Excess Power / Burst Search — unmodeled transient detection via time-frequency excess power (Anderson et al., PRD 2001)
- Q-Transform — constant-Q time-frequency representation for GW signal visualization (Brown et al., CQG 2004; Chatterji et al., 2004)
- Bayesian Blocks — adaptive time-frequency tiling for burst detection (Scargle, 1998; adapted for GW)
- Wiener Filtering — noise whitening and optimal linear filtering for GW strain (Allen et al., PRD 2012)

_Pipeline & Bayesian Methods (2005-2018):_
- PyCBC — matched-filter search pipeline with chi-squared signal consistency test (Usman et al., CQG 2016; Nitz et al., 2017)
- GstLAL — matched-filter pipeline using singular value decomposition of templates (Messick et al., PRD 2017)
- MBTA — Multi-Band Template Analysis for low-latency detection (Adams et al., CQG 2016)
- cWB (Coherent WaveBurst) — unmodeled burst search using constrained likelihood in wavelet domain (Klimenko et al., PRD 2008, 2016)
- BayesWave — Bayesian signal vs. glitch discrimination using sine-Gaussian wavelets (Cornish & Littenberg, CQG 2015)
- LALInference — Bayesian parameter estimation via MCMC and nested sampling (Veitch et al., PRD 2015) — standard LVC parameter estimation
- Bilby — modular Bayesian inference for GW parameter estimation (Ashton et al., ApJS 2019)
- RIFT — Rapid Parameter Inference via Iterative Fitting (Pankow et al., PRD 2015; Lange et al., PRD 2018)
- Power Spectral Density estimation — BayesLine and median-mean methods (Littenberg & Cornish, PRD 2015)

_Deep Learning (2017-2026):_
- Deep Filtering — 1D CNN for GW detection (George & Huerta, PLB 2018) — first deep learning GW detection paper
- GW signal denoising with autoencoders (Shen et al., 2019; Ormiston et al., PRL 2020)
- CGAN for GW signal generation (McGinn et al., 2021)
- Neural posterior estimation (NPE) / simulation-based inference for GW parameter estimation (Green et al., ML:ST 2020; Dax et al., PRL 2021)
- DINGO — Deep Inference for Gravitational-wave Observations; normalizing flow PE (Dax et al., PRL 2021)
- Aframe — attention-based GW detection (Chatterjee et al., 2023)
- Vitamin — variational inference for GW transient analysis (Gabbard et al., Nature Physics 2022)
- Transformer for GW signal detection (Ravichandran et al., 2023)
- Real-time ML GW detection pipelines for O4 (2024)
- Foundation model for multi-messenger GW astrophysics (2025)
- Diffusion model for GW waveform generation (McGinn et al., 2024)

#### Step 3: Update Gravitational Wave Solvers

After listing all gravitational wave solvers, update `algorithm_base/gravitational_wave/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All gravitational wave solvers use the data format: `y` (num_detectors, num_time_samples) strain time series at sample rate 2048-16384 Hz, `psd` (num_detectors, num_freq_bins) one-sided power spectral density, `template_bank` (num_templates, num_time_samples) precomputed waveform templates (for matched filtering). The `GWOperator` handles forward waveform generation via IMRPhenom/EOB/NR surrogate models, noise-weighted inner product computation, and time-frequency transforms.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for gravitational wave:**
- GWOSC O1 (GW150914 recovery): Matched Filter SNR ~24, PyCBC FAR < 1/100yr, cWB detection significance > 5 sigma
- MLGWSC-1 (BBH at SNR 8-12): PyCBC sensitivity ~95%, GstLAL ~93%, Deep Filtering CNN ~88%, Aframe ~91%
- Parameter estimation (GW150914): LALInference chirp mass 1% accuracy, Bilby 1% accuracy, DINGO 2% accuracy at 1000x speedup
- Metrics: detection sensitivity (true positive rate at fixed FAR), matched filter SNR recovery, parameter estimation accuracy (posterior width), computational latency
- All reference values from published LVC papers and MLGWSC-1 challenge results

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'gravitational_wave' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/gravitational_wave/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for gravitational wave detection. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/gravitational_wave/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/gravitational_wave/standard/`

---

### Particle Calorimetry (`particle_calorimetry`) Modality Template

#### Step 1: Verify Standard Dataset

For particle calorimetry, what dataset do you use to verify? Is this dataset used for particle calorimetry popular algorithms? Please ensure the standard dataset in `datasets/benchmark/particle_calorimetry/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original particle calorimetry standard dataset.

**Popular datasets to consider:**
- **CaloChallenge 2022 Datasets (Krause et al., 2022)** — Calochallenge standardized datasets for fast calorimeter simulation; three datasets of increasing complexity (photons and pions in different detector geometries); the primary benchmark for generative calorimeter models
- **ATLAS Open Data for Jet Reconstruction (ATLAS Collaboration, 2020)** — topological cluster and jet reconstruction data from ATLAS calorimeters; used for particle flow and jet calibration studies
- **CMS Open Data Calorimeter (CMS Collaboration, 2021)** — ECAL and HCAL cell-level data from CMS; used for shower reconstruction and energy calibration benchmarks
- **ILD Calorimeter Simulation (ILC TDR, 2013)** — International Large Detector full simulation of highly granular calorimeters; standard for particle flow algorithm development
- **Geant4 Calorimeter Shower Libraries (Geant4 Collaboration)** — parameterized and full shower simulation libraries; canonical ground truth for calorimeter response modeling

**Decision criteria:** CaloChallenge 2022 is the gold standard for calorimeter simulation benchmarking (2022-2026); ATLAS/CMS open data for real-detector reconstruction. Use the dataset that appears in the largest number of calorimetry ML/reconstruction papers.

#### Step 2: List All Particle Calorimetry Algorithms

Please first ensure all the particle calorimetry algorithms have been listed in `\Physics_World_Model\algorithm_base\particle_calorimetry\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/particle_calorimetry. Besides, you need to search all algorithms from 1950 to 2026. After listing all the particle calorimetry solvers, please update the particle calorimetry solver.

**Key algorithms to cover (1950-2026):**

_Analytic / Classical (1960s-2005):_
- Sampling Calorimetry Calibration — energy reconstruction from sampled ionization E = sum(w_i * s_i) (Wigmans, Calorimetry 2000)
- Software Compensation — hadronic energy correction using cell energy density weighting (ATLAS/CMS, 1990s)
- Topological Clustering (TopoClusters) — noise-suppressed 3D clustering of calorimeter cells (ATLAS Collaboration, 2017; roots 1990s)
- Sliding-Window Clustering — fixed-cone electron/photon clustering (ATLAS, 1996)
- H1-style Weighting — local hadronic calibration weighting (H1 Collaboration, 1993; adapted by ATLAS)
- Electromagnetic Shower Parameterization — longitudinal/lateral profile fitting (Longo & Sestili, NIM 1975; Grindhammer et al., 1990)

_Particle Flow & Advanced Methods (2005-2020):_
- Particle Flow Algorithm (PFA) / PandoraPFA — individual particle reconstruction combining tracker and calorimeter (Thomson, NIM A 2009; Marshall et al., 2015) — the standard for ILC/CLIC detectors
- ATLAS Particle Flow (2017) — combined tracking + calorimeter reconstruction for ATLAS
- CMS Particle Flow (CMS Collaboration, Sirunyan et al., JINST 2017) — the CMS reconstruction paradigm
- Local Hadronic Calibration (LC) — cell-level dead material and out-of-cluster corrections (ATLAS, 2009)
- Global Sequential Calibration (GSC) — jet-level multivariate calibration (ATLAS, 2015)
- Jet Energy Scale (JES) Calibration — in-situ methods using Z+jet, gamma+jet, multijet balance (ATLAS/CMS standard)
- Gaussian Mixture Model shower parameterization (Wigmans & Zeyrek, 2002)
- BDT-based energy regression for electrons/photons (CMS, 2013)

_Deep Learning (2017-2026):_
- CaloGAN — GAN-based fast calorimeter simulation (de Oliveira et al., Computing and Software for Big Science 2018; Paganini et al., PRL 2018) — first GAN for calorimetry
- CaloFlow — normalizing flow for fast shower generation (Krause & Shih, PRD 2021)
- CaloScore — score-based diffusion for calorimeter simulation (Mikuni & Nachman, PRD 2023)
- CaloDiffusion — diffusion model for calorimeter showers (Amram & Shih, PRD 2023)
- CaloPointFlow — point-cloud flow for variable-size showers (Kansal et al., 2023)
- Graph Neural Network for Particle Flow (Qasim et al., EPJC 2019; Pata et al., EPJC 2021) — MLPF
- Object Condensation for calorimeter clustering (Kieseler, 2020; Qasim et al., 2022)
- Attention-based calorimeter reconstruction (Mikuni & Nachman, 2021)
- L2LFlows — layer-by-layer normalizing flows for calorimeter simulation (Buckley et al., 2023)
- SuperCalo — super-resolution for fast calorimeter upsampling (2024)
- Foundation model for detector simulation (2025)
- Classifier-based calibration and unfolding (Andreassen et al., PRL 2020)

#### Step 3: Update Particle Calorimetry Solvers

After listing all particle calorimetry solvers, update `algorithm_base/particle_calorimetry/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All particle calorimetry solvers use the data format: `y` (num_events, num_layers, num_cells_eta, num_cells_phi) calorimeter cell energy deposits or (num_events, num_hits, 4) point cloud format (x, y, z, E), `particle_type` (num_events,) incident particle ID, `incident_energy` (num_events,) true particle energy, `incident_angle` (num_events,) incidence angle. The `CaloOperator` handles forward shower simulation via Geant4 interface or parameterized models and energy/position reconstruction from cell-level data.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for particle calorimetry:**
- CaloChallenge Dataset 1 (photon, low granularity): Geant4 baseline, CaloGAN FPD ~20, CaloFlow FPD ~5, CaloScore FPD ~3, CaloDiffusion FPD ~2
- CaloChallenge Dataset 2 (photon, high granularity): CaloFlow FPD ~15, CaloScore FPD ~8, CaloDiffusion FPD ~6
- CaloChallenge Dataset 3 (pion, high granularity): CaloFlow FPD ~25, L2LFlows FPD ~10, CaloDiffusion FPD ~12
- Particle Flow jet energy resolution: TopoClusters sigma/E ~12%, PandoraPFA sigma/E ~4%, MLPF sigma/E ~3.5%
- Metrics: Frechet Physics Distance (FPD), KPD, shower shape distributions (longitudinal/lateral profiles), energy resolution sigma(E)/E, separation power, generation time speedup over Geant4
- All reference values from CaloChallenge leaderboard and published HEP papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'particle_calorimetry' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/particle_calorimetry/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for particle calorimetry. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/particle_calorimetry/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/particle_calorimetry/standard/`
