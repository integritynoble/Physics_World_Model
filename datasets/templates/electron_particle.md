
---

## Electron Microscopy & Coherent Imaging — Modality Templates

---

### TEM (`tem`) Modality Template

#### Step 1: Verify Standard Dataset

For TEM, what dataset do you use to verify? Is this dataset used for TEM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/tem/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original TEM standard dataset.

**Popular datasets to consider:**
- **NIST TEM Benchmark (NIST, 2015–ongoing)** — calibrated TEM images of standard reference materials (Au nanoparticles, Si lattice) with known lattice spacings; widely used for resolution and contrast verification
- **HRTEM Simulation Benchmark (Kirkland, 2010)** — multislice-simulated TEM image stacks with known atomic structure ground truth; the canonical numerical benchmark for image simulation codes (QSTEM, Dr. Probe, abTEM)
- **Materials Data Facility — TEM Collections (Blaiszik et al., 2016)** — curated high-resolution TEM datasets of metals, ceramics, and nanostructures; used for deep learning segmentation and defect detection
- **EMNIST / Atomic Resolution TEM Dataset (Ziatdinov et al., 2017)** — labeled atomic-resolution STEM/TEM images for machine learning; defect identification in 2D materials (graphene, MoS2)
- **CTF Challenge Dataset (Ophus et al., 2016)** — TEM images with known CTF parameters for benchmarking CTF correction methods

**Decision criteria:** NIST TEM Benchmark and HRTEM Simulation Benchmark (Kirkland multislice) are the gold standards for TEM image fidelity verification. Multislice simulations with known ground truth are essential for quantitative algorithm validation. Use the dataset that appears in the largest number of TEM reconstruction/simulation papers.

#### Step 2: List All TEM Algorithms

Please first ensure all the TEM algorithms have been listed in `\pwm\public\algorithm_base\tem\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/tem. Besides, you need to search all algorithms from 1950 to 2026. After listing all the TEM solvers, please update the TEM solver.

**Key algorithms to cover (1950–2026):**

_Image Formation & Simulation (1957–2020):_
- Multislice algorithm — iterative slice-by-slice electron wave propagation through a specimen (Cowley & Moodie, Acta Cryst 1957; Goodman & Moodie, 1974) — the foundational TEM image simulation method
- Bloch wave method — dynamical diffraction calculation using eigenvalue decomposition (Bethe, Ann Phys 1928; Humphreys, Rep Prog Phys 1979)
- Frozen phonon multislice — thermal diffuse scattering via ensemble averaging of displaced atom configurations (Loane et al., Acta Cryst A 1991)
- QSTEM — quantitative TEM/STEM simulation package (Koch, 2002)
- Dr. Probe — multislice TEM/STEM simulation with PRISM support (Barthel, Ultramicroscopy 2018)
- abTEM — ab initio TEM simulation using GPU-accelerated multislice (Madsen & Susi, Open Res Europe 2021)
- PRISM algorithm — fast STEM simulation via plane-wave reciprocal-space interpolation (Ophus, Adv Struct Chem Imaging 2017)

_CTF Correction & Phase Recovery (1968–2026):_
- Wiener filter CTF correction — optimal linear deconvolution of CTF from TEM images (Frank, 1973)
- Focal series reconstruction — exit-wave reconstruction from through-focus TEM image series (Coene et al., PRL 1992; Thust et al., Ultramicroscopy 1996)
- Gerchberg-Saxton algorithm — iterative phase retrieval from intensity measurements (Gerchberg & Saxton, Optik 1972)
- Maximum likelihood exit-wave reconstruction (Coene et al., Ultramicroscopy 1996; Op de Beeck et al., 1996)
- Transport of Intensity Equation (TIE) — phase retrieval from defocus series (Teague, JOSA 1983; Paganin & Nugent, PRL 1998)
- Ptychographic reconstruction for TEM — iterative overlap-scan phase retrieval (Rodenburg et al., PRL 2007)

_Denoising & Enhancement (2005–2026):_
- Non-local means denoising for TEM (Buades et al., 2005; adapted for EM, Mevenkamp et al., 2015)
- BM3D for TEM image denoising (Dabov et al., 2007; applied to EM)
- Deep learning denoising — Noise2Noise (Lehtinen et al., ICML 2018), Noise2Void (Krull et al., CVPR 2019), Topaz-Denoise (Bepler et al., Nat Commun 2020)
- Atomic column detection via CNN — AtomNet / atom finding neural networks (Ziatdinov et al., ACS Nano 2017)
- Super-resolution TEM via deep learning (de Haan et al., 2019)
- Foundation model for TEM image analysis (2025–2026)

_Segmentation & Analysis (2010–2026):_
- Watershed segmentation for nanoparticle analysis (Beucher & Lantuejoul, 1979; applied to TEM)
- U-Net for TEM segmentation (Ronneberger et al., MICCAI 2015; adapted for materials EM)
- ASTAR crystal orientation mapping — automated crystal phase/orientation from diffraction (Rauch et al., Ultramicroscopy 2005)
- Graph neural networks for defect detection in TEM (2023)
- Vision transformer for TEM defect classification (2024)

#### Step 3: Update TEM Solvers

After listing all TEM solvers, update `algorithm_base/tem/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All TEM solvers use the data format: `y` (H, W) or (N, H, W) TEM image or focal series stack, `ctf_params` (defocus, Cs, voltage, aperture) CTF parameters, `specimen_potential` (Nz, H, W) 3D electrostatic potential slices for multislice. The `TEMOperator` handles forward (multislice propagation -> CTF -> detector) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for TEM:**
- Multislice Si [110] 200kV: lattice fringe spacing within 0.5% of known 1.36 A d-spacing
- Focal series exit-wave reconstruction: phase accuracy <0.1 rad on NIST Au nanoparticle benchmark
- CTF correction: PSNR >35 dB on simulated data with known ground truth
- Deep learning denoising: PSNR improvement of 3–6 dB over Wiener filter on low-dose TEM
- Segmentation: Dice >0.90 on nanoparticle TEM benchmark
- All reference values from published papers and simulation ground truth

**Verification criteria:**
- `done` — PWM within 3 dB PSNR (image tasks) or 0.5% lattice accuracy (simulation) of reference
- `partial` — 3–10 dB shortfall or 0.5–2% lattice error
- `gap` — >10 dB shortfall or >2% lattice error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'tem' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/tem/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/tem/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/tem/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for TEM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/tem/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/tem/standard/`

---

### STEM (`stem`) Modality Template

#### Step 1: Verify Standard Dataset

For STEM, what dataset do you use to verify? Is this dataset used for STEM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/stem/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original STEM standard dataset.

**Popular datasets to consider:**
- **Ronchigram Benchmark Dataset (Lupini et al., 2010)** — calibrated STEM Ronchigrams for aberration measurement validation; used to benchmark aberration corrector alignment algorithms
- **STEM-HAADF Atomic Resolution Benchmark (LeBeau et al., 2008)** — quantitative HAADF-STEM images of SrTiO3 with absolute intensity calibration; the standard for quantitative STEM
- **Materials Data Facility — STEM Collections (Blaiszik et al., 2016)** — curated aberration-corrected STEM datasets of oxides, metals, alloys; used for ML-based analysis
- **2D Materials STEM Dataset (Ziatdinov et al., 2017; Maksov et al., 2019)** — labeled atomic-resolution STEM images of graphene, MoS2, WS2; atom-by-atom annotation for ML training
- **NIST STEM Reference Images (NIST, 2018)** — standard reference materials imaged under calibrated STEM conditions; traceable resolution and contrast metrics

**Decision criteria:** STEM-HAADF quantitative benchmark (LeBeau, SrTiO3) is the gold standard for quantitative STEM validation. 2D materials datasets (Ziatdinov) are the standard for ML-based atomic analysis. Use the dataset that appears in the largest number of STEM analysis papers.

#### Step 2: List All STEM Algorithms

Please first ensure all the STEM algorithms have been listed in `\pwm\public\algorithm_base\stem\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/stem. Besides, you need to search all algorithms from 1950 to 2026. After listing all the STEM solvers, please update the STEM solver.

**Key algorithms to cover (1970–2026):**

_HAADF/ADF Imaging & Quantification (1970–2020):_
- HAADF-STEM Z-contrast imaging — high-angle annular dark field for compositional contrast (Crewe et al., Science 1970; Pennycook & Jesson, PRL 1990)
- ABF-STEM — annular bright field for light-atom imaging (Findlay et al., APL 2009; Okunishi et al., Microsc Microanal 2009)
- Quantitative HAADF — absolute intensity calibration via detector normalization (LeBeau & Stemmer, Ultramicroscopy 2008)
- Atom counting from HAADF — statistical decomposition of integrated column intensities (Van Aert et al., Ultramicroscopy 2011)
- Multislice STEM simulation — frozen-phonon HAADF/ABF simulation (Kirkland, 2010; Koch, 2002)
- PRISM — fast STEM simulation via reciprocal-space interpolation (Ophus, Adv Struct Chem Imaging 2017)

_Aberration Correction & Diagnostics (1998–2020):_
- Zemlin tableau — aberration measurement from diffractogram tilt series (Zemlin et al., Ultramicroscopy 1978)
- Ronchigram analysis — aberration diagnosis from convergent-beam shadow images (Lupini et al., 2010)
- Automated aberration correction — feedback-loop correction using Ronchigram or diffractogram analysis (Krivanek et al., Ultramicroscopy 1999; Haider et al., Nature 1998)
- ptychographic aberration measurement — post-acquisition aberration determination (Yang et al., Ultramicroscopy 2015)

_Ptychography & Phase Retrieval (2012–2026):_
- Electron ptychography — focused-probe 4D-STEM ptychographic reconstruction (Jiang et al., Nature 2018; Chen et al., Science 2021) — achieved sub-angstrom resolution
- Single-sideband (SSB) ptychography — linear phase retrieval from segmented detector (Pennycook et al., Ultramicroscopy 2015)
- Wigner Distribution Deconvolution (WDD) — phase retrieval via Wigner function (Rodenburg & Bates, Phil Trans 1992)
- Iterative ptychographic engines — ePIE (Maiden & Rodenburg, Ultramicroscopy 2009), rPIE, ML-based ptychography
- Multi-slice ptychography — depth-resolved phase retrieval (Chen et al., Science 2021)
- Mixed-state ptychography — partial coherence modeling (Thibault & Menzel, Nature 2013)

_Deep Learning for STEM (2017–2026):_
- Deep learning STEM denoising — Noise2Atom (Ede & Beanland, 2020), self-supervised denoising for atomic STEM
- AtomSegNet — atomic column segmentation and quantification (Lin et al., 2021)
- STEM super-resolution via deep learning (de Haan et al., 2019)
- Defect detection in STEM via CNN/GNN (Ziatdinov et al., ACS Nano 2017; Li et al., npj Comp Mat 2023)
- Automated STEM simulation parameter fitting via neural networks (2023)
- Foundation model for STEM image interpretation (2025–2026)

_Spectroscopic STEM (2010–2026):_
- Spectrum imaging — EELS/EDX acquisition at each STEM probe position (Jeanguillaume & Colliex, Ultramicroscopy 1989)
- Compressed sensing spectrum imaging — sub-Nyquist STEM-EELS/EDX (Stevens et al., Microscopy 2014)
- Multivariate statistical analysis (MSA) / PCA for spectrum images (Bonnet, Ultramicroscopy 1990)
- Non-negative matrix factorization for STEM spectral unmixing (Shiga et al., 2016)

#### Step 3: Update STEM Solvers

After listing all STEM solvers, update `algorithm_base/stem/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All STEM solvers use the data format: `y` (H, W) HAADF/ABF image or (Ny, Nx, Ky, Kx) 4D-STEM dataset, `probe` (Ky, Kx) convergent probe function, `detector_geometry` inner/outer angles for ADF. The `STEMOperator` handles forward (probe convolution -> scattering -> detector integration) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for STEM:**
- Quantitative HAADF SrTiO3: column intensity ratio Sr/Ti within 5% of Bloch wave simulation
- PRISM simulation: agreement with multislice within 1% intensity for benchmark structures
- Electron ptychography: phase precision <10 mrad on thin specimen benchmark
- Deep learning denoising: PSNR improvement of 4–8 dB on low-dose STEM
- Atom counting: correct column count for >95% of columns on SrTiO3 benchmark
- All reference values from published papers and simulation ground truth

**Verification criteria:**
- `done` — PWM within 5% intensity accuracy or 10 mrad phase accuracy of reference
- `partial` — 5–15% intensity error or 10–50 mrad phase error
- `gap` — >15% intensity error or >50 mrad phase error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'stem' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/stem/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/stem/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/stem/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for STEM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/stem/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/stem/standard/`

---

### SEM (`sem`) Modality Template

#### Step 1: Verify Standard Dataset

For SEM, what dataset do you use to verify? Is this dataset used for SEM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sem/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SEM standard dataset.

**Popular datasets to consider:**
- **NIST SEM Resolution Standard (NIST, 2014)** — calibrated SEM images of tin-on-carbon resolution test specimens; traceable resolution measurement using SMART algorithm; the accepted standard for SEM resolution benchmarking
- **SEM Dataset for Segmentation (Arganda-Carreras et al., 2015)** — ISBI segmentation challenge SEM dataset of neural tissue (ssTEM); widely used for EM segmentation benchmarks
- **Semiconductor Defect SEM Dataset (SEMI, ongoing)** — SEM images of semiconductor wafer defects (particle, scratch, pattern defect); used for automated defect review benchmarking
- **Powder Morphology SEM Dataset (NIST, 2020)** — SEM images of metal powders with annotated particle boundaries; used for particle size analysis algorithm benchmarking
- **Materials Microstructure SEM Dataset (Holm et al., 2020)** — UHCS (Ultra-High Carbon Steel) SEM micrographs with labeled microstructural constituents; canonical benchmark for ML microstructure classification

**Decision criteria:** NIST SEM Resolution Standard is the gold standard for SEM imaging performance. ISBI ssTEM dataset (Arganda-Carreras) is the standard for segmentation. UHCS dataset (Holm) for microstructure classification. Use the dataset that appears in the largest number of SEM analysis papers.

#### Step 2: List All SEM Algorithms

Please first ensure all the SEM algorithms have been listed in `\pwm\public\algorithm_base\sem\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sem. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SEM solvers, please update the SEM solver.

**Key algorithms to cover (1965–2026):**

_Image Formation & Signal Detection (1965–2010):_
- Secondary electron (SE) imaging — surface topography contrast from low-energy secondaries (Everhart & Thornley, J Sci Instrum 1960; Joy, 1995)
- Backscattered electron (BSE) imaging — compositional/channeling contrast from high-energy backscatters (Wells, 1974)
- Monte Carlo electron trajectory simulation — electron-matter interaction modeling (Joy, 1995; Casino, Drouin et al., 2007; MCNP-based)
- CASINO simulation — Monte Carlo simulation of electron trajectories in solids (Drouin et al., Scanning 2007)
- JMONSEL — Java Monte Carlo simulation of secondary electron emission (Villarrubia & Ding, NIST 2009)
- Charge contrast imaging — for insulating specimens (Cazaux, J Appl Phys 2004)

_Resolution & Metrology (1986–2026):_
- SMART algorithm — SEM resolution measurement via spectral analysis (Joy, 1986; ISO 2014)
- CD-SEM linewidth metrology — critical dimension measurement from SEM profiles (Villarrubia, JVST B 2005)
- Model-based SEM metrology — physics-based edge detection using electron-matter interaction models (Villarrubia & Vladár, Ultramicroscopy 2010)
- Contour-based CD measurement — edge extraction and averaging for semiconductor patterns (Mack, SPIE 2013)
- Machine learning CD-SEM — deep learning for critical dimension measurement (2020–2026)

_Segmentation & Feature Extraction (2005–2026):_
- Trainable Weka Segmentation (TWS) — machine learning pixel classification (Arganda-Carreras et al., Bioinformatics 2017)
- U-Net for SEM segmentation — encoder-decoder CNN for EM image segmentation (Ronneberger et al., MICCAI 2015)
- Mask R-CNN for SEM particle/defect instance segmentation (He et al., ICCV 2017; applied to SEM)
- Watershed + marker-controlled segmentation for grain boundaries (Vincent & Soille, IEEE TPAMI 1991)
- SegNet / DeepLab for SEM microstructure segmentation (2018)
- Vision Transformer (ViT) for SEM image classification (Dosovitskiy et al., ICLR 2021; applied to materials SEM)
- Segment Anything Model (SAM) adapted for SEM (2023–2024)

_Denoising & Enhancement (2010–2026):_
- Frame averaging and integration — classical SNR improvement by averaging multiple scans
- Non-local means for SEM denoising (Buades et al., 2005; applied to SEM)
- Deep learning SEM denoising — Noise2Noise, self-supervised approaches (2019–2024)
- GAN-based SEM image enhancement — super-resolution and denoising (2020)
- SEM image colorization and style transfer for visualization (2021)

_Automated Defect Inspection (2015–2026):_
- Template matching defect detection — pattern comparison for semiconductor inspection
- CNN-based defect classification — ResNet/VGG for wafer defect review (2017)
- Anomaly detection — autoencoder and GAN-based defect detection (2019)
- YOLOv5/v8 for real-time SEM defect detection (2022–2024)
- Foundation model for semiconductor SEM inspection (2025–2026)

#### Step 3: Update SEM Solvers

After listing all SEM solvers, update `algorithm_base/sem/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SEM solvers use the data format: `y` (H, W) SEM image (SE or BSE), `beam_params` (voltage, current, working_distance, spot_size), `detector_type` (SE, BSE, InLens). The `SEMOperator` handles forward (beam-specimen interaction -> signal generation -> detection) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SEM:**
- SMART resolution measurement: reproducible within 5% on NIST tin-on-carbon standard
- ISBI ssTEM segmentation: U-Net Rand error ~0.045, Warping error ~0.0003
- UHCS microstructure classification: accuracy >93% (DeCost & Holm, 2017)
- Monte Carlo simulation: BSE yield within 3% of experimental values for standard materials
- Deep learning denoising: PSNR improvement of 3–5 dB on low-dose SEM
- All reference values from published papers and benchmark competitions

**Verification criteria:**
- `done` — PWM within 5% of reference metric values
- `partial` — 5–15% shortfall
- `gap` — >15% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sem' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sem/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sem/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sem/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SEM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sem/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sem/standard/`

---

### FIB-SEM (`fib_sem`) Modality Template

#### Step 1: Verify Standard Dataset

For FIB-SEM, what dataset do you use to verify? Is this dataset used for FIB-SEM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/fib_sem/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original FIB-SEM standard dataset.

**Popular datasets to consider:**
- **FIBSEM Drosophila Brain (Zheng et al., Cell 2018)** — full adult Drosophila brain FIB-SEM volume at 4x4x4 nm voxels; the landmark connectomics dataset; used for segmentation and neuron tracing benchmarks
- **OpenOrganelle / COSEM (Heinrich et al., Nature 2021)** — multi-organelle FIB-SEM volumes of cultured cells with dense annotation; the reference benchmark for organelle segmentation from FIB-SEM
- **CREMI Challenge Dataset (2016–ongoing)** — serial-section EM neurite segmentation challenge; 5x5x5 nm FIB-SEM-like volumes with dense synapse and neurite annotations
- **MICrONS Dataset (Consortium, 2021)** — cubic millimeter mouse visual cortex; FIB-SEM-derived connectome; largest mammalian connectomics dataset
- **EMPIAR FIB-SEM Entries (2018–ongoing)** — community-deposited FIB-SEM volumes for cell biology

**Decision criteria:** OpenOrganelle/COSEM (Heinrich et al., 2021) is the gold standard for FIB-SEM organelle segmentation benchmarking. Drosophila brain (Zheng et al., 2018) for connectomics. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All FIB-SEM Algorithms

Please first ensure all the FIB-SEM algorithms have been listed in `\pwm\public\algorithm_base\fib_sem\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/fib_sem. Besides, you need to search all algorithms from 1950 to 2026. After listing all the FIB-SEM solvers, please update the FIB-SEM solver.

**Key algorithms to cover (1990–2026):**

_Volume Acquisition & Alignment (1990–2020):_
- Serial FIB milling + SEM imaging — automated slice-and-view volume acquisition (Heymann et al., JSB 2006)
- Cross-correlation stack alignment — rigid translation alignment of serial sections (Kremer et al., JSB 1996)
- SIFT-based elastic alignment — scale-invariant feature matching for non-rigid registration (Lowe, IJCV 2004; applied to FIB-SEM, Cardona et al., 2012)
- TrakEM2 — elastic montaging and alignment for EM volumes (Cardona et al., PLoS One 2012)
- Linear stack alignment with SIFT — FIJI/ImageJ plugin (Saalfeld et al., 2012)
- Intensity normalization across slices — histogram matching and CLAHE for FIB-SEM stacks

_3D Segmentation (2012–2026):_
- Flood-filling networks (FFN) — recurrent neural network for neurite segmentation (Januszewski et al., Nat Methods 2018) — state-of-the-art for connectomics
- 3D U-Net for volumetric segmentation (Cicek et al., MICCAI 2016; applied to FIB-SEM)
- Cellpose — generalist cell segmentation adapted for EM (Stringer et al., Nat Methods 2021)
- SegCLR — contrastive learning for EM segmentation (Xenes et al., 2023)
- Watershed + agglomeration — over-segmentation followed by region merging (Funke et al., MICCAI 2019)
- MALA — agglomeration framework for neuron segmentation (Funke et al., 2019)
- Mask R-CNN for organelle instance segmentation in FIB-SEM (He et al., 2017; applied to COSEM)
- StarDist 3D — star-convex polyhedra for 3D nuclei detection (Weigert et al., 2020)
- Empanada — panoptic segmentation for EM volumes (Conrad & Bhatt, 2023)
- Vision Transformer for volumetric EM segmentation (2024)
- Segment Anything 3D for FIB-SEM (2024–2025)

_Surface Reconstruction & Visualization (2005–2026):_
- Marching cubes — isosurface extraction from segmented volumes (Lorensen & Cline, SIGGRAPH 1987)
- IMOD — 3D reconstruction and visualization for EM tomography (Kremer et al., JSB 1996)
- Amira/Avizo — commercial 3D visualization and analysis (Stalling et al., 2005)
- Neuroglancer — web-based volumetric data viewer (Maitin-Shepard, 2021)

_Artifact Correction (2015–2026):_
- Curtaining artifact removal — FIB milling streak correction via wavelet filtering (Liu et al., Ultramicroscopy 2018)
- Charging artifact correction — adaptive histogram equalization and destriping
- Deep learning artifact removal — CNN-based curtaining and charging correction (2022)
- Missing slice interpolation — frame interpolation for skipped/damaged slices (2020)

#### Step 3: Update FIB-SEM Solvers

After listing all FIB-SEM solvers, update `algorithm_base/fib_sem/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All FIB-SEM solvers use the data format: `y` (Nz, H, W) volumetric FIB-SEM stack (serial sections), `voxel_size` (dz, dy, dx) voxel dimensions in nm, `slice_metadata` per-slice milling and imaging parameters. The `FIBSEMOperator` handles forward (3D structure -> serial sectioning -> SEM imaging) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for FIB-SEM:**
- CREMI neurite segmentation: FFN CREMI score ~0.15, adapted Rand error ~0.08
- COSEM organelle segmentation: 3D U-Net IoU >0.75 for mitochondria, >0.65 for ER
- Alignment: SIFT-based registration <1 pixel residual on Drosophila brain dataset
- Curtaining removal: PSNR improvement >5 dB on synthetic curtaining benchmark
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 10% relative error of reference segmentation metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'fib_sem' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/fib_sem/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/fib_sem/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/fib_sem/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for FIB-SEM. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/fib_sem/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/fib_sem/standard/`

---

### Cryo-ET (`cryo_et`) Modality Template

#### Step 1: Verify Standard Dataset

For Cryo-ET, what dataset do you use to verify? Is this dataset used for Cryo-ET popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cryo_et/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Cryo-ET standard dataset.

**Popular datasets to consider:**
- **SHREC Cryo-ET Challenge (Gubins et al., 2019–2024)** — synthetic and real cryo-ET volumes with ground-truth particle positions and orientations; the primary benchmark for subtomogram detection and classification
- **EMPIAR Cryo-ET Entries (2018–ongoing)** — raw tilt series from community depositions; includes HIV-1 Gag (EMPIAR-10164), ribosome tomograms, in situ cellular tomograms
- **CryoET Data Portal (Chan Zuckerberg Initiative, 2023–ongoing)** — curated cryo-ET datasets with standardized annotations; designed as community benchmark
- **EMAN2 Tomography Tutorial Dataset (Galaz-Montoya et al., JSB 2016)** — HIV-1 VLP tilt series; the standard tutorial/benchmark for cryo-ET processing pipelines
- **Himes & Zhang Benchmark (2018)** — simulated cryo-ET datasets with known ground truth for reconstruction and subtomogram averaging validation

**Decision criteria:** SHREC challenge datasets are the gold standard for subtomogram detection/classification benchmarking. CryoET Data Portal (CZI, 2023) for standardized community benchmarks. EMPIAR-10164 (HIV-1 Gag) for tilt-series reconstruction validation. Use the dataset with broadest algorithm coverage.

#### Step 2: List All Cryo-ET Algorithms

Please first ensure all the Cryo-ET algorithms have been listed in `\pwm\public\algorithm_base\cryo_et\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cryo_et. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Cryo-ET solvers, please update the Cryo-ET solver.

**Key algorithms to cover (1968–2026):**

_Tilt Series Alignment (1968–2026):_
- Fiducial-based alignment — gold bead tracking for tilt series registration (Mastronarde, JSB 1997; IMOD)
- Patch tracking alignment — fiducial-free cross-correlation-based alignment (Mastronarde, JSB 2005; IMOD)
- AreTomo — GPU-accelerated tilt series alignment and reconstruction with local motion correction (Zheng et al., J Struct Biol X 2022) — fastest alignment tool
- IMOD / etomo — the canonical cryo-ET processing pipeline (Kremer et al., JSB 1996; Mastronarde & Held, JSB 2017)
- SerialEM — automated tilt series acquisition (Mastronarde, Microsc Microanal 2003)
- Markerless alignment via projection matching (Castano-Diez et al., JSB 2010)

_Tomographic Reconstruction (1970–2026):_
- Weighted back-projection (WBP) — filtered back-projection with Crowther weighting for the missing wedge (Crowther et al., Nature 1970; Radermacher, 1988)
- SIRT — Simultaneous Iterative Reconstruction Technique (Gilbert, JSB 1972) — standard iterative method for cryo-ET
- ART — Algebraic Reconstruction Technique (Gordon et al., J Theor Biol 1970)
- SART — Simultaneous Algebraic Reconstruction Technique (Andersen & Kak, Ultrason Imaging 1984)
- Compressed sensing ET — total variation and L1 regularization for missing wedge mitigation (Leary et al., Ultramicroscopy 2013; Goris et al., 2012)
- ICON / IsoNet — deep learning missing wedge correction and isotropic reconstruction (Liu et al., J Struct Biol 2022)
- MBIR — Model-Based Iterative Reconstruction for ET (Venkatakrishnan et al., 2015)
- CryoET-DeepGrasp — deep learning reconstruction (2024)
- Differentiable cryo-ET reconstruction (2024–2025)

_CTF Estimation & Correction for Tilt Series (2014–2026):_
- CTF estimation for tilted images — defocus gradient correction across tilted micrographs (Xiong et al., JSB 2009)
- Novactf — 3D-CTF correction for cryo-ET (Turonova et al., JSB 2017) — standard CTF correction for subtomogram averaging
- CTFPlotter — CTF estimation integrated with IMOD (Mastronarde, 2018)
- Per-particle CTF refinement for subtomograms (Himes & Zhang, 2018)

_Subtomogram Averaging (2003–2026):_
- PEET — Particle Estimation for Electron Tomography (Nicastro et al., Science 2006; Heumann et al., JSB 2011)
- Dynamo — flexible subtomogram averaging and analysis framework (Castano-Diez et al., JSB 2012) — widely used for in situ structural biology
- EMAN2 subtomogram averaging — e2spt pipeline (Galaz-Montoya et al., JSB 2016)
- RELION 3D subtomogram averaging — Bayesian approach adapted for tomography (Bharat & Scheres, Structure 2016; RELION-4 Zivanov et al., eLife 2022)
- emClarity — high-resolution subtomogram averaging with per-particle CTF (Himes & Zhang, Nat Methods 2018)
- M — multi-particle refinement for cryo-ET (Tegunov et al., Nat Methods 2021) — achieved atomic resolution from tomographic data
- WarpTools/WARP — real-time processing for cryo-ET (Tegunov & Cramer, Nat Methods 2019)

_Particle Picking in Tomograms (2017–2026):_
- Template matching in 3D — cross-correlation search in reconstructed tomograms (Frangakis et al., PNAS 2002)
- DeepFinder — CNN-based particle localization in cryo-ET volumes (Moebel et al., Nat Methods 2021)
- crYOLO for tomograms — adapted YOLO for particle picking in tomographic slices (Wagner et al., 2019)
- TomoTwin — metric learning for particle picking in cryo-ET (Rice et al., Nat Methods 2023)
- CryoET Object Detection Challenge models (CZI, 2024)
- Segmentation-based approaches — 3D U-Net for organelle/particle detection (2022)

_Denoising & Enhancement (2019–2026):_
- CryoCARE — content-aware image restoration for cryo-ET (Buchholz et al., ISBI 2019) — widely adopted denoising for cryo-ET
- Topaz-Denoise 3D — self-supervised denoising for tomographic volumes (Bepler et al., 2020)
- DeepDeWedge — deep learning missing wedge restoration (2024)
- Diffusion-prior tomogram denoising (2025)

#### Step 3: Update Cryo-ET Solvers

After listing all Cryo-ET solvers, update `algorithm_base/cryo_et/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Cryo-ET solvers use the data format: `y` (Ntilt, H, W) tilt series stack, `tilt_angles` (Ntilt,) tilt angles in degrees, `ctf_params` per-tilt CTF parameters, `fiducial_coords` (N, 3) gold bead positions for alignment. The `CryoETOperator` handles forward (3D volume -> tilt -> project -> CTF -> noise) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Cryo-ET:**
- WBP reconstruction: PSNR ~18–22 dB on simulated cryo-ET with known ground truth
- SIRT reconstruction: 2–4 dB improvement over WBP on missing wedge-affected data
- IsoNet missing wedge: correlation coefficient improvement >0.15 over WBP
- Subtomogram averaging (RELION/M): <4 A resolution on HIV-1 Gag (EMPIAR-10164)
- SHREC particle picking: F1 score >0.80 on benchmark volumes
- CryoCARE denoising: PSNR improvement 3–6 dB on cryo-ET volumes
- All reference values from published papers and challenge leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB PSNR of reference (reconstruction) or within 0.5 A resolution (averaging)
- `partial` — 3–10 dB shortfall or 0.5–2.0 A resolution gap
- `gap` — >10 dB shortfall or >2.0 A resolution gap
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cryo_et' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cryo_et/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cryo_et/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cryo_et/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Cryo-ET. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cryo_et/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cryo_et/standard/`

---

### Electron Tomography (`electron_tomography`) Modality Template

#### Step 1: Verify Standard Dataset

For Electron Tomography, what dataset do you use to verify? Is this dataset used for Electron Tomography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/electron_tomography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Electron Tomography standard dataset.

**Popular datasets to consider:**
- **Atomic Electron Tomography (AET) Benchmark (Miao et al., 2015–2021)** — tilt series of metallic nanoparticles (FePt, Au, Pt) with atomic-resolution 3D reconstruction ground truth; the canonical benchmark for atomic-resolution electron tomography
- **HAADF-STEM Tomography of Nanoparticles (Goris et al., Nano Lett 2012)** — tilt series of Au nanorods and core-shell nanoparticles; widely used for discrete tomography and compressed sensing validation
- **Materials Electron Tomography Dataset (Leary et al., 2013)** — HAADF-STEM tilt series with simulated ground truth for testing reconstruction algorithms under limited tilt range
- **Biological Electron Tomography — IMOD Tutorial (Mastronarde, 2017)** — plastic-section ET tilt series for dual-axis tomography reconstruction benchmarking
- **In Situ ET Dataset (Liao et al., 2020)** — electron tomography of catalyst nanoparticles under reaction conditions; benchmark for time-resolved ET

**Decision criteria:** AET nanoparticle datasets (Miao group) are the gold standard for atomic-resolution electron tomography. HAADF-STEM tilt series of nanoparticles (Goris et al.) for materials science ET. Use the dataset that appears in the largest number of electron tomography reconstruction papers.

#### Step 2: List All Electron Tomography Algorithms

Please first ensure all the Electron Tomography algorithms have been listed in `\pwm\public\algorithm_base\electron_tomography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/electron_tomography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Electron Tomography solvers, please update the Electron Tomography solver.

**Key algorithms to cover (1968–2026):**

_Classical Reconstruction (1968–2000):_
- Filtered back-projection (FBP) — Ram-Lak / Shepp-Logan filtered back-projection for tilt series (Crowther et al., Nature 1970; DeRosier & Klug, Nature 1968)
- Weighted back-projection (WBP) — tilt-angle-weighted FBP for non-uniform angular sampling (Radermacher, 1988)
- ART — Algebraic Reconstruction Technique (Gordon et al., J Theor Biol 1970)
- SIRT — Simultaneous Iterative Reconstruction Technique (Gilbert, JSB 1972)
- SART — Simultaneous Algebraic Reconstruction Technique (Andersen & Kak, 1984)
- DART — Discrete Algebraic Reconstruction Technique for binary/few-level materials (Batenburg & Sijbers, IEEE TIP 2011)

_Compressed Sensing & Advanced Optimization (2006–2020):_
- Total variation (TV) regularized ET — sparse-gradient tomography for limited-angle data (Goris et al., Ultramicroscopy 2012; Leary et al., 2013)
- FISTA-TV — fast iterative shrinkage-thresholding for TV-regularized ET (Beck & Teboulle, 2009; applied to ET)
- Total generalized variation (TGV) for ET — higher-order regularization (Bredies et al., 2010; applied to ET 2016)
- Low-rank + sparse decomposition for dynamic ET (2016)
- Dictionary learning electron tomography (2015)
- Compressive real-time ET — acquisition-adaptive reconstruction (Levin et al., 2018)

_Atomic Electron Tomography (AET) (2012–2026):_
- GENFIRE — GENeralized Fourier Iterative REconstruction for atomic ET (Pryor et al., Sci Rep 2017) — standard for atomic-resolution ET
- RESIRE — Real Space Iterative Reconstruction (Miao group, 2021) — state-of-the-art for atomic ET
- REal-space Constraint Oversampling (RECO) — oversampled iterative real-space constraint (Yang et al., Nature 2017)
- Atom tracing — identification and classification of individual atom positions from 3D reconstruction (Xu et al., Nat Mater 2015; Yang et al., Nature 2017)
- Dynamic AET — 4D atomic electron tomography for tracking atom motion (Zhou et al., Nature 2019)

_Deep Learning for ET (2018–2026):_
- Deep learning tomographic reconstruction — CNN-based artifact removal and reconstruction (Yang et al., 2018)
- TomoGAN — GAN-based denoising for electron tomography (Liu et al., Nat Mach Intell 2020)
- Neural implicit tomographic reconstruction — coordinate network for continuous 3D representation (2022)
- Self-supervised ET reconstruction — Noise2Inverse (Hendriksen et al., IEEE TCI 2020)
- Diffusion-prior electron tomography (2024)
- Transformer-based ET reconstruction (2024–2025)
- Neural Radiance Fields (NeRF) adapted for ET (2023–2024)

_Dual-Axis & Advanced Acquisition (1992–2020):_
- Dual-axis tomography — combining two perpendicular tilt series for reduced missing wedge (Penczek et al., 1995; Mastronarde, JSB 1997)
- Conical tilt reconstruction — alternative geometry for reduced artifacts (Lanzavecchia et al., 1999)
- On-axis rotation tomography — needle specimen geometry eliminating the missing wedge (Ke et al., JSB 2010)
- Dose-symmetric tilt scheme — optimal tilt ordering for radiation-sensitive specimens (Hagen et al., JSB 2017)

#### Step 3: Update Electron Tomography Solvers

After listing all Electron Tomography solvers, update `algorithm_base/electron_tomography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Electron Tomography solvers use the data format: `y` (Ntilt, H, W) tilt series projections, `tilt_angles` (Ntilt,) tilt angles in degrees, `tilt_axes` tilt axis orientation(s). The `ElectronTomographyOperator` handles forward (3D volume -> tilt -> project) and adjoint (back-projection) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Electron Tomography:**
- FBP/WBP reconstruction: PSNR ~20–24 dB on Shepp-Logan phantom with missing wedge
- SIRT vs WBP: 2–5 dB PSNR improvement on limited-angle data
- TV-regularized ET: 4–8 dB improvement over WBP on sparse nanoparticle phantoms
- DART: correct voxel classification >95% for binary phantoms with 5+ projections
- GENFIRE atomic ET: atom position accuracy <20 pm on FePt nanoparticle benchmark
- TomoGAN denoising: PSNR improvement 3–5 dB on experimental ET volumes
- All reference values from published papers and simulation ground truth

**Verification criteria:**
- `done` — PWM within 3 dB PSNR of reference (volumetric tasks) or <20 pm atom position error (AET)
- `partial` — 3–10 dB shortfall or 20–50 pm error
- `gap` — >10 dB shortfall or >50 pm error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'electron_tomography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/electron_tomography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Electron Tomography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/electron_tomography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_tomography/standard/`

---

### 4D-STEM Electron Diffraction (`electron_diffraction`) Modality Template

#### Step 1: Verify Standard Dataset

For 4D-STEM Electron Diffraction, what dataset do you use to verify? Is this dataset used for 4D-STEM popular algorithms? Please ensure the standard dataset in `datasets/benchmark/electron_diffraction/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original 4D-STEM standard dataset.

**Popular datasets to consider:**
- **4D-STEM Tutorial Dataset (Ophus, Microsc Microanal 2019)** — well-characterized 4D-STEM datasets of SrTiO3 and twisted bilayer MoS2; the canonical tutorial/benchmark for 4D-STEM analysis pipelines (py4DSTEM)
- **py4DSTEM Benchmark Collection (Savitzky et al., Microsc Microanal 2021)** — curated 4D-STEM datasets with ground-truth strain, orientation, and phase maps; used to validate py4DSTEM analysis routines
- **Prismatic Simulation Benchmark (Pryor et al., 2017)** — simulated 4D-STEM datasets with known crystal structure; used for validating strain mapping and orientation determination
- **MicroED / 3D ED Benchmark (Nannenga & Gonen, 2019)** — electron diffraction datasets of known protein/small-molecule crystals for structure determination validation
- **ASTAR / NanoMEGAS Orientation Mapping Dataset (Rauch et al., 2005)** — precession electron diffraction datasets with EBSD-validated crystal orientation ground truth

**Decision criteria:** py4DSTEM benchmark datasets (Ophus, Savitzky) are the gold standard for 4D-STEM analysis validation. ASTAR datasets for orientation mapping. MicroED benchmarks for structure determination. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All 4D-STEM Electron Diffraction Algorithms

Please first ensure all the 4D-STEM Electron Diffraction algorithms have been listed in `\pwm\public\algorithm_base\electron_diffraction\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/electron_diffraction. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Electron Diffraction solvers, please update the Electron Diffraction solver.

**Key algorithms to cover (1950–2026):**

_Diffraction Pattern Analysis (1950–2010):_
- Kinematical diffraction — Bragg peak positions from Ewald sphere construction (von Laue, 1912; adapted for electron diffraction)
- Dynamical diffraction theory — Bloch wave and multislice for electron diffraction simulation (Bethe, 1928; Cowley & Moodie, 1957)
- Selected area electron diffraction (SAED) analysis — zone axis identification and lattice parameter measurement (classical crystallography)
- Convergent beam electron diffraction (CBED) — thickness and symmetry determination (Tanaka & Terauchi, 1985)
- Precession electron diffraction (PED) — quasi-kinematical diffraction via beam precession (Vincent & Midgley, Ultramicroscopy 1994)
- ASTAR / orientation mapping — automated crystal orientation from PED patterns (Rauch & Dupuy, Arch Met Mat 2005)

_4D-STEM Virtual Imaging & Analysis (2014–2026):_
- Virtual detector imaging — arbitrary detector geometry applied to 4D-STEM data (Ophus, Microsc Microanal 2019)
- Center of mass (CoM) / differential phase contrast (DPC) — electric/magnetic field mapping from beam deflection (Lazic et al., Ultramicroscopy 2016; Muller et al., Nat Commun 2014)
- Integrated CoM (iCoM) — phase retrieval from integrated center of mass (Muller et al., 2014)
- Symmetry-STEM — symmetry analysis of CBED patterns for local symmetry determination (Ophus et al., 2022)
- Fluctuation EM (FEM) — nanoscale structural order from diffraction variance (Voyles et al., Ultramicroscopy 2002)

_Strain Mapping (2010–2026):_
- Geometric phase analysis (GPA) — strain from Fourier space peak analysis (Hytch et al., Ultramicroscopy 1998)
- Peak finding strain mapping — Bragg disk position fitting in 4D-STEM (Pekin et al., Ultramicroscopy 2017)
- Template matching strain mapping — cross-correlation with simulated patterns (Ophus et al., 2022)
- Deep learning strain mapping — CNN-based strain extraction from 4D-STEM (Oelerich et al., 2022)
- Nano-beam electron diffraction (NBED) strain mapping — sub-nm spatial resolution strain (Beche et al., Ultramicroscopy 2009)

_Phase Retrieval from 4D-STEM (2012–2026):_
- Single-sideband (SSB) ptychography — linear phase retrieval (Pennycook et al., 2015)
- Wigner Distribution Deconvolution (WDD) — non-iterative phase retrieval (Rodenburg & Bates, 1992)
- ePIE / rPIE — iterative ptychographic engines applied to 4D-STEM (Maiden & Rodenburg, 2009)
- Multi-slice ptychography from 4D-STEM — depth-resolved reconstruction (Chen et al., Science 2021)
- Parallax / phase-contrast STEM — tilt-corrected BF-STEM phase imaging (Gao et al., Ultramicroscopy 2017)

_MicroED / 3D Electron Diffraction (2013–2026):_
- Rotation electron diffraction (RED) — continuous-rotation data collection (Wan et al., J Appl Cryst 2013)
- MicroED — electron diffraction for protein/small-molecule structure determination (Shi et al., eLife 2013; Nannenga & Gonen, Nat Methods 2019)
- DIALS for electron diffraction — adapting X-ray data processing for MicroED (Clabbers et al., Acta Cryst D 2018)
- XDS / SHELX adapted for electron diffraction — integration and structure solution (Kabsch, 2010; Sheldrick, 2015; adapted for ED)
- Dynamical refinement — structure refinement accounting for dynamical scattering (Palatinus et al., Acta Cryst B 2015)
- CryoEM-based MicroED processing (SerialEM + RELION pipeline, 2020)

_Machine Learning for 4D-STEM (2020–2026):_
- Neural network crystal phase identification (Kaufmann et al., Science 2020)
- Unsupervised clustering of diffraction patterns — NMF, VAE-based (Martinolich et al., 2022)
- Deep learning Bragg disk detection (2022)
- Graph neural networks for crystal structure from diffraction (2024)
- Foundation model for electron diffraction analysis (2025–2026)

#### Step 3: Update Electron Diffraction Solvers

After listing all Electron Diffraction solvers, update `algorithm_base/electron_diffraction/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Electron Diffraction solvers use the data format: `y` (Ny, Nx, Ky, Kx) 4D-STEM dataset (scan positions x diffraction pattern), `probe` (Ky, Kx) convergent probe, `calibrations` (pixel_size_real, pixel_size_diffraction, voltage). The `ElectronDiffractionOperator` handles forward (crystal structure -> dynamical diffraction -> detector) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for 4D-STEM Electron Diffraction:**
- Strain mapping: strain accuracy <0.05% on SrTiO3 benchmark (Pekin et al.)
- DPC/CoM: electric field accuracy within 10% of known fields on simulated data
- Ptychography from 4D-STEM: phase accuracy <15 mrad on thin SrTiO3
- Crystal orientation mapping: orientation accuracy <1 degree vs EBSD ground truth
- MicroED: R-factor <0.20 on lysozyme benchmark
- All reference values from published papers and simulation ground truth

**Verification criteria:**
- `done` — PWM within stated accuracy thresholds of reference
- `partial` — 2–5x larger error than reference threshold
- `gap` — >5x error or qualitative failure
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'electron_diffraction' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/electron_diffraction/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Electron Diffraction. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/electron_diffraction/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_diffraction/standard/`

---

### Electron Holography (`electron_holography`) Modality Template

#### Step 1: Verify Standard Dataset

For Electron Holography, what dataset do you use to verify? Is this dataset used for Electron Holography popular algorithms? Please ensure the standard dataset in `datasets/benchmark/electron_holography/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Electron Holography standard dataset.

**Popular datasets to consider:**
- **Off-Axis Holography MgO Benchmark (Lichte et al., 2008)** — off-axis electron holograms of MgO nanocubes with known mean inner potential; the canonical benchmark for phase reconstruction accuracy in electron holography
- **Magnetic Nanoparticle Holography Dataset (Dunin-Borkowski et al., Science 1998)** — electron holograms of magnetite nanoparticles chains; standard for magnetic phase shift measurement validation
- **p-n Junction Holography Benchmark (McCartney & Smith, 2007)** — semiconductor device cross-sections with known doping profiles; used to validate electrostatic potential mapping
- **Simulated Hologram Benchmark (Lehmann & Lichte, 2002)** — synthetic electron holograms with known phase/amplitude ground truth; used for validating reconstruction algorithms
- **In Situ Holography Dataset (Beleggia et al., 2014)** — time-resolved electron holograms of dynamic processes (charging, magnetic switching)

**Decision criteria:** MgO cube holography (Lichte et al.) is the gold standard for phase accuracy validation. Magnetic nanoparticle dataset (Dunin-Borkowski) for magnetic holography. Simulated holograms (Lehmann & Lichte) for quantitative algorithm comparison. Use the dataset with broadest algorithm coverage.

#### Step 2: List All Electron Holography Algorithms

Please first ensure all the Electron Holography algorithms have been listed in `\pwm\public\algorithm_base\electron_holography\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/electron_holography. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Electron Holography solvers, please update the Electron Holography solver.

**Key algorithms to cover (1955–2026):**

_Off-Axis Holography Reconstruction (1955–2010):_
- Gabor holography — in-line holography (Gabor, Nature 1948; first electron demonstration, 1955)
- Off-axis Fourier reconstruction — sideband extraction and inverse FFT phase recovery (Takeda et al., JOSA 1982; applied to electron holography by Tonomura, 1987)
- Double-exposure holography — reference/specimen hologram subtraction for artifact removal (Lehmann & Lichte, Ultramicroscopy 2002)
- Phase unwrapping — Goldstein branch-cut (Goldstein et al., Radio Science 1988), quality-guided, and Lp-norm methods for electron holography
- Sideband filtering optimization — optimal mask shape and size for phase/amplitude extraction (Lehmann & Lichte, 2002)
- Fresnel fringe removal — correction for biprism-induced artifacts

_In-Line Holography & Focal Series (1980–2020):_
- In-line (Gabor) holographic reconstruction — iterative twin-image removal (Latychevskaia & Fink, PRL 2007)
- Focal series phase reconstruction — TIE-based or iterative approach (Coene et al., PRL 1992; Thust et al., 1996)
- Lorentz microscopy — magnetic domain imaging in defocused mode (Chapman, J Phys D 1984)
- Differential phase contrast from quadrant/segmented detector — magnetic/electric field mapping

_Quantitative Phase Analysis (1998–2026):_
- Mean inner potential measurement — quantitative electrostatic potential from holographic phase (Gajdardziska-Josifovska et al., Ultramicroscopy 1993)
- Magnetic flux quantization — Aharonov-Bohm phase shift analysis for magnetic induction mapping (Tonomura, Rev Mod Phys 1987)
- 3D electrostatic potential tomography — holographic tomography combining holography + tilt series (Lai et al., JSB 2000; Wolf et al., 2010)
- Charge density mapping — from Laplacian of holographic phase (Beleggia et al., 2014)
- Space-charge field mapping in semiconductor devices (McCartney et al., Ultramicroscopy 2007)

_Advanced & Computational Methods (2010–2026):_
- Split-image holography — biprism-free holography using crystal-edge diffraction (Ru, J Appl Phys 1994)
- Dark-field electron holography — strain mapping via holographic diffracted beam interference (Hytch et al., Nature 2008)
- Time-resolved electron holography — ultrafast dynamics with pulsed electron sources (Arbouet et al., 2018)
- Compressed sensing holographic reconstruction — sparse phase recovery (2016)
- Deep learning phase unwrapping for electron holography (2020)
- Neural network holographic reconstruction — learned phase retrieval (2022)
- Self-supervised hologram denoising (2023)
- Diffusion model for holographic phase recovery (2025)

_Vector Field Tomography (2007–2026):_
- Holographic vector field electron tomography — 3D magnetic induction mapping from holographic tilt series (Phatak et al., Ultramicroscopy 2008)
- Model-based magnetic reconstruction — constrained optimization for divergence-free B fields (Humphrey et al., Ultramicroscopy 2014)
- Machine learning for magnetic induction mapping (2024)

#### Step 3: Update Electron Holography Solvers

After listing all Electron Holography solvers, update `algorithm_base/electron_holography/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Electron Holography solvers use the data format: `y` (H, W) electron hologram (interference fringe pattern), `reference_hologram` (H, W) vacuum reference hologram, `biprism_params` (voltage, fringe_spacing, fringe_contrast). The `ElectronHolographyOperator` handles forward (object wave -> biprism interference -> hologram) and adjoint (sideband extraction -> phase/amplitude recovery) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Electron Holography:**
- Off-axis phase reconstruction: phase accuracy <0.05 rad on MgO benchmark (Lichte et al.)
- Mean inner potential MgO: measured value within 0.5 V of accepted 13.0 V
- Magnetic phase shift: flux quantization accuracy within 5% on magnetite chain
- Phase unwrapping: zero unwrapping errors on benchmark with known topology
- Dark-field holography strain: accuracy <0.05% on Si/SiGe heterostructure
- All reference values from published papers and simulation ground truth

**Verification criteria:**
- `done` — PWM within 0.05 rad phase accuracy or 5% field accuracy of reference
- `partial` — 0.05–0.2 rad phase error or 5–15% field error
- `gap` — >0.2 rad phase error or >15% field error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'electron_holography' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/electron_holography/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/electron_holography/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/electron_holography/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Electron Holography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/electron_holography/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/electron_holography/standard/`

---

### EBSD (`ebsd`) Modality Template

#### Step 1: Verify Standard Dataset

For EBSD, what dataset do you use to verify? Is this dataset used for EBSD popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ebsd/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original EBSD standard dataset.

**Popular datasets to consider:**
- **DREAM.3D Synthetic EBSD Benchmark (Groeber & Jackson, IMMI 2014)** — synthetically generated EBSD maps with known grain structure, texture, and misorientation; the standard benchmark for EBSD analysis and reconstruction algorithms
- **MTEX Example Datasets (Bachmann et al., 2010)** — curated EBSD datasets of various metals and minerals distributed with the MTEX toolbox; widely used for texture analysis validation
- **Ni Superalloy EBSD Round Robin (Wilkinson et al., 2006)** — community round-robin EBSD dataset of nickel superalloy; used for comparing indexing accuracy across software packages
- **EDAX/Oxford EBSD Standard Materials (ongoing)** — calibration EBSD patterns from Si, Ni, Cu single crystals with known orientation; used for detector geometry calibration
- **HR-EBSD Cross-Correlation Benchmark (Wilkinson et al., Ultramicroscopy 2006)** — EBSD patterns with sub-pixel shift ground truth for validating high-angular-resolution methods

**Decision criteria:** DREAM.3D synthetic benchmarks are the gold standard for grain reconstruction algorithm validation. MTEX example datasets for texture analysis. HR-EBSD benchmark (Wilkinson) for high-angular-resolution methods. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All EBSD Algorithms

Please first ensure all the EBSD algorithms have been listed in `\pwm\public\algorithm_base\ebsd\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ebsd. Besides, you need to search all algorithms from 1950 to 2026. After listing all the EBSD solvers, please update the EBSD solver.

**Key algorithms to cover (1973–2026):**

_Pattern Indexing (1973–2020):_
- Hough transform indexing — band detection via Hough/Radon transform of Kikuchi patterns (Krieger Lassen et al., Scanning Microscopy 1992) — the standard EBSD indexing method for 30 years
- Triplet voting indexing — band triplet matching for orientation determination (Wright & Adams, Met Trans A 1992)
- Dictionary indexing — template matching against pre-computed dictionary of simulated patterns (Chen et al., Microsc Microanal 2015; Rauch & Dupuy, 2005)
- Spherical harmonic indexing — orientation determination via spherical function cross-correlation (Lenthe et al., Ultramicroscopy 2019)
- Dynamical pattern simulation — Bloch wave / master pattern calculation for accurate template generation (Callahan & De Graef, Microsc Microanal 2013)

_HR-EBSD & Strain Analysis (2006–2026):_
- HR-EBSD cross-correlation — sub-pixel pattern shift measurement for elastic strain/rotation (Wilkinson et al., Ultramicroscopy 2006; Britton & Wilkinson, 2012)
- Remapping / pattern remapping — integrated forward model EBSD (Winkelmann et al., 2007)
- GND density estimation — geometrically necessary dislocation density from orientation gradients (Pantleon, Scripta Mater 2008)
- Kernel Average Misorientation (KAM) — local misorientation analysis
- Pattern center refinement — iterative calibration of detector geometry (Britton et al., Ultramicroscopy 2010)

_Grain Reconstruction & Analysis (1990–2026):_
- Grain boundary detection and misorientation analysis — standard crystallographic analysis (Humphreys, 2001)
- DREAM.3D pipeline — grain segmentation, feature extraction, and synthetic microstructure generation (Groeber & Jackson, 2014)
- Orientation Distribution Function (ODF) estimation — MTEX harmonic method (Hielscher & Schaeben, J Appl Cryst 2008)
- Pole figure calculation and texture analysis (Bunge, 1982; MTEX, 2010)
- Parent grain reconstruction — Austenite reconstruction from martensite EBSD (Nyyssonen et al., 2016)
- EBSD cleaning / denoising — half-quadratic filtering, Kuwahara filter for orientation maps

_Deep Learning for EBSD (2019–2026):_
- Deep learning EBSD indexing — CNN-based orientation from Kikuchi patterns (Kaufmann et al., Science 2020; Shen et al., Acta Mater 2019)
- Neural network pattern center determination (Pang et al., 2020)
- GAN-based EBSD pattern denoising and super-resolution (2021)
- Physics-informed neural network for EBSD (2023)
- Self-supervised EBSD pattern analysis (2024)
- Foundation model for EBSD indexing (2025–2026)

_Transmission Kikuchi Diffraction (TKD) (2012–2026):_
- t-EBSD / TKD — transmission mode EBSD for thin specimens with improved spatial resolution (Keller & Geiss, JMICRO 2012)
- On-axis TKD — improved geometry for TKD with direct electron detector (Fundenberger et al., Ultramicroscopy 2016)
- TKD indexing via dictionary matching (2018)

#### Step 3: Update EBSD Solvers

After listing all EBSD solvers, update `algorithm_base/ebsd/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All EBSD solvers use the data format: `y` (Ny, Nx, Hp, Wp) EBSD pattern array (scan grid x pattern pixels), `master_pattern` (H, W) or (theta, phi) dynamical master pattern, `detector_params` (pattern_center, detector_tilt, sample_tilt). The `EBSDOperator` handles forward (crystal orientation -> dynamical diffraction -> Kikuchi pattern) and adjoint (pattern -> orientation) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for EBSD:**
- Hough indexing: indexing success rate >98% on clean Si/Ni patterns, angular accuracy ~0.5-1.0 degree
- Dictionary indexing: angular accuracy ~0.1-0.3 degree on benchmark (Lenthe et al.)
- HR-EBSD: strain sensitivity ~1e-4 on Si reference (Wilkinson et al.)
- Deep learning indexing: angular accuracy ~0.3 degree with 10x speed improvement (Kaufmann et al.)
- ODF estimation: L2 error <5% vs X-ray texture ground truth (MTEX benchmark)
- All reference values from published papers and round-robin studies

**Verification criteria:**
- `done` — PWM within stated angular accuracy or strain sensitivity of reference
- `partial` — 2–5x larger angular error or strain error than reference
- `gap` — >5x error or qualitative failure
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ebsd' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ebsd/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ebsd/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ebsd/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for EBSD. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ebsd/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ebsd/standard/`

---

### EELS (`eels`) Modality Template

#### Step 1: Verify Standard Dataset

For EELS, what dataset do you use to verify? Is this dataset used for EELS popular algorithms? Please ensure the standard dataset in `datasets/benchmark/eels/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original EELS standard dataset.

**Popular datasets to consider:**
- **EELS Atlas / EELS DB (Ahn & Krivanek, 1983; updated online)** — comprehensive database of core-loss and low-loss EELS reference spectra for all elements; the canonical reference for edge identification and quantification
- **Gatan EELS Reference Spectra (Gatan, ongoing)** — instrument-specific reference spectra used for energy calibration and edge fine structure validation
- **SrTiO3 EELS Benchmark (Muller et al., Nature 2008)** — atomic-resolution EELS mapping of SrTiO3 with Ti-L2,3 and O-K edges; gold standard for spatial resolution and fine structure analysis
- **STEM-EELS Spectrum Image Dataset (Bonnet et al., 1990; Trebbia & Bonnet, 1990)** — multivariate analysis benchmark spectrum images; used for validating PCA, NMF, and ICA decomposition methods
- **HyperSpy EELS Tutorial Dataset (de la Pena et al., 2017)** — curated EELS spectrum images distributed with HyperSpy; standard for testing processing routines

**Decision criteria:** EELS Atlas/DB is the essential reference for edge identification. SrTiO3 atomic-resolution EELS (Muller) for spatial resolution benchmarking. HyperSpy datasets for algorithm development and testing. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All EELS Algorithms

Please first ensure all the EELS algorithms have been listed in `\pwm\public\algorithm_base\eels\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/eels. Besides, you need to search all algorithms from 1950 to 2026. After listing all the EELS solvers, please update the EELS solver.

**Key algorithms to cover (1962–2026):**

_Spectrum Processing & Background Subtraction (1962–2010):_
- Power-law background model — AE^(-r) background fitting and subtraction for core-loss edges (Egerton, EELS in the Electron Microscope, 1986) — the standard EELS background method
- Hartree-Slater cross-section calculation — theoretical ionization cross-sections for quantification (Egerton, 1979; Rez, 1982)
- Fourier-log deconvolution — plural scattering removal from low-loss EELS (Johnson & Spence, J Phys D 1974)
- Fourier-ratio deconvolution — plural scattering correction using low-loss as kernel (Egerton, 1986)
- Richardson-Lucy deconvolution — iterative deconvolution for EELS energy resolution improvement (Richardson, 1972; Lucy, 1974; applied to EELS)
- Maximum entropy deconvolution for EELS (Overwijk & Reefman, Micron 2000)
- Zero-loss peak alignment and calibration — energy drift correction

_Quantification & Fine Structure (1982–2020):_
- Elemental quantification — edge integration with Hartree-Slater cross-sections (Egerton, 1986)
- ELNES/XANES analysis — Energy Loss Near Edge Structure fingerprinting for chemical state (Kurata & Colliex, PRB 1993)
- EXELFS — Extended Energy Loss Fine Structure for local bonding environment (Leapman & Cosslett, J Phys D 1976)
- Multiple linear least squares (MLLS) fitting — reference spectrum fitting for overlapping edges (Riegler & Kothleitner, Ultramicroscopy 2010)
- Kramers-Kronig analysis — dielectric function extraction from low-loss EELS (Egerton, 1986; Stoger-Pollach, Micron 2008)
- FEFF simulation — ab initio EELS fine structure simulation (Rehr & Albers, Rev Mod Phys 2000)

_Multivariate Analysis (1990–2020):_
- PCA / SVD for EELS spectrum images — noise reduction and component extraction (Bonnet et al., Ultramicroscopy 1990; Trebbia & Bonnet, 1990)
- Independent Component Analysis (ICA) for EELS — blind source separation (de la Pena et al., Ultramicroscopy 2011)
- Non-negative Matrix Factorization (NMF) — physically meaningful spectral decomposition (Shiga et al., 2016)
- Vertex Component Analysis (VCA) — endmember extraction for EELS (Dobigeon & Brun, Ultramicroscopy 2012)
- Bayesian spectral unmixing for EELS (Dobigeon et al., 2009)

_Monochromated & Vibrational EELS (2014–2026):_
- Vibrational EELS — phonon spectroscopy at sub-10 meV resolution (Krivanek et al., Nature 2014) — transformative advance
- Aloof-beam vibrational EELS — damage-free vibrational spectroscopy (Hachtel et al., Science 2019)
- Isotope identification via vibrational EELS (Hachtel et al., 2019)
- Surface phonon polariton mapping (Govyadinov et al., Nat Commun 2017)
- Momentum-resolved EELS — q-dependent loss function mapping (Senga et al., Nature 2019)

_Deep Learning for EELS (2019–2026):_
- Deep learning EELS denoising — autoencoder-based noise reduction (Potapov, Ultramicroscopy 2020)
- Neural network background subtraction — learned background model (2021)
- CNN for EELS fine structure classification (2022)
- Self-supervised EELS spectrum denoising (2023)
- Foundation model for spectroscopy / EELS analysis (2025–2026)
- Transfer learning for ELNES fingerprinting (2024)

#### Step 3: Update EELS Solvers

After listing all EELS solvers, update `algorithm_base/eels/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All EELS solvers use the data format: `y` (Ny, Nx, E) EELS spectrum image (spatial x energy channels), `energy_axis` (E,) energy loss values in eV, `beam_params` (voltage, convergence_angle, collection_angle). The `EELSOperator` handles forward (specimen composition -> inelastic scattering -> energy loss spectrum) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for EELS:**
- Power-law background: quantification accuracy within 10% for standard edges (Egerton benchmark)
- Fourier-log deconvolution: zero-loss peak FWHM reduced by >50% on monochromated data
- PCA denoising: SNR improvement >10x on SrTiO3 atomic-resolution EELS
- MLLS fitting: composition accuracy within 5% for known multi-element standards
- Vibrational EELS: phonon peak position accuracy <2 meV on BN reference
- All reference values from published papers and database comparisons

**Verification criteria:**
- `done` — PWM within 10% quantification accuracy or 2 meV energy accuracy of reference
- `partial` — 10–25% quantification error or 2–10 meV energy error
- `gap` — >25% quantification error or >10 meV energy error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'eels' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/eels/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/eels/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/eels/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for EELS. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/eels/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/eels/standard/`

---

### STEM-EDX Elemental Mapping (`edx_mapping`) Modality Template

#### Step 1: Verify Standard Dataset

For STEM-EDX, what dataset do you use to verify? Is this dataset used for STEM-EDX popular algorithms? Please ensure the standard dataset in `datasets/benchmark/edx_mapping/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original STEM-EDX standard dataset.

**Popular datasets to consider:**
- **NIST DTSA-II Standard Spectra (Newbury & Ritchie, 2013–ongoing)** — simulated and experimental EDX spectra of standard reference materials; the canonical benchmark for EDX quantification algorithms; distributed with the DTSA-II software
- **SRM 2063a Thin Film Standard (NIST)** — multi-element thin film standard for EDX calibration; traceable elemental composition; the hardware standard for EDX quantification
- **HyperSpy EDX Tutorial Dataset (de la Pena et al., 2017)** — curated STEM-EDX spectrum images for algorithm testing; includes core-shell nanoparticles and multilayer thin films
- **Atomic-Resolution EDX Benchmark (Chu et al., PRL 2010; D'Alfonso et al., PRB 2010)** — SrTiO3 atomic-resolution STEM-EDX maps; used to validate channeling effects and quantification at atomic scale
- **SEM-EDX Phase Mapping Dataset (Ritchie et al., 2012)** — multi-phase mineral/alloy EDX maps with known mineralogy; used for phase classification benchmarking

**Decision criteria:** NIST DTSA-II standards are the gold standard for EDX quantification validation. SrTiO3 atomic-resolution EDX (Chu et al.) for spatial resolution benchmarking. HyperSpy datasets for algorithm development. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All STEM-EDX Algorithms

Please first ensure all the STEM-EDX algorithms have been listed in `\pwm\public\algorithm_base\edx_mapping\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/edx_mapping. Besides, you need to search all algorithms from 1950 to 2026. After listing all the STEM-EDX solvers, please update the STEM-EDX solver.

**Key algorithms to cover (1968–2026):**

_Spectrum Processing & Peak Identification (1968–2010):_
- Gaussian peak fitting — EDX characteristic peak identification and deconvolution (Reed, 1975)
- Background modeling — Bremsstrahlung background fitting via Kramers' law or polynomial (Kramers, 1923; Fiori et al., 1976)
- Peak deconvolution — overlapping peak separation via MLLS and Bayesian methods (Statham, XRMS 2002)
- Escape peak and sum peak correction — artifact removal from detector response
- Energy calibration — gain and offset correction from known peak positions
- DTSA/DTSA-II spectrum simulation — Monte Carlo EDX simulation (Newbury & Ritchie, JRES NIST 2013)

_Quantification Methods (1975–2020):_
- Cliff-Lorimer thin film quantification — k-factor method for thin specimens (Cliff & Lorimer, JMICRO 1975) — the standard STEM-EDX quantification method
- Zeta-factor method — mass-thickness and composition from EDX (Watanabe & Williams, JMICRO 2006)
- Absorption correction for thick specimens — self-absorption and path-length correction (Goldstein et al., 1986)
- ZAF/Phi-Rho-Z correction — matrix correction for bulk SEM-EDX (Pouchou & Pichoir, 1991)
- Standardless quantification — theoretical k-factor calculation (2000)
- DTSA-II Monte Carlo quantification — physics-based quantification via electron trajectory simulation (Ritchie, 2009)

_Spectrum Image Analysis (1990–2026):_
- PCA / SVD for EDX spectrum images — noise reduction and component extraction (Kotula et al., Microsc Microanal 2003)
- NMF for EDX spectral unmixing — non-negative decomposition for elemental maps (Shiga et al., Ultramicroscopy 2016)
- ICA for EDX — blind source separation of spectral components (de la Pena et al., 2011)
- Machine learning phase classification — k-means, Gaussian mixture, and supervised classification of EDX spectra (Parish & Brewer, Microsc Microanal 2010)
- Bayesian spectral unmixing for EDX (Dobigeon et al., 2009)

_Atomic-Resolution EDX (2010–2026):_
- Channeling-corrected EDX quantification — accounting for electron beam channeling effects at atomic resolution (Lugg et al., PRB 2014)
- Frozen-phonon EDX simulation — inelastic scattering simulation for atomic-resolution EDX (Forbes et al., PRB 2010; Allen et al., Ultramicroscopy 2015)
- Deconvolution of probe-broadened EDX maps — super-resolution elemental mapping (2018)
- Deep learning EDX super-resolution and denoising (2022)

_Compressed Sensing & Advanced Methods (2014–2026):_
- Compressed sensing EDX — sub-Nyquist spectrum image acquisition (Stevens et al., Microscopy 2014)
- Inpainting for sparse EDX acquisition — dictionary learning and low-rank recovery (2016)
- Total variation regularization for EDX denoising (2017)
- Deep learning EDX denoising — CNN-based noise reduction for low-dose EDX maps (2021)
- Self-supervised EDX spectrum denoising (2023)
- Foundation model for EDX analysis (2025–2026)

#### Step 3: Update STEM-EDX Solvers

After listing all STEM-EDX solvers, update `algorithm_base/edx_mapping/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All STEM-EDX solvers use the data format: `y` (Ny, Nx, E) EDX spectrum image (spatial x energy channels), `energy_axis` (E,) energy values in keV, `detector_params` (solid_angle, takeoff_angle, window_type, detector_response). The `EDXOperator` handles forward (elemental composition -> X-ray generation -> detection) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for STEM-EDX:**
- Cliff-Lorimer quantification: composition accuracy within 5% relative for standard thin films (SRM 2063a)
- DTSA-II simulation: peak positions within 5 eV and intensities within 10% of experiment
- PCA denoising: SNR improvement >5x on atomic-resolution EDX spectrum images
- NMF phase mapping: correct identification of >95% of known phases in multi-phase alloys
- Atomic-resolution EDX: correct column assignment for >90% of atom columns in SrTiO3
- All reference values from published papers and NIST standards

**Verification criteria:**
- `done` — PWM within 5% relative composition accuracy of reference
- `partial` — 5–15% composition error
- `gap` — >15% composition error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'edx_mapping' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/edx_mapping/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for STEM-EDX. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/edx_mapping/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/edx_mapping/standard/`

---

### Coherent Diffractive Imaging / Phase Retrieval (`phase_retrieval`) Modality Template

#### Step 1: Verify Standard Dataset

For Phase Retrieval / CDI, what dataset do you use to verify? Is this dataset used for Phase Retrieval popular algorithms? Please ensure the standard dataset in `datasets/benchmark/phase_retrieval/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Phase Retrieval standard dataset.

**Popular datasets to consider:**
- **CXIDB (Coherent X-ray Imaging Data Bank, Maia, 2012)** — the central repository for coherent diffractive imaging data; hosts X-ray and electron CDI patterns with deposited reconstructions; the standard data source for CDI algorithm benchmarking
- **Miao CDI Benchmark (Miao et al., Nature 1999; Acta Cryst A 2015)** — foundational CDI diffraction patterns (Au nanoparticles, nanocrystals) with known structure; the original CDI demonstration datasets
- **Ptychography Benchmark (Thibault et al., 2008; Maiden & Rodenburg, 2009)** — curated ptychographic datasets with known probe and object; used to validate iterative ptychographic engines
- **SHARP/COSMIC Ptychography Benchmark (Shapiro et al., 2014)** — synchrotron soft X-ray ptychography with high-quality ground truth; standard for ptychographic reconstruction evaluation
- **Simulated CDI Benchmark (Marchesini et al., 2003)** — synthetic diffraction patterns with known 2D/3D object support and phase; used for systematic comparison of phase retrieval algorithms

**Decision criteria:** CXIDB is the primary repository for CDI benchmarking. Miao CDI datasets for plane-wave CDI. Ptychography benchmarks (Thibault, Maiden) for scanning CDI. Simulated benchmarks (Marchesini) for systematic algorithm comparison. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All Phase Retrieval Algorithms

Please first ensure all the Phase Retrieval algorithms have been listed in `\pwm\public\algorithm_base\phase_retrieval\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/phase_retrieval. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Phase Retrieval solvers, please update the Phase Retrieval solver.

**Key algorithms to cover (1952–2026):**

_Classical Phase Retrieval (1952–1990):_
- Sayre's equation — half-wavelength phase relationship for oversampled diffraction (Sayre, Acta Cryst 1952)
- Gerchberg-Saxton (GS) algorithm — iterative Fourier projection between real and reciprocal space (Gerchberg & Saxton, Optik 1972) — the foundational iterative phase retrieval algorithm
- Error Reduction (ER) algorithm — real-space support constraint projection (Fienup, Appl Opt 1982)
- Hybrid Input-Output (HIO) algorithm — relaxed real-space constraint with feedback (Fienup, Appl Opt 1982) — the workhorse algorithm for CDI
- Oversampling phasing method — connection between oversampling ratio and phase retrieval uniqueness (Miao et al., JOSA A 1998)

_Advanced Iterative Methods (1990–2015):_
- Shrinkwrap — adaptive support determination via Gaussian-blurred autocorrelation (Marchesini et al., PRB 2003) — widely used for automated CDI
- RAAR — Relaxed Averaged Alternating Reflections (Luke, Inverse Problems 2005)
- Difference Map — generalized projection algorithm (Elser, JOSA A 2003)
- Charge flipping — sign-flipping phase retrieval (Oszlanyi & Suto, Acta Cryst A 2004)
- OSS — Oversampling Smoothness regularization (Rodriguez et al., J Appl Cryst 2013)
- Saddle-point optimization for phase retrieval (2010)
- Wirtinger Flow — gradient descent for phase retrieval with convergence guarantees (Candes et al., IEEE TIT 2015)
- Truncated Wirtinger Flow (Chen & Candes, 2017)
- Phase retrieval via PhaseLift — semidefinite relaxation (Candes et al., CPAM 2013)

_Ptychographic Phase Retrieval (2004–2026):_
- PIE — Ptychographic Iterative Engine (Rodenburg & Faulkner, APL 2004)
- ePIE — extended PIE with simultaneous probe and object recovery (Maiden & Rodenburg, Ultramicroscopy 2009) — the standard ptychographic engine
- rPIE — regularized PIE with improved convergence (Maiden et al., 2017)
- DM ptychography — Difference Map for ptychography (Thibault et al., Science 2008)
- Maximum likelihood ptychography — Poisson noise model (Thibault & Guizar-Sicairos, New J Phys 2012)
- Multi-slice ptychography — thick specimen depth-resolved reconstruction (Maiden et al., JOSA A 2012; Tsai et al., 2016)
- Mixed-state ptychography — partial coherence and multimode reconstruction (Thibault & Menzel, Nature 2013)
- Blind ptychography — simultaneous probe, object, and position recovery (Maiden et al., Ultramicroscopy 2015)
- Near-field ptychography — Fresnel regime ptychographic reconstruction (Stockmar et al., Sci Rep 2013)

_3D CDI & Tomographic Phase Retrieval (2006–2026):_
- 3D CDI — ab initio 3D structure from single-particle diffraction patterns (Chapman et al., Nat Phys 2006)
- Bragg CDI — coherent diffraction around Bragg peaks for nanocrystal strain mapping (Robinson & Harder, Nat Mater 2009)
- Ptychographic tomography — 3D reconstruction from ptycho-tilt-series (Dierolf et al., Nature 2010)
- Ankylography — single-shot 3D structure from sufficiently oversampled 2D pattern (Raines et al., Nature 2010)
- Multi-wavelength CDI — spectroscopic phase retrieval (2015)

_Deep Learning for Phase Retrieval (2017–2026):_
- PtychoNN — neural network for ptychographic reconstruction (Cherukara et al., Appl Phys Lett 2020)
- PhaseGAN — GAN-based phase retrieval (2020)
- Deep learning CDI — end-to-end learned phase retrieval replacing iterative algorithms (Rivenson et al., Light Sci Appl 2018)
- Unrolled optimization for phase retrieval — algorithm unrolling with learned parameters (Metzler et al., 2018)
- AutoPhaseNN — automated neural network for CDI (Wu et al., 2021)
- Physics-informed neural network phase retrieval (2022)
- Diffusion-prior phase retrieval (2024)
- Transformer-based ptychographic reconstruction (2024)
- Neural implicit representation for CDI (2023)
- Foundation model for coherent imaging (2025–2026)

#### Step 3: Update Phase Retrieval Solvers

After listing all Phase Retrieval solvers, update `algorithm_base/phase_retrieval/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Phase Retrieval solvers use the data format: `y` (H, W) or (N, H, W) oversampled diffraction pattern(s) (intensity = |F{object}|^2), `support` (H, W) binary support constraint, `probe` (H, W) illumination function for ptychography, `scan_positions` (N, 2) probe positions. The `PhaseRetrievalOperator` handles forward (object -> propagate -> |.|^2 -> diffraction pattern) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Phase Retrieval:**
- HIO reconstruction: phase error <0.1 rad on simulated CDI with known ground truth (Marchesini benchmark)
- Shrinkwrap CDI: converges to correct support for >90% of random initializations on standard benchmarks
- ePIE ptychography: phase accuracy <15 mrad on standard test objects (Maiden & Rodenburg benchmark)
- Bragg CDI: strain sensitivity ~1e-4 on nanocrystal benchmark (Robinson)
- PtychoNN: reconstruction speed 100–1000x faster than iterative with <5% quality loss
- Deep learning CDI: PSNR >30 dB on simulated CDI patterns
- All reference values from published papers and CXIDB depositions

**Verification criteria:**
- `done` — PWM within 0.1 rad phase error or 5% amplitude error of reference
- `partial` — 0.1–0.5 rad phase error or 5–15% amplitude error
- `gap` — >0.5 rad phase error or >15% amplitude error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged or phase retrieval stagnated

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'phase_retrieval' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/phase_retrieval/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Phase Retrieval. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/phase_retrieval/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/phase_retrieval/standard/`

---

### Talbot-Lau X-ray Grating Interferometry (`talbot_lau`) Modality Template

#### Step 1: Verify Standard Dataset

For Talbot-Lau Interferometry, what dataset do you use to verify? Is this dataset used for Talbot-Lau popular algorithms? Please ensure the standard dataset in `datasets/benchmark/talbot_lau/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Talbot-Lau standard dataset.

**Popular datasets to consider:**
- **Pfeiffer Group Grating Interferometry Benchmark (Pfeiffer et al., Nat Phys 2006; Nat Mater 2008)** — foundational Talbot-Lau phase-contrast and dark-field datasets of biological and materials specimens; the original demonstration datasets widely referenced in all subsequent GI work
- **Paul Scherrer Institut (PSI) GI Dataset (Thuering & Stampanoni, 2014)** — calibrated grating interferometry data with known phase and dark-field ground truth from monochromatized synchrotron and polychromatic lab sources; used for algorithm validation
- **Munich Compact Light Source GI Dataset (Bech et al., 2009)** — Talbot-Lau data from compact X-ray source; benchmark for lab-based phase-contrast imaging algorithms
- **ANPC Phase-Contrast Breast Imaging Dataset (Stampanoni et al., 2011; Arboleda et al., 2017)** — clinical grating-based phase-contrast mammography; used for medical imaging algorithm benchmarking
- **Simulated Talbot-Lau Benchmark (Revol et al., 2010)** — wave-optics simulated Talbot carpet and phase-stepping data with known phase, absorption, and scattering ground truth

**Decision criteria:** Pfeiffer group datasets are the foundational benchmark for Talbot-Lau GI. PSI calibrated datasets for quantitative algorithm validation. Simulated benchmarks (Revol) for systematic algorithm comparison. Use the dataset with the broadest algorithm coverage.

#### Step 2: List All Talbot-Lau Algorithms

Please first ensure all the Talbot-Lau algorithms have been listed in `\pwm\public\algorithm_base\talbot_lau\README.md` and `\pwm\public\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/talbot_lau. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Talbot-Lau solvers, please update the Talbot-Lau solver.

**Key algorithms to cover (1836–2026):**

_Talbot Effect & Grating Theory (1836–2006):_
- Talbot self-imaging — fractional Talbot effect theory for grating interferometry design (Talbot, 1836; Rayleigh, 1881; Cloetens et al., J Phys D 1999)
- Lau effect — extended source self-imaging with source grating (Lau, Ann Phys 1948)
- Wave-optical grating simulation — Fresnel propagation through grating structures (Weitkamp et al., Opt Express 2005)
- Talbot-Lau interferometer design — three-grating system for incoherent sources (Pfeiffer et al., Nat Phys 2006)
- Geometric optics approximation — ray-tracing for GI design (Donath et al., JSR 2009)

_Phase Stepping & Signal Extraction (2002–2020):_
- Phase stepping — multi-exposure Moire fringe sampling for absorption, phase, and dark-field extraction (Weitkamp et al., Opt Express 2005) — the standard acquisition method
- Fourier analysis of phase-stepping curves — DFT extraction of 0th, 1st, and 2nd harmonics (Pfeiffer et al., Nat Phys 2006)
- Single-shot Moire analysis — spatial harmonic extraction from single Moire pattern (Takeda et al., JOSA 1982; applied to GI by Bevins et al., 2012)
- Hilbert transform single-shot retrieval — 1D phase/dark-field extraction (Marschner et al., 2016)
- Least-squares phase stepping — optimal signal extraction from noisy stepping data (Seifert et al., 2019)
- Polychromatic correction — spectral averaging correction for broadband sources (Thuering et al., 2013)

_Phase Retrieval & Reconstruction (2006–2026):_
- Differential phase integration — 1D integration of phase gradient for quantitative phase (Pfeiffer et al., 2006)
- 2D phase integration — Poisson solver or Fourier-based integration of x/y phase gradients (Kottler et al., Opt Express 2007)
- Iterative phase integration — weighted least-squares Poisson integration for noisy data (2012)
- CT reconstruction of phase-contrast data — FBP with Hilbert filter for differential phase (Pfeiffer et al., 2007)
- Dark-field CT reconstruction — separate tomographic reconstruction of scattering signal (Bech et al., Phys Med Biol 2010)
- Statistical iterative reconstruction for GI-CT — Poisson noise model for low-dose phase-contrast CT (Xu et al., 2012)
- Compressed sensing GI — sparse acquisition with regularized reconstruction (2015)

_Dark-Field & Scattering Analysis (2008–2026):_
- Dark-field signal modeling — small-angle X-ray scattering interpretation (Yashiro et al., Opt Express 2010; Lynch et al., Appl Opt 2011)
- Directional dark-field imaging — anisotropic scattering detection via grating rotation (Jensen et al., Opt Express 2010)
- Quantitative dark-field — correlation length and particle size extraction (Strobl, Sci Rep 2014)
- Dark-field tomography for fiber orientation — 3D scattering tensor reconstruction (Sharma et al., PNAS 2017)
- Energy-resolved dark-field — spectral dark-field for material discrimination (2018)

_Deep Learning for Grating Interferometry (2019–2026):_
- Deep learning phase retrieval from reduced phase steps — CNN-based extraction from fewer exposures (Ge et al., 2020)
- Neural network Moire demodulation — single-shot deep learning signal extraction (2021)
- GAN-based artifact removal for GI — ring artifact and grating defect correction (2022)
- Deep learning GI-CT reconstruction — learned iterative reconstruction for phase-contrast CT (2023)
- Self-supervised denoising for GI — Noise2Noise adapted for phase-stepping data (2023)
- Physics-informed neural network for Talbot-Lau (2024)
- Foundation model for X-ray phase-contrast imaging (2025–2026)

_Advanced Interferometer Designs (2010–2026):_
- Dual-phase grating interferometer — two-grating system without analyzer grating (Miao et al., Nat Phys 2016)
- Coded aperture / structured illumination GI — non-periodic mask-based phase-contrast (2018)
- Polychromatic design optimization — grating period and duty cycle optimization for broadband sources (Thuering & Stampanoni, Phil Trans A 2014)
- Neutron grating interferometry — Talbot-Lau adapted for neutron imaging (Pfeiffer et al., PRL 2006)
- Edge illumination — alternative phase-contrast technique compatible with GI analysis methods (Olivo et al., 2001)

#### Step 3: Update Talbot-Lau Solvers

After listing all Talbot-Lau solvers, update `algorithm_base/talbot_lau/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Talbot-Lau solvers use the data format: `y` (Nsteps, H, W) phase-stepping image series, `stepping_positions` (Nsteps,) grating displacement values, `grating_params` (periods, distances, Talbot_order), `energy_spectrum` source energy distribution. The `TalbotLauOperator` handles forward (object -> absorption + phase shift + scattering -> Talbot interference -> detector) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Talbot-Lau:**
- Phase-stepping extraction: absorption/phase/dark-field SNR within 5% of theoretical limits on calibrated phantom
- Phase integration: phase accuracy <0.05 rad on known PMMA cylinder benchmark
- Dark-field quantification: scattering coefficient within 10% of SAXS-derived ground truth
- GI-CT FBP reconstruction: PSNR >30 dB on Shepp-Logan phase phantom
- Single-shot retrieval: within 15% of multi-step reference (SNR trade-off expected)
- Deep learning reduced-step: 3-step matches 8-step SNR within 2 dB
- All reference values from published papers and simulation ground truth

**Verification criteria:**
- `done` — PWM within 5% SNR or 0.05 rad phase accuracy of reference
- `partial` — 5–15% SNR shortfall or 0.05–0.2 rad phase error
- `gap` — >15% SNR shortfall or >0.2 rad phase error
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged or signal extraction failed

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'talbot_lau' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/talbot_lau/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Talbot-Lau. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/talbot_lau/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/talbot_lau/standard/`

---
