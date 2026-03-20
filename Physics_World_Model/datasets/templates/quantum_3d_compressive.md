
---

## Quantum & Novel Imaging — Modality Templates

---

### Ghost Imaging (`ghost_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For Ghost Imaging, what dataset do you use to verify? Is this dataset used for ghost imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/ghost_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ghost imaging standard dataset.

**Popular datasets to consider:**
- **Rice Single-Pixel Camera Experimental Data (Duarte et al., IEEE SPM 2008)** — the foundational single-pixel/ghost imaging experimental dataset; random binary patterns illuminating objects with a single bucket detector; used to validate computational ghost imaging and CS ghost imaging algorithms
- **Computational GI Benchmarks (Shapiro, PRA 2008; Bromberg et al., PRA 2009)** — simulated and experimental computational ghost imaging data using structured illumination; used by most GI algorithm papers for comparison
- **Quantum GI Datasets (Pittman et al., PRA 1995; Strekalov et al., PRL 1995)** — SPDC-based entangled photon ghost imaging data; the original quantum GI experiments; used as reference for quantum advantage studies
- **MNIST/CIFAR Computational GI Simulations** — standard natural image sets with simulated bucket detector measurements; widely used for deep learning ghost imaging papers (2018-2025)
- **Hadamard Basis GI Data (Sun et al., Opt. Express 2012)** — structured illumination ghost imaging using Hadamard patterns; efficient sampling benchmark
- **Shanghai Institute of Optics GI Data (Gong et al., 2016)** — experimental single-pixel imaging datasets at various compression ratios; frequently cited in Chinese GI research community

**Decision criteria:** Computational GI with DMD-based structured illumination data is the most widely used experimental benchmark. MNIST/CIFAR simulated GI data is the standard for deep learning GI papers. Use the dataset that appears in the largest number of ghost imaging reconstruction papers (2008-2026).

#### Step 2: List All Ghost Imaging Algorithms

Please first ensure all the ghost imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\ghost_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/ghost_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ghost imaging solvers, please update the ghost imaging solver.

**Key algorithms to cover (1950-2026):**

_Classical Correlation-Based (1995-2009):_
- Thermal light ghost imaging — intensity correlation reconstruction (Bennink et al., PRL 2002)
- Second-order correlation GI — G(2) correlation function reconstruction (Valencia et al., PRL 2005)
- Differential ghost imaging — DGI for background noise subtraction (Ferri et al., PRL 2010; roots 2005)
- Normalized ghost imaging — NGI for improved SNR (Sun et al., Opt. Lett. 2012; roots 2009)
- Higher-order correlation ghost imaging — G(3) and beyond (Chen et al., Opt. Express 2010)
- Pseudo-thermal ghost imaging with rotating ground glass (Scarcelli et al., PRL 2005)
- Computational ghost imaging — CGI with known patterns, no reference arm (Shapiro, PRA 2008)

_Compressive Sensing & Optimization (2009-2016):_
- CS Ghost Imaging — compressed sensing reconstruction from sub-Nyquist bucket measurements (Katz et al., APL 2009) — the foundational CS-GI paper
- Hadamard basis GI — structured illumination with Hadamard patterns for efficient sampling (Sun et al., 2012)
- Total variation regularized GI (Yu et al., Opt. Express 2014)
- Sparse Bayesian learning GI (2014)
- OMP-based ghost imaging reconstruction (2013)
- ADMM-based ghost imaging (2015)
- Fourier single-pixel imaging — Fourier basis patterns for efficient reconstruction (Zhang & Zhong, Opt. Express 2015)
- Sinusoidal structured illumination GI (2013)
- Iterative projection GI — noise-robust iterative reconstruction (2012)
- Gradient projection for sparse GI (2014)

_Deep Learning (2018-2026):_
- Deep learning ghost imaging — CNN reconstruction from bucket signals (Lyu et al., Sci. Rep. 2017; Shimobaba et al., Opt. Commun. 2018) — first DL-GI papers
- U-Net ghost imaging — encoder-decoder for GI (He et al., 2018)
- Physics-informed neural network GI (Li et al., Opt. Express 2020)
- GAN-based ghost imaging enhancement (Wang et al., Opt. Express 2019)
- Recurrent neural network for temporal GI (2020)
- Untrained neural network GI — DIP-style reconstruction (Boominathan et al., 2020)
- Transformer-based ghost imaging (2023)
- Self-supervised ghost imaging from limited measurements (2022)
- Diffusion-model ghost imaging reconstruction (2024)
- Foundation model for computational imaging including GI (2025)

#### Step 3: Update Ghost Imaging Solvers

After listing all ghost imaging solvers, update `algorithm_base/ghost_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ghost imaging solvers use the data format: `y` (M,) bucket detector measurements (single values per illumination pattern), `patterns` (M, H, W) illumination patterns (random, Hadamard, Fourier, etc.), where M is the number of measurements (M << H*W for compressive). The `GhostImagingOperator` handles the forward model `y_i = <pattern_i, x>` (inner product of pattern and scene) and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Ghost Imaging:**
- Computational GI at 50% sampling: correlation ~18 dB, CS (Katz) ~25 dB, DL-GI ~30 dB
- Computational GI at 10% sampling: correlation ~12 dB, CS ~20 dB, DL-GI ~25 dB
- MNIST GI at 25% sampling: CS ~22 dB, U-Net ~28 dB, Transformer ~30 dB
- Published PSNR/SSIM from DL ghost imaging papers (2018-2025)
- All reference values from original papers and recent survey articles

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'ghost_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/ghost_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/ghost_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/ghost_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ghost imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/ghost_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/ghost_imaging/standard/`

---

### Quantum Illumination (`quantum_illumination`) Modality Template

#### Step 1: Verify Standard Dataset

For Quantum Illumination, what dataset do you use to verify? Is this dataset used for QI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/quantum_illumination/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original QI standard dataset.

**Popular datasets to consider:**
- **MIT Lincoln Lab QI Radar Data (Barzanjeh et al., 2020; Luong et al., 2020)** — experimental microwave quantum illumination radar data demonstrating quantum advantage in target detection; entangled signal-idler pairs with thermal noise background; the first experimental QI demonstration
- **Simulated QI Data (Lloyd, Science 2008; Tan et al., PRL 2008)** — Monte Carlo simulated quantum illumination detection data with known signal-to-noise ratios; used for benchmarking quantum receivers against classical bounds; the standard theoretical benchmark
- **Entangled Photon Detection Benchmarks (Lopaeva et al., PRL 2013)** — optical-domain quantum illumination experimental data demonstrating 6 dB quantum advantage; photon counting statistics with known target reflectivity
- **Microwave Quantum Radar Simulations (Barzanjeh et al., PRL 2015)** — simulated electro-optomechanical QI data for microwave target detection; used for receiver design optimization
- **Gaussian State QI Simulation (Weedbrook et al., 2012)** — Gaussian quantum information framework simulations for continuous-variable QI protocols

**Decision criteria:** Simulated QI data with known theoretical bounds is essential for algorithm validation since experimental QI data is extremely scarce. Lopaeva et al. optical data provides real experimental reference. Use simulated data with published theoretical ROC curves for quantitative algorithm comparison.

#### Step 2: List All Quantum Illumination Algorithms

Please first ensure all the quantum illumination algorithms have been listed in `\Physics_World_Model\algorithm_base\quantum_illumination\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/quantum_illumination. Besides, you need to search all algorithms from 1950 to 2026. After listing all the QI solvers, please update the QI solver.

**Key algorithms to cover (1950-2026):**

_Theoretical Foundations (2008-2012):_
- Optimal quantum receiver for QI — quantum hypothesis testing with entangled signal (Lloyd, Science 2008) — the foundational QI paper
- Quantum illumination with Gaussian states — TMSV state protocol (Tan et al., PRL 2008) — proved 6 dB advantage over optimal classical
- Classical benchmark — coherent state illumination with homodyne detection (baseline)
- Helstrom bound computation — optimal quantum detection limit (Helstrom, 1969; applied to QI)
- Chernoff bound for QI — asymptotic error exponent (Audenaert et al., PRL 2007; applied 2009)

_Receiver Designs (2009-2016):_
- OPA receiver — optical parametric amplifier receiver for QI (Guha & Erkmen, PRA 2009) — first structured receiver achieving partial quantum advantage
- Phase-conjugate receiver — PC receiver for QI (Guha, PRA 2009) — achieves 3 dB of the 6 dB advantage
- Sum-frequency generation receiver — SFG receiver (Zhuang et al., PRL 2017; roots 2013)
- Feed-forward SFG receiver — iterated SFG for improved detection (Zhuang et al., PRA 2017)
- Quantum hypothesis testing — Neyman-Pearson and Bayesian detection frameworks
- Classical correlation detection — intensity correlation as classical baseline
- Heterodyne/homodyne detection — standard quantum optics measurement baselines

_Modern & Enhanced (2017-2026):_
- Microwave quantum illumination — electro-optomechanical transduction for radar (Barzanjeh et al., PRL 2015; experimental 2020)
- Quantum-enhanced lidar — QI applied to ranging and imaging (2019)
- Quantum illumination with non-Gaussian states — photon-subtracted TMSV (2020)
- Machine learning quantum state discrimination for QI (2021)
- Adaptive quantum illumination — measurement-adaptive protocols (2022)
- Quantum illumination in the presence of jamming (2023)
- Continuous-variable QI with heterodyne detection (2020)
- Quantum radar cross-section estimation (2024)
- Hybrid classical-quantum detection networks (2024)
- Quantum-enhanced target detection with deep learning post-processing (2025)

#### Step 3: Update Quantum Illumination Solvers

After listing all QI solvers, update `algorithm_base/quantum_illumination/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All QI solvers use the data format: `y_signal` (N_modes, N_samples) returned signal mode measurements, `y_idler` (N_modes, N_samples) retained idler mode measurements, `n_background` mean thermal background photon number, `n_signal` mean signal photon number per mode. The `QIOperator` handles the forward model for target present/absent hypotheses and quantum state evolution.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Quantum Illumination:**
- QI vs. classical illumination: 6 dB advantage in error exponent (Tan et al., PRL 2008 theoretical)
- OPA receiver: achieves 3 dB of the 6 dB advantage (Guha & Erkmen, PRA 2009)
- Phase-conjugate receiver: 3 dB advantage over classical (Guha, PRA 2009)
- SFG receiver: approaches full 6 dB advantage asymptotically (Zhuang et al., PRL 2017)
- Lopaeva et al. experimental: demonstrated quantum advantage at low signal brightness (N_S << 1)
- ROC curves: quantum vs. classical detection probability at fixed false alarm rate

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'quantum_illumination' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/quantum_illumination/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/quantum_illumination/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/quantum_illumination/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for quantum illumination. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/quantum_illumination/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/quantum_illumination/standard/`

---

### Entangled Photon (`entangled_photon`) Modality Template

#### Step 1: Verify Standard Dataset

For Entangled Photon Imaging, what dataset do you use to verify? Is this dataset used for entangled photon imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/entangled_photon/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original entangled photon imaging standard dataset.

**Popular datasets to consider:**
- **SPDC Photon Pair Imaging Data (Pittman et al., PRA 1995; Abouraddy et al., PRL 2001)** — spontaneous parametric down-conversion entangled photon pair imaging data; coincidence-counted images with known objects; the foundational entangled photon imaging dataset
- **Quantum Imaging Benchmark (Brida et al., Nature Photonics 2010)** — sub-shot-noise quantum imaging with entangled photon pairs; demonstrates quantum advantage in imaging below classical noise floor; standard benchmark for quantum-enhanced imaging
- **Ghost Imaging with Entangled Photons (Strekalov et al., PRL 1995; Aspden et al., 2013)** — ghost imaging using SPDC-generated entangled photon pairs; used for comparing quantum vs. classical ghost imaging
- **Interaction-Free Imaging Data (Elitzur & Vaidman, 1993; White et al., PRA 1998; experimental 2015)** — imaging using entangled photons that never interact with the object; quantum Zeno-based protocols
- **Undetected Photon Imaging Data (Lemos et al., Nature 2014)** — imaging with photons that never interact with the object via induced coherence; mid-IR imaging using visible detection; breakthrough quantum imaging modality
- **Quantum-Secured Imaging Data (Malik et al., APL 2012)** — imaging with photon number correlations for authentication; used for quantum imaging security benchmarks

**Decision criteria:** SPDC photon pair data with coincidence counting is the most widely used experimental platform. Brida et al. sub-shot-noise data for quantum advantage demonstration. Use the dataset that appears in the largest number of entangled photon imaging papers (1995-2026).

#### Step 2: List All Entangled Photon Algorithms

Please first ensure all the entangled photon imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\entangled_photon\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/entangled_photon. Besides, you need to search all algorithms from 1950 to 2026. After listing all the entangled photon imaging solvers, please update the entangled photon imaging solver.

**Key algorithms to cover (1950-2026):**

_Foundational Quantum Imaging (1995-2009):_
- Coincidence counting imaging — spatial correlation measurement with SPDC pairs (Pittman et al., PRA 1995) — the first entangled photon imaging demonstration
- Quantum ghost imaging with entangled photons — two-photon correlation imaging (Strekalov et al., PRL 1995)
- Sub-shot-noise imaging — exploiting photon number correlations below classical limit (Brida et al., Nature Photonics 2010; roots 2005)
- Quantum lithography — two-photon absorption for sub-Rayleigh resolution (Boto et al., PRL 2000)
- Entangled two-photon absorption imaging (Lee & Goodson, JPCL 2006)
- Quantum-secured direct communication imaging (2005)

_Quantum Protocols & Tomography (2010-2016):_
- Quantum illumination protocols for entangled imaging — QI applied to spatial imaging (Lloyd, 2008; Lopaeva et al., PRL 2013)
- Quantum state tomography for imaging — full quantum state reconstruction of photon pairs (James et al., PRA 2001; applied to imaging 2010)
- Interaction-free imaging — quantum Zeno effect-based measurement (Elitzur & Vaidman, 1993; experimental imaging 2015)
- Imaging with undetected photons — induced coherence protocol (Lemos et al., Nature 2014)
- Quantum-enhanced phase imaging — entangled photon interferometry (Ono et al., Nature Communications 2013)
- Quantum holography — entangled photon holographic reconstruction (Abouraddy et al., PRL 2001; extended 2012)
- Biphoton spatial mode analysis (2013)
- Heralded single-photon imaging (Aspden et al., Optica 2015)

_Deep Learning & Modern (2020-2026):_
- Deep learning quantum image reconstruction — CNN for coincidence image enhancement (2023)
- Neural network photon counting image restoration (2022)
- Physics-informed neural network for entangled imaging (2024)
- Quantum-classical hybrid imaging with ML post-processing (2023)
- Generative model for quantum image super-resolution (2024)
- Transformer-based coincidence image denoising (2025)
- Self-supervised learning for low-photon-count quantum imaging (2024)
- Diffusion-model quantum image enhancement (2025)

#### Step 3: Update Entangled Photon Solvers

After listing all entangled photon imaging solvers, update `algorithm_base/entangled_photon/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All entangled photon imaging solvers use the data format: `y_coincidence` (H, W) coincidence count image from spatially-resolved detectors, `y_singles_signal` (H, W) singles counts on signal arm, `y_singles_idler` (H, W) singles counts on idler arm, `pump_profile` (H, W) pump beam spatial profile. The `EntangledPhotonOperator` handles the forward model for biphoton correlation, coincidence counting, and quantum state propagation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Entangled Photon Imaging:**
- Sub-shot-noise imaging: demonstrated 1.5 dB below classical shot noise limit (Brida et al., 2010)
- Coincidence imaging SNR: ~10-15 dB improvement over singles counting in high-background regime
- Interaction-free imaging: >90% image fidelity with <25% object interrogation (Elitzur-Vaidman bomb tester efficiency)
- Undetected photon imaging: spatial resolution matching detected photon wavelength, not interacting wavelength
- Published metrics from entangled photon imaging papers (visibility, contrast, SNR)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'entangled_photon' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/entangled_photon/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for entangled photon imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/entangled_photon/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/entangled_photon/standard/`

---

## 3D Scene Reconstruction — Modality Templates

---

### NeRF (`nerf`) Modality Template

#### Step 1: Verify Standard Dataset

For NeRF (Neural Radiance Fields), what dataset do you use to verify? Is this dataset used for NeRF popular algorithms? Please ensure the standard dataset in `datasets/benchmark/nerf/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original NeRF standard dataset.

**Popular datasets to consider:**
- **Synthetic-NeRF / Blender Dataset (Mildenhall et al., ECCV 2020)** — 8 synthetic scenes (Chair, Drums, Ficus, Hotdog, Lego, Materials, Mic, Ship) rendered with Blender at 800x800; 100 training + 200 test views per scene; the canonical NeRF benchmark dataset used by virtually all NeRF papers
- **LLFF — Local Light Field Fusion (Mildenhall et al., SIGGRAPH 2019)** — 8 real forward-facing scenes (Fern, Flower, Fortress, Horns, Leaves, Orchids, Room, T-Rex) captured with a cellphone; used for forward-facing novel view synthesis evaluation
- **Mip-NeRF 360 Dataset (Barron et al., CVPR 2022)** — 9 unbounded real scenes (indoor + outdoor) captured at varying scales; 360-degree captures; the standard benchmark for unbounded NeRF methods
- **DTU MVS Dataset (Jensen et al., 2014)** — 124 indoor scenes with structured-light ground-truth 3D geometry; used for NeRF geometry quality evaluation
- **Tanks and Temples (Knapitsch et al., SIGGRAPH 2017)** — large-scale outdoor scenes with LiDAR ground truth; used for evaluating scalability of NeRF methods
- **ScanNet++ (Yeshwanth et al., ICCV 2023)** — 1500+ high-quality indoor scenes with iPhone + DSLR + LiDAR; emerging standard for indoor reconstruction

**Decision criteria:** The Synthetic-NeRF/Blender dataset is the undisputed gold standard for NeRF benchmarking (2020-2026); Mip-NeRF 360 for unbounded scenes; LLFF for forward-facing. Use the dataset that appears in the largest number of NeRF/neural rendering papers.

#### Step 2: List All NeRF Algorithms

Please first ensure all the NeRF algorithms have been listed in `\Physics_World_Model\algorithm_base\nerf\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/nerf. Besides, you need to search all algorithms from 1950 to 2026. After listing all the NeRF solvers, please update the NeRF solver.

**Key algorithms to cover (2020-2026):**

_Foundational NeRF (2020-2021):_
- NeRF — Neural Radiance Fields (Mildenhall et al., ECCV 2020) — the foundational paper; MLP mapping (x,y,z,theta,phi) to (RGB, sigma); positional encoding; hierarchical volume rendering
- NeRF++ — analyzing and improving NeRF for unbounded scenes (Zhang et al., 2020)
- NeRF-in-the-Wild (NeRF-W) — appearance and transient embeddings for internet photo collections (Martin-Brualla et al., CVPR 2021)
- Mip-NeRF — anti-aliased cone tracing replacing ray tracing (Barron et al., ICCV 2021) — integrated positional encoding for multi-scale rendering
- NSVF — Neural Sparse Voxel Fields with sparse octree (Liu et al., NeurIPS 2020)
- PixelNeRF — conditioning NeRF on image features for few-shot (Yu et al., CVPR 2021)
- IBRNet — image-based rendering with NeRF (Wang et al., CVPR 2021)
- MVSNeRF — multi-view stereo + NeRF (Chen et al., ICCV 2021)

_Efficient NeRF (2022):_
- Instant-NGP — hash-encoded multi-resolution feature grids for 5-second training (Muller et al., SIGGRAPH 2022) — breakthrough in training speed
- Plenoxels — sparse voxel grid without neural networks (Fridovich-Keil et al., CVPR 2022) — proved MLPs are not necessary
- TensoRF — tensorial radiance fields with VM decomposition (Chen et al., ECCV 2022)
- Mip-NeRF 360 — unbounded anti-aliased NeRF with contraction and distortion loss (Barron et al., CVPR 2022)
- DirectVoxGO — direct voxel grid optimization (Sun et al., CVPR 2022)
- Point-NeRF — point cloud-based neural radiance fields (Xu et al., CVPR 2022)

_Nerfstudio & Modern (2023-2024):_
- Nerfacto (Nerfstudio, Tancik et al., SIGGRAPH 2023) — production-quality NeRF combining best practices from multiple methods
- ZipNeRF — anti-aliased grid-based NeRF combining Mip-NeRF 360 and Instant-NGP (Barron et al., ICCV 2023) — state-of-the-art quality
- Tri-MipRF — tri-plane mip radiance fields (Hu et al., 2023)
- K-Planes — explicit decomposition for static and dynamic scenes (Fridovich-Keil et al., CVPR 2023)
- 3D Gaussian Splatting (Kerbl et al., SIGGRAPH 2023) — not strictly NeRF but the dominant alternative
- NeuS2 — neural implicit surface fast training (Wang et al., 2023)
- Neuralangelo — high-fidelity neural surface reconstruction (Li et al., CVPR 2023)

_Dynamic NeRF:_
- D-NeRF — deformable NeRF for dynamic scenes (Pumarola et al., CVPR 2021)
- Nerfies — deformable neural radiance fields (Park et al., ICCV 2021)
- HyperNeRF — higher-dimensional NeRF for topological changes (Park et al., SIGGRAPH Asia 2021)
- Neural Scene Flow Fields (Li et al., CVPR 2021)
- NeRF-DS — dynamic specular NeRF (2023)
- DynIBaR — neural dynamic image-based rendering (Li et al., CVPR 2023)

_Special Capabilities:_
- Block-NeRF — large-scale scene NeRF (Tancik et al., CVPR 2022)
- Mega-NeRF — city-scale NeRF (Turki et al., CVPR 2022)
- Urban Radiance Fields — LiDAR-supervised NeRF (Rematas et al., CVPR 2022)
- Instruct-NeRF2NeRF — text-based NeRF editing (2023)
- DreamFusion — text-to-3D via SDS (Poole et al., ICLR 2023)
- SplaTAM — dense visual SLAM using 3DGS (2024)

#### Step 3: Update NeRF Solvers

After listing all NeRF solvers, update `algorithm_base/nerf/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All NeRF solvers use the data format: `images` (N, H, W, 3) posed input images, `poses` (N, 4, 4) camera-to-world transformation matrices, `intrinsics` (3, 3) or (N, 3, 3) camera intrinsics, `bounds` (N, 2) near/far depth bounds. The `NeRFOperator` handles volume rendering (ray marching, quadrature), positional encoding, and novel view synthesis.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for NeRF:**
- Synthetic-NeRF (Blender, average over 8 scenes): NeRF ~31.0 dB, Mip-NeRF ~33.1 dB, Instant-NGP ~33.2 dB, TensoRF ~33.1 dB, Plenoxels ~31.7 dB, Nerfacto ~31.5 dB, ZipNeRF ~33.6 dB, 3DGS ~33.3 dB
- LLFF (average over 8 scenes): NeRF ~26.5 dB, Mip-NeRF ~26.9 dB
- Mip-NeRF 360 (average): Mip-NeRF 360 ~27.7 dB (indoor) / ~24.5 dB (outdoor), ZipNeRF ~28.5 dB / ~25.3 dB, 3DGS ~27.5 dB / ~24.6 dB
- All SSIM and LPIPS metrics from published papers and Nerfstudio benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'nerf' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/nerf/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/nerf/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/nerf/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for NeRF. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/nerf/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/nerf/standard/`

---

### Gaussian Splatting (`gaussian_splatting`) Modality Template

#### Step 1: Verify Standard Dataset

For 3D Gaussian Splatting, what dataset do you use to verify? Is this dataset used for Gaussian Splatting popular algorithms? Please ensure the standard dataset in `datasets/benchmark/gaussian_splatting/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Gaussian Splatting standard dataset.

**Popular datasets to consider:**
- **Mip-NeRF 360 Dataset (Barron et al., CVPR 2022)** — 9 unbounded real scenes (indoor: bonsai, counter, kitchen, room; outdoor: bicycle, flowers, garden, stump, treehill); the primary benchmark for 3DGS and all follow-up papers; 360-degree captures at varying scales
- **Tanks and Temples (Knapitsch et al., SIGGRAPH 2017)** — large-scale outdoor scenes (Truck, Train) with LiDAR ground truth; used for evaluating 3DGS scalability and geometry accuracy
- **Deep Blending Dataset (Hedman et al., SIGGRAPH Asia 2018)** — indoor scenes (DrJohnson, Playroom) for view synthesis with challenging view-dependent effects; standard 3DGS evaluation dataset
- **DTU MVS Dataset (Jensen et al., 2014)** — 124 structured indoor scenes; used for geometry evaluation of Gaussian Splatting methods (especially SuGaR, 2DGS)
- **ScanNet++ (Yeshwanth et al., ICCV 2023)** — 1500+ high-quality indoor scenes; emerging standard for indoor 3DGS evaluation
- **Synthetic-NeRF / Blender (Mildenhall et al., 2020)** — 8 synthetic scenes; occasionally used for 3DGS but less standard than Mip-NeRF 360

**Decision criteria:** Mip-NeRF 360 is the undisputed gold standard for 3DGS benchmarking (2023-2026); Tanks & Temples and Deep Blending are secondary benchmarks used by the original 3DGS paper. Use the dataset that appears in the largest number of Gaussian Splatting papers.

#### Step 2: List All Gaussian Splatting Algorithms

Please first ensure all the Gaussian Splatting algorithms have been listed in `\Physics_World_Model\algorithm_base\gaussian_splatting\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/gaussian_splatting. Besides, you need to search all algorithms from 1950 to 2026. After listing all the Gaussian Splatting solvers, please update the Gaussian Splatting solver.

**Key algorithms to cover (2023-2026):**

_Original & Core (2023):_
- 3D Gaussian Splatting — 3DGS (Kerbl et al., SIGGRAPH 2023) — the foundational paper; anisotropic 3D Gaussians with learnable position, covariance, opacity, SH color; differentiable tile-based rasterizer; real-time rendering at >100 FPS
- SuGaR — Surface-Aligned Gaussian Splatting (Guedon & Lepetit, CVPR 2024; arXiv 2023) — extracts mesh from 3D Gaussians via regularization to approximate surfaces
- GaussianPro — 3D Gaussian Splatting with Progressive Propagation (Cheng et al., ICML 2024; arXiv 2024) — progressive densification using depth and normal priors

_Quality & Anti-Aliasing (2024):_
- Mip-Splatting — alias-free 3D Gaussian Splatting (Yu et al., CVPR 2024) — 3D smoothing filter and 2D Mip filter for anti-aliased multi-scale rendering
- 2D Gaussian Splatting — 2DGS (Huang et al., SIGGRAPH 2024) — flat 2D Gaussian disks for better surface reconstruction and reduced floaters
- Scaffold-GS — structured 3D Gaussians with anchor points (Lu et al., CVPR 2024) — anchor-based representation reducing redundancy
- GaussianShader — 3D Gaussians with shading functions (Jiang et al., CVPR 2024) — normal-based shading for view-dependent appearance

_Efficient & Compressed (2024):_
- Compressed 3DGS — compact 3D scene representation (Niedermayr et al., CVPR 2024) — sensitivity-aware quantization and entropy coding
- LightGaussian — unbounded 3D Gaussian compression (Fan et al., NeurIPS 2024; arXiv 2024) — Gaussian pruning + SH distillation + quantization for 10-25x compression
- Mini-Splatting — representing scenes with a configurable number of Gaussians (Fang et al., ECCV 2024)
- HAC — Hash-grid Assisted Context for 3DGS compression (Chen et al., ECCV 2024)
- Compact3D — compressing Gaussian splat radiance fields (Lee et al., CVPR 2024)

_Dynamic & 4D (2023-2024):_
- Dynamic 3D Gaussians — tracking via persistent dynamic view synthesis (Luiten et al., 3DV 2024) — Gaussian trajectories for dynamic scenes
- 4D Gaussian Splatting — 4DGS for real-time dynamic scene rendering (Wu et al., CVPR 2024) — 4D Gaussian primitives with spatial-temporal structure
- Deformable 3D Gaussians — deformable 3DGS for monocular video (Yang et al., CVPR 2024) — deformation MLP applied to canonical Gaussians
- SC-GS — sparse-controlled Gaussian Splatting for editable dynamic scenes (Huang et al., CVPR 2024)
- GaussianFlow — splatting and advecting for dynamic scene (Lin et al., 2024)

_Generative & Special (2024-2025):_
- DreamGaussian — text-to-3D Gaussian generation (Tang et al., ICLR 2024) — SDS-based 3DGS generation from text prompts
- GaussianEditor — editing 3D Gaussians via semantic guidance (Chen et al., CVPR 2024)
- SplaTAM — dense SLAM with 3D Gaussians (Keetha et al., CVPR 2024)
- Gaussian Grouping — segment anything in 3D Gaussians (Ye et al., ECCV 2024)
- PhysGaussian — physics-integrated 3D Gaussians (Xie et al., CVPR 2024)
- GaussianAvatar — animatable human avatar (2024)
- GOF — Gaussian Opacity Fields for efficient surface reconstruction (Yu et al., 2024)
- StopThePop — sorted Gaussian Splatting for view-consistent rendering (Radl et al., SIGGRAPH 2024)

#### Step 3: Update Gaussian Splatting Solvers

After listing all Gaussian Splatting solvers, update `algorithm_base/gaussian_splatting/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All Gaussian Splatting solvers use the data format: `images` (N, H, W, 3) posed input images, `poses` (N, 4, 4) camera-to-world transformation matrices, `intrinsics` (3, 3) or (N, 3, 3) camera intrinsics, `points_init` (P, 3) initial SfM point cloud from COLMAP. The `GaussianSplattingOperator` handles differentiable Gaussian rasterization, densification/pruning control, and novel view rendering.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Gaussian Splatting:**
- Mip-NeRF 360 (average over 9 scenes): 3DGS ~27.2 dB / 0.815 SSIM / 0.214 LPIPS, Mip-Splatting ~27.6 dB, 2DGS ~26.8 dB, Scaffold-GS ~27.5 dB
- Tanks and Temples: 3DGS ~23.1 dB (Truck) / ~21.2 dB (Train), Mip-Splatting ~23.5 dB (Truck)
- Deep Blending: 3DGS ~29.4 dB (DrJohnson) / ~30.0 dB (Playroom)
- Training time: 3DGS ~25 min, Mip-Splatting ~40 min, 2DGS ~30 min (single A6000 GPU)
- Rendering speed: 3DGS >100 FPS at 1080p, LightGaussian >100 FPS with 10x compression
- All reference values from published papers and gsplat benchmarks

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'gaussian_splatting' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/gaussian_splatting/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for Gaussian Splatting. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/gaussian_splatting/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/gaussian_splatting/standard/`

---

## Compressive & Ultrafast — Modality Templates

---

### CUP (`cup`) Modality Template

#### Step 1: Verify Standard Dataset

For Compressed Ultrafast Photography (CUP), what dataset do you use to verify? Is this dataset used for CUP popular algorithms? Please ensure the standard dataset in `datasets/benchmark/cup/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original CUP standard dataset.

**Popular datasets to consider:**
- **CUP Benchmark Data (Liang et al., Nature 2014)** — the foundational CUP dataset; streak camera-based compressed ultrafast photography captures at 100 billion frames/s; includes laser pulse propagation, photon racing through scattering media, and fluorescence lifetime imaging; the most widely cited CUP dataset
- **T-CUP Data (Liang et al., Light: Science & Applications 2018)** — trillion-frame-per-second CUP data at 10 Tframe/s; captures at 10x faster temporal resolution; ultrafast transient phenomena at femtosecond timescales
- **CUP Simulation Benchmark (Gao et al., 2014)** — simulated CUP measurements with known ground-truth 3D (x,y,t) datacubes; used for quantitative algorithm comparison
- **Ultrafast Phenomena Reference Data** — laser ablation, plasma dynamics, photonic Mach cone, ultrafast fluorescence decay datacubes with known temporal profiles
- **STAMP Data (Nakagawa et al., Nature Photonics 2014)** — Sequentially Timed All-optical Mapping Photography; complementary ultrafast imaging dataset for comparison

**Decision criteria:** Liang et al. 2014 CUP data is the gold standard for CUP algorithm evaluation. Simulated CUP data with known ground truth for quantitative PSNR/SSIM comparison. Use the dataset most widely referenced in CUP/ultrafast imaging papers (2014-2026).

#### Step 2: List All CUP Algorithms

Please first ensure all the CUP algorithms have been listed in `\Physics_World_Model\algorithm_base\cup\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/cup. Besides, you need to search all algorithms from 1950 to 2026. After listing all the CUP solvers, please update the CUP solver.

**Key algorithms to cover (2014-2026):**

_Foundational CUP Reconstruction (2014-2016):_
- TwIST for CUP — Two-step Iterative Shrinkage/Thresholding for CUP reconstruction (Bioucas-Dias & Figueiredo, TIP 2007; applied to CUP by Liang et al., Nature 2014) — the original CUP reconstruction algorithm; TV + wavelet regularization
- FISTA for CUP — Fast Iterative Shrinkage-Thresholding for CUP (Beck & Teboulle, 2009; applied to CUP 2015)
- Augmented Lagrangian for CUP — constrained optimization framework (2015)
- Compressed sensing baseline for CUP — standard L1 minimization (2014)

_Optimization-Based (2016-2020):_
- GAP-TV CUP — Generalized Alternating Projection with Total Variation for CUP (Yuan et al., 2016; adapted for ultrafast) — efficient TV-based CUP reconstruction
- PnP CUP — Plug-and-Play priors for CUP reconstruction (Chan et al., 2017; applied to CUP 2019) — using pretrained denoisers (BM3D, DnCNN) as regularizers
- ADMM CUP — Alternating Direction Method of Multipliers for CUP (2018)
- Spatial-temporal sparsity CUP — joint spatial and temporal sparsity exploitation (Liang et al., 2015)
- Multi-scale CUP reconstruction — hierarchical CUP recovery (2017)
- Rank minimization CUP — low-rank + sparse for temporal correlation (2018)
- Bayesian CUP reconstruction (2019)

_Deep Learning (2020-2026):_
- Deep learning CUP — CNN-based CUP reconstruction (2020) — end-to-end learned CUP recovery
- U-Net CUP — encoder-decoder architecture for CUP (2020)
- Deep unfolding CUP — ADMM-Net / ISTA-Net adapted for CUP (2021)
- PnP-DRUNet CUP — Plug-and-Play with deep denoiser for CUP (2022)
- Transformer-based CUP reconstruction (2023)
- Physics-informed neural network CUP — encoding streak camera physics (2023)
- Diffusion-model CUP — score-based reconstruction for ultrafast imaging (2024)
- Self-supervised CUP — training without ground truth (2024)
- Foundation model for compressive ultrafast imaging (2025)

#### Step 3: Update CUP Solvers

After listing all CUP solvers, update `algorithm_base/cup/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All CUP solvers use the data format: `y` (H, W_sheared) 2D streak camera image (spatially encoded + temporally sheared measurement), `mask` (H, W) spatial encoding mask (DMD or physical mask), `shear_params` dict containing temporal shear rate and number of temporal frames. The `CUPOperator` handles the forward model `y = C * T * x` where C is spatial encoding, T is temporal shearing by the streak camera, and x is the (H, W, N_t) spatiotemporal datacube.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for CUP:**
- Simulated CUP 256x256x50 datacube: TwIST ~25 dB, GAP-TV ~28 dB, PnP-BM3D ~30 dB, DL-CUP ~33 dB
- Liang 2014 laser pulse: qualitative agreement with streak camera ground truth (temporal profile RMSE <5%)
- T-CUP femtosecond: temporal resolution ~0.1 ps verified against pump-probe reference
- Published PSNR/SSIM from CUP reconstruction papers (Liang group, 2014-2025)

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'cup' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/cup/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/cup/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/cup/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for CUP. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/cup/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/cup/standard/`

---

### SD-CASSI (`sd_cassi`) Modality Template

#### Step 1: Verify Standard Dataset

For Single-Disperser CASSI (SD-CASSI), what dataset do you use to verify? Is this dataset used for SD-CASSI popular algorithms? Please ensure the standard dataset in `datasets/benchmark/sd_cassi/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original SD-CASSI standard dataset.

**Popular datasets to consider:**
- **KAIST Dataset Adapted for SD-CASSI (Choi et al., 2017)** — 30 hyperspectral scenes captured with a custom hyperspectral camera at 2704x3376 spatial and 28 spectral bands; widely used for DD-CASSI and adapted for SD-CASSI simulation by applying single-disperser forward model
- **CAVE Hyperspectral Dataset (Yasuma et al., 2010)** — 32 indoor scenes at 512x512 spatial and 31 spectral bands (400-700 nm at 10 nm); one of the most widely used HSI datasets; commonly used for SD-CASSI simulation
- **ICVL Hyperspectral Dataset (Arad & Ben-Shahar, ICCV 2016)** — 201 outdoor/indoor scenes at 1392x1300 spatial and 31 spectral bands; large-scale benchmark for spectral reconstruction
- **SD-CASSI Prototype Captures (Wagadarikar et al., Appl. Opt. 2008; Kittle et al., Appl. Opt. 2010)** — real single-disperser CASSI measurements from prototype systems; includes calibrated coded aperture and prism dispersion; used for validating SD-CASSI reconstruction on real hardware
- **Harvard Hyperspectral Dataset (Chakrabarti & Zickler, CVPR 2011)** — 50 indoor/outdoor scenes at 31 bands; used for HSI recovery benchmarks

**Decision criteria:** CAVE and ICVL with simulated SD-CASSI forward model are the most common for algorithm development. SD-CASSI prototype data from Wagadarikar/Kittle for real-data validation. Use the dataset that appears in the largest number of SD-CASSI reconstruction papers (2008-2026).

#### Step 2: List All SD-CASSI Algorithms

Please first ensure all the SD-CASSI algorithms have been listed in `\Physics_World_Model\algorithm_base\sd_cassi\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/sd_cassi. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SD-CASSI solvers, please update the SD-CASSI solver.

**Key algorithms to cover (2007-2026):**

_Classical SD-CASSI (2007-2012):_
- TwIST for SD-CASSI — Two-step Iterative Shrinkage/Thresholding with TV+wavelet (Bioucas-Dias & Figueiredo, TIP 2007; applied to CASSI by Wagadarikar et al., Appl. Opt. 2008) — the original SD-CASSI reconstruction algorithm
- GPSR for SD-CASSI — Gradient Projection for Sparse Reconstruction (Figueiredo et al., JSTSP 2007; applied 2008)
- Inversion by pseudoinverse — simple least-squares baseline for SD-CASSI (2008)
- Truncated matrix inversion — regularized direct inversion (2009)

_Optimization-Based (2012-2019):_
- GAP-TV for SD-CASSI — Generalized Alternating Projection with Total Variation (Yuan, 2016; adapted for SD-CASSI) — efficient TV-based SD-CASSI reconstruction
- ADMM SD-CASSI — Alternating Direction Method of Multipliers for SD-CASSI (Tan et al., J. Opt. Soc. Am. A 2015) — handles the SD-CASSI forward model with spectrally-varying shift
- GPSR-BB for SD-CASSI — Barzilai-Borwein gradient projection (2013)
- 3D-TV for SD-CASSI — 3D total variation exploiting joint spatial-spectral smoothness (2014)
- Dictionary learning SD-CASSI — adaptive sparse representation (2014)
- Low-rank and sparse SD-CASSI — exploiting spectral low-rank structure (Golbabaee & Vandergheynst, 2012)
- PnP-ADMM SD-CASSI — Plug-and-Play with BM3D/DnCNN (2018)
- Bayesian SD-CASSI (2015)

_Deep Learning (2019-2026):_
- Lambda-Net adapted for SD-CASSI — spectral-spatial CNN (Miao et al., ICCV 2019; adapted for SD-CASSI forward model)
- MST adapted for SD-CASSI — Mask-guided Spectral-wise Transformer (Cai et al., CVPR 2022; adapted for SD-CASSI by modifying the mask-aware attention to single-disperser shift pattern)
- TSA-Net for SD-CASSI — spatial-spectral self-attention (Meng et al., ECCV 2020; adapted)
- DGSMP adapted for SD-CASSI — deep Gaussian scale mixture prior (Huang et al., CVPR 2021; adapted)
- GAP-Net for SD-CASSI — deep unfolding GAP for SD-CASSI (2020)
- CST for SD-CASSI — spectral transformer (Cai et al., ECCV 2022; adapted)
- DAUHST for SD-CASSI — degradation-aware unfolding with HSI spatial-spectral transformer (Cai et al., NeurIPS 2022; adapted)
- Deep unfolding SD-CASSI — ADMM-Net for single-disperser (2021)
- Diffusion-model SD-CASSI — score-based spectral reconstruction (2024)
- DL SD-CASSI end-to-end with joint mask optimization (2023)
- Foundation model for spectral imaging including SD-CASSI (2025)

#### Step 3: Update SD-CASSI Solvers

After listing all SD-CASSI solvers, update `algorithm_base/sd_cassi/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SD-CASSI solvers use the data format: `y` (H, W+L-1) 2D compressed measurement (spatial encoding + spectral dispersion), `mask` (H, W) coded aperture pattern, `dispersion` float or array specifying the prism dispersion (wavelength-dependent spatial shift in pixels). The `SDCASSIOperator` handles the forward model `y(x',y') = sum_lambda mask(x,y) * scene(x, y'-d(lambda), lambda)` where d(lambda) is the spectral dispersion, and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SD-CASSI:**
- CAVE 256x256x28 simulation: TwIST ~25 dB, GAP-TV ~28 dB, ADMM ~28.5 dB, Lambda-Net ~32 dB, MST-adapted ~35 dB, DAUHST-adapted ~36 dB
- ICVL simulation: TwIST ~26 dB, GAP-TV ~29 dB, MST-adapted ~36 dB
- SD-CASSI prototype real data: qualitative comparison with reference spectral measurements
- Note: SD-CASSI typically yields 1-2 dB lower than DD-CASSI due to spectral ambiguity in the single-shot measurement

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'sd_cassi' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/sd_cassi/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/sd_cassi/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/sd_cassi/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SD-CASSI. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/sd_cassi/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/sd_cassi/standard/`

---

### SPC Block (`spc_block`) Modality Template

#### Step 1: Verify Standard Dataset

For Block-Diagonal SPC (Single-Pixel Camera with block-diagonal measurement matrices), what dataset do you use to verify? Is this dataset used for block-diagonal SPC popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spc_block/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original block-diagonal SPC standard dataset.

**Popular datasets to consider:**
- **Set11 with Block-Diagonal Measurements** — 11 standard grayscale test images (Barbara, Boats, Cameraman, Fingerprint, Flinstones, Foreman, House, Lena, Monarch, Parrots, Peppers) at 256x256; divided into non-overlapping 33x33 blocks with block-diagonal Gaussian measurement matrix at 25% sampling ratio; the canonical benchmark for block-CS and ISTA-Net+
- **BSD68 with Block-Diagonal Measurements (Martin et al., ICCV 2001)** — 68 natural images from Berkeley Segmentation Dataset; 33x33 block-wise measurements at 1%, 4%, 10%, 25%, 50% sampling ratios; widely used for deep CS algorithm evaluation
- **ISTA-Net Benchmark 25% Ratio (Zhang & Ghanem, CVPR 2018)** — the specific benchmark setup: 33x33 blocks, Gaussian measurement matrix, 25% CS ratio; used by ISTA-Net, ISTA-Net+, AMP-Net, TransCS, and virtually all deep unfolding CS papers
- **Urban100 / DIV2K with Block CS** — high-resolution images for block-wise CS evaluation at various ratios
- **Standard CS Measurement Matrices** — Gaussian, Bernoulli, partial Hadamard, learned measurement matrices at block level

**Decision criteria:** Set11 + BSD68 at 25% block-diagonal measurement ratio (33x33 blocks) is the undisputed standard for block CS benchmarking, established by ISTA-Net and used by all subsequent deep unfolding papers. Use this exact setup for fair comparison.

#### Step 2: List All SPC Block Algorithms

Please first ensure all the SPC block algorithms have been listed in `\Physics_World_Model\algorithm_base\spc_block\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spc_block. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SPC block solvers, please update the SPC block solver.

**Key algorithms to cover (2006-2026):**

_Classical Block CS (2006-2014):_
- Block-wise OMP — Orthogonal Matching Pursuit applied independently to each block (Tropp & Gilbert, TIT 2007; block extension)
- Block-wise Basis Pursuit — L1 minimization per block (Chen et al., 2001; block extension)
- Block ISTA — Iterative Shrinkage-Thresholding per block (Daubechies et al., CPAM 2004)
- Block FISTA — Fast ISTA per block (Beck & Teboulle, SIAM J Imaging 2009)
- Block CoSaMP — compressive sampling matching pursuit per block (Needell & Tropp, 2009)
- Block-wise total variation CS (Lustig et al., 2007; block adaptation)
- Block-wise AMP — Approximate Message Passing per block (Donoho et al., PNAS 2009)
- BM3D-CS — BM3D regularized block CS (Egiazarian et al., 2007)
- D-AMP — denoising-based AMP for block CS (Metzler et al., ICASSP 2014)

_Deep Unfolding (2018-2022):_
- ISTA-Net — learned ISTA for block CS (Zhang & Ghanem, CVPR 2018) — the foundational deep unfolding CS paper; learned soft-thresholding with CNN proximal
- ISTA-Net+ — enhanced ISTA-Net with improved proximal operator (Zhang & Ghanem, CVPR 2018) — the "++" variant; learned step size and threshold
- OPINE-Net — deep unfolding for block CS with learned sampling (Zhang et al., JSTSP 2020)
- AMP-Net — AMP-inspired network for block CS (Zhang et al., TIP 2021) — learned denoising within AMP iterations
- COAST — controllable arbitrary-sampling-ratio CS (You et al., TIP 2021)
- MADUN — memory-augmented deep unfolding network (Song et al., CVPR 2021)
- CASNet — deep CS adaptive sensing (2021)

_Transformer & Modern (2022-2026):_
- TransCS — Transformer-based block CS (Shen et al., 2022) — self-attention for inter-block correlation exploitation
- CSformer — convolution-free Transformer for block CS (Ye et al., 2023)
- DPC-DUN — deep progressive cross-domain unfolding (2023)
- SCSNet — scalable CS network (Shi et al., 2019; refined 2022)
- FSOINet — feature space optimization inspired network (Chen et al., CVPR 2022)
- Diffusion-CS — diffusion model for block CS recovery (2024)
- Foundation model for compressive sensing (2025)

#### Step 3: Update SPC Block Solvers

After listing all SPC block solvers, update `algorithm_base/spc_block/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SPC block solvers use the data format: `y` (N_blocks, M) block-wise measurements where M = cs_ratio * block_size^2, `Phi` (M, block_size^2) measurement matrix per block (same for all blocks in standard setup), `block_size` int (typically 33). The `SPCBlockOperator` handles the block-diagonal forward model `y_i = Phi * x_i` for each block i and the reassembly of blocks into the full image.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SPC Block:**
- Set11 at 25% CS ratio (33x33 blocks): ISTA-Net ~29.5 dB, ISTA-Net+ ~31.5 dB, AMP-Net ~32.0 dB, TransCS ~33.0 dB, CSformer ~33.5 dB
- Set11 at 10% CS ratio: ISTA-Net ~25.0 dB, ISTA-Net+ ~27.0 dB, TransCS ~29.0 dB
- Set11 at 50% CS ratio: ISTA-Net+ ~35.5 dB, TransCS ~37.0 dB
- BSD68 at 25% CS ratio: ISTA-Net ~28.5 dB, ISTA-Net+ ~30.5 dB, TransCS ~32.0 dB
- All reference values from published ISTA-Net, AMP-Net, TransCS papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spc_block' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spc_block/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spc_block/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spc_block/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SPC block. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spc_block/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spc_block/standard/`

---

### SPC Kronecker (`spc_kronecker`) Modality Template

#### Step 1: Verify Standard Dataset

For Kronecker Product SPC (Single-Pixel Camera with Kronecker-structured measurement matrices), what dataset do you use to verify? Is this dataset used for Kronecker SPC popular algorithms? Please ensure the standard dataset in `datasets/benchmark/spc_kronecker/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original Kronecker SPC standard dataset.

**Popular datasets to consider:**
- **Set11 / BSD68 with Kronecker Measurements** — standard grayscale test images measured with Kronecker product measurement matrices Phi = Phi_row kron Phi_col; same images as block SPC but with structured Kronecker measurement; used for comparing Kronecker CS against block-diagonal and full-matrix approaches
- **ISTA-Net Benchmark with Kronecker Matrices** — same test setup as ISTA-Net but with Kronecker-structured measurement matrices replacing the block-diagonal Gaussian matrix; used for Kronecker deep unfolding evaluation
- **Kronecker CS Benchmark (Duarte & Baraniuk, IEEE TIP 2012)** — the original Kronecker CS paper benchmark; 256x256 images with separable row/column measurement matrices at various compression ratios
- **2D Compressive Imaging Data** — images measured with separable 2D random projections; Phi_2D = Phi_v * X * Phi_h^T; used for separable CS algorithms
- **SPC Hardware Kronecker Data** — single-pixel camera captures using row-column scanning with Kronecker measurement structure

**Decision criteria:** Set11/BSD68 with Kronecker measurement matrices matching Duarte & Baraniuk 2012 setup is the standard for Kronecker CS benchmarking. Use the same images as block SPC for direct comparison of measurement structures.

#### Step 2: List All SPC Kronecker Algorithms

Please first ensure all the SPC Kronecker algorithms have been listed in `\Physics_World_Model\algorithm_base\spc_kronecker\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/spc_kronecker. Besides, you need to search all algorithms from 1950 to 2026. After listing all the SPC Kronecker solvers, please update the SPC Kronecker solver.

**Key algorithms to cover (2008-2026):**

_Classical Kronecker CS (2008-2014):_
- Kronecker CS — compressed sensing with Kronecker product measurement matrices (Duarte & Baraniuk, IEEE TIP 2012) — the foundational paper; exploits separable structure for reduced storage and fast matrix-vector products; Phi = Phi_1 kron Phi_2 where each factor is small
- Separable 2D OMP — row-column OMP exploiting Kronecker structure (2010)
- Kronecker ISTA — ISTA with Kronecker-structured measurement operator (2012)
- Kronecker FISTA — Fast ISTA with Kronecker matrix-vector products (2013)
- Row-column compressed sensing — measuring rows and columns separately (Lim et al., 2011)
- Multi-dimensional CS with Kronecker sparsifying basis (Caiafa & Cichocki, IEEE TSP 2013)
- Kronecker basis pursuit (2012)

_Optimization & Structured (2014-2020):_
- Block Kronecker ISTA — combining block-diagonal and Kronecker structure (2016)
- Kronecker TV — total variation regularized Kronecker CS (2015)
- Kronecker AMP — approximate message passing with Kronecker operators (2017)
- Structured random matrices for Kronecker CS — partial Hadamard, scrambled Fourier (2014)
- Tensor CS with Kronecker measurement — exploiting tensor structure (Sidiropoulos et al., 2017)
- Kronecker ADMM — ADMM with Kronecker operator splitting (2018)
- PnP Kronecker CS — Plug-and-Play with Kronecker forward model (2019)

_Deep Learning (2020-2026):_
- Deep unfolding Kronecker CS — ISTA-Net with Kronecker measurement operator (2020) — learned Kronecker recovery with efficient forward/adjoint
- Kronecker-structured learned measurement — jointly learning Phi_1, Phi_2 as Kronecker factors (2021)
- Deep Kronecker CS with 2D attention (2022)
- Separable measurement network — learning row and column measurement matrices independently (2022)
- Kronecker tensor decomposition with neural networks (2023)
- Efficient Transformer for Kronecker CS (2024)
- Foundation model for structured compressive sensing (2025)

#### Step 3: Update SPC Kronecker Solvers

After listing all SPC Kronecker solvers, update `algorithm_base/spc_kronecker/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All SPC Kronecker solvers use the data format: `y` (M_row * M_col,) or (M_row, M_col) vectorized Kronecker measurements, `Phi_row` (M_row, N_row) row measurement matrix, `Phi_col` (M_col, N_col) column measurement matrix, where the full measurement is Phi = Phi_row kron Phi_col and image size is N_row x N_col. The `SPCKroneckerOperator` handles the Kronecker forward model `y = vec(Phi_row * X * Phi_col^T)` and adjoint `X_adj = Phi_row^T * mat(y) * Phi_col` without forming the full Kronecker product.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for SPC Kronecker:**
- Set11 at 25% Kronecker ratio: Kronecker ISTA ~27.5 dB, Kronecker FISTA ~28.5 dB, Deep unfolding Kronecker ~31.0 dB
- Set11 at 10% Kronecker ratio: Kronecker ISTA ~23.0 dB, Deep unfolding ~27.0 dB
- Comparison with block-diagonal: Kronecker typically 0.5-1.5 dB below block-diagonal at same ratio due to less flexible measurement structure, but with significantly lower storage and computation
- Duarte & Baraniuk 2012 benchmark: Kronecker CS within 1-2 dB of full random matrix CS at 25% ratio
- Published PSNR/SSIM from Kronecker CS papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'spc_kronecker' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/spc_kronecker/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/spc_kronecker/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/spc_kronecker/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for SPC Kronecker. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/spc_kronecker/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/spc_kronecker/standard/`

---

### Streak Camera (`streak_camera`) Modality Template

#### Step 1: Verify Standard Dataset

For Streak Camera imaging, what dataset do you use to verify? Is this dataset used for streak camera popular algorithms? Please ensure the standard dataset in `datasets/benchmark/streak_camera/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original streak camera standard dataset.

**Popular datasets to consider:**
- **Ultrafast Streak Camera Captures (ps-scale)** — streak camera images of ultrafast optical phenomena (fluorescence decay, laser pulse propagation, scintillation) at picosecond temporal resolution; raw streak images with known temporal profiles for algorithm validation
- **Synchroscan Streak Data (Hamamatsu, 2000s-2020s)** — repetitive high-frequency streak camera data from synchroscan mode; used for high-repetition-rate temporal measurements in particle physics and synchrotron experiments
- **FLIM Streak Data (Becker, 2005; Elson et al., 2004)** — Fluorescence Lifetime Imaging Microscopy using streak camera detection; 2D spatial + temporal streak images with known fluorophore lifetimes; standard for FLIM algorithm validation
- **Streak Camera Simulation Benchmark** — Monte Carlo simulated streak images with known instrument response function (IRF) and temporal profiles; used for deconvolution algorithm evaluation
- **Single-Shot Streak Camera Data** — single-shot temporal profile measurements of laser pulses, plasma emission, and ultrafast chemical reactions

**Decision criteria:** FLIM streak data with known fluorophore lifetimes is the most widely used quantitative benchmark. Synchroscan data for high-repetition applications. Use simulated streak data with known ground truth for algorithm PSNR/SSIM comparison.

#### Step 2: List All Streak Camera Algorithms

Please first ensure all the streak camera algorithms have been listed in `\Physics_World_Model\algorithm_base\streak_camera\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/streak_camera. Besides, you need to search all algorithms from 1950 to 2026. After listing all the streak camera solvers, please update the streak camera solver.

**Key algorithms to cover (1970-2026):**

_Classical Temporal Analysis (1970-2009):_
- Temporal profile extraction — direct readout of streak image intensity along time axis (basic streak camera analysis)
- Deconvolution with instrument response function — Richardson-Lucy and Wiener deconvolution for removing IRF broadening (Richardson, 1972; Lucy, 1974; applied to streak 1980s)
- Least-squares exponential fitting — multi-exponential decay fitting for fluorescence lifetimes (Marquardt, 1963)
- Phasor analysis for FLIM — frequency-domain lifetime analysis from streak data (Digman et al., Biophys J 2008; applied to streak)
- Maximum likelihood estimation for photon counting streak (2000)
- Background subtraction and flat-field correction for streak images (1990s)

_Compressive & Optimization (2010-2019):_
- CS streak camera — compressed sensing reconstruction from coded streak measurements (Gao et al., Nature 2014) — single-shot 3D (x,y,t) recovery from 2D streak image via spatial encoding
- TV-regularized streak deconvolution (2015)
- Sparse deconvolution for streak camera — L1-regularized temporal recovery (2014)
- Low-rank streak image recovery — exploiting temporal redundancy (2016)
- Bayesian lifetime estimation from streak data (2012)
- ADMM-based streak reconstruction (2017)
- PnP streak camera — Plug-and-Play denoising for streak (2018)
- Compressed streak imaging with structured illumination (2016)

_Deep Learning (2022-2026):_
- DL streak camera deconvolution — CNN-based temporal deconvolution (2022)
- U-Net streak image denoising (2022)
- Deep learning FLIM from streak data — learned lifetime estimation (2023)
- Physics-informed neural network for streak camera — encoding streak physics (2023)
- End-to-end deep streak reconstruction — joint encoding + reconstruction (2024)
- Transformer-based temporal profile recovery (2024)
- Diffusion-model streak camera — score-based temporal reconstruction (2025)
- Self-supervised streak denoising from paired acquisitions (2024)

#### Step 3: Update Streak Camera Solvers

After listing all streak camera solvers, update `algorithm_base/streak_camera/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All streak camera solvers use the data format: `y` (H_slit, W_time) 2D streak image where the horizontal axis is time and vertical axis is the slit spatial dimension, `irf` (W_time,) instrument response function, `time_axis` (W_time,) calibrated time values in picoseconds, `slit_pos` (H_slit,) spatial positions along the entrance slit. The `StreakCameraOperator` handles the forward model `y(s,t) = IRF(t) * x(s,t) + noise` where * denotes temporal convolution, and adjoint (matched filter) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Streak Camera:**
- FLIM benchmark: lifetime estimation accuracy <5% error for single-exponential (tau = 1-10 ns), <10% for bi-exponential
- Streak deconvolution: temporal resolution recovery from 10 ps IRF to ~2 ps effective resolution
- CS streak (Gao et al., 2014): spatial-temporal datacube recovery ~25 dB PSNR from single streak image
- DL streak deconvolution: 3-5 dB improvement over Richardson-Lucy
- Published lifetime accuracy and temporal resolution from streak camera papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'streak_camera' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/streak_camera/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/streak_camera/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/streak_camera/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for streak camera. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/streak_camera/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/streak_camera/standard/`

---

### Pump-Probe (`pump_probe`) Modality Template

#### Step 1: Verify Standard Dataset

For Pump-Probe Spectroscopy/Imaging, what dataset do you use to verify? Is this dataset used for pump-probe popular algorithms? Please ensure the standard dataset in `datasets/benchmark/pump_probe/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original pump-probe standard dataset.

**Popular datasets to consider:**
- **Transient Absorption Spectroscopy Benchmarks (Berera et al., 2009)** — femtosecond transient absorption (TA) spectra of well-characterized molecular systems (e.g., beta-carotene, chlorophyll, porphyrins); wavelength-time 2D maps with known kinetic models; the standard benchmark for global analysis algorithms
- **Pump-Probe Microscopy Data (Fischer et al., 2016)** — spatially-resolved pump-probe imaging of semiconductor nanostructures, melanin, and hemoglobin; spatial maps at multiple time delays
- **Ultrafast Dynamics Reference Data (Stolow et al., 2004)** — time-resolved photoelectron spectroscopy benchmark; well-characterized molecular dynamics for testing kinetic analysis
- **GloTarAn Test Datasets (Snellenburg et al., 2012)** — Global and Target Analysis software benchmark datasets; simulated and experimental TA data with known species-associated spectra and rate constants; the canonical benchmark for global/target analysis
- **Photosynthesis TA Data (van Grondelle & Novoderezhkin, 2006)** — transient absorption spectra of photosynthetic complexes (LH2, LHCII, PSII); multi-component kinetics with known energy transfer pathways

**Decision criteria:** GloTarAn test datasets with known ground-truth kinetic models are the standard for algorithm validation. Beta-carotene TA for standardized spectral benchmark. Use the dataset most widely referenced in pump-probe analysis papers (2000-2026).

#### Step 2: List All Pump-Probe Algorithms

Please first ensure all the pump-probe algorithms have been listed in `\Physics_World_Model\algorithm_base\pump_probe\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/pump_probe. Besides, you need to search all algorithms from 1950 to 2026. After listing all the pump-probe solvers, please update the pump-probe solver.

**Key algorithms to cover (1980-2026):**

_Classical Analysis (1980-2009):_
- SVD / global analysis — singular value decomposition for extracting principal spectral components and kinetics (van Stokkum et al., BBA 2004) — the standard analysis method for transient absorption data
- Target analysis — fitting to a specific kinetic model with species-associated spectra (van Stokkum et al., BBA 2004) — assumed kinetic model -> extract spectra
- Multi-exponential fitting — fitting transient kinetics at each wavelength to sum of exponentials (Marquardt, 1963)
- Chirp correction — correcting for group velocity dispersion in the probe pulse (Kovalenko et al., 1999)
- Lifetime density analysis — regularized distribution of lifetimes without assumed number of components (2005)
- Singular value decomposition with self-modeling (1990s)

_Model-Based & Optimization (2010-2018):_
- GloTarAn — Global and Target Analysis software framework (Snellenburg et al., J. Stat. Software 2012) — the standard tool for pump-probe data analysis; sequential and parallel kinetic models
- Bayesian target analysis — uncertainty quantification in kinetic fitting (2015)
- Evolutionary fitting for pump-probe — genetic algorithm optimization of kinetic parameters (2012)
- Maximum entropy method for lifetime distributions (2010)
- LASSO-based sparse kinetic analysis — sparsity-promoting spectral decomposition (2016)
- 2D correlation spectroscopy analysis — synchronous/asynchronous correlation (Noda, 2000; applied to pump-probe 2012)
- Tikhonov-regularized multiexponential analysis (2014)

_Deep Learning (2022-2026):_
- DL pump-probe analysis — neural network for automatic kinetic parameter extraction (2022)
- Autoencoder for transient absorption decomposition — unsupervised spectral unmixing (2022)
- Physics-informed neural network for pump-probe — encoding rate equations (2023)
- Convolutional neural network for chirp correction (2023)
- Neural ODE for pump-probe kinetics — continuous-time dynamics modeling (2023)
- Transformer-based spectral-temporal analysis (2024)
- Self-supervised learning for pump-probe denoising (2024)
- Diffusion-model pump-probe — denoising spectral-temporal maps (2025)
- Foundation model for ultrafast spectroscopy (2025)

#### Step 3: Update Pump-Probe Solvers

After listing all pump-probe solvers, update `algorithm_base/pump_probe/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All pump-probe solvers use the data format: `y` (N_wavelengths, N_delays) transient absorption 2D map (Delta-OD as function of probe wavelength and pump-probe time delay), `wavelengths` (N_wavelengths,) probe wavelength axis in nm, `delays` (N_delays,) pump-probe delay axis in ps/fs, `irf_width` float instrument response function width (Gaussian FWHM). The `PumpProbeOperator` handles the forward model `y(lambda, t) = sum_i c_i(t) * s_i(lambda)` where c_i(t) are concentration profiles governed by kinetic rate equations and s_i(lambda) are species-associated spectra, convolved with the IRF.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Pump-Probe:**
- GloTarAn simulated 3-component system: SVD recovers correct number of components, target analysis rate constants within 5% of ground truth
- Beta-carotene TA: S2 lifetime ~150 fs, S1 lifetime ~9 ps; algorithms must recover within 10% of known values
- Photosynthesis LH2: energy transfer time constants matching published kinetics (van Grondelle et al.)
- Chirp correction: residual chirp <10 fs after correction
- DL pump-probe: 2-3 dB improvement in noisy TA data recovery over SVD/global analysis

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'pump_probe' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/pump_probe/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/pump_probe/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/pump_probe/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for pump-probe. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/pump_probe/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/pump_probe/standard/`

---

### Radio Interferometry (`radio_interferometry`) Modality Template

#### Step 1: Verify Standard Dataset

For Radio Interferometry, what dataset do you use to verify? Is this dataset used for radio interferometry popular algorithms? Please ensure the standard dataset in `datasets/benchmark/radio_interferometry/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original radio interferometry standard dataset.

**Popular datasets to consider:**
- **VLA Calibrator Survey Data (NRAO)** — Karl G. Jansky Very Large Array visibility data of calibrator sources with known structure; standard for imaging algorithm development and testing; multi-configuration (A/B/C/D) data available
- **ALMA Science Verification Data (ALMA Observatory, 2012-)** — Atacama Large Millimeter Array calibrated visibility datasets released for community testing; includes continuum and spectral line observations with known source structure; widely used for ALMA imaging benchmarks
- **LOFAR Visibility Data (van Haarlem et al., 2013)** — Low-Frequency Array data with direction-dependent calibration challenges; used for wide-field imaging algorithm evaluation
- **VLBI Fringe-Fitting Benchmarks (EHT Collaboration, 2019)** — Event Horizon Telescope visibility data and synthetic datasets for VLBI imaging; used for sparse uv-coverage imaging algorithm validation; includes M87* and Sgr A* datasets
- **PURIFY/SARA Benchmark Data (Carrillo et al., MNRAS 2014)** — simulated radio interferometric data with known ground truth for compressed sensing radio imaging algorithm evaluation; standard benchmark for compressive radio imaging
- **MeerKAT Data (Jonas, 2016)** — MeerKAT radio telescope visibility data; emerging standard for next-generation radio interferometry

**Decision criteria:** VLA calibrator data is the most widely used for classical imaging benchmarks. ALMA SV data for millimeter-wave imaging. EHT data for sparse VLBI. PURIFY/SARA simulations for compressive imaging algorithm evaluation. Use the dataset most widely referenced in radio interferometric imaging papers (2000-2026).

#### Step 2: List All Radio Interferometry Algorithms

Please first ensure all the radio interferometry algorithms have been listed in `\Physics_World_Model\algorithm_base\radio_interferometry\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/radio_interferometry. Besides, you need to search all algorithms from 1950 to 2026. After listing all the radio interferometry solvers, please update the radio interferometry solver.

**Key algorithms to cover (1960-2026):**

_Classical Radio Imaging (1960-2000):_
- Fringe fitting — determination of interferometric phase, delay, and rate from visibilities (Cotton, 1995; Schwab & Cotton, 1983) — fundamental VLBI calibration step
- Closure quantities — closure phase and closure amplitude for calibration-independent imaging (Jennison, 1958; Readhead et al., 1980)
- CLEAN — iterative deconvolution of dirty beam (Hogbom, 1974) — the foundational radio imaging algorithm; finds point sources iteratively
- Clark CLEAN — major/minor cycle CLEAN for efficient computation (Clark, 1980)
- Cotton-Schwab CLEAN — wide-field CLEAN with w-projection (Cotton, 1999)
- Multi-scale CLEAN — extended source CLEAN with multi-scale components (Cornwell, 2008)
- Hybrid mapping — iterative self-calibration + CLEAN (Readhead & Wilkinson, 1978; Pearson & Readhead, 1984) — standard for VLBI imaging
- MEM — Maximum Entropy Method for radio imaging (Cornwell & Evans, 1985)
- Uniform weighting, natural weighting, Briggs robust weighting (Briggs, 1995)

_Calibration & Wide-Field (2000-2016):_
- AIPS calibration pipeline — Astronomical Image Processing System (NRAO, Greisen 2003) — standard radio data calibration and imaging package
- CASA calibration pipeline — Common Astronomy Software Applications (McMullin et al., 2007) — the modern standard for radio interferometry data processing
- W-projection — correcting non-coplanar baseline effects (Cornwell et al., 2008)
- A-projection — direction-dependent calibration correction during imaging (Bhatnagar et al., 2008)
- AWimager — A/W-projection for wide-field imaging (Tasse et al., 2013)
- DDFacet — direction-dependent faceted imaging (Tasse et al., A&A 2018)
- Self-calibration — iterative model-based gain calibration (Cornwell & Fomalont, 1999)
- Peeling — direction-dependent calibration for individual sources (Noordam, 2004)
- KillMS — direction-dependent calibration (Tasse, 2014)

_Compressive & Sparse (2011-2020):_
- Compressed sensing radio imaging — sparsity-based radio image reconstruction (Wiaux et al., MNRAS 2009)
- SARA — Sparsity Averaging Reweighted Analysis for radio interferometric imaging (Carrillo et al., MNRAS 2012)
- PURIFY — sparse radio interferometric imaging (Carrillo et al., MNRAS 2014)
- ADMM radio imaging — proximal splitting for radio deconvolution (2015)
- Total variation radio imaging (2013)
- Bayesian radio imaging — RESOLVE (Junklewitz et al., 2016)
- uSARA — unconstrained SARA with forward-backward algorithm (Terris et al., MNRAS 2022; roots 2018)

_Deep Learning & Modern (2020-2026):_
- ML RFI excision — machine learning radio frequency interference detection and flagging (Akeret et al., 2017; Vafaei Sadr et al., 2020) — CNN/RNN for automated RFI removal
- Deep learning CLEAN — CNN-based radio image deconvolution (Connor et al., 2022)
- R2D2 — Residual-to-Residual DNN for radio imaging (Terris et al., 2023)
- POLISH — deep learning post-processing for radio images (2021)
- Neural network self-calibration (2022)
- Variational inference for radio imaging (2023)
- Score-based diffusion radio imaging (2024)
- EHT imaging with deep learning — neural network reconstruction for sparse VLBI (2023)
- Foundation model for radio astronomical image reconstruction (2025)
- Physics-informed neural network for visibility modeling (2024)
- GAN-based radio image super-resolution (2022)

#### Step 3: Update Radio Interferometry Solvers

After listing all radio interferometry solvers, update `algorithm_base/radio_interferometry/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All radio interferometry solvers use the data format: `y` (N_baselines, N_channels, N_time) complex visibilities, `uvw` (N_baselines, 3) baseline coordinates in wavelengths, `weights` (N_baselines, N_channels, N_time) visibility weights/flags, `freq` (N_channels,) channel frequencies in Hz. The `RadioInterferometryOperator` handles the forward model (non-uniform FFT from image to visibilities: `V(u,v) = FFT{I(l,m) * primary_beam(l,m)}`) and adjoint (dirty image formation) operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for Radio Interferometry:**
- PURIFY/SARA simulated benchmark: CLEAN ~30 dB (point sources), SARA ~35 dB, uSARA ~37 dB
- VLA calibrator: dynamic range >1000:1 with self-calibration + multi-scale CLEAN
- ALMA SV data: image fidelity >0.95 for well-calibrated continuum observations
- EHT M87* benchmark: image domain fidelity metrics from EHT Collaboration papers (2019)
- ML RFI excision: >95% detection rate with <1% false positive rate on flagged data
- Published image quality metrics (dynamic range, fidelity, PSNR) from radio imaging papers

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3-10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'radio_interferometry' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/radio_interferometry/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for radio interferometry. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/radio_interferometry/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/radio_interferometry/standard/`
