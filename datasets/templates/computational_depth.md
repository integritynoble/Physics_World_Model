
---

## Computational Photography, Depth Imaging & Optics — Modality Templates

---

### Coded Exposure / Flutter Shutter (`coded_exposure`) Modality Template

#### Step 1: Verify Standard Dataset

For Coded Exposure, what dataset do you use to verify? Is this dataset used for coded exposure / flutter shutter popular algorithms? Please ensure the standard dataset in `datasets/benchmark/coded_exposure/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original coded exposure standard dataset.

**Popular datasets to consider:**
- **Raskar Flutter Shutter Dataset (Raskar et al., SIGGRAPH 2006)** — the original flutter shutter capture sequences with ground-truth sharp images; the canonical benchmark for coded exposure deblurring
- **Levin Uniform Blur Dataset (Levin et al., CVPR 2009)** — 32 blur kernels x 4 images; the standard benchmark for blind and non-blind image deconvolution; widely used for evaluating coded exposure reconstruction
- **Kohler Motion Blur Dataset (Kohler et al., ECCV 2012)** — 48 blurred images from 12 real camera motion trajectories on 4 images; used for spatially varying motion deblur evaluation
- **GoPro Deblur Dataset (Nah et al., CVPR 2017)** — 3,214 pairs of blurred/sharp images from high-speed video; the dominant deep learning motion deblur benchmark applicable to coded exposure pipelines
- **HIDE Dataset (Shen et al., ICCV 2019)** — human-aware image deblurring dataset; used to evaluate coded exposure methods in dynamic scenes

**Decision criteria:** Raskar's original dataset is the gold standard for flutter shutter specifically; the GoPro dataset is the most popular for general motion deblur deep learning methods; Levin's dataset for classical deconvolution evaluation. Use the dataset that appears in the largest number of coded exposure reconstruction papers.

#### Step 2: List All Coded Exposure Algorithms

Please first ensure all the coded exposure algorithms have been listed in `\Physics_World_Model\algorithm_base\coded_exposure\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/coded_exposure. Besides, you need to search all algorithms from 1950 to 2026. After listing all the coded exposure solvers, please update the coded exposure solver.

**Key algorithms to cover (1950–2026):**

_Classical / Analytic (1990s–2006):_
- Wiener Deconvolution — frequency-domain linear deconvolution baseline (Wiener, 1949; applied to imaging 1960s onward)
- Richardson-Lucy Deconvolution — iterative MLE deconvolution (Richardson 1972, Lucy 1974)
- Tikhonov-regularized Deconvolution — L2-regularized frequency-domain inversion (1960s–present)
- Flutter Shutter — Coded Exposure Photography (Raskar et al., SIGGRAPH 2006) — the foundational coded exposure paper; broadband temporal code for invertible PSF

_Optimization-Based (2007–2016):_
- Sparse Gradient Deconvolution — total variation and sparsity priors for deblurring (Fergus et al., SIGGRAPH 2006; Chan & Wong, TIP 1998)
- Blind Deconvolution with Sparse Priors (Levin et al., CVPR 2009; Krishnan et al., NIPS 2011)
- Half-Quadratic Splitting Deconvolution (Krishnan & Fergus, NIPS 2009)
- Hyper-Laplacian Priors for Deblurring (Krishnan & Fergus, NIPS 2009)
- Bayesian Blind Deconvolution (Fergus et al., SIGGRAPH 2006)
- Normalized Sparsity for Blind Deconvolution (Krishnan et al., CVPR 2011)
- Optimal Coded Exposure Design via Information Theory (Tendero et al., SIAM J. Imaging Sciences 2013)
- Coded Rolling Shutter for Motion Deblurring (Gu et al., ECCV 2010)
- Adaptive Coded Aperture + Exposure (Holloway et al., ICCP 2012)
- ADMM-based Coded Exposure Reconstruction (Boyd et al., 2011)

_Deep Learning (2017–2026):_
- DeblurGAN — generative adversarial deblurring (Kupyn et al., CVPR 2018)
- DeblurGAN-v2 — improved GAN-based deblurring (Kupyn et al., ICCV 2019)
- SRN-DeblurNet — Scale-Recurrent Network for deblurring (Tao et al., CVPR 2018)
- DMPHN — Deep Multi-Patch Hierarchical Network (Zhang et al., CVPR 2019)
- MPRNet — Multi-Stage Progressive Restoration (Zamir et al., CVPR 2021)
- Restormer — Efficient Transformer for image restoration (Zamir et al., CVPR 2022)
- NAFNet — Nonlinear Activation Free Network for restoration (Chen et al., ECCV 2022)
- Stripformer — Strip Transformer for deblurring (Tsai et al., ECCV 2022)
- FFTformer — Frequency-aware Transformer for deblurring (Kong et al., CVPR 2023)
- Learned Coded Exposure via Differentiable Optics (Martel et al., SIGGRAPH 2020)
- Neural Coded Exposure Optimization (Chang & Wetzstein, CVPR 2024)

#### Step 3: Update Coded Exposure Solvers

After listing all coded exposure solvers, update `algorithm_base/coded_exposure/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All coded exposure solvers use the data format: `y` (H, W, C) coded/blurred sensor image, `code` (T,) binary temporal exposure code sequence, `exposure_time` float total integration time. The `CodedExposureOperator` handles forward `y = sum(code[t] * frame[t])` convolution and adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for coded exposure:**
- Raskar dataset: Wiener ~26.0 dB, Flutter Shutter + Wiener ~31.5 dB, Sparse prior ~33.0 dB
- GoPro test set: SRN-DeblurNet ~30.26 dB, MPRNet ~32.66 dB, Restormer ~32.92 dB, NAFNet ~33.71 dB
- Levin dataset: Richardson-Lucy ~28.0 dB, Hyper-Laplacian ~34.0 dB, Learned Coded ~36.0 dB
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'coded_exposure' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/coded_exposure/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for coded exposure. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/coded_exposure/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/coded_exposure/standard/`

---

### Event Camera / Dynamic Vision Sensor (`event_camera`) Modality Template

#### Step 1: Verify Standard Dataset

For Event Camera, what dataset do you use to verify? Is this dataset used for event camera / dynamic vision sensor popular algorithms? Please ensure the standard dataset in `datasets/benchmark/event_camera/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original event camera standard dataset.

**Popular datasets to consider:**
- **DAVIS Dataset (Mueggler et al., IJRR 2017)** — the canonical event camera benchmark; simultaneous frames + events from a DAVIS sensor; includes camera trajectories with ground truth; used by virtually all event-based vision papers
- **MVSEC — Multi Vehicle Stereo Event Camera Dataset (Zhu et al., RA-L 2018)** — stereo event camera data from driving and indoor scenes with LiDAR ground-truth depth; widely used for event-based depth and optical flow
- **DSEC — Driving Stereo Event Camera Dataset (Gehrig et al., RA-L 2021)** — large-scale stereo event + frame driving dataset with disparity ground truth; the dominant benchmark for event-based stereo since 2021
- **ECD — Event Camera Dataset (Mueggler et al., IJRR 2017)** — sequences with 6-DOF ground truth from motion capture; used for event-based visual odometry evaluation
- **N-Caltech101 / N-MNIST (Orchard et al., 2015)** — neuromorphic versions of Caltech-101 and MNIST captured with a DVS; used for event-based classification benchmarks

**Decision criteria:** DAVIS dataset is the gold standard for general event camera reconstruction and flow benchmarks (2017–2026); DSEC for stereo depth; MVSEC for multi-modal depth estimation. Use the dataset that appears in the largest number of event camera reconstruction papers.

#### Step 2: List All Event Camera Algorithms

Please first ensure all the event camera algorithms have been listed in `\Physics_World_Model\algorithm_base\event_camera\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/event_camera. Besides, you need to search all algorithms from 1950 to 2026. After listing all the event camera solvers, please update the event camera solver.

**Key algorithms to cover (1950–2026):**

_Event Representation & Classical (2008–2016):_
- Event Integration / Accumulation — baseline event-to-frame by summing events (Lichtsteiner et al., JSSC 2008)
- Event Histogram — spatial-temporal histogram of events
- Time Surface / Surface of Active Events (Benosman et al., TNNLS 2014)
- Contrast Maximization Framework (Gallego et al., CVPR 2018; roots in earlier work)
- Event-based Lucas-Kanade optical flow (Benosman et al., Neural Computation 2012)
- Asynchronous Event-based Corner Detection (Vasco et al., IROS 2016)

_Model-Based & Optimization (2014–2020):_
- Event-driven Stereo Matching (Rogister et al., TNNLS 2012)
- Simultaneous Optical Flow and Intensity Estimation (Pan et al., CVPR 2019)
- Event-based Multi-View Stereo (Rebecq et al., IJCV 2018)
- Continuous-Time Trajectory Estimation (Mueggler et al., RSS 2015)
- Contrast Maximization for Motion Estimation (Gallego et al., CVPR 2018)
- Event-based Visual Odometry with IMU (Rebecq et al., BMVC 2017)
- EVO — Event-based Visual Odometry (Rebecq et al., RA-L 2017)
- EMVS — Event-based Multi-View Stereo (Rebecq et al., IJCV 2018)
- ESIM — Event camera simulator (Rebecq et al., CoRL 2018)

_Deep Learning (2018–2026):_
- E2VID — Events-to-Video via recurrent neural network (Rebecq et al., TPAMI 2020) — landmark event-to-intensity reconstruction
- FireNet — lightweight event-to-video (Scheerlinck et al., 2020)
- ERAFT — Event-based Recurrent All-pairs Field Transforms for optical flow (Gehrig et al., 3DV 2021)
- Event-based Learned Stereo (Tulyakov et al., CVPR 2019)
- Spike-FlowNet — spiking neural network for optical flow (Lee et al., ECCV 2020)
- SPADE-E2VID — spatially adaptive denormalization for E2VID (Cadena et al., 2021)
- ET-Net — Event Transformer Network (Weng et al., 2021)
- TimeReplayer — time-aware event representation learning (He et al., CVPR 2022)
- EventNeRF — event-based neural radiance field (Rudnev et al., CVPR 2023)
- HyperE2VID — hypernet-based event-to-video (Ercan et al., CVPR 2024)
- Diffusion-based Event Reconstruction (Wu et al., 2024)
- State Space Model for Event Streams (Zubic et al., ECCV 2024)

#### Step 3: Update Event Camera Solvers

After listing all event camera solvers, update `algorithm_base/event_camera/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All event camera solvers use the data format: `events` (N, 4) array of (x, y, timestamp, polarity), `sensor_size` (H, W) spatial resolution, `time_window` (t_start, t_end) reconstruction interval. The `EventCameraOperator` handles event integration, contrast maximization, and event-to-frame forward/adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for event camera:**
- DAVIS reconstruction: Event Integration ~18.0 dB, E2VID ~21.5 dB PSNR / 0.58 SSIM, FireNet ~20.0 dB, HyperE2VID ~23.0 dB
- MVSEC optical flow: Spike-FlowNet ~0.86 AEE, ERAFT ~0.55 AEE
- DSEC stereo: Event stereo disparity ~1.5 px MAE, ERAFT-stereo ~1.2 px MAE
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 10% for flow/depth metrics)
- `partial` — 3–10 dB shortfall (or 10–30% metric gap)
- `gap` — >10 dB shortfall (or >30% metric gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'event_camera' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/event_camera/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/event_camera/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/event_camera/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for event camera. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/event_camera/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/event_camera/standard/`

---

### High Dynamic Range Imaging (`hdr_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For HDR Imaging, what dataset do you use to verify? Is this dataset used for high dynamic range imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/hdr_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original HDR imaging standard dataset.

**Popular datasets to consider:**
- **Kalantari HDR Dataset (Kalantari & Ramamoorthi, SIGGRAPH 2017)** — 74 training + 15 test scenes of multi-exposure LDR bracket sets with ground-truth HDR; the dominant benchmark for deep multi-exposure HDR merging
- **Prabhakar HDR Dataset (Prabhakar et al., CVPR 2019)** — dynamic scene multi-exposure HDR dataset with moving objects and ghosting artifacts; used for deghosting evaluation
- **Tursun Dynamic HDR Dataset (Tursun et al., CGF 2016)** — multi-exposure sequences with large motion; used for evaluating ghost removal in HDR
- **HDR+ Burst Dataset (Hasinoff et al., SIGGRAPH Asia 2016)** — Google HDR+ burst photography raw data; used for burst HDR and denoising evaluation
- **SI-HDR Dataset (Eilertsen et al., SIGGRAPH 2017)** — single-image HDR reconstruction benchmark with diverse scenes

**Decision criteria:** Kalantari's dataset is the gold standard for multi-exposure HDR merging benchmarks (2017–2026); SI-HDR for single-image inverse tone mapping. Use the dataset that appears in the largest number of HDR reconstruction papers.

#### Step 2: List All HDR Imaging Algorithms

Please first ensure all the HDR imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\hdr_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/hdr_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the HDR imaging solvers, please update the HDR imaging solver.

**Key algorithms to cover (1950–2026):**

_Classical Multi-Exposure Fusion (1990s–2010):_
- Debevec-Malik HDR — Recovering High Dynamic Range Radiance Maps (Debevec & Malik, SIGGRAPH 1997) — the foundational HDR imaging paper
- Robertson HDR — Estimation-Maximization CRF recovery (Robertson et al., 1999)
- Mitsunaga-Nayar Radiometric Self-Calibration (Mitsunaga & Nayar, CVPR 1999)
- Mertens Exposure Fusion — contrast-saturation-exposedness weighted multi-exposure fusion without HDR (Mertens et al., PG 2007)
- Fattal Gradient Domain HDR Compression (Fattal et al., SIGGRAPH 2002)
- Reinhard Global/Local Tone Mapping (Reinhard et al., SIGGRAPH 2002)
- Bilateral Filter Tone Mapping (Durand & Dorsey, SIGGRAPH 2002)

_Ghost Removal & Optimization (2003–2016):_
- Khan Ghost Removal — detecting and removing ghosts in HDR (Khan et al., TIP 2006)
- Sen Robust Patch-Based HDR (Sen et al., SIGGRAPH 2012) — patch-based alignment and reconstruction for dynamic HDR
- Hu Moving Object Detection for HDR (Hu et al., TIP 2013)
- Photomatix weighted average HDR merging (commercial, widely used)
- Rank Minimization for HDR (Oh et al., CVPR 2014)
- Superpixel-based Ghost Detection (Pece & Kautz, EGSR 2010)

_Deep Learning (2017–2026):_
- Kalantari Deep HDR — CNN for multi-exposure HDR merging (Kalantari & Ramamoorthi, SIGGRAPH 2017)
- DeepHDR — end-to-end deep HDR imaging (Wu et al., ECCV 2018)
- AHDRNet — Attention-guided HDR (Yan et al., CVPR 2019)
- HDR-GAN — generative adversarial HDR (Niu et al., AAAI 2021)
- HDRUNET — single-image HDR reconstruction (Chen et al., CVPRW 2021)
- ExpandNet — single-image LDR-to-HDR CNN (Marnerides et al., EGSR 2018)
- SingleHDR — single-image HDR via modulation (Liu et al., CVPR 2020)
- SCTNet — Spatial-Channel Transformer for HDR (2022)
- HDR-Transformer — Transformer for HDR deghosting (Liu et al., AAAI 2022)
- SMAE — Self-supervised Masked Autoencoder for HDR (2023)
- Diffusion-HDR — diffusion model for single-image HDR reconstruction (2024)
- Neural HDR with Implicit Exposure Control (2025)

#### Step 3: Update HDR Imaging Solvers

After listing all HDR imaging solvers, update `algorithm_base/hdr_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All HDR imaging solvers use the data format: `y` list of (H, W, 3) LDR images at different exposures, `exposure_times` list of float exposure durations, `crf` (256,) camera response function. The `HDROperator` handles forward tone mapping `LDR = crf(HDR * exposure_time)` and inverse operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for HDR imaging:**
- Kalantari test set (PSNR-mu / SSIM-mu): Debevec ~38.0 dB, Sen ~40.8 dB, Kalantari Deep HDR ~42.7 dB, AHDRNet ~43.6 dB, HDR-Transformer ~44.2 dB
- SI-HDR single image: ExpandNet ~28.0 dB, SingleHDR ~31.0 dB, Diffusion-HDR ~33.5 dB
- HDR-VDP-2 scores: Mertens fusion ~55, Sen ~62, Deep HDR ~67
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference
- `partial` — 3–10 dB shortfall
- `gap` — >10 dB shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'hdr_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/hdr_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for HDR imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/hdr_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/hdr_imaging/standard/`

---

### Panorama Multi-Focus Fusion (`panorama`) Modality Template

#### Step 1: Verify Standard Dataset

For Panorama Multi-Focus Fusion, what dataset do you use to verify? Is this dataset used for panorama stitching and multi-focus fusion popular algorithms? Please ensure the standard dataset in `datasets/benchmark/panorama/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original panorama standard dataset.

**Popular datasets to consider:**
- **Adobe Panorama Dataset (Lin et al., ECCV 2016)** — 50 challenging panorama sequences with ground-truth homographies; the standard benchmark for image stitching evaluation
- **UDIS-D Dataset (Nie et al., TIP 2021)** — Unsupervised Deep Image Stitching dataset with 10,440 image pairs; the dominant deep stitching benchmark
- **Lytro Multi-Focus Dataset (Nejati et al., SP 2015)** — 20 pairs of multi-focus images with ground-truth all-in-focus images; widely used for focus fusion evaluation
- **MFFW Dataset (Xu et al., Information Fusion 2020)** — Multi-Focus Fusion in the Wild; real-world multi-focus image pairs for fusion evaluation
- **DHW Dataset (Nie et al., AAAI 2020)** — Deep Homography Warping dataset for stitching; large-scale with diverse scenes

**Decision criteria:** Adobe Panorama is the gold standard for classical stitching evaluation; UDIS-D for deep learning stitching; Lytro for multi-focus fusion. Use the dataset that appears in the largest number of panorama/fusion papers.

#### Step 2: List All Panorama Algorithms

Please first ensure all the panorama algorithms have been listed in `\Physics_World_Model\algorithm_base\panorama\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/panorama. Besides, you need to search all algorithms from 1950 to 2026. After listing all the panorama solvers, please update the panorama solver.

**Key algorithms to cover (1950–2026):**

_Classical Stitching & Fusion (1990s–2010):_
- Cylindrical/Spherical Projection Stitching (Szeliski & Shum, SIGGRAPH 1997) — foundational panorama stitching via cylindrical warping
- AutoStitch — Automatic Panoramic Image Stitching (Brown & Lowe, IJCV 2007) — SIFT-based feature matching + RANSAC + multi-band blending
- Laplacian Pyramid Blending (Burt & Adelson, TOG 1983)
- Multi-Band Blending (Brown & Lowe, IJCV 2007)
- Graph-Cut Seam Finding (Kwatra et al., SIGGRAPH 2003)
- APAP — As-Projective-As-Possible warping (Zaragoza et al., CVPR 2013)
- Wavelet-based Multi-Focus Fusion (Li et al., Pattern Recognition 1995)
- Laplacian Pyramid Multi-Focus Fusion (Burt & Adelson, 1983)

_Optimization-Based (2010–2018):_
- SPHP — Shape-Preserving Half-Projective warping (Chang et al., CVPR 2014)
- NIS — Natural Image Stitching with Global Similarity Prior (Chen & Chuang, ECCV 2016)
- AANAP — Adaptive As-Natural-As-Possible warping (Lin et al., CVPR 2015)
- REW — Robust Elastic Warping for panorama (Li et al., TIP 2017)
- GLP — Guided Laplacian Pyramid Multi-Focus Fusion (Wang & Li, SP 2014)
- Dense SIFT Multi-Focus Fusion (Liu et al., Information Fusion 2015)
- CNN-based Focus Measure (Pertuz et al., Pattern Recognition 2013)
- Gradient-domain Stitching (Levin et al., ECCV 2004)

_Deep Learning (2018–2026):_
- Deep Homography Estimation (DeTone et al., 2016; updated Nguyen et al., 2018)
- UDIS — Unsupervised Deep Image Stitching (Nie et al., TIP 2021)
- UDIS++ — improved unsupervised stitching with composition (Nie et al., TPAMI 2023)
- Deep Multi-Focus Fusion with CNN (Liu et al., Information Fusion 2017)
- SESF-Fuse — Self-supervised multi-focus fusion (Ma et al., 2020)
- U2Fusion — unified unsupervised image fusion (Xu et al., TPAMI 2022)
- TransFuse — Transformer for multi-focus fusion (2022)
- Parallax-Tolerant Image Stitching via Learning (Dai et al., CVPR 2022)
- RecRecNet — recurrent rectangular stitching (Nie et al., ICCV 2023)

#### Step 3: Update Panorama Solvers

After listing all panorama solvers, update `algorithm_base/panorama/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All panorama solvers use the data format: `y` list of (H, W, 3) input images from different viewpoints or focal planes, `homographies` list of (3, 3) inter-image homography matrices, `focus_maps` list of (H, W) focus quality maps. The `PanoramaOperator` handles warping, blending mask generation, and composition forward/adjoint operations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for panorama:**
- Adobe Panorama stitching: RMSE alignment error — AutoStitch ~2.5 px, APAP ~1.8 px, NIS ~1.5 px, UDIS ~1.2 px, UDIS++ ~1.0 px
- UDIS-D test set: PSNR — Deep Homography ~25.0 dB, UDIS ~30.5 dB, UDIS++ ~32.0 dB
- Lytro multi-focus fusion: PSNR — Laplacian Pyramid ~33.0 dB, Dense SIFT ~35.5 dB, CNN Fusion ~37.0 dB, U2Fusion ~38.5 dB
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 3 dB of reference (or within 10% for alignment metrics)
- `partial` — 3–10 dB shortfall (or 10–30% metric gap)
- `gap` — >10 dB shortfall (or >30% metric gap)
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'panorama' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/panorama/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/panorama/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/panorama/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for panorama. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/panorama/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/panorama/standard/`

---

### Flash LiDAR (`flash_lidar`) Modality Template

#### Step 1: Verify Standard Dataset

For Flash LiDAR, what dataset do you use to verify? Is this dataset used for flash LiDAR popular algorithms? Please ensure the standard dataset in `datasets/benchmark/flash_lidar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original flash LiDAR standard dataset.

**Popular datasets to consider:**
- **Middlebury Stereo / Depth v3 (Scharstein et al., GCPR 2014)** — the canonical structured-light depth benchmark with sub-pixel ground truth; widely used for depth completion and enhancement evaluation applicable to flash LiDAR
- **NYU Depth V2 (Silberman et al., ECCV 2012)** — 1,449 aligned RGB-D pairs from Kinect; used as a standard benchmark for depth enhancement and completion; applicable to flash LiDAR upsampling
- **KITTI Depth Completion (Uhrig et al., 3DV 2017)** — sparse LiDAR depth maps with semi-dense ground truth from outdoor driving; the dominant benchmark for depth completion from sparse measurements
- **CSIC Single-Photon LiDAR Dataset (Shin et al., Nature Communications 2016)** — single-photon avalanche diode flash LiDAR data at extreme photon counts; used for SPAD-based reconstruction evaluation
- **Lindell SPAD Dataset (Lindell et al., SIGGRAPH 2018)** — single-photon 3D imaging data with ground truth; used for flash LiDAR denoising and depth reconstruction

**Decision criteria:** KITTI Depth Completion is the dominant benchmark for sparse-to-dense depth completion; NYU Depth V2 for indoor flash LiDAR scenarios; Lindell SPAD for single-photon flash LiDAR. Use the dataset that appears in the largest number of flash LiDAR reconstruction papers.

#### Step 2: List All Flash LiDAR Algorithms

Please first ensure all the flash LiDAR algorithms have been listed in `\Physics_World_Model\algorithm_base\flash_lidar\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/flash_lidar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the flash LiDAR solvers, please update the flash LiDAR solver.

**Key algorithms to cover (1950–2026):**

_Classical / Model-Based (1990s–2012):_
- Matched Filter Detection — correlate return signal with emitted pulse template (classical radar/LiDAR, 1950s onward)
- Histogram Peak Detection — bin photon arrival times and find peaks (basic SPAD processing)
- Cross-Correlation Range Estimation — cross-correlate transmitted and received waveforms (standard LiDAR)
- Bilateral Filter Depth Denoising (Tomasi & Manduchi, ICCV 1998; applied to depth)
- Joint Bilateral Upsampling — guided depth upsampling using RGB (Kopf et al., SIGGRAPH 2007)
- Markov Random Field Depth Refinement (Diebel & Thrun, NIPS 2005)

_Optimization / Compressed Sensing (2012–2018):_
- Poisson Denoising for Photon-Limited Depth (Shin et al., Nature Communications 2016)
- Convex Optimization for SPAD Depth Recovery (Rapp & Goyal, IEEE TSP 2017)
- Sparse Signal Recovery for Flash LiDAR (Kirmani et al., Science 2014) — first-photon depth imaging
- Total Variation Depth Regularization (Ferstl et al., ICCV 2013)
- Guided Depth Super-Resolution (Ferstl et al., ICCV 2013)
- Non-Local Means Depth Denoising (Park et al., ECCV 2014)
- Multi-Scale Depth Completion (Ku et al., CRV 2018)
- Compressive Depth Acquisition with SPAD (Howland et al., Applied Physics Letters 2013)

_Deep Learning (2018–2026):_
- CSPN — Convolutional Spatial Propagation Network for depth completion (Cheng et al., ECCV 2018)
- DeepLiDAR — Deep Surface Normal Guided Depth Prediction (Qiu et al., CVPR 2019)
- S2D — Sparse-to-Dense depth prediction (Ma & Karaman, ICRA 2018)
- GuideNet — guided depth completion (Tang et al., TIP 2020)
- NLSPN — Non-Local Spatial Propagation Network (Park et al., ECCV 2020)
- PENet — Precise and Efficient Depth Completion (Hu et al., ICRA 2021)
- CFormer — Transformer for depth completion (Zhang et al., 2023)
- CompletionFormer — depth completion with Transformer (Zhang et al., CVPR 2023)
- SPADNet — Deep SPAD LiDAR Reconstruction (Lindell et al., SIGGRAPH 2018)
- Deep Single-Photon 3D Imaging (Peng et al., Nature Photonics 2020)
- Diffusion-based Depth Completion (Saxena et al., 2024)

#### Step 3: Update Flash LiDAR Solvers

After listing all flash LiDAR solvers, update `algorithm_base/flash_lidar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All flash LiDAR solvers use the data format: `y` (H, W, T) photon-count histogram or (H, W) sparse depth map, `rgb_guide` (H, W, 3) optional co-registered RGB image, `pulse_template` (T,) emitted pulse shape. The `FlashLiDAROperator` handles forward Poisson observation model and depth-to-histogram mapping.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for flash LiDAR:**
- KITTI depth completion: S2D ~1.29 m RMSE, CSPN ~1.02 m, NLSPN ~0.092 m iRMSE, PENet ~0.073 m iRMSE, CompletionFormer ~0.068 m iRMSE
- NYU Depth V2 upsampling (x8): Bilateral ~5.2 cm RMSE, CSPN ~2.0 cm, NLSPN ~1.5 cm
- Lindell SPAD: Matched Filter ~15.0 cm RMSE, SPADNet ~2.5 cm, Deep Single-Photon ~1.0 cm
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'flash_lidar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/flash_lidar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/flash_lidar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/flash_lidar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for flash LiDAR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/flash_lidar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/flash_lidar/standard/`

---

### LiDAR Scanner (`lidar`) Modality Template

#### Step 1: Verify Standard Dataset

For LiDAR Scanner, what dataset do you use to verify? Is this dataset used for LiDAR scanner popular algorithms? Please ensure the standard dataset in `datasets/benchmark/lidar/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original LiDAR standard dataset.

**Popular datasets to consider:**
- **KITTI 3D Object Detection / LiDAR (Geiger et al., CVPR 2012)** — the canonical outdoor LiDAR benchmark; Velodyne 64-beam point clouds with 3D bounding box ground truth; used by virtually all LiDAR perception papers
- **nuScenes (Caesar et al., CVPR 2020)** — large-scale autonomous driving dataset with 32-beam LiDAR; 1,000 scenes with full 3D annotations; the dominant modern LiDAR benchmark
- **Waymo Open Dataset (Sun et al., CVPR 2020)** — high-resolution 64-beam LiDAR with dense annotations; largest LiDAR driving dataset
- **SemanticKITTI (Behley et al., ICCV 2019)** — point-wise semantic labels on KITTI LiDAR; the standard benchmark for LiDAR semantic segmentation
- **ScanNet (Dai et al., CVPR 2017)** — indoor RGB-D/LiDAR reconstructions with semantic annotations; used for indoor 3D scene understanding

**Decision criteria:** KITTI is the undisputed gold standard for outdoor LiDAR 3D detection (2012–2026); nuScenes for modern multi-frame LiDAR; SemanticKITTI for semantic segmentation. Use the dataset that appears in the largest number of LiDAR processing papers.

#### Step 2: List All LiDAR Algorithms

Please first ensure all the LiDAR algorithms have been listed in `\Physics_World_Model\algorithm_base\lidar\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/lidar. Besides, you need to search all algorithms from 1950 to 2026. After listing all the LiDAR solvers, please update the LiDAR solver.

**Key algorithms to cover (1950–2026):**

_Classical Point Cloud Processing (1990s–2014):_
- ICP — Iterative Closest Point for point cloud registration (Besl & McKay, TPAMI 1992)
- Normal Estimation via PCA — local surface normal estimation from point neighborhoods
- Voxel Grid Downsampling — spatial hashing for point cloud simplification
- RANSAC Plane Fitting — robust planar surface extraction (Fischler & Bolles, CACM 1981)
- Octree-based Spatial Indexing for LiDAR (Meagher, Computer Graphics 1982)
- Ground Segmentation via height thresholding and morphological filtering

_3D Detection & Segmentation (2015–2019):_
- VoxelNet — End-to-End Learning for 3D point cloud detection (Zhou & Tuzel, CVPR 2018)
- PointNet — Deep Learning on Point Sets (Qi et al., CVPR 2017)
- PointNet++ — Deep Hierarchical Feature Learning on Point Sets (Qi et al., NIPS 2017)
- SECOND — Sparsely Embedded Convolutional Detection (Yan et al., Sensors 2018)
- PointPillars — Fast Encoders for 3D detection (Lang et al., CVPR 2019)
- PointRCNN — 3D Object Proposal Generation and Detection (Shi et al., CVPR 2019)
- RangeNet++ — LiDAR semantic segmentation (Milioto et al., IROS 2019)
- SqueezeSeg — convolutional neural net for LiDAR segmentation (Wu et al., ICRA 2018)
- KPConv — Kernel Point Convolution (Thomas et al., ICCV 2019)

_Modern Deep Learning (2020–2026):_
- CenterPoint — Center-based 3D Object Detection (Yin et al., CVPR 2021) — dominant detection method
- PV-RCNN — Point-Voxel Region-based CNN (Shi et al., CVPR 2020)
- Cylinder3D — cylindrical 3D LiDAR segmentation (Zhu et al., CVPR 2021)
- TransFusion — LiDAR-Camera Transformer (Bai et al., CVPR 2022)
- VoxFormer — Sparse Voxel Transformer for 3D occupancy (Li et al., CVPR 2023)
- LiDARFormer — Transformer for LiDAR 3D detection (2023)
- UniTR — Unified multi-modal Transformer (Wang et al., ICCV 2023)
- LargeKernel3D — large kernel 3D backbone (Chen et al., CVPR 2023)
- FlatFormer — flattened window Transformer for LiDAR (Liu et al., CVPR 2023)
- Foundation model for LiDAR perception (2025–2026)

#### Step 3: Update LiDAR Solvers

After listing all LiDAR solvers, update `algorithm_base/lidar/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All LiDAR solvers use the data format: `y` (N, 4+) point cloud array of (x, y, z, intensity, ...), `voxel_size` (3,) voxelization resolution, `range_image` (H, W, C) range-view projection. The `LiDAROperator` handles voxelization, range projection, and point-to-voxel / voxel-to-point transformations.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for LiDAR:**
- KITTI 3D detection (car, moderate): PointPillars ~77.3% AP, SECOND ~83.1%, PointRCNN ~85.9%, PV-RCNN ~84.8%, CenterPoint ~85.2%
- nuScenes 3D detection: PointPillars ~45.3 NDS, CenterPoint ~67.3 NDS, TransFusion ~71.7 NDS, UniTR ~73.1 NDS
- SemanticKITTI segmentation: RangeNet++ ~52.2 mIoU, Cylinder3D ~68.9 mIoU, LiDARFormer ~72.0 mIoU
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 2% AP/mIoU of reference
- `partial` — 2–5% shortfall
- `gap` — >5% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'lidar' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/lidar/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/lidar/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/lidar/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for LiDAR. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/lidar/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/lidar/standard/`

---

### Photometric Stereo (`photometric_stereo`) Modality Template

#### Step 1: Verify Standard Dataset

For Photometric Stereo, what dataset do you use to verify? Is this dataset used for photometric stereo popular algorithms? Please ensure the standard dataset in `datasets/benchmark/photometric_stereo/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original photometric stereo standard dataset.

**Popular datasets to consider:**
- **DiLiGenT (Shi et al., TPAMI 2019)** — Directional Lighting for Generic; 10 objects with calibrated lighting, 96 images each, ground-truth normals from laser scanner; the gold standard benchmark for photometric stereo
- **DiLiGenT-10^2 (Ren et al., CVPR 2022)** — extended DiLiGenT with 100 lighting directions per object; larger-scale PS benchmark
- **Gourd&Apple Dataset (Alldrin et al., CVPR 2008)** — real objects under varied illumination for uncalibrated PS
- **Light Stage Data Gallery (Debevec et al., USC ICT)** — high-quality relighting and reflectance data; used for evaluating PS under complex BRDF
- **Harvard Photometric Stereo Dataset (Hertzmann & Seitz, CVPR 2005)** — multi-view photometric stereo captures with ground truth

**Decision criteria:** DiLiGenT is the undisputed gold standard for calibrated photometric stereo benchmarking (2019–2026); DiLiGenT-10^2 for large-scale evaluation. Use the dataset that appears in the largest number of photometric stereo papers.

#### Step 2: List All Photometric Stereo Algorithms

Please first ensure all the photometric stereo algorithms have been listed in `\Physics_World_Model\algorithm_base\photometric_stereo\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/photometric_stereo. Besides, you need to search all algorithms from 1950 to 2026. After listing all the photometric stereo solvers, please update the photometric stereo solver.

**Key algorithms to cover (1950–2026):**

_Classical Lambertian (1980–2005):_
- Woodham's Photometric Stereo — least-squares normal estimation under Lambertian assumption (Woodham, Optical Engineering 1980) — the foundational photometric stereo paper
- Robust Least Squares PS — outlier-robust normal estimation (Coleman & Jain, 1982)
- Rank-3 Factorization for Uncalibrated PS (Hayakawa, JOSA A 1994)
- Bas-Relief Ambiguity in Uncalibrated PS (Belhumeur et al., IJCV 1999)
- Lambertian PS with Shadows — shadow detection and exclusion (Barsky & Petrou, BMVC 2003)

_Non-Lambertian & Robust (2005–2016):_
- Robust PCA Photometric Stereo — separate Lambertian and specular components (Wu et al., ICCV 2011)
- Sparse Bayesian Regression PS (Ikehata et al., CVPR 2012)
- Example-Based PS for Non-Lambertian Surfaces (Hertzmann & Seitz, CVPR 2005)
- Isotropy-based PS for General Reflectance (Shi et al., CVPR 2012)
- Bivariate BRDF PS (Shi et al., ECCV 2014)
- Matrix Rank Minimization PS (Wu & Tan, TPAMI 2013)
- Near-Light Photometric Stereo (Queau et al., SSVM 2017)
- Multi-View Photometric Stereo (Hernandez et al., TPAMI 2008)

_Deep Learning (2017–2026):_
- DPSN — Deep Photometric Stereo Network (Santo et al., CVPRW 2017) — first deep PS network
- PS-FCN — Photometric Stereo Fully Convolutional Network (Chen et al., ECCV 2018)
- CNN-PS — CNN for per-pixel PS (Ikehata, ECCV 2018)
- IRPS — Interreflection-aware PS (Taniai & Maehara, CVPR 2018)
- SDPS-Net — Self-calibrating Deep PS Network (Chen et al., CVPR 2019)
- GPS-Net — Graph-based PS Network (Yao et al., ICCV 2020)
- PS-Transformer — Attention-based PS (Ikehata, BMVC 2021)
- UniPS — Universal Photometric Stereo (Ikehata, CVPR 2023) — state of the art; handles arbitrary materials
- SDM-UniPS — scalable diffusion model for universal PS (Ikehata, CVPR 2024)
- NeIF-PS — Neural Incident Field for PS (Wang et al., 2023)
- GR-PSN — Graph Reasoning PS Network (2023)

#### Step 3: Update Photometric Stereo Solvers

After listing all photometric stereo solvers, update `algorithm_base/photometric_stereo/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All photometric stereo solvers use the data format: `y` (K, H, W, C) images under K different lighting directions, `light_dirs` (K, 3) calibrated light direction vectors, `light_intensities` (K, 3) light RGB intensities. The `PhotometricStereoOperator` handles forward rendering `I_k = albedo * max(0, n . l_k)` under Lambertian or general BRDF models.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for photometric stereo:**
- DiLiGenT (mean angular error in degrees): Woodham LS ~17.0 deg, Robust PCA ~10.5 deg, Sparse Bayesian ~9.0 deg, CNN-PS ~8.4 deg, PS-FCN ~7.5 deg, SDPS-Net ~7.0 deg, UniPS ~5.5 deg, SDM-UniPS ~5.0 deg
- DiLiGenT challenging objects (BALL, READING): Woodham ~25 deg, UniPS ~7.0 deg
- All reference values from published papers and DiLiGenT leaderboard

**Verification criteria:**
- `done` — PWM within 1.0 deg MAE of reference
- `partial` — 1.0–3.0 deg shortfall
- `gap` — >3.0 deg shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'photometric_stereo' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/photometric_stereo/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for photometric stereo. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/photometric_stereo/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/photometric_stereo/standard/`

---

### Structured-Light Depth Camera (`structured_light`) Modality Template

#### Step 1: Verify Standard Dataset

For Structured-Light Depth Camera, what dataset do you use to verify? Is this dataset used for structured-light depth popular algorithms? Please ensure the standard dataset in `datasets/benchmark/structured_light/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original structured-light standard dataset.

**Popular datasets to consider:**
- **Middlebury Stereo v3 (Scharstein et al., GCPR 2014)** — sub-pixel accurate ground-truth disparity from structured light; the canonical benchmark for stereo and structured-light depth evaluation
- **Middlebury 2005/2006 Structured Light Dataset (Scharstein & Szeliski, 2003–2006)** — structured-light ground truth for stereo evaluation; original SL benchmark
- **Guo Structured Light Dataset (Guo et al., Optics Express 2004)** — Phase-shifting + Gray code patterns with ground-truth depth; used for evaluating fringe analysis methods
- **RREAL Dataset (Fanello et al., CVPR 2017)** — real structured-light depth from Intel RealSense; used for depth enhancement
- **ICL-NUIM Dataset (Handa et al., ICRA 2014)** — synthetic RGB-D data with perfect ground truth; used for evaluating depth camera SLAM and reconstruction

**Decision criteria:** Middlebury Stereo v3 is the gold standard for structured-light depth evaluation; Guo's dataset for phase-shifting fringe analysis. Use the dataset that appears in the largest number of structured-light reconstruction papers.

#### Step 2: List All Structured-Light Algorithms

Please first ensure all the structured-light algorithms have been listed in `\Physics_World_Model\algorithm_base\structured_light\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/structured_light. Besides, you need to search all algorithms from 1950 to 2026. After listing all the structured-light solvers, please update the structured-light solver.

**Key algorithms to cover (1950–2026):**

_Classical Pattern Coding (1970s–2005):_
- Gray Code Structured Light — binary spatial coding for correspondence (Inokuchi et al., 1984) — foundational temporal coding method
- Phase-Shifting Profilometry — sinusoidal fringe projection with N-step phase shifts (Srinivasan et al., Applied Optics 1984)
- Phase Unwrapping — temporal and spatial methods for resolving 2-pi ambiguity (Huntley & Saldner, 1993)
- Multi-Frequency Phase Unwrapping — hierarchical frequency approach (Gushov & Solodkin, 1991)
- De Bruijn Sequence Patterns — single-shot color stripe coding (Zhang et al., 2002)
- Micro Phase Shifting (Gupta et al., ICCV 2011)

_Optimization & Robust Methods (2005–2017):_
- Fourier Transform Profilometry — single-shot depth from carrier-frequency fringe (Takeda & Mutoh, Applied Optics 1983)
- Windowed Fourier Transform for fringe analysis (Kemao, Applied Optics 2004)
- Phase-Measuring Deflectometry (Knauer et al., 2004)
- Speckle Pattern Structured Light — random dot projection (Konolige, IROS 2010; Kinect v1)
- Ensemble of Multi-Pattern SL (Gupta et al., TPAMI 2013)
- Robust Structured Light via Mutual Information (Kim et al., TPAMI 2014)
- Stereo-Phase Unwrapping — combining stereo geometry with phase (Weise et al., TPAMI 2007)
- High-Speed Structured Light (Gong & Zhang, Optics Express 2010)
- Depth-Discontinuity-Preserving Phase Unwrapping (Zuo et al., Optics Express 2016)

_Deep Learning (2018–2026):_
- Deep Phase Unwrapping (Spoorthi et al., ICASSP 2018)
- Learned Single-Shot Structured Light (Riegler et al., 3DV 2019)
- ActiveStereoNet — end-to-end self-supervised active stereo (Zhang et al., ECCV 2018)
- Deep Fringe Analysis Network (Feng et al., Advanced Photonics 2019)
- PhaseNet — deep phase retrieval from fringe patterns (2020)
- Neural Structured Light — differentiable pattern optimization (Baek et al., CVPR 2021)
- Self-Supervised Depth from Structured Light (Fan et al., 2022)
- Transformer-based Fringe Analysis (2023)
- SL-NeRF — structured light with neural radiance fields (2024)
- Foundation model for structured-light depth (2025)

#### Step 3: Update Structured-Light Solvers

After listing all structured-light solvers, update `algorithm_base/structured_light/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All structured-light solvers use the data format: `y` (K, H, W) captured images under K projected patterns, `patterns` (K, H_p, W_p) projected structured-light patterns, `calibration` dict with projector-camera intrinsics and extrinsics. The `StructuredLightOperator` handles forward projection-capture model and phase/correspondence recovery.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for structured-light:**
- Middlebury v3 (bad-2.0 % error): Gray Code ~8.0%, Phase Shifting (4-step) ~3.5%, Multi-Freq Unwrapping ~2.0%, ActiveStereoNet ~4.5%, Neural SL ~1.8%
- Middlebury v3 (RMSE in pixels): Phase Shifting ~0.8 px, Micro Phase Shifting ~0.3 px, Learned SL ~0.25 px
- Guo phase dataset: Fourier Transform ~1.5 mm, Windowed Fourier ~0.8 mm, Deep Phase ~0.4 mm
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'structured_light' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/structured_light/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/structured_light/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/structured_light/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for structured-light. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/structured_light/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/structured_light/standard/`

---

### Time-of-Flight Depth Camera (`tof_camera`) Modality Template

#### Step 1: Verify Standard Dataset

For Time-of-Flight Depth Camera, what dataset do you use to verify? Is this dataset used for ToF depth camera popular algorithms? Please ensure the standard dataset in `datasets/benchmark/tof_camera/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original ToF camera standard dataset.

**Popular datasets to consider:**
- **FLAT Dataset (Guo et al., ECCV 2018)** — synthetic multi-frequency ToF data with ground-truth depth and multi-path interference labels; the standard benchmark for ToF depth correction
- **Agresti ToF-Stereo Dataset (Agresti et al., Sensors 2017)** — real ToF + stereo data with ground-truth depth; used for ToF depth enhancement evaluation
- **REAL3 Extended Dataset (Hansard et al., TPAMI 2012)** — real ToF camera data from PMD sensors with ground truth; used for multi-path and denoising evaluation
- **Cornell ToF Dataset (Freedman et al., CVPR 2014)** — multi-frequency continuous-wave ToF data with multi-path ground truth
- **ScanNet (Dai et al., CVPR 2017)** — indoor scenes captured with depth sensors including ToF; used for depth completion and enhancement

**Decision criteria:** FLAT dataset is the gold standard for ToF depth correction and multi-path interference removal (2018–2026); Agresti for real-data ToF evaluation. Use the dataset that appears in the largest number of ToF correction papers.

#### Step 2: List All ToF Camera Algorithms

Please first ensure all the ToF camera algorithms have been listed in `\Physics_World_Model\algorithm_base\tof_camera\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/tof_camera. Besides, you need to search all algorithms from 1950 to 2026. After listing all the ToF camera solvers, please update the ToF camera solver.

**Key algorithms to cover (1950–2026):**

_Classical Continuous-Wave ToF (2000–2012):_
- Phase-Based Depth Estimation — standard CW-ToF phase-to-depth conversion (Lange & Seitz, 2001) — foundational ToF principle
- Four-Phase / Four-Bucket Demodulation — standard homodyne demodulation (Spirig et al., 1995)
- Multi-Frequency Unwrapping for ToF — resolving depth ambiguity via dual/multi-frequency (Jongenelen et al., 2011)
- Wiggling Error Correction — systematic phase nonlinearity correction (Lindner et al., 2010)
- Fixed Pattern Noise Calibration for ToF (Kahlmann et al., 2006)

_Multi-Path & Scattering Correction (2010–2018):_
- Multi-Path Interference Separation — AMCW ToF multi-path decomposition (Dorrington et al., 2011)
- Sparse Coding Multi-Path Correction (Freedman et al., CVPR 2014)
- Closed-Form Multi-Path Resolution for Dual-Frequency ToF (Godbaz et al., 2012)
- Epipolar ToF Imaging — multi-path separation via epipolar geometry (O'Toole et al., SIGGRAPH 2015)
- Transient Rendering for Multi-Path Simulation (Jarabo et al., SIGGRAPH 2014)
- Bilateral ToF Denoising — joint spatial-range filtering (Hahne et al., 2013)
- Guided ToF Upsampling — RGB-guided ToF super-resolution (Ferstl et al., ICCV 2013)
- KinectFusion for Depth Integration (Newcombe et al., ISMAR 2011)

_Deep Learning (2018–2026):_
- DeepToF — deep multi-path correction (Marco et al., ECCV 2017)
- Deep End-to-End ToF Imaging (Su et al., CVPR 2018)
- FLAT-Net — multi-frequency ToF correction network (Guo et al., ECCV 2018)
- Agresti Deep ToF Denoising (Agresti & Zanuttigh, CVPRW 2019)
- ToF-KPN — Kernel Prediction Network for ToF denoising (2020)
- Multi-Path Neural Network (Son et al., 2021)
- Learned ToF Multi-Frequency Fusion (Qiu et al., 2022)
- Transformer-based ToF Depth Correction (2023)
- Neural Transient Rendering for ToF (2024)
- Self-Supervised ToF Multi-Path Correction (2025)

#### Step 3: Update ToF Camera Solvers

After listing all ToF camera solvers, update `algorithm_base/tof_camera/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All ToF camera solvers use the data format: `y` (K, H, W) raw correlation measurements at K phase offsets or frequencies, `frequencies` (K,) modulation frequencies in MHz, `integration_time` float sensor integration time. The `ToFOperator` handles forward CW-ToF correlation model `C_k = a * cos(2*pi*f*2d/c + phi_k) + b` and phase-to-depth conversion.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for ToF camera:**
- FLAT dataset (depth RMSE in mm): Four-Phase ~45.0 mm, Multi-Freq ~25.0 mm, DeepToF ~12.0 mm, FLAT-Net ~8.0 mm, Learned Fusion ~5.5 mm
- FLAT multi-path scenes: Phase-based ~80.0 mm, Sparse Coding ~30.0 mm, FLAT-Net ~10.0 mm
- Agresti real data: Bilateral ~15.0 mm, Deep ToF ~8.0 mm, ToF-KPN ~5.0 mm
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'tof_camera' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/tof_camera/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/tof_camera/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/tof_camera/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for ToF camera. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/tof_camera/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/tof_camera/standard/`

---

### Integral Photography (`integral`) Modality Template

#### Step 1: Verify Standard Dataset

For Integral Photography, what dataset do you use to verify? Is this dataset used for integral photography / integral imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/integral/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original integral photography standard dataset.

**Popular datasets to consider:**
- **Stanford Lenslet Light Field Archive (Levoy & Hanrahan, SIGGRAPH 1996)** — the original light field / integral imaging dataset; used for depth estimation and view synthesis from microlens arrays
- **INRIA Lenslet Dataset (Dansereau et al., 2013)** — raw lenslet camera (Lytro) data with calibration; used for integral imaging reconstruction
- **Heidelberg 4D Light Field Benchmark (Honauer et al., ACCV 2016)** — synthetic and real light field data with ground-truth depth and disparity; used for integral imaging depth evaluation
- **EPFL Light Field Dataset (Rerabek & Ebrahimi, 2016)** — Lytro Illum captures for quality evaluation of light field / integral imaging
- **HCI 4D Light Field Dataset (Wanner et al., VMV 2013)** — synthetic light fields with ground truth; used for disparity and depth estimation benchmarks

**Decision criteria:** Heidelberg 4D is the gold standard for quantitative integral imaging / light field depth evaluation; Stanford archive for historical significance; INRIA for real lenslet data. Use the dataset that appears in the largest number of integral photography reconstruction papers.

#### Step 2: List All Integral Photography Algorithms

Please first ensure all the integral photography algorithms have been listed in `\Physics_World_Model\algorithm_base\integral\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/integral. Besides, you need to search all algorithms from 1950 to 2026. After listing all the integral photography solvers, please update the integral photography solver.

**Key algorithms to cover (1950–2026):**

_Classical Integral Imaging (1908–2005):_
- Lippmann Integral Photography — original concept (Lippmann, 1908; computational revival Okoshi 1976)
- Computational Refocusing — synthetic aperture refocusing from microlens array (Ng et al., Stanford Tech Report 2005)
- CIIR — Computational Integral Imaging Reconstruction (Jang & Javidi, Optics Letters 2002)
- Fresnel Propagation Reconstruction — wave-optics based reconstruction for integral imaging (Frauel et al., 2006)
- Elemental Image Array Rendering — ray-based reconstruction from elemental images (Arai et al., Applied Optics 2006)

_Model-Based & Optimization (2006–2017):_
- Depth Estimation from Micro-Lens Defocus (Bishop & Favaro, TPAMI 2012)
- LFBP — Light Field Back-Projection for 3D reconstruction (Lim et al., Optics Express 2009)
- Bayesian Depth Estimation for Integral Imaging (Wanner & Goldluecke, GCPR 2012)
- TV-Regularized Depth from Integral Images (Goldluecke & Wanner, ECCV 2012)
- Multi-View Stereo from Integral Images (Kim et al., 3DTV 2008)
- Super-Resolution Integral Imaging (Park et al., Optics Express 2003)
- Compressive Light Field Integral Imaging (Marwah et al., SIGGRAPH 2013)
- Occlusion-Aware Depth Estimation (Wang et al., CVPR 2015)

_Deep Learning (2018–2026):_
- EPI-based CNN for Light Field Depth — Epipolar Plane Image analysis (Shin et al., CVPR 2018)
- LFattNet — Attention-based Light Field depth estimation (Tsai et al., TIP 2020)
- OACC-Net — Occlusion-Aware Cost Constructor for LF depth (Wang et al., CVPR 2022)
- DistgDisp — disentangling disparity for light field (Wang et al., TPAMI 2022)
- SubFocal — Sub-Aperture Feature Aggregation (2022)
- LF-Transformer — Transformer for light field depth (Wang et al., 2023)
- Neural Integral Imaging — NeRF-based integral image reconstruction (2024)
- Diffusion-based Integral Image Super-Resolution (2025)

#### Step 3: Update Integral Photography Solvers

After listing all integral photography solvers, update `algorithm_base/integral/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All integral photography solvers use the data format: `y` (U, V, H, W, C) 4D light field or (H_raw, W_raw, C) raw lenslet image, `microlens_params` dict with pitch, focal length, grid geometry, `calibration` dict with main lens parameters. The `IntegralOperator` handles lenslet-to-sub-aperture conversion, computational refocusing, and depth reconstruction.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for integral photography:**
- HCI 4D LF benchmark (MSE x100 / BadPix 0.07): CIIR ~8.0 / 25%, TV-Regularized ~3.5 / 15%, EPI-CNN ~1.5 / 8%, LFattNet ~1.0 / 5%, OACC-Net ~0.7 / 3.5%
- Heidelberg benchmark: Bayesian ~2.5 MSE, DistgDisp ~0.6 MSE
- Lytro dataset refocusing PSNR: Comp. Refocusing ~32.0 dB, SR Integral ~35.0 dB, Neural Integral ~38.0 dB
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'integral' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/integral/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/integral/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/integral/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for integral photography. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/integral/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/integral/standard/`

---

### Light Field Imaging (`light_field`) Modality Template

#### Step 1: Verify Standard Dataset

For Light Field Imaging, what dataset do you use to verify? Is this dataset used for light field imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/light_field/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original light field standard dataset.

**Popular datasets to consider:**
- **Stanford Light Field Archive (Levoy & Hanrahan, SIGGRAPH 1996; Vaish et al., 2004)** — the original multi-camera array light field captures; foundational dataset for light field research
- **Heidelberg 4D Light Field Benchmark (Honauer et al., ACCV 2016)** — the gold standard quantitative benchmark; 24 synthetic + 4 real scenes with ground-truth depth/disparity; used by all light field depth papers
- **INRIA Lytro Dataset (Dansereau et al., 2013)** — decoded Lytro camera light fields; widely used for view synthesis and super-resolution evaluation
- **HCI Light Field Dataset (Wanner et al., VMV 2013)** — synthetic 4D light fields with ground-truth disparity; the standard evaluation set for LF depth algorithms
- **Kalantari Light Field Video Dataset (Kalantari et al., SIGGRAPH 2016)** — sparse-to-dense light field view synthesis benchmark

**Decision criteria:** Heidelberg 4D / HCI is the undisputed gold standard for light field depth estimation benchmarking (2013–2026); Stanford archive for historical and novel view synthesis. Use the dataset that appears in the largest number of light field papers.

#### Step 2: List All Light Field Algorithms

Please first ensure all the light field algorithms have been listed in `\Physics_World_Model\algorithm_base\light_field\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/light_field. Besides, you need to search all algorithms from 1950 to 2026. After listing all the light field solvers, please update the light field solver.

**Key algorithms to cover (1950–2026):**

_Classical Light Field (1996–2010):_
- Light Field Rendering — image-based rendering from dense samples (Levoy & Hanrahan, SIGGRAPH 1996) — foundational light field paper
- Lumigraph — unstructured light field rendering (Gortler et al., SIGGRAPH 1996)
- Light Field Photography — digital refocusing from plenoptic camera (Ng et al., Stanford 2005)
- EPI Analysis — depth from slope in Epipolar Plane Images (Bolles et al., IJCV 1987)
- Structure-from-Light-Field — depth estimation from sub-aperture views (Tao et al., ICCV 2013; roots earlier)
- Fourier Disparity Layer Representation (Le Pendu et al., TIP 2019; concept roots 2005)

_Optimization & Variational (2010–2017):_
- Global Variational Light Field Depth (Wanner & Goldluecke, IJCV 2014)
- Spinning Parallelogram Operator for LF depth (Zhang et al., CVPR 2016)
- Depth from Light Field Defocus and Correspondence (Tao et al., ICCV 2013)
- Light Field Super-Resolution via Gaussian Mixture (Wanner & Goldluecke, ECCV 2012)
- Angular Super-Resolution via Sparse Coding (Shi et al., 2014)
- Low-Rank Light Field Representation (Kamal et al., 2016)
- Dictionary Learning for Light Field Super-Resolution (Farrugia & Guillemot, TIP 2017)
- Robust Depth from Light Field via MRF (Chen et al., 2014)
- Bilateral Consistency for LF Depth (Jeon et al., CVPR 2015)
- Multi-Label Optimization for LF Depth (Strecke et al., 2017)

_Deep Learning (2017–2026):_
- EPINET — Epipolar Plane Image Network for LF depth (Shin et al., CVPR 2018) — dominant deep LF depth method
- LFattNet — Attention-based Light Field depth estimation (Tsai et al., TIP 2020)
- OACC-Net — Occlusion-Aware Cost Constructor (Wang et al., CVPR 2022)
- DistgDisp — Disentangling Disparity for Light Field (Wang et al., TPAMI 2022)
- LF-InterNet — Spatial-Angular Interaction for LF SR (Wang et al., 2020)
- DistgSSR — Disentangling Light Field SR (Wang et al., CVPR 2022)
- LFT — Light Field Transformer (Liang et al., 2022)
- EPIT — EPI Transformer for LF depth (2023)
- Neural Light Field — NeRF for light field (Sitzmann et al., NeurIPS 2021)
- DiffusionLF — diffusion for light field view synthesis (2024)
- Foundation model for light field processing (2025)

#### Step 3: Update Light Field Solvers

After listing all light field solvers, update `algorithm_base/light_field/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All light field solvers use the data format: `y` (U, V, H, W, C) 4D light field array with (U, V) angular and (H, W) spatial dimensions, `disparity_range` (d_min, d_max) search range, `baseline` float inter-camera spacing. The `LightFieldOperator` handles view warping, EPI extraction, refocusing, and disparity-to-depth conversion.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for light field:**
- HCI 4D benchmark (MSE x100 / BadPix 0.07): EPI classical ~3.0 / 12%, Spinning Parallelo ~2.5 / 10%, EPINET ~0.8 / 4.5%, LFattNet ~0.7 / 3.8%, OACC-Net ~0.5 / 3.0%, DistgDisp ~0.45 / 2.8%
- Heidelberg 4D: Variational ~2.0 MSE, EPINET ~0.9, DistgDisp ~0.5
- Light field spatial SR (x2 PSNR): Bicubic ~33.0 dB, LF-InterNet ~37.5 dB, DistgSSR ~38.2 dB, LFT ~38.8 dB
- All reference values from published papers and benchmark leaderboards

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'light_field' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/light_field/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/light_field/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/light_field/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for light field. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/light_field/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/light_field/standard/`

---

### Stellar Coronagraphy (`coronagraphy`) Modality Template

#### Step 1: Verify Standard Dataset

For Stellar Coronagraphy, what dataset do you use to verify? Is this dataset used for stellar coronagraphy / high-contrast imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/coronagraphy/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original coronagraphy standard dataset.

**Popular datasets to consider:**
- **VLT/SPHERE-IRDIS Exoplanet Imaging Data Challenge (Cantalloube et al., 2020)** — standardized coronagraphic ADI datasets from VLT/SPHERE with injected companions at known positions; the primary benchmark for high-contrast imaging algorithms
- **Gemini/GPI Exoplanet Survey Data (Macintosh et al., PNAS 2014)** — archival GPI coronagraphic data with known companions; used for algorithm validation
- **HST Coronagraphic Reference Library (Schneider et al., 2014)** — Hubble Space Telescope coronagraphic images with PSF references; used for circumstellar disk and companion detection
- **JWST NIRCam/MIRI Coronagraphic Data (Kammerer et al., 2022)** — James Webb Space Telescope coronagraphic observations; newest benchmark data
- **Vortex Imaging Processing (VIP) Tutorial Data** — standardized ADI+SDI sequences bundled with the VIP library for algorithm testing

**Decision criteria:** The SPHERE Exoplanet Data Challenge is the gold standard for coronagraphic post-processing benchmarking (2020–2026); GPI data for real companion validation. Use the dataset that appears in the largest number of high-contrast imaging algorithm papers.

#### Step 2: List All Coronagraphy Algorithms

Please first ensure all the coronagraphy algorithms have been listed in `\Physics_World_Model\algorithm_base\coronagraphy\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/coronagraphy. Besides, you need to search all algorithms from 1950 to 2026. After listing all the coronagraphy solvers, please update the coronagraphy solver.

**Key algorithms to cover (1950–2026):**

_Classical PSF Subtraction (1984–2006):_
- Classical ADI — Angular Differential Imaging with median PSF subtraction (Marois et al., ApJ 2006) — foundational high-contrast imaging technique
- Roll Subtraction — telescope roll-based reference subtraction (Schneider & Silverstone, 2003)
- Reference Star Differential Imaging (RDI) — using reference star observations for PSF subtraction (Ruane et al., 2019)
- SDI — Spectral Differential Imaging for companion detection (Sparks & Ford, ApJ 2002)
- Simple Median Combination — median of derotated frames (baseline)

_Advanced PSF Modeling (2007–2016):_
- LOCI — Locally Optimized Combination of Images (Lafreniere et al., ApJ 2007) — the dominant pre-PCA method
- PCA / KLIP — Karhunen-Loeve Image Projection for PSF subtraction (Soummer et al., ApJL 2012; Amara & Quanz 2012) — the current standard method
- NMF — Non-Negative Matrix Factorization for coronagraphy (Ren et al., ApJ 2018)
- TLOCI — Template LOCI (Marois et al., 2014)
- pyKLIP — Python KLIP implementation (Wang et al., 2015)
- ANDROMEDA — Bayesian matched filter detection (Cantalloube et al., A&A 2015)
- PACO — PAtch COvariance for exoplanet detection (Flasseur et al., A&A 2018)
- Forward Model Matched Filter (Ruffio et al., ApJ 2017)

_Deep Learning & Statistical (2018–2026):_
- SODINN — deep learning for exoplanet detection (Gomez Gonzalez et al., A&A 2018)
- Supervised ML Detection — Random Forest / SVM for companion classification (Yip et al., 2020)
- Deep PACO — deep learning enhanced PACO (Flasseur et al., 2023)
- ContraNet — CNN for high-contrast imaging (Cantero et al., 2023)
- Starflow — normalizing flow for coronagraphic speckle subtraction (2023)
- NA-SODINN — noise-aware deep detection (Cantero et al., 2024)
- Half-Sibling Regression for speckle removal (Gebhard et al., A&A 2022)
- Diffusion-based Coronagraphic Reconstruction (2025)
- Foundation model for high-contrast imaging (2026)

#### Step 3: Update Coronagraphy Solvers

After listing all coronagraphy solvers, update `algorithm_base/coronagraphy/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All coronagraphy solvers use the data format: `y` (T, H, W) temporal cube of coronagraphic frames, `parallactic_angles` (T,) derotation angles for ADI, `wavelengths` (T,) or (L,) for SDI. The `CoronagraphyOperator` handles field rotation, PSF model subtraction, frame derotation, and signal-to-noise map computation.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for coronagraphy:**
- SPHERE Data Challenge (TPR @ FPR=1e-4): Median ADI ~0.10, LOCI ~0.25, PCA/KLIP ~0.40, PACO ~0.65, SODINN ~0.55, NA-SODINN ~0.70
- SPHERE Data Challenge (5-sigma contrast at 0.5"): Classical ADI ~1e-4, PCA ~5e-5, PACO ~2e-5, Deep PACO ~1e-5
- GPI benchmark: PCA ~5e-6 contrast at 1", NMF ~3e-6, ANDROMEDA ~2e-6
- All reference values from published papers and Exoplanet Data Challenge leaderboards

**Verification criteria:**
- `done` — PWM within 0.5x contrast factor of reference
- `partial` — 0.5x–2x contrast shortfall
- `gap` — >2x contrast shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'coronagraphy' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/coronagraphy/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/coronagraphy/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/coronagraphy/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for coronagraphy. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/coronagraphy/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/coronagraphy/standard/`

---

### Adaptive Optics Imaging (`adaptive_optics`) Modality Template

#### Step 1: Verify Standard Dataset

For Adaptive Optics Imaging, what dataset do you use to verify? Is this dataset used for adaptive optics popular algorithms? Please ensure the standard dataset in `datasets/benchmark/adaptive_optics/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original adaptive optics standard dataset.

**Popular datasets to consider:**
- **Keck AO Archive (Wizinowich et al., 2000)** — adaptive optics imaging data from Keck Observatory; the most widely used ground-based AO dataset with Strehl ratio measurements
- **VLT/MUSE AO Dataset (Bacon et al., 2010)** — MUSE integral-field spectrograph AO-corrected data; used for AO PSF reconstruction evaluation
- **ESO AO PSF Reconstruction Benchmark (Beltramo-Martin et al., 2020)** — standardized AO PSF reconstruction data with ground-truth PSFs from marginal wavefront sensing
- **Gemini/NIRI AO Dataset (Herriot et al., 2000)** — natural guide star and laser guide star AO data for PSF estimation benchmarks
- **COMPASS Simulation Toolkit Data (Ferreira et al., 2018)** — synthetic AO telemetry with known atmospheric parameters; used for algorithm development and validation

**Decision criteria:** ESO AO PSF Reconstruction Benchmark is the emerging gold standard for AO PSF estimation (2020–2026); Keck AO archive for real astronomical AO; COMPASS for simulation-based evaluation. Use the dataset that appears in the largest number of AO reconstruction papers.

#### Step 2: List All Adaptive Optics Algorithms

Please first ensure all the adaptive optics algorithms have been listed in `\Physics_World_Model\algorithm_base\adaptive_optics\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/adaptive_optics. Besides, you need to search all algorithms from 1950 to 2026. After listing all the adaptive optics solvers, please update the adaptive optics solver.

**Key algorithms to cover (1950–2026):**

_Classical Wavefront Sensing & Correction (1953–2000):_
- Babcock Concept — original adaptive optics concept for astronomical seeing compensation (Babcock, PASP 1953)
- Shack-Hartmann Wavefront Sensor — lenslet-array wavefront slope measurement (Shack & Platt, 1971)
- Zernike Modal Decomposition — wavefront reconstruction via Zernike polynomial fitting (Noll, JOSA 1976)
- Least-Squares Wavefront Reconstruction — direct slope-to-phase conversion via matrix inversion
- Curvature Wavefront Sensing (Roddier, Applied Optics 1988)
- Pyramid Wavefront Sensor (Ragazzoni, Journal of Modern Optics 1996)

_Advanced Reconstruction & Control (2000–2016):_
- MMSE Wavefront Reconstruction — Minimum Mean Square Error tomographic reconstruction (Ellerbroek, JOSA A 2002)
- Fourier Transform Wavefront Reconstruction (Poyneer et al., JOSA A 2002)
- CuReD — Cumulative Reconstructor with Domain Decomposition (Rosensteiner, JOSA A 2012)
- Multi-Conjugate AO (MCAO) Tomographic Reconstruction (Beckers, 1988; Fusco et al., 2001)
- MOAO — Multi-Object AO open-loop control (Hammer et al., 2004)
- LTAO — Laser Tomography AO (Foy & Labeyrie, 1985; implemented 2000s)
- Predictive Control for AO — temporal prediction of turbulence evolution (Dessenne et al., 1998)
- PSF Reconstruction from AO Telemetry (Veran et al., JOSA A 1997)
- Sparse Wavefront Reconstruction (Helin & Yudytskiy, Inverse Problems 2013)

_Deep Learning (2017–2026):_
- Deep Wavefront Sensing — CNN for phase retrieval from focal plane images (Nishizaki et al., Optics Express 2019)
- Neural Network AO Control — NN-based wavefront prediction and correction (Swanson et al., 2018)
- Deep Reinforcement Learning for AO (Landman & Haffert, A&A 2020)
- U-Net PSF Reconstruction from AO telemetry (Beltramo-Martin et al., 2020)
- Phase Diversity with Deep Learning (Nishizaki et al., 2020)
- Recurrent NN for Predictive AO Control (Wong et al., 2021)
- GAN-based AO Image Deconvolution (Schreiber et al., 2022)
- Transformer for Multi-Conjugate AO (2023)
- Physics-Informed Neural Network for Turbulence Tomography (2024)
- Foundation model for AO wavefront correction (2025)

#### Step 3: Update Adaptive Optics Solvers

After listing all adaptive optics solvers, update `algorithm_base/adaptive_optics/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All adaptive optics solvers use the data format: `y` (N_sub, 2) Shack-Hartmann slope measurements or (H, W) focal-plane image, `interaction_matrix` (N_act, N_sub*2) DM-to-slope influence matrix, `r0` float Fried parameter, `L0` float outer scale. The `AdaptiveOpticsOperator` handles forward wavefront-to-slopes/PSF model and DM command-to-wavefront mapping.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for adaptive optics:**
- ESO benchmark (residual wavefront RMS in nm): Least-Squares ~180 nm, MMSE ~120 nm, Fourier ~125 nm, CuReD ~115 nm, NN Prediction ~95 nm
- Strehl ratio (H-band): Uncorrected ~0.02, Least-Squares ~0.35, MMSE ~0.55, Predictive ~0.60, RL-AO ~0.65
- PSF reconstruction fidelity: Classical Veran ~85% correlation, U-Net ~92%, Physics-Informed NN ~95%
- All reference values from published papers and AO system telemetry

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'adaptive_optics' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/adaptive_optics/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for adaptive optics. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/adaptive_optics/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/adaptive_optics/standard/`

---

### Lucky Imaging (`lucky_imaging`) Modality Template

#### Step 1: Verify Standard Dataset

For Lucky Imaging, what dataset do you use to verify? Is this dataset used for lucky imaging / speckle imaging popular algorithms? Please ensure the standard dataset in `datasets/benchmark/lucky_imaging/standard` is a popular dataset used by most algorithms. If not, please find some popular dataset to replace the original lucky imaging standard dataset.

**Popular datasets to consider:**
- **USNO Speckle Interferometry Archive (Hartkopf et al., 2001)** — the largest speckle/lucky imaging archive; thousands of binary star observations with known separations and magnitudes; the canonical reference for speckle interferometry validation
- **FastCam Lucky Imaging Dataset (Oscoz et al., PASP 2008)** — lucky imaging observations from the FastCam instrument at WHT/NOT; real data with diffraction-limited ground truth from HST comparison
- **AstraLux Lucky Imaging Survey (Hormuth et al., A&A 2008)** — large lucky imaging survey data from Calar Alto; used for binary star detection and photometry validation
- **Cambridge Lucky Imaging Data (Law et al., A&A 2006)** — original demonstration data from the Cambridge LuckyCam system; used for algorithm development and evaluation
- **Gemini/DSSI Speckle Dataset (Horch et al., AJ 2012)** — differential speckle survey instrument data with well-characterized point sources

**Decision criteria:** FastCam/AstraLux datasets are the most widely used for lucky imaging algorithm evaluation; USNO Speckle Archive for speckle interferometry; Cambridge data for historical lucky imaging methods. Use the dataset that appears in the largest number of lucky imaging papers.

#### Step 2: List All Lucky Imaging Algorithms

Please first ensure all the lucky imaging algorithms have been listed in `\Physics_World_Model\algorithm_base\lucky_imaging\README.md` and `\Physics_World_Model\datasets\benchmark\algorithm_state.md`. You can refer to https://pwm.platformai.org/benchmark/lucky_imaging. Besides, you need to search all algorithms from 1950 to 2026. After listing all the lucky imaging solvers, please update the lucky imaging solver.

**Key algorithms to cover (1950–2026):**

_Classical Lucky Imaging & Speckle (1970–2000):_
- Fried's Lucky Imaging Concept — select and co-add best short-exposure frames (Fried, JOSA 1978) — foundational lucky imaging paper
- Speckle Interferometry — Fourier analysis of short-exposure speckle patterns (Labeyrie, A&A 1970) — foundational speckle technique
- Shift-and-Add — register and stack short exposures on brightest speckle (Christou, MNRAS 1991)
- Knox-Thompson Method — phase recovery from cross-spectra (Knox & Thompson, ApJL 1974)
- Bispectral Analysis / Triple Correlation — closure-phase image reconstruction (Weigelt, Optics Communications 1977; Lohmann et al., Applied Optics 1983)

_Advanced Selection & Reconstruction (2000–2015):_
- Percentage Selection Lucky Imaging — select top N% frames by Strehl/sharpness (Law et al., A&A 2006) — the modern lucky imaging paper
- Lucky Region Selection — spatially varying frame selection (Law et al., ApJ 2006)
- FITSTARS — Frame Selection and Image Reconstruction algorithm (Mackay et al., 2004)
- Drizzle Integration for Lucky Imaging — sub-pixel sampling via dithered lucky frames (Hook & Lucy, 2004)
- Wiener Filter Post-Processing for lucky-imaging stacks (Garrel et al., 2012)
- Multi-Frame Blind Deconvolution (Schulz, JOSA A 1993)
- Speckle Holography — holographic image reconstruction from speckle patterns (Pehlemann et al., A&A 1992)
- BSMEM — BiSpectrum Maximum Entropy Method (Buscher, 1994)
- Iterative Blind Deconvolution for speckle stacks (Jefferies & Christou, ApJ 1993)
- Phase Diversity Speckle Reconstruction (Paxman et al., JOSA A 1992)

_Deep Learning (2018–2026):_
- Deep Lucky Imaging — CNN-based frame selection and stacking (Staley, 2019)
- Neural Frame Quality Assessment — learned Strehl/quality metric for selection (2020)
- GAN-based Super-Resolution for Lucky Imaging stacks (2021)
- Multi-Frame Super-Resolution Network — deep fusion of short-exposure frames (2021)
- Physics-Aware Lucky Imaging Network — incorporating atmospheric model (2022)
- Transformer for Multi-Frame Registration and Fusion (2023)
- Diffusion-based Atmospheric Deblurring from short exposures (2024)
- Self-Supervised Lucky Imaging — no ground truth required (2025)
- Foundation model for atmospheric-degraded image restoration (2025)

#### Step 3: Update Lucky Imaging Solvers

After listing all lucky imaging solvers, update `algorithm_base/lucky_imaging/solvers.py` to include implementations for each algorithm. Ensure each solver follows the standard interface:

```python
def run_solver(y: np.ndarray, operator: Any, cfg: dict) -> np.ndarray
```

All lucky imaging solvers use the data format: `y` (T, H, W) temporal cube of short-exposure frames, `exposure_time` float individual frame exposure time, `wavelength` float observation wavelength, `D` float telescope aperture diameter. The `LuckyImagingOperator` handles frame quality assessment (Strehl ratio estimation), shift-and-add registration, and atmospheric PSF modeling.

#### Step 4: Verify Each Algorithm

Then you need to verify each algorithm one by one. You test every algorithm based on the standard dataset. The algorithm which can achieve the same results as in the reference can be marked as done.

**Reference benchmarks for lucky imaging:**
- FastCam binary star (resolution in mas): Long exposure ~500 mas, 10% selection ~120 mas, 1% selection ~80 mas, Shift-and-Add ~100 mas, Speckle Interferometry ~60 mas
- AstraLux survey (Strehl ratio, I-band): Long exposure ~0.03, 10% lucky ~0.15, 1% lucky ~0.25, Deep Lucky ~0.30
- Bispectral reconstruction dynamic range: Speckle Interferometry ~5 mag, Bispectral ~7 mag, BSMEM ~8 mag
- All reference values from published papers and observatory reports

**Verification criteria:**
- `done` — PWM within 10% of reference metric
- `partial` — 10–30% shortfall
- `gap` — >30% shortfall
- `no_ckpt` — Algorithm documented but pretrained weights not available
- `fail` — Solver diverged

#### Step 5: Upload Checkpoints to GCS

After verifying all the algorithms, you can upload the checkpoints into GCS. You first create the 'lucky_imaging' folder in `/pwm-benchmark-datasets/checkpoint/` if this path has no that folder, then you can upload checkpoints for each GPU algorithm into `/pwm-benchmark-datasets/checkpoint/lucky_imaging/`.

#### Step 6: Upload Standard Dataset to GCS

You also need to upload the standard dataset into `/pwm-benchmark-datasets/datasets/Benchmark/lucky_imaging/standard/`. If there are some datasets in `/pwm-benchmark-datasets/datasets/Benchmark/lucky_imaging/standard/`, you should first compare them and ensure the best popular dataset should be the standard dataset for lucky imaging. You keep the most popular dataset for local and GCS.

#### Step 7: Push to GitHub

Then you push into GitHub but don't push the checkpoint. Don't push standard dataset into GitHub. You also need to ensure the standard dataset is uploaded to GCS before pushing.

**Checkpoint Storage:** `gs://pwm-benchmark-datasets/checkpoint/lucky_imaging/`
**Dataset Storage:** `gs://pwm-benchmark-datasets/datasets/Benchmark/lucky_imaging/standard/`
