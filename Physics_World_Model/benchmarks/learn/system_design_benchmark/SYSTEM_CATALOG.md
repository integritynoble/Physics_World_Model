# PWM System Catalog — All 168 Imaging Modalities

Auto-generated from `platform/scripts/generate_system_catalog.py`.
Source data: MODALITY_CATALOG, algorithm catalog, CATEGORY_REAL_SCORES.

---

## Summary

- **Total modalities:** 168
- **Categories:** 19
- **Total algorithm entries:** 1367
- **Average best PSNR:** 35.1 dB

## Category Overview

| Category | Count | Avg Best PSNR (dB) | Avg Capital Cost (k USD) |
|----------|-------|---------------------|--------------------------|
| astronomy | 4 | 30.4 | 126762 |
| coherent | 5 | 35.1 | 96 |
| compressive | 4 | 38.6 | 34 |
| computational | 2 | 35.6 | 102 |
| computational_photography | 5 | 35.8 | 2 |
| depth_imaging | 5 | 35.4 | 16 |
| electron_microscopy | 11 | 33.2 | 1609 |
| experimental_science | 11 | 32.9 | 106511 |
| industrial_inspection | 10 | 35.6 | 75 |
| medical | 37 | 37.5 | 1054 |
| microscopy | 24 | 36.8 | 216 |
| multi_modal_fusion | 6 | 34.2 | 2183 |
| neural_rendering | 2 | 35.9 | 1 |
| quantum | 3 | 29.6 | 103 |
| remote_sensing | 11 | 34.3 | 14052 |
| scanning_probe | 4 | 32.5 | 142 |
| scientific_instrumentation | 12 | 32.8 | 18208 |
| spectroscopy | 8 | 33.0 | 312 |
| ultrafast | 4 | 32.9 | 125145 |

## Astronomy (4 modalities)

### Stellar Coronagraphy
**ID:** `coronagraphy` | **Carrier:** Photon | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 5 µm |
| Output dimensionality | 2D |
| Capital cost | $5000k |
| Operator skill | specialist |
| Solver latency | 60s |

**Mismatch parameters (4):** `c_m` (Coronagraph mask centering (lambda/D)), `w_e` (Wavefront error (WFE) (-)), `s_l` (Stellar leakage (contrast)), `s_l` (Speckle lifetime (s))

**Best solver:** ANDROMEDA — **PSNR:** 28.0 dB, **SSIM:** 0.84
**Worst PSNR:** 18.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Statistical

**Sample:** in-vivo capable

---

### Event Horizon Telescope (EHT) Imaging
**ID:** `eht_imaging` | **Carrier:** RF | **DAG:** `F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 2D |
| Capital cost | $500000.0k |
| Operator skill | specialist |
| Solver latency | 86400s |

**Mismatch parameters (4):** `a_o` (Atmospheric opacity (tau) (nepers)), `s_g` (Station gain calibration (-)), `u_s` (uv-coverage sparsity (-)), `i_s` (Interstellar scattering (uas broadening))

**Best solver:** PRIMO — **PSNR:** 31.2 dB, **SSIM:** 0.905
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

---

### Lucky Imaging
**ID:** `lucky_imaging` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 100 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D |
| Capital cost | $50k |
| Operator skill | expert |
| Solver latency | 30s |

**Mismatch parameters (4):** `f_p` (Fried parameter (r0) (cm)), `f_s` (Frame selection threshold (-)), `i_a` (Isoplanatic angle (arcsec)), `r_e` (Registration error (px))

**Best solver:** PRIMO — **PSNR:** 31.2 dB, **SSIM:** 0.905
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### Solar EUV/X-ray Imaging
**ID:** `solar_imaging` | **Carrier:** Photon/EUV | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D/3D |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `p_d` (PSF degradation (mirror aging) (-)), `s_l` (Stray light (-)), `f_e` (Flat-field error (-)), `p_j` (Pointing jitter (arcsec))

**Best solver:** PRIMO — **PSNR:** 31.2 dB, **SSIM:** 0.905
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

## Coherent (5 modalities)

### Digital Holographic Microscopy
**ID:** `holography` | **Carrier:** Photon | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+phase |
| Capital cost | $50k |
| Operator skill | expert |
| Solver latency | 0.1s |

**Mismatch parameters (3):** `r_a` (Reference angle error (deg)), `c_f` (Carrier frequency error (-)), `v` (Vibration (-))

**Best solver:** ScorePhase — **PSNR:** 35.82 dB, **SSIM:** 0.968
**Worst PSNR:** 21.5 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Optical Diffraction Tomography (ODT)
**ID:** `odt` | **Carrier:** Photon | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 50 |
| Max FPS | 1 |
| Spatial resolution | 200 nm |
| Output dimensionality | 3D(RI) |
| Capital cost | $100k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `i_a` (Illumination angle error (deg per angle)), `m_c` (Missing cone artifact (deg)), `r_i` (Refractive index of medium (-)), `m_s` (Multiple scattering (-))

**Best solver:** Rytov-Former — **PSNR:** 34.0 dB, **SSIM:** 0.935
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Coherent Diffractive Imaging / Phase Retrieval
**ID:** `phase_retrieval` | **Carrier:** Photon/Electron | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 3 |
| Max FPS | 30 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+phase |
| Capital cost | $30k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (3):** `s_m` (Support mask error (-)), `o_r` (Oversampling ratio (-)), `p_c` (Partial coherence (-))

**Best solver:** ScorePhase — **PSNR:** 35.82 dB, **SSIM:** 0.968
**Worst PSNR:** 21.5 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, Score-based, Vision Transformer

---

### Ptychographic Imaging
**ID:** `ptychography` | **Carrier:** Electron/Photon | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 10 nm |
| Output dimensionality | 2D+phase |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (3):** `p_p` (Probe position error (px)), `d` (Defocus (nm)), `p_c` (Partial coherence (-))

**Best solver:** AutoPhaseNN — **PSNR:** 34.0 dB, **SSIM:** 0.935
**Worst PSNR:** 25.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning

---

### Talbot-Lau X-ray Grating Interferometry
**ID:** `talbot_lau` | **Carrier:** X-ray | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 4 |
| Max FPS | 5 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D(abs,dpc,df) |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 0.01s |

**Mismatch parameters (4):** `g_a` (Grating alignment (rotation) (deg)), `i_d` (Inter-grating distance error (-)), `p_s` (Phase stepping error (per step)), `g_d` (Grating defect fraction (-))

**Best solver:** ScorePhase — **PSNR:** 35.82 dB, **SSIM:** 0.968
**Worst PSNR:** 21.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

## Compressive (4 modalities)

### Coded Aperture Compressive Temporal Imaging (CACTI)
**ID:** `cacti` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100,000,000 |
| Spatial resolution | 5 µm |
| Output dimensionality | 3D(x,y,t) |
| Capital cost | $15k |
| Operator skill | expert |
| Solver latency | 2.1s |

**Mismatch parameters (5):** `s_s` (Spatial shift x,y (px)), `r` (Rotation (deg)), `t_c` (Temporal clock error (frame frac)), `g_/` (Gain / offset (- / counts)), `f_g` (Frame-dependent gain (-))

**Best solver:** FlowHSI — **PSNR:** 38.58 dB, **SSIM:** 0.982
**Worst PSNR:** 26.83 dB | **Algorithms in catalog:** 5 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Deep Unfolding, PnP, Transformer

**Sample:** in-vivo capable

---

### Coded Aperture Snapshot Spectral Imaging (CASSI)
**ID:** `cassi` | **Carrier:** Photon | **DAG:** `M → W → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 5 µm |
| Output dimensionality | 3D(x,y,lam) |
| Capital cost | $20k |
| Operator skill | expert |
| Solver latency | 3.5s |

**Mismatch parameters (7):** `m_s` (Mask shift dx (px)), `m_s` (Mask shift dy (px)), `m_r` (Mask rotation (deg)), `d_s` (Dispersion slope a1 (px/band)), `d_o` (Dispersion offset alpha (px)), `g` (Gain (-)), `r_n` (Read noise (e-))

**Best solver:** FlowHSI — **PSNR:** 38.58 dB, **SSIM:** 0.982
**Worst PSNR:** 26.83 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Diffusion, Generative, PnP, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Generic Matrix Sensing
**ID:** `matrix` | **Carrier:** Photon | **DAG:** `M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 300 nm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $100k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (2):** `m_p` (Matrix perturbation (A)), `c_n` (Condition number change (-))

**Best solver:** FlowHSI — **PSNR:** 38.58 dB, **SSIM:** 0.982
**Worst PSNR:** 26.83 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Diffusion, Generative, PnP, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Single-Pixel Camera (SPC)
**ID:** `spc` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 1 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D |
| Capital cost | $2k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Mismatch parameters (3):** `g_d` (Gain drift alpha (-)), `m_n` (Measurement noise sigma_y (-)), `p_e` (Pattern error (bit flips) (-))

**Best solver:** FlowHSI — **PSNR:** 38.58 dB, **SSIM:** 0.982
**Worst PSNR:** 26.83 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 14

**Algorithm types:** Classical, Deep Learning, Diffusion, Generative, PnP, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

## Computational (2 modalities)

### Integral Photography
**ID:** `integral` | **Carrier:** Photon | **DAG:** `C → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 10 µm |
| Output dimensionality | 3D(x,y,lam) |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 2s |

**Best solver:** DistgSSR — **PSNR:** 35.8 dB, **SSIM:** 0.95
**Worst PSNR:** 25.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Light Field Imaging
**ID:** `light_field` | **Carrier:** Photon | **DAG:** `C → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 10 µm |
| Output dimensionality | 4D(x,y,u,v) |
| Capital cost | $5k |
| Operator skill | untrained |
| Solver latency | 0.5s |

**Best solver:** DistgSSR — **PSNR:** 35.5 dB, **SSIM:** 0.948
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

## Computational Photography (5 modalities)

### Coded Exposure / Flutter Shutter
**ID:** `coded_exposure` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 5 µm |
| Output dimensionality | 2D+blur |
| Capital cost | $3k |
| Operator skill | technician |
| Solver latency | 0.01s |

**Mismatch parameters (3):** `s_c` (Shutter code timing error (-)), `m_b` (Motion blur PSF mismatch (velocity error)), `s_r` (Sensor readout noise (e-))

**Best solver:** DiffusionPhoto — **PSNR:** 38.82 dB, **SSIM:** 0.978
**Worst PSNR:** 27.8 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Event Camera / Dynamic Vision Sensor (DVS)
**ID:** `event_camera` | **Carrier:** Photon | **DAG:** `M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1,000,000 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D(events) |
| Capital cost | $5k |
| Operator skill | technician |
| Solver latency | 0.01s |

**Mismatch parameters (4):** `c_t` (Contrast threshold (log intensity)), `r_p` (Refractory period (us)), `n_e` (Noise event rate (of real events)), `h_p` (Hot pixel fraction (-))

**Best solver:** SPADE-E2VID — **PSNR:** 33.0 dB, **SSIM:** 0.93
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### High Dynamic Range (HDR) Imaging
**ID:** `hdr_imaging` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 3 |
| Max FPS | 10 |
| Spatial resolution | 5 µm |
| Output dimensionality | 2D(HDR) |
| Capital cost | $1k |
| Operator skill | untrained |
| Solver latency | 0.1s |

**Mismatch parameters (3):** `c_r` (Camera response function error (-)), `e_r` (Exposure ratio error (-)), `g_a` (Ghost artifact (motion between exposures) (px))

**Best solver:** DiffusionPhoto — **PSNR:** 38.82 dB, **SSIM:** 0.978
**Worst PSNR:** 27.8 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Lensless (Diffuser Camera) Imaging
**ID:** `lensless` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 50 µm |
| Output dimensionality | 2D/3D |
| Capital cost | $0.5k |
| Operator skill | untrained |
| Solver latency | 0.1s |

**Best solver:** Uformer — **PSNR:** 33.5 dB, **SSIM:** 0.92
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Panorama Multi-Focus Fusion
**ID:** `panorama` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10 |
| Max FPS | 5 |
| Spatial resolution | 5 µm |
| Output dimensionality | 2D(360) |
| Capital cost | $0.5k |
| Operator skill | untrained |
| Solver latency | 1s |

**Best solver:** PanoFormer — **PSNR:** 35.0 dB, **SSIM:** 0.95
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

## Depth Imaging (5 modalities)

### Flash LiDAR
**ID:** `flash_lidar` | **Carrier:** Photon | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 3D(depth) |
| Capital cost | $20k |
| Operator skill | untrained |
| Solver latency | 0.05s |

**Mismatch parameters (4):** `s_j` (SPAD jitter (ps)), `a_p` (Ambient photon rate (-)), `p_d` (Pile-up distortion (at high flux)), `p_c` (Pixel cross-talk (-))

**Best solver:** SPADNet — **PSNR:** 33.2 dB, **SSIM:** 0.93
**Worst PSNR:** 23.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### LiDAR Scanner
**ID:** `lidar` | **Carrier:** Photon | **DAG:** `P → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 10 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 3D(point) |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Best solver:** DiffusionDepth — **PSNR:** 37.68 dB, **SSIM:** 0.978
**Worst PSNR:** 25.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Photometric Stereo
**ID:** `photometric_stereo` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 4 |
| Max FPS | 5 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D(normal) |
| Capital cost | $2k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Mismatch parameters (4):** `l_d` (Light direction error (deg per source)), `l_i` (Light intensity calibration (-)), `n_s` (Non-Lambertian surface fraction (-)), `c_s` (Cast shadow fraction (of pixels))

**Best solver:** PS-Transformer — **PSNR:** 34.2 dB, **SSIM:** 0.945
**Worst PSNR:** 25.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Structured-Light Depth Camera
**ID:** `structured_light` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 5 |
| Max FPS | 10 |
| Spatial resolution | 50 µm |
| Output dimensionality | 3D(depth) |
| Capital cost | $5k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Best solver:** DiffusionDepth — **PSNR:** 37.68 dB, **SSIM:** 0.978
**Worst PSNR:** 25.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Time-of-Flight Depth Camera
**ID:** `tof_camera` | **Carrier:** Photon/IR | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 3D(depth) |
| Capital cost | $2k |
| Operator skill | untrained |
| Solver latency | 0.01s |

**Best solver:** MPI-Former — **PSNR:** 34.0 dB, **SSIM:** 0.93
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

## Electron Microscopy (11 modalities)

### Cryo-Electron Tomography (Cryo-ET)
**ID:** `cryo_et` | **Carrier:** Electron | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 60 |
| Max FPS | 0 |
| Spatial resolution | 2 nm |
| Output dimensionality | 3D |
| Capital cost | $3000k |
| Operator skill | specialist |
| Solver latency | 1800s |

**Mismatch parameters (5):** `t_a` (Tilt axis offset (px)), `t_a` (Tilt angle accuracy (deg per tilt)), `d_s` (Dose-induced shrinkage (-)), `c_p` (CTF per-tilt variation (um)), `m_w` (Missing wedge (deg))

**Best solver:** ScoreCryoEM — **PSNR:** 34.58 dB, **SSIM:** 0.947
**Worst PSNR:** 22.3 dB | **Algorithms in catalog:** 12 | **Benchmark results:** 11

**Algorithm types:** Classical, Deep Learning, Diffusion, Score-based, Transformer, Vision Transformer

**Sample:** destructive

---

### Electron Backscatter Diffraction (EBSD)
**ID:** `ebsd` | **Carrier:** Electron | **DAG:** `R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 50 nm |
| Output dimensionality | 2D(orient) |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 5s |

**Best solver:** EBSD-Former — **PSNR:** 32.5 dB, **SSIM:** 0.915
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** destructive

---

### STEM-EDX Elemental Mapping
**ID:** `edx_mapping` | **Carrier:** Electron | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 10 nm |
| Output dimensionality | 2D+element |
| Capital cost | $300k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `a_c` (Absorption correction error (-)), `d_s` (Detector solid angle (sr)), `p_o` (Peak overlap (spectral) (-)), `b_b` (Bremsstrahlung background (-))

**Best solver:** SwinIR — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** destructive

---

### Electron Energy Loss Spectroscopy (EELS)
**ID:** `eels` | **Carrier:** Electron | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 1D+spec |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 5s |

**Best solver:** EELS-Net — **PSNR:** 32.0 dB, **SSIM:** 0.91
**Worst PSNR:** 23.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** destructive

---

### 4D-STEM Electron Diffraction
**ID:** `electron_diffraction` | **Carrier:** Electron | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1 |
| Spatial resolution | 0 nm |
| Output dimensionality | 2D(recip) |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 10s |

**Best solver:** AutoPhaseNN — **PSNR:** 33.0 dB, **SSIM:** 0.925
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning

**Sample:** destructive

---

### Electron Holography
**ID:** `electron_holography` | **Carrier:** Electron | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1 |
| Spatial resolution | 0 nm |
| Output dimensionality | 2D+phase |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 5s |

**Best solver:** PhaseNet-EH — **PSNR:** 34.5 dB, **SSIM:** 0.94
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** destructive

---

### Electron Tomography
**ID:** `electron_tomography` | **Carrier:** Electron | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 1 nm |
| Output dimensionality | 3D |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 600s |

**Best solver:** CryoAI — **PSNR:** 32.0 dB, **SSIM:** 0.91
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning

**Sample:** destructive

---

### Focused Ion Beam SEM (FIB-SEM)
**ID:** `fib_sem` | **Carrier:** Electron | **DAG:** `S → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 5 nm |
| Output dimensionality | 3D |
| Capital cost | $1000k |
| Operator skill | specialist |
| Solver latency | 600s |

**Mismatch parameters (4):** `s_t` (Slice thickness variation (-)), `c_a` (Curtaining artifact (relative)), `c` (Charging (V)), `d_b` (Drift between slices (nm))

**Best solver:** SwinIR — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** destructive

---

### Scanning Electron Microscopy (SEM)
**ID:** `sem` | **Carrier:** Electron | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 1 nm |
| Output dimensionality | 2D |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 1s |

**Best solver:** SwinIR — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** destructive

---

### Scanning Transmission Electron Microscopy (STEM)
**ID:** `stem` | **Carrier:** Electron | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 2D |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 2s |

**Best solver:** SwinIR — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** destructive

---

### Transmission Electron Microscopy (TEM)
**ID:** `tem` | **Carrier:** Electron | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 0 nm |
| Output dimensionality | 2D |
| Capital cost | $3000k |
| Operator skill | specialist |
| Solver latency | 5s |

**Best solver:** SwinIR — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** destructive

---

## Experimental Science (11 modalities)

### Acoustic Emission Testing (AE)
**ID:** `acoustic_emission` | **Carrier:** Acoustic | **DAG:** `P → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1,000,000 |
| Spatial resolution | 5.0 mm |
| Output dimensionality | 2D(source) |
| Capital cost | $10k |
| Operator skill | technician |
| Solver latency | 5s |

**Mismatch parameters (4):** `s_l` (Source location error (mm)), `w_s` (Wave speed error (m/s)), `s_c` (Sensor coupling gain (-)), `a_t` (Arrival time bias (us))

**Best solver:** DiffusionAE — **PSNR:** 35.5 dB, **SSIM:** 0.95
**Worst PSNR:** 20.5 dB | **Algorithms in catalog:** 9 | **Benchmark results:** 9

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** contact, in-vivo capable

---

### Adaptive Optics (AO) Imaging
**ID:** `adaptive_optics` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1,000 |
| Spatial resolution | 500 nm |
| Output dimensionality | 2D |
| Capital cost | $1000k |
| Operator skill | specialist |
| Solver latency | 0.001s |

**Mismatch parameters (4):** `d_a` (DM actuator gain (-)), `w_c` (WFS centroid bias (px)), `f_p` (Fried parameter r0 (m)), `s_l` (Servo lag (ms))

**Best solver:** DiffusionAO — **PSNR:** 35.0 dB, **SSIM:** 0.948
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 8 | **Benchmark results:** 8

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Transformer

**Sample:** in-vivo capable

---

### Bioluminescence Tomography (BLT)
**ID:** `bioluminescence_tomo` | **Carrier:** Photon | **DAG:** `Src → R → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 3D |
| Capital cost | $100k |
| Operator skill | specialist |
| Solver latency | 30s |

**Mismatch parameters (3):** `o_p` (Optical property error (mu_a, mu_s') (relative)), `s_d` (Source depth ambiguity (mm)), `a_b` (Autofluorescence background (-))

**Best solver:** SwinIR — **PSNR:** 34.1 dB, **SSIM:** 0.942
**Worst PSNR:** 25.4 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Full-Waveform Inversion (FWI)
**ID:** `fwi` | **Carrier:** Seismic/Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 10000.0 mm |
| Output dimensionality | 3D(velocity) |
| Capital cost | $5000k |
| Operator skill | specialist |
| Solver latency | 3600s |

**Mismatch parameters (4):** `s_v` (Starting velocity model error (-)), `s_w` (Source wavelet error (-)), `a_a` (Anelastic attenuation (Q) (-)), `s_l` (Source location error (m))

**Best solver:** VelocityGAN — **PSNR:** 32.2 dB, **SSIM:** 0.91
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning

---

### Gravitational Wave Detection
**ID:** `gravitational_wave` | **Carrier:** Gravitational | **DAG:** `P → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 16,000 |
| Spatial resolution | -1000 nm |
| Output dimensionality | 1D(strain) |
| Capital cost | $1000000.0k |
| Operator skill | specialist |
| Solver latency | 1s |

**Mismatch parameters (4):** `c_a` (Calibration amplitude (-)), `p_c` (Phase calibration (rad)), `p_s` (Power spectral density (1/Hz)), `t_o` (Timing offset (s))

**Best solver:** WaveFormer — **PSNR:** 30.5 dB, **SSIM:** 0.895
**Worst PSNR:** 20.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

### Electrical Impedance Tomography (EIT)
**ID:** `impedance_tomo` | **Carrier:** Electric | **DAG:** `M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 10 |
| Spatial resolution | 50.0 mm |
| Output dimensionality | 2D |
| Capital cost | $10k |
| Operator skill | technician |
| Solver latency | 1s |

**Mismatch parameters (4):** `c_i` (Contact impedance (ohm)), `e_p` (Electrode position error (mm)), `b_c` (Background conductivity (S/m)), `c_a` (Current amplitude drift (-))

**Best solver:** EIT-Former — **PSNR:** 30.0 dB, **SSIM:** 0.88
**Worst PSNR:** 21.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

### Magnetic Particle Imaging (MPI)
**ID:** `magnetic_particle` | **Carrier:** Magnetic | **DAG:** `M → F → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 40 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 3D |
| Capital cost | $500k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `d_f` (Drive field amplitude (mT)), `s_f` (Selection field gradient (T/m)), `p_r` (Particle relaxation time (us)), `r_c` (Receive coil sensitivity (-))

**Best solver:** SwinIR — **PSNR:** 34.1 dB, **SSIM:** 0.942
**Worst PSNR:** 25.4 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

---

### Ocean Acoustic Tomography
**ID:** `ocean_acoustic_tomo` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 100000.0 mm |
| Output dimensionality | 3D(sound-spd) |
| Capital cost | $5000k |
| Operator skill | specialist |
| Solver latency | 300s |

**Mismatch parameters (4):** `s_s` (Sound speed profile error (-)), `m_i` (Multipath identification (-)), `s_p` (Source/receiver position (m)), `c_v` (Current velocity error (m/s))

**Best solver:** SwinIR — **PSNR:** 34.1 dB, **SSIM:** 0.942
**Worst PSNR:** 25.4 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** contact, in-vivo capable

---

### Particle Calorimetry
**ID:** `particle_calorimetry` | **Carrier:** Particle | **DAG:** `R → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 40,000,000 |
| Spatial resolution | 100.0 mm |
| Output dimensionality | 3D(energy) |
| Capital cost | $100000.0k |
| Operator skill | specialist |
| Solver latency | 0.001s |

**Mismatch parameters (4):** `e_s` (Energy scale factor (-)), `p_r` (Position resolution (mm)), `s_f` (Sampling fraction (-)), `p_f` (Pile-up fraction (-))

**Best solver:** CaloDiffusion — **PSNR:** 31.5 dB, **SSIM:** 0.9
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion

---

### Radio Aperture Synthesis
**ID:** `radio_astronomy` | **Carrier:** RF | **DAG:** `F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 2D |
| Capital cost | $50000k |
| Operator skill | specialist |
| Solver latency | 60s |

**Mismatch parameters (4):** `a_g` (Antenna gain error (-)), `p_c` (Phase calibration error (deg)), `b_s` (Bandpass slope (1/MHz)), `p_o` (Pointing offset (arcsec))

**Best solver:** PRIMO — **PSNR:** 31.2 dB, **SSIM:** 0.905
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

---

### Seismic Tomography
**ID:** `seismic_tomo` | **Carrier:** Seismic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 100000.0 mm |
| Output dimensionality | 3D(velocity) |
| Capital cost | $10000k |
| Operator skill | specialist |
| Solver latency | 3600s |

**Mismatch parameters (4):** `v_m` (Velocity model error (m/s)), `s_l` (Source location error (m)), `r_c` (Receiver coupling (-)), `t_e` (Timing error (s))

**Best solver:** SwinIR — **PSNR:** 34.1 dB, **SSIM:** 0.942
**Worst PSNR:** 25.4 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

---

## Industrial Inspection (10 modalities)

### Scanning Acoustic Microscopy (SAM)
**ID:** `acoustic_microscopy` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 1 µm |
| Output dimensionality | 2D |
| Capital cost | $80k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `c_m` (Coupling medium speed (m/s)), `f_d` (Focus depth error (um)), `l_a` (Lens aberration (waves)), `g_p` (Gate position error (-))

**Best solver:** DiffusionSAM — **PSNR:** 35.0 dB, **SSIM:** 0.948
**Worst PSNR:** 21.5 dB | **Algorithms in catalog:** 8 | **Benchmark results:** 8

**Algorithm types:** Classical, Deep Learning, Diffusion, Physics-Informed, PnP, Self-Supervised, Transformer

**Sample:** contact, in-vivo capable

---

### Active Thermography (IR)
**ID:** `active_thermography` | **Carrier:** IR | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 500 µm |
| Output dimensionality | 2D+t(IR) |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 2s |

**Mismatch parameters (4):** `e_e` (Emissivity error (-)), `h_s` (Heat source power drift (-)), `b_t` (Background temperature (C)), `i_t` (Integration time offset (s))

**Best solver:** DiffusionThermo — **PSNR:** 35.5 dB, **SSIM:** 0.95
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 8 | **Benchmark results:** 8

**Algorithm types:** Classical, Deep Learning, Diffusion, Physics-Informed, PnP, Transformer

---

### Eddy Current Imaging
**ID:** `eddy_current` | **Carrier:** EM | **DAG:** `F → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 10 |
| Spatial resolution | 500 µm |
| Output dimensionality | 2D |
| Capital cost | $15k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Mismatch parameters (4):** `l_d` (Liftoff distance (mm)), `c_e` (Conductivity error (MS/m)), `e_f` (Excitation frequency drift (kHz)), `p_t` (Probe tilt (deg))

**Best solver:** ECT-Former — **PSNR:** 33.5 dB, **SSIM:** 0.925
**Worst PSNR:** 23.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

### Industrial X-ray CT
**ID:** `industrial_ct` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 5 µm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $300k |
| Operator skill | expert |
| Solver latency | 30s |

**Mismatch parameters (4):** `c_o` (Center of rotation offset (px)), `s_v` (Source voltage drift (kV)), `d_t` (Detector tilt (deg)), `b_h` (Beam hardening coefficient (-))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, PnP

**Sample:** in-vivo capable

---

### Machine Vision / AOI
**ID:** `machine_vision` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D |
| Capital cost | $5k |
| Operator skill | untrained |
| Solver latency | 0.01s |

**Mismatch parameters (4):** `f_d` (Focus distance error (mm)), `l_d` (Lens distortion k1 (-)), `e_t` (Exposure time drift (ms)), `w_b` (White balance gain (-))

**Best solver:** LSTM-NDT — **PSNR:** 34.8 dB, **SSIM:** 0.95
**Worst PSNR:** 26.2 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Shearography
**ID:** `shearography` | **Carrier:** Photon | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D(strain) |
| Capital cost | $40k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Mismatch parameters (3):** `s_a` (Shearing amount error (-)), `s_d` (Speckle decorrelation (-)), `l_n` (Loading non-uniformity (-))

**Best solver:** PhaseFormer — **PSNR:** 34.0 dB, **SSIM:** 0.935
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Terahertz Imaging (THz)
**ID:** `terahertz` | **Carrier:** THz | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 200 µm |
| Output dimensionality | 2D+spec |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `p_c` (Pulse chirp (ps^2)), `w_v` (Water vapor absorption (1/cm)), `b_a` (Beam alignment error (mm)), `d_r` (Dynamic range drift (-))

**Best solver:** THz-Former — **PSNR:** 34.5 dB, **SSIM:** 0.94
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

### Ultrasonic Phased Array (TFM/FMC)
**ID:** `ultrasonic_phased_array` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 300 µm |
| Output dimensionality | 2D/3D |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Mismatch parameters (4):** `e_p` (Element pitch error (mm)), `s_v` (Sound velocity (m/s)), `w_a` (Wedge angle error (deg)), `d_e` (Dead element fraction (-))

**Best solver:** FMC-Former — **PSNR:** 34.5 dB, **SSIM:** 0.94
**Worst PSNR:** 25.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** contact, in-vivo capable

---

### X-ray NDT (Radiography)
**ID:** `xray_ndt` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Mismatch parameters (4):** `s_p` (Source position error (mm)), `b_h` (Beam hardening (-)), `d_g` (Detector gain drift (-)), `g_m` (Geometric magnification (-))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### X-ray Fluorescence (XRF) Imaging
**ID:** `xrf_imaging` | **Carrier:** X-ray | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 20 µm |
| Output dimensionality | 2D+element |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `e_e` (Excitation energy drift (keV)), `d_r` (Detector resolution (eV)), `m_a` (Matrix absorption (-)), `b_s` (Beam spot size (um))

**Best solver:** SpectraFormer — **PSNR:** 34.0 dB, **SSIM:** 0.935
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

## Medical (37 modalities)

### X-ray Angiography
**ID:** `angiography` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10 |
| Max FPS | 15 |
| Spatial resolution | 200 µm |
| Output dimensionality | 2D(t) |
| Capital cost | $500k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Arterial Spin Labeling (ASL) MRI
**ID:** `asl_mri` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 2.0 mm |
| Output dimensionality | 2D+perfusion |
| Capital cost | $2000k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (3):** `l_e` (Labeling efficiency (-)), `t_d` (Transit delay (s)), `t_b` (T1 blood error (-))

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### Brachytherapy Imaging
**ID:** `brachytherapy_img` | **Carrier:** Gamma/X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1 |
| Spatial resolution | 500 µm |
| Output dimensionality | 3D |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `s_p` (Source position error (mm)), `a_c` (Attenuation coefficient (1/cm)), `d_g` (Detector gain drift (-)), `s_f` (Scatter fraction (-))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Cone-Beam Computed Tomography (CBCT)
**ID:** `cbct` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 300 |
| Max FPS | 0 |
| Spatial resolution | 200 µm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $200k |
| Operator skill | technician |
| Solver latency | 5s |

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### CEST MRI
**ID:** `cest_mri` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 2D+Z-spec |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (4):** `b_i` (B0 inhomogeneity (Hz)), `b_i` (B1 inhomogeneity (-)), `s_p` (Saturation power error (-)), `m_c` (MT contamination (-))

**Best solver:** CEST-Former — **PSNR:** 33.0 dB, **SSIM:** 0.92
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Contrast-Enhanced Ultrasound (CEUS)
**ID:** `ceus` | **Carrier:** Acoustic | **DAG:** `P → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 300 µm |
| Output dimensionality | 2D+perfusion |
| Capital cost | $80k |
| Operator skill | expert |
| Solver latency | 0.05s |

**Mismatch parameters (3):** `b_c` (Bubble concentration (relative)), `n_h` (Nonlinear harmonic extraction (-)), `m_b` (Motion between frames (mm))

**Best solver:** ScoreUS — **PSNR:** 36.28 dB, **SSIM:** 0.962
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** contact, in-vivo capable

---

### Confocal Laser Endomicroscopy (CLE)
**ID:** `confocal_endomicroscopy` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 1 µm |
| Output dimensionality | 2D |
| Capital cost | $100k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Mismatch parameters (3):** `f_b` (Fiber bundle honeycomb pattern (-)), `m_a` (Motion artifact (px/frame)), `f_c` (Fluorescein concentration variation (relative))

**Best solver:** EndoL2H — **PSNR:** 33.2 dB, **SSIM:** 0.93
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### X-ray Computed Tomography (CT)
**ID:** `ct` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 300 µm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $1000k |
| Operator skill | technician |
| Solver latency | 5s |

**Mismatch parameters (5):** `c_o` (Center-of-rotation offset (px)), `a_o` (Angular offset (deg)), `d_t` (Detector tilt (deg)), `b_h` (Beam hardening coeff (-)), `r_a` (Ring artifact amplitude (counts))

**Best solver:** CTFlow — **PSNR:** 40.15 dB, **SSIM:** 0.985
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Dual-Energy X-ray Absorptiometry (DEXA)
**ID:** `dexa` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1 |
| Spatial resolution | 500 µm |
| Output dimensionality | 2D(dual-E) |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, PnP

**Sample:** in-vivo capable

---

### Diffusion MRI (DTI)
**ID:** `diffusion_mri` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 1.5 mm |
| Output dimensionality | 3D+diffusion |
| Capital cost | $2000k |
| Operator skill | expert |
| Solver latency | 10s |

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### Digital Breast Tomosynthesis (DBT)
**ID:** `digital_breast_tomo` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 15 |
| Max FPS | 1 |
| Spatial resolution | 100 µm |
| Output dimensionality | 3D |
| Capital cost | $400k |
| Operator skill | technician |
| Solver latency | 5s |

**Mismatch parameters (3):** `a_r` (Angular range error (deg total)), `d_m` (Detector motion blur (px)), `s_f` (Scatter fraction (-))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Doppler Ultrasound
**ID:** `doppler_ultrasound` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 50 |
| Spatial resolution | 300 µm |
| Output dimensionality | 2D+velocity |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 0.005s |

**Best solver:** ScoreUS — **PSNR:** 36.28 dB, **SSIM:** 0.962
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** contact, in-vivo capable

---

### Diffuse Optical Tomography (DOT)
**ID:** `dot` | **Carrier:** Photon | **DAG:** `M → R → P → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 1 |
| Spatial resolution | 5.0 mm |
| Output dimensionality | 3D |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 10s |

**Best solver:** DeepDOT — **PSNR:** 30.5 dB, **SSIM:** 0.89
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### Shear-Wave Elastography
**ID:** `elastography` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 2 |
| Max FPS | 20 |
| Spatial resolution | 500 µm |
| Output dimensionality | 2D+stiffness |
| Capital cost | $80k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Best solver:** ElastNet — **PSNR:** 33.0 dB, **SSIM:** 0.92
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** contact, in-vivo capable

---

### Fiber Bundle Endoscopy
**ID:** `endoscopy` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 50 µm |
| Output dimensionality | 2D |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 0.03s |

**Best solver:** EndoL2H — **PSNR:** 33.2 dB, **SSIM:** 0.93
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### Fluoroscopy
**ID:** `fluoroscopy` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 200 µm |
| Output dimensionality | 2D(t) |
| Capital cost | $300k |
| Operator skill | technician |
| Solver latency | 0.03s |

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Functional MRI (BOLD fMRI)
**ID:** `fmri` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 2.0 mm |
| Output dimensionality | 3D+t(BOLD) |
| Capital cost | $2000k |
| Operator skill | expert |
| Solver latency | 2s |

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### Fundus Camera
**ID:** `fundus` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D |
| Capital cost | $20k |
| Operator skill | technician |
| Solver latency | 0.05s |

**Best solver:** Swin-Fundus — **PSNR:** 34.2 dB, **SSIM:** 0.94
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Intravascular Ultrasound (IVUS)
**ID:** `ivus` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D(cross-sec) |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 0.01s |

**Mismatch parameters (3):** `c_r` (Catheter rotation non-uniformity (-)), `r_a` (Ring-down artifact (-)), `s_s` (Sound speed in plaque (m/s))

**Best solver:** ScoreUS — **PSNR:** 36.28 dB, **SSIM:** 0.962
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** contact, in-vivo capable

---

### Mammography
**ID:** `mammography` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 2 |
| Spatial resolution | 70 µm |
| Output dimensionality | 2D |
| Capital cost | $200k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### MR Elastography (MRE)
**ID:** `mr_elastography` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 2.0 mm |
| Output dimensionality | 3D+stiffness |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (4):** `s_w` (Shear wave frequency error (-)), `w_a` (Wave attenuation model (-)), `m_e` (Motion encoding gradient error (-)), `b_r` (Boundary reflection (amplitude))

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### MR Fingerprinting (MRF)
**ID:** `mr_fingerprinting` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 2D+multi-param |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (3):** `d_r` (Dictionary resolution (T1, T2) (-)), `b_i` (B1 inhomogeneity (-)), `u_a` (Undersampling artifact (-))

**Best solver:** MRF-Former — **PSNR:** 33.5 dB, **SSIM:** 0.93
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### MR Angiography (MRA)
**ID:** `mra` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 500 µm |
| Output dimensionality | 3D(vessels) |
| Capital cost | $2000k |
| Operator skill | technician |
| Solver latency | 5s |

**Mismatch parameters (3):** `c_t` (Contrast timing error (s)), `b_s` (Background suppression (-)), `v_e` (Velocity encoding error (-))

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### Magnetic Resonance Imaging (MRI)
**ID:** `mri` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 500 µm |
| Output dimensionality | 3D+contrast |
| Capital cost | $2000k |
| Operator skill | technician |
| Solver latency | 2s |

**Mismatch parameters (4):** `c_s` (Coil sensitivity error (relative)), `k_t` (k-space trajectory deviation (-)), `o_(` (Off-resonance (B0) (Hz)), `a_f` (Acceleration factor (-))

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### MR Spectroscopy (MRS)
**ID:** `mrs` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 1D+chem |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 2s |

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### Functional Near-Infrared Spectroscopy (fNIRS)
**ID:** `nirs_brain` | **Carrier:** Photon | **DAG:** `M → R → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 2D(cortex) |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 1s |

**Mismatch parameters (4):** `s_c` (Source-detector coupling (-)), `s_d` (Scalp-brain distance variation (mm)), `m_a` (Motion artifact (head) (-)), `s_p` (Systemic physiology contamination (-))

**Best solver:** DeepDOT — **PSNR:** 30.5 dB, **SSIM:** 0.89
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### Optical Coherence Tomography (OCT)
**ID:** `oct` | **Carrier:** Photon | **DAG:** `P+P → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100,000 |
| Spatial resolution | 5 µm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $80k |
| Operator skill | technician |
| Solver latency | 1s |

**Mismatch parameters (3):** `d_g` (Dispersion GDD (fs^2)), `r_a` (Reference arm position (um)), `k_e` (K-linearization error (relative))

**Best solver:** ScoreOCT — **PSNR:** 37.95 dB, **SSIM:** 0.973
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### OCT Angiography (OCTA)
**ID:** `octa` | **Carrier:** Photon | **DAG:** `P+P → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 2 |
| Max FPS | 50,000 |
| Spatial resolution | 10 µm |
| Output dimensionality | 3D(vessels) |
| Capital cost | $100k |
| Operator skill | technician |
| Solver latency | 2s |

**Best solver:** ScoreOCT — **PSNR:** 37.95 dB, **SSIM:** 0.973
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Positron Emission Tomography (PET)
**ID:** `pet` | **Carrier:** Gamma | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 4.0 mm |
| Output dimensionality | 3D |
| Capital cost | $2000k |
| Operator skill | technician |
| Solver latency | 30s |

**Mismatch parameters (5):** `a_m` (Attenuation map error (HU-to-LAC)), `s_f` (Scatter fraction (-)), `r_f` (Randoms fraction (-)), `n_e` (Normalization error (-)), `t_t` (TOF timing offset (ps))

**Best solver:** TransEM — **PSNR:** 33.7 dB, **SSIM:** 0.938
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Photoacoustic Imaging
**ID:** `photoacoustic` | **Carrier:** Acoustic | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 5 |
| Spatial resolution | 50 µm |
| Output dimensionality | 3D |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 5s |

**Best solver:** PAT-Former — **PSNR:** 33.5 dB, **SSIM:** 0.92
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** contact, in-vivo capable

---

### Portal Imaging (EPID)
**ID:** `portal_imaging` | **Carrier:** MV | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 15 |
| Spatial resolution | 400 µm |
| Output dimensionality | 2D |
| Capital cost | $200k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Mismatch parameters (4):** `i_s` (Isocenter shift (mm)), `b_e` (Beam energy variation (MV)), `d_s` (Detector sag (mm)), `s_k` (Scatter kernel width (mm))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Proton Therapy Imaging
**ID:** `proton_therapy_img` | **Carrier:** Proton | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 3D |
| Capital cost | $10000k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (4):** `r_u` (Range uncertainty (mm)), `s_p` (Scattering power error (-)), `d_e` (Detector efficiency drift (-)), `s_e` (Setup error (mm))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Single Photon Emission CT (SPECT)
**ID:** `spect` | **Carrier:** Gamma | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 8.0 mm |
| Output dimensionality | 3D |
| Capital cost | $500k |
| Operator skill | technician |
| Solver latency | 30s |

**Mismatch parameters (3):** `c_r` (Collimator response error (-)), `c` (Center-of-rotation (px)), `a_e` (Attenuation error (relative))

**Best solver:** TransEM — **PSNR:** 33.7 dB, **SSIM:** 0.938
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Photon-Counting Spectral CT
**ID:** `spectral_ct` | **Carrier:** X-ray | **DAG:** `Π → W → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 300 µm |
| Output dimensionality | 3D+E |
| Capital cost | $2000k |
| Operator skill | technician |
| Solver latency | 10s |

**Mismatch parameters (4):** `e_t` (Energy threshold calibration (keV per bin)), `c_s` (Charge sharing fraction (-)), `p_a` (Pile-up at high flux (-)), `m_d` (Material decomposition basis error (-))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

### Susceptibility-Weighted Imaging (SWI)
**ID:** `swi` | **Carrier:** Spin/RF | **DAG:** `M → F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 500 µm |
| Output dimensionality | 3D+suscept |
| Capital cost | $2000k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (3):** `p_u` (Phase unwrapping error (-)), `b_f` (Background field removal error (-)), `d_i` (Dipole inversion regularization (-))

**Best solver:** MRDynamo — **PSNR:** 40.45 dB, **SSIM:** 0.938
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 10

**Algorithm types:** Classical, Compressed Sensing, Deep Learning, Deep Unrolling, Diffusion, Physics-Informed, PnP, Transformer

**Sample:** in-vivo capable

---

### Ultrasound B-mode Imaging
**ID:** `ultrasound` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 300 µm |
| Output dimensionality | 2D |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 0.001s |

**Mismatch parameters (4):** `s_o` (Speed of sound (m/s)), `p_a` (Phase aberration (ns rms)), `e_s` (Element sensitivity (-)), `a` (Attenuation (dB/cm/MHz))

**Best solver:** ScoreUS — **PSNR:** 36.28 dB, **SSIM:** 0.962
**Worst PSNR:** 24.5 dB | **Algorithms in catalog:** 14 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** contact, in-vivo capable

---

### X-ray Radiography
**ID:** `xray_radiography` | **Carrier:** X-ray | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 0.05s |

**Mismatch parameters (3):** `s_f` (Scatter fraction (-)), `b_h` (Beam hardening (-)), `d_l` (Detector lag (fraction))

**Best solver:** Score-CT — **PSNR:** 39.92 dB, **SSIM:** 0.984
**Worst PSNR:** 27.38 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Transformer, Vision Transformer

**Sample:** in-vivo capable

---

## Microscopy (24 modalities)

### Confocal 3D Z-Stack
**ID:** `confocal_3d` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 500 |
| Max FPS | 0 |
| Spatial resolution | 200 nm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `a_p` (Axial PSF sigma (px)), `r_i` (Refractive index (-)), `a_c` (Attenuation coeff (per slice)), `s_a` (Spherical aberration (waves))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Confocal Live-Cell Microscopy
**ID:** `confocal_livecell` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 50 |
| Max FPS | 5 |
| Spatial resolution | 200 nm |
| Output dimensionality | 3D+t |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 0.5s |

**Mismatch parameters (4):** `p_s` (PSF sigma (px)), `d_r` (Drift rate (px/frame)), `b_r` (Bleaching rate (per frame)), `p_m` (Pinhole misalignment (AU offset))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Dark-Field Microscopy
**ID:** `dark_field` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 500 nm |
| Output dimensionality | 2D |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 0.05s |

**Mismatch parameters (3):** `c_n` (Condenser NA vs objective NA ratio (-)), `s_l` (Stray light (relative)), `s_a` (Scattering angle range (-))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Differential Interference Contrast (DIC)
**ID:** `dic` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 200 nm |
| Output dimensionality | 2D+gradient |
| Capital cost | $80k |
| Operator skill | technician |
| Solver latency | 0.1s |

**Mismatch parameters (3):** `s_a` (Shear amount (nm)), `b_r` (Bias retardation (nm)), `p_o` (Prism orientation (deg))

**Best solver:** PhaseFormer — **PSNR:** 33.5 dB, **SSIM:** 0.93
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### DNA-PAINT Super-Resolution
**ID:** `dna_paint` | **Carrier:** Photon | **DAG:** `M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 50000 |
| Max FPS | 0 |
| Spatial resolution | 10 nm |
| Output dimensionality | 2D/3D |
| Capital cost | $150k |
| Operator skill | specialist |
| Solver latency | 120s |

**Mismatch parameters (4):** `b_o` (Binding on-rate (relative)), `i_s` (Imager strand concentration (nM)), `d_r` (Drift rate (nm/frame)), `b_f` (Background from non-specific binding (-))

**Best solver:** DECODE — **PSNR:** 32.1 dB, **SSIM:** 0.915
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### Expansion Microscopy (ExM)
**ID:** `expansion` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1 |
| Spatial resolution | 70 nm |
| Output dimensionality | 3D |
| Capital cost | $50k |
| Operator skill | expert |
| Solver latency | 10s |

**Mismatch parameters (3):** `e_f` (Expansion factor (x)), `l_d` (Local distortion (relative)), `a_e` (Anisotropic expansion (x vs y))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Fluorescence Lifetime Imaging (FLIM)
**ID:** `flim` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+tau |
| Capital cost | $150k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `i_w` (IRF width (ps)), `i_s` (IRF shift (ps)), `a` (Afterpulsing (relative)), `p_f` (Pile-up fraction (-))

**Best solver:** FLIM-Former — **PSNR:** 33.5 dB, **SSIM:** 0.93
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Fourier Ptychographic Microscopy (FPM)
**ID:** `fpm` | **Carrier:** Photon | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 100 nm |
| Output dimensionality | 2D+phase |
| Capital cost | $20k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `l_p` (LED position error (mm)), `l_i` (LED intensity variation (relative)), `p_a` (Pupil aberration (Zernike) (waves)), `d` (Defocus (um))

**Best solver:** PtychoDV — **PSNR:** 33.8 dB, **SSIM:** 0.935
**Worst PSNR:** 25.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Deep Unrolling

**Sample:** in-vivo capable

---

### Image Scanning Microscopy (ISM)
**ID:** `ism` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 5 |
| Spatial resolution | 150 nm |
| Output dimensionality | 2D |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (2):** `d_e` (Detector element offset (px)), `m_e` (Magnification error (relative))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Lattice Light-Sheet Microscopy
**ID:** `lattice_lightsheet` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 200 |
| Max FPS | 10 |
| Spatial resolution | 200 nm |
| Output dimensionality | 3D(x,y,z)+t |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 2s |

**Mismatch parameters (4):** `l_p` (Lattice period error (relative)), `d_r` (Dithering range (-)), `s_n` (Sheet NA error (-)), `e_p` (Excitation PSF sidelobe (relative))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Light-Sheet Fluorescence Microscopy (LSFM)
**ID:** `lightsheet` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 200 |
| Max FPS | 5 |
| Spatial resolution | 400 nm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $150k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `s_t` (Sheet thickness (um)), `s_t` (Sheet tilt (deg)), `s_s` (Stripe strength (relative)), `a_c` (Attenuation coeff (per slice))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### MINFLUX Nanoscopy
**ID:** `minflux` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 2 nm |
| Output dimensionality | 3D |
| Capital cost | $800k |
| Operator skill | specialist |
| Solver latency | 60s |

**Mismatch parameters (2):** `b_c` (Beam center error (nm)), `p_c` (Photon count (photons))

**Best solver:** DECODE — **PSNR:** 32.1 dB, **SSIM:** 0.915
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### PALM/STORM Single-Molecule Localization
**ID:** `palm_storm` | **Carrier:** Photon | **DAG:** `M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 20 nm |
| Output dimensionality | 2D/3D |
| Capital cost | $150k |
| Operator skill | expert |
| Solver latency | 30s |

**Mismatch parameters (4):** `d_r` (Drift rate (x, y) (nm/frame)), `b_p` (Background photons (per px)), `p_c` (Photon count/event (photons)), `p_s` (Pixel size (nm))

**Best solver:** DECODE — **PSNR:** 32.1 dB, **SSIM:** 0.915
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### Phase Contrast Microscopy
**ID:** `phase_contrast` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+phase |
| Capital cost | $60k |
| Operator skill | expert |
| Solver latency | 0.1s |

**Mismatch parameters (3):** `p_r` (Phase ring alignment (um offset)), `h_a` (Halo artifact strength (relative)), `p_r` (Phase ring absorption (-))

**Best solver:** PhaseFormer — **PSNR:** 35.0 dB, **SSIM:** 0.945
**Worst PSNR:** 25.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Polarization Microscopy
**ID:** `polarization` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 5 µm |
| Output dimensionality | 2D+Stokes |
| Capital cost | $20k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Mismatch parameters (3):** `a_a` (Analyzer angle offset (deg)), `r_o` (Retardance offset (nm)), `e_r` (Extinction ratio (-))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Second Harmonic Generation (SHG) Microscopy
**ID:** `shg` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 500 |
| Max FPS | 1 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D |
| Capital cost | $300k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (3):** `p_m` (Phase matching error (-)), `e_p` (Excitation power fluctuation (-)), `c_n` (Collection NA mismatch (-))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Structured Illumination Microscopy (SIM)
**ID:** `sim` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 9 |
| Max FPS | 10 |
| Spatial resolution | 100 nm |
| Output dimensionality | 2D/3D |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 0.5s |

**Mismatch parameters (4):** `p_f` (Pattern frequency (cycles/px)), `p_s` (Phase shifts (rad)), `m_d` (Modulation depth (-)), `p_o` (Pattern orientation (deg))

**Best solver:** SIMformer — **PSNR:** 36.5 dB, **SSIM:** 0.96
**Worst PSNR:** 28.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Spinning Disk Confocal Microscopy
**ID:** `spinning_disk` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 10 |
| Spatial resolution | 250 nm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $250k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (3):** `p_c` (Pinhole crosstalk (-)), `d_r` (Disk rotation wobble (px)), `i_n` (Illumination non-uniformity (-))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### STED Microscopy
**ID:** `sted` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 500 |
| Max FPS | 1 |
| Spatial resolution | 50 nm |
| Output dimensionality | 2D/3D |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 2s |

**Mismatch parameters (3):** `d_b` (Depletion beam alignment (nm offset)), `s_f` (Saturation factor (-)), `e_p` (Effective PSF FWHM (nm))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Three-Photon Microscopy
**ID:** `three_photon` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 500 |
| Max FPS | 0 |
| Spatial resolution | 300 nm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $600k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (3):** `s_c` (Scattering coeff (mm^-1)), `e_w` (Excitation wavelength shift (nm)), `d_p` (Depth-dependent PSF (-))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### TIRF Microscopy
**ID:** `tirf` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 200 nm |
| Output dimensionality | 2D(surface) |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (3):** `i_a` (Incidence angle (deg)), `e_d` (Evanescent depth (nm)), `b_(` (Background (non-TIRF) (relative))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Two-Photon / Multiphoton Microscopy
**ID:** `two_photon` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 500 |
| Max FPS | 1 |
| Spatial resolution | 300 nm |
| Output dimensionality | 3D(x,y,z) |
| Capital cost | $400k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `s_c` (Scattering coeff (mm^-1)), `p_d` (PSF depth scaling (-)), `e_a` (Excitation attenuation (per um)), `m_a` (Motion artifact (um))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Widefield Fluorescence Microscopy
**ID:** `widefield` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 0.05s |

**Mismatch parameters (5):** `p_s` (PSF sigma (px)), `b_l` (Background level (counts)), `g` (Gain (-)), `f_n` (Flatfield non-uniformity (peak-to-peak)), `p_r` (Photobleaching rate (per frame))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Low-Dose Widefield Microscopy
**ID:** `widefield_lowdose` | **Carrier:** Photon | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 100 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D |
| Capital cost | $30k |
| Operator skill | technician |
| Solver latency | 0.05s |

**Mismatch parameters (4):** `p_r` (Photon rate alpha (photons/px)), `r_n` (Read noise sigma (e-)), `b` (Background (counts)), `d_c` (Dark current (e-/px/s))

**Best solver:** ScoreMicro — **PSNR:** 38.48 dB, **SSIM:** 0.981
**Worst PSNR:** 27.1 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 13

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

## Multi Modal Fusion (6 modalities)

### Correlative Light-Electron Microscopy (CLEM)
**ID:** `clem` | **Carrier:** Photon | **DAG:** `(C → D) + (C → D) → ⊕`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 10 nm |
| Output dimensionality | 2D(correlated) |
| Capital cost | $3000k |
| Operator skill | specialist |
| Solver latency | 300s |

**Mismatch parameters (3):** `r_e` (Registration error (LM to EM) (nm)), `s_d` (Sample deformation (fixation) (shrinkage)), `f_p` (Fluorescence preservation (-))

**Best solver:** PPMF-Net — **PSNR:** 34.3 dB, **SSIM:** 0.945
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning

**Sample:** in-vivo capable

---

### CT + Fluorescence (FLIT)
**ID:** `ct_fluorescence` | **Carrier:** X-ray | **DAG:** `(Π → D) + (M → R → P → D) → ⊕`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 200 µm |
| Output dimensionality | 3D+element |
| Capital cost | $1000k |
| Operator skill | specialist |
| Solver latency | 30s |

**Mismatch parameters (3):** `o_p` (Optical property assignment error (-)), `a` (Autofluorescence (-)), `r_(` (Registration (CT to optical) (mm))

**Best solver:** PPMF-Net — **PSNR:** 34.3 dB, **SSIM:** 0.945
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### PET/CT Fusion
**ID:** `pet_ct` | **Carrier:** X-ray | **DAG:** `(Π → D) + (Π → D) → ⊕`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 2.0 mm |
| Output dimensionality | 3D+anat |
| Capital cost | $3000k |
| Operator skill | technician |
| Solver latency | 30s |

**Mismatch parameters (4):** `c_r` (CT-PET registration error (mm)), `a_m` (Attenuation map from CT error (HU-to-LAC)), `r_m` (Respiratory motion mismatch (mm)), `c_c` (CT contrast agent artifact (attenuation))

**Best solver:** PPMF-Net — **PSNR:** 34.3 dB, **SSIM:** 0.945
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### PET/MR Fusion
**ID:** `pet_mr` | **Carrier:** Gamma | **DAG:** `(Π → D) + (M → F → S → D) → ⊕`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 2.0 mm |
| Output dimensionality | 3D+multi |
| Capital cost | $5000k |
| Operator skill | specialist |
| Solver latency | 60s |

**Mismatch parameters (4):** `m_a` (MR-based attenuation error (-)), `s_a` (Susceptibility artifact at air/tissue (mm)), `t_s` (Timing synchronization (ms)), `t_(` (Truncation (MR FOV < PET FOV) (of body))

**Best solver:** PPMF-Net — **PSNR:** 34.3 dB, **SSIM:** 0.945
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### SPECT/CT Fusion
**ID:** `spect_ct` | **Carrier:** Gamma | **DAG:** `(Π → D) + (Π → D) → ⊕`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 5.0 mm |
| Output dimensionality | 3D+anat |
| Capital cost | $1000k |
| Operator skill | technician |
| Solver latency | 30s |

**Mismatch parameters (3):** `r_e` (Registration error (mm)), `c_a` (CT-based attenuation error (-)), `s_c` (Scatter correction error (-))

**Best solver:** TransEM — **PSNR:** 33.7 dB, **SSIM:** 0.938
**Worst PSNR:** 24.8 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

**Sample:** in-vivo capable

---

### US/MRI Fusion
**ID:** `us_mri` | **Carrier:** Acoustic | **DAG:** `(P → D) + (M → F → S → D) → ⊕`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 500 µm |
| Output dimensionality | 3D(fused) |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (3):** `r_e` (Registration error (deformable) (mm)), `p_p` (Probe pressure deformation (mm)), `m_d` (MR distortion (mm))

**Best solver:** PPMF-Net — **PSNR:** 34.3 dB, **SSIM:** 0.945
**Worst PSNR:** 25.6 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** contact, in-vivo capable

---

## Neural Rendering (2 modalities)

### 3D Gaussian Splatting (3DGS)
**ID:** `gaussian_splatting` | **Carrier:** Photon | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 50 |
| Max FPS | 30 |
| Spatial resolution | 10 µm |
| Output dimensionality | 3D(radiance) |
| Capital cost | $1k |
| Operator skill | untrained |
| Solver latency | 120s |

**Best solver:** NeRFactor2 — **PSNR:** 35.85 dB, **SSIM:** 0.966
**Worst PSNR:** 26.4 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Vision Transformer

**Sample:** in-vivo capable

---

### Neural Radiance Fields (NeRF)
**ID:** `nerf` | **Carrier:** Photon | **DAG:** `M → P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 50 |
| Max FPS | 0 |
| Spatial resolution | 10 µm |
| Output dimensionality | 3D(radiance) |
| Capital cost | $1k |
| Operator skill | untrained |
| Solver latency | 300s |

**Best solver:** NeRFactor2 — **PSNR:** 35.85 dB, **SSIM:** 0.966
**Worst PSNR:** 26.4 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Vision Transformer

**Sample:** in-vivo capable

---

## Quantum (3 modalities)

### Entangled Photon Microscopy
**ID:** `entangled_photon` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 50 µm |
| Output dimensionality | 2D |
| Capital cost | $100k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `p_g` (Pair generation rate (-)), `c_w` (Coincidence window (ns)), `a_c` (Accidental coincidence rate (-)), `p_l` (Photon loss (per arm) (dB))

**Best solver:** Ghost-ViT — **PSNR:** 30.1 dB, **SSIM:** 0.885
**Worst PSNR:** 21.2 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Ghost Imaging
**ID:** `ghost_imaging` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D |
| Capital cost | $10k |
| Operator skill | expert |
| Solver latency | 0.5s |

**Mismatch parameters (4):** `b_d` (Bucket detector efficiency (-)), `s_c` (Speckle correlation mismatch (-)), `b_c` (Background counts (-)), `n_o` (Number of measurements (-))

**Best solver:** Ghost-ViT — **PSNR:** 30.1 dB, **SSIM:** 0.885
**Worst PSNR:** 21.2 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Quantum Illumination
**ID:** `quantum_illumination` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 0 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 2D |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 1s |

**Mismatch parameters (4):** `e_q` (Entanglement quality (concurrence) (-)), `b_t` (Background thermal noise (-)), `d_d` (Detector dark count rate (Hz)), `c_l` (Channel loss (dB))

**Best solver:** QuantumFormer — **PSNR:** 28.5 dB, **SSIM:** 0.84
**Worst PSNR:** 18.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

## Remote Sensing (11 modalities)

### Ground-Penetrating Radar (GPR)
**ID:** `gpr` | **Carrier:** RF | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 1 |
| Spatial resolution | 100.0 mm |
| Output dimensionality | 2D/3D |
| Capital cost | $20k |
| Operator skill | technician |
| Solver latency | 5s |

**Mismatch parameters (4):** `s_p` (Soil permittivity error (-)), `a_h` (Antenna height (m)), `t_z` (Time zero offset (ns)), `v_m` (Velocity model error (m/ns))

**Best solver:** HyperDet — **PSNR:** 31.5 dB, **SSIM:** 0.905
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning

---

### Hyperspectral Remote Sensing
**ID:** `hyperspectral_remote` | **Carrier:** Photon | **DAG:** `M → W → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 500.0 mm |
| Output dimensionality | 2D+lam |
| Capital cost | $500k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `s_s` (Spectral shift (nm)), `s_d` (Smile distortion (px)), `k_d` (Keystone distortion (px)), `r_g` (Radiometric gain (-))

**Best solver:** MST++ — **PSNR:** 36.8 dB, **SSIM:** 0.955
**Worst PSNR:** 26.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Interferometric SAR (InSAR)
**ID:** `insar` | **Carrier:** RF | **DAG:** `F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 2 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 2D(deform) |
| Capital cost | $15000k |
| Operator skill | specialist |
| Solver latency | 30s |

**Mismatch parameters (4):** `p_u` (Phase unwrapping error (-)), `b_e` (Baseline estimation error (m)), `a_p` (Atmospheric phase screen (rad rms)), `t_d` (Temporal decorrelation (coherence loss))

**Best solver:** InSAR-Former — **PSNR:** 33.0 dB, **SSIM:** 0.92
**Worst PSNR:** 23.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

---

### Multispectral Satellite Imaging
**ID:** `multispectral_sat` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 2D+bands |
| Capital cost | $5000k |
| Operator skill | specialist |
| Solver latency | 2s |

**Mismatch parameters (4):** `b_r` (Band registration error (px)), `a_t` (Atmospheric transmittance (-)), `r_c` (Radiometric calibration (-)), `p_j` (Pointing jitter (px))

**Best solver:** FlowCompute — **PSNR:** 38.35 dB, **SSIM:** 0.98
**Worst PSNR:** 26.5 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 12

**Algorithm types:** Classical, Deep Learning, Diffusion, Generative, PnP, Vision Transformer

**Sample:** in-vivo capable

---

### Ocean Color Remote Sensing
**ID:** `ocean_color` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 10000.0 mm |
| Output dimensionality | 2D+bands |
| Capital cost | $3000k |
| Operator skill | specialist |
| Solver latency | 2s |

**Mismatch parameters (3):** `a_c` (Atmospheric correction error (-)), `s_g` (Sun glint contamination (-)), `v_c` (Vicarious calibration offset (-))

**Best solver:** AquaFormer — **PSNR:** 32.5 dB, **SSIM:** 0.91
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Passive Microwave Radiometry
**ID:** `passive_microwave` | **Carrier:** RF | **DAG:** `Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 100000.0 mm |
| Output dimensionality | 2D+freq |
| Capital cost | $5000k |
| Operator skill | specialist |
| Solver latency | 2s |

**Mismatch parameters (4):** `a_b` (Antenna beam width error (deg)), `r_g` (Receiver gain drift (-)), `b_t` (Brightness temperature offset (K)), `c_l` (Cross-polarization leakage (-))

**Best solver:** DiffusionSAR — **PSNR:** 35.42 dB, **SSIM:** 0.955
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Transformer

---

### Polarimetric SAR (PolSAR)
**ID:** `polsar` | **Carrier:** RF | **DAG:** `F → M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 2D+pol |
| Capital cost | $15000k |
| Operator skill | specialist |
| Solver latency | 15s |

**Mismatch parameters (3):** `c_b` (Cross-talk between polarizations (dB)), `c_i` (Channel imbalance (dB)), `f_r` (Faraday rotation (deg))

**Best solver:** DiffusionSAR — **PSNR:** 35.42 dB, **SSIM:** 0.955
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Transformer, Vision Transformer

---

### Radio Interferometry (VLBI)
**ID:** `radio_interferometry` | **Carrier:** RF | **DAG:** `F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 2D |
| Capital cost | $100000.0k |
| Operator skill | specialist |
| Solver latency | 3600s |

**Mismatch parameters (4):** `b_e` (Baseline error (m)), `p_c` (Phase calibration (deg)), `a_c` (Amplitude calibration (-)), `c_o` (Clock offset (ns))

**Best solver:** PRIMO — **PSNR:** 31.2 dB, **SSIM:** 0.905
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP

---

### Synthetic Aperture Radar (SAR)
**ID:** `sar` | **Carrier:** RF | **DAG:** `F → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 2D(complex) |
| Capital cost | $10000k |
| Operator skill | specialist |
| Solver latency | 10s |

**Best solver:** DiffusionSAR — **PSNR:** 35.42 dB, **SSIM:** 0.955
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 13 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Transformer, Vision Transformer

---

### Sonar Imaging
**ID:** `sonar` | **Carrier:** Acoustic | **DAG:** `P → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 1 |
| Spatial resolution | 100.0 mm |
| Output dimensionality | 2D/3D |
| Capital cost | $50k |
| Operator skill | technician |
| Solver latency | 2s |

**Best solver:** SwinIR — **PSNR:** 34.1 dB, **SSIM:** 0.942
**Worst PSNR:** 25.4 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** contact, in-vivo capable

---

### Weather / Doppler Radar
**ID:** `weather_radar` | **Carrier:** RF | **DAG:** `P → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 3D(reflectivity) |
| Capital cost | $1000k |
| Operator skill | technician |
| Solver latency | 0.5s |

**Mismatch parameters (4):** `c_b` (Calibration bias (dBZ)), `b_e` (Beam elevation error (deg)), `a_c` (Attenuation correction (-)), `g_c` (Ground clutter leakage (-))

**Best solver:** Earthformer — **PSNR:** 33.5 dB, **SSIM:** 0.935
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

---

## Scanning Probe (4 modalities)

### Atomic Force Microscopy (AFM)
**ID:** `afm` | **Carrier:** Mechanical | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 2D(topo) |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `t_s` (Tip shape convolution (-)), `p_n` (Piezo nonlinearity (-)), `t_d` (Thermal drift (nm/s)), `s_h` (Scanner hysteresis (-))

**Best solver:** DiffusionAFM — **PSNR:** 34.5 dB, **SSIM:** 0.94
**Worst PSNR:** 20.0 dB | **Algorithms in catalog:** 7 | **Benchmark results:** 7

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Self-Supervised, Transformer

**Sample:** contact

---

### Magnetic Force Microscopy (MFM)
**ID:** `mfm` | **Carrier:** Magnetic | **DAG:** `S → M → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 30 nm |
| Output dimensionality | 2D(magnetic) |
| Capital cost | $120k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (3):** `l_h` (Lift height (nm)), `t_m` (Tip magnetization model (-)), `e_c` (Electrostatic coupling (-))

**Best solver:** E2E-BTR — **PSNR:** 31.8 dB, **SSIM:** 0.908
**Worst PSNR:** 23.2 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** contact

---

### Near-field Scanning Optical Microscopy (NSOM)
**ID:** `nsom` | **Carrier:** Photon | **DAG:** `M → C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 50 nm |
| Output dimensionality | 2D+optical |
| Capital cost | $150k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `t_d` (Tip-sample distance (nm)), `a_s` (Aperture size error (-)), `t_c` (Topographic coupling (-)), `f_b` (Far-field background (-))

**Best solver:** E2E-BTR — **PSNR:** 31.8 dB, **SSIM:** 0.908
**Worst PSNR:** 23.2 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** contact

---

### Scanning Tunneling Microscopy (STM)
**ID:** `stm` | **Carrier:** Electron | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 2D(LDOS) |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `t_e` (Tip electronic structure (-)), `p_c` (Piezo creep (-)), `t_b` (Tunneling barrier height (eV)), `v_a` (Vibration amplitude (pm))

**Best solver:** E2E-BTR — **PSNR:** 31.8 dB, **SSIM:** 0.908
**Worst PSNR:** 23.2 dB | **Algorithms in catalog:** 10 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** contact

---

## Scientific Instrumentation (12 modalities)

### Atom Probe Tomography (APT)
**ID:** `atom_probe` | **Carrier:** Ion | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 3D+chem |
| Capital cost | $2000k |
| Operator skill | specialist |
| Solver latency | 300s |

**Mismatch parameters (4):** `f_p` (Flight path error (mm)), `v_c` (Voltage calibration (-)), `d_e` (Detection efficiency (-)), `t_r` (Tip radius error (nm))

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** destructive

---

### Cathodoluminescence (CL) Imaging
**ID:** `cathodoluminescence` | **Carrier:** Electron | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 10 nm |
| Output dimensionality | 2D+spec |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 5s |

**Mismatch parameters (4):** `b_c` (Beam current drift (-)), `c_e` (Collection efficiency variation (spatial)), `s_c` (Spectral calibration error (nm)), `c_c` (Carbon contamination (signal loss))

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** destructive

---

### Cryo-EM Single Particle Analysis
**ID:** `cryo_em` | **Carrier:** Electron | **DAG:** `C → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 3D |
| Capital cost | $3000k |
| Operator skill | specialist |
| Solver latency | 3600s |

**Mismatch parameters (4):** `d_e` (Defocus error (nm)), `a` (Astigmatism (nm)), `b_t` (Beam tilt (mrad)), `i_t` (Ice thickness variation (nm))

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** destructive

---

### MALDI Mass Spectrometry Imaging
**ID:** `maldi_msi` | **Carrier:** Ion | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D+mass |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (4):** `l_f` (Laser fluence drift (-)), `m_a` (Mass accuracy (ppm)), `e_d` (Extraction delay (ns)), `m_c` (Matrix crystallization (-))

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** destructive

---

### Muon Tomography
**ID:** `muon_tomo` | **Carrier:** Muon | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000000.0 |
| Max FPS | 0 |
| Spatial resolution | 1000.0 mm |
| Output dimensionality | 3D |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 3600s |

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

---

### Neutron Diffraction
**ID:** `neutron_diffraction` | **Carrier:** Neutron | **DAG:** `R → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 1 nm |
| Output dimensionality | 3D(n-density) |
| Capital cost | $100000.0k |
| Operator skill | specialist |
| Solver latency | 300s |

**Mismatch parameters (4):** `w_c` (Wavelength calibration (-)), `a_c` (Absorption correction (-)), `t_o` (Texture/preferred orientation (-)), `t_f` (TOF frame overlap (-))

**Best solver:** DiffFormer — **PSNR:** 32.5 dB, **SSIM:** 0.915
**Worst PSNR:** 23.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

---

### Neutron Radiography / Tomography
**ID:** `neutron_tomo` | **Carrier:** Neutron | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 50 µm |
| Output dimensionality | 3D |
| Capital cost | $100000.0k |
| Operator skill | specialist |
| Solver latency | 60s |

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

---

### Proton Radiography
**ID:** `proton_radiography` | **Carrier:** Proton | **DAG:** `Π → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 500 µm |
| Output dimensionality | 2D(RSP) |
| Capital cost | $10000k |
| Operator skill | specialist |
| Solver latency | 30s |

**Best solver:** pCT-Former — **PSNR:** 33.0 dB, **SSIM:** 0.92
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, PnP, Transformer

**Sample:** in-vivo capable

---

### Small-Angle X-ray Scattering (SAXS)
**ID:** `saxs` | **Carrier:** X-ray | **DAG:** `R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 10.0 mm |
| Output dimensionality | 1D/2D(recip) |
| Capital cost | $500k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `s_d` (Sample-detector distance (mm)), `b_c` (Beam center x (px)), `b_c` (Beam center y (px)), `w_e` (Wavelength error (nm))

**Best solver:** ScatterFormer — **PSNR:** 33.5 dB, **SSIM:** 0.925
**Worst PSNR:** 24.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Wide-Angle X-ray Scattering (WAXS)
**ID:** `waxs` | **Carrier:** X-ray | **DAG:** `R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10 |
| Spatial resolution | 1.0 mm |
| Output dimensionality | 1D/2D(recip) |
| Capital cost | $500k |
| Operator skill | expert |
| Solver latency | 2s |

**Mismatch parameters (4):** `d_d` (Detector distance error (-)), `b_c` (Beam center error (px)), `p_c` (Polarization correction (-)), `a_s` (Air scatter background (-))

**Best solver:** CrystalFormer — **PSNR:** 33.0 dB, **SSIM:** 0.92
**Worst PSNR:** 23.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### X-ray Crystallography
**ID:** `xray_crystallography` | **Carrier:** X-ray | **DAG:** `F → S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 0 nm |
| Output dimensionality | 3D(e-density) |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 60s |

**Mismatch parameters (4):** `c_o` (Crystal orientation error (deg)), `b_d` (Beam divergence (mrad)), `a_c` (Absorption correction (-)), `s_f` (Scale factor (-))

**Best solver:** CrystFormer — **PSNR:** 32.5 dB, **SSIM:** 0.915
**Worst PSNR:** 22.0 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### X-ray Fluorescence Tomography
**ID:** `xrf_tomo` | **Carrier:** X-ray | **DAG:** `Π → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100000.0 |
| Max FPS | 0 |
| Spatial resolution | 50 µm |
| Output dimensionality | 3D+element |
| Capital cost | $500k |
| Operator skill | specialist |
| Solver latency | 60s |

**Mismatch parameters (4):** `s_c` (Self-absorption correction (-)), `r_a` (Rotation axis offset (px)), `f_y` (Fluorescence yield error (-)), `d_t` (Dead time at high count rate (-))

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

## Spectroscopy (8 modalities)

### Brillouin Microscopy
**ID:** `brillouin` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+mech |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (3):** `b_s` (Brillouin shift calibration (MHz)), `v_f` (VIPA FSR error (-)), `e_s` (Elastic scattering leakage (-))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Coherent Anti-Stokes Raman (CARS) Microscopy
**ID:** `cars` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 30 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+spec |
| Capital cost | $300k |
| Operator skill | expert |
| Solver latency | 0.5s |

**Mismatch parameters (3):** `p_f` (Pump-Stokes frequency offset (cm^-1)), `n_b` (Non-resonant background (-)), `c_m` (Chirp mismatch (fs^2))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### DESI Mass Spectrometry Imaging
**ID:** `desi` | **Carrier:** Ion | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 100 µm |
| Output dimensionality | 2D+mass |
| Capital cost | $300k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `s_a` (Spray angle error (deg)), `s_f` (Solvent flow variation (-)), `i_s` (Ion suppression (matrix effect) (-)), `s_r` (Spatial resolution degradation (-))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** destructive

---

### FTIR Spectroscopic Imaging
**ID:** `ftir_imaging` | **Carrier:** IR | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 100 |
| Max FPS | 0 |
| Spatial resolution | 5 µm |
| Output dimensionality | 2D+IR-spec |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 5s |

**Mismatch parameters (4):** `w_c` (Wavenumber calibration (cm^-1)), `w_v` (Water vapor absorption (-)), `d_n` (Detector nonlinearity (-)), `a_c` (ATR crystal RI error (-))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

---

### Laser-Induced Breakdown Spectroscopy (LIBS) Imaging
**ID:** `libs` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 10 |
| Spatial resolution | 50 µm |
| Output dimensionality | 2D+element |
| Capital cost | $100k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (4):** `l_e` (Laser energy fluctuation (-)), `m_e` (Matrix effect (-)), `s_c` (Self-absorption correction (-)), `c_v` (Crater-to-crater variation (-))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Raman Imaging / Microscopy
**ID:** `raman_imaging` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 500 nm |
| Output dimensionality | 2D+spec |
| Capital cost | $200k |
| Operator skill | expert |
| Solver latency | 10s |

**Mismatch parameters (4):** `s_c` (Spectral calibration shift (cm^-1)), `f_b` (Fluorescence background (relative)), `l_p` (Laser power fluctuation (-)), `c_r` (Cosmic ray artifact (-))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Secondary Ion Mass Spectrometry (SIMS) Imaging
**ID:** `sims` | **Carrier:** Ion | **DAG:** `S → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 10000 |
| Max FPS | 0 |
| Spatial resolution | 50 nm |
| Output dimensionality | 2D+mass |
| Capital cost | $1000k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (4):** `m_c` (Mass calibration drift (ppm)), `m_e` (Matrix effect (sputter yield) (-)), `c_e` (Crater edge effect (-)), `c_(` (Charging (insulating samples) (V))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** destructive

---

### Stimulated Raman Scattering (SRS) Microscopy
**ID:** `srs` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 500 |
| Max FPS | 1 |
| Spatial resolution | 300 nm |
| Output dimensionality | 2D+lam |
| Capital cost | $300k |
| Operator skill | expert |
| Solver latency | 1s |

**Mismatch parameters (3):** `l_p` (Lock-in phase error (deg)), `c_m` (Cross-phase modulation (-)), `l_i` (Laser intensity noise (RIN) (dBc/Hz))

**Best solver:** Cascade-UNet — **PSNR:** 33.0 dB, **SSIM:** 0.922
**Worst PSNR:** 24.3 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

## Ultrafast (4 modalities)

### Compressed Ultrafast Photography (CUP)
**ID:** `cup` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 10,000,000,000 |
| Spatial resolution | 100 µm |
| Output dimensionality | 3D(x,y,t) |
| Capital cost | $80k |
| Operator skill | specialist |
| Solver latency | 30s |

**Mismatch parameters (3):** `d_e` (DMD encoding error (-)), `s_s` (Streak sweep calibration (-)), `t_c` (Temporal-spatial coupling (-))

**Best solver:** AL-DL — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.6 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### Pump-Probe Microscopy
**ID:** `pump_probe` | **Carrier:** Photon | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1000 |
| Max FPS | 1,000,000,000,000,000 |
| Spatial resolution | 10 µm |
| Output dimensionality | 2D+t |
| Capital cost | $300k |
| Operator skill | specialist |
| Solver latency | 10s |

**Mismatch parameters (4):** `t_d` (Time-zero drift (fs)), `p_p` (Pump power fluctuation (-)), `c_(` (Chirp (GDD) (fs^2)), `s_o` (Spatial overlap error (-))

**Best solver:** DynFormer — **PSNR:** 32.0 dB, **SSIM:** 0.905
**Worst PSNR:** 22.5 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

### Streak Camera Imaging
**ID:** `streak_camera` | **Carrier:** Photon | **DAG:** `M → Σ → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 1,000,000,000,000 |
| Spatial resolution | 50 µm |
| Output dimensionality | 1D+t |
| Capital cost | $200k |
| Operator skill | specialist |
| Solver latency | 0.01s |

**Mismatch parameters (4):** `s_n` (Sweep nonlinearity (-)), `t_r` (Temporal resolution (ps)), `d_r` (Dynamic range saturation (-)), `t_j` (Trigger jitter (ps))

**Best solver:** AL-DL — **PSNR:** 33.4 dB, **SSIM:** 0.93
**Worst PSNR:** 24.6 dB | **Algorithms in catalog:** 11 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Deep Unrolling, Diffusion, PnP, Score-based, Vision Transformer

**Sample:** in-vivo capable

---

### XFEL Serial Femtosecond Crystallography (SFX)
**ID:** `xfel_sfx` | **Carrier:** X-ray | **DAG:** `M → R → D`

| Property | Value |
|----------|-------|
| Shots per datacube | 1 |
| Max FPS | 120 |
| Spatial resolution | 0 nm |
| Output dimensionality | 3D(e-density) |
| Capital cost | $500000.0k |
| Operator skill | specialist |
| Solver latency | 3600s |

**Mismatch parameters (4):** `h_r` (Hit rate (-)), `i_a` (Indexing ambiguity (-)), `p_m` (Partiality model error (-)), `b_f` (Background from jet/carrier (-))

**Best solver:** CalibFormer — **PSNR:** 32.8 dB, **SSIM:** 0.92
**Worst PSNR:** 24.1 dB | **Algorithms in catalog:** 4 | **Benchmark results:** 4

**Algorithm types:** Classical, Deep Learning, Transformer

**Sample:** in-vivo capable

---

## Cross-Modality Comparison

### Top 10 by Best PSNR

| Rank | Modality | PSNR (dB) | SSIM | Best Method |
|------|----------|-----------|------|-------------|
| 1 | Arterial Spin Labeling (ASL) MRI | 40.45 | 0.938 | MRDynamo |
| 2 | Diffusion MRI (DTI) | 40.45 | 0.938 | MRDynamo |
| 3 | Functional MRI (BOLD fMRI) | 40.45 | 0.938 | MRDynamo |
| 4 | MR Elastography (MRE) | 40.45 | 0.938 | MRDynamo |
| 5 | MR Angiography (MRA) | 40.45 | 0.938 | MRDynamo |
| 6 | Magnetic Resonance Imaging (MRI) | 40.45 | 0.938 | MRDynamo |
| 7 | MR Spectroscopy (MRS) | 40.45 | 0.938 | MRDynamo |
| 8 | Susceptibility-Weighted Imaging (SWI) | 40.45 | 0.938 | MRDynamo |
| 9 | X-ray Computed Tomography (CT) | 40.15 | 0.985 | CTFlow |
| 10 | X-ray Angiography | 39.92 | 0.984 | Score-CT |

### 10 Most Affordable Systems

| Rank | Modality | Capital Cost | Best PSNR | Operator |
|------|----------|-------------|-----------|----------|
| 1 | Lensless (Diffuser Camera) Imaging | $0.5k | 33.5 dB | untrained |
| 2 | Panorama Multi-Focus Fusion | $0.5k | 35.0 dB | untrained |
| 3 | 3D Gaussian Splatting (3DGS) | $1k | 35.85 dB | untrained |
| 4 | High Dynamic Range (HDR) Imaging | $1k | 38.82 dB | untrained |
| 5 | Neural Radiance Fields (NeRF) | $1k | 35.85 dB | untrained |
| 6 | Photometric Stereo | $2k | 34.2 dB | technician |
| 7 | Single-Pixel Camera (SPC) | $2k | 38.58 dB | technician |
| 8 | Time-of-Flight Depth Camera | $2k | 34.0 dB | untrained |
| 9 | Coded Exposure / Flutter Shutter | $3k | 38.82 dB | technician |
| 10 | Event Camera / Dynamic Vision Sensor (DVS) | $5k | 33.0 dB | technician |

### 10 Most Expensive Systems

| Rank | Modality | Capital Cost | Best PSNR | Operator |
|------|----------|-------------|-----------|----------|
| 1 | Gravitational Wave Detection | $1000000.0k | 30.5 dB | specialist |
| 2 | Event Horizon Telescope (EHT) Imaging | $500000.0k | 31.2 dB | specialist |
| 3 | XFEL Serial Femtosecond Crystallography (SFX) | $500000.0k | 32.8 dB | specialist |
| 4 | Neutron Diffraction | $100000.0k | 32.5 dB | specialist |
| 5 | Neutron Radiography / Tomography | $100000.0k | 32.8 dB | specialist |
| 6 | Particle Calorimetry | $100000.0k | 31.5 dB | specialist |
| 7 | Radio Interferometry (VLBI) | $100000.0k | 31.2 dB | specialist |
| 8 | Radio Aperture Synthesis | $50000k | 31.2 dB | specialist |
| 9 | Interferometric SAR (InSAR) | $15000k | 33.0 dB | specialist |
| 10 | Polarimetric SAR (PolSAR) | $15000k | 35.42 dB | specialist |

### Single-Shot Systems

| Modality | Max FPS | Resolution | Capital Cost | Best PSNR |
|----------|---------|------------|-------------|-----------|
| Brachytherapy Imaging | 1 | 500 µm | $500k | 39.92 dB |
| Dual-Energy X-ray Absorptiometry (DEXA) | 1 | 500 µm | $50k | 39.92 dB |
| Fluoroscopy | 30 | 200 µm | $300k | 39.92 dB |
| Mammography | 2 | 70 µm | $200k | 39.92 dB |
| Portal Imaging (EPID) | 15 | 400 µm | $200k | 39.92 dB |
| X-ray NDT (Radiography) | 10 | 100 µm | $50k | 39.92 dB |
| X-ray Radiography | 30 | 100 µm | $50k | 39.92 dB |
| Coded Exposure / Flutter Shutter | 100 | 5 µm | $3k | 38.82 dB |
| Coded Aperture Compressive Temporal Imaging (CACTI) | 100,000,000 | 5 µm | $15k | 38.58 dB |
| Coded Aperture Snapshot Spectral Imaging (CASSI) | 30 | 5 µm | $20k | 38.58 dB |
| Dark-Field Microscopy | 30 | 500 nm | $50k | 38.48 dB |
| Expansion Microscopy (ExM) | 1 | 70 nm | $50k | 38.48 dB |
| Polarization Microscopy | 30 | 5 µm | $20k | 38.48 dB |
| TIRF Microscopy | 100 | 200 nm | $100k | 38.48 dB |
| Widefield Fluorescence Microscopy | 100 | 300 nm | $30k | 38.48 dB |
| Low-Dose Widefield Microscopy | 100 | 300 nm | $30k | 38.48 dB |
| Multispectral Satellite Imaging | 0 | 1000.0 mm | $5000k | 38.35 dB |
| Optical Coherence Tomography (OCT) | 100,000 | 5 µm | $80k | 37.95 dB |
| Hyperspectral Remote Sensing | 0 | 500.0 mm | $500k | 36.8 dB |
| Contrast-Enhanced Ultrasound (CEUS) | 30 | 300 µm | $80k | 36.28 dB |
| Doppler Ultrasound | 50 | 300 µm | $50k | 36.28 dB |
| Intravascular Ultrasound (IVUS) | 30 | 100 µm | $200k | 36.28 dB |
| Ultrasound B-mode Imaging | 100 | 300 µm | $30k | 36.28 dB |
| Digital Holographic Microscopy | 100 | 300 nm | $50k | 35.82 dB |
| Integral Photography | 10 | 10 µm | $200k | 35.8 dB |
| Acoustic Emission Testing (AE) | 1,000,000 | 5.0 mm | $10k | 35.5 dB |
| Active Thermography (IR) | 100 | 500 µm | $30k | 35.5 dB |
| Light Field Imaging | 30 | 10 µm | $5k | 35.5 dB |
| Passive Microwave Radiometry | 0 | 100000.0 mm | $5000k | 35.42 dB |
| Polarimetric SAR (PolSAR) | 0 | 1000.0 mm | $15000k | 35.42 dB |
| Synthetic Aperture Radar (SAR) | 0 | 1000.0 mm | $10000k | 35.42 dB |
| Adaptive Optics (AO) Imaging | 1,000 | 500 nm | $1000k | 35.0 dB |
| Phase Contrast Microscopy | 100 | 300 nm | $60k | 35.0 dB |
| Machine Vision / AOI | 100 | 10 µm | $5k | 34.8 dB |
| Electron Holography | 1 | 0 nm | $2000k | 34.5 dB |
| Ultrasonic Phased Array (TFM/FMC) | 100 | 300 µm | $30k | 34.5 dB |
| US/MRI Fusion | 10 | 500 µm | $100k | 34.3 dB |
| Fundus Camera | 10 | 10 µm | $20k | 34.2 dB |
| Bioluminescence Tomography (BLT) | 0 | 1.0 mm | $100k | 34.1 dB |
| Shearography | 30 | 100 µm | $40k | 34.0 dB |
| Time-of-Flight Depth Camera | 30 | 1.0 mm | $2k | 34.0 dB |
| Differential Interference Contrast (DIC) | 100 | 200 nm | $80k | 33.5 dB |
| Lensless (Diffuser Camera) Imaging | 30 | 50 µm | $0.5k | 33.5 dB |
| Small-Angle X-ray Scattering (SAXS) | 10 | 10.0 mm | $500k | 33.5 dB |
| Weather / Doppler Radar | 0 | 1000.0 mm | $1000k | 33.5 dB |
| Compressed Ultrafast Photography (CUP) | 10,000,000,000 | 100 µm | $80k | 33.4 dB |
| Streak Camera Imaging | 1,000,000,000,000 | 50 µm | $200k | 33.4 dB |
| Transmission Electron Microscopy (TEM) | 10 | 0 nm | $3000k | 33.4 dB |
| Confocal Laser Endomicroscopy (CLE) | 30 | 1 µm | $100k | 33.2 dB |
| Fiber Bundle Endoscopy | 30 | 50 µm | $30k | 33.2 dB |
| Flash LiDAR | 30 | 10.0 mm | $20k | 33.2 dB |
| Coherent Anti-Stokes Raman (CARS) Microscopy | 30 | 300 nm | $300k | 33.0 dB |
| 4D-STEM Electron Diffraction | 1 | 0 nm | $2000k | 33.0 dB |
| Event Camera / Dynamic Vision Sensor (DVS) | 1,000,000 | 10 µm | $5k | 33.0 dB |
| Wide-Angle X-ray Scattering (WAXS) | 10 | 1.0 mm | $500k | 33.0 dB |
| XFEL Serial Femtosecond Crystallography (SFX) | 120 | 0 nm | $500000.0k | 32.8 dB |
| Ocean Color Remote Sensing | 0 | 10000.0 mm | $3000k | 32.5 dB |
| Particle Calorimetry | 40,000,000 | 100.0 mm | $100000.0k | 31.5 dB |
| Solar EUV/X-ray Imaging | 10 | 100 µm | $2000k | 31.2 dB |
| Gravitational Wave Detection | 16,000 | -1000 nm | $1000000.0k | 30.5 dB |
| Functional Near-Infrared Spectroscopy (fNIRS) | 10 | 10.0 mm | $30k | 30.5 dB |

### Carrier Distribution

| Carrier | Count | Modalities |
|---------|-------|------------|
| Photon | 72 | adaptive_optics, bioluminescence_tomo, brillouin, cacti, cars, ... +67 more |
| X-ray | 20 | angiography, cbct, ct, ct_fluorescence, dexa, ... +15 more |
| Electron | 14 | cathodoluminescence, cryo_em, cryo_et, ebsd, edx_mapping, ... +9 more |
| Acoustic | 12 | acoustic_emission, acoustic_microscopy, ceus, doppler_ultrasound, elastography, ... +7 more |
| Spin/RF | 10 | asl_mri, cest_mri, diffusion_mri, fmri, mr_elastography, ... +5 more |
| RF | 9 | eht_imaging, gpr, insar, passive_microwave, polsar, ... +4 more |
| Ion | 4 | atom_probe, desi, maldi_msi, sims |
| Gamma | 4 | pet, pet_mr, spect, spect_ct |
| IR | 2 | active_thermography, ftir_imaging |
| Magnetic | 2 | magnetic_particle, mfm |
| Neutron | 2 | neutron_diffraction, neutron_tomo |
| Proton | 2 | proton_radiography, proton_therapy_img |
| Mechanical | 1 | afm |
| Gamma/X-ray | 1 | brachytherapy_img |
| EM | 1 | eddy_current |
| Seismic/Acoustic | 1 | fwi |
| Gravitational | 1 | gravitational_wave |
| Electric | 1 | impedance_tomo |
| Muon | 1 | muon_tomo |
| Particle | 1 | particle_calorimetry |
| Photon/Electron | 1 | phase_retrieval |
| MV | 1 | portal_imaging |
| Electron/Photon | 1 | ptychography |
| Seismic | 1 | seismic_tomo |
| Photon/EUV | 1 | solar_imaging |
| THz | 1 | terahertz |
| Photon/IR | 1 | tof_camera |
