# PWM Imaging Modality Standards

This document defines the standard datasets, imaging configurations, and classical reconstruction algorithms for each of the 17 supported imaging modalities. These standards are designed to ensure reproducibility and alignment with academic benchmarks.

---

## Table of Contents

1. [Widefield Microscopy](#1-widefield-microscopy)
2. [Widefield Low-Dose](#2-widefield-low-dose)
3. [Confocal Live-Cell](#3-confocal-live-cell)
4. [Confocal 3D Stack](#4-confocal-3d-stack)
5. [SIM (Structured Illumination Microscopy)](#5-sim-structured-illumination-microscopy)
6. [CASSI (Coded Aperture Spectral Imaging)](#6-cassi-coded-aperture-spectral-imaging)
7. [SPC (Single-Pixel Camera)](#7-spc-single-pixel-camera)
8. [CACTI (Video Snapshot Compressive Imaging)](#8-cacti-video-snapshot-compressive-imaging)
9. [Lensless / DiffuserCam](#9-lensless--diffusercam)
10. [Light-Sheet Microscopy](#10-light-sheet-microscopy)
11. [CT (Computed Tomography)](#11-ct-computed-tomography)
12. [MRI (Magnetic Resonance Imaging)](#12-mri-magnetic-resonance-imaging)
13. [Ptychography](#13-ptychography)
14. [Holography](#14-holography)
15. [NeRF (Neural Radiance Fields)](#15-nerf-neural-radiance-fields)
16. [3D Gaussian Splatting](#16-3d-gaussian-splatting)
17. [Matrix Operator (Generic)](#17-matrix-operator-generic)

---

## 1. Widefield Microscopy

### Standard Datasets
| Dataset | Resolution | Description | Download |
|---------|------------|-------------|----------|
| **Fluorescence Microscopy Deconvolution (FMD)** | 512×512 | Synthetic + real fluorescence images | [GitHub](https://github.com/zj-dong/FMD) |
| **BioSR** | 512×512 | Super-resolution microscopy benchmark | [Zenodo](https://zenodo.org/record/5654567) |
| **Widefield2SIM** | 512×512 | Paired widefield/SIM data | [Harvard Dataverse](https://dataverse.harvard.edu) |
| **DeconvolutionLab2** | Various | Standard test images (Ely, CElegans) | [EPFL](http://bigwww.epfl.ch/deconvolution/) |

### Imaging Configuration
```yaml
physics:
  modality: widefield
  psf:
    type: gaussian  # or measured PSF
    sigma: 2.0      # pixels (NA-dependent)
    size: 31        # PSF kernel size
  noise:
    shot_noise: poisson
    read_noise: 5.0  # electrons
```

### Classical Algorithms
| Algorithm | Reference | Implementation |
|-----------|-----------|----------------|
| **Richardson-Lucy (RL)** | Richardson 1972, Lucy 1974 | scikit-image, DeconvolutionLab2 |
| **Wiener Filter** | Wiener 1949 | scipy, MATLAB |
| **ADMM-TV** | Boyd 2011 | Custom, ProxTV |
| **Landweber** | Landweber 1951 | DeconvolutionLab2 |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "richardson_lucy", "params": {"iters": 50}},
      {"id": "pnp_hqs", "params": {"denoiser": "bm3d", "iters": 30}}
    ]
  }
}
```

---

## 2. Widefield Low-Dose

### Standard Datasets
| Dataset | Description | Download |
|---------|-------------|----------|
| **W2S (Widefield2STORM)** | Low-SNR widefield images | [Nature Methods](https://www.nature.com/articles/s41592-021-01236-x) |
| **FMD Low-Photon** | Simulated low-photon data | [GitHub](https://github.com/zj-dong/FMD) |

### Imaging Configuration
```yaml
physics:
  modality: widefield
  budget:
    max_photons: 100  # Very low photon count
  noise:
    shot_noise: poisson
    read_noise: 10.0
    background: 50.0  # High background
```

### Classical Algorithms
| Algorithm | Reference |
|-----------|-----------|
| **PURE-LET** | Luisier 2010 |
| **VST + BM3D** | Makitalo 2012 |
| **Noise2Noise** | Lehtinen 2018 |
| **CARE** | Weigert 2018 |

---

## 3. Confocal Live-Cell

### Standard Datasets
| Dataset | Resolution | Description | Download |
|---------|------------|-------------|----------|
| **Cell Image Library** | Various | Diverse cell images | [CIL](http://www.cellimagelibrary.org/) |
| **Allen Cell Explorer** | 3D stacks | Live cell imaging | [Allen Institute](https://www.allencell.org/) |
| **ISBI Cell Tracking** | Time-lapse | Motion tracking benchmark | [ISBI](http://celltrackingchallenge.net/) |

### Imaging Configuration
```yaml
physics:
  modality: confocal
  psf:
    sigma: 1.5  # Sharper than widefield
  sample:
    motion:
      type: drift
      rate_px_per_frame: 0.1
  budget:
    max_photons: 500  # Limited for live cell
```

### Classical Algorithms
| Algorithm | Reference |
|-----------|-----------|
| **Deconvolution + Motion Correction** | Huygens, Imaris |
| **Drift Correction** | Cross-correlation based |
| **CARE (Content-Aware)** | Weigert 2018 |

---

## 4. Confocal 3D Stack

### Standard Datasets
| Dataset | Size | Description | Download |
|---------|------|-------------|----------|
| **BBBC (Broad Institute)** | 3D stacks | Cell biology benchmarks | [Broad](https://bbbc.broadinstitute.org/) |
| **OpenCell** | 3D confocal | Protein localization | [OpenCell](https://opencell.czbiohub.org/) |

### Imaging Configuration
```yaml
physics:
  modality: confocal
  dims: [512, 512, 64]  # H, W, Z
  psf:
    axial_sigma: 3.0  # Elongated axially
    lateral_sigma: 1.0
  attenuation:
    model: exponential
    coefficient: 0.03
```

### Classical Algorithms
| Algorithm | Reference |
|-----------|-----------|
| **3D Richardson-Lucy** | McNally 1999 |
| **Iterative Constrained** | Rosen 2001 |
| **Multi-View Fusion** | Preibisch 2014 |

---

## 5. SIM (Structured Illumination Microscopy)

### Standard Datasets
| Dataset | Resolution | Patterns | Download |
|---------|------------|----------|----------|
| **BioSR** | 512×512 | 9 (3×3) | [Zenodo](https://zenodo.org/record/5654567) |
| **SIMData** | 256×256 | 9, 15 | [GitHub](https://github.com/HenriquesLab) |
| **FairSIM Test** | 256×256 | 9 | [FairSIM](https://github.com/fairSIM/fairSIM) |

### Imaging Configuration
```yaml
physics:
  modality: sim
  patterns:
    n_angles: 3      # Orientation angles
    n_phases: 3      # Phase shifts per angle
    frequency: 0.1   # Pattern frequency (cycles/px)
    mod_depth: 0.8   # Modulation depth
  psf:
    sigma: 1.5       # Detection PSF
```

### Classical Algorithms
| Algorithm | Reference | Implementation |
|-----------|-----------|----------------|
| **Wiener-SIM** | Gustafsson 2000 | fairSIM, OpenSIM |
| **HiFi-SIM** | Huang 2018 | MATLAB |
| **ML-SIM** | Christensen 2021 | Python/TensorFlow |
| **JSFR-SIM** | Jin 2020 | MATLAB |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "wiener_sim", "params": {"wiener_param": 0.001}},
      {"id": "pnp_hqs", "params": {"iters": 50}}
    ]
  }
}
```

---

## 6. CASSI (Coded Aperture Spectral Imaging)

### Standard Datasets
| Dataset | Size | Bands | Description | Download |
|---------|------|-------|-------------|----------|
| **KAIST** | 512×512 | 31 | Real HSI scenes | [KAIST](http://vclab.kaist.ac.kr/siggraphasia2017p1/) |
| **ICVL** | 1392×1300 | 31 | Natural hyperspectral | [ICVL](http://icvl.cs.bgu.ac.il/hyperspectral/) |
| **CAVE** | 512×512 | 31 | Indoor objects | [Columbia](https://www.cs.columbia.edu/CAVE/databases/multispectral/) |
| **Harvard** | 1392×1040 | 31 | Indoor/outdoor | [Harvard](http://vision.seas.harvard.edu/hyperspec/) |
| **ARAD_1K** | 482×512 | 31 | NTIRE challenge | [CVPR Workshop](https://codalab.lisn.upsaclay.fr/) |

### Imaging Configuration
```yaml
physics:
  modality: cassi
  dims: [256, 256, 28]  # H, W, L (spectral bands)
  mask:
    type: binary_random  # or coded aperture
    density: 0.5
  dispersion:
    model: linear  # or polynomial
    shift_per_band: 2  # pixels
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **TwIST** | Bioucas-Dias 2007 | MATLAB |
| **GAP-TV** | Yuan 2016 | [GitHub](https://github.com/yuanxy92/GAP-TV) |
| **ADMM** | Boyd 2011 | Custom |
| **DeSCI** | Liu 2018 | [GitHub](https://github.com/liuyang12/DeSCI) |
| **TSA-Net** | Meng 2020 | [GitHub](https://github.com/mengziyi64/TSA-Net) |
| **MST** | Cai 2022 | [GitHub](https://github.com/caiyuanhao1998/MST) |
| **CST** | Cai 2022 | [GitHub](https://github.com/caiyuanhao1998/MST) |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "gap_tv", "params": {"lambda": 0.01, "iters": 100}},
      {"id": "admm_tv", "params": {"rho": 1.0, "iters": 50}}
    ]
  }
}
```

---

## 7. SPC (Single-Pixel Camera)

### Standard Datasets
| Dataset | Images | Resolution | Description | Download |
|---------|--------|------------|-------------|----------|
| **Set11** | 11 | 256×256 | Classic CS benchmark | Included in most CS papers |
| **BSD68** | 68 | 481×321 | Denoising benchmark | [Berkeley](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/) |
| **Set14** | 14 | Various | Super-resolution | Standard benchmark |
| **Urban100** | 100 | Various | Urban scenes | [GitHub](https://github.com/jbhuang0604/SelfExSR) |

### Imaging Configuration
```yaml
physics:
  modality: spc
  dims: [64, 64]  # or 256×256
  measurement:
    pattern_type: hadamard  # hadamard, gaussian, binary
    sampling_rate: 0.25     # 25% measurements
    n_measurements: 1024    # For 64×64 image at 25%
```

### Pattern Types
| Pattern | Properties | Usage |
|---------|------------|-------|
| **Hadamard** | Orthogonal, ±1 | Most common, best SNR |
| **Gaussian** | Random, continuous | RIP guarantee |
| **Binary** | Random 0/1 | DMD-friendly |
| **Fourier** | Structured | Frequency domain |

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **OMP** | Tropp 2007 | sklearn |
| **ISTA/FISTA** | Beck 2009 | Custom |
| **TVAL3** | Li 2013 | [MATLAB](https://www.caam.rice.edu/~optimization/L1/TVAL3/) |
| **D-AMP** | Metzler 2016 | [GitHub](https://github.com/ricedsp/D-AMP_Toolbox) |
| **ISTA-Net** | Zhang 2018 | [GitHub](https://github.com/jianzhangcs/ISTA-Net) |
| **ReconNet** | Kulkarni 2016 | [GitHub](https://github.com/KulkarniLab/ReconNet) |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "ista_net", "params": {"layers": 9}},
      {"id": "pnp_hqs", "params": {"denoiser": "bm3d", "iters": 50, "sigma": 0.15}}
    ]
  }
}
```

---

## 8. CACTI (Video Snapshot Compressive Imaging)

### Standard Datasets (6 Benchmark Videos)
| Video | Resolution | Frames | Description | Download |
|-------|------------|--------|-------------|----------|
| **Kobe** | 256×256×8 | 8 | Basketball player | [GitHub](https://github.com/mq0829/DUN-3DUnet) |
| **Traffic** | 256×256×8 | 8 | Highway traffic | Same |
| **Runner** | 256×256×8 | 8 | Running athlete | Same |
| **Drop** | 256×256×8 | 8 | Water droplet | Same |
| **Crash** | 256×256×8 | 8 | Car crash | Same |
| **Aerial** | 256×256×8 | 8 | Aerial view | Same |

### Additional Datasets
| Dataset | Description | Download |
|---------|-------------|----------|
| **DAVIS** | Video segmentation | [DAVIS](https://davischallenge.org/) |
| **UCF101** | Action recognition | [UCF](https://www.crcv.ucf.edu/data/UCF101.php) |

### Imaging Configuration
```yaml
physics:
  modality: cacti
  dims: [256, 256, 8]  # H, W, T (8 frames standard)
  mask:
    type: shifting_binary
    shift_type: vertical  # vertical shift per frame
    density: 0.5
  compression_ratio: 8  # 8 frames → 1 snapshot
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **GAP-TV** | Yuan 2016 | [GitHub](https://github.com/yuanxy92/GAP-TV) |
| **DeSCI** | Liu 2018 | [GitHub](https://github.com/liuyang12/DeSCI) |
| **PnP-FFDNet** | Yuan 2020 | [GitHub](https://github.com/yuanxy92/PnP-SCI) |
| **E2E-CNN** | Qiao 2020 | [GitHub](https://github.com/mq0829/E2E-CNN) |
| **STFormer** | Wang 2022 | [GitHub](https://github.com/ucaswangls/STFormer) |
| **EfficientSCI** | Wang 2023 | [GitHub](https://github.com/ucaswangls/EfficientSCI) |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "gap_tv", "params": {"lambda": 0.01, "iters": 100}},
      {"id": "pnp_ffdnet", "params": {"iters": 50}}
    ]
  }
}
```

---

## 9. Lensless / DiffuserCam

### Standard Datasets
| Dataset | Resolution | Description | Download |
|---------|------------|-------------|----------|
| **DiffuserCam** | 270×480 | Real lensless captures | [Waller Lab](https://waller-lab.github.io/DiffuserCam/) |
| **FlatCam** | 256×256 | Amplitude mask camera | [Rice DSP](https://ricedsp.github.io/) |
| **PhlatCam** | 512×512 | Phase mask lensless | [CMU](https://www.cs.cmu.edu/~motoMDK/) |

### Imaging Configuration
```yaml
physics:
  modality: lensless
  dims: [256, 256]
  psf:
    type: diffuser  # or phase_mask, amplitude_mask
    model: measured  # Use measured PSF
    # Or simulated:
    # model: random_phase
    # feature_size: 8  # pixels
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **ADMM** | Boyd 2011 | [DiffuserCam](https://github.com/Waller-Lab/DiffuserCam) |
| **Gradient Descent** | — | Standard |
| **FlatNet** | Khan 2020 | [GitHub](https://github.com/vccimaging/FlatNet) |
| **U-Net Lensless** | Monakhova 2019 | [GitHub](https://github.com/Waller-Lab/LenslessLearning) |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "admm", "params": {"rho": 1.0, "iters": 100}},
      {"id": "pnp_hqs", "params": {"denoiser": "bm3d", "iters": 50}}
    ]
  }
}
```

---

## 10. Light-Sheet Microscopy

### Standard Datasets
| Dataset | Description | Download |
|---------|-------------|----------|
| **OpenSPIM** | Multi-view light-sheet | [OpenSPIM](https://openspim.org/) |
| **Zeiss Z.1** | Commercial light-sheet | Instrument data |
| **ClearMap** | Cleared tissue imaging | [GitHub](https://github.com/ChristophKirst/ClearMap) |

### Imaging Configuration
```yaml
physics:
  modality: lightsheet
  dims: [512, 512, 200]  # Large 3D volume
  psf:
    axial_sigma: 1.0   # Thin sheet
    lateral_sigma: 1.5
  artifacts:
    stripes:
      enabled: true
      strength: 0.2
    attenuation:
      model: exponential
      coefficient: 0.02
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **Multi-View Fusion** | Preibisch 2010 | [Fiji/BigStitcher](https://imagej.net/plugins/bigstitcher/) |
| **Stripe Removal** | Münch 2009 | [Fiji](https://imagej.net/plugins/remove-stripes) |
| **CARE** | Weigert 2018 | [GitHub](https://github.com/CSBDeep/CSBDeep) |
| **Destripe** | Liu 2022 | [GitHub](https://github.com/peng-lab/destripe) |

---

## 11. CT (Computed Tomography)

### Standard Datasets
| Dataset | Size | Views | Description | Download |
|---------|------|-------|-------------|----------|
| **AAPM Low-Dose CT** | 512×512 | Full | Grand Challenge 2016 | [AAPM](https://www.aapm.org/GrandChallenge/LowDoseCT/) |
| **Mayo Clinic** | 512×512 | Various | Clinical CT | [TCIA](https://www.cancerimagingarchive.net/) |
| **LoDoPaB-CT** | 362×362 | Various | Low-dose benchmark | [Zenodo](https://zenodo.org/record/3384092) |
| **Walnut** | 501×501 | 1200 | Micro-CT | [Zenodo](https://zenodo.org/record/2686726) |

### Imaging Configuration
```yaml
physics:
  modality: ct
  dims: [512, 512]
  geometry:
    type: fan_beam  # or parallel, cone_beam
    n_angles: 180   # Full: 720, Sparse: 60-180
    detector_count: 512
  dose:
    level: low  # low, quarter, full
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **FBP** | Feldkamp 1984 | ASTRA, ODL |
| **ART/SART** | Andersen 1984 | ASTRA |
| **ADMM-TV** | Sidky 2008 | Custom |
| **FBPConvNet** | Jin 2017 | [GitHub](https://github.com/panakino/FBPConvNet) |
| **RED-CNN** | Chen 2017 | [GitHub](https://github.com/SSinyu/RED-CNN) |
| **LEARN** | Chen 2018 | Custom |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "fbp", "params": {"filter": "ram-lak"}},
      {"id": "sart", "params": {"iters": 20}},
      {"id": "pnp_hqs", "params": {"denoiser": "bm3d", "iters": 30}}
    ]
  }
}
```

---

## 12. MRI (Magnetic Resonance Imaging)

### Standard Datasets
| Dataset | Size | Coils | Description | Download |
|---------|------|-------|-------------|----------|
| **fastMRI** | 320×320 | 15 | Knee, Brain MRI | [fastMRI](https://fastmri.org/) |
| **Calgary-Campinas** | 256×256 | 12 | Brain MRI | [GitHub](https://sites.google.com/view/calgary-campinas-dataset) |
| **IXI** | 256×256 | 1 | Brain MRI | [IXI](https://brain-development.org/ixi-dataset/) |
| **OASIS** | 256×256 | 1 | Brain MRI | [OASIS](https://www.oasis-brains.org/) |

### Imaging Configuration
```yaml
physics:
  modality: mri
  dims: [320, 320]
  acquisition:
    trajectory: cartesian  # cartesian, radial, spiral
    acceleration: 4        # 4x undersampling
    acs_lines: 24          # Auto-calibration signal
  coils:
    n_coils: 8
    sensitivity: estimated  # or measured
```

### Sampling Patterns
| Pattern | Description | Usage |
|---------|-------------|-------|
| **Cartesian** | Uniform + random | Standard |
| **Variable Density** | Dense center | Compressed sensing |
| **Radial** | Star pattern | Motion robust |
| **Spiral** | Spiral trajectory | Fast imaging |

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **SENSE** | Pruessmann 1999 | BART |
| **GRAPPA** | Griswold 2002 | BART |
| **ESPIRiT** | Uecker 2014 | [BART](https://mrirecon.github.io/bart/) |
| **L1-ESPIRiT** | — | BART |
| **VarNet** | Sriram 2020 | [fastMRI](https://github.com/facebookresearch/fastMRI) |
| **E2E-VarNet** | Sriram 2020 | Same |

### Recommended PWM Config
```json
{
  "recon": {
    "solvers": [
      {"id": "espirit", "params": {"maps": "estimated"}},
      {"id": "pnp_hqs", "params": {"denoiser": "bm3d", "iters": 30}}
    ]
  }
}
```

---

## 13. Ptychography

### Standard Datasets
| Dataset | Resolution | Positions | Download |
|---------|------------|-----------|----------|
| **PtychoNN** | 256×256 | 16-64 | [GitHub](https://github.com/mcherukara/PtychoNN) |
| **Synthetic Ptycho** | Various | Various | Generated |
| **APS Data** | Real X-ray | — | [APS](https://www.aps.anl.gov/) |

### Imaging Configuration
```yaml
physics:
  modality: ptychography
  dims: [256, 256]
  scan:
    n_positions: 64
    overlap: 0.7  # 70% overlap
    pattern: grid  # or spiral, random
  probe:
    size: 64
    type: gaussian  # or measured
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **ePIE** | Maiden 2009 | [GitHub](https://github.com/AdvancedPhotonSource/ptychography) |
| **rPIE** | Maiden 2017 | Same |
| **RAAR** | Luke 2005 | Custom |
| **Adam-ptycho** | Nashed 2017 | [GitHub](https://github.com/AdvancedPhotonSource/ptychonn) |
| **PtychoNN** | Cherukara 2020 | [GitHub](https://github.com/mcherukara/PtychoNN) |

---

## 14. Holography

### Standard Datasets
| Dataset | Description | Download |
|---------|-------------|----------|
| **DIV2K** | High-res images (simulation) | [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) |
| **USAF Targets** | Resolution targets | Standard |
| **Holography Benchmark** | Phase objects | Custom |

### Imaging Configuration
```yaml
physics:
  modality: holography
  dims: [512, 512]
  setup:
    type: off_axis  # off_axis, inline, phase_shifting
    wavelength: 632.8e-9  # HeNe laser
    carrier_freq: 0.2     # cycles/pixel
    reference_angle: 5    # degrees
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **Angular Spectrum** | Goodman 2005 | Custom |
| **Fresnel Propagation** | — | Custom |
| **Phase Shifting** | Yamaguchi 1997 | Custom |
| **TIE** | Teague 1983 | [GitHub](https://github.com/bionanoimaging/TIE) |
| **Gerchberg-Saxton** | Gerchberg 1972 | Custom |
| **HoloNet** | Wu 2018 | [GitHub](https://github.com/Waller-Lab) |

---

## 15. NeRF (Neural Radiance Fields)

### Standard Datasets
| Dataset | Scenes | Views | Resolution | Download |
|---------|--------|-------|------------|----------|
| **Synthetic-NeRF** | 8 | 100 | 800×800 | [NeRF](https://github.com/bmild/nerf) |
| **LLFF** | 8 | 20-60 | 1008×756 | [LLFF](https://github.com/Fyusion/LLFF) |
| **DTU** | 124 | 49-64 | 1600×1200 | [DTU](https://roboimagedata.compute.dtu.dk/) |
| **Mip-NeRF 360** | 9 | 100+ | Various | [GitHub](https://github.com/google-research/multinerf) |
| **Tanks & Temples** | 14 | Many | HD | [T&T](https://www.tanksandtemples.org/) |

### Imaging Configuration
```yaml
physics:
  modality: nerf
  dims: [400, 400, 128]  # H, W, D
  rendering:
    n_views: 100
    n_samples: 64  # samples per ray
    near: 2.0
    far: 6.0
  camera:
    model: pinhole
    poses: provided
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **NeRF** | Mildenhall 2020 | [GitHub](https://github.com/bmild/nerf) |
| **Instant-NGP** | Müller 2022 | [GitHub](https://github.com/NVlabs/instant-ngp) |
| **Plenoxels** | Yu 2022 | [GitHub](https://github.com/sxyu/plenoxel) |
| **TensoRF** | Chen 2022 | [GitHub](https://github.com/apchenstu/TensoRF) |
| **Nerfacto** | Tancik 2023 | [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) |

---

## 16. 3D Gaussian Splatting

### Standard Datasets
| Dataset | Description | Download |
|---------|-------------|----------|
| **Mip-NeRF 360** | Unbounded scenes | [GitHub](https://github.com/google-research/multinerf) |
| **Tanks & Temples** | Large-scale | [T&T](https://www.tanksandtemples.org/) |
| **Deep Blending** | Indoor | [DB](https://repo-sam.inria.fr/fungraph/deep-blending/) |

### Imaging Configuration
```yaml
physics:
  modality: gaussian_splatting
  dims: [400, 400, 128]
  rendering:
    n_views: 100
    n_gaussians: 100000  # Initial count
  optimization:
    densify_interval: 100
    opacity_reset: 3000
```

### Classical Algorithms
| Algorithm | Reference | Code |
|-----------|-----------|------|
| **3D Gaussian Splatting** | Kerbl 2023 | [GitHub](https://github.com/graphdeco-inria/gaussian-splatting) |
| **Mip-Splatting** | Yu 2023 | [GitHub](https://github.com/autonomousvision/mip-splatting) |
| **2D Gaussian Splatting** | Huang 2024 | [GitHub](https://github.com/hbb1/2d-gaussian-splatting) |

---

## 17. Matrix Operator (Generic)

### Standard Datasets
| Problem | Matrix Type | Size | Download |
|---------|-------------|------|----------|
| **Deblurring** | Toeplitz | n×n | Custom |
| **Inpainting** | Sampling | m×n | Custom |
| **Compressed Sensing** | Gaussian | m×n | Generated |
| **Tomography** | Radon | m×n² | Generated |

### Imaging Configuration
```yaml
physics:
  modality: matrix
  operator:
    kind: matrix
    source: path/to/A.npy
  dims: [64, 64]
```

### Classical Algorithms
| Algorithm | Reference | Usage |
|-----------|-----------|-------|
| **CG** | Hestenes 1952 | Positive definite |
| **LSQR** | Paige 1982 | General least squares |
| **ADMM** | Boyd 2011 | With regularization |
| **Proximal GD** | Parikh 2014 | Composite objectives |

---

## Summary Table

| # | Modality | Standard Dataset | Resolution | Key Algorithm |
|---|----------|------------------|------------|---------------|
| 1 | Widefield | FMD, BioSR | 512×512 | Richardson-Lucy |
| 2 | Widefield Low-Dose | W2S | 512×512 | VST+BM3D |
| 3 | Confocal Live-Cell | Cell Image Library | 512×512 | RL + Motion Corr |
| 4 | Confocal 3D | BBBC | 512×512×64 | 3D RL |
| 5 | SIM | BioSR, SIMData | 256×256×9 | Wiener-SIM |
| 6 | CASSI | KAIST, ICVL | 256×256×28 | GAP-TV, TSA-Net |
| 7 | SPC | Set11, BSD68 | 256×256 | ISTA-Net |
| 8 | CACTI | 6 Videos | 256×256×8 | GAP-TV, STFormer |
| 9 | Lensless | DiffuserCam | 256×256 | ADMM |
| 10 | Light-Sheet | OpenSPIM | 512×512×200 | Multi-View Fusion |
| 11 | CT | AAPM, Mayo | 512×512 | FBP, SART |
| 12 | MRI | fastMRI | 320×320 | ESPIRiT, VarNet |
| 13 | Ptychography | PtychoNN | 256×256 | ePIE |
| 14 | Holography | DIV2K | 512×512 | Angular Spectrum |
| 15 | NeRF | Synthetic-NeRF | 800×800 | NeRF, Instant-NGP |
| 16 | 3DGS | Mip-NeRF 360 | HD | 3D Gaussian Splatting |
| 17 | Matrix | Custom | Various | CG, ADMM |

---

## Implementation Priority

### Phase 1: Core Algorithms (Immediate)
1. **Richardson-Lucy** - Widefield/Confocal
2. **GAP-TV** - CASSI/CACTI
3. **ISTA-Net** - SPC
4. **FBP/SART** - CT
5. **ESPIRiT** - MRI

### Phase 2: Advanced Methods
1. **Wiener-SIM** - SIM reconstruction
2. **ePIE** - Ptychography
3. **Angular Spectrum** - Holography
4. **Multi-View Fusion** - Light-Sheet

### Phase 3: Deep Learning
1. **TSA-Net/MST** - CASSI
2. **STFormer** - CACTI
3. **VarNet** - MRI
4. **PtychoNN** - Ptychography
5. **NeRF/3DGS** - Novel view synthesis

---

## References

Key papers for each modality are listed in the algorithm tables above. For comprehensive reviews:

1. **Computational Imaging**: Mait et al., "Computational imaging", Advances in Optics and Photonics, 2018
2. **Compressed Sensing**: Candès & Wakin, "An Introduction To Compressive Sampling", IEEE SPM, 2008
3. **Deep Learning for Imaging**: Ongie et al., "Deep Learning Techniques for Inverse Problems in Imaging", IEEE JSTSP, 2020
