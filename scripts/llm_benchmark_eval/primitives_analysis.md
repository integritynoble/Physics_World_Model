# Primitives Analysis — 11 Primitives Cover All 65 Imaging Modalities

## Summary

The Physics World Model uses exactly **11 primitives** to express every imaging modality. Each primitive is a physics building block; the modality-specific behavior comes from the **operator** (parameter) inside it. Currently 125 unique operators are used across 65 modalities, but each primitive can support hundreds more.

---

## The 11 Primitives

| # | Primitive | Symbol | Meaning | Variants Using | Unique Operators |
|---|-----------|--------|---------|----------------|-----------------|
| 1 | **C** | `C(·)` | Convolution / Point Spread Function | 19 | 15 |
| 2 | **D** | `D(·)` | Detector (gain + noise) | 65 | 40+ |
| 3 | **F** | `F(·)` | Fourier / k-space sampling | 6 | 5 |
| 4 | **M** | `M(·)` | Mask / spatial modulation | 6 | 5 |
| 5 | **P** | `P(·)` | Propagation | 22 | 15 |
| 6 | **R** | `R(·)` | Rotation | 5 | 3 |
| 7 | **S** | `S(·)` | Structured illumination | 4 | 4 |
| 8 | **W** | `W(·)` | Wavelength dispersion | 1 | 1 |
| 9 | **Π** | `Π(·)` | Projection | 19 | 16 |
| 10 | **Σ** | `Σ(·)` | Summation / integration | 24 | 19 |
| 11 | **Λ** | `Λ(·)` | Energy selection / filtering | 2 | 2 |

**Total unique operators in use: ~125**

---

## Operators per Primitive

### C — Convolution / PSF (19 variants)

| Operator (params) | Label | Variants |
|--------------------|-------|----------|
| `PSF` | PSF Convolution | widefield, widefield_lowdose, sim, flim, polarization |
| `PSF` | Single-Molecule PSF | palm_storm |
| `PSF_2P` | Two-Photon PSF | two_photon |
| `PSF_3D` | 3D Confocal PSF | confocal_3d |
| `PSF_NA` | Low-NA PSF | fpm |
| `PSF_STED` | STED Effective PSF | sted |
| `PSF_TIRF` | Evanescent-Field PSF | tirf |
| `PSF_confocal` | Confocal PSF | confocal_livecell |
| `PSF_fiber` | Fiber Bundle PSF | endoscopy |
| `PSF_focus` | Depth-Dependent PSF | panorama |
| `PSF_optic` | Ophthalmic PSF | fundus |
| `PSF_sheet` | Light-Sheet PSF | lightsheet |
| `CTF` | Contrast Transfer Function | tem |
| `probe` | Probe Formation / Scanning | stem, sem |

### D — Detector (65 variants — universal)

Every modality ends with a detector. The operator specifies gain `g` and noise level `η`:

| Noise Level | Detector Types | Variants |
|-------------|---------------|----------|
| `η₁` | CCD, CMOS, RF Coil, Flat-Panel, SPAD, Radar, Sensor Array, ... | 42 variants |
| `η₂` | Piezo Array, Hydrophone, Ultrasound Transducer | 4 variants (acoustic) |
| `η₃` | PMT, EMCCD, sCMOS, Gamma Camera, Scintillation, TCSPC, APD | 17 variants (photon-counting) |
| `η₄` | Coded-aperture Detector | 2 variants (cacti, sd_cassi) |

### P — Propagation (22 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `acoustic` | Acoustic Propagation | ultrasound, doppler_ultrasound, photoacoustic, sonar |
| `e⁻` | Electron Beam / Wave / Biprism | tem, sem, stem, ebsd, eels, electron_holography, electron_tomography, electron_diffraction |
| `Fresnel` | Fresnel Propagation | holography |
| `diffuse` | Diffuse Propagation | dot |
| `diffuser` | Diffuser Propagation | lensless |
| `far-field` | Far-Field Propagation | phase_retrieval |
| `low-coherence` | Low-Coherence Source | oct, octa |
| `modulated` | Modulated Light | tof_camera |
| `probe` | Probe Illumination | ptychography |
| `pulsed` | Pulsed Laser | lidar |
| `shear` | Shear-Wave Propagation | elastography |

### Π — Projection (19 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `fan` | Fan-Beam Projection | ct |
| `cone` | Cone-Beam Projection | cbct |
| `proj` | X-ray / Generic Projection | xray_radiography, angiography, fluoroscopy, dexa, electron_tomography |
| `contact` | Contact Projection | mammography |
| `parallel` | Parallel-Hole Collimator | spect |
| `LOR` | Line-of-Response | pet |
| `ray` | Ray Casting | nerf |
| `splat` | Gaussian Splatting | gaussian_splatting |
| `micro-lens` | Micro-Lens Array | light_field |
| `lens-array` | Lens Array | integral |
| `triangulation` | Triangulation | structured_light |
| `neutron` | Neutron Attenuation | neutron_tomo |
| `proton` | Proton Transmission | proton_radiography |
| `muon` | Muon Scattering | muon_tomo |
| `backscatter` | Kikuchi Pattern | ebsd |

### Σ — Summation / Integration (24 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `t` | Temporal Integration | ultrasound, doppler_ultrasound, photoacoustic, pet, fluoroscopy, elastography, cacti, fmri, sonar, flim |
| `λ` | Spectral Sum | sd_cassi |
| `θ` | Angular Sum | fpm |
| `φ` | Phase-Shift Sum | sim |
| `f` | Focus Stack Sum | panorama |
| `E` | Energy Window Sum | spect |
| `interference` | Interferometric Sum | oct, octa, electron_holography |
| `volume` | Volume Rendering | nerf |
| `alpha` | Alpha Compositing | gaussian_splatting |
| `correlation` | Correlation Integration | tof_camera |
| `return` | Return Signal Integration | lidar |
| *(empty)* | Spatial / Boundary Sum | spc_block, dot |

### F — Fourier / k-Space Sampling (6 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `k-traj` | k-Space Sampling | mri |
| `EPI` | Echo-Planar Imaging | fmri, diffusion_mri |
| `FID` | Free Induction Decay | mrs |
| `diffraction` | Diffraction Pattern | electron_diffraction |
| `azimuth×range` | Range-Doppler | sar |

### M — Mask / Modulation (6 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `mask` | Coded Aperture | sd_cassi |
| `Φ` | Sensing Matrix | spc_block, matrix |
| `H⊗W` | Kronecker Sensing | spc_kronecker |
| `m_t` | Temporal Mask | cacti |
| `polarizer` | Polarizer / Analyzer | polarization |

### R — Rotation (5 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `θ` | Gantry Rotation | ct, cbct |
| `θ` | Sample Rotation / Tilt | neutron_tomo, electron_tomography |
| `θ_cosmic` | Cosmic Muon Incidence | muon_tomo |

### S — Structured Illumination (4 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `LED` | LED Array Illumination | fpm |
| `block` | Block Illumination | spc_block |
| `grating` | Sinusoidal Illumination | sim |
| `pattern` | Projected Pattern | structured_light |

### W — Wavelength Dispersion (1 variant)

| Operator | Label | Variants |
|----------|-------|----------|
| `α, a` | Prism Dispersion | sd_cassi |

### Λ — Energy Selection (2 variants)

| Operator | Label | Variants |
|----------|-------|----------|
| `E₁,E₂` | Dual-Energy Selection | dexa |
| `energy` | Energy Disperser | eels |

---

## All 65 Modality Specs

| Variant | Spec (Primitive Chain) |
|---------|----------------------|
| angiography | `Π(proj) → D(g, η₁)` |
| cacti | `M(m_t) → Σ_t → D(g, η₄)` |
| cbct | `R(θ) → Π(cone) → D(g, η₁)` |
| confocal_3d | `C(PSF_3D) → D(g, η₃)` |
| confocal_livecell | `C(PSF_confocal) → D(g, η₃)` |
| ct | `R(θ) → Π(fan) → D(g, η₁)` |
| dexa | `Λ(E₁,E₂) → Π(proj) → D(g, η₁)` |
| diffusion_mri | `F(EPI) → D(g, η₁)` |
| doppler_ultrasound | `P(acoustic) → Σ_t → D(g, η₂)` |
| dot | `P(diffuse) → Σ → D(g, η₃)` |
| ebsd | `P(e⁻) → Π(backscatter) → D(g, η₁)` |
| eels | `P(e⁻) → Λ(energy) → D(g, η₁)` |
| elastography | `P(shear) → Σ_t → D(g, η₂)` |
| electron_diffraction | `P(e⁻) → F(diffraction) → D(g, η₁)` |
| electron_holography | `P(e⁻) → Σ(interference) → D(g, η₁)` |
| electron_tomography | `R(θ) → P(e⁻) → Π(proj) → D(g, η₁)` |
| endoscopy | `C(PSF_fiber) → D(g, η₁)` |
| flim | `C(PSF) → Σ_t → D(g, η₃)` |
| fluoroscopy | `Π(proj) → Σ_t → D(g, η₁)` |
| fmri | `F(EPI) → Σ_t → D(g, η₁)` |
| fpm | `S(LED array) → C(PSF_NA) → Σ_θ → D(g, η₁)` |
| fundus | `C(PSF_optic) → D(g, η₁)` |
| gaussian_splatting | `Π(splat) → Σ(alpha) → D(g, η₁)` |
| holography | `P(Fresnel) → D(g, η₁)` |
| integral | `Π(lens-array) → D(g, η₁)` |
| lensless | `P(diffuser) → D(g, η₁)` |
| lidar | `P(pulsed) → Σ(return) → D(g, η₁)` |
| light_field | `Π(micro-lens) → D(g, η₁)` |
| lightsheet | `C(PSF_sheet) → D(g, η₃)` |
| mammography | `Π(contact) → D(g, η₁)` |
| matrix | `M(Φ) → D(g, η₁)` |
| mri | `F(k-traj) → D(g, η₁)` |
| mrs | `F(FID) → D(g, η₁)` |
| muon_tomo | `R(θ_cosmic) → Π(muon) → D(g, η₁)` |
| nerf | `Π(ray) → Σ(volume) → D(g, η₁)` |
| neutron_tomo | `R(θ) → Π(neutron) → D(g, η₁)` |
| oct | `P(low-coherence) → Σ(interference) → D(g, η₁)` |
| octa | `P(low-coherence) → Σ(interference) → D(g, η₁)` |
| palm_storm | `C(PSF) → D(g, η₃)` |
| panorama | `C(PSF_focus) → Σ_f → D(g, η₁)` |
| pet | `Π(LOR) → Σ_t → D(g, η₃)` |
| phase_retrieval | `P(far-field) → D(g, η₁)` |
| photoacoustic | `P(acoustic) → Σ_t → D(g, η₂)` |
| polarization | `M(polarizer) → C(PSF) → D(g, η₁)` |
| proton_radiography | `Π(proton) → D(g, η₁)` |
| ptychography | `P(probe) → D(g, η₁)` |
| sar | `F(azimuth×range) → D(g, η₁)` |
| sd_cassi | `M(mask) → W(α, a) → Σ_λ → D(g, η₄)` |
| sem | `P(e⁻ beam) → C(probe) → D(g, η₁)` |
| sim | `S(grating) → C(PSF) → Σ_φ → D(g, η₃)` |
| sonar | `P(acoustic) → Σ_t → D(g, η₂)` |
| spc_block | `S(block) → M(Φ) → Σ → D(g, η₁)` |
| spc_kronecker | `M(H⊗W) → D(g, η₁)` |
| spect | `Π(parallel) → Σ_E → D(g, η₃)` |
| sted | `C(PSF_STED) → D(g, η₃)` |
| stem | `P(e⁻) → C(probe) → D(g, η₁)` |
| structured_light | `S(pattern) → Π(triangulation) → D(g, η₁)` |
| tem | `P(e⁻) → C(CTF) → D(g, η₁)` |
| tirf | `C(PSF_TIRF) → D(g, η₃)` |
| tof_camera | `P(modulated) → Σ(correlation) → D(g, η₁)` |
| two_photon | `C(PSF_2P) → D(g, η₃)` |
| ultrasound | `P(acoustic) → Σ_t → D(g, η₂)` |
| widefield | `C(PSF) → D(g, η₃)` |
| widefield_lowdose | `C(PSF) → D(g, η₃)` |
| xray_radiography | `Π(proj) → D(g, η₁)` |

---

## Key Observations

1. **Every modality is a chain of 2–4 primitives.** The shortest specs have 2 (e.g., `Π(proj) → D(g, η₁)` for X-ray radiography); the longest have 4 (e.g., `S(LED array) → C(PSF_NA) → Σ_θ → D(g, η₁)` for FPM).

2. **D is universal** — every spec ends with a detector. The noise level (η₁–η₄) groups detectors by physics domain.

3. **Operators carry the modality-specific physics.** The primitive `P` (Propagation) covers X-rays, electrons, acoustic waves, photons, and neutrons — the operator (`e⁻`, `acoustic`, `Fresnel`, etc.) specifies which.

4. **125 operators today, thousands possible.** Each primitive can support hundreds of operators. For example:
   - `C(·)` — Gaussian, Airy, Moffat, Zernike, motion blur, atmospheric turbulence, Bessel beam, ...
   - `P(·)` — plane wave, Gaussian beam, fiber mode, terahertz, gamma ray, positron, ...
   - `Π(·)` — helical, laminography, limited-angle, sparse-angle, interior tomography, ...

5. **W and Λ are the least used** (1 and 2 variants). They could potentially be merged into a generalized "spectral/energy filtering" primitive, but keeping them separate preserves physical clarity.

---

## Proposed Workflow: Prompt-Driven Modality Design

```
User prompt: "I want to image brain tissue with two-photon excitation
              and add a deformable mirror for adaptive optics"

Step 1 — PWM recognizes base modality:
         two_photon → C(PSF_2P) → D(g, η₃)

Step 2 — PWM retrieves operators from primitives database:
         C primitive has operators: PSF_2P, PSF_2P_AO (adaptive optics variant)
         Suggests: C(PSF_2P_AO) → D(g, η₃)

Step 3 — User refines: "Also add temporal gating for FLIM"
         PWM reorganizes: C(PSF_2P_AO) → Σ_t → D(g, η₃)

Step 4 — Final spec returned with full operator metadata
```

Each operator in the database would include:
- Mathematical definition (transfer function / matrix)
- Physical parameters and their ranges
- Compatible primitives it can chain with
- Reference implementations in the PWM codebase
