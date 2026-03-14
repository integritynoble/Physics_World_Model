# How spec.md Designs Any Imaging System

## The Core Question

If users want to design any imaging system, how can a 7-field schema with 11 primitives satisfy designing any imaging modality?

## The Answer: Composition over a Finite Primitive Basis

Despite 170+ modalities looking very different physically, they all do the same thing -- **encode an object into measurements through a sequence of linear operations**:

```
Object → [transform chain] → Measurements + Noise
```

The companion paper (Paper II) proved the **Finite Primitive Basis (FPB) theorem**: every imaging forward model across 5 carrier families can be decomposed into a composition of 11 canonical primitives. The difference between modalities is just which primitives appear, in what order, and with what parameters.

---

## The 7-Field Schema

```yaml
modality:       # What system? → names the DAG template
carrier:        # What physical carrier? → constrains which primitives apply
geometry:       # What measurement geometry? → sets primitive parameters
object:         # What are we imaging? → defines domain constraints
forward_model:  # What is A? → the primitive composition chain
noise:          # What corrupts measurements? → defines data-fidelity term
target:         # What quality do we need? → guides algorithm selection
```

Each field serves a distinct role in going from intent to math:

| Field | Determines | Used by |
|---|---|---|
| `modality` | Which DAG template to instantiate | Plan Agent |
| `carrier` | Physical constraints (wavelength, coherence, speed) | Primitive selection |
| `geometry` | All numerical parameters (angles, coils, pixels, mask) | Operator construction |
| `object` | Domain of x (size, non-negativity, sparsity basis) | Regularization choice |
| `forward_model` | The operator A = P_n ... P_1 | Forward/adjoint implementation |
| `noise` | The likelihood p(y|x) | Data-fidelity term in cost function |
| `target` | Acceptable reconstruction quality | Stopping criterion, algorithm tier |

---

## The 11 Primitives

These are the **atomic operations** that appear across all known imaging physics:

| What physics does | Primitive | Symbol | Math | Examples |
|---|---|---|---|---|
| Integrate along lines | Projection | Pi | Line integrals of attenuation | CT, PET, SPECT |
| Multiply by a pattern | Modulation | M | Element-wise multiply | Coil maps, coded apertures, SLM patterns, structured illumination |
| Transform to frequency domain | Fourier encoding | F | FFT / partial Fourier | MRI k-space encoding, Fourier ptychography |
| Keep subset of data | Sampling | S | Subsampling operator P_Omega | MRI undersampling, compressed sensing |
| Blur by a kernel | Convolution | C | PSF convolution | Microscopy PSF, lens aberrations, diffraction |
| Propagate waves | Propagation | P | Fresnel / angular spectrum | Holography, ptychography, OCT, ultrasound |
| Shift by wavelength | Dispersion | W | Wavelength-dependent spatial shift | CASSI, spectrometers, prism-based systems |
| Sum over an axis | Accumulation | Sigma | Sum over spectral/temporal axis | Snapshot spectral/temporal compression |
| Bounce/scatter | Reflection | R | Scattering / backscatter operator | Ultrasound, DOT, seismic imaging |
| Convert to detectable signal | Detection | D | Intensity, amplitude, interferometric | All modalities (final stage) |
| Corrupt the signal | Noise | N | Poisson, Gaussian, mixed | All modalities |

---

## Three Worked Examples

### (a) Clinical CT

```yaml
modality: computed_tomography
carrier: xray
geometry: parallel_beam, n_angles=128
object: 128x128 image, non-negative
forward_model: Radon(Pi) -> Detect(D, intensity)
noise: Poisson, I_0=1e4
target: PSNR >= 30dB
```

**Mathematical derivation from spec.md:**

```
y = Poisson( I_0 * exp( -R_theta * x ) )
```

- `Pi` (Radon projection): `[R_theta * x]_{i,j} = integral along ray(i,j) of x(r) dr` -- line integrals at 128 angles
- `D` (Detect, intensity): Beer-Lambert law converts attenuation to photon counts
- `noise: Poisson, I_0=1e4`: shot noise on detected intensity

**Reconstruction**: Solve `x_hat = argmin_x ||A(x) - y||^2 + lambda * TV(x)` where A is the Radon transform.

### (b) CASSI (Spectral Imaging)

```yaml
modality: cassi
carrier: photon
geometry: coded_aperture + disperser, lambda=[400,700]nm
object: 256x256x28 spectral cube
forward_model: Modulate(M) -> Disperse(W) -> Accumulate(Sigma) -> Detect(D)
noise: Gaussian, sigma=0.01
target: PSNR >= 28dB
```

**Mathematical derivation from spec.md:**

```
y(r,c) = sum_{k=1}^{28} M(r,c) * x(r, c - k*Delta, lambda_k) + n
```

- `M` (Modulate): element-wise multiply by coded aperture mask M in {0,1}^{256x256}
- `W` (Disperse): shift band k by k*Delta pixels along detector axis
- `Sigma` (Accumulate): sum all 28 shifted/modulated bands into one 2D measurement
- `D` (Detect): integrate photons; `noise: Gaussian, sigma=0.01`

**Reconstruction**: GAP-TV or TwIST with 3D-TV regularization to separate overlapping bands.

### (c) Accelerated MRI

```yaml
modality: mri
carrier: spin
geometry: cartesian_kspace, acceleration=4x
object: 256x256 complex image
forward_model: Modulate(M, coil) -> Encode(F, kspace) -> Sample(S) -> Detect(D)
noise: Gaussian, SNR=30dB
target: SSIM >= 0.9
```

**Mathematical derivation from spec.md:**

```
y_c = P_Omega * F * (S_c .* x) + n_c,   c = 1, ..., N_coils
```

- `M` (Modulate, coil): multiply image by coil sensitivity S_c
- `F` (Encode, kspace): 2D discrete Fourier transform
- `S` (Sample): apply undersampling mask P_Omega (keep 1/4 of k-space lines)
- `D` (Detect): measure complex k-space; `noise: Gaussian, SNR=30dB`

**Reconstruction**: CG-SENSE or FISTA with wavelet/TV regularization.

---

## How Composition Creates New Modalities

A user designs a new modality by composing primitives in a new order. No new math is needed -- just a new DAG:

| Modality | Forward Model Chain | What it does |
|---|---|---|
| CT | `Pi -> D` | Project, then detect |
| PET | `Pi -> D` | Same chain, different carrier (positron) |
| MRI | `M -> F -> S -> D` | Modulate by coils, Fourier encode, subsample, detect |
| CASSI | `M -> W -> Sigma -> D` | Mask, disperse, accumulate bands, detect |
| CACTI | `M -> Sigma -> D` | Mask temporal frames, accumulate, detect |
| Lensless | `C -> D` | Convolve by PSF, detect |
| Holography | `P -> D` | Propagate wavefront, detect intensity |
| Ptychography | `M -> P -> D` | Modulate by probe, propagate, detect |
| SIM | `M -> C -> D` | Structured illumination, convolve, detect |
| DOT | `M -> R -> P -> R -> D` | Modulate source, scatter, propagate, scatter, detect |
| OCT | `P + P -> Sigma -> D` | Two propagation paths (reference + sample), interfere, detect |
| Ultrasound | `P -> R -> P -> D` | Transmit pulse, reflect off tissue, propagate back, detect |
| Light field | `M -> C -> S -> D` | Microlens modulate, convolve, subsample, detect |
| Electron ptycho | `M -> P -> D` | Probe modulate, propagate (electron), detect |

**Novel system design example:**

- **Existing**: Lensless imaging = `C -> D`
- **New idea**: Lensless + coded aperture = `M -> C -> D` (add a mask before the PSF)
- **Another idea**: Snapshot spectral lensless = `M -> W -> C -> Sigma -> D` (add dispersion and accumulation)

The user doesn't derive new math -- they just:
1. Identify which physical operations their system performs
2. Map each to a primitive
3. Write the chain in `forward_model`
4. Fill in the parameters in `geometry`

---

## From spec.md to Reconstruction: The Full Pipeline

```
spec.md  ──→  Plan Agent  ──→  6-Gate Compiler  ──→  Forward Model A, A^T  ──→  Reconstruction
   │              │                   │                        │                      │
   │         Generates           Validates:              Constructs:            Solves:
   │         spec.md          1. Schema check          A(x) from chain      x_hat = argmin
   │                          2. Primitive check       A^T(y) (adjoint)     ||A(x)-y||^2
   │                          3. DAG acyclicity        Adjoint test:        + lambda*R(x)
   │                          4. Lipschitz bound       <Ax,y> = <x,A^Ty>
   │                          5. Param bounds
   │                          6. FPB coverage
   │
   └── 7 fields map to every piece of information needed for reconstruction
```

A user reading `forward_model: M -> F -> S -> D` can:
1. Map each symbol to its mathematical operator
2. Compose them left-to-right to get the full forward model A
3. Derive the adjoint A^T by reversing the chain and transposing each operator
4. Plug into any optimization framework: `x_hat = argmin_x ||A(x) - y||^2 + lambda * R(x)`

---

## What About Truly Novel Physics?

The paper acknowledges limits (Definition 1 -- "designable" scope):

### Covered by the tier system
Each primitive has 4 fidelity tiers:
- **Tier-0** (geometric): ray optics, ideal sampling
- **Tier-1** (Fourier): diffraction, bandwidth limits
- **Tier-2** (shift-variant): spatially varying PSF, field inhomogeneity
- **Tier-3** (full-physics): multiple scattering, nonlinear interactions

A new modality might start at Tier-1 and tier-lift to Tier-3 as physics models improve. The tier-lifting protocol (Proposition 2 in the paper) provides a formal way to upgrade fidelity while preserving the DAG structure.

### The 5 carrier families cover all known imaging
X-ray, photon, spin, acoustic, electron/particle. If a new carrier were discovered, a new primitive might be needed -- but this hasn't happened across the 170 modalities surveyed.

### Current limitations
- **Nonlinear forward models**: The framework assumes the chain is approximately linear or linearizable. Strongly nonlinear systems (e.g., deep tissue scattering in DOT) have large epsilon_unmod at lower tiers.
- **Coupled multi-physics**: Systems where two carriers interact (e.g., photoacoustic = photon + acoustic) require multi-DAG composition, which is supported but less validated.

---

## Validation: The Expert Study

The expert study directly validates that spec.md is **mathematically sufficient**:

- 5 independent methods received the same spec.md for 3 modalities
- All 5 derived the same forward model A from the specification
- All 5 achieved similar reconstruction quality (inter-method PSNR CoV < 5.7%)
- The small variation confirms: **the specification determines the quality, not the algorithm**

This means spec.md contains enough information for any user to:
1. Read the 7 fields
2. Derive the forward model mathematically
3. Implement reconstruction
4. Achieve quality comparable to expert-tuned libraries

---

## Summary

```
spec.md works for "any" imaging system because:

1. Physics constrains the design space -- all imaging is:
   object -> [linear operator chain] -> measurements + noise

2. 11 primitives span the operator space -- proved across 170 modalities,
   5 carrier families (companion paper)

3. 7 fields separate concerns -- modality/carrier/geometry set the DAG,
   object/noise/target guide reconstruction

4. Composition creates new modalities -- novel systems are new DAGs
   of existing primitives, not new math

5. Tier system handles fidelity -- same DAG, upgradeable physics
```
