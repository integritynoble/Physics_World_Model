# 01 — MRI Physics Fundamentals

## 1. Nuclear Magnetic Resonance (NMR)

### 1.1 Proton Spin and Magnetic Moment

Atomic nuclei with an odd number of protons or neutrons possess **spin angular
momentum**. Hydrogen-1 (¹H) — the most abundant nucleus in the body (water,
fat) — has spin-½. When placed in an external magnetic field **B₀**, the
spins align either parallel (low energy) or anti-parallel (high energy).

The net magnetisation **M** arises from the slight population excess in the
parallel state. It is this net magnetisation that MRI detects.

### 1.2 The Larmor Equation

A nucleus in a field B₀ precesses at the **Larmor frequency**:

```
ω₀ = γ · B₀
```

where **γ** is the gyromagnetic ratio. For ¹H:

```
γ / 2π = 42.577 MHz/T
```

| Field Strength | Larmor Frequency |
|---------------|------------------|
| 1.5 T         | 63.87 MHz        |
| 3.0 T         | 127.73 MHz       |
| 7.0 T         | 298.03 MHz       |

The Larmor equation is the foundation of spatial encoding — by adding
spatially varying gradient fields, we make the precession frequency depend on
position.

---

## 2. The Main Magnetic Field (B₀)

The superconducting magnet produces a strong, static field **B₀** along the
bore axis (conventionally the z-axis). Key characteristics:

- **Strength**: clinical scanners use 1.5 T or 3.0 T; research scanners up
  to 7 T and beyond.
- **Homogeneity**: specified in parts per million (ppm) over a defined
  sphere. Typical spec: < 1 ppm over a 40 cm DSV (diameter spherical
  volume).
- **Inhomogeneity effects**: off-resonance → signal dephasing, geometric
  distortion. The PWM benchmark models this as `B0_inhomog_hz` ranging
  from 5 Hz (mild) to 60 Hz (severe) — see
  `datasets/benchmark/mri/build_dataset.py`, `SPEC_RANGES`.

---

## 3. Radiofrequency (RF) Excitation

### 3.1 The RF Pulse

To observe the nuclear signal we must tip **M** away from the z-axis into
the transverse (x-y) plane. This is done by transmitting a short burst of
RF energy at the Larmor frequency through a **B₁ coil**.

The flip angle α is:

```
α = γ · ∫ B₁(t) dt
```

Common flip angles:
- **90°**: maximum transverse magnetisation (spin-echo, gradient-echo)
- **180°**: inversion pulse (inversion recovery, refocusing in SE)
- **Small angles** (5–30°): fast gradient-echo sequences (FLASH, SPGR)

### 3.2 Slice Selection

Applying a linear gradient **Gₛ** along the slice direction (z) during the
RF pulse makes the resonance condition spatially selective:

```
ω(z) = γ · (B₀ + Gₛ · z)
```

Only spins within the bandwidth of the RF pulse are excited. The slice
thickness is:

```
Δz = BW_rf / (γ · Gₛ)
```

---

## 4. Relaxation

After excitation the magnetisation returns to equilibrium through two
independent processes.

### 4.1 T1 Relaxation (Spin-Lattice)

The longitudinal magnetisation Mz recovers exponentially:

```
Mz(t) = M₀ · (1 − e^(−t/T1))
```

- **Mechanism**: energy transfer from spins to the surrounding lattice.
- **Typical brain values at 3 T**:
  - Grey matter: ~1600 ms
  - White matter: ~800 ms
  - CSF: ~4000 ms

### 4.2 T2 Relaxation (Spin-Spin)

The transverse magnetisation Mxy decays exponentially:

```
Mxy(t) = Mxy(0) · e^(−t/T2)
```

- **Mechanism**: loss of phase coherence due to spin-spin interactions.
- **Typical brain values at 3 T**:
  - Grey matter: ~80 ms
  - White matter: ~70 ms
  - CSF: ~2000 ms

### 4.3 T2* Relaxation

Includes additional dephasing from local B₀ inhomogeneities:

```
1/T2* = 1/T2 + 1/T2'
```

T2* is always shorter than T2. Gradient-echo sequences are T2*-weighted,
while spin-echo sequences refocus T2' effects to give pure T2 contrast.

---

## 5. Spatial Encoding with Gradients

Three orthogonal gradient coils (Gx, Gy, Gz) superimpose linear field
variations on B₀:

```
B(x,y,z) = B₀ + Gx·x + Gy·y + Gz·z
```

This makes the Larmor frequency depend on position.

### 5.1 Frequency Encoding (Readout)

During signal acquisition, a gradient **Gx** (the readout gradient) is
applied along the x-axis. The instantaneous frequency at position x is:

```
ω(x) = γ · (B₀ + Gx · x)
```

The Fourier transform of the acquired signal separates frequencies →
spatial positions.

### 5.2 Phase Encoding

Before each readout, a gradient **Gy** is pulsed briefly along y. This
imparts a position-dependent phase:

```
φ(y) = γ · Gy · y · Δt
```

By repeating the experiment with different Gy amplitudes we sample the
second dimension of k-space. The number of phase-encode (PE) steps equals
the matrix size in the PE direction (e.g. 320 in the PWM benchmark).

### 5.3 Gradient Nonlinearity

Real gradient coils are not perfectly linear — the field deviates from the
ideal `G·r` at the edges of the FOV. This causes:
- **Geometric distortion** (warping)
- **Signal intensity variation**

The PWM benchmark models this as a smooth warp field `warp_field(2,H,W)` in
pixels. The `gradient_nonlin_frac` parameter ranges from 0.1% (mild) to 2%
(severe).

---

## 6. k-Space

### 6.1 The 2D Fourier Relationship

The MRI signal at time t is:

```
s(t) = ∫∫ ρ(x,y) · e^(−i2π(kx(t)·x + ky(t)·y)) dx dy
```

where `kx(t)` and `ky(t)` are the k-space trajectory determined by the
gradient waveforms:

```
kx(t) = γ/(2π) · ∫₀ᵗ Gx(τ) dτ
ky(t) = γ/(2π) · ∫₀ᵗ Gy(τ) dτ
```

Each readout line samples one row of k-space. Phase-encode steps move to
different ky positions.

### 6.2 k-Space Properties

```
                ky
                 ↑
     high freq   |   high freq
     (edges)     |   (edges)
    ─────────────┼───────────── kx
     high freq   |   high freq
     (edges)     |   (edges)
                 |
```

- **Centre of k-space** (low spatial frequencies): overall image contrast,
  brightness, large structures.
- **Periphery** (high spatial frequencies): edges, fine detail, sharp
  boundaries.
- **Symmetry**: for a real-valued object, k-space is Hermitian symmetric:
  `S(-k) = S*(k)`.

### 6.3 Cartesian vs Non-Cartesian Sampling

| Pattern | Trajectory | Pros | Cons |
|---------|-----------|------|------|
| Cartesian | Parallel lines in ky | Simple FFT recon, robust | Slow, coherent aliasing |
| Radial | Spokes through centre | Motion-robust, incoherent aliasing | Gridding needed, streaks |
| Spiral | Archimedes spiral | Very fast, efficient | Off-resonance sensitive |

The PWM benchmark uses **Cartesian sampling** with variable-density
undersampling (see `generate_vds_mask` in `build_dataset.py`).

### 6.4 Undersampling and Acceleration

To speed up acquisition, we skip PE lines:

```
Acceleration factor R = total_lines / sampled_lines
```

The PWM benchmark uses **R = 4** (4× acceleration). A fully sampled
**auto-calibration signal (ACS)** region (8% of k-space centre) is always
acquired for coil sensitivity estimation.

The mask is stored as a 1D array `mask(320,)` of `uint8` (0 or 1),
indicating which ky lines were sampled.

---

## 7. Multi-Coil Arrays

### 7.1 Phased-Array Coils

Modern MRI uses arrays of small surface coils (4–128 elements) instead of a
single volume coil. Benefits:
- Higher local SNR (closer to tissue)
- Enables parallel imaging (SENSE, GRAPPA)
- Each coil has a **spatial sensitivity profile** S_c(x,y)

### 7.2 Coil Sensitivity Maps

The signal received by coil c is:

```
s_c(t) = ∫∫ S_c(x,y) · ρ(x,y) · e^(−i2π(kx·x + ky·y)) dx dy
```

Sensitivity maps are complex-valued, smooth spatial functions. They are
typically estimated from the ACS region.

In the PWM benchmark:
- **15 coils** (`N_COILS = 15` in `build_dataset.py`)
- Stored as `coil_maps(15, 320, 320)` complex64
- Estimated via simplified ESPIRiT (see `estimate_sensitivity_maps` in
  `mri_solvers.py`)

### 7.3 Root-Sum-of-Squares (RSS) Combination

The simplest way to combine multi-coil images is RSS:

```
x_rss(x,y) = √(Σ_c |x_c(x,y)|²)
```

where x_c is the image from coil c. This is the baseline reconstruction
in the PWM benchmark (`zero_filled_reconstruction` in `mri_solvers.py`).

---

## 8. The Full Signal Equation

Combining all elements, the multi-coil MRI signal in k-space is:

```
y_c(kx, ky) = M(kx, ky) · F{ S_c(x,y) · ρ(x,y) · e^(−i2π·ΔB₀(x,y)·TE) } + n_c
```

where:
- `ρ(x,y)` — proton density (the image we want to recover)
- `S_c(x,y)` — sensitivity map of coil c (complex)
- `M(kx, ky)` — binary undersampling mask
- `F{}` — 2D Fourier transform
- `ΔB₀(x,y)` — B₀ inhomogeneity field map
- `TE` — echo time (25 ms in the PWM benchmark)
- `n_c` — complex Gaussian noise

In operator notation:

```
y = M · F · S · x + n
```

This is the **forward model** that reconstruction algorithms must invert.
The next file (02) explores this inverse problem in detail.

---

## 9. Summary

| Concept | Key Equation | Parameter |
|---------|-------------|-----------|
| Precession | ω₀ = γ B₀ | γ/2π = 42.577 MHz/T |
| Flip angle | α = γ ∫ B₁ dt | 90° for max signal |
| T1 recovery | Mz = M₀(1 − e^(−t/T1)) | 800–4000 ms (brain) |
| T2 decay | Mxy = Mxy₀ e^(−t/T2) | 40–2000 ms (brain) |
| k-space | s(t) = ∫∫ ρ e^(−i2π k·r) dr | 320 × 320 matrix |
| Multi-coil | y_c = MFS_c x + n | 15 coils, R=4 |

---

*Next: [02 — MRI as an Inverse Problem](02_mri_as_inverse_problem.md)*
