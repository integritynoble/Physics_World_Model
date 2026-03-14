# Multidimensional Lensless Imaging Systems: Proposal

## Motivation

A bare diffuser camera already encodes 3D (x, y, z) via depth-dependent PSFs. By adding physical diversity mechanisms — spectral dispersion, temporal modulation, or streak encoding — we can extend lensless imaging to 4D and 5D datacubes, all from a single 2D sensor readout.

| System | Dimensions | Object shape | Compression | Chain |
|--------|-----------|-------------|-------------|-------|
| 3D Lensless (baseline) | x, y, z | (Nz, Ny, Nx) | Nz:1 | C → Σ → D |
| **4D Spectral-Depth** | x, y, z, λ | (Nz, Nλ, Ny, Nx) | Nz·Nλ:1 | W → C → Σ → D |
| **4D Temporal-Depth** | x, y, z, t | (Nz, Nt, Ny, Nx) | Nz·Nt:1 | M → C → Σ → D |
| **5D Full** | x, y, z, λ, t | (Nz, Nλ, Nt, Ny, Nx) | Nz·Nλ·Nt:1 | M → W → C → Σ → D |

---

## System 1: 4D Spectral-Depth Lensless (x, y, z, λ)

### Chain: W → C → Σ_zλ → D

### Physical Concept

The diffuser PSF is **both depth-dependent and wavelength-dependent**:
- **Depth diversity**: PSF changes with object distance (defocus + magnification shift)
- **Spectral diversity**: A dispersive element (prism/grating) shifts each wavelength laterally before it hits the diffuser

These two diversity mechanisms are **physically independent**, providing a combined measurement operator with good conditioning.

### Forward Model

```
y(r) = Σ_z Σ_λ  H_z * S_λ(x_{z,λ}(r))  +  n(r)
```

where:
- `x_{z,λ}(r)` is the object at depth z, wavelength λ, spatial position r
- `S_λ` is the wavelength-dependent lateral shift (prism dispersion)
- `H_z` is the depth-dependent diffuser PSF
- Σ sums over all depth planes and spectral bands
- `n` is Poisson + Gaussian noise

### System Diagram

```
[Broadband scene]  →  [Prism/Grating]  →  [Phase Diffuser]  →  [Bare CMOS]  →  y
   (Nz × Nλ)           (λ-shift)         (z-dependent PSF)     (sum + noise)

   x_{z,λ}    →    S_λ(x_{z,λ})    →    H_z * S_λ(x_{z,λ})   →   Σ_{z,λ} + n
```

### Hardware

| Component | Specification | Cost |
|-----------|-------------|------|
| Broadband LED | 400–700 nm, 150 mW | $50 |
| Dispersive prism (BK7) | 37.5 nm/px dispersion | $100 |
| Phase diffuser | Ground glass, 25.4 mm | $50 |
| Bare CMOS sensor | 512×512, 5.5 μm pixel | $200 |
| **Total** | | **~$400** |

### Dimensions and Compression

| Parameter | Value |
|-----------|-------|
| Depth planes (Nz) | 8 |
| Spectral bands (Nλ) | 8 |
| Spatial resolution | 256 × 256 |
| Object size | 8 × 8 × 256 × 256 = 4,194,304 voxels |
| Measurement size | 256 × 256 = 65,536 pixels |
| **Compression ratio** | **64:1** |

### Diversity Analysis

The combined operator `A = Σ_{z,λ} H_z · S_λ` has two independent diversity sources:
1. **Depth**: H_z varies smoothly with z (defocus + lateral shift). Adjacent depth planes have correlated PSFs, but planes separated by >2 Rayleigh depths are nearly independent.
2. **Wavelength**: S_λ applies a lateral shift proportional to λ. With 37.5 nm/px dispersion and 8 bands spanning 400–700 nm, the maximum spectral shift is ~20 pixels — sufficient to decorrelate adjacent bands.

**Key advantage**: The diversity is inherent in the physics — no active modulation needed. This is a **completely passive** system.

### Reconstruction Algorithm

**GAP-TV** adapted for 4D:
```
Initialize: x_{z,λ}^0 = H_z^T · S_λ^T(y) / (Nz · Nλ)

for k = 1,...,100:
    y_est = Σ_{z,λ} H_z * S_λ(x_{z,λ}^k) / (Nz · Nλ)
    residual = y - y_est
    for each (z, λ):
        v_{z,λ} = x_{z,λ}^k + H_z^T · S_λ^T(residual) / (Nz · Nλ)
        x_{z,λ}^{k+1} = denoise_tv(v_{z,λ}, weight=λ_tv)
```

### Feasibility Assessment

| Criterion | Assessment |
|-----------|-----------|
| **Physical realizability** | Yes — all components are off-the-shelf |
| **Passive system** | Yes — no active modulation needed |
| **Calibration** | Requires per-(z,λ) PSF measurement: Nz × Nλ = 64 calibration images using a narrowband point source at each depth |
| **Compression (64:1)** | Aggressive but tractable — two independent diversity sources |
| **Expected PSNR** | 10–15 dB per (z,λ) slice (limited by 64:1 compression and cross-talk) |
| **Reconstruction time** | ~5–10 min on CPU (100 GAP-TV iterations × 64 slices × FFT convolutions) |
| **Practical limitation** | Spectral-depth cross-talk: if depth and wavelength produce similar PSF shifts, the operator becomes ill-conditioned |

**Verdict: FEASIBLE. This is the most practical of the three systems.**

### Prior Art

- Monakhova et al., "Spectral DiffuserCam: lensless snapshot hyperspectral imaging with a spectral filter array," Optica 7(10), 2020 — does (x, y, λ) but NOT depth
- Antipa et al., "DiffuserCam: Lensless single-exposure 3D imaging," Optica 5(1), 2018 — does (x, y, z) but NOT spectral
- **Our system combines both** — first (x, y, z, λ) lensless system

---

## System 2: 4D Temporal-Depth Lensless (x, y, z, t)

### Chain: M → C → Σ_zt → D

### Physical Concept

A fast temporal modulator (DMD or LCD shutter) provides time-varying masks that encode different video frames, while the diffuser's depth-dependent PSF encodes the depth dimension.

- **Depth diversity**: Depth-dependent PSF from diffuser (same as 3D lensless)
- **Temporal diversity**: Binary mask pattern m_t changes at each time step, modulating frame x_t differently before the diffuser

### Forward Model

```
y(r) = Σ_t Σ_z  H_z * (m_t(r) · x_{z,t}(r))  +  n(r)
```

where:
- `x_{z,t}(r)` is the object at depth z, time frame t
- `m_t(r)` is the binary temporal mask at time t
- `H_z` is the depth-dependent diffuser PSF
- The detector integrates over all time frames and depth planes

### System Diagram

```
[Dynamic scene]  →  [DMD mask m_t]  →  [Phase Diffuser]  →  [Bare CMOS]  →  y
   (Nz × Nt)         (temporal)        (z-dependent PSF)    (sum + noise)

   x_{z,t}    →    m_t · x_{z,t}   →  H_z * (m_t · x_{z,t})  → Σ_{z,t} + n
```

### Hardware

| Component | Specification | Cost |
|-----------|-------------|------|
| Broadband LED | 400–700 nm, 100 mW | $50 |
| DMD (temporal modulator) | 1024×768, 22 kHz switching | $800 |
| Phase diffuser | Ground glass, 25.4 mm | $50 |
| Bare CMOS sensor | 512×512, 5.5 μm pixel | $200 |
| Synchronization electronics | Trigger + timing | $100 |
| **Total** | | **~$1,200** |

### Dimensions and Compression

| Parameter | Value |
|-----------|-------|
| Depth planes (Nz) | 8 |
| Time frames (Nt) | 8 |
| Spatial resolution | 256 × 256 |
| Object size | 8 × 8 × 256 × 256 = 4,194,304 voxels |
| Measurement size | 256 × 256 = 65,536 pixels |
| **Compression ratio** | **64:1** |

### Diversity Analysis

1. **Depth**: H_z varies with object distance — inherent physical diversity (passive)
2. **Temporal**: m_t are random binary masks — designed diversity (active modulation)

The two mechanisms are fully independent: depth diversity comes from the diffuser optics, temporal diversity from the electronic mask. The combined operator should be well-conditioned.

### Reconstruction Algorithm

**GAP-TV** adapted for 4D:
```
for k = 1,...,100:
    y_est = Σ_{z,t} H_z * (m_t · x_{z,t}^k) / (Nz · Nt)
    residual = y - y_est
    for each (z, t):
        v_{z,t} = x_{z,t}^k + m_t · H_z^T(residual) / (Nz · Nt)
        x_{z,t}^{k+1} = denoise_tv(v_{z,t}, weight=λ_tv)
```

### Feasibility Assessment

| Criterion | Assessment |
|-----------|-----------|
| **Physical realizability** | Yes — DMD + diffuser + CMOS, all commercial |
| **Active modulation** | Yes — DMD requires synchronization electronics |
| **Calibration** | Per-depth PSF calibration (8 images) + mask pattern verification |
| **Compression (64:1)** | Tractable — temporal masks provide strong diversity |
| **Expected PSNR** | 12–16 dB per (z,t) slice (temporal masks improve conditioning vs. passive-only) |
| **Reconstruction time** | ~5–10 min on CPU |
| **Practical limitation** | DMD switching speed limits temporal resolution; scene motion during exposure causes blur |

**Verdict: FEASIBLE. Requires active modulation but uses standard CACTI-style hardware.**

### Prior Art

- Yuan et al., "Snapshot compressive imaging: Theory, algorithms, and applications," IEEE SPM, 2021 — CACTI does (x, y, t) but NOT depth
- Antipa et al., 2018 — DiffuserCam does (x, y, z) but NOT temporal
- **Our system combines both** — first (x, y, z, t) lensless system

---

## System 3: 5D Full Lensless (x, y, z, λ, t)

### Chain: M → W → C → Σ_zλt → D

### Physical Concept

The most ambitious design combines ALL three diversity mechanisms:
1. **Temporal modulation** (M): DMD or streak camera encodes time
2. **Spectral dispersion** (W): Prism shifts each wavelength differently
3. **Depth-dependent PSF** (C): Diffuser changes PSF with depth

### Forward Model

```
y(r) = Σ_t Σ_z Σ_λ  H_z * S_λ(m_t(r) · x_{z,λ,t}(r))  +  n(r)
```

where:
- `x_{z,λ,t}(r)` is the 5D object (depth z, wavelength λ, time t, spatial r)
- `m_t` is the temporal mask (or streak encoding)
- `S_λ` is wavelength-dependent lateral shift
- `H_z` is depth-dependent diffuser PSF

### Option A: DMD-Based (practical, lower cost)

```
[Dynamic spectral scene]  →  [DMD m_t]  →  [Prism S_λ]  →  [Diffuser H_z]  →  [CMOS]  →  y
     (Nz × Nλ × Nt)         (temporal)     (spectral)     (depth PSF)        (sum + n)
```

**Hardware cost: ~$1,500**

- Temporal resolution: limited by DMD switching rate (~22 kHz → ~45 μs per frame)
- Practical for video-rate (30 fps) with Nt=8 coded sub-frames per video frame

### Option B: Streak Camera (ultrafast, high cost)

```
[Ultrafast spectral scene]  →  [Prism S_λ]  →  [Diffuser H_z]  →  [Streak Camera]  →  y
     (Nz × Nλ × Nt)            (spectral)     (depth PSF)         (time → space)
```

**Hardware cost: ~$100,000–200,000**

A streak camera converts time into a spatial axis by sweeping photoelectrons with a ramped electric field. This replaces the temporal mask with a continuous time-to-space mapping:
- Instead of `m_t · x_{z,λ,t}`, the streak camera gives `x_{z,λ}(r_x, r_y + v·t)` where v is the streak velocity
- Temporal resolution: sub-picosecond (10^-12 s) — orders of magnitude faster than DMD
- The streak direction (y-axis) encodes time, while the orthogonal axis (x) retains spatial information

The forward model becomes:
```
y(r_x, r_y) = Σ_z Σ_λ  H_z * S_λ(x_{z,λ}(r_x, r_y - v·t_streak))  +  n
```

where `t_streak` maps the y-pixel to a time instant.

### Dimensions and Compression

For practical operation, reduce each dimension to keep compression manageable:

| Configuration | Nz | Nλ | Nt | Spatial | Total voxels | Compression |
|--------------|----|----|-----|---------|-------------|-------------|
| Aggressive | 8 | 8 | 8 | 256×256 | 33.6M | 512:1 |
| **Practical** | **4** | **4** | **4** | **256×256** | **4.2M** | **64:1** |
| Conservative | 2 | 4 | 4 | 256×256 | 2.1M | 32:1 |

**Recommendation**: Use the practical configuration (4×4×4) for 64:1 compression, matching the 4D systems.

### Diversity Analysis

Three independent diversity sources:
1. **Depth** (passive): H_z varies with z — continuous, physics-driven
2. **Spectral** (passive): S_λ shifts with λ — continuous, dispersion-driven
3. **Temporal** (active/passive): m_t or streak encoding — designed or physics-driven

With three independent mechanisms, the combined operator `A = Σ_{z,λ,t} H_z · S_λ · M_t` should have sufficient diversity **if** each mechanism provides at least partial orthogonality across its dimension.

**Critical condition**: The three diversity mechanisms must produce **distinguishable signatures**. Specifically:
- A depth change δz must produce a different PSF change than a wavelength change δλ
- A temporal mask change must be independent of both depth and wavelength effects
- If any two mechanisms produce similar effects (e.g., depth defocus looks like wavelength blur), the combined operator becomes ill-conditioned

### Reconstruction Algorithm

**Alternating GAP-TV** with dimension-wise splitting:

```
Initialize: x_{z,λ,t}^0 via adjoint back-projection

for k = 1,...,200:
    # Forward projection
    y_est = Σ_{z,λ,t} H_z * S_λ(m_t · x_{z,λ,t}^k) / (Nz · Nλ · Nt)

    # Residual back-projection
    residual = y - y_est
    for each (z, λ, t):
        v = x_{z,λ,t}^k + m_t · S_λ^T(H_z^T(residual)) / (Nz · Nλ · Nt)
        x_{z,λ,t}^{k+1} = denoise_tv(v, weight=λ_tv)
```

For the streak camera variant, replace `m_t · ...` with the streak-to-time mapping.

### Feasibility Assessment

| Criterion | Option A (DMD) | Option B (Streak) |
|-----------|---------------|-------------------|
| **Physical realizability** | Yes | Yes (specialized lab) |
| **Cost** | ~$1,500 | ~$100,000–200,000 |
| **Temporal resolution** | ~45 μs (DMD-limited) | <1 ps (streak-limited) |
| **Compression (64:1 at 4×4×4)** | Tractable | Tractable |
| **Compression (512:1 at 8×8×8)** | Very challenging | Very challenging |
| **Expected PSNR (4×4×4)** | 8–12 dB | 10–14 dB (better SNR from streak) |
| **Expected PSNR (8×8×8)** | 3–7 dB (likely too low) | 5–9 dB |
| **Calibration complexity** | Nz × Nλ PSFs + mask verification | Nz × Nλ PSFs + streak velocity calibration |
| **Reconstruction time** | ~30 min CPU / ~2 min GPU | ~30 min CPU / ~2 min GPU |
| **Key risk** | Cross-dimensional ambiguity at high compression | Streak camera cost and fragility |

**Verdict (Option A - DMD): FEASIBLE at 4×4×4, MARGINAL at 8×8×8.**

**Verdict (Option B - Streak): FEASIBLE at 4×4×4, but cost-prohibitive for most labs.**

### Prior Art

- No prior system recovers all 5 dimensions (x, y, z, λ, t) from a single shot
- Closest: CASSI+DiffuserCam hybrids do (x, y, λ) or (x, y, z) separately
- Streak cameras are used in CUP (Compressed Ultrafast Photography) for (x, y, t) at ps resolution, but without depth or spectral recovery

**This would be the FIRST 5D single-shot imaging system.**

---

## Comparative Summary

| System | Chain | Diversity | Passive? | Cost | Compression | Expected PSNR | Feasibility |
|--------|-------|-----------|----------|------|-------------|--------------|-------------|
| 3D Lensless | C→Σ→D | depth | Yes | $400 | 8:1 | 15–20 dB | Proven |
| **4D Spectral-Depth** | W→C→Σ→D | depth + spectral | **Yes** | $400 | 64:1 | 10–15 dB | **High** |
| **4D Temporal-Depth** | M→C→Σ→D | depth + temporal | No | $1,200 | 64:1 | 12–16 dB | **High** |
| **5D Full (DMD)** | M→W→C→Σ→D | depth + spectral + temporal | No | $1,500 | 64:1 (4³) | 8–12 dB | **Medium** |
| **5D Full (Streak)** | W→C→Σ→D_streak | depth + spectral + streak | No | $150,000 | 64:1 (4³) | 10–14 dB | **Low (cost)** |

### Key Insight

The progression from 3D to 5D follows a clear pattern:
- Each additional physical dimension adds one diversity mechanism to the chain
- The compression ratio multiplies: 8 → 64 → 64 (at reduced per-dim resolution)
- The reconstruction difficulty grows, but the forward model remains **linear** in all cases
- All systems share the same `Σ → D` (accumulate → detect) back-end

### Recommendation

1. **Build the 4D Spectral-Depth first** — it's completely passive ($400), needs no active electronics, and the calibration protocol is straightforward (scan a narrowband point source across depths)
2. **Add temporal modulation second** — upgrading to 5D requires only adding a DMD (+$800) and synchronization electronics
3. **The streak camera option** is for ultrafast applications only (femtosecond dynamics) — not recommended for general imaging due to cost

---

## Canonical Chains in FPB Notation

Using the 11 primitives B = {P, M, Π, F, C, Σ, D, S, W, R, Λ}:

| System | FPB Chain | Primitives used |
|--------|----------|----------------|
| 3D Lensless | C → Σ → D | C, Σ, D |
| 4D Spectral-Depth | W → C → Σ → D | W, C, Σ, D |
| 4D Temporal-Depth | M → C → Σ → D | M, C, Σ, D |
| 5D Full (DMD) | M → W → C → Σ → D | M, W, C, Σ, D |
| 5D Full (Streak) | W → C → Σ → D | W, C, Σ, D + streak detector |

Note: The streak camera is a special detector that maps time→space, so it could be modeled as a modified D primitive (D_streak) or as an additional Σ_t primitive.
