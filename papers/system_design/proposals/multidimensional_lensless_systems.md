# Multidimensional Lensless Imaging Systems: Proposal

## Motivation

A bare diffuser camera already encodes 3D (x, y, z) via depth-dependent PSFs. By adding physical diversity mechanisms — spectral dispersion (W_λ), temporal modulation (M), or temporal dispersion via streak camera (W_t) — we can extend lensless imaging to 4D and 5D datacubes, all from a single 2D sensor readout.

**Key insight:** The streak camera and the spectral disperser are the **same primitive W** — both are parameterized shift operators `W_ξ : x(r) → x(r + f(ξ))`. This means no new primitives are needed beyond the 11 FPB building blocks B = {P, M, Π, F, C, Σ, D, S, W, R, Λ}.

| System | Dimensions | Object shape | Compression | Chain | Passive? |
|--------|-----------|-------------|-------------|-------|----------|
| 3D Lensless (baseline) | x, y, z | (Nz, Ny, Nx) | Nz:1 | C → Σ → D | Yes |
| **4D Spectral-Depth** | x, y, z, λ | (Nz, Nλ, Ny, Nx) | Nz·Nλ:1 | W_λ → C → Σ → D | Yes |
| **4D Temporal-Depth (DMD)** | x, y, z, t | (Nz, Nt, Ny, Nx) | Nz·Nt:1 | M → C → Σ → D | No |
| **4D Temporal-Depth (Streak)** | x, y, z, t | (Nz, Nt, Ny, Nx) | Nz·Nt:1 | W_t → C → Σ → D | **Yes** |
| **5D Full (DMD)** | x, y, z, λ, t | (Nz, Nλ, Nt, Ny, Nx) | Nz·Nλ·Nt:1 | M → W_λ → C → Σ → D | No |
| **5D Full (Streak)** | x, y, z, λ, t | (Nz, Nλ, Nt, Ny, Nx) | Nz·Nλ·Nt:1 | W_λ → W_t → C → Σ → D | **Yes** |

---

## System 1: 4D Spectral-Depth Lensless (x, y, z, λ)

### Chain: W_λ → C → Σ → D

### Physical Concept

The diffuser PSF is **both depth-dependent and wavelength-dependent**:
- **Depth diversity** (C): PSF changes with object distance (defocus + magnification shift)
- **Spectral diversity** (W_λ): A dispersive element (prism/grating) shifts each wavelength laterally before it hits the diffuser

These two diversity mechanisms are **physically independent**, providing a combined measurement operator with good conditioning.

### Forward Model

```
y(r) = Σ_z Σ_λ  H_z * W_λ(x_{z,λ}(r))  +  n(r)
```

where:
- `x_{z,λ}(r)` is the object at depth z, wavelength λ, spatial position r
- `W_λ : x(r) → x(r + d(λ))` is the spectral dispersion operator (prism)
- `H_z` is the depth-dependent diffuser PSF (convolution operator C)
- Σ sums over all depth planes and spectral bands (accumulation Σ)
- `n` is Poisson + Gaussian noise (detection D)

### System Diagram

```
[Broadband scene]  →  [Prism/Grating W_λ]  →  [Diffuser C_z]  →  [CMOS Σ+D]  →  y
   (Nz × Nλ)           (λ-shift)              (z-dependent PSF)   (sum + noise)

   x_{z,λ}    →    W_λ(x_{z,λ})    →    H_z * W_λ(x_{z,λ})   →   Σ_{z,λ} + n
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

The combined operator `A = Σ_{z,λ} H_z · W_λ` has two independent diversity sources:
1. **Depth** (C): H_z varies smoothly with z (defocus + lateral shift). Adjacent depth planes have correlated PSFs, but planes separated by >2 Rayleigh depths are nearly independent.
2. **Wavelength** (W_λ): W_λ applies a lateral shift proportional to λ. With 37.5 nm/px dispersion and 8 bands spanning 400–700 nm, the maximum spectral shift is ~20 pixels — sufficient to decorrelate adjacent bands.

**Key advantage**: The diversity is inherent in the physics — no active modulation needed. This is a **completely passive** system.

### Reconstruction Algorithm

**GAP-TV** adapted for 4D:
```
Initialize: x_{z,λ}^0 = H_z^T · W_λ^T(y) / (Nz · Nλ)

for k = 1,...,100:
    y_est = Σ_{z,λ} H_z * W_λ(x_{z,λ}^k) / (Nz · Nλ)
    residual = y - y_est
    for each (z, λ):
        v_{z,λ} = x_{z,λ}^k + W_λ^T · H_z^T(residual) / (Nz · Nλ)
        x_{z,λ}^{k+1} = denoise_tv(v_{z,λ}, weight=λ_tv)
```

### Feasibility Assessment

| Criterion | Assessment |
|-----------|-----------|
| **Physical realizability** | Yes — all components are off-the-shelf |
| **Passive system** | Yes — no active modulation needed |
| **Calibration** | Requires per-(z,λ) PSF measurement: Nz × Nλ = 64 calibration images using a narrowband point source at each depth. W_λ shift calibrated from known spectral lines. |
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

Two approaches exist for encoding time: **active modulation** (DMD, M primitive) and **passive dispersion** (streak camera, W_t primitive). Both exploit the diffuser's depth-dependent PSF (C) for the depth dimension.

### Option A: DMD-Based (Active) — Chain: M → C → Σ → D

#### Physical Concept

A fast temporal modulator (DMD or LCD shutter) provides time-varying masks that encode different video frames, while the diffuser's depth-dependent PSF encodes the depth dimension.

- **Depth diversity** (C): Depth-dependent PSF from diffuser — passive, inherent
- **Temporal diversity** (M): Binary mask pattern m_t changes at each time step — active, designed

#### Forward Model

```
y(r) = Σ_t Σ_z  H_z * (m_t(r) · x_{z,t}(r))  +  n(r)
```

where:
- `x_{z,t}(r)` is the object at depth z, time frame t
- `m_t(r)` is the binary temporal mask at time t (modulation M)
- `H_z` is the depth-dependent diffuser PSF (convolution C)
- Σ accumulates over all time frames and depth planes, D detects

#### System Diagram

```
[Dynamic scene]  →  [DMD mask M_t]  →  [Diffuser C_z]  →  [CMOS Σ+D]  →  y
   (Nz × Nt)         (active)         (z-dependent PSF)   (sum + noise)

   x_{z,t}    →    m_t · x_{z,t}   →  H_z * (m_t · x_{z,t})  → Σ_{z,t} + n
```

#### Hardware

| Component | Specification | Cost |
|-----------|-------------|------|
| Broadband LED | 400–700 nm, 100 mW | $50 |
| DMD (temporal modulator) | 1024×768, 22 kHz switching | $800 |
| Phase diffuser | Ground glass, 25.4 mm | $50 |
| Bare CMOS sensor | 512×512, 5.5 μm pixel | $200 |
| Synchronization electronics | Trigger + timing | $100 |
| **Total** | | **~$1,200** |

#### Reconstruction Algorithm (GAP-TV)

```
for k = 1,...,100:
    y_est = Σ_{z,t} H_z * (m_t · x_{z,t}^k) / (Nz · Nt)
    residual = y - y_est
    for each (z, t):
        v_{z,t} = x_{z,t}^k + m_t · H_z^T(residual) / (Nz · Nt)
        x_{z,t}^{k+1} = denoise_tv(v_{z,t}, weight=λ_tv)
```

### Option B: Streak Camera (Passive) — Chain: W_t → C → Σ → D

#### Physical Concept

A streak camera maps time onto a spatial axis via electrostatic deflection: `W_t : x(r) → x(r + v·t)`, where v is the streak velocity. This is **mathematically identical** to spectral dispersion `W_λ : x(r) → x(r + d(λ))` — the same W primitive with a different parameter.

- **Depth diversity** (C): Depth-dependent PSF from diffuser — passive, inherent
- **Temporal diversity** (W_t): Streak camera disperses time onto space — passive, physics-driven

#### Forward Model

```
y(r) = Σ_z  H_z * W_t(x_{z}(r, t))  +  n(r)
     = Σ_z  H_z * x_{z}(r_x, r_y + v·t)  +  n(r)
```

where:
- `W_t` shifts the signal along the y-axis proportional to time
- `v` is the streak velocity (calibrated parameter)
- The streak direction (y) encodes time; orthogonal axis (x) retains spatial information

#### System Diagram

```
[Dynamic scene]  →  [Streak tube W_t]  →  [Diffuser C_z]  →  [CMOS Σ+D]  →  y
   (Nz × Nt)        (time → space)       (z-dependent PSF)  (sum + noise)

   x_{z,t}    →    x_{z}(r + v·t)    →  H_z * x_{z}(r + v·t)  → Σ_{z} + n
```

#### Hardware

| Component | Specification | Cost |
|-----------|-------------|------|
| Pulsed light source | ps/fs laser or fast LED | $5,000–50,000 |
| Streak camera | Sub-ps time resolution | $50,000–100,000 |
| Phase diffuser | Ground glass, 25.4 mm | $50 |
| **Total** | | **~$60,000–150,000** |

#### Reconstruction Algorithm (GAP-TV)

```
for k = 1,...,100:
    y_est = Σ_z H_z * W_t(x_{z,t}^k) / (Nz · Nt)
    residual = y - y_est
    for each (z, t):
        v_{z,t} = x_{z,t}^k + W_t^T · H_z^T(residual) / (Nz · Nt)
        x_{z,t}^{k+1} = denoise_tv(v_{z,t}, weight=λ_tv)
```

Note: `W_t^T` is the adjoint shift `x(r) → x(r - v·t)` — identical structure to `W_λ^T`.

### Dimensions and Compression (both options)

| Parameter | Value |
|-----------|-------|
| Depth planes (Nz) | 8 |
| Time frames (Nt) | 8 |
| Spatial resolution | 256 × 256 |
| Object size | 8 × 8 × 256 × 256 = 4,194,304 voxels |
| Measurement size | 256 × 256 = 65,536 pixels |
| **Compression ratio** | **64:1** |

### Feasibility Comparison

| Criterion | Option A (DMD) | Option B (Streak) |
|-----------|---------------|-------------------|
| **FPB chain** | M → C → Σ → D | W_t → C → Σ → D |
| **Passive?** | No (DMD requires sync) | **Yes** (pure physics) |
| **Cost** | ~$1,200 | ~$60,000–150,000 |
| **Temporal resolution** | ~45 μs (DMD-limited) | <1 ps (streak-limited) |
| **Compression (64:1)** | Tractable | Tractable |
| **Expected PSNR** | 12–16 dB | 14–18 dB (better from continuous W_t) |
| **Calibration** | 8 PSFs + mask verification | 8 PSFs + streak velocity calibration |
| **Practical limitation** | DMD speed limits time resolution | Cost; streak tube fragility |

**Verdict (DMD): FEASIBLE. Standard CACTI-style hardware, ~$1,200 total.**

**Verdict (Streak): FEASIBLE but expensive. The passive W_t → C → Σ → D chain is elegant — no active electronics.**

### Prior Art

- Yuan et al., "Snapshot compressive imaging: Theory, algorithms, and applications," IEEE SPM, 2021 — CACTI does (x, y, t) but NOT depth
- Antipa et al., 2018 — DiffuserCam does (x, y, z) but NOT temporal
- Liang et al., "Single-shot real-time femtosecond imaging of temporal focusing," Light: Sci. & Appl., 2018 — CUP does (x, y, t) via streak but NOT depth
- **Our system combines both** — first (x, y, z, t) lensless system. The streak variant (W_t → C → Σ → D) is additionally the first **passive** 4D temporal lensless system.

---

## System 3: 5D Full Lensless (x, y, z, λ, t)

Two approaches exist, differing only in how time is encoded:

| Option | Chain | Time encoding | Passive? |
|--------|-------|--------------|----------|
| **A (DMD)** | M → W_λ → C → Σ → D | Active modulation (M) | No |
| **B (Streak)** | W_λ → W_t → C → Σ → D | Passive dispersion (W_t) | **Yes** |

### Physical Concept

The most ambitious design combines ALL three diversity mechanisms:
1. **Depth-dependent PSF** (C): Diffuser changes PSF with depth — passive
2. **Spectral dispersion** (W_λ): Prism shifts each wavelength differently — passive
3. **Temporal encoding**: DMD mask (M, active) **or** streak camera (W_t, passive)

### Option A: DMD-Based — Chain: M → W_λ → C → Σ → D

#### Forward Model

```
y(r) = Σ_t Σ_z Σ_λ  H_z * W_λ(m_t(r) · x_{z,λ,t}(r))  +  n(r)
```

#### System Diagram

```
[Dynamic spectral scene]  →  [DMD M_t]  →  [Prism W_λ]  →  [Diffuser C_z]  →  [CMOS Σ+D]  →  y
     (Nz × Nλ × Nt)         (active)      (dispersion)    (depth PSF)        (sum + noise)
```

**Hardware cost: ~$1,500** (DMD $800 + prism $100 + diffuser $50 + CMOS $200 + sync $100 + LED $50)

- Temporal resolution: limited by DMD switching rate (~22 kHz → ~45 μs per frame)
- Practical for video-rate (30 fps) with Nt=8 coded sub-frames per video frame

### Option B: Streak Camera — Chain: W_λ → W_t → C → Σ → D

#### Forward Model

```
y(r) = Σ_z Σ_λ  H_z * W_λ(W_t(x_{z,λ,t}(r)))  +  n(r)
     = Σ_z Σ_λ  H_z * x_{z,λ}(r_x + d(λ), r_y + v·t)  +  n(r)
```

where:
- `W_λ : x(r) → x(r_x + d(λ), r_y)` — spectral shift along x-axis (prism)
- `W_t : x(r) → x(r_x, r_y + v·t)` — temporal shift along y-axis (streak)
- The two W operators act on **orthogonal axes** — spectral along x, temporal along y

#### System Diagram

```
[Ultrafast spectral scene]  →  [Prism W_λ]  →  [Streak W_t]  →  [Diffuser C_z]  →  [CMOS Σ+D]  →  y
     (Nz × Nλ × Nt)            (λ → x-shift)  (t → y-shift)    (depth PSF)        (sum + noise)
```

**Hardware cost: ~$100,000–200,000** (streak camera $50,000–150,000 + pulsed source $5,000–50,000 + optics $300)

- Temporal resolution: sub-picosecond (10^-12 s) — orders of magnitude faster than DMD
- **Fully passive 5D system** — no electronics, no masks, no modulators. Just a prism (W_λ), a streak tube (W_t), a diffuser (C), and a bare sensor (Σ+D). All 5 dimensions encoded by physics alone.

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
1. **Depth** (C, passive): H_z varies with z — continuous, physics-driven
2. **Spectral** (W_λ, passive): W_λ shifts with λ — continuous, dispersion-driven
3. **Temporal**: M_t (active, designed masks) **or** W_t (passive, streak dispersion)

With three independent mechanisms, the combined operator should have sufficient diversity **if** each mechanism provides at least partial orthogonality across its dimension.

**Option B advantage**: In the streak system, all three mechanisms (C, W_λ, W_t) act on **different physical axes**:
- W_λ disperses along x (prism)
- W_t disperses along y (streak)
- C varies the PSF shape (diffuser)
This orthogonality ensures the combined operator `A = Σ_{z,λ,t} H_z · W_λ · W_t` is better conditioned than Option A where M_t acts pixel-wise on the same spatial axes as the PSF.

**Critical condition**: The three diversity mechanisms must produce **distinguishable signatures**:
- A depth change δz must produce a different PSF change than a wavelength change δλ
- Temporal encoding (M_t or W_t) must be independent of both depth and wavelength effects
- If any two mechanisms produce similar effects (e.g., depth defocus looks like wavelength blur), the combined operator becomes ill-conditioned

### Reconstruction Algorithm

**Alternating GAP-TV** with dimension-wise splitting:

```
Initialize: x_{z,λ,t}^0 via adjoint back-projection

# Option A (DMD): M → W_λ → C → Σ → D
for k = 1,...,200:
    y_est = Σ_{z,λ,t} H_z * W_λ(m_t · x_{z,λ,t}^k) / (Nz · Nλ · Nt)
    residual = y - y_est
    for each (z, λ, t):
        v = x_{z,λ,t}^k + m_t · W_λ^T(H_z^T(residual)) / (Nz · Nλ · Nt)
        x_{z,λ,t}^{k+1} = denoise_tv(v, weight=λ_tv)

# Option B (Streak): W_λ → W_t → C → Σ → D
for k = 1,...,200:
    y_est = Σ_{z,λ,t} H_z * W_λ(W_t(x_{z,λ,t}^k)) / (Nz · Nλ · Nt)
    residual = y - y_est
    for each (z, λ, t):
        v = x_{z,λ,t}^k + W_t^T(W_λ^T(H_z^T(residual))) / (Nz · Nλ · Nt)
        x_{z,λ,t}^{k+1} = denoise_tv(v, weight=λ_tv)
```

Note: The adjoint of each W is simply the reverse shift: `W_λ^T(x)(r) = x(r - d(λ))`, `W_t^T(x)(r) = x(r - v·t)`. The algorithm structure is identical for both options — only the forward/adjoint operators differ.

### Feasibility Assessment

| Criterion | Option A: DMD (M→W_λ→C→Σ→D) | Option B: Streak (W_λ→W_t→C→Σ→D) |
|-----------|-------------------------------|-------------------------------------|
| **Passive?** | No (DMD requires sync) | **Yes** (all physics) |
| **Cost** | ~$1,500 | ~$100,000–200,000 |
| **Temporal resolution** | ~45 μs (DMD-limited) | <1 ps (streak-limited) |
| **Compression (64:1 at 4³)** | Tractable | Tractable |
| **Compression (512:1 at 8³)** | Very challenging | Very challenging |
| **Expected PSNR (4³)** | 8–12 dB | 10–14 dB (better from orthogonal W_λ,W_t) |
| **Expected PSNR (8³)** | 3–7 dB (likely too low) | 5–9 dB |
| **Calibration** | Nz×Nλ PSFs + mask verify | Nz×Nλ PSFs + W_t velocity calibration |
| **Reconstruction time** | ~30 min CPU / ~2 min GPU | ~30 min CPU / ~2 min GPU |
| **Key risk** | Cross-dimensional ambiguity | Cost; streak tube fragility |

**Verdict (Option A - DMD): FEASIBLE at 4×4×4, MARGINAL at 8×8×8.**

**Verdict (Option B - Streak): FEASIBLE at 4×4×4, but cost-prohibitive for most labs. Uniquely, this is a fully passive 5D system — the only one possible with 11 FPB primitives.**

### Prior Art

- No prior system recovers all 5 dimensions (x, y, z, λ, t) from a single shot
- Closest: CASSI+DiffuserCam hybrids do (x, y, λ) or (x, y, z) separately
- Liang et al., CUP (Compressed Ultrafast Photography) — streak-based (x, y, t) at ps resolution, but without depth or spectral recovery
- Gao et al., "Single-shot compressed ultrafast photography at hundred billion frames per second," Nature 516, 2014 — M + streak for (x, y, t), no depth/spectral

**This would be the FIRST 5D single-shot imaging system.**

---

## Comparative Summary

| System | Chain | Diversity | Passive? | Cost | Compression | Expected PSNR | Feasibility |
|--------|-------|-----------|----------|------|-------------|--------------|-------------|
| 3D Lensless | C→Σ→D | depth | Yes | $400 | 8:1 | 15–20 dB | Proven |
| **4D Spectral-Depth** | W_λ→C→Σ→D | depth + spectral | **Yes** | $400 | 64:1 | 10–15 dB | **High** |
| **4D Temporal-Depth (DMD)** | M→C→Σ→D | depth + temporal | No | $1,200 | 64:1 | 12–16 dB | **High** |
| **4D Temporal-Depth (Streak)** | W_t→C→Σ→D | depth + temporal | **Yes** | $80,000 | 64:1 | 14–18 dB | **Medium (cost)** |
| **5D Full (DMD)** | M→W_λ→C→Σ→D | all three | No | $1,500 | 64:1 (4³) | 8–12 dB | **Medium** |
| **5D Full (Streak)** | W_λ→W_t→C→Σ→D | all three | **Yes** | $150,000 | 64:1 (4³) | 10–14 dB | **Low (cost)** |

---

## Unifying Insight: W_ξ as a General Dispersion Primitive

A crucial observation: **the streak camera and the spectral disperser are the same primitive W** — both are shift operators that map a non-spatial dimension onto a spatial axis of the detector:

| Device | Primitive | Coordinate ξ | Forward operator | Adjoint | Physics |
|--------|-----------|-------------|-----------------|---------|---------|
| Prism / Grating | W_λ | wavelength | x(r) → x(r + d(λ)) | x(r) → x(r - d(λ)) | Refractive dispersion |
| Streak camera | W_t | time | x(r) → x(r + v·t) | x(r) → x(r - v·t) | Electrostatic deflection |

Mathematically, both are parameterized shift operators: **`W_ξ : x(r) → x(r + f(ξ))`** where ξ ∈ {λ, t, ...}. The physics differs but the linear algebra is identical. This means:

1. **No new primitive is needed** — the existing W primitive covers both spectral and temporal dispersion
2. **The 5D streak system** is simply two W's in series: `W_λ → W_t → C → Σ → D`
3. **The reconstruction algorithm** is the same — just apply the adjoint shift `W_ξ^T : x(r) → x(r - f(ξ))` in the appropriate dimension
4. **The DMD alternative** replaces W_t with M (modulation instead of dispersion), trading ultrafast resolution for lower cost
5. **Future extensions**: Any new dispersion mechanism (e.g., angular W_θ) slots in as another W_ξ

This unification strengthens the FPB framework: the 11 primitives B = {P, M, Π, F, C, Σ, D, S, W, R, Λ} are sufficient to describe even the most ambitious multidimensional systems without inventing new operators.

---

## Progression Pattern

Each additional physical dimension adds exactly one primitive to the chain:

```
3D (x,y,z):             C  → Σ → D           depth via PSF diversity (passive)
4D (x,y,z,λ):       W_λ → C  → Σ → D         + spectral dispersion   (passive)
4D (x,y,z,t):        M  → C  → Σ → D         + temporal modulation    (active)
4D (x,y,z,t):       W_t → C  → Σ → D         + temporal dispersion    (passive)
5D (x,y,z,λ,t):  M → W_λ → C → Σ → D        + modulation + spectral  (active)
5D (x,y,z,λ,t): W_λ → W_t → C → Σ → D       + both dispersions       (passive)
```

The compression ratio multiplies: 8 → 64 → 64 (at reduced per-dim resolution). The forward model remains **linear** in all cases. All systems share the same `Σ → D` back-end (accumulation + detection).

**Passive vs. Active trade-off**: Every system that uses M (mask/DMD) is active; every system that uses only W, C, Σ, D is passive. The streak camera enables passive temporal encoding (W_t), while the DMD gives active temporal encoding (M) at 1/100th the cost.

---

## Recommendation

1. **Build the 4D Spectral-Depth first** — completely passive ($400), no active electronics, calibration is straightforward (scan a narrowband point source across depths). Chain: W_λ → C → Σ → D.
2. **Add temporal encoding second**:
   - **Budget path**: DMD ($1,200 total) → M → C → Σ → D for 4D temporal, then M → W_λ → C → Σ → D for 5D
   - **Premium path**: Streak camera ($80,000+) → W_t → C → Σ → D for passive 4D temporal, then W_λ → W_t → C → Σ → D for fully passive 5D
3. **The ultimate system** `W_λ → W_t → C → Σ → D` is a fully passive 5D imager — no electronics, no masks, no modulators, just optics and physics

---

## Canonical Chains in FPB Notation

Using the 11 primitives B = {P, M, Π, F, C, Σ, D, S, W, R, Λ}:

| System | FPB Chain | Primitives used | Active? | Cost |
|--------|----------|----------------|---------|------|
| 3D Lensless | C → Σ → D | C, Σ, D | Passive | $400 |
| 4D Spectral-Depth | W_λ → C → Σ → D | W, C, Σ, D | Passive | $400 |
| 4D Temporal-Depth (DMD) | M → C → Σ → D | M, C, Σ, D | Active | $1,200 |
| 4D Temporal-Depth (Streak) | W_t → C → Σ → D | W, C, Σ, D | **Passive** | $80,000 |
| 5D Full (DMD) | M → W_λ → C → Σ → D | M, W, C, Σ, D | Active | $1,500 |
| 5D Full (Streak) | W_λ → W_t → C → Σ → D | W, W, C, Σ, D | **Passive** | $150,000 |

The streak-based 5D system `W_λ → W_t → C → Σ → D` is remarkable: it is a **fully passive** 5D imaging system — no electronics, no masks, no modulators. Just a prism, a streak tube, a diffuser, and a bare sensor. All 5 dimensions are encoded by physics alone.
