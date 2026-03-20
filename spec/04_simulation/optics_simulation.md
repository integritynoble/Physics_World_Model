# Fresnel Diffraction — Physics Simulation

> **Use Case 4: Scientific Simulation**
> From: `papers/universal_simulation/benchmark/09_optics/spec.md`

---

## Physics Equations

```
U(x, y, z) = (exp(ikz) / (iλz)) · FT{ U₀(xₐ, yₐ) · exp(iπ(xₐ²+yₐ²)/(λz)) }

I(x, y) = |U(x, y, z)|²

Fresnel number: N_F = R² / (λ · z)
```

| Variable | Description | Value |
|----------|-------------|-------|
| λ | Wavelength | 632.8 nm (HeNe) |
| R | Aperture radius | 0.5 mm |
| z | Propagation distance | 0.1 m |
| N_F | Fresnel number | 3.95 |
| Grid | Computation grid | 512 × 512 |

---

## Simulation Code

```python
# ============================================================
# Fresnel Diffraction Simulation — PWM Run Button
# ============================================================
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------
# 1. System parameters (customize here)
# -------------------------------------------------------
LAMBDA = 632.8e-9       # Wavelength (m) — HeNe laser
R_APERTURE = 0.5e-3     # Aperture radius (m)
Z = 0.1                 # Propagation distance (m)
N_GRID = 512            # Grid size (pixels)
PIXEL_SIZE = 5e-6       # Aperture plane pixel size (m)

# Derived
k = 2 * np.pi / LAMBDA
N_F = R_APERTURE**2 / (LAMBDA * Z)
print(f"Fresnel number: N_F = {N_F:.3f}")
print(f"Regime: {'Near-field (Fresnel)' if N_F > 1 else 'Far-field (Fraunhofer)'}")

# -------------------------------------------------------
# 2. Aperture function
# -------------------------------------------------------
x = np.linspace(-N_GRID//2, N_GRID//2 - 1, N_GRID) * PIXEL_SIZE
xa, ya = np.meshgrid(x, x)
r = np.sqrt(xa**2 + ya**2)

# Circular aperture
U0 = (r <= R_APERTURE).astype(np.complex128)
print(f"Aperture: circular, R={R_APERTURE*1e3:.2f} mm, grid: {N_GRID}×{N_GRID}")

# -------------------------------------------------------
# 3. Fresnel propagation via FFT
# -------------------------------------------------------
# Quadratic phase factor
fresnel_phase = np.exp(1j * np.pi * (xa**2 + ya**2) / (LAMBDA * Z))
U_fresnel = U0 * fresnel_phase

# FFT
U_fft = np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(U_fresnel)))

# Observation plane coordinates
freq_x = np.fft.fftshift(np.fft.fftfreq(N_GRID, d=PIXEL_SIZE))
dx_obs = LAMBDA * Z * freq_x.max() / N_GRID  # observation pixel size (approximate)

# Propagation prefactor
prefactor = np.exp(1j * k * Z) / (1j * LAMBDA * Z)
U_out = prefactor * U_fft * PIXEL_SIZE**2

# Intensity
I = np.abs(U_out)**2

# Normalize to power = 1
I = I / I.sum()

print(f"Output: intensity pattern {I.shape}")
print(f"Peak intensity: {I.max():.6e}")

# -------------------------------------------------------
# 4. Analytical validation (on-axis, Lommel functions)
# -------------------------------------------------------
# For circular aperture, on-axis intensity is:
# I(0,0,z) = 4 * sin²(π*N_F/2) / (π*N_F)²   [normalized]
I_onaxis_analytical = 4 * np.sin(np.pi * N_F / 2)**2 / (np.pi * N_F)**2
I_onaxis_simulated = I[N_GRID//2, N_GRID//2]
print(f"\nOn-axis intensity validation:")
print(f"  Analytical: {I_onaxis_analytical:.6e}")
print(f"  Simulated:  {I_onaxis_simulated:.6e}")
print(f"  Relative error: {abs(I_onaxis_analytical - I_onaxis_simulated)/I_onaxis_analytical:.2e}")

# -------------------------------------------------------
# 5. Compute observables
# -------------------------------------------------------
# Radial intensity profile
r_obs = np.sqrt((np.arange(N_GRID) - N_GRID//2)**2)
# Find center
cy, cx = N_GRID//2, N_GRID//2
radial_profile = I[cy, cx:]

# On-axis intensity vs. z
z_values = np.linspace(0.01, 0.5, 100)
I_onaxis = []
for zi in z_values:
    N_Fi = R_APERTURE**2 / (LAMBDA * zi)
    I_onaxis.append(4 * np.sin(np.pi * N_Fi / 2)**2 / max((np.pi * N_Fi)**2, 1e-10))
I_onaxis = np.array(I_onaxis)

# -------------------------------------------------------
# 6. Visualize
# -------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# Aperture
im0 = axes[0, 0].imshow(np.abs(U0), cmap='gray', extent=[-N_GRID//2*PIXEL_SIZE*1e3]*2 + [N_GRID//2*PIXEL_SIZE*1e3]*2)
axes[0, 0].set_title('Aperture |U₀| (mm)'); axes[0, 0].set_xlabel('x (mm)'); axes[0, 0].set_ylabel('y (mm)')

# Intensity pattern
im1 = axes[0, 1].imshow(I, cmap='inferno',
                         extent=[-N_GRID//2*dx_obs*1e3]*2 + [N_GRID//2*dx_obs*1e3]*2)
axes[0, 1].set_title(f'Intensity I(x,y,z)\nz={Z*100:.1f}cm, N_F={N_F:.2f}')
plt.colorbar(im1, ax=axes[0, 1])

# Log-scale intensity
im2 = axes[0, 2].imshow(np.log10(I + I.max()*1e-6), cmap='inferno')
axes[0, 2].set_title('Intensity (log scale)')
plt.colorbar(im2, ax=axes[0, 2])

# Radial profile
axes[1, 0].plot(radial_profile[:100], 'b-', linewidth=2)
axes[1, 0].set_title('Radial intensity profile I(r)'); axes[1, 0].set_xlabel('r (pixels)')

# On-axis vs z
axes[1, 1].plot(z_values * 100, I_onaxis, 'r-', linewidth=2)
axes[1, 1].axvline(Z * 100, color='k', linestyle='--', label=f'z={Z*100:.1f}cm')
axes[1, 1].set_title('On-axis intensity I(0,0,z)'); axes[1, 1].set_xlabel('z (cm)')
axes[1, 1].legend()

# Encircled energy
from scipy.integrate import cumulative_trapezoid
r_pix = np.arange(len(radial_profile))
encircled = cumulative_trapezoid(radial_profile * 2 * np.pi * r_pix, r_pix, initial=0)
encircled /= encircled[-1] + 1e-10
axes[1, 2].plot(r_pix[:100], encircled[:100], 'g-', linewidth=2)
axes[1, 2].set_title('Encircled energy E(r)'); axes[1, 2].set_xlabel('r (pixels)')
axes[1, 2].set_ylabel('Fraction of total power')

plt.suptitle(f'Fresnel Diffraction: λ={LAMBDA*1e9:.0f}nm, R={R_APERTURE*1e3:.1f}mm, z={Z*100:.0f}cm', fontsize=14)
plt.tight_layout()
plt.savefig('fresnel_diffraction.png', dpi=150, bbox_inches='tight')
plt.show()
print("Saved: fresnel_diffraction.png")
```

---

## Validation Tolerance

From `papers/universal_simulation/benchmark/09_optics/spec.md`:
- On-axis intensity relative error: ≤ 1 × 10⁻⁵
- Radial profile L2 relative: ≤ 1 × 10⁻⁵

---

## Task Variations

| Variation | Description |
|-----------|-------------|
| Aperture shape | Rectangular, slit, annular |
| Wavelength | 400–800 nm (visible), 0.1 nm (X-ray) |
| z | Near-field (N_F>>1) to far-field (N_F<<1) |
| Coherence | Partial coherence (mutual coherence function) |
| Aberrations | Defocus, astigmatism (add phase to U₀) |

---

## References

- **Fresnel diffraction**: Born & Wolf, "Principles of Optics", Ch. 8
- **Lommel functions**: Born & Wolf, §8.8
- **FFT propagation**: Goodman, "Introduction to Fourier Optics", Ch. 4
