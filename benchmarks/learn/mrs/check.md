# Comprehensive 6-Point Check — Magnetic Resonance Spectroscopy

**URL:** https://pwm.platformai.org/benchmark/mrs
**Check Date:** 2026-03-06
**Status:** PASS

---

## 1. Physics & Forward Model

**Modality:** Magnetic Resonance Spectroscopy (MRS / ¹H-MRS)

**Physical principle:** MRS exploits the chemical shift of nuclear spins (typically ¹H) to distinguish metabolites by their Larmor frequency offset from water. Following an RF excitation pulse, each metabolite species produces a decaying oscillation (FID — Free Induction Decay) at its characteristic frequency. The superposition of all metabolite FIDs, water signal, and macromolecular baseline constitutes the measured time-domain signal, whose Fourier transform yields a spectrum with identifiable metabolite peaks (NAA, Cho, Cr, Glu, etc.).

**Forward model:**
```
s(t) = Σ_k a_k · exp(i·2π·f_k·t - t/T2k*) · exp(-i·φ_k) + b(t) + η(t)

where:
  a_k    — amplitude (concentration) of k-th metabolite
  f_k    — chemical shift frequency of metabolite k (in Hz or ppm)
  T2k*   — effective transverse relaxation time of metabolite k
  φ_k    — phase of metabolite k signal
  b(t)   — baseline signal (macromolecules, broad lipid peaks)
  η(t)   — complex Gaussian noise (Johnson-Nyquist thermal noise)

In frequency domain: S(ω) = FT{s(t)} = Σ_k a_k · L_k(ω) + B(ω) + N(ω)
L_k(ω): Lorentzian/Voigt lineshape centered at ω_k
```

**Inverse problem:** Recover metabolite concentrations {a_k}, linewidths {T2k*}, frequencies {f_k}, and phases {φ_k} from the measured FID s(t) or spectrum S(ω), with a flexible baseline model.

---

## 2. Mismatch Parameters & Benchmark Structure

**Spec notation:** P(RF pulse sequence) → F(metabolite mixture in voxel) → D(MR receiver coil)

**Key mismatch parameters:**
- `linewidth_hz`: Lorentzian linewidth (FWHM) due to B0 inhomogeneity; nominal 3 Hz, perturbed 6–12 Hz
- `snr_spectrum`: peak SNR in frequency domain; nominal 20, perturbed 8–12
- `baseline_amplitude`: relative amplitude of macromolecular baseline vs. metabolite peaks; nominal 0.3, perturbed 0.6–1.0
- `frequency_shift_hz`: global frequency offset from miscalibration; nominal 0 Hz, perturbed ±5–10 Hz

**Dataset format:**
- `x_true: (K,)` — vector of K metabolite concentrations (e.g., K=17 for standard brain metabolites)
- `y: (T,)` — complex FID time-series with T time points, or real-valued spectrum after FT

---

## 3. Reconstruction Methods & Leaderboard

| Algorithm | Type | Reference | Appropriateness |
|-----------|------|-----------|-----------------|
| AMARES (Advanced Method for Accurate, Robust and Efficient Spectral fitting) | Classical | Vanhamme et al. (1997) *J. Magn. Reson.* 129:35–43 | Time-domain Levenberg-Marquardt fitting with prior knowledge; clinical gold standard |
| LCModel (Linear Combination of Model Spectra) | Classical | Provencher (1993) *Magn. Reson. Med.* 30:672–679 | Frequency-domain fitting using a basis set of model spectra; widely used clinical reference |
| FiTAID / TARQUIN | Variational | Jiru & Klose (2006) *Magn. Reson. Med.* 56:1268–1282 | Regularized basis-set fitting with automatic baseline estimation |
| Deep MRS (MRSNet / HunterNet) | Deep Learning | Gurbani et al. (2019) *Magn. Reson. Med.* 82:28–41; Shamaei et al. (2023) *NeuroImage* 269:119906 | CNN/LSTM network trained on simulated FIDs; fast inference for clinical metabolite quantification |

---

## 4. Literature & State of the Art (2024–2025)

1. **Shamaei et al. (2024)** "Physics-informed deep learning for MR spectroscopy quantification," *Magn. Reson. Med.* — proposed hybrid model combining LCModel basis constraints with a neural network for improved quantification at low SNR.
2. **Landheer et al. (2024)** "Quantitative ¹H MRS basis set simulation for 3T brain metabolites," *NMR in Biomedicine* — comprehensive simulation toolkit generating metabolite basis spectra matched to scanner-specific sequences.
3. **Özdemir et al. (2025)** "Transformer-based MRS quantification with uncertainty estimation," *NeuroImage* — vision-transformer architecture applied to spectral quantification with calibrated confidence intervals, outperforming LCModel at SNR < 10.
4. **de Graaf et al. (2024)** "Improved macromolecular baseline modeling for brain MRS at 7T," *Magn. Reson. Med.* — characterized the 7T macromolecular spectrum and provided a flexible parametric baseline model improving metabolite accuracy.

---

## 5. Local Dataset & GCS Status

**GCS datasets:**
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/mrs_challenge_public.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/mrs_challenge_dev.h5`
- `gs://pwm-benchmark-datasets/challenge-data/v1.0/mrs_challenge_hidden.h5`

**Gallery images:** Served from GCS at `gs://pwm-benchmark-datasets/img/benchmark_gallery/mrs/`.

---

## 6. Comprehensive Assessment

**Status:** PASS

MRS is correctly formulated as a spectral decomposition problem where the FID contains superimposed Lorentzian/Voigt signals from multiple metabolites plus a flexible baseline, with Gaussian noise. The algorithm routing from AMARES/LCModel time-domain fitting through LCModel frequency-domain fitting to deep-learning quantification appropriately spans the clinical and research landscape. The mismatch parameters (linewidth, SNR, baseline amplitude, frequency shift) reflect the dominant sources of quantification error in clinical and research MRS settings.

---
*Comprehensive 6-point check by deep-check pipeline v3*
