# Benchmark Review -- dic (Differential Interference Contrast Microscopy)

**URL:** https://pwm.platformai.org/benchmark/dic
**Review Date:** 2026-03-03
**Modality ID:** dic
**Category:** Microscopy | Carrier: Photon | DAG: M --> C --> D

---

## 1. Platform Page Assessment

The benchmark page at https://pwm.platformai.org/benchmark/dic loads successfully
(HTTP 200, ~55 kB). The page title reads "Differential Interference Contrast
(DIC) -- Physics World Model" and describes DIC as a blind reconstruction task
under physics model mismatch.

**What the page provides:**

- Forward model pipeline: M (Modulation) --> C (Convolution) --> D (Detector),
  encoding the DIC shearing interferometry process as PSF convolution plus
  directional gradient modulation and sensor readout with gain and noise.
- Three evaluation tiers: Public (5 scenes, full ground truth), Dev (5 scenes,
  blind), and Hidden (5 scenes, server-side containerized evaluation).
- Composite scoring: 40% PSNR (normalised) + 40% SSIM + 20% Consistency
  (1 - ||y - H_hat x_hat|| / ||y||).
- Mismatch parameters with tier-specific ranges:
  - shear_amount: 80--140 nm (public), 76--136 nm (dev), 86--146 nm (hidden)
  - bias_retardation: -0.15 to 0.15 nm (all tiers)
  - prism_orientation: -0.6 to 1.2 deg (varies by tier)
- Leaderboard with 4 baseline entries: Restormer + gradient (0.724),
  CARE + gradient (0.671), PnP-FISTA + gradient (0.669), Richardson-Lucy +
  gradient (0.612). Public PSNR range: 25.5--34.5 dB, SSIM: 0.818--0.964.
- Challenge data in HDF5 format on GCS for public/dev tiers; Docker container
  submissions for hidden tier.

**Discrepancies vs local YAML config:**

The local `benchmarks/configs/dic.yaml` specifies simpler mismatch ranges
(shear_amount 50--200 nm, bias_retardation 0--0 nm, prism_orientation -3 to
3 deg) while the platform page shows narrower tier-specific ranges. The YAML
uses 64x64 image shapes whereas the expanded config (`dic_expanded.yaml`)
defines four size tiers (128, 256, 512, 1024). The platform leaderboard
includes "PolScope-Former + gradient" and other methods not present in the
local solver registry (which only lists Richardson-Lucy and CARE). These
discrepancies suggest the platform has evolved beyond the local configuration.

---

## 2. Physics and Forward Model Evaluation

DIC microscopy converts phase gradients into intensity contrast through
Nomarski prism shearing interferometry. The specimen is illuminated with
polarised light split into two sheared beams by a Wollaston/Nomarski prism;
the beams traverse slightly different optical paths through the sample, and
their recombination produces interference-based intensity contrast proportional
to the optical path length gradient along the shear direction.

**Forward model fidelity:**

The benchmark models DIC as a linear operator (y = Ax + n) with PSF
convolution via the `microscopy_psf` category module. This is a simplification.
The true DIC forward model is:

    I(x,y) = I_0 [1 + cos(dphi(x,y) + delta)]

where dphi is the phase gradient difference along the shear direction and
delta is the bias retardation. This is inherently nonlinear in the phase.
The benchmark partially captures this through its mismatch parameters
(shear_amount, bias_retardation, prism_orientation), but the underlying
operator is classified as `linear_operator`, which only approximates the
intensity-level restoration problem rather than the full phase-retrieval
inverse problem.

The `modify_plan.md` file in the local repository explicitly acknowledges
this gap: "the distinguishing reconstruction problem in DIC is quantitative
phase retrieval from the gradient-contrast image" and notes that algorithms
like Transport-of-Intensity Equation (TIE) solvers or Wiener deconvolution
with phase-gradient kernels would be more domain-appropriate.

**Noise model:** Poisson (photon counting) + Gaussian (read noise, dark
current). Physical parameters: sigma=2.0, read_noise_e=5.0, pixel_size_um=6.5.

---

## 3. Literature and State of the Art

Recent research in DIC phase retrieval and reconstruction includes:

- **Artifact-free phase reconstruction for DIC (Optics Express, March 2025):**
  A deep learning method using a trained Integral GAN that maps experimental
  differential phase to specimen phase, achieving quantitative optical thickness
  reconstruction with ~5 ms inference time for 512x512 images. This represents
  the current frontier for DIC-specific deep learning phase retrieval.

- **Rotational-diversity phase estimation (JOSA A):** Classical approach
  exploiting multiple DIC images at different prism orientations to decouple
  phase from shear direction, enabling orientation-independent phase recovery.

- **PhaseStain / Label-free DIC (Ounkomol et al., 2018):** CNN-based mapping
  from DIC to fluorescence-equivalent images, demonstrating that deep networks
  can learn the DIC-to-phase relationship end-to-end.

- **BioSR dataset (Qiao et al., Nature Methods):** While primarily SIM-focused,
  this benchmark dataset has become the de facto standard for evaluating
  microscopy reconstruction algorithms. The PWM platform page references it
  for DIC evaluation context, though BioSR does not contain DIC-specific data.

- **Physics-informed diffusion models (2024):** Recent work on microscopy
  image reconstruction using physics-informed denoising diffusion probabilistic
  models (PI-DDPM) that incorporate forward model constraints, applicable to
  DIC and related modalities.

- **Computing metasurfaces for vectorial DIC (ACS Photonics):** Hardware-level
  advances enabling broad-band vectorial DIC without traditional Nomarski
  prisms, potentially changing the forward model assumptions.

The benchmark's current baselines (Restormer, CARE, PnP-FISTA, Richardson-Lucy)
are generic restoration methods. DIC-specific methods like TIE solvers,
gradient-integration phase retrievers, and the Integral GAN approach are
absent from the leaderboard.

---

## 4. Local Dataset and Code Status

**Dataset:** No local DIC dataset directory exists at `datasets/benchmark/dic/`.
The benchmark config specifies `data_source.fallback: generated` with
`synthetic_generator: shepp_logan`, meaning all current benchmark data is
synthetically generated rather than derived from real DIC microscopy
acquisitions. The `dataset_id` and `dataset_url` fields are empty. The
expanded config confirms the only data source is `dic_generated` (type:
generated) applying to all variants.

**Learning materials:** A complete set of 5 learning documents plus README
exists at `benchmarks/learn/dic/`:
- `README.md` (1,458 B) -- overview and navigation
- `01_physics_fundamentals.md` (2,234 B) -- photon physics, PSF, noise
- `02_forward_model.md` (2,704 B) -- linear operator model, mismatch params
- `03_reconstruction_algorithms.md` (2,158 B) -- solver table and details
- `04_pwm_benchmark.md` (2,310 B) -- tier structure, scoring, runner commands
- `05_hands_on_tutorial.md` (3,559 B) -- code walkthrough

**Code:** The benchmark uses `benchmarks/runners/run_expanded.py` with
`--modality dic`. The default solver is `dic_gradient_integration` (a
DIC-specific gradient integration approach), though the solver registry only
exposes Richardson-Lucy (CPU, 0 params) and CARE (GPU, 2M params). Metrics
are computed via `benchmarks/framework/metrics.py` (PSNR and SSIM).

**Maturity:** M0 (earliest stage). Tier A benchmark. Total expanded cases:
252 (B1: 12, B2: 80, B3: 80, B4: 80). Four noise levels (clean 60 dB,
low 40 dB, medium 30 dB, high 20 dB). Five mismatch levels (M0 nominal
through M4 adversarial).

---

## 5. Identified Gaps and Recommendations

**GAP-1: Synthetic-only data.** The benchmark relies entirely on Shepp-Logan
phantom-based synthetic generation. Real DIC microscopy data (e.g., biological
cells, tissue sections) would substantially improve ecological validity.
Recommendation: incorporate publicly available DIC datasets such as the
Allen Cell Collection or generate DIC images from existing fluorescence
datasets using computational DIC simulation.

**GAP-2: Phase retrieval not tested.** The current metrics (PSNR, SSIM)
evaluate intensity-domain restoration quality but do not assess quantitative
phase recovery, which is the primary scientific value of DIC reconstruction.
Recommendation: add phase-domain metrics (phase RMSE, phase SSIM) if
ground-truth phase maps are available.

**GAP-3: Missing DIC-specific solvers.** The solver registry lacks
domain-specific methods: TIE-based phase retrieval, Wiener deconvolution
with DIC kernel, gradient-integration methods (despite `dic_gradient_integration`
being the default solver), and recent deep learning approaches (Integral GAN,
PhaseStain). Recommendation: expand the solver pool with at least one
classical phase-retrieval method and one DIC-specific deep learning method.

**GAP-4: Bias retardation range is degenerate.** The local YAML config
specifies `bias_retardation` range as [0, 0], meaning no mismatch is
introduced for this parameter. The platform page shows [-0.15, 0.15] nm.
The local config should be updated to match.

**GAP-5: Image size mismatch.** The base config uses 64x64 which is
unrealistically small for DIC microscopy (typical: 512x512 to 2048x2048).
While the expanded config offers up to 1024x1024, the default benchmark
runs at 64x64, limiting practical relevance.

**GAP-6: Leaderboard/config divergence.** The platform leaderboard shows
methods (PolScope-Former, FLIM-Net, Phasor-FLIM, PnP-BM3D) that do not
appear in the local solver registry or config files. This suggests either
the platform has community-contributed methods not tracked locally, or there
is a synchronisation issue between the platform and the repository.

---

## 6. Summary Verdict

| Aspect                  | Rating      | Notes                                    |
|-------------------------|-------------|------------------------------------------|
| Platform page           | PASS        | Loads, leaderboard present, HDF5 hosted  |
| Physics model           | PARTIAL     | Linear approx of nonlinear phase problem |
| Forward model fidelity  | ADEQUATE    | PSF convolution captures restoration     |
| Data realism            | WEAK        | Synthetic only, no real DIC acquisitions  |
| Solver diversity        | WEAK        | Generic methods, no DIC-specific solvers  |
| Learning materials      | PASS        | Complete 5-document set with tutorial     |
| Metric coverage         | PARTIAL     | Intensity metrics only, no phase metrics  |
| Config consistency      | WARNING     | Local YAML vs platform ranges diverge    |
| Maturity                | M0          | Earliest stage, substantial gaps remain   |

The DIC benchmark is structurally complete (platform page, 3-tier evaluation,
learning materials, runner infrastructure) but scientifically shallow. It
treats DIC as a generic PSF deconvolution problem rather than the phase-gradient
retrieval modality it truly is. The reliance on synthetic Shepp-Logan phantoms,
absence of phase-domain evaluation, and lack of DIC-specific reconstruction
algorithms are the most significant gaps. Upgrading to M1 maturity should
prioritise: (1) real or realistic DIC data, (2) phase-retrieval metrics,
and (3) at least one domain-specific solver (e.g., TIE or gradient integration
with DIC kernel).

---

*Comprehensive 6-point review on 2026-03-03. Sources: [PWM platform](https://pwm.platformai.org/benchmark/dic), [Artifact-free DIC phase reconstruction (Optics Express 2025)](https://www.researchgate.net/publication/389013540), [BioSR dataset (figshare)](https://figshare.com/articles/dataset/BioSR/13264793), [Physics-informed DDPM for microscopy (PMC 2024)](https://pmc.ncbi.nlm.nih.gov/articles/PMC11683148/), [Vectorial DIC via metasurfaces (ACS Photonics)](https://pubs.acs.org/doi/10.1021/acsphotonics.2c00882), local repository configs and learning materials.*