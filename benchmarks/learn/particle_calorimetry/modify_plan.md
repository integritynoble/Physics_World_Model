# Modify Plan: particle_calorimetry

## Current State (After Fix)

- **Category:** experimental_science
- **Sub-category pool:** hep_calorimetry (particle physics override)
- **Algorithms:** PandoraPFA, GARFIELD++, GravNet, CaloDiffusion

## Assessment

Algorithms are now domain-appropriate.

The previous pool (Tikhonov, PnP-RED, ResUNet, SwinIR) served as generic experimental science reconstruction baselines. While not incorrect for the inverse-problem framework, none of these algorithms are from the high-energy physics literature and would not be recognized by the particle physics community benchmarking against ATLAS or CMS reconstruction pipelines.

The new pool reflects the actual state of calorimeter reconstruction:
- **PandoraPFA** (Thomson, Eur. Phys. J. C 2009): The gold-standard particle flow algorithm used at ILC, ILD, and adapted for CMS HGCAL. Clusters calorimeter hits and tracks into particle candidates.
- **GARFIELD++** (Veenhof et al., CERN 2017): CERN's primary detector simulation and response modeling framework, used for calibration reference reconstruction.
- **GravNet** (Qasim et al., Eur. Phys. J. C 2019): Graph neural network with dynamic graph construction for HGCAL cluster reconstruction — current state of the art for 3D calorimeter hit clustering.
- **CaloDiffusion** (Acosta et al., arXiv:2308.03876 2023): Score-based diffusion model for calorimeter shower reconstruction, achieving GEANT4-quality shower shapes at 10,000× faster inference.

## Verdict

No further code changes needed.
