# Modify Plan: particle_calorimetry

## Current State
- **Category:** experimental_science
- **Carrier:** Particle
- **Score key:** experimental_science
- **Algorithms:**
  1. Tikhonov (Classical) -- Analytical baseline
  2. PnP-RED (PnP) -- Romano et al., IEEE TIP 2017
  3. ResUNet (Deep Learning) -- Residual U-Net baseline
  4. SwinIR (Transformer) -- Liang et al., ICCVW 2021

## Assessment

Particle calorimetry involves reconstructing particle energy and shower profiles from calorimeter cell readings in high-energy physics detectors (e.g., at ATLAS, CMS). The category `experimental_science` is appropriate. The leaderboard methods (Sci-Former, ResUNet, Tikhonov, PnP-BM3D) are generic but acceptable.

However, more domain-specific algorithms exist:
- CaloGAN (de Oliveira et al., PRD 2018) -- GAN-based shower simulation
- CaloFlow (Krause & Shih, PRD 2023) -- normalizing flow for calorimeter showers
- Graph Neural Network (Qasim et al., EPJC 2019) -- GNN for calorimeter clustering

The current generic algorithms (Tikhonov, PnP-RED, ResUNet, SwinIR) are not wrong for a reconstruction benchmark. They represent the standard inverse-problem solver classes. The mismatch is mild -- these are reasonable baselines.

## Required Changes

No code changes needed. The generic experimental_science algorithms are acceptable for a calorimeter energy reconstruction benchmark.
