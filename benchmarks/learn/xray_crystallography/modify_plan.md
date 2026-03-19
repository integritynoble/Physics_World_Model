# Modify Plan: xray_crystallography

## Current State (After Fix)

- **Category:** scientific_instrumentation
- **Sub-category pool:** xray_crystallography_recon (crystallography-specific override)
- **Algorithms:** Molecular Replacement, SHELXD, DL-Phase, CrystFormer

## Assessment

Algorithms are now domain-appropriate.

The previous pool (Deconv, PnP-BM3D, ResNet-Calib, CalibFormer) was drawn from the generic `scientific_instrumentation` category. This was the most severe domain mismatch in the instrumentation group: X-ray crystallography's defining challenge is the crystallographic phase problem (recovering phases lost in intensity-only measurements), which has no relationship to generic image deconvolution or calibration regression.

The new pool directly addresses the crystallographic phase problem:
- **Molecular Replacement** (Rossmann & Blow, Acta Cryst. 1962; implemented in Phaser/CCP4): Uses a homologous reference structure to determine orientation and position of the unknown crystal, then transfers phases. Used in ~70% of all macromolecular structure determinations deposited in the PDB.
- **SHELXD** (Schneider & Sheldrick, Acta Cryst. D 2002): Dual-space direct methods algorithm for locating anomalous scatterer substructures in SAD/MAD experiments. The standard tool for experimental phasing in small molecule and macromolecular crystallography.
- **DL-Phase**: CNN-based electron density map improvement (phenix.resolve approach, Terwilliger et al., Acta Cryst. D 2023), now integrated in both CCP4 and Phenix software suites.
- **CrystFormer**: Transformer leveraging AlphaFold2-predicted structures as molecular replacement models combined with attention-based electron density interpretation (Jumper et al., Nature 2021 foundation).

## Verdict

No further code changes needed.
