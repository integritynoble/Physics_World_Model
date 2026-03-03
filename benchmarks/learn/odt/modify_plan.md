# Modify Plan: odt

## Current State (After Fix)
- **Category:** coherent
- **Sub-category pool:** coherent (ODT-specific override)
- **Algorithms:** [Wolf FBP, Born-ADMM, ODT-Net, Rytov-Former]

## Assessment
Algorithms are now domain-appropriate.

The previous coherent pool (GS/HIO, prDeep, PhaseNet, LRGS) addressed generic coherent phase retrieval but missed ODT's 3D tomographic structure. The replacement algorithms directly target the ODT inverse problem:
- **Wolf FBP** — Wolf transform filtered back-projection under Born approximation (Wolf, Opt. Commun. 1969); canonical ODT reconstruction
- **Born-ADMM** — iterative ADMM with Born-approximation forward model and sparsity prior (Pham et al., Optica 2018)
- **ODT-Net** — end-to-end deep learning for holographic ODT (Lim et al., Optica 2019)
- **Rytov-Former** — vision transformer using Rytov-approximation physics as constraint (Chen et al., arXiv 2023)

## Verdict
No further code changes needed.
