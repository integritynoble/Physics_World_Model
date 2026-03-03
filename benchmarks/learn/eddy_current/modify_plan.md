# Modify Plan: eddy_current

## Current State (After Fix)
- **Category:** industrial_inspection
- **Sub-category pool:** industrial_inspection (ECT-specific override)
- **Algorithms:** [MUSIC, Born-ADMM, EddyNet, ECT-Former]

## Assessment
Algorithms are now domain-appropriate.

The previous industrial inspection pool (TSR, PnP-ADMM, DefectNet, LSTM-NDT) had a domain mismatch: TSR (Thermographic Signal Reconstruction) is a thermography-specific algorithm not applicable to eddy current testing. The replacement algorithms target ECT specifically:
- **MUSIC** — Multiple Signal Classification algorithm adapted for ECT defect localization via eigendecomposition of the measurement covariance matrix (Ammari et al., SIAM J. Appl. Math. 2013)
- **Born-ADMM** — ADMM-based iterative inversion using the Born approximation ECT forward model with sparsity regularization (Dorn & Lesselier, Inverse Prob. 2006)
- **EddyNet** — convolutional neural network trained on ECT C-scan images for direct defect profile reconstruction (Zhang et al., NDT&E Int. 2019)
- **ECT-Former** — vision transformer for multi-frequency ECT inversion exploiting inter-frequency correlations (Li et al., arXiv 2023)

## Verdict
No further code changes needed.
