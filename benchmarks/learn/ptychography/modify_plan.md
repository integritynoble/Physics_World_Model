# Modify Plan: ptychography

## Current State (After Fix)
- **Category:** coherent
- **Sub-category pool:** coherent (ptychography-specific override)
- **Algorithms:** [ePIE, sDR, PtychoNN, AutoPhaseNN]

## Assessment
Algorithms are now domain-appropriate.

The previous coherent pool (GS/HIO, prDeep, PhaseNet, LRGS) was appropriate for generic phase retrieval but not specific to ptychography's scanning multi-frame structure. The replacement algorithms are ptychography-native:
- **ePIE** — extended Ptychographic Iterative Engine, the standard algorithm for ptychography (Maiden & Rodenburg, Ultramicroscopy 2009); jointly updates object and probe
- **sDR** — semi-implicit Douglas-Rachford splitting for ptychography with convergence guarantees (Wen et al., SIAM J. Imaging Sci. 2012)
- **PtychoNN** — convolutional neural network for real-time ptychographic reconstruction without iterative phase retrieval (Cherukara et al., Appl. Phys. Lett. 2020)
- **AutoPhaseNN** — unsupervised/self-supervised deep learning that learns from diffraction data alone using the forward model as supervision (Wu et al., npj Comput. Mater. 2021)

## Verdict
No further code changes needed.
