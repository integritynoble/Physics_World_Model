"""Spec Primitives — the 11 building blocks of any imaging forward model."""

from __future__ import annotations

SPEC_PRIMITIVES = {
    "P": {
        "symbol": "P",
        "name": "Propagation",
        "description": "Free-space or medium propagation kernel (Fresnel, Rayleigh-Sommerfeld).",
    },
    "M": {
        "symbol": "M",
        "name": "Mask / Modulation",
        "description": "Spatial or spatio-temporal amplitude modulation (coded aperture, SLM pattern).",
    },
    "Pi": {
        "symbol": "\u03a0",
        "name": "Projection",
        "description": "Geometric projection operator (Radon transform, fan-beam, cone-beam).",
    },
    "F": {
        "symbol": "F",
        "name": "Fourier Sampling",
        "description": "Sampling in the Fourier / k-space domain (MRI, ptychography).",
    },
    "C": {
        "symbol": "C",
        "name": "Convolution",
        "description": "Shift-invariant convolution with a point-spread function (PSF).",
    },
    "Sigma": {
        "symbol": "\u03a3",
        "name": "Summation / Integration",
        "description": "Summation along a physical dimension (spectral, temporal, angular).",
    },
    "D": {
        "symbol": "D",
        "name": "Detector",
        "description": "Sensor readout with gain g and noise model \u03b7 (Gaussian, Poisson, mixed).",
    },
    "S": {
        "symbol": "S",
        "name": "Structured Illumination",
        "description": "Patterned illumination (block, Hadamard, random) applied to the scene.",
    },
    "W": {
        "symbol": "W",
        "name": "Wavelength Dispersion",
        "description": "Spectral dispersion element (prism, grating) with shift \u03b1 and aperture a.",
    },
    "R": {
        "symbol": "R",
        "name": "Rotation / Motion",
        "description": "Sample or gantry rotation (CT, electron tomography).",
    },
    "Lambda": {
        "symbol": "\u039b",
        "name": "Wavelength Selection",
        "description": "Spectral filter or monochromator selecting a wavelength band.",
    },
}
