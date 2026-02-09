"""
pwm_core.agents.physics_stage_visualizer
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Deterministic physics-stage visualizer that produces before/after/noise
images for each element in an imaging system's optical chain.

This module is used for debugging, teaching, and pre-flight inspection.
Given a ground-truth signal ``x_gt``, it walks the element chain and
applies each transfer function and noise source *independently*, producing
a visualization record per element.

Design invariants
-----------------
* All random operations use a seeded ``numpy.random.Generator`` so that
  outputs are fully reproducible given the same seed.
* Transfer functions are simplified analytical approximations, not the
  full physics operators (those live in the operator modules). The goal
  is *visualization*, not bit-exact forward modeling.
* Complex-valued signals are handled by the ``CoherentVisualizer`` subclass,
  which additionally tracks magnitude and phase.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from .contracts import (
    ElementSpec,
    ImagingSystem,
    NoiseKind,
    TransferKind,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class ElementVisualization:
    """Visualization record for a single optical element.

    Attributes
    ----------
    element_name : str
        Human-readable name of the element.
    before : np.ndarray
        Signal *before* this element's transfer function is applied.
    after : np.ndarray
        Signal *after* the deterministic transfer function, before noise.
    after_noisy : np.ndarray
        Signal after transfer *and* noise injection.
    noise_map : np.ndarray
        The additive noise contribution (``after_noisy - after``).
    metadata : dict
        Element-specific metadata (local SNR, throughput, transfer kind,
        noise kinds, parameters used, etc.).
    """

    element_name: str
    before: np.ndarray
    after: np.ndarray
    after_noisy: np.ndarray
    noise_map: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# PhysicsStageVisualizer
# ---------------------------------------------------------------------------


class PhysicsStageVisualizer:
    """Walk an imaging system's element chain and produce per-element
    before/after/noise visualizations.

    The visualizer applies simplified transfer functions and noise models
    deterministically, using a seeded RNG for all stochastic operations.
    """

    def visualize_chain(
        self,
        system: ImagingSystem,
        x_gt: np.ndarray,
        seed: int = 42,
    ) -> List[ElementVisualization]:
        """Produce a visualization for every element in the imaging chain.

        Parameters
        ----------
        system : ImagingSystem
            The imaging system whose element chain will be walked.
        x_gt : np.ndarray
            Ground-truth signal. Shape should be compatible with the
            system's ``signal_dims``. Typically (H, W) for 2D or
            (C, H, W) / (H, W, C) for spectral/volumetric data.
        seed : int
            Random seed for reproducible noise injection.

        Returns
        -------
        list[ElementVisualization]
            One visualization record per element, in chain order.
        """
        rng = np.random.default_rng(seed)
        visualizations: List[ElementVisualization] = []

        # Current signal propagating through the chain
        current = x_gt.astype(np.float64, copy=True)

        for element in system.elements:
            before = current.copy()

            # Apply the deterministic transfer function
            after = self._apply_transfer(element, current)

            # Apply noise per the element's noise_kinds
            after_noisy = self._apply_noise(element, after, rng)

            # Compute noise map
            noise_map = after_noisy - after

            # Compute local SNR
            local_snr = self._compute_local_snr(after, noise_map)

            metadata: Dict[str, Any] = {
                "element_type": element.element_type,
                "transfer_kind": element.transfer_kind.value,
                "noise_kinds": [nk.value for nk in element.noise_kinds],
                "throughput": element.throughput,
                "parameters": dict(element.parameters),
                "local_snr_db": local_snr,
                "signal_range": [
                    float(np.min(after)),
                    float(np.max(after)),
                ],
                "noise_std": float(np.std(noise_map)) if noise_map.size > 0 else 0.0,
            }

            visualizations.append(
                ElementVisualization(
                    element_name=element.name,
                    before=before,
                    after=after,
                    after_noisy=after_noisy,
                    noise_map=noise_map,
                    metadata=metadata,
                )
            )

            # Propagate the noisy signal to the next element
            current = after_noisy.copy()

        return visualizations

    # ------------------------------------------------------------------
    # Transfer function dispatch
    # ------------------------------------------------------------------

    def _apply_transfer(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Apply a simplified transfer function based on the element's
        ``transfer_kind``.

        Parameters
        ----------
        element : ElementSpec
            The optical element specification.
        signal : np.ndarray
            Input signal.

        Returns
        -------
        np.ndarray
            Transformed signal (same dtype, may differ in shape for
            integration).
        """
        kind = element.transfer_kind

        if kind == TransferKind.convolution:
            return self._transfer_convolution(element, signal)
        elif kind == TransferKind.modulation:
            return self._transfer_modulation(element, signal)
        elif kind == TransferKind.dispersion:
            return self._transfer_dispersion(element, signal)
        elif kind == TransferKind.integration:
            return self._transfer_integration(element, signal)
        elif kind == TransferKind.identity:
            return signal.copy()
        elif kind == TransferKind.projection:
            return self._transfer_projection(element, signal)
        elif kind == TransferKind.interference:
            return self._transfer_throughput_scale(element, signal)
        elif kind == TransferKind.propagation:
            return self._transfer_throughput_scale(element, signal)
        elif kind == TransferKind.sampling:
            return self._transfer_throughput_scale(element, signal)
        elif kind == TransferKind.nonlinear:
            return self._transfer_throughput_scale(element, signal)
        else:
            logger.warning(
                "Unknown transfer kind '%s' for element '%s'; "
                "applying throughput scaling only.",
                kind,
                element.name,
            )
            return self._transfer_throughput_scale(element, signal)

    def _transfer_convolution(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Apply Gaussian blur with sigma from element parameters.

        Looks for ``sigma`` or ``psf_sigma`` in the element's parameters.
        Falls back to ``sigma=1.0`` if not specified.
        """
        sigma = element.parameters.get(
            "sigma",
            element.parameters.get("psf_sigma", 1.0),
        )
        sigma = float(sigma)

        if sigma <= 0:
            return signal * element.throughput

        # Apply Gaussian blur via FFT-based convolution for efficiency
        blurred = self._gaussian_blur(signal, sigma)
        return blurred * element.throughput

    def _transfer_modulation(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Multiply signal by a modulation pattern.

        If the element is a mask (element_type == "mask"), generates a
        deterministic binary random pattern based on the element name hash.
        Otherwise applies throughput scaling.
        """
        if element.element_type == "mask":
            # Generate a deterministic binary mask from the element name
            mask_seed = abs(hash(element.name)) % (2**31)
            mask_rng = np.random.default_rng(mask_seed)
            mask_ratio = element.parameters.get("density", 0.5)
            mask = (mask_rng.random(signal.shape) < mask_ratio).astype(
                signal.dtype
            )
            return signal * mask * element.throughput
        elif element.element_type == "modulator":
            # Sinusoidal modulation pattern
            freq = element.parameters.get("frequency", 10.0)
            # Apply along the last spatial axis
            x_coords = np.linspace(0, 2 * np.pi * freq, signal.shape[-1])
            pattern = 0.5 * (1.0 + np.sin(x_coords))
            return signal * pattern * element.throughput
        else:
            return signal * element.throughput

    def _transfer_dispersion(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Simulate spectral dispersion by shifting channels.

        For signals with >=3 dimensions, treats the first axis as the
        spectral axis and shifts each channel by an increasing offset.
        """
        step = element.parameters.get("dispersion_step", 1.0)
        step = float(step)

        if signal.ndim < 3:
            # Cannot disperse a 2D signal; just apply throughput
            return signal * element.throughput

        dispersed = np.zeros_like(signal)
        n_channels = signal.shape[0]

        for ch in range(n_channels):
            shift = int(round(ch * step))
            if signal.ndim == 3:
                # (C, H, W) -> shift along W axis
                dispersed[ch] = np.roll(signal[ch], shift, axis=-1)
            else:
                dispersed[ch] = np.roll(signal[ch], shift, axis=-1)

        return dispersed * element.throughput

    def _transfer_integration(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Sum along the spectral (first) axis.

        Models detector integration that collapses a spectral cube to a
        2D measurement. If the signal is already 2D, returns a copy.
        """
        if signal.ndim >= 3:
            integrated = np.sum(signal, axis=0)
        else:
            integrated = signal.copy()

        return integrated * element.throughput

    def _transfer_projection(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Simplified projection (e.g., Radon-like line integrals).

        For visualization purposes, sums along axis 0 as a proxy for
        a single projection angle.
        """
        if signal.ndim >= 2:
            projected = np.sum(signal, axis=0, keepdims=True)
        else:
            projected = signal.copy()

        return projected * element.throughput

    def _transfer_throughput_scale(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Default passthrough with throughput scaling only."""
        return signal * element.throughput

    # ------------------------------------------------------------------
    # Noise injection
    # ------------------------------------------------------------------

    def _apply_noise(
        self,
        element: ElementSpec,
        signal: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Inject noise according to the element's noise_kinds.

        Multiple noise sources are applied additively in sequence.

        Parameters
        ----------
        element : ElementSpec
            Element specification with noise kinds.
        signal : np.ndarray
            Clean (post-transfer) signal.
        rng : numpy.random.Generator
            Seeded RNG for reproducibility.

        Returns
        -------
        np.ndarray
            Signal with noise added.
        """
        noisy = signal.copy()

        for noise_kind in element.noise_kinds:
            if noise_kind == NoiseKind.none:
                continue
            elif noise_kind == NoiseKind.shot_poisson:
                noisy = self._noise_shot_poisson(noisy, rng)
            elif noise_kind == NoiseKind.read_gaussian:
                noisy = self._noise_read_gaussian(noisy, element, rng)
            elif noise_kind == NoiseKind.quantization:
                noisy = self._noise_quantization(noisy, element)
            elif noise_kind == NoiseKind.fixed_pattern:
                noisy = self._noise_fixed_pattern(noisy, element, rng)
            elif noise_kind == NoiseKind.thermal:
                noisy = self._noise_thermal(noisy, element, rng)
            else:
                # aberration, alignment, acoustic -- treat as small
                # additive Gaussian for visualization purposes
                sigma = element.parameters.get("noise_sigma", 0.01)
                noise = rng.normal(0, float(sigma), noisy.shape)
                noisy = noisy + noise

        return noisy

    def _noise_shot_poisson(
        self, signal: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Apply shot noise (Poisson statistics).

        Negative signal values are clipped to zero before applying
        Poisson sampling, since Poisson requires non-negative rates.
        """
        clipped = np.clip(signal, 0, None)
        # Avoid overflow for very large values by capping
        max_photons = 1e8
        capped = np.minimum(clipped, max_photons)
        noisy = rng.poisson(capped).astype(signal.dtype)
        return noisy

    def _noise_read_gaussian(
        self,
        signal: np.ndarray,
        element: ElementSpec,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply additive read noise (Gaussian).

        Read noise sigma is taken from the element's parameters
        (``read_noise_sigma`` or ``read_noise``), defaulting to 5.0
        electrons.
        """
        read_noise = element.parameters.get(
            "read_noise_sigma",
            element.parameters.get("read_noise", 5.0),
        )
        read_noise = float(read_noise)
        noise = rng.normal(0, read_noise, signal.shape)
        return signal + noise

    def _noise_quantization(
        self, signal: np.ndarray, element: ElementSpec
    ) -> np.ndarray:
        """Apply quantization noise (rounding to integer ADC levels)."""
        n_bits = element.parameters.get("adc_bits", 12)
        n_levels = 2 ** int(n_bits)
        sig_range = np.max(signal) - np.min(signal)
        if sig_range < 1e-12:
            return signal.copy()
        step = sig_range / n_levels
        quantized = np.round(signal / step) * step
        return quantized

    def _noise_fixed_pattern(
        self,
        signal: np.ndarray,
        element: ElementSpec,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply fixed-pattern noise (spatially correlated, static)."""
        fpn_sigma = element.parameters.get("fpn_sigma", 0.02)
        # Fixed pattern is deterministic per element, so use element-name
        # seeded RNG
        fpn_seed = abs(hash(element.name + "_fpn")) % (2**31)
        fpn_rng = np.random.default_rng(fpn_seed)
        pattern = fpn_rng.normal(1.0, float(fpn_sigma), signal.shape)
        return signal * pattern

    def _noise_thermal(
        self,
        signal: np.ndarray,
        element: ElementSpec,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Apply thermal (dark current) noise."""
        dark_current = element.parameters.get("dark_current", 0.1)
        exposure = element.parameters.get("exposure_s", 1.0)
        dark_signal = float(dark_current) * float(exposure)
        noise = rng.poisson(dark_signal, signal.shape).astype(signal.dtype)
        return signal + noise

    # ------------------------------------------------------------------
    # Utility methods
    # ------------------------------------------------------------------

    def _gaussian_blur(
        self, signal: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Apply Gaussian blur via FFT convolution.

        Works for 2D and higher-dimensional signals by blurring the
        last two spatial dimensions.
        """
        if sigma <= 0 or signal.size == 0:
            return signal.copy()

        original_shape = signal.shape

        if signal.ndim == 1:
            # 1D Gaussian blur
            n = signal.shape[0]
            x = np.fft.fftfreq(n)
            kernel_fft = np.exp(-2.0 * (np.pi * sigma * x) ** 2)
            blurred = np.real(
                np.fft.ifft(np.fft.fft(signal) * kernel_fft)
            )
            return blurred

        # For 2D and higher, blur the last two spatial dimensions
        if signal.ndim == 2:
            return self._gaussian_blur_2d(signal, sigma)

        # Reshape to (batch, H, W) for batch processing
        batch_shape = signal.shape[:-2]
        h, w = signal.shape[-2], signal.shape[-1]
        flat = signal.reshape(-1, h, w)

        result = np.empty_like(flat)
        for i in range(flat.shape[0]):
            result[i] = self._gaussian_blur_2d(flat[i], sigma)

        return result.reshape(original_shape)

    def _gaussian_blur_2d(
        self, img: np.ndarray, sigma: float
    ) -> np.ndarray:
        """Apply 2D Gaussian blur to a single (H, W) image via FFT."""
        h, w = img.shape
        fy = np.fft.fftfreq(h).reshape(-1, 1)
        fx = np.fft.fftfreq(w).reshape(1, -1)
        kernel_fft = np.exp(
            -2.0 * (np.pi * sigma) ** 2 * (fy**2 + fx**2)
        )
        blurred = np.real(
            np.fft.ifft2(np.fft.fft2(img) * kernel_fft)
        )
        return blurred

    def _compute_local_snr(
        self, clean: np.ndarray, noise_map: np.ndarray
    ) -> float:
        """Compute local SNR in dB.

        SNR = 10 * log10(signal_power / noise_power).
        Returns 0.0 if noise power is zero or signal is empty.
        """
        if clean.size == 0:
            return 0.0

        signal_power = float(np.mean(clean**2))
        noise_power = float(np.mean(noise_map**2))

        if noise_power < 1e-30:
            return 100.0  # effectively infinite SNR
        if signal_power < 1e-30:
            return 0.0

        return float(10.0 * np.log10(signal_power / noise_power))


# ---------------------------------------------------------------------------
# CoherentVisualizer
# ---------------------------------------------------------------------------


class CoherentVisualizer(PhysicsStageVisualizer):
    """Extension for complex-valued (coherent) signals.

    In addition to the standard before/after/noise images, this visualizer
    tracks magnitude and phase separately and stores them in the element
    metadata.

    Suitable for holography, ptychography, and other coherent imaging
    modalities where the signal is complex-valued.
    """

    def visualize_chain(
        self,
        system: ImagingSystem,
        x_gt: np.ndarray,
        seed: int = 42,
    ) -> List[ElementVisualization]:
        """Produce visualizations with additional magnitude/phase tracking.

        Parameters
        ----------
        system : ImagingSystem
            The imaging system to visualize.
        x_gt : np.ndarray
            Ground-truth signal. May be complex-valued.
        seed : int
            Random seed for reproducibility.

        Returns
        -------
        list[ElementVisualization]
            Visualization records with magnitude and phase in metadata.
        """
        # Ensure the signal is complex
        if not np.iscomplexobj(x_gt):
            x_gt = x_gt.astype(np.complex128)

        rng = np.random.default_rng(seed)
        visualizations: List[ElementVisualization] = []

        current = x_gt.copy()

        for element in system.elements:
            before = current.copy()

            # Apply transfer function (operates on complex signal)
            after = self._apply_transfer_complex(element, current)

            # Apply noise (operates on magnitude, preserves phase)
            after_noisy = self._apply_noise_complex(element, after, rng)

            # Compute noise map (in complex domain)
            noise_map = after_noisy - after

            # Compute local SNR on magnitude
            local_snr = self._compute_local_snr(
                np.abs(after), np.abs(noise_map)
            )

            # Extract magnitude and phase for metadata
            before_mag = np.abs(before)
            before_phase = np.angle(before)
            after_mag = np.abs(after)
            after_phase = np.angle(after)
            after_noisy_mag = np.abs(after_noisy)
            after_noisy_phase = np.angle(after_noisy)

            metadata: Dict[str, Any] = {
                "element_type": element.element_type,
                "transfer_kind": element.transfer_kind.value,
                "noise_kinds": [nk.value for nk in element.noise_kinds],
                "throughput": element.throughput,
                "parameters": dict(element.parameters),
                "local_snr_db": local_snr,
                "signal_range": [
                    float(np.min(after_mag)),
                    float(np.max(after_mag)),
                ],
                "noise_std": float(np.std(np.abs(noise_map))),
                "is_complex": True,
                "before_magnitude": before_mag,
                "before_phase": before_phase,
                "after_magnitude": after_mag,
                "after_phase": after_phase,
                "after_noisy_magnitude": after_noisy_mag,
                "after_noisy_phase": after_noisy_phase,
                "phase_range": [
                    float(np.min(after_phase)),
                    float(np.max(after_phase)),
                ],
            }

            visualizations.append(
                ElementVisualization(
                    element_name=element.name,
                    before=before,
                    after=after,
                    after_noisy=after_noisy,
                    noise_map=noise_map,
                    metadata=metadata,
                )
            )

            current = after_noisy.copy()

        return visualizations

    # ------------------------------------------------------------------
    # Complex-valued transfer functions
    # ------------------------------------------------------------------

    def _apply_transfer_complex(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Apply transfer function in the complex domain.

        For most transfer kinds, applies the standard real-valued
        transfer to the magnitude while preserving phase. For
        propagation and interference, handles complex field properly.
        """
        kind = element.transfer_kind

        if kind == TransferKind.propagation:
            return self._propagation_complex(element, signal)
        elif kind == TransferKind.interference:
            return self._interference_complex(element, signal)
        else:
            # Apply standard transfer to magnitude, preserve phase
            magnitude = np.abs(signal)
            phase = np.angle(signal)
            transformed_mag = np.abs(
                self._apply_transfer(element, magnitude)
            )
            return transformed_mag * np.exp(1j * phase)

    def _propagation_complex(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Simulate free-space propagation using angular spectrum method.

        Applies a quadratic phase factor in the Fourier domain.
        """
        wavelength = element.parameters.get("wavelength_nm", 550.0)
        distance = element.parameters.get("propagation_distance_um", 100.0)
        pixel_pitch = element.parameters.get("pixel_pitch_um", 1.0)

        wavelength_um = float(wavelength) / 1000.0
        distance_um = float(distance)
        dx = float(pixel_pitch)

        if signal.ndim < 2:
            return signal * element.throughput

        h, w = signal.shape[-2], signal.shape[-1]
        fy = np.fft.fftfreq(h, d=dx).reshape(-1, 1)
        fx = np.fft.fftfreq(w, d=dx).reshape(1, -1)

        # Angular spectrum transfer function
        k = 2.0 * np.pi / wavelength_um
        kz_sq = (k**2) - (2 * np.pi * fx) ** 2 - (2 * np.pi * fy) ** 2
        # Clamp to avoid evanescent waves
        kz_sq = np.maximum(kz_sq, 0)
        kz = np.sqrt(kz_sq)
        H = np.exp(1j * kz * distance_um)

        if signal.ndim == 2:
            propagated = np.fft.ifft2(np.fft.fft2(signal) * H)
        else:
            # Batch over leading dimensions
            batch_shape = signal.shape[:-2]
            flat = signal.reshape(-1, h, w)
            result = np.empty_like(flat)
            for i in range(flat.shape[0]):
                result[i] = np.fft.ifft2(np.fft.fft2(flat[i]) * H)
            propagated = result.reshape(signal.shape)

        return propagated * element.throughput

    def _interference_complex(
        self, element: ElementSpec, signal: np.ndarray
    ) -> np.ndarray:
        """Simulate interference with a reference beam.

        Models off-axis holography by adding a tilted plane wave
        reference and returning the resulting intensity.
        """
        tilt_deg = element.parameters.get("reference_tilt_deg", 5.0)
        ref_amplitude = element.parameters.get("reference_amplitude", 1.0)

        tilt_rad = float(tilt_deg) * np.pi / 180.0
        ref_amp = float(ref_amplitude)

        if signal.ndim < 2:
            # 1D interference
            n = signal.shape[-1]
            x = np.arange(n, dtype=np.float64)
            ref_wave = ref_amp * np.exp(
                1j * 2 * np.pi * np.sin(tilt_rad) * x / n
            )
            combined = signal + ref_wave
        else:
            h, w = signal.shape[-2], signal.shape[-1]
            x = np.arange(w, dtype=np.float64).reshape(1, -1)
            ref_wave = ref_amp * np.exp(
                1j * 2 * np.pi * np.sin(tilt_rad) * x / w
            )
            combined = signal + ref_wave

        return combined * element.throughput

    # ------------------------------------------------------------------
    # Complex noise injection
    # ------------------------------------------------------------------

    def _apply_noise_complex(
        self,
        element: ElementSpec,
        signal: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Inject noise into a complex-valued signal.

        Shot noise is applied to the magnitude (intensity), while read
        noise is applied independently to real and imaginary components.
        Other noise types fall back to magnitude-based injection with
        phase preservation.
        """
        noisy = signal.copy()

        for noise_kind in element.noise_kinds:
            if noise_kind == NoiseKind.none:
                continue
            elif noise_kind == NoiseKind.shot_poisson:
                # Shot noise on intensity, then take square root for amplitude
                intensity = np.abs(noisy) ** 2
                clipped = np.clip(intensity, 0, 1e8)
                noisy_intensity = rng.poisson(clipped).astype(np.float64)
                # Preserve phase, update magnitude
                phase = np.angle(noisy)
                new_mag = np.sqrt(np.maximum(noisy_intensity, 0))
                noisy = new_mag * np.exp(1j * phase)
            elif noise_kind == NoiseKind.read_gaussian:
                read_noise = element.parameters.get(
                    "read_noise_sigma",
                    element.parameters.get("read_noise", 5.0),
                )
                read_noise = float(read_noise)
                noise_real = rng.normal(0, read_noise, noisy.shape)
                noise_imag = rng.normal(0, read_noise, noisy.shape)
                noisy = noisy + noise_real + 1j * noise_imag
            else:
                # Generic small noise on real and imaginary parts
                sigma = element.parameters.get("noise_sigma", 0.01)
                noise = rng.normal(
                    0, float(sigma), noisy.shape
                ) + 1j * rng.normal(0, float(sigma), noisy.shape)
                noisy = noisy + noise

        return noisy
