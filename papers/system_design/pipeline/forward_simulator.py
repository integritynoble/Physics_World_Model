"""Forward Simulator — executes each flowchart element in order.

Design principle: each element type has a dispatcher method.
The simulator walks the element graph in topological order,
applies noise and mismatch at each stage, and returns the
final measurement array.
"""
from __future__ import annotations
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter

from ..schemas import ForwardAction, FlowchartElement
from ..utils.noise import (
    apply_poisson_noise,
    apply_gaussian_noise,
    apply_dark_current,
)


class ForwardSimulator:
    """Simulates a full imaging forward model defined by a ForwardAction."""

    def simulate(
        self,
        action: ForwardAction,
        phantom: np.ndarray,
        modality: str = "",
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Execute all elements in topological order.

        Returns:
            (measurements, element_outputs) where element_outputs maps
            element_id → intermediate array for debugging.
        """
        # Build adjacency for topological sort
        order = self._topological_sort(action.elements)
        element_map = {el.id: el for el in action.elements}

        outputs: dict[str, np.ndarray] = {"_phantom": phantom}
        current = phantom.copy().astype(np.float64)

        for el_id in order:
            el = element_map[el_id]
            current = self._dispatch(el, current, phantom, modality)
            outputs[el_id] = current.copy()

        return current.astype(np.float32), outputs

    # ── Element dispatchers ────────────────────────────────────────────────────

    def _dispatch(
        self,
        el: FlowchartElement,
        signal: np.ndarray,
        phantom: np.ndarray,
        modality: str,
    ) -> np.ndarray:
        """Route to element-type handler."""
        handler = {
            "source":       self._handle_source,
            "interaction":  self._handle_interaction,
            "geometry":     self._handle_geometry,
            "detector":     self._handle_detector,
            "digitization": self._handle_digitization,
            "processing":   self._handle_processing,
            "other":        self._handle_other,
        }.get(el.type, self._handle_other)
        return handler(el, signal, phantom, modality)

    def _handle_source(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        """Source elements generate the incident field / beam."""
        flux = el.parameters.get("flux_photons_per_s", el.parameters.get("power_mW", 1e5))
        return np.full_like(phantom, float(flux), dtype=np.float64)

    def _handle_interaction(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        """Phantom-beam interaction (Beer-Lambert, Bloch, PSF convolution, etc.)."""
        model = el.parameters.get("model", "beer_lambert")
        if model == "beer_lambert":
            # I = I0 * exp(-mu * x)
            mu = el.parameters.get("mu_water_cm", 0.206)
            return signal * np.exp(-mu * phantom)
        elif model == "bloch_equations":
            # Simplified: signal proportional to magnetization
            T1 = el.parameters.get("T1_range_ms", [1000])[0] / 1000.0
            return signal * (1.0 - np.exp(-0.5 / T1)) * phantom
        else:
            # Generic multiplicative interaction
            return signal * phantom

    def _handle_geometry(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        """Geometric projection (Radon for CT, k-space sampling for MRI, etc.)."""
        scan_type = el.parameters.get("scan_type", el.parameters.get("trajectory", ""))

        if scan_type in ("parallel_beam", "fan_beam") or modality == "ct":
            return self._radon_transform(phantom, el.parameters)
        elif scan_type == "cartesian" or modality == "mri":
            return self._kspace_sample(phantom, el.parameters)
        else:
            # Identity pass-through (geometry already embedded in interaction)
            return signal

    def _handle_detector(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        """Apply detector noise models."""
        out = signal.copy()
        for noise_spec in el.noise:
            out = self._apply_noise(out, noise_spec.type, noise_spec.parameters)
        return np.maximum(out, 0.0)

    def _handle_digitization(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        """Quantize to ADC bit depth."""
        bit_depth = el.parameters.get("bit_depth", 12)
        max_val = signal.max()
        if max_val > 0:
            quantized = np.round(signal / max_val * (2**bit_depth - 1))
            return quantized / (2**bit_depth - 1) * max_val
        return signal

    def _handle_processing(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        """Generic processing (PSF convolution, lens, etc.)."""
        na   = el.parameters.get("NA", 0.0)
        wl   = el.parameters.get("emission_nm", el.parameters.get("wavelength_nm", 500))
        if na > 0:
            sigma_px = (wl / (2 * na)) / el.parameters.get("effective_pixel_nm", 100)
            return gaussian_filter(signal.astype(np.float64), sigma=max(sigma_px, 0.5))
        return signal

    def _handle_other(
        self, el: FlowchartElement, signal: np.ndarray,
        phantom: np.ndarray, modality: str
    ) -> np.ndarray:
        return signal

    # ── Noise helper ──────────────────────────────────────────────────────────

    def _apply_noise(
        self, signal: np.ndarray, noise_type: str, params: dict
    ) -> np.ndarray:
        if noise_type == "poisson":
            I0 = params.get("I0", params.get("incident_photons", 1e5))
            return apply_poisson_noise(signal, I0=float(I0))
        elif noise_type in ("gaussian", "thermal_johnson", "readout"):
            sigma = params.get("sigma_electrons", params.get("sigma", 1.0))
            return apply_gaussian_noise(signal, sigma=float(sigma))
        elif noise_type == "dark_current":
            rate = params.get("electrons_per_s", 0.1)
            exposure = params.get("exposure_s", 0.1)
            return apply_dark_current(signal, rate=float(rate), exposure_s=float(exposure))
        else:
            return signal  # unknown noise type → no-op

    # ── Geometry helpers ──────────────────────────────────────────────────────

    def _radon_transform(
        self, phantom: np.ndarray, params: dict
    ) -> np.ndarray:
        """Simple parallel-beam Radon transform via skimage."""
        try:
            from skimage.transform import radon
        except ImportError:
            return phantom  # fallback

        num_angles  = int(params.get("num_angles", 180))
        theta       = np.linspace(0.0, 180.0, num_angles, endpoint=False)
        sino = radon(phantom, theta=theta, circle=True)
        return sino.astype(np.float64)

    def _kspace_sample(
        self, phantom: np.ndarray, params: dict
    ) -> np.ndarray:
        """Accelerated Cartesian k-space sampling."""
        acceleration = int(params.get("acceleration_factor", 4))
        kspace = np.fft.fftshift(np.fft.fft2(phantom))
        # Simple regular undersampling mask with fully-sampled centre
        h, w = kspace.shape
        mask = np.zeros((h, w), dtype=bool)
        mask[::acceleration, :] = True
        # Always sample centre 8% of k-space
        centre = int(h * 0.08)
        mask[h//2 - centre:h//2 + centre, :] = True
        return (kspace * mask).astype(np.complex64)

    # ── Topological sort ──────────────────────────────────────────────────────

    def _topological_sort(self, elements: list[FlowchartElement]) -> list[str]:
        """Kahn's algorithm on the connects_to graph."""
        from collections import deque

        in_degree = {el.id: 0 for el in elements}
        adj: dict[str, list[str]] = {el.id: [] for el in elements}

        for el in elements:
            for nxt in el.connects_to:
                if nxt in adj:
                    adj[el.id].append(nxt)
                    in_degree[nxt] += 1

        queue = deque(eid for eid, deg in in_degree.items() if deg == 0)
        order: list[str] = []
        while queue:
            eid = queue.popleft()
            order.append(eid)
            for nxt in adj[eid]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        # Append any elements not reached (disconnected nodes)
        remaining = [el.id for el in elements if el.id not in order]
        return order + remaining
