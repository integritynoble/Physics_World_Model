"""pwm_core.core.simulator

Measurement simulator for PWM pipeline.

Generates test phantoms, applies forward operator, and adds noise
based on SensorState configuration.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np

from pwm_core.api.types import ExperimentSpec, SensorState
from pwm_core.physics.base import PhysicsOperator
from pwm_core.noise.gaussian import GaussianNoise
from pwm_core.noise.poisson import PoissonNoise
from pwm_core.noise.poisson_gaussian import PoissonGaussianNoise


def generate_phantom(
    spec: ExperimentSpec,
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Generate a test phantom (ground truth signal x).

    Creates a simple structured phantom based on the modality and dimensions.

    Args:
        spec: ExperimentSpec with physics dimensions.
        rng: Optional random generator for reproducibility.

    Returns:
        Ground truth signal x as numpy array.
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Get dimensions from spec
    dims = spec.states.physics.dims
    if dims is None:
        x_shape = (64, 64)
    elif isinstance(dims, dict):
        h = dims.get('H', dims.get('height', 64))
        w = dims.get('W', dims.get('width', 64))
        l = dims.get('L', dims.get('bands', None))
        if l is not None:
            x_shape = (int(h), int(w), int(l))
        else:
            x_shape = (int(h), int(w))
    elif isinstance(dims, (list, tuple)):
        x_shape = tuple(int(d) for d in dims)
    else:
        x_shape = (64, 64)

    modality = spec.states.physics.modality.lower()

    if modality == "cassi" and len(x_shape) == 3:
        # Hyperspectral phantom: spatial pattern with spectral variation
        return _generate_hyperspectral_phantom(x_shape, rng)
    else:
        # 2D spatial phantom with geometric features
        return _generate_2d_phantom(x_shape[:2], rng)


def _generate_2d_phantom(shape: Tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    """Generate a 2D phantom with circles and rectangles."""
    H, W = shape
    x = np.zeros((H, W), dtype=np.float32)

    # Background
    x[:] = 0.1

    # Add some circles
    yy, xx = np.ogrid[:H, :W]

    # Large circle in center
    cy, cx, r = H // 2, W // 2, min(H, W) // 4
    mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2
    x[mask] = 0.8

    # Smaller circle offset
    cy2, cx2, r2 = H // 3, 2 * W // 3, min(H, W) // 8
    mask2 = (yy - cy2) ** 2 + (xx - cx2) ** 2 < r2 ** 2
    x[mask2] = 0.6

    # Rectangle
    x[H // 4 : H // 4 + H // 8, W // 8 : W // 8 + W // 4] = 0.9

    # Add a bit of texture
    x += 0.05 * rng.random((H, W)).astype(np.float32)

    return np.clip(x, 0.0, 1.0)


def _generate_hyperspectral_phantom(
    shape: Tuple[int, int, int],
    rng: np.random.Generator
) -> np.ndarray:
    """Generate a hyperspectral phantom (H, W, L)."""
    H, W, L = shape
    x = np.zeros((H, W, L), dtype=np.float32)

    # Create spatial base
    base_2d = _generate_2d_phantom((H, W), rng)

    # Spectral signature per region
    for l in range(L):
        # Spectral modulation factor (smooth variation)
        spectral_weight = 0.5 + 0.5 * np.sin(2 * np.pi * l / L)
        x[:, :, l] = base_2d * spectral_weight

    # Add spectral variation in different regions
    yy, xx = np.ogrid[:H, :W]
    cy, cx, r = H // 2, W // 2, min(H, W) // 4
    center_mask = (yy - cy) ** 2 + (xx - cx) ** 2 < r ** 2

    for l in range(L):
        # Different spectral signature in center
        x[:, :, l][center_mask] *= (1.0 + 0.3 * np.cos(4 * np.pi * l / L))

    return np.clip(x, 0.0, 1.0)


def apply_noise(
    y_clean: np.ndarray,
    sensor: Optional[SensorState],
    rng: Optional[np.random.Generator] = None
) -> np.ndarray:
    """Apply noise to clean measurements based on SensorState config.

    Args:
        y_clean: Clean measurements from forward operator.
        sensor: SensorState with noise parameters.
        rng: Optional random generator.

    Returns:
        Noisy measurements.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Handle complex data: apply Gaussian noise to real and imag parts
    if np.iscomplexobj(y_clean):
        sigma = 0.02
        if sensor is not None and sensor.read_noise_sigma is not None:
            sigma = sensor.read_noise_sigma
        noise_real = rng.normal(0, sigma, y_clean.shape).astype(np.float32)
        noise_imag = rng.normal(0, sigma, y_clean.shape).astype(np.float32)
        return (y_clean + noise_real + 1j * noise_imag).astype(np.complex64)

    if sensor is None:
        # Default: mild Gaussian noise
        sigma = 0.02
        noise_model = GaussianNoise(noise_id="gaussian", params={"sigma": sigma})
        return noise_model.apply(y_clean, rng)

    # Determine noise model from sensor state
    has_shot = sensor.shot_noise is not None
    read_sigma = sensor.read_noise_sigma

    if has_shot and read_sigma is not None and read_sigma > 0:
        # Poisson-Gaussian mixed noise
        shot_config = sensor.shot_noise if isinstance(sensor.shot_noise, dict) else {}
        gain = shot_config.get("gain", 100.0)
        noise_model = PoissonGaussianNoise(
            noise_id="poisson_gaussian",
            params={"gain": gain, "sigma": read_sigma}
        )
        return noise_model.apply(y_clean, rng)

    elif has_shot:
        # Poisson only
        shot_config = sensor.shot_noise if isinstance(sensor.shot_noise, dict) else {}
        gain = shot_config.get("gain", 100.0)
        noise_model = PoissonNoise(noise_id="poisson", params={"gain": gain})
        return noise_model.apply(y_clean, rng)

    elif read_sigma is not None and read_sigma > 0:
        # Gaussian only
        noise_model = GaussianNoise(noise_id="gaussian", params={"sigma": read_sigma})
        return noise_model.apply(y_clean, rng)

    else:
        # Default mild noise
        sigma = 0.02
        noise_model = GaussianNoise(noise_id="gaussian", params={"sigma": sigma})
        return noise_model.apply(y_clean, rng)


def apply_sensor_effects(
    y: np.ndarray,
    sensor: Optional[SensorState]
) -> np.ndarray:
    """Apply additional sensor effects: saturation, quantization.

    Args:
        y: Measurements (possibly noisy).
        sensor: SensorState with sensor parameters.

    Returns:
        Measurements with sensor effects applied.
    """
    if sensor is None:
        return y

    result = y.copy()

    # Saturation
    if sensor.saturation_level is not None:
        result = np.clip(result, None, sensor.saturation_level)

    # Quantization (simulate ADC)
    if sensor.quantization_bits is not None:
        bits = sensor.quantization_bits
        levels = 2 ** bits
        # Normalize to [0, levels-1], round, then back
        y_min, y_max = result.min(), result.max()
        if y_max > y_min:
            result_norm = (result - y_min) / (y_max - y_min)
            result_quant = np.round(result_norm * (levels - 1)) / (levels - 1)
            result = result_quant * (y_max - y_min) + y_min

    return result.astype(np.float32)


def simulate_measurement(
    spec: ExperimentSpec,
    operator: PhysicsOperator,
    rng: Optional[np.random.Generator] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Simulate measurements given an ExperimentSpec and operator.

    Args:
        spec: ExperimentSpec with physics and sensor configuration.
        operator: Physics operator with forward model.
        rng: Optional random generator for reproducibility.

    Returns:
        Tuple of (x_true, y_noisy) where:
            x_true: Ground truth signal
            y_noisy: Noisy measurements
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # Generate ground truth
    x_true = generate_phantom(spec, rng)

    # Apply forward model
    y_clean = operator.forward(x_true)

    # Apply noise
    sensor = spec.states.sensor
    y_noisy = apply_noise(y_clean, sensor, rng)

    # Apply sensor effects (saturation, quantization)
    y_noisy = apply_sensor_effects(y_noisy, sensor)

    return x_true, y_noisy
