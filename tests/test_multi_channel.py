"""Tests for multi-channel sensor support."""

import numpy as np
import pytest

from pwm_core.graph.primitives import PhotonSensor, CoilSensor, GenericSensor


class TestPhotonSensorMultiChannel:
    def test_single_channel_shape(self):
        sensor = PhotonSensor(params={"quantum_efficiency": 0.9, "gain": 1.0})
        x = np.random.rand(32, 32)
        y = sensor.forward(x)
        assert y.shape == (32, 32)

    def test_multi_channel_shape(self):
        sensor = PhotonSensor(params={
            "quantum_efficiency": 0.9, "gain": 1.0, "n_channels": 3
        })
        x = np.random.rand(32, 32)
        y = sensor.forward(x)
        assert y.shape == (3, 32, 32)

    def test_multi_channel_with_responses(self):
        sensor = PhotonSensor(params={
            "quantum_efficiency": 0.9, "gain": 1.0,
            "n_channels": 3, "channel_responses": [0.8, 1.0, 0.6]
        })
        x = np.ones((16, 16))
        y = sensor.forward(x)
        assert y.shape == (3, 16, 16)
        # First channel scaled by 0.8
        np.testing.assert_allclose(y[0], x * 0.9 * 1.0 * 0.8)

    def test_adjoint_reduces_channels(self):
        sensor = PhotonSensor(params={
            "quantum_efficiency": 0.9, "gain": 1.0, "n_channels": 3
        })
        y = np.random.rand(3, 16, 16)
        x_adj = sensor.adjoint(y)
        assert x_adj.shape == (16, 16)


class TestCoilSensorMultiCoil:
    def test_single_coil_shape(self):
        sensor = CoilSensor(params={"sensitivity": 1.0})
        x = np.random.rand(32, 32)
        y = sensor.forward(x)
        assert y.shape == (32, 32)
        assert np.iscomplexobj(y)

    def test_multi_coil_shape(self):
        sensor = CoilSensor(params={"sensitivity": 1.0, "n_coils": 8})
        x = np.random.rand(32, 32)
        y = sensor.forward(x)
        assert y.shape == (8, 32, 32)
        assert np.iscomplexobj(y)

    def test_adjoint_reduces_coils(self):
        sensor = CoilSensor(params={"sensitivity": 1.0, "n_coils": 8})
        y = (np.random.rand(8, 16, 16) + 1j * np.random.rand(8, 16, 16))
        x_adj = sensor.adjoint(y)
        assert x_adj.shape == (16, 16)
        assert np.iscomplexobj(x_adj)


class TestGenericSensorMultiChannel:
    def test_single_channel(self):
        sensor = GenericSensor(params={"gain": 2.0})
        x = np.ones((16, 16))
        y = sensor.forward(x)
        assert y.shape == (16, 16)
        np.testing.assert_allclose(y, 2.0)

    def test_multi_channel(self):
        sensor = GenericSensor(params={"gain": 2.0, "n_channels": 4})
        x = np.ones((16, 16))
        y = sensor.forward(x)
        assert y.shape == (4, 16, 16)

    def test_adjoint_multi_channel(self):
        sensor = GenericSensor(params={"gain": 2.0, "n_channels": 4})
        y = np.ones((4, 16, 16))
        x_adj = sensor.adjoint(y)
        assert x_adj.shape == (16, 16)


class TestMultiChannelNoRegression:
    def test_all_sensors_default_single_channel(self):
        """Default params should give single-channel (no regression)."""
        for SensorCls, extra_params in [
            (PhotonSensor, {"quantum_efficiency": 0.9, "gain": 1.0}),
            (CoilSensor, {"sensitivity": 1.0}),
            (GenericSensor, {"gain": 1.0}),
        ]:
            sensor = SensorCls(params=extra_params)
            x = np.random.rand(16, 16)
            y = sensor.forward(x)
            assert y.ndim == 2, f"{SensorCls.__name__} default broke: shape={y.shape}"
