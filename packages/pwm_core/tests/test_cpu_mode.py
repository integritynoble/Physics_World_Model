"""Failing tests for cpu_mode portfolio flag and device propagation.

These tests are written BEFORE the implementation exists (TDD).
Tests 1, 3, and 4 are expected to FAIL until the feature is implemented.
Test 2 may pass or fail depending on current behaviour.
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Minimal physics stub
# ---------------------------------------------------------------------------

class _PhysicsStub:
    def __init__(self, shape=(8, 8)):
        self.x_shape = shape
        self.y_shape = shape
        self.all_linear = True
        self.mask = np.ones(shape, dtype=np.float32)
        self.n_bands = 4

    def forward(self, x):
        return x.flatten()[: np.prod(self.y_shape)].reshape(self.y_shape)

    def adjoint(self, y):
        return y.reshape(self.x_shape)

    def info(self):
        return {
            "modality": "cassi",
            "mask": self.mask,
            "n_bands": self.n_bands,
            "x_shape": self.x_shape,
        }


# ---------------------------------------------------------------------------
# Test 1 – cpu_mode=True injects device="cpu" into the cfg passed to run_gap_tv
# ---------------------------------------------------------------------------

def test_cpu_mode_injects_device_cpu(monkeypatch):
    """When cpu_mode=True, run_portfolio must set device='cpu' in cfg before
    passing it to the underlying solver (run_gap_tv here)."""
    from pwm_core.recon import portfolio

    captured: dict = {}

    def fake_run_gap_tv(y, physics, cfg):
        captured["cfg"] = dict(cfg)
        return y.astype(np.float32), {"solver": "gap_tv_fake"}

    monkeypatch.setattr("pwm_core.recon.portfolio.run_gap_tv", fake_run_gap_tv)

    rng = np.random.default_rng(0)
    y = rng.random((8, 8)).astype(np.float32)
    physics = _PhysicsStub()

    portfolio.run_portfolio(y, physics, {"candidates": ["gap_tv"], "cpu_mode": True})

    assert "cfg" in captured, "run_gap_tv was never called"
    assert captured["cfg"].get("device") == "cpu", (
        f"Expected device='cpu' in cfg passed to run_gap_tv, got: {captured['cfg']}"
    )


# ---------------------------------------------------------------------------
# Test 2 – without cpu_mode the cfg is NOT forced to device="cpu"
# ---------------------------------------------------------------------------

def test_cpu_mode_false_does_not_force_device(monkeypatch):
    """When cpu_mode is absent, run_portfolio must NOT inject device='cpu'."""
    from pwm_core.recon import portfolio

    captured: dict = {}

    def fake_run_gap_tv(y, physics, cfg):
        captured["cfg"] = dict(cfg)
        return y.astype(np.float32), {"solver": "gap_tv_fake"}

    monkeypatch.setattr("pwm_core.recon.portfolio.run_gap_tv", fake_run_gap_tv)

    rng = np.random.default_rng(0)
    y = rng.random((8, 8)).astype(np.float32)
    physics = _PhysicsStub()

    portfolio.run_portfolio(y, physics, {"candidates": ["gap_tv"]})

    # Either run_gap_tv was not called (fallback path) OR device was not set to "cpu"
    if "cfg" in captured:
        assert captured["cfg"].get("device") != "cpu", (
            "device should NOT be 'cpu' when cpu_mode is absent"
        )


# ---------------------------------------------------------------------------
# Test 3 – _run_operator_solver passes device from cfg to mst_recon_cassi
# ---------------------------------------------------------------------------

def test_mst_entry_passes_device_from_cfg(monkeypatch):
    """_run_operator_solver('mst', cfg) must forward cfg['device'] to
    mst_recon_cassi as a keyword argument."""
    import pwm_core.recon.mst as mst_module
    from pwm_core.recon import portfolio

    captured: dict = {}

    def fake_mst_recon_cassi(y, mask, n_bands, **kwargs):
        captured["kwargs"] = kwargs
        # Return a plausible 3-D cube
        return np.zeros((*y.shape, n_bands), dtype=np.float32)

    # Patch the symbol on the mst module so the local import inside portfolio
    # picks up the patched version (Python module cache).
    monkeypatch.setattr(mst_module, "mst_recon_cassi", fake_mst_recon_cassi)

    rng = np.random.default_rng(1)
    y = rng.random((8, 8)).astype(np.float32)
    physics = _PhysicsStub()

    portfolio._run_operator_solver(y, physics, "mst", {"device": "cpu"})

    assert "kwargs" in captured, "mst_recon_cassi was never called"
    assert captured["kwargs"].get("device") == "cpu", (
        f"Expected device='cpu' kwarg, got: {captured['kwargs']}"
    )


# ---------------------------------------------------------------------------
# Test 4 – run_hsi_sdecnn passes device from cfg to gap_sdecnn_cassi
# ---------------------------------------------------------------------------

def test_hsi_sdecnn_passes_device_from_cfg(monkeypatch):
    """run_hsi_sdecnn must forward cfg['device'] to gap_sdecnn_cassi."""
    import pwm_core.recon.hsi_sdecnn as hsi_module

    captured: dict = {}

    def fake_gap_sdecnn_cassi(y, mask, n_bands=28, weights_path=None,
                               iters=50, acc=1.0, **kwargs):
        captured["kwargs"] = kwargs
        return np.zeros((*y.shape, n_bands), dtype=np.float32)

    monkeypatch.setattr(hsi_module, "gap_sdecnn_cassi", fake_gap_sdecnn_cassi)

    rng = np.random.default_rng(2)
    y = rng.random((8, 8)).astype(np.float32)
    physics = _PhysicsStub()

    hsi_module.run_hsi_sdecnn(y, physics, {"device": "cpu"})

    assert "kwargs" in captured, "gap_sdecnn_cassi was never called"
    assert captured["kwargs"].get("device") == "cpu", (
        f"Expected device='cpu' kwarg, got: {captured['kwargs']}"
    )
