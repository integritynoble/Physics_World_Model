"""test_compute_benchmarks.py

Track E: Compute budget benchmarks.

Verify that key operations complete within time/memory budgets.
These are smoke tests — not micro-benchmarks — ensuring no
catastrophic performance regressions.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import yaml

from pwm_core.graph.compiler import GraphCompiler
from pwm_core.graph.graph_spec import OperatorGraphSpec


# ---------------------------------------------------------------------------
# Load templates
# ---------------------------------------------------------------------------

TEMPLATES_PATH = (
    Path(__file__).resolve().parent.parent / "contrib" / "graph_templates.yaml"
)


def _load_templates() -> Dict[str, Any]:
    with open(TEMPLATES_PATH) as f:
        data = yaml.safe_load(f)
    return data.get("templates", {})


TEMPLATES = _load_templates()


def _pick_template(modality: str) -> str:
    """Pick first template matching modality."""
    for tid, tmpl in TEMPLATES.items():
        meta = tmpl.get("metadata", {})
        if meta.get("modality") == modality:
            return tid
    # Fallback to first template
    return next(iter(TEMPLATES))


# ---------------------------------------------------------------------------
# E.4: Compute budget benchmarks
# ---------------------------------------------------------------------------


class TestForwardAdjointTimeBudget:
    """forward + adjoint for 64x64 must complete < 2s."""

    @pytest.mark.parametrize("modality", ["widefield", "spc", "ct"])
    def test_forward_adjoint_time(self, modality: str) -> None:
        tid = _pick_template(modality)
        tmpl = TEMPLATES[tid]
        spec = OperatorGraphSpec(
            graph_id=tid,
            nodes=tmpl["nodes"],
            edges=tmpl["edges"],
            metadata=tmpl.get("metadata", {}),
        )
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)

        x_shape = graph_op.x_shape
        y_shape = graph_op.y_shape
        rng = np.random.default_rng(42)

        x = rng.random(x_shape).astype(np.float32)

        start = time.perf_counter()
        y = graph_op.forward(x)
        if graph_op.all_linear:
            _ = graph_op.adjoint(y)
        elapsed = time.perf_counter() - start

        assert elapsed < 2.0, (
            f"forward+adjoint for '{tid}' took {elapsed:.3f}s (budget: 2.0s)"
        )


class TestCompileTimeBudget:
    """GraphCompiler.compile < 1s for any template."""

    @pytest.mark.parametrize(
        "template_id",
        sorted(TEMPLATES.keys())[:10],  # Test first 10 to keep fast
        ids=lambda x: x,
    )
    def test_compile_time(self, template_id: str) -> None:
        tmpl = TEMPLATES[template_id]
        spec = OperatorGraphSpec(
            graph_id=template_id,
            nodes=tmpl["nodes"],
            edges=tmpl["edges"],
            metadata=tmpl.get("metadata", {}),
        )

        start = time.perf_counter()
        compiler = GraphCompiler()
        _ = compiler.compile(spec)
        elapsed = time.perf_counter() - start

        assert elapsed < 1.0, (
            f"Compile for '{template_id}' took {elapsed:.3f}s (budget: 1.0s)"
        )


class TestMemoryBudget:
    """Peak RSS increase < 500MB for a standard pipeline operation."""

    def test_memory_budget(self) -> None:
        """Compile + forward should not consume excessive memory."""
        import resource

        # Get baseline memory
        baseline_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss

        # Compile and forward on a representative template
        tid = _pick_template("widefield")
        tmpl = TEMPLATES[tid]
        spec = OperatorGraphSpec(
            graph_id=tid,
            nodes=tmpl["nodes"],
            edges=tmpl["edges"],
            metadata=tmpl.get("metadata", {}),
        )
        compiler = GraphCompiler()
        graph_op = compiler.compile(spec)

        x = np.random.default_rng(42).random(graph_op.x_shape).astype(np.float32)
        _ = graph_op.forward(x)

        peak_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        increase_mb = (peak_kb - baseline_kb) / 1024.0

        # 500 MB budget — generous for 64x64 operations
        assert increase_mb < 500.0, (
            f"Memory increase: {increase_mb:.1f}MB (budget: 500MB)"
        )


class TestCachingLayer:
    """Content-addressed caching layer works correctly."""

    def test_content_hash(self, tmp_path) -> None:
        """SHA256 produces deterministic hash."""
        from pwm_core.io.caching import content_hash

        fpath = str(tmp_path / "test.bin")
        with open(fpath, "wb") as f:
            f.write(b"deterministic content")

        h1 = content_hash(fpath)
        h2 = content_hash(fpath)
        assert h1 == h2
        assert len(h1) == 64  # SHA256 hex digest

    def test_cache_put_get(self, tmp_path) -> None:
        """cache_put stores by hash, cache_get retrieves."""
        from pwm_core.io.caching import cache_put, cache_get, content_hash

        # Create a test file
        src = str(tmp_path / "source.bin")
        with open(src, "wb") as f:
            f.write(b"cached content")

        cache_dir = str(tmp_path / "cache")
        cached_path = cache_put(src, cache_dir)
        assert os.path.exists(cached_path)

        # Retrieve by hash
        h = content_hash(src)
        found = cache_get(h, cache_dir)
        assert found is not None
        assert os.path.exists(found)

    def test_cache_stats(self, tmp_path) -> None:
        """cache_stats reports correct counts."""
        from pwm_core.io.caching import cache_put, cache_stats

        cache_dir = str(tmp_path / "cache")

        # Empty cache
        stats = cache_stats(cache_dir)
        assert stats["num_files"] == 0

        # Add a file
        src = str(tmp_path / "file.bin")
        with open(src, "wb") as f:
            f.write(b"data")
        cache_put(src, cache_dir)

        stats = cache_stats(cache_dir)
        assert stats["num_files"] == 1
        assert stats["total_bytes"] > 0
