"""pwm_core.viewer.components

Small Streamlit helpers for the viewer.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import streamlit as st


def load_manifest(rb: Path) -> Dict[str, Any]:
    p = rb / "runbundle_manifest.json"
    if not p.exists():
        st.error(f"Missing manifest: {p}")
        st.stop()
    return json.loads(p.read_text(encoding="utf-8"))


def load_array_if_exists(p: Path) -> Optional[np.ndarray]:
    if p.exists():
        try:
            return np.load(p, allow_pickle=True)
        except Exception as e:
            st.warning(f"Failed to load {p.name}: {e}")
            return None
    return None


def show_manifest_summary(manifest: Dict[str, Any]):
    st.caption(f"Run ID: `{manifest.get('run_id','')}`  |  Spec: `{manifest.get('spec_version','')}`")
    data = manifest.get("data", {})
    st.write("**Data mode:**", data.get("mode", "unknown"))
    if data.get("mode") == "reference":
        st.write("**Data reference:**", data.get("path", ""))


def show_json_block(obj: Any):
    st.json(obj)


def _to_image(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    if x.ndim == 3:
        # show first slice / channel
        x = x[..., 0]
    x = x.astype(np.float32)
    x = x - x.min()
    if x.max() > 0:
        x = x / x.max()
    return x


def show_side_by_side_images(xgt: Optional[np.ndarray], xhat: Optional[np.ndarray], y: Optional[np.ndarray]):
    cols = st.columns(3)
    with cols[0]:
        st.subheader("Ground Truth (if any)")
        if xgt is None:
            st.info("No x_gt.npy")
        else:
            st.image(_to_image(xgt), clamp=True)
    with cols[1]:
        st.subheader("Reconstruction")
        if xhat is None:
            st.info("No x_hat.npy")
        else:
            st.image(_to_image(xhat), clamp=True)
    with cols[2]:
        st.subheader("Measurement y")
        if y is None:
            st.info("No y.npy")
        else:
            st.image(_to_image(y), clamp=True)


def show_markdown_report(report_path: Path):
    if report_path.exists():
        st.markdown(report_path.read_text(encoding="utf-8"))
    else:
        st.info("No report.md found.")
