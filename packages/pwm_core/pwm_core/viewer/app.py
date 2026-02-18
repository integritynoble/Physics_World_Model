"""pwm_core.viewer.app

Minimal Streamlit app to view a RunBundle.

Usage:
  streamlit run pwm_core/viewer/app.py -- <path_to_runbundle>

This starter expects a runbundle_manifest.json and optional files:
- artifacts/recon/x_hat.npy
- artifacts/sim/y.npy
- report.md
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

from pwm_core.viewer.components import (
    load_manifest,
    load_array_if_exists,
    show_manifest_summary,
    show_side_by_side_images,
    show_json_block,
    show_markdown_report,
)


def main():
    st.set_page_config(page_title="PWM RunBundle Viewer", layout="wide")
    st.title("PWM RunBundle Viewer")

    if len(sys.argv) < 2:
        st.info("Usage: streamlit run app.py -- <runbundle_dir>")
        st.stop()

    rb = Path(sys.argv[1]).expanduser().resolve()
    manifest = load_manifest(rb)
    show_manifest_summary(manifest)

    # Load arrays
    y = load_array_if_exists(rb / "artifacts" / "sim" / "y.npy")
    xhat = load_array_if_exists(rb / "artifacts" / "recon" / "x_hat.npy")
    xgt = load_array_if_exists(rb / "artifacts" / "sim" / "x_gt.npy")

    st.header("Images")
    show_side_by_side_images(xgt, xhat, y)

    st.header("Diagnosis / Metrics")
    diag_path = rb / "artifacts" / "analysis" / "diagnosis.json"
    met_path = rb / "artifacts" / "analysis" / "metrics.json"
    diag = json.loads(diag_path.read_text()) if diag_path.exists() else {}
    met = json.loads(met_path.read_text()) if met_path.exists() else {}

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Diagnosis")
        show_json_block(diag)
    with col2:
        st.subheader("Metrics")
        show_json_block(met)

    st.header("Report")
    report_path = rb / "artifacts" / "analysis" / "report.md"
    show_markdown_report(report_path)

    st.header("Manifest (raw)")
    show_json_block(manifest)


if __name__ == "__main__":
    main()
