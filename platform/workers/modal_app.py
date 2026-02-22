# platform/workers/modal_app.py
"""Modal GPU workers for PWM Platform.

These functions run on Modal's GPU infrastructure. They receive an ExperimentSpec
dict, run pwm_core, and return the result dict with artifacts base64-encoded.
The Celery worker on the CPU server decodes the artifacts and saves them to
/data/workspace/runbundles/{run_id}/.

No cloud storage credentials needed -- results returned in-memory as dicts.
"""
import pathlib
import modal
from modal import Image

# Path to the locally-built pwm_core wheel (produced by `python -m build`)
_PWM_WHEEL = pathlib.Path(__file__).parent.parent.parent / "packages" / "pwm_core" / "dist" / "pwm_core-0.2.1-py3-none-any.whl"
# Fallback: pre-built wheel in /tmp (produced by CI or local build)
if not _PWM_WHEEL.exists():
    _PWM_WHEEL = pathlib.Path("/tmp/pwm_wheels/pwm_core-0.2.1-py3-none-any.whl")

pwm_image = (
    Image.debian_slim(python_version="3.11")
    .pip_install("uv")
    .add_local_file(str(_PWM_WHEEL), "/tmp/pwm_core-0.2.1-py3-none-any.whl", copy=True)
    .run_commands(
        "uv pip install --system 'torch' 'deepinv>=0.2.0' /tmp/pwm_core-0.2.1-py3-none-any.whl"
    )
)

app = modal.App("pwm-platform")


@app.function(
    image=pwm_image,
    gpu="T4",
    timeout=600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def simulate_gpu(spec_dict: dict) -> dict:
    """Run simulate() on GPU, return result dict with base64-encoded artifacts.

    The CPU-side Celery worker will decode and save artifacts to local disk.
    No GCS or cloud storage required.
    """
    import tempfile
    import os
    import base64

    with tempfile.TemporaryDirectory() as tmp:
        from pwm_core.api.endpoints import run
        result = run(spec=spec_dict, out_dir=tmp)

        # Serialize RunBundle files for transport back to CPU server
        bundle_dir = result.get("runbundle_path", tmp)
        artifacts: dict[str, str] = {}
        if bundle_dir and os.path.isdir(bundle_dir):
            for fname in os.listdir(bundle_dir):
                fpath = os.path.join(bundle_dir, fname)
                if os.path.isfile(fpath) and os.path.getsize(fpath) < 200 * 1024 * 1024:
                    with open(fpath, "rb") as f:
                        artifacts[fname] = base64.b64encode(f.read()).decode()

        result["_artifacts_b64"] = artifacts
        result.pop("runbundle_path", None)  # Path is not valid on the CPU server
        return result


@app.function(
    image=pwm_image,
    gpu="T4",
    timeout=600,
    retries=modal.Retries(max_retries=2, backoff_coefficient=2.0),
)
def reconstruct_gpu(spec_dict: dict) -> dict:
    """Run reconstruct() on GPU, return result dict with base64-encoded artifacts."""
    import tempfile
    import os
    import base64

    with tempfile.TemporaryDirectory() as tmp:
        from pwm_core.api.endpoints import run
        # Force reconstruction-only task
        spec_dict = {**spec_dict, "states": {**spec_dict.get("states", {}),
                     "task": {"kind": "recon_analyze"}}}
        result = run(spec=spec_dict, out_dir=tmp)

        bundle_dir = result.get("runbundle_path", tmp)
        artifacts: dict[str, str] = {}
        if bundle_dir and os.path.isdir(bundle_dir):
            for fname in os.listdir(bundle_dir):
                fpath = os.path.join(bundle_dir, fname)
                if os.path.isfile(fpath) and os.path.getsize(fpath) < 200 * 1024 * 1024:
                    with open(fpath, "rb") as f:
                        artifacts[fname] = base64.b64encode(f.read()).decode()

        result["_artifacts_b64"] = artifacts
        result.pop("runbundle_path", None)
        return result
