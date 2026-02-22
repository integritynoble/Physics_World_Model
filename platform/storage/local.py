"""Local filesystem storage for RunBundles, datasets, and manifests.
No cloud dependency required.
"""
import base64
import os
import pathlib
from typing import Optional


class LocalStorage:
    def __init__(self, workspace: Optional[str] = None):
        self.workspace = pathlib.Path(
            workspace or os.environ.get("PWM_WORKSPACE_DIR", "/data/workspace")
        )

    def runbundle_path(self, run_id: str) -> str:
        return str(self.workspace / "runbundles" / run_id)

    def dataset_path(self, modality: str, kind: str, version: str) -> str:
        return str(self.workspace / "datasets" / kind / modality / version)

    def bootstrap_proposal_path(self, proposal_id: str) -> str:
        return str(self.workspace / "bootstrap_proposals" / proposal_id)

    def ensure_runbundle_dir(self, run_id: str) -> str:
        path = pathlib.Path(self.runbundle_path(run_id))
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def ensure_dataset_dir(self, modality: str, kind: str, version: str) -> str:
        path = pathlib.Path(self.dataset_path(modality, kind, version))
        path.mkdir(parents=True, exist_ok=True)
        return str(path)

    def save_artifacts(self, bundle_dir: str, artifacts_b64: dict) -> None:
        """Write base64-encoded artifacts (returned from Modal) to local disk."""
        for fname, b64data in artifacts_b64.items():
            dest = pathlib.Path(bundle_dir) / fname
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(base64.b64decode(b64data))

    def list_runbundles(self) -> list:
        base = self.workspace / "runbundles"
        if not base.exists():
            return []
        return [d.name for d in base.iterdir() if d.is_dir()]

    def runbundle_exists(self, run_id: str) -> bool:
        return pathlib.Path(self.runbundle_path(run_id)).is_dir()
