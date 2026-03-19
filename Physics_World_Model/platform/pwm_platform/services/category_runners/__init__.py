"""Category runner registry.

Maps category_module strings to CategoryRunner instances.
"""

from __future__ import annotations

from typing import Optional

from ._base import CategoryRunner, MethodResult
from .ct_radon import CTRadonRunner
from .mri_kspace import MRIKspaceRunner
from .microscopy_psf import MicroscopyPSFRunner
from .electron_ctf import ElectronCTFRunner
from .compressive_mask import CompressiveMaskRunner
from .remote_sensing import RemoteSensingRunner
from .scanning_probe import ScanningProbeRunner

_RUNNERS = {
    "medical_ct_radon": CTRadonRunner(),
    "medical_mri_kspace": MRIKspaceRunner(),
    "microscopy_psf": MicroscopyPSFRunner(),
    "electron_ctf": ElectronCTFRunner(),
    "compressive_mask": CompressiveMaskRunner(),
    "remote_sensing_sar": RemoteSensingRunner(),
    "scanning_probe": ScanningProbeRunner(),
}

# Default fallback runner (most generic physics)
_DEFAULT = MicroscopyPSFRunner()


def get_runner(category_module: str) -> CategoryRunner:
    """Get the runner for a category_module string."""
    return _RUNNERS.get(category_module, _DEFAULT)


__all__ = [
    "CategoryRunner",
    "MethodResult",
    "get_runner",
]
