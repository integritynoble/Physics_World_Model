"""Solvers for Coded Aperture Compressive Temporal Imaging (CACTI) (cacti).

Each function wraps a solver from pwm_core.recon.*.
All follow the standard interface: fn(y, operator, cfg) -> x_hat
"""

from __future__ import annotations
import importlib
import numpy as np
from typing import Any, Dict, Optional

MODALITY_ID = "cacti"
DISPLAY_NAME = "Coded Aperture Compressive Temporal Imaging (CACTI)"


# Solver registry for cacti
SOLVERS = {
    "traditional_cpu": {
        "name": "GAP-TV",
        "module": "pwm_core.recon.gap_tv",
        "function": "run_gap_tv",
        "gpu": False,
        "reference": "Yuan et al. 2016",
    },
    "best_quality": {
        "name": "EfficientSCI",
        "module": "pwm_core.recon.efficientsci",
        "function": "run_efficientsci",
        "gpu": False,
        "reference": "Wang et al. CVPR 2023",
    },
    "famous_dl": {
        "name": "ELP-Unfolding",
        "module": "pwm_core.recon.elp_unfolding",
        "function": "run_elp_unfolding",
        "gpu": False,
        "reference": "Yang et al. ECCV 2022",
    },
    "small_gpu": {
        "name": "EfficientSCI-T",
        "module": "pwm_core.recon.efficientsci",
        "function": "run_efficientsci",
        "gpu": False,
        "reference": "",
    },
    "pnp_ffdnet": {
        "name": "PnP-FFDNet",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "pnp_ffdnet_cacti",
        "gpu": False,
        "reference": "Yuan et al., CVPR 2020",
    },
    "hisvit9": {
        "name": "HiSViT-9",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "hisvit_cacti",
        "gpu": True,
        "reference": "Chen et al., ICCV 2023",
    },
    "hisvit13": {
        "name": "HiSViT-13",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "hisvit_cacti",
        "gpu": True,
        "reference": "Wang et al., ECCV 2024",
    },
    "desci": {
        "name": "DeSCI",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "desci_cacti",
        "gpu": False,
        "reference": "Liu et al., TPAMI 2019",
    },
    "birnat": {
        "name": "BIRNAT",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "birnat_cacti",
        "gpu": True,
        "reference": "Cheng et al., ECCV 2020",
    },
    "revsci": {
        "name": "RevSCI",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "revsci_cacti",
        "gpu": True,
        "reference": "Cheng et al., CVPR 2021",
    },
    "metasci": {
        "name": "MetaSCI",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "metasci_cacti",
        "gpu": True,
        "reference": "Wang et al., CVPR 2021",
    },
    "stformer": {
        "name": "STFormer",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "stformer_cacti",
        "gpu": True,
        "reference": "Wang et al., TPAMI 2023",
    },
    "gap_net": {
        "name": "GAP-Net",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "gap_net_cacti",
        "gpu": True,
        "reference": "Meng et al., IJCV 2023",
    },
    "dun_3dunet": {
        "name": "DUN-3DUnet",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "dun_3dunet_cacti",
        "gpu": True,
        "reference": "Wu et al., ICCV 2021",
    },
    "ctm_sci": {
        "name": "CTM-SCI",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "ctm_sci_cacti",
        "gpu": True,
        "reference": "Zheng et al., ICCV 2023",
    },
    "e2e_cnn": {
        "name": "E2E-CNN",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "e2e_cnn_cacti",
        "gpu": True,
        "reference": "Qiao et al., APL Photonics 2020",
    },
    "pnp_fastdvdnet": {
        "name": "PnP-FastDVDnet",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "pnp_fastdvdnet_cacti",
        "gpu": False,
        "reference": "Yuan et al., TPAMI 2022",
    },
    "efficientsci_l": {
        "name": "EfficientSCI-L",
        "module": "pwm_core.recon.efficientsci",
        "function": "run_efficientsci",
        "gpu": False,
        "reference": "Wang et al., CVPR 2023",
    },
    "mambasci": {
        "name": "MambaSCI",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "mambasci_cacti",
        "gpu": True,
        "reference": "Pan et al., NeurIPS 2024",
    },
    "q_sci": {
        "name": "Q-SCI",
        "module": "pwm_core.recon.cacti_solvers",
        "function": "q_sci_cacti",
        "gpu": True,
        "reference": "Cao et al., ECCV 2024",
    },
}


def _load_fn(solver_key: str):
    """Dynamically load solver function."""
    spec = SOLVERS[solver_key]
    mod = importlib.import_module(spec["module"])
    return getattr(mod, spec["function"])


def run_solver(solver_key: str, y: np.ndarray, operator: Any = None,
               cfg: Optional[Dict] = None) -> np.ndarray:
    """Run a solver by key.

    Args:
        solver_key: One of the keys in SOLVERS dict
        y: Measurement data (float32)
        operator: Forward operator
        cfg: Hyperparameters (optional)

    Returns:
        x_hat: Reconstructed signal
    """
    if solver_key not in SOLVERS:
        raise ValueError(f"Unknown solver {solver_key}. Available: {list(SOLVERS.keys())}")
    fn = _load_fn(solver_key)
    result = fn(y.astype(np.float32), operator, cfg or {})
    if isinstance(result, tuple):
        return np.asarray(result[0], dtype=np.float32)
    return np.asarray(result, dtype=np.float32)


def list_solvers():
    """List all available solvers for cacti."""
    return [(k, v) for k, v in SOLVERS.items()]


def run_traditional_cpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-TV. CPU only.
    Reference: Yuan et al. 2016
    """
    return run_solver("traditional_cpu", y, operator, cfg)

def run_best_quality(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EfficientSCI. CPU only.
    Reference: Wang et al. CVPR 2023
    """
    return run_solver("best_quality", y, operator, cfg)

def run_famous_dl(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """ELP-Unfolding. CPU only.
    Reference: Yang et al. ECCV 2022
    """
    return run_solver("famous_dl", y, operator, cfg)

def run_small_gpu(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EfficientSCI-T. CPU only.
    """
    return run_solver("small_gpu", y, operator, cfg)

def run_pnp_ffdnet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PnP-FFDNet. CPU only.
    Reference: Yuan et al., CVPR 2020
    """
    return run_solver("pnp_ffdnet", y, operator, cfg)

def run_hisvit9(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HiSViT-9. GPU required.
    Reference: Chen et al., ICCV 2023
    """
    return run_solver("hisvit9", y, operator, cfg)

def run_hisvit13(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """HiSViT-13. GPU required.
    Reference: Wang et al., ECCV 2024
    """
    return run_solver("hisvit13", y, operator, cfg)

def run_desci(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DeSCI. CPU only.
    Reference: Liu et al., TPAMI 2019
    """
    return run_solver("desci", y, operator, cfg)

def run_birnat(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """BIRNAT. GPU required.
    Reference: Cheng et al., ECCV 2020
    """
    return run_solver("birnat", y, operator, cfg)

def run_revsci(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """RevSCI. GPU required.
    Reference: Cheng et al., CVPR 2021
    """
    return run_solver("revsci", y, operator, cfg)

def run_metasci(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MetaSCI. GPU required.
    Reference: Wang et al., CVPR 2021
    """
    return run_solver("metasci", y, operator, cfg)

def run_stformer(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """STFormer. GPU required.
    Reference: Wang et al., TPAMI 2023
    """
    return run_solver("stformer", y, operator, cfg)

def run_gap_net(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """GAP-Net. GPU required.
    Reference: Meng et al., IJCV 2023
    """
    return run_solver("gap_net", y, operator, cfg)

def run_dun_3dunet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """DUN-3DUnet. GPU required.
    Reference: Wu et al., ICCV 2021
    """
    return run_solver("dun_3dunet", y, operator, cfg)

def run_ctm_sci(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """CTM-SCI. GPU required.
    Reference: Zheng et al., ICCV 2023
    """
    return run_solver("ctm_sci", y, operator, cfg)

def run_e2e_cnn(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """E2E-CNN. GPU required.
    Reference: Qiao et al., APL Photonics 2020
    """
    return run_solver("e2e_cnn", y, operator, cfg)

def run_pnp_fastdvdnet(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """PnP-FastDVDnet. CPU only.
    Reference: Yuan et al., TPAMI 2022
    """
    return run_solver("pnp_fastdvdnet", y, operator, cfg)

def run_efficientsci_l(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """EfficientSCI-L. CPU only.
    Reference: Wang et al., CVPR 2023
    """
    return run_solver("efficientsci_l", y, operator, cfg)

def run_mambasci(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """MambaSCI. GPU required.
    Reference: Pan et al., NeurIPS 2024
    """
    return run_solver("mambasci", y, operator, cfg)

def run_q_sci(y: np.ndarray, operator: Any = None, cfg: Optional[Dict] = None) -> np.ndarray:
    """Q-SCI. GPU required.
    Reference: Cao et al., ECCV 2024
    """
    return run_solver("q_sci", y, operator, cfg)
