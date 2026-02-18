"""pwm_core.counterfactual.base_generator
=========================================

Abstract base class for counterfactual pack generators.

Each modality (CASSI, SPC, CACTI) subclasses ``BaseCounterfactualGenerator``
and implements ``load_scenes``, ``forward_model``, ``get_param_config``, and
``get_red_team_configs``.
"""

from __future__ import annotations

import hashlib
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from pwm_core.counterfactual.schema import (
    CounterfactualPackManifest,
    ExpectedBaseline,
    MismatchConfig,
    NoiseConfig,
    RedTeamCategory,
    RegimeKind,
    ScenarioSpec,
    SplitKind,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared physics helpers
# ---------------------------------------------------------------------------


def add_poisson_gaussian_noise(
    y: np.ndarray,
    peak: float = 100_000,
    sigma: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Poisson + Gaussian noise to measurement.

    Args:
        y: clean measurement (any shape).
        peak: Poisson peak (photon count at max signal).
        sigma: Gaussian noise std (in normalized units).
        rng: NumPy Generator for reproducibility.

    Returns:
        Noisy measurement, clipped to >= 0, float32.
    """
    if rng is None:
        rng = np.random.default_rng()

    y = np.nan_to_num(y, nan=0.0, posinf=0.0, neginf=0.0)
    y = np.maximum(y, 0)
    y_max = float(np.max(y))
    if y_max <= 0:
        y_max = 1.0

    y_scaled = (y / y_max) * peak
    y_scaled = np.maximum(y_scaled, 0)

    y_poisson = rng.poisson(y_scaled.astype(np.int64)).astype(np.float64)
    y_noisy = y_poisson + rng.normal(0, sigma * peak, y_poisson.shape)
    y_noisy = y_noisy / peak * y_max

    return np.maximum(y_noisy, 0).astype(np.float32)


def add_gaussian_noise(
    y: np.ndarray,
    sigma: float = 0.01,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Add Gaussian-only noise to measurement."""
    if rng is None:
        rng = np.random.default_rng()
    return (y + sigma * rng.standard_normal(y.shape).astype(np.float32)).astype(
        np.float32
    )


def compute_psnr(x_gt: np.ndarray, x_rec: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio (dB)."""
    mse = float(np.mean((x_gt.astype(np.float64) - x_rec.astype(np.float64)) ** 2))
    if mse < 1e-15:
        return 100.0
    data_max = float(np.max(x_gt))
    if data_max < 1e-15:
        data_max = 1.0
    return float(10 * np.log10(data_max**2 / mse))


def compute_ssim(x_gt: np.ndarray, x_rec: np.ndarray) -> float:
    """Simple SSIM (luminance + contrast, single-channel average)."""
    x = x_gt.astype(np.float64).ravel()
    y = x_rec.astype(np.float64).ravel()
    mu_x, mu_y = x.mean(), y.mean()
    sig_x, sig_y = x.std(), y.std()
    sig_xy = ((x - mu_x) * (y - mu_y)).mean()
    c1 = (0.01 * float(np.max(x_gt))) ** 2
    c2 = (0.03 * float(np.max(x_gt))) ** 2
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sig_xy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sig_x**2 + sig_y**2 + c2)
    )
    return float(ssim_val)


def sha256_file(path: Path) -> str:
    """Compute SHA-256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------


class BaseCounterfactualGenerator(ABC):
    """Abstract base for counterfactual pack generation.

    Subclasses implement modality-specific loading, forward model, parameter
    configs, and red-team injection configurations.
    """

    def __init__(
        self,
        modality: str,
        pack_id: str,
        seed_public: int = 2026_02_18,
        seed_hidden: int = 9999_02_18,
    ):
        self.modality = modality
        self.pack_id = pack_id
        self.seed_public = seed_public
        self.seed_hidden = seed_hidden

    # ----- abstract interface -----

    @abstractmethod
    def load_scenes(self) -> List[Tuple[str, np.ndarray, np.ndarray]]:
        """Return list of (scene_id, x_gt, mask_or_operator).

        Each entry represents one source scene/image/clip with its
        ground truth and nominal measurement operator.
        """

    @abstractmethod
    def forward_model(
        self,
        x_gt: np.ndarray,
        mask: np.ndarray,
        mismatch_params: Dict[str, float],
        noise_config: Dict[str, float],
        rng: np.random.Generator,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply forward model with mismatch and noise.

        Returns:
            (y_meas, mask_corrupted) — measurement and the actually-used operator.
        """

    @abstractmethod
    def get_param_config(self, split: SplitKind) -> Dict[str, Dict[str, float]]:
        """Return parameter ranges for the given split.

        Returns dict mapping param_name -> {"min": ..., "max": ..., "unit": ...}.
        """

    @abstractmethod
    def get_red_team_configs(
        self, split: SplitKind
    ) -> Dict[str, Dict[str, Any]]:
        """Return red-team injection configs for gate_flip, oof, compute_trap.

        Returns dict mapping regime name -> injection config dict.
        """

    @abstractmethod
    def get_nominal_noise(self, split: SplitKind) -> Dict[str, float]:
        """Return default noise config for nominal scenarios."""

    def get_severity_fractions(self) -> List[float]:
        """Return severity fractions for single-param sweeps (0=min, 1=max)."""
        return [0.25, 0.5, 1.0]

    # ----- generation pipeline -----

    def generate_pack(self, out_dir: Path) -> CounterfactualPackManifest:
        """Generate the full counterfactual pack to *out_dir*."""
        out_dir.mkdir(parents=True, exist_ok=True)
        scenes = self.load_scenes()
        all_scenarios: List[ScenarioSpec] = []
        file_hashes: Dict[str, str] = {}

        for split in [SplitKind.probe, SplitKind.hidden]:
            seed = self.seed_public if split == SplitKind.probe else self.seed_hidden
            rng = np.random.default_rng(seed)
            cfg = self.get_param_config(split)

            split_dir = out_dir / split.value
            split_dir.mkdir(parents=True, exist_ok=True)

            for scene_id, x_gt, mask in scenes:
                # Nominal
                specs = self._generate_nominal(
                    scene_id, x_gt, mask, split, rng, split_dir, file_hashes
                )
                all_scenarios.extend(specs)

                # Single-param sweep
                specs = self._generate_single_param(
                    scene_id, x_gt, mask, split, cfg, rng, split_dir, file_hashes
                )
                all_scenarios.extend(specs)

                # Compound (3 draws)
                specs = self._generate_compound(
                    scene_id, x_gt, mask, split, cfg, rng, split_dir, file_hashes
                )
                all_scenarios.extend(specs)

                # Red-team scenarios
                specs = self._generate_red_team(
                    scene_id, x_gt, mask, split, cfg, rng, split_dir, file_hashes
                )
                all_scenarios.extend(specs)

            # Write split-level scenario list
            self._write_scenarios_json(
                split_dir,
                [s for s in all_scenarios if s.split == split],
                file_hashes,
            )

        manifest = CounterfactualPackManifest(
            pack_id=self.pack_id,
            modality=self.modality,
            seeds={"probe": self.seed_public, "hidden": self.seed_hidden},
            n_scenarios=len(all_scenarios),
            regimes=sorted({s.regime.value for s in all_scenarios}),
            solvers=[],
            scenarios=all_scenarios,
            expected_baselines=[],
            file_hashes=file_hashes,
        )

        manifest_path = out_dir / "manifest.json"
        manifest_path.write_text(
            manifest.model_dump_json(indent=2), encoding="utf-8"
        )
        logger.info(
            "Pack '%s' generated: %d scenarios -> %s",
            self.pack_id,
            len(all_scenarios),
            out_dir,
        )
        return manifest

    # ----- scenario generators -----

    def _generate_nominal(
        self,
        scene_id: str,
        x_gt: np.ndarray,
        mask: np.ndarray,
        split: SplitKind,
        rng: np.random.Generator,
        split_dir: Path,
        file_hashes: Dict[str, str],
    ) -> List[ScenarioSpec]:
        """Generate one nominal (clean) scenario per scene."""
        noise_cfg = self.get_nominal_noise(split)
        mismatch_dict: Dict[str, float] = {}  # zero mismatch
        scenario_id = f"{split.value}_{scene_id}_nominal"
        seed_val = int(rng.integers(0, 2**31))

        sub_rng = np.random.default_rng(seed_val)
        y_meas, mask_corrupted = self.forward_model(
            x_gt, mask, mismatch_dict, noise_cfg, sub_rng
        )
        self._save_scenario_data(
            split_dir, scenario_id, scene_id, y_meas, x_gt, mask,
            mask_corrupted, split, mismatch_dict, file_hashes,
        )

        spec = ScenarioSpec(
            scenario_id=scenario_id,
            split=split,
            regime=RegimeKind.nominal,
            scene_id=scene_id,
            mismatch_params=[],
            noise_config=NoiseConfig(
                noise_alpha=noise_cfg.get("noise_alpha"),
                noise_sigma=noise_cfg["noise_sigma"],
            ),
            seed=seed_val,
        )
        return [spec]

    def _generate_single_param(
        self,
        scene_id: str,
        x_gt: np.ndarray,
        mask: np.ndarray,
        split: SplitKind,
        cfg: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
        split_dir: Path,
        file_hashes: Dict[str, str],
    ) -> List[ScenarioSpec]:
        """Generate single-param sweep scenarios (3 severities per param)."""
        noise_cfg = self.get_nominal_noise(split)
        fracs = self.get_severity_fractions()
        specs: List[ScenarioSpec] = []

        for param_name, param_cfg in cfg.items():
            pmin, pmax = param_cfg["min"], param_cfg["max"]
            unit = param_cfg["unit"]
            for i, frac in enumerate(fracs):
                val = pmin + frac * (pmax - pmin)
                mismatch_dict = {param_name: val}
                sev_label = f"sev{i}"
                scenario_id = f"{split.value}_{scene_id}_single_{param_name}_{sev_label}"
                seed_val = int(rng.integers(0, 2**31))

                sub_rng = np.random.default_rng(seed_val)
                y_meas, mask_corrupted = self.forward_model(
                    x_gt, mask, mismatch_dict, noise_cfg, sub_rng
                )
                self._save_scenario_data(
                    split_dir, scenario_id, scene_id, y_meas, x_gt, mask,
                    mask_corrupted, split, mismatch_dict, file_hashes,
                )

                mc = MismatchConfig(
                    name=param_name, value=val, unit=unit,
                    range_min=pmin, range_max=pmax,
                )
                spec = ScenarioSpec(
                    scenario_id=scenario_id,
                    split=split,
                    regime=RegimeKind.single_param,
                    red_team_category=RedTeamCategory.mismatch_escalation,
                    scene_id=scene_id,
                    mismatch_params=[mc],
                    noise_config=NoiseConfig(
                        noise_alpha=noise_cfg.get("noise_alpha"),
                        noise_sigma=noise_cfg["noise_sigma"],
                    ),
                    seed=seed_val,
                )
                specs.append(spec)
        return specs

    def _generate_compound(
        self,
        scene_id: str,
        x_gt: np.ndarray,
        mask: np.ndarray,
        split: SplitKind,
        cfg: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
        split_dir: Path,
        file_hashes: Dict[str, str],
    ) -> List[ScenarioSpec]:
        """Generate compound scenarios (3 random draws, all params)."""
        noise_cfg = self.get_nominal_noise(split)
        specs: List[ScenarioSpec] = []

        for draw_idx in range(3):
            mismatch_dict: Dict[str, float] = {}
            mc_list: List[MismatchConfig] = []
            for param_name, param_cfg in cfg.items():
                pmin, pmax = param_cfg["min"], param_cfg["max"]
                val = float(rng.uniform(pmin, pmax))
                mismatch_dict[param_name] = val
                mc_list.append(MismatchConfig(
                    name=param_name, value=val, unit=param_cfg["unit"],
                    range_min=pmin, range_max=pmax,
                ))

            scenario_id = f"{split.value}_{scene_id}_compound_{draw_idx}"
            seed_val = int(rng.integers(0, 2**31))

            sub_rng = np.random.default_rng(seed_val)
            y_meas, mask_corrupted = self.forward_model(
                x_gt, mask, mismatch_dict, noise_cfg, sub_rng
            )
            self._save_scenario_data(
                split_dir, scenario_id, scene_id, y_meas, x_gt, mask,
                mask_corrupted, split, mismatch_dict, file_hashes,
            )

            spec = ScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                regime=RegimeKind.compound,
                red_team_category=RedTeamCategory.compound_mismatch,
                scene_id=scene_id,
                mismatch_params=mc_list,
                noise_config=NoiseConfig(
                    noise_alpha=noise_cfg.get("noise_alpha"),
                    noise_sigma=noise_cfg["noise_sigma"],
                ),
                seed=seed_val,
            )
            specs.append(spec)
        return specs

    def _generate_red_team(
        self,
        scene_id: str,
        x_gt: np.ndarray,
        mask: np.ndarray,
        split: SplitKind,
        cfg: Dict[str, Dict[str, float]],
        rng: np.random.Generator,
        split_dir: Path,
        file_hashes: Dict[str, str],
    ) -> List[ScenarioSpec]:
        """Generate red-team scenarios: gate_flip, oof, compute_trap."""
        rt_cfgs = self.get_red_team_configs(split)
        specs: List[ScenarioSpec] = []

        regime_map = {
            "gate_flip": (RegimeKind.gate_flip, RedTeamCategory.gate_flip),
            "oof": (RegimeKind.oof, RedTeamCategory.out_of_family),
            "compute_trap": (RegimeKind.compute_trap, RedTeamCategory.compute_trap),
        }

        for rt_name, (regime, category) in regime_map.items():
            if rt_name not in rt_cfgs:
                continue
            rt_cfg = rt_cfgs[rt_name]
            mismatch_dict = rt_cfg.get("mismatch", {})
            noise_dict = rt_cfg.get("noise", self.get_nominal_noise(split))
            meta = rt_cfg.get("metadata", {})

            scenario_id = f"{split.value}_{scene_id}_{rt_name}"
            seed_val = int(rng.integers(0, 2**31))

            sub_rng = np.random.default_rng(seed_val)

            # For oof/compute_trap, apply_injection may modify y after forward
            y_meas, mask_corrupted = self.forward_model(
                x_gt, mask, mismatch_dict, noise_dict, sub_rng
            )

            # Apply post-forward injection if provided
            injection_fn = rt_cfg.get("post_injection")
            if injection_fn is not None:
                y_meas = injection_fn(y_meas, x_gt, mask, sub_rng)

            mc_list = [
                MismatchConfig(
                    name=k, value=v, unit=cfg.get(k, {}).get("unit", ""),
                    range_min=cfg.get(k, {}).get("min", v),
                    range_max=cfg.get(k, {}).get("max", v),
                )
                for k, v in mismatch_dict.items()
            ]

            self._save_scenario_data(
                split_dir, scenario_id, scene_id, y_meas, x_gt, mask,
                mask_corrupted, split, mismatch_dict, file_hashes,
            )

            spec = ScenarioSpec(
                scenario_id=scenario_id,
                split=split,
                regime=regime,
                red_team_category=category,
                scene_id=scene_id,
                mismatch_params=mc_list,
                noise_config=NoiseConfig(
                    noise_alpha=noise_dict.get("noise_alpha"),
                    noise_sigma=noise_dict.get("noise_sigma", 0.01),
                ),
                seed=seed_val,
                metadata=meta,
            )
            specs.append(spec)
        return specs

    # ----- I/O helpers -----

    def _save_scenario_data(
        self,
        split_dir: Path,
        scenario_id: str,
        scene_id: str,
        y_meas: np.ndarray,
        x_gt: np.ndarray,
        mask_ideal: np.ndarray,
        mask_corrupted: np.ndarray,
        split: SplitKind,
        mismatch_dict: Dict[str, float],
        file_hashes: Dict[str, str],
    ) -> None:
        """Save scenario .npz and params.json to disk."""
        scene_dir = split_dir / scenario_id
        scene_dir.mkdir(parents=True, exist_ok=True)

        # Measurement .npz
        npz_path = scene_dir / "measurement.npz"
        save_kwargs = {
            "y": y_meas,
            "x_gt": x_gt,
            "mask_ideal": mask_ideal,
        }
        if split == SplitKind.probe:
            save_kwargs["mask_corrupted"] = mask_corrupted
        np.savez_compressed(str(npz_path), **save_kwargs)
        rel_npz = str(npz_path.relative_to(split_dir.parent))
        file_hashes[rel_npz] = sha256_file(npz_path)

        # Params JSON
        params_path = scene_dir / "params.json"
        if split == SplitKind.probe:
            params_data = {
                "scenario_id": scenario_id,
                "scene_id": scene_id,
                "mismatch_params": mismatch_dict,
            }
        else:
            # Hidden: redact mismatch values
            params_data = {
                "scenario_id": scenario_id,
                "scene_id": scene_id,
                "mismatch_params": "REDACTED",
            }
        params_path.write_text(json.dumps(params_data, indent=2), encoding="utf-8")
        rel_params = str(params_path.relative_to(split_dir.parent))
        file_hashes[rel_params] = sha256_file(params_path)

    def _write_scenarios_json(
        self,
        split_dir: Path,
        scenarios: List[ScenarioSpec],
        file_hashes: Dict[str, str],
    ) -> None:
        """Write scenarios.json for a split."""
        path = split_dir / "scenarios.json"
        data = [s.model_dump(mode="json") for s in scenarios]
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        rel = str(path.relative_to(split_dir.parent))
        file_hashes[rel] = sha256_file(path)
