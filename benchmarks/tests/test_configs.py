"""Validate all 168 benchmark configs load correctly."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "packages" / "pwm_core"))


class TestAllConfigsLoad:
    """Verify every YAML config parses and validates."""

    def test_168_configs_exist(self):
        configs_dir = ROOT / "benchmarks" / "configs"
        yamls = [p for p in configs_dir.glob("*.yaml") if not p.name.startswith("_")]
        assert len(yamls) == 168, f"Expected 168 configs, found {len(yamls)}"

    def test_all_configs_load(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        assert len(configs) == 168

    def test_all_configs_have_modality_id(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            assert cfg.modality_id == mid, f"Config {mid} has mismatched modality_id: {cfg.modality_id}"

    def test_all_configs_have_category(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            assert cfg.category, f"{mid} missing category"

    def test_all_configs_have_dag(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            assert cfg.canonical_dag, f"{mid} missing canonical_dag"
            assert "-->" in cfg.canonical_dag, f"{mid} DAG format invalid: {cfg.canonical_dag}"

    def test_all_configs_have_valid_tier(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            assert cfg.tier in ("A", "B", "C"), f"{mid} invalid tier: {cfg.tier}"

    def test_all_configs_have_valid_dims(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            assert len(cfg.x_shape) >= 2, f"{mid} x_shape too short: {cfg.x_shape}"
            assert all(d > 0 for d in cfg.x_shape), f"{mid} x_shape has non-positive dim"

    def test_all_configs_have_category_module(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        valid_modules = {
            "microscopy_psf", "compressive_mask", "medical_ct_radon",
            "medical_mri_kspace", "electron_ctf", "scanning_probe",
            "remote_sensing_sar", "nuclear_emission",
        }
        for mid, cfg in configs.items():
            assert cfg.category_module in valid_modules, \
                f"{mid} has invalid category_module: {cfg.category_module}"

    def test_all_configs_have_source_attribution(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            sa = cfg.source_attribution
            assert sa.ground_truth, f"{mid} missing ground_truth attribution"
            assert sa.forward_model, f"{mid} missing forward_model attribution"

    def test_tier_a_configs_have_solvers(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        tier_a = {k: v for k, v in configs.items() if v.tier == "A"}
        for mid, cfg in tier_a.items():
            assert cfg.solvers, f"Tier A modality {mid} has no solvers"

    def test_mismatch_params_valid(self):
        from benchmarks.framework.benchmark_config import load_all_configs
        configs = load_all_configs()
        for mid, cfg in configs.items():
            for mp in cfg.mismatch_params:
                assert mp.name, f"{mid} has unnamed mismatch param"
                assert mp.range[0] <= mp.range[1], \
                    f"{mid} mismatch {mp.name} range inverted"


class TestConfigConsistency:
    """Cross-validate configs against registries."""

    def test_modalities_yaml_has_all(self):
        """All 168 modality IDs exist in modalities.yaml."""
        import yaml
        with open(ROOT / "packages" / "pwm_core" / "contrib" / "modalities.yaml") as f:
            data = yaml.safe_load(f)
        mod_ids = set(data.get("modalities", {}).keys())

        from benchmarks.framework.benchmark_config import load_all_configs
        config_ids = set(load_all_configs().keys())

        missing = config_ids - mod_ids
        assert not missing, f"Modalities missing from modalities.yaml: {missing}"

    def test_solver_registry_has_all(self):
        """All 168 modality IDs have solver entries."""
        import yaml
        with open(ROOT / "packages" / "pwm_core" / "contrib" / "solver_registry.yaml") as f:
            data = yaml.safe_load(f)
        solver_ids = set(data.keys()) - {"version"}

        from benchmarks.framework.benchmark_config import load_all_configs
        config_ids = set(load_all_configs().keys())

        missing = config_ids - solver_ids
        assert not missing, f"Modalities missing from solver_registry.yaml: {missing}"

    def test_graph_templates_coverage(self):
        """All 168 modalities have at least one graph template."""
        import yaml
        with open(ROOT / "packages" / "pwm_core" / "contrib" / "graph_templates.yaml") as f:
            data = yaml.safe_load(f)
        template_mods = set()
        for tid in data.get("templates", {}):
            parts = tid.rsplit("_graph_", 1)
            if len(parts) == 2:
                template_mods.add(parts[0])

        from benchmarks.framework.benchmark_config import load_all_configs
        config_ids = set(load_all_configs().keys())

        missing = config_ids - template_mods
        assert not missing, f"Modalities missing from graph_templates.yaml: {missing}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
