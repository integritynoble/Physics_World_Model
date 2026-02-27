"""Factory — expands compact registry entries into full VARIANT_DATABASE entries."""

from __future__ import annotations

from ._algorithm_catalog import CATEGORY_BENCHMARK_DATASETS
from ._challenge_data import CHALLENGE_CONFIG


# ── Challenge benchmark builder ───────────────────────────────────────────────

def _make_b_challenge(key: str, display: str, leaderboard: list | None = None,
                      category: str | None = None) -> dict | None:
    """Build a Blind Reconstruction Challenge benchmark if the variant is configured."""
    cfg = CHALLENGE_CONFIG.get(key)
    if cfg is None:
        return None

    scene_count = cfg["scene_count"]
    tiers_cfg = cfg["tiers"]

    bm_dataset = CATEGORY_BENCHMARK_DATASETS.get(category) if category else None

    return {
        "id": "Challenge",
        "title": "Blind Reconstruction Challenge",
        "description": (
            "Given measurements with unknown mismatch and spec ranges (not exact params), "
            "reconstruct the original signal. A method must be evaluated on all three tiers "
            "for a complete score. Scored on a composite metric: "
            f"{cfg['scoring']['formula_display']}."
        ),
        "input": "Measurements y, ideal forward model H, spec ranges",
        "output": "Reconstructed signal x̂",
        "is_challenge": True,
        "scoring": cfg["scoring"],
        "spec_ranges": cfg["spec_ranges"],
        "gallery_variant": cfg.get("gallery_variant", key),
        "benchmark_dataset": bm_dataset,
        "leaderboard": leaderboard or [],
        "tiers": {
            "public": {
                "name": "Public",
                "description": (
                    f"All {scene_count} scenes with measurements, ideal operator, "
                    "spec ranges, ground truth, and true spec. Includes all InverseNet datasets."
                ),
                "count": scene_count,
                "includes_ground_truth": True,
                "introduction": tiers_cfg["public"].get("introduction"),
                "preview_image": tiers_cfg["public"].get("preview_image"),
                "visible_data": tiers_cfg["public"].get("visible_data", []),
                "true_spec": tiers_cfg["public"].get("true_spec"),
                "spec_ranges": tiers_cfg["public"].get("spec_ranges", []),
                "dataset": {
                    "name": f"{display} Challenge Public Dataset",
                    "format": "HDF5",
                    "num_samples": scene_count,
                    "gcs_object_path": f"challenge-data/v1.0/{key}_challenge_public.h5",
                    "download_url": None,
                },
            },
            "dev": {
                "name": "Dev",
                "description": (
                    f"All {scene_count} scenes with measurements + ideal operator + "
                    "spec ranges (no ground truth). Submit your reconstruction."
                ),
                "count": scene_count,
                "introduction": tiers_cfg["dev"].get("introduction"),
                "preview_image": tiers_cfg["dev"].get("preview_image"),
                "visible_data": tiers_cfg["dev"].get("visible_data", []),
                "true_spec": tiers_cfg["dev"].get("true_spec"),
                "spec_ranges": tiers_cfg["dev"].get("spec_ranges", []),
                "dataset": {
                    "name": f"{display} Challenge Dev Dataset",
                    "format": "HDF5",
                    "num_samples": scene_count,
                    "gcs_object_path": f"challenge-data/v1.0/{key}_challenge_dev.h5",
                    "download_url": None,
                },
            },
            "hidden": {
                "name": "Hidden",
                "description": (
                    f"All {scene_count} scenes held server-side. "
                    "Submit your algorithm; we run it on hidden data."
                ),
                "count": scene_count,
                "introduction": tiers_cfg["hidden"].get("introduction"),
                "preview_image": tiers_cfg["hidden"].get("preview_image"),
                "visible_data": tiers_cfg["hidden"].get("visible_data", []),
                "true_spec": tiers_cfg["hidden"].get("true_spec"),
                "spec_ranges": tiers_cfg["hidden"].get("spec_ranges", []),
            },
        },
        "baselines": cfg["baselines"],
        "links": {
            "submit_reconstruction": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-reconstruction.md",
            "submit_algorithm": "https://github.com/InverseNet/benchmark-data/issues/new?template=submit-algorithm.md",
        },
        "credits": {"winner_share_pct": 30, "pool_source": "platform_profit"},
    }


# ── Benchmark list builder ────────────────────────────────────────────────────

def _build_benchmarks(key: str, display: str, lb: dict,
                      category: str | None = None) -> list[dict]:
    """Build benchmark list — Challenge only."""
    benchmarks = []
    challenge = _make_b_challenge(key, display, leaderboard=lb.get("challenge"),
                                  category=category)
    if challenge is not None:
        benchmarks.append(challenge)
    return benchmarks


# ── Public factory ────────────────────────────────────────────────────────────

def build_variant(key: str, registry_entry: dict, leaderboard: dict | None = None) -> dict:
    """Expand a compact registry entry into a full VARIANT_DATABASE entry."""
    display = registry_entry["display_name"]
    category = registry_entry["category"]
    lb = leaderboard or {}

    return {
        "display_name": display,
        "full_name": registry_entry["full_name"],
        "parent_modality": registry_entry["parent_modality"],
        "category": category,
        "spec_notation": registry_entry["spec_notation"],
        "spec_dag": registry_entry["spec_dag"],
        "mismatch_params": registry_entry["mismatch_params"],
        "benchmarks": _build_benchmarks(key, display, lb, category=category),
        "credits_config": {
            "profit_pool_pct": 40,
            "winner_share_pct": 30,
            "min_withdrawal_usd": 100,
        },
    }
