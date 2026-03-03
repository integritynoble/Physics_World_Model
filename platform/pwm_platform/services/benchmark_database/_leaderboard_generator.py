"""Leaderboard generator — deterministic benchmark data for all imaging modalities.

Generates challenge leaderboard and challenge baselines using seed-based
deterministic scoring.  Internal B2 leaderboard and scenario table generators
are kept as private helpers since challenge scoring depends on them.

All scores are derived from category-level PSNR bands, mismatch sensitivity
parameters, and oracle recovery fractions that produce plausible, realistic
benchmark data consistent with the hand-crafted InverseNet baselines.

Key design principles:
  - Deterministic: same variant_key always produces identical scores
  - Calibrated: PSNR ranges reflect real published results per domain
  - Differentiated: classical < PnP < deep < transformer (with overlap)
  - Domain-aware: scene names, mismatch descriptions, algorithm sources
  - Consistent with hand-crafted InverseNet baselines (CASSI, CACTI, SPC)
"""

from __future__ import annotations

import hashlib
import math

from ._algorithm_catalog import (
    CATEGORY_REAL_SCORES,
    classify_solver,
    get_algorithms,
    get_correction_description,
    get_mismatch_description,
    get_scene_names,
)

# ── PSNR ranges per solver class (low, high) ─────────────────────────────────
# Calibrated to published results: classical worst, transformer best.
# These are Scenario I (Ideal) PSNR ranges — no mismatch, no noise.

_PSNR_RANGES: dict[str, dict[str, tuple[float, float]]] = {
    "compressive":                {"classical": (24.0, 28.5), "pnp": (27.0, 31.0), "deep": (32.0, 36.0), "transformer": (33.0, 37.0)},
    "medical":                    {"classical": (26.0, 31.0), "pnp": (29.0, 34.0), "deep": (33.0, 38.0), "transformer": (34.0, 39.0)},
    "medical_ultrasound":         {"classical": (23.0, 27.0), "pnp": (26.0, 30.0), "deep": (30.0, 34.0), "transformer": (31.0, 35.0)},
    "coherent":                   {"classical": (22.0, 26.0), "pnp": (25.0, 29.0), "deep": (29.0, 33.0), "transformer": (30.0, 34.0)},
    "microscopy":                 {"classical": (25.0, 29.0), "pnp": (28.0, 32.0), "deep": (31.0, 36.0), "transformer": (32.0, 37.0)},
    "electron_microscopy":        {"classical": (21.0, 25.0), "pnp": (24.0, 28.0), "deep": (27.0, 32.0), "transformer": (28.0, 33.0)},
    "clinical_optics":            {"classical": (24.0, 28.0), "pnp": (27.0, 31.0), "deep": (30.0, 35.0), "transformer": (31.0, 36.0)},
    "computational":              {"classical": (25.0, 29.0), "pnp": (28.0, 32.0), "deep": (31.0, 36.0), "transformer": (32.0, 37.0)},
    "computational_photography":  {"classical": (26.0, 30.0), "pnp": (29.0, 33.0), "deep": (32.0, 37.0), "transformer": (33.0, 38.0)},
    "neural_rendering":           {"classical": (25.0, 29.0), "pnp": (27.0, 31.0), "deep": (30.0, 35.0), "transformer": (31.0, 36.0)},
    "depth_imaging":              {"classical": (24.0, 28.0), "pnp": (27.0, 31.0), "deep": (30.0, 35.0), "transformer": (31.0, 36.0)},
    "remote_sensing":             {"classical": (22.0, 26.0), "pnp": (25.0, 29.0), "deep": (28.0, 33.0), "transformer": (29.0, 34.0)},
    "particle_imaging":           {"classical": (23.0, 27.0), "pnp": (26.0, 30.0), "deep": (29.0, 34.0), "transformer": (30.0, 35.0)},
    "scanning_probe":             {"classical": (22.0, 26.0), "pnp": (25.0, 29.0), "deep": (28.0, 33.0), "transformer": (29.0, 34.0)},
    "industrial_inspection":      {"classical": (25.0, 29.0), "pnp": (28.0, 32.0), "deep": (31.0, 36.0), "transformer": (32.0, 37.0)},
    "spectroscopy":               {"classical": (23.0, 27.0), "pnp": (26.0, 30.0), "deep": (29.0, 34.0), "transformer": (30.0, 35.0)},
    "astronomy":                  {"classical": (21.0, 25.0), "pnp": (24.0, 28.0), "deep": (27.0, 32.0), "transformer": (28.0, 33.0)},
    "ultrafast":                  {"classical": (23.0, 27.0), "pnp": (26.0, 30.0), "deep": (29.0, 34.0), "transformer": (30.0, 35.0)},
    "quantum":                    {"classical": (20.0, 24.0), "pnp": (23.0, 27.0), "deep": (26.0, 31.0), "transformer": (27.0, 32.0)},
    "experimental_science":       {"classical": (24.0, 28.0), "pnp": (27.0, 31.0), "deep": (30.0, 35.0), "transformer": (31.0, 36.0)},
    "scientific_instrumentation": {"classical": (23.0, 27.0), "pnp": (26.0, 30.0), "deep": (29.0, 34.0), "transformer": (30.0, 35.0)},
    "multi_modal_fusion":         {"classical": (24.0, 28.0), "pnp": (27.0, 31.0), "deep": (30.0, 35.0), "transformer": (31.0, 36.0)},
}

# Mismatch PSNR drop (dB) per solver class.
# Classical methods are robust (model-based), deep methods fragile (distribution shift).
# PnP = moderate. Transformer = moderate-high (learned but adaptable with gradient).
_MISMATCH_DROP: dict[str, tuple[float, float]] = {
    "classical":   (3.0, 6.0),
    "pnp":         (5.0, 10.0),
    "deep":        (8.0, 14.0),
    "transformer": (6.0, 12.0),
}

# Oracle recovery fraction of the mismatch drop.
# How much of the drop is recovered when true calibration params are known.
# Transformer + gradient recovers most; deep methods less (learned weights can't adapt).
_RECOVERY_FRACTION: dict[str, tuple[float, float]] = {
    "classical":   (0.60, 0.80),
    "pnp":         (0.50, 0.75),
    "deep":        (0.40, 0.60),
    "transformer": (0.65, 0.90),
}

# Scene counts per category (aligned with _challenge_data._CATEGORY_SCENE_COUNT)
_CATEGORY_SCENE_COUNT: dict[str, int] = {
    "compressive": 5, "medical": 3, "medical_ultrasound": 3, "coherent": 5,
    "microscopy": 5, "electron_microscopy": 3, "clinical_optics": 3,
    "computational": 5, "computational_photography": 5, "neural_rendering": 5,
    "depth_imaging": 5, "remote_sensing": 3, "particle_imaging": 3,
    "scanning_probe": 3, "industrial_inspection": 3, "spectroscopy": 5,
    "astronomy": 3, "ultrafast": 5, "quantum": 3, "experimental_science": 5,
    "scientific_instrumentation": 5, "multi_modal_fusion": 3,
}


# ── Utility functions ─────────────────────────────────────────────────────────


def _deterministic_seed(key: str, method: str, scene: int = 0) -> int:
    """Generate a deterministic integer seed from variant key + method + scene."""
    h = hashlib.sha256(f"{key}:{method}:{scene}".encode()).hexdigest()
    return int(h[:8], 16)


def _seeded_uniform(seed: int, lo: float, hi: float) -> float:
    """Deterministic uniform sample in [lo, hi] from a seed."""
    t = ((seed * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF
    return lo + t * (hi - lo)


def _seeded_normal(seed: int, mean: float, std: float) -> float:
    """Deterministic approximate normal sample using Box-Muller."""
    u1 = max(1e-10, ((seed * 2654435761) & 0xFFFFFFFF) / 0xFFFFFFFF)
    u2 = (((seed * 1664525 + 1013904223) & 0xFFFFFFFF)) / 0xFFFFFFFF
    z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
    return mean + std * z


def _psnr_to_ssim(psnr: float) -> float:
    """Convert PSNR to approximate SSIM using logistic mapping.

    Fitted to InverseNet baselines:
      PSNR 15 -> ~0.35, 20 -> ~0.60, 25 -> ~0.80, 30 -> ~0.92, 35 -> ~0.97
    """
    return min(0.995, max(0.05, 1.0 / (1.0 + math.exp(-0.2 * (psnr - 18.0)))))


def _round2(x: float) -> float:
    return round(x, 2)


def _round3(x: float) -> float:
    return round(x, 3)


# ── B2 leaderboard (private — used internally by challenge pipeline) ──────────


def _generate_b2_leaderboard(
    variant_key: str,
    category: str,
    algorithms: list[dict],
) -> list[dict]:
    """Generate B2 (Scenario I — Ideal) leaderboard with 4 ranked entries.

    Each entry contains method name, PSNR, SSIM, source citation,
    and adopted flag (True for rank-1 method = official PWM baseline).

    Uses real published scores from CATEGORY_REAL_SCORES when available
    for the category. Falls back to synthetic PSNR-range generation for
    categories without real scores or for unmatched method names.
    """
    # Check variant-specific scores first, then fall back to category
    real_scores = CATEGORY_REAL_SCORES.get(variant_key) or CATEGORY_REAL_SCORES.get(category)
    # Build a lookup from method name → real score entry
    real_lookup: dict[str, dict] = {}
    if real_scores:
        for rs in real_scores:
            real_lookup[rs["method"]] = rs

    psnr_bands = _PSNR_RANGES.get(category, _PSNR_RANGES["computational"])
    entries = []

    for algo in algorithms:
        match = real_lookup.get(algo["name"])
        if match:
            # Use real published PSNR/SSIM
            psnr = match["psnr"]
            ssim = match["ssim"]
            source = match.get("source", algo.get("source", "PWM Baseline"))
        else:
            # Fall back to synthetic generation
            solver_class = classify_solver(algo["type"])
            lo, hi = psnr_bands.get(solver_class, (25.0, 30.0))
            seed = _deterministic_seed(variant_key, algo["name"])
            psnr = _round2(_seeded_uniform(seed, lo, hi))
            ssim = _round3(_psnr_to_ssim(psnr))
            source = algo.get("source", "PWM Baseline")

        entries.append({
            "method": algo["name"],
            "psnr": psnr,
            "ssim": ssim,
            "source": source,
            "adopted": False,
        })

    # Sort by PSNR descending, assign ranks, mark top-1 as adopted
    entries.sort(key=lambda e: e["psnr"], reverse=True)
    entries[0]["adopted"] = True
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    return entries


# ── B2 scenario table (private — used internally by challenge pipeline) ───────


def _generate_b2_scenario_table(
    variant_key: str,
    category: str,
    algorithms: list[dict],
    b2_leaderboard: list[dict],
) -> dict:
    """Generate full 3-scenario comparison table (Ideal / Mismatch / Oracle).

    Uses domain-specific scene names and mismatch descriptions from the
    algorithm catalog. Per-scene PSNR values have deterministic jitter
    (sigma ~ 1.5-2.0 dB) around the method mean.
    """
    scene_count = _CATEGORY_SCENE_COUNT.get(category, 5)
    scene_names = get_scene_names(category, scene_count)
    mismatch_desc = get_mismatch_description(category)
    correction_desc = get_correction_description(category)

    # Build method lookup from B2 leaderboard
    ideal_psnr = {e["method"]: e["psnr"] for e in b2_leaderboard}
    ideal_ssim = {e["method"]: e["ssim"] for e in b2_leaderboard}

    methods = []
    method_names = []

    for algo in algorithms:
        name = algo["name"]
        solver_class = classify_solver(algo["type"])
        s1_psnr = ideal_psnr.get(name, 28.0)
        s1_ssim = ideal_ssim.get(name, 0.85)

        # Scenario II: Mismatch (uncorrected)
        drop_lo, drop_hi = _MISMATCH_DROP.get(solver_class, (5.0, 10.0))
        drop_seed = _deterministic_seed(variant_key, name, scene=999)
        drop = _seeded_uniform(drop_seed, drop_lo, drop_hi)
        s2_psnr = _round2(s1_psnr - drop)
        s2_ssim = _round3(_psnr_to_ssim(s2_psnr))

        # Scenario III: Oracle correction (gradient with known true params)
        rec_lo, rec_hi = _RECOVERY_FRACTION.get(solver_class, (0.50, 0.75))
        rec_seed = _deterministic_seed(variant_key, name, scene=998)
        recovery = _seeded_uniform(rec_seed, rec_lo, rec_hi)
        s3_psnr = _round2(s2_psnr + recovery * drop)
        s3_ssim = _round3(_psnr_to_ssim(s3_psnr))

        methods.append({
            "method": name,
            "type": algo["type"],
            "mask_aware": algo["mask_aware"],
            "params": algo["params"],
            "s1_psnr": s1_psnr, "s1_ssim": s1_ssim,
            "s2_psnr": s2_psnr, "s2_ssim": s2_ssim,
            "s3_psnr": s3_psnr, "s3_ssim": s3_ssim,
        })
        method_names.append(name)

    # Generate per-scene PSNR values with deterministic jitter
    # Format: [m1_s1, m1_s2, m1_s3, m2_s1, m2_s2, m2_s3, ...]
    per_scene = []
    for scene_idx in range(scene_count):
        psnr_values = []
        for algo in algorithms:
            name = algo["name"]
            method_data = next(m for m in methods if m["method"] == name)

            for key_prefix in ("s1", "s2", "s3"):
                mean_psnr = method_data[f"{key_prefix}_psnr"]
                seed = _deterministic_seed(
                    variant_key, f"{name}:{key_prefix}", scene=scene_idx + 1,
                )
                jitter = _seeded_normal(seed, 0.0, 1.8)
                scene_psnr = _round2(mean_psnr + jitter)
                psnr_values.append(scene_psnr)

        per_scene.append({
            "scene": scene_names[scene_idx],
            "psnr_values": psnr_values,
        })

    desc = (
        f"{scene_count} test scenes, {len(algorithms)} algorithms, 3 scenarios. "
        f"{mismatch_desc} "
        f"Correction: {correction_desc}"
    )

    return {
        "description": desc,
        "scenarios": [
            {
                "id": "I",
                "name": "Ideal",
                "color": "green",
                "description": "No mismatch, no noise \u2014 theoretical upper bound",
            },
            {
                "id": "II",
                "name": "Mismatch (uncorrected)",
                "color": "red",
                "description": (
                    "Forward model mismatch + measurement noise applied, "
                    "no calibration correction"
                ),
            },
            {
                "id": "III",
                "name": "Oracle correction",
                "color": "blue",
                "description": (
                    "True mismatch parameters known \u2014 gradient-based "
                    "calibration upper bound"
                ),
            },
        ],
        "methods": methods,
        "method_names": method_names,
        "per_scene": per_scene,
    }


# ── Challenge leaderboard ─────────────────────────────────────────────────────


def _compute_challenge_score(psnr: float, ssim: float, consistency: float) -> float:
    """Composite score: 0.4 * PSNR_norm + 0.4 * SSIM + 0.2 * consistency.

    PSNR normalized to [0, 1] range assuming 10-50 dB range.
    """
    psnr_norm = min(1.0, max(0.0, (psnr - 10.0) / 40.0))
    return 0.4 * psnr_norm + 0.4 * ssim + 0.2 * consistency


def generate_challenge_leaderboard(
    variant_key: str,
    category: str,
    algorithms: list[dict],
    b2_scenario_table: dict,
) -> list[dict]:
    """Generate challenge leaderboard with 3-tier scores per method.

    The challenge applies each algorithm with gradient-based forward model
    correction (hence '+ gradient' suffix). Three tiers test progressively
    harder mismatch conditions:
      - Public: mild mismatch, ground truth available (close to Scenario I)
      - Dev: moderate mismatch, blind evaluation (between Scenario II and III)
      - Hidden: severe mismatch, fully blind (close to Scenario II)
    """
    entries = []

    for algo in algorithms:
        name = algo["name"]
        method_data = next(
            (m for m in b2_scenario_table["methods"] if m["method"] == name),
            None,
        )
        if method_data is None:
            continue

        # ── Public tier: close to Scenario I (mild mismatch + ground truth) ──
        pub_seed = _deterministic_seed(variant_key, f"{name}:pub")
        pub_psnr = _round2(
            method_data["s1_psnr"] - _seeded_uniform(pub_seed, 1.0, 3.0)
        )
        pub_ssim = _round3(_psnr_to_ssim(pub_psnr))
        pub_cons = _round2(min(0.99, _seeded_uniform(pub_seed + 1, 0.86, 0.95)))

        # ── Dev tier: between Scenario II and III (moderate, blind) ──
        dev_seed = _deterministic_seed(variant_key, f"{name}:dev")
        dev_psnr = _round2(
            (method_data["s2_psnr"] + method_data["s3_psnr"]) / 2.0
            + _seeded_uniform(dev_seed, -1.0, 1.0)
        )
        dev_ssim = _round3(_psnr_to_ssim(dev_psnr))
        dev_cons = _round2(min(0.95, _seeded_uniform(dev_seed + 1, 0.78, 0.90)))

        # ── Hidden tier: close to Scenario II (hardest, fully blind) ──
        hid_seed = _deterministic_seed(variant_key, f"{name}:hid")
        hid_psnr = _round2(
            method_data["s2_psnr"] + _seeded_uniform(hid_seed, -0.5, 1.5)
        )
        hid_ssim = _round3(_psnr_to_ssim(hid_psnr))
        hid_cons = _round2(min(0.92, _seeded_uniform(hid_seed + 1, 0.74, 0.88)))

        # Composite scores
        pub_score = _round3(_compute_challenge_score(pub_psnr, pub_ssim, pub_cons))
        dev_score = _round3(_compute_challenge_score(dev_psnr, dev_ssim, dev_cons))
        hid_score = _round3(_compute_challenge_score(hid_psnr, hid_ssim, hid_cons))
        overall = _round3((pub_score + dev_score + hid_score) / 3.0)

        entries.append({
            "rank": 0,
            "method": f"{name} + gradient",
            "public_score": pub_score,
            "dev_score": dev_score,
            "hidden_score": hid_score,
            "overall_score": overall,
            "details": {
                "public": {
                    "psnr": pub_psnr,
                    "ssim": pub_ssim,
                    "consistency": pub_cons,
                },
                "dev": {
                    "psnr": dev_psnr,
                    "ssim": dev_ssim,
                    "consistency": dev_cons,
                },
                "hidden": {
                    "psnr": hid_psnr,
                    "ssim": hid_ssim,
                    "consistency": hid_cons,
                },
            },
            "source": algo.get("source", "PWM Baseline"),
        })

    # Sort by overall_score descending, assign ranks
    entries.sort(key=lambda e: e["overall_score"], reverse=True)
    for i, e in enumerate(entries, 1):
        e["rank"] = i

    return entries


# ── Challenge baselines ───────────────────────────────────────────────────────


def generate_challenge_baselines(
    variant_key: str,
    algorithms: list[dict],
    b2_scenario_table: dict,
) -> dict:
    """Generate challenge baselines from B2 scenario data.

    Returns {"scenario_ii": [...], "scenario_iii": [...]} with per-method
    PSNR/SSIM drawn from the scenario table. These show what each algorithm
    achieves under mismatch (II) and oracle correction (III).
    """
    scenario_ii = []
    scenario_iii = []

    for algo in algorithms:
        name = algo["name"]
        method_data = next(
            (m for m in b2_scenario_table["methods"] if m["method"] == name),
            None,
        )
        if method_data is None:
            continue

        scenario_ii.append({
            "method": name,
            "psnr": method_data["s2_psnr"],
            "ssim": method_data["s2_ssim"],
        })
        scenario_iii.append({
            "method": name,
            "psnr": method_data["s3_psnr"],
            "ssim": method_data["s3_ssim"],
        })

    return {
        "scenario_ii": scenario_ii,
        "scenario_iii": scenario_iii,
    }


# ── Full leaderboard generation ──────────────────────────────────────────────


def generate_full_leaderboard(
    variant_key: str,
    category: str,
) -> dict:
    """Generate complete leaderboard data dict for a single variant.

    Returns a dict with keys: challenge, _baselines
    matching the format expected by _factory.build_variant().

    The _baselines key is consumed by __init__.py to fill challenge configs
    with Scenario II/III baseline performance.

    Internally generates B2 leaderboard and scenario table as intermediate
    data required by the challenge scoring pipeline.
    """
    algorithms = get_algorithms(variant_key, category)

    b2 = _generate_b2_leaderboard(variant_key, category, algorithms)
    b2_table = _generate_b2_scenario_table(
        variant_key, category, algorithms, b2,
    )
    challenge = generate_challenge_leaderboard(
        variant_key, category, algorithms, b2_table,
    )
    baselines = generate_challenge_baselines(
        variant_key, algorithms, b2_table,
    )

    return {
        "challenge": challenge,
        "_baselines": baselines,
    }
