#!/usr/bin/env python3
"""
Generate 3,600 benchmark tasks for the universal simulation paper.
12 domains x 100 instances per domain x 3 tiers = 3,600 tasks.

Each task is a YAML file specifying parameter variations from the base spec.md.
Tiers:
  - public:  ground truth provided, standard parameters
  - dev:     blind evaluation, modified parameters
  - hidden:  adversarial instances, edge cases

Seeds ensure reproducibility across independent runs.
"""

import os
import json
import hashlib
import numpy as np
from pathlib import Path

BENCHMARK_ROOT = Path(__file__).parent
SEED_BASE = 20260317  # date-based seed for reproducibility

DOMAINS = {
    "01_classical_mechanics": {
        "name": "Granular Flow with Frictional Contact",
        "base_params": {
            "N": 5000, "mu_friction": 0.5, "sigma_top_kPa": 10.0,
            "R_mean_mm": 2.0, "R_std_mm": 0.5, "gravity_angle_deg": 0.0,
        },
        "variations": {
            "public": {
                "N": [3000, 4000, 5000, 6000, 7000],
                "mu_friction": [0.3, 0.4, 0.5, 0.6, 0.7],
                "sigma_top_kPa": [1, 5, 10, 20, 30],
            },
            "dev": {
                "N": [3500, 4500, 5500, 6500, 8000],
                "mu_friction": [0.2, 0.35, 0.55, 0.75, 0.8],
                "sigma_top_kPa": [2, 8, 15, 25, 40],
                "gravity_angle_deg": [0, 5, 10, 15, 20],
            },
            "hidden": {
                "N": [2000, 3000, 5000, 8000, 10000],
                "mu_friction": [0.1, 0.2, 0.5, 0.9, 1.0],
                "sigma_top_kPa": [0.5, 50, 100, 5, 10],
                "R_std_mm": [0.0, 0.1, 0.5, 1.0, 1.5],
                "gravity_angle_deg": [0, 10, 20, 30, 45],
            },
        },
    },
    "02_electromagnetics": {
        "name": "Rectangular Waveguide Mode Analysis",
        "base_params": {
            "a_mm": 22.86, "b_mm": 10.16, "epsilon_r": 1.0,
            "sigma_wall": 1e30, "n_modes": 10, "freq_min_GHz": 8.0, "freq_max_GHz": 12.0,
        },
        "variations": {
            "public": {
                "a_mm": [22.86, 34.85, 47.55, 72.14, 109.22],
                "epsilon_r": [1.0, 1.0, 1.0, 1.0, 1.0],
            },
            "dev": {
                "a_mm": [15.80, 28.50, 40.39, 58.17, 86.36],
                "epsilon_r": [1.0, 2.1, 4.0, 9.8, 2.1],
            },
            "hidden": {
                "a_mm": [22.86, 22.86, 22.86, 34.85, 47.55],
                "epsilon_r": [1.0, 2.1, 4.0, 1.0, 9.8],
                "sigma_wall": [5.8e7, 3.77e7, 1e6, 5.8e7, 1e5],
            },
        },
    },
    "03_quantum_chemistry": {
        "name": "Helium-like Atom Ground State Energy",
        "base_params": {
            "Z": 2, "basis": "STO-6G", "n_max": 4, "l_max": 3, "method": "Full-CI",
        },
        "variations": {
            "public": {
                "Z": [2, 2, 2, 2, 2],
                "basis": ["STO-3G", "STO-6G", "cc-pVDZ", "cc-pVTZ", "cc-pVQZ"],
            },
            "dev": {
                "Z": [3, 3, 4, 4, 2],
                "basis": ["STO-6G", "cc-pVDZ", "STO-6G", "cc-pVDZ", "aug-cc-pVDZ"],
                "method": ["HF", "CISD", "HF", "CISD", "Full-CI"],
            },
            "hidden": {
                "Z": [2, 3, 4, 5, 2],
                "basis": ["cc-pVQZ", "cc-pVTZ", "cc-pVTZ", "cc-pVDZ", "cc-pV5Z"],
                "n_max": [5, 5, 5, 4, 6],
            },
        },
    },
    "04_fluid_dynamics": {
        "name": "Turbulent Backward-Facing Step",
        "base_params": {
            "Re": 5100, "expansion_ratio": 2.0, "T_averaging": 300,
        },
        "variations": {
            "public": {
                "Re": [1000, 2000, 5100, 8000, 10000],
            },
            "dev": {
                "Re": [1500, 3000, 5100, 7000, 9000],
                "expansion_ratio": [1.5, 1.75, 2.0, 2.5, 3.0],
            },
            "hidden": {
                "Re": [500, 2500, 5100, 12000, 20000],
                "expansion_ratio": [1.25, 2.0, 2.0, 2.0, 3.0],
            },
        },
    },
    "05_thermodynamics": {
        "name": "2D Heat Conduction",
        "base_params": {
            "alpha": 0.01, "T_final": 1.0, "bc_type": "Dirichlet",
            "source": "none", "domain_shape": "square",
        },
        "variations": {
            "public": {
                "alpha": [0.001, 0.005, 0.01, 0.05, 0.1],
                "T_final": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
            "dev": {
                "alpha": [0.002, 0.008, 0.02, 0.08, 0.2],
                "bc_type": ["Dirichlet", "Neumann", "mixed", "Dirichlet", "Robin"],
            },
            "hidden": {
                "alpha": [0.01, 0.01, 0.01, 0.01, 0.01],
                "domain_shape": ["L-shaped", "annular", "square", "square", "square"],
                "source": ["none", "Gaussian", "none", "discontinuous", "time-dependent"],
            },
        },
    },
    "06_structural_mechanics": {
        "name": "Topology Optimization of Bracket",
        "base_params": {
            "V_frac": 0.4, "E_GPa": 210, "load_N": 1000,
            "support": "cantilever", "penalization": 3,
        },
        "variations": {
            "public": {
                "V_frac": [0.2, 0.3, 0.4, 0.5, 0.6],
                "support": ["cantilever", "cantilever", "cantilever", "MBB", "MBB"],
            },
            "dev": {
                "V_frac": [0.25, 0.35, 0.45, 0.55, 0.4],
                "load_N": [500, 1000, 2000, 5000, 1000],
                "support": ["cantilever", "bridge", "MBB", "cantilever", "L-bracket"],
            },
            "hidden": {
                "V_frac": [0.3, 0.4, 0.3, 0.4, 0.5],
                "support": ["cantilever", "cantilever", "MBB", "MBB", "bridge"],
                "penalization": [3, 3, 3, 4, 3],
            },
        },
    },
    "07_chemical_kinetics": {
        "name": "GRI-Mech 3.0 Ignition Delay",
        "base_params": {
            "T_K": 1500, "p_atm": 10, "phi": 1.0,
            "fuel": "CH4", "reactor": "constant_volume",
        },
        "variations": {
            "public": {
                "T_K": [1000, 1200, 1500, 1800, 2000],
                "p_atm": [1, 5, 10, 20, 50],
            },
            "dev": {
                "T_K": [900, 1100, 1400, 1600, 1900],
                "p_atm": [2, 8, 15, 30, 40],
                "phi": [0.5, 0.8, 1.0, 1.2, 2.0],
            },
            "hidden": {
                "T_K": [800, 850, 900, 950, 1000],
                "p_atm": [10, 10, 10, 10, 10],
                "phi": [0.5, 1.0, 1.5, 2.0, 0.3],
            },
        },
    },
    "08_epidemiology": {
        "name": "COVID-19 County-Level SEIR-D",
        "base_params": {
            "county": "Fulton_GA", "beta_0": 0.35, "sigma": 0.2,
            "gamma": 0.1, "delta": 0.005, "T_days": 180,
        },
        "variations": {
            "public": {
                "county": ["Fulton_GA", "Cook_IL", "LosAngeles_CA", "Harris_TX",
                           "Maricopa_AZ", "King_WA", "MiamiDade_FL", "Wayne_MI"],
            },
            "dev": {
                "county": ["NewYork_NY", "Philadelphia_PA", "Dallas_TX", "SanDiego_CA",
                           "Hennepin_MN", "Allegheny_PA", "Suffolk_MA", "OrangeCounty_CA",
                           "DeKalb_GA", "Bexar_TX", "Clark_NV", "Middlesex_MA"],
            },
            "hidden": {
                "county": ["Fulton_GA", "Cook_IL", "LosAngeles_CA", "Harris_TX",
                           "King_WA"],
                "T_days": [365, 365, 365, 365, 365],
            },
        },
    },
    "09_optics": {
        "name": "Fresnel Diffraction",
        "base_params": {
            "lambda_nm": 632.8, "R_mm": 0.5, "z_m": 0.1,
            "aperture_shape": "circular", "grid_N": 512,
        },
        "variations": {
            "public": {
                "lambda_nm": [400, 500, 632.8, 700, 800],
                "z_m": [0.01, 0.05, 0.1, 0.5, 1.0],
            },
            "dev": {
                "lambda_nm": [450, 550, 632.8, 650, 750],
                "aperture_shape": ["circular", "rectangular", "slit", "circular", "annular"],
            },
            "hidden": {
                "lambda_nm": [632.8, 632.8, 400, 800, 550],
                "aperture_shape": ["circular", "phase_only", "circular", "annular", "double_slit"],
                "z_m": [0.001, 0.1, 10.0, 0.05, 0.1],
            },
        },
    },
    "10_inverse_problems": {
        "name": "CT Reconstruction (LoDoPaB)",
        "base_params": {
            "n_angles": 128, "image_size": 128, "noise": "Poisson",
            "geometry": "parallel_beam", "regularization": "TV",
        },
        "variations": {
            "public": {
                "n_angles": [64, 96, 128, 180, 256],
            },
            "dev": {
                "n_angles": [32, 48, 72, 128, 512],
                "regularization": ["TV", "Tikhonov", "TV", "wavelet", "TV"],
            },
            "hidden": {
                "n_angles": [16, 32, 64, 128, 128],
                "geometry": ["parallel_beam", "parallel_beam", "fan_beam",
                             "limited_angle", "truncated"],
            },
        },
    },
    "11_seismic": {
        "name": "Seismic Full-Waveform Inversion (Marmousi)",
        "base_params": {
            "model": "Marmousi-2", "n_sources": 240, "freq_max_Hz": 15,
            "starting_model": "1D_smoothed", "lambda_TV": 1e-3,
        },
        "variations": {
            "public": {
                "freq_max_Hz": [5, 8, 10, 12, 15],
                "n_sources": [60, 120, 240, 240, 240],
            },
            "dev": {
                "freq_max_Hz": [5, 10, 15, 10, 15],
                "starting_model": ["1D_smoothed", "1D_smoothed", "1D_smoothed",
                                    "constant", "water_velocity"],
            },
            "hidden": {
                "model": ["Marmousi-2", "Marmousi-2", "Overthrust", "Marmousi-2", "Marmousi-2"],
                "freq_max_Hz": [5, 15, 10, 15, 15],
                "starting_model": ["constant", "1D_smoothed", "1D_smoothed",
                                    "1D_smoothed", "1D_smoothed"],
            },
        },
    },
    "12_molecular_dynamics": {
        "name": "Protein Conformational Sampling",
        "base_params": {
            "protein": "HP35", "force_field": "ff14SB", "solvent": "GB-Neck2",
            "T_K": 300, "T_sim_ns": 1000, "dt_fs": 2.0,
        },
        "variations": {
            "public": {
                "T_K": [280, 290, 300, 310, 320],
            },
            "dev": {
                "T_K": [285, 295, 300, 315, 340],
                "force_field": ["ff14SB", "CHARMM36m", "ff14SB", "OPLS-AA", "ff14SB"],
            },
            "hidden": {
                "protein": ["HP35", "HP35", "Trp-cage", "chignolin", "HP35"],
                "T_K": [300, 350, 300, 300, 280],
                "solvent": ["GB-Neck2", "GB-Neck2", "GB-Neck2", "GB-Neck2", "TIP3P"],
            },
        },
    },
}


def make_seed(domain, tier, instance):
    """Deterministic seed from domain/tier/instance for reproducibility."""
    s = f"{SEED_BASE}_{domain}_{tier}_{instance}"
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


def generate_instances(domain_key, domain_info, tier, n_target):
    """Generate n_target task instances for a domain/tier combination."""
    variations = domain_info["variations"].get(tier, {})
    base = domain_info["base_params"].copy()
    tasks = []

    # Determine number of variation combinations
    var_keys = list(variations.keys())
    if not var_keys:
        var_keys = list(base.keys())

    for i in range(n_target):
        seed = make_seed(domain_key, tier, i)
        rng = np.random.RandomState(seed)

        task = base.copy()
        task["_task_id"] = f"{domain_key}_{tier}_{i:03d}"
        task["_seed"] = seed
        task["_tier"] = tier
        task["_domain"] = domain_key

        # Select parameter values from variation lists
        for k, vals in variations.items():
            if isinstance(vals, list) and len(vals) > 0:
                idx = i % len(vals)
                # Add small random perturbation for non-categorical params
                val = vals[idx]
                if isinstance(val, (int, float)) and tier != "public":
                    val = val * (1 + rng.uniform(-0.05, 0.05))
                    if isinstance(vals[idx], int):
                        val = int(round(val))
                task[k] = val

        tasks.append(task)

    return tasks


def write_task(task, domain_dir):
    """Write a single task as a JSON file."""
    tier = task["_tier"]
    tid = task["_task_id"]
    tier_dir = domain_dir / tier
    tier_dir.mkdir(parents=True, exist_ok=True)

    fpath = tier_dir / f"{tid}.json"
    with open(fpath, "w") as f:
        json.dump(task, f, indent=2, default=str)
    return fpath


def main():
    total = 0
    catalog = {
        "benchmark": "Universal Simulation Benchmark v1.0",
        "total_tasks": 0,
        "domains": {},
        "tiers": {
            "public": "Ground truth provided, standard parameters",
            "dev": "Blind evaluation, modified parameters",
            "hidden": "Adversarial instances, edge cases",
        },
        "tasks_per_domain": 300,
        "instances_per_tier": {"public": 100, "dev": 100, "hidden": 100},
    }

    for domain_key, domain_info in DOMAINS.items():
        domain_dir = BENCHMARK_ROOT / domain_key
        domain_tasks = {"public": 0, "dev": 0, "hidden": 0}

        for tier, n in [("public", 100), ("dev", 100), ("hidden", 100)]:
            tasks = generate_instances(domain_key, domain_info, tier, n)
            for task in tasks:
                write_task(task, domain_dir)
                domain_tasks[tier] += 1
                total += 1

        catalog["domains"][domain_key] = {
            "name": domain_info["name"],
            "spec": f"{domain_key}/spec.md",
            "tasks": domain_tasks,
            "total": sum(domain_tasks.values()),
        }

    catalog["total_tasks"] = total

    # Write catalog
    catalog_path = BENCHMARK_ROOT / "catalog.json"
    with open(catalog_path, "w") as f:
        json.dump(catalog, f, indent=2)

    print(f"Generated {total} tasks across {len(DOMAINS)} domains")
    print(f"Catalog written to {catalog_path}")

    # Verification
    for domain_key in DOMAINS:
        domain_dir = BENCHMARK_ROOT / domain_key
        for tier in ["public", "dev", "hidden"]:
            tier_dir = domain_dir / tier
            count = len(list(tier_dir.glob("*.json")))
            print(f"  {domain_key}/{tier}: {count} tasks")


if __name__ == "__main__":
    main()
