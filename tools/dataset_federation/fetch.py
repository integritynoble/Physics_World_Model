"""tools/dataset_federation/fetch.py

Federated dataset registry query tool.

Usage
-----
    python fetch.py list                        # list all datasets
    python fetch.py list --modality ct          # filter by modality
    python fetch.py info lodopab_ct             # show details for one dataset
    python fetch.py emit-card lodopab_ct        # emit a PWM DatasetCard JSON
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

_REGISTRY_PATH = Path(__file__).parent / "registry.yaml"


def _load_registry():
    with open(_REGISTRY_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_list(args):
    reg = _load_registry()
    datasets = reg["datasets"]
    if args.modality:
        datasets = [d for d in datasets if d["modality"] == args.modality]
    print(f"\n{'ID':<45} {'Modality':<30} {'Samples':>8}  License")
    print("-" * 100)
    for d in datasets:
        print(f"{d['id']:<45} {d['modality']:<30} {d['n_samples']:>8}  {d['license']}")
    print(f"\n{len(datasets)} dataset(s) found.\n")


def cmd_info(args):
    reg = _load_registry()
    entry = next((d for d in reg["datasets"] if d["id"] == args.dataset_id), None)
    if entry is None:
        print(f"ERROR: dataset '{args.dataset_id}' not found.")
        sys.exit(1)
    for key, val in entry.items():
        print(f"  {key:<30} {val}")


def cmd_emit_card(args):
    reg = _load_registry()
    entry = next((d for d in reg["datasets"] if d["id"] == args.dataset_id), None)
    if entry is None:
        print(f"ERROR: dataset '{args.dataset_id}' not found.")
        sys.exit(1)

    # Build a minimal PWM DatasetCard
    try:
        from pwm_core.cards.dataset_card import DatasetCard
        card = DatasetCard(
            dataset_id=entry["pwm_datasetcard_id"],
            display_name=entry["display_name"],
            modality=entry["modality"],
            license=entry["license"],
            source_url=entry.get("source_url", ""),
            n_samples=entry["n_samples"],
            image_size=entry.get("image_size", []),
            noise_model=entry.get("noise_model", ""),
            split_structure=entry.get("split_structure", {}),
            tags=entry.get("tags", []),
        )
        print(json.dumps(card.model_dump(), indent=2, default=str))
    except ImportError:
        # Fallback: just print the registry entry as JSON
        print(json.dumps(entry, indent=2))


def main():
    p = argparse.ArgumentParser(description="PWM Federated Dataset Registry tool")
    sub = p.add_subparsers(dest="cmd")

    p_list = sub.add_parser("list")
    p_list.add_argument("--modality", default="")
    p_list.set_defaults(func=cmd_list)

    p_info = sub.add_parser("info")
    p_info.add_argument("dataset_id")
    p_info.set_defaults(func=cmd_info)

    p_emit = sub.add_parser("emit-card")
    p_emit.add_argument("dataset_id")
    p_emit.set_defaults(func=cmd_emit_card)

    args = p.parse_args()
    if not args.cmd:
        p.print_help()
        return
    args.func(args)


if __name__ == "__main__":
    main()
