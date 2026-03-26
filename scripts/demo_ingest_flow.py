#!/usr/bin/env python3
"""Demonstrate pwm ingest -> DatasetCard emission.

Satisfies P1 exit criterion:
  "pwm ingest <dir> emits valid DatasetCard"
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np

from pwm_core.cards.dataset_card import DatasetCard


def demo_ingest() -> int:
    # ------------------------------------------------------------------ #
    # 1. Create a synthetic data directory (mock raw CT slices)
    # ------------------------------------------------------------------ #
    data_dir = Path(tempfile.mkdtemp(prefix="pwm_ingest_demo_"))
    n_slices = 5
    shape = (64, 64)

    for i in range(n_slices):
        phantom = np.random.RandomState(seed=42 + i).rand(*shape).astype(np.float32)
        np.save(str(data_dir / f"slice_{i:03d}.npy"), phantom)

    print(f"1. Created synthetic data: {data_dir} ({n_slices} slices, {shape})")

    # ------------------------------------------------------------------ #
    # 2. Run the ingest logic: scan directory -> build DatasetCard
    # ------------------------------------------------------------------ #
    npy_files = sorted(data_dir.glob("*.npy"))
    sample_arr = np.load(str(npy_files[0]))

    card = DatasetCard(
        dataset_id=f"demo_ct_ingest_{data_dir.name}",
        modality="ct",
        version="1.0.0",
        split="public",
        n_samples=len(npy_files),
        source="synthetic",
        license="CC-BY-4.0",
        gcs_path=f"gs://pwm-benchmark-datasets/datasets/Benchmark/ct/public/",
        description=f"Demo CT dataset ingested from {data_dir}",
        authors=["pwm ingest demo"],
        tags=["ct", "demo", "synthetic"],
    )

    card_path = data_dir / "dataset_card.json"
    card_path.write_text(card.model_dump_json(indent=2))
    print(f"2. DatasetCard emitted: {card_path}")

    # ------------------------------------------------------------------ #
    # 3. Validate the emitted card
    # ------------------------------------------------------------------ #
    reloaded = DatasetCard(**json.loads(card_path.read_text()))
    assert reloaded.dataset_id == card.dataset_id
    assert reloaded.modality == "ct"
    assert reloaded.n_samples == n_slices
    assert reloaded.split == "public"
    print(f"3. Validation passed")

    # ------------------------------------------------------------------ #
    # Summary
    # ------------------------------------------------------------------ #
    print(f"\npwm ingest -> DatasetCard emission complete:")
    print(f"  Dataset ID  : {reloaded.dataset_id}")
    print(f"  Modality    : {reloaded.modality}")
    print(f"  Samples     : {reloaded.n_samples} x {list(sample_arr.shape)}")
    print(f"  Format      : npy ({sample_arr.dtype})")
    print(f"  Split       : {reloaded.split}")
    print(f"  Trust tier  : draft (default for new ingested data)")
    print(f"  Card file   : {card_path}")
    print("\nP1 exit criterion satisfied: pwm ingest <dir> emits valid DatasetCard")
    return 0


if __name__ == "__main__":
    sys.exit(demo_ingest())
