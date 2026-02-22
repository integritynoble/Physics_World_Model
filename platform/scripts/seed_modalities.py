"""
Seed the modality_basics knowledge base with all 64 imaging modalities.

Pulls comprehensive data from the Physics World Model modality database
(pwm_platform.services.modality_database) so that the DB always mirrors
the authoritative in-code knowledge base.

Run: python scripts/seed_modalities.py
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pwm_platform.services.modality_database import (
    MODALITY_DATABASE,
    list_all_modality_keys,
)

# Fields that map directly to ModalityBasics columns.
_DB_FIELDS = [
    "display_name",
    "category",
    "physics_class",
    "forward_model_family",
    "primitive_gates",
    "wave_model",
    "sensor_type",
    "source_type",
    "geometry",
    "typical_x_dims",
    "typical_y_dims",
    "calibration_params",
    "mismatch_modes",
    "noise_model",
    "default_solver",
    "evaluation_metrics",
    "default_experiment_spec",
    "default_operator_graph",
    "canonical_references",
    "tags",
]


def _build_seed_data() -> list[dict]:
    """Convert MODALITY_DATABASE entries into ModalityBasics-compatible dicts."""
    rows = []
    for key in list_all_modality_keys():
        entry = MODALITY_DATABASE[key]
        row = {"modality_key": key}
        for field in _DB_FIELDS:
            if field in entry:
                row[field] = entry[field]
        rows.append(row)
    return rows


async def main():
    from sqlalchemy import select
    from pwm_platform.db.database import async_session_factory, init_db
    from pwm_platform.db.models import ModalityBasics

    await init_db()

    seed_data = _build_seed_data()

    async with async_session_factory() as db:
        seeded = 0
        updated = 0
        skipped = 0

        for data in seed_data:
            key = data["modality_key"]
            result = await db.execute(
                select(ModalityBasics).where(ModalityBasics.modality_key == key)
            )
            existing = result.scalar_one_or_none()

            if existing:
                # Update existing entry with any new fields
                changed = False
                for field, value in data.items():
                    if field == "modality_key":
                        continue
                    if hasattr(existing, field) and getattr(existing, field) != value:
                        setattr(existing, field, value)
                        changed = True
                if changed:
                    updated += 1
                    print(f"  Update: {key}")
                else:
                    skipped += 1
                    print(f"  Skip (unchanged): {key}")
                continue

            m = ModalityBasics(**data)
            db.add(m)
            seeded += 1
            print(f"  Seed: {key} — {data['display_name']}")

        await db.commit()
        total = len(seed_data)
        print(
            f"\nSeeding complete. {total} modalities total: "
            f"{seeded} new, {updated} updated, {skipped} unchanged."
        )


if __name__ == "__main__":
    asyncio.run(main())
