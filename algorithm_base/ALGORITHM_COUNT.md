# Algorithm Count Reconciliation

The strategy claims 2,732 solvers across 172 modalities.

## Counting methodology
- **Unique algorithm implementations**: 171 .py files in algorithm_base/ (excluding __init__ and _registry)
- **Unique algorithm names in catalog**: 911 distinct names in _algorithm_catalog.py
- **Variant override entries**: 112 variant keys with 766 algorithm entries in _VARIANT_OVERRIDES
- **Category pool entries**: 158 category keys with 293 algorithm entries in _CATEGORY_ALGORITHMS
- **Total catalog entries**: 1,059 algorithm entries across both sections
- **Algorithm x modality pairs**: Up to 2,732+ when each algorithm is available for multiple modalities via carrier routing
- **Modalities**: 170 in modalities.yaml

The 2,732 figure counts algorithm-modality pairs, not unique implementations.
Each algorithm family (e.g., TV-ADMM) applies to multiple modalities through
the 3-tier routing system:
1. _VARIANT_OVERRIDES -- hand-crafted per-variant algorithm lists
2. _CARRIER_ROUTING -- maps (category, carrier) to algorithm pool key
3. _CATEGORY_ALGORITHMS -- fallback category pools

A single algorithm like "TV-ADMM" appears once in code but serves CT, CBCT,
PET, SPECT, mammography, and other ray-based modalities through carrier routing,
yielding multiple algorithm-modality pairs from one implementation.
