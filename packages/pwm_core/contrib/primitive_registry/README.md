# Primitive Registry

Versioned, append-only registry of computational and imaging physics primitives.

## Structure

```
primitive_registry/
├── general/v1/primitives.yaml       — 12 universal computational primitives
├── imaging/v1/primitives.yaml       — 11 imaging physics primitives
└── mappings/
    └── imaging_to_general/v1/
        └── mappings.yaml            — decomposition rules: imaging → general
```

## Rules

1. **Append-only** — never delete or rename an existing entry.
2. **New version** — breaking changes require a new version directory (`v2/`).
3. **All entries require a stable `id`** in the form `namespace/vN/Name`.
4. **Mappings** stay in sync with `graph/canonical_decompositions.py`.

## Usage

```python
import yaml
from pathlib import Path

registry_dir = Path("packages/pwm_core/contrib/primitive_registry")
general = yaml.safe_load((registry_dir / "general/v1/primitives.yaml").read_text())
imaging = yaml.safe_load((registry_dir / "imaging/v1/primitives.yaml").read_text())
```
