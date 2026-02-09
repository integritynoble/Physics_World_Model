# Registry Conventions Contract

**Status:** FROZEN — All registry entries across all tasks must follow these conventions.

## ID Format

```
<domain>_<name>_v<N>
```

Examples:
- `cassi_gap_tv_v1`
- `spc_random_mask_v1`
- `ct_radon_fbp_v2`
- `mismatch_disp_step_v1`

## Rules

1. **Lowercase only** — no uppercase letters.
2. **Underscores** — use `_` as separator, no hyphens or spaces.
3. **Monotonic versioning** — version suffix `_v<N>` where N is a positive integer. New versions increment N; old versions are never deleted.
4. **Domain prefix** — first token identifies the modality or subsystem (e.g., `cassi`, `cacti`, `spc`, `ct`, `mri`, `common`).
5. **No special characters** — only `[a-z0-9_]`.

## Lookup Semantics

```python
entry = registry.get(id)
# Raises KeyError on miss — NEVER silent fallback.
# No fuzzy matching, no "did you mean?" at runtime.
```

- `registry.get(id)` returns the entry dict or raises `KeyError`.
- There is no implicit default. If an ID is not found, the caller must handle the error explicitly.

## LLM Outputs

- LLM agents must output **registry IDs only** (not freeform strings).
- IDs are validated mechanically immediately after LLM response.
- If LLM hallucinates an invalid ID → fall back to deterministic default.

## Registry Files

All registries live in `packages/pwm_core/contrib/`:

| File | Content |
|------|---------|
| `modalities.yaml` | 26 modality definitions |
| `solver_registry.yaml` | 43+ solver definitions |
| `compression_db.yaml` | Compression/recoverability tables |
| `mismatch_db.yaml` | Mismatch family definitions |
| `photon_db.yaml` | Photon budget models |
| `metrics_db.yaml` | Quality metric definitions |
| `primitives.yaml` | Primitive operators (NEW, Task B) |
| `graph_templates.yaml` | Graph skeleton templates (NEW, Task B) |

## Adding New Entries

1. Choose an ID following the format above.
2. Add the entry to the appropriate YAML file.
3. Run `make check` to verify registry integrity.
4. If the entry introduces a new domain prefix, document it in a comment at the top of the YAML file.
