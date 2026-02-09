# CLI Conventions Contract

**Status:** FROZEN — All CLI subcommands must follow these signatures and conventions.

## Existing Commands (do not modify signatures)

```
pwm run --prompt "..." [--out-dir DIR]
pwm run --spec spec.json [--out-dir DIR]
pwm fit-operator --y FILE --operator ID [--out-dir DIR]
pwm calib-recon --y FILE --operator ID [--out-dir DIR]
pwm view <runbundle_dir>
```

## New Commands (Task A implements these)

```
pwm demo <modality> [--preset NAME] [--run] [--open-viewer] [--export-sharepack]
pwm validate <runbundle_dir>
pwm gallery build [--output-dir DIR]
```

### `pwm demo`

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `modality` | positional | YES | — | Modality key from modalities.yaml |
| `--preset` | string | NO | first available | Preset name (e.g., `tissue`, `satellite`) |
| `--run` | flag | NO | False | Execute simulation + reconstruction |
| `--open-viewer` | flag | NO | False | Launch Streamlit viewer after run |
| `--export-sharepack` | flag | NO | False | Export SharePack to `sharepack/` |

**Behavior:**
1. Load CasePack for `<modality>` with optional `--preset`.
2. If `--run`: execute pipeline, produce RunBundle.
3. If `--export-sharepack`: generate `sharepack/` with teaser.png, summary.md, metrics.json, reproduce.sh, and teaser.mp4 (or skip video with message if encoder unavailable).
4. If `--open-viewer`: launch `pwm view` on the RunBundle.

### `pwm validate`

| Argument | Type | Required | Description |
|----------|------|----------|-------------|
| `runbundle_dir` | positional | YES | Path to RunBundle directory |

**Behavior:**
1. Check `runbundle_manifest.json` exists and parses.
2. Verify all required fields per `runbundle_schema.md`.
3. Verify all artifact files exist.
4. Verify all SHA256 hashes match.
5. Print pass/fail + details. Exit code 0 on pass, 1 on fail.

### `pwm gallery build`

| Argument | Type | Required | Default | Description |
|----------|------|----------|---------|-------------|
| `--output-dir` | string | NO | `docs/gallery/` | Output directory for static site |

**Behavior:**
1. Read RunBundle index or `benchmark_results.json` as fallback.
2. Generate static HTML gallery with 26 modality cards.
3. Write to output directory.

## General Conventions

- All commands use `argparse` (no click/typer dependency).
- Output directories are created automatically if they don't exist.
- All commands support `--help`.
- Exit code 0 = success, 1 = error.
- JSON output goes to stdout; progress/status messages go to stderr.
