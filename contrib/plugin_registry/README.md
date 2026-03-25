# PWM Plugin Registry

The PWM Plugin Registry is a versioned, append-only catalog of community-contributed solver plugins. Plugins extend PWM with additional reconstruction algorithms without modifying the core package.

## Registry format

Each entry in `registry.yaml` has the following fields:

| Field | Type | Description |
|-------|------|-------------|
| `id` | string | Unique plugin identifier (snake_case) |
| `version` | string | Semantic version (MAJOR.MINOR.PATCH) |
| `display_name` | string | Human-readable name |
| `description` | string | Short description of the plugin |
| `author` | string | Author or organization |
| `license` | string | SPDX license identifier |
| `modalities` | list | PWM modality IDs this plugin supports |
| `entry_point` | string | Python entry point (`module:function`) |
| `source_url` | string | Canonical source URL |
| `sha256` | string | SHA-256 of the installable artifact |
| `trust_score` | float | Community trust score (0–5) |
| `certified_runs` | int | Number of verified benchmark runs |

## How to submit a plugin

1. Fork the `Physics_World_Model` repository.
2. Add your plugin entry to `contrib/plugin_registry/registry.yaml` following the format above. Do not modify existing entries (the registry is append-only within a schema version).
3. Set `sha256` to the SHA-256 hex digest of your plugin's installable artifact (wheel or tarball).
4. Open a pull request. The PWM maintainers will review the entry and, if approved, run at least one certified benchmark to increment `certified_runs`.

## Installing a plugin

```bash
pwm install gap_tv_enhanced          # install latest version
pwm install gap_tv_enhanced==1.2.0   # install specific version
pwm list-plugins                     # list all available plugins
pwm list-plugins --modality cassi    # filter by modality
pwm uninstall gap_tv_enhanced        # remove a plugin
```

## Trust scores

Trust scores are maintained by the PWM core team and reflect:
- Number of certified benchmark runs
- Absence of security issues
- Code review status
- Community usage statistics

A score of 4.0+ indicates a well-tested, reviewed plugin. Scores below 2.0 indicate early-stage or unreviewed contributions.
