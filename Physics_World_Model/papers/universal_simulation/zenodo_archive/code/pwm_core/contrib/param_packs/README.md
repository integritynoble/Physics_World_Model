# PWM contrib/param_packs

Param packs are small, reusable bundles of **physical parameters** used by operators:
- PSF models (widefield/confocal)
- SIM illumination patterns
- CASSI masks + dispersion parameterizations
- Sensor pipelines (quantization, saturation, FPN)

## Rules
- Keep packs small and versioned.
- Provide a JSON metadata file + optional NPZ/pt payloads.
- Include units and parameter ranges.

Generated on: 2026-02-01
