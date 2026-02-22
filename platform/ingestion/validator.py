"""Validate dataset config YAML before ingestion."""
REQUIRED_FIELDS = ["dataset_id", "name", "modality", "kind", "version", "local_data_dir"]

def validate_config(cfg: dict) -> list:
    """Return list of error strings. Empty = valid."""
    errors = []
    for field in REQUIRED_FIELDS:
        if field not in cfg:
            errors.append(f"Missing required field: {field}")
    valid_kinds = {"simulation", "real", "calibration"}
    if cfg.get("kind") not in valid_kinds:
        errors.append(f"'kind' must be one of {valid_kinds}, got: {cfg.get('kind')!r}")
    local_dir = cfg.get("local_data_dir", "")
    if local_dir and not __import__("os").path.isdir(local_dir):
        errors.append(f"local_data_dir does not exist: {local_dir!r}")
    return errors
