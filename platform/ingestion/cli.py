import click, yaml, json, os, shutil, pathlib
from ingestion.manifest import generate_manifest


@click.group()
def main():
    """PWM dataset ingestion CLI."""


@main.command()
@click.option("--config", required=True, type=click.Path(exists=True))
@click.option("--dry-run", is_flag=True, default=False)
def ingest(config, dry_run):
    """Validate, checksum, manifest, and copy a dataset to PWM_WORKSPACE_DIR."""
    from ingestion.validator import validate_config
    with open(config) as f:
        cfg = yaml.safe_load(f)

    errors = validate_config(cfg)
    if errors:
        for e in errors:
            click.echo(f"ERROR: {e}", err=True)
        raise SystemExit(1)

    src_dir = cfg["local_data_dir"].rstrip("/")
    workspace = cfg.get("workspace_dir", os.environ.get("PWM_WORKSPACE_DIR", "/data/workspace"))
    dest_dir = os.path.join(workspace, "datasets", cfg["kind"], cfg["modality"], cfg["version"])

    click.echo(f"Generating manifest for {src_dir}...")
    manifest = generate_manifest(src_dir, dest_dir)
    click.echo(f"Found {len(manifest['files'])} files.")

    if dry_run:
        click.echo(f"[dry-run] Skipping copy to {dest_dir}.")
        return

    pathlib.Path(dest_dir).mkdir(parents=True, exist_ok=True)
    for entry in manifest["files"]:
        src_fpath = os.path.join(src_dir, entry["path"])
        dst_fpath = os.path.join(dest_dir, entry["path"])
        pathlib.Path(dst_fpath).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_fpath, dst_fpath)
        click.echo(f"  Copied {entry['path']}")

    manifest_path = os.path.join(dest_dir, "manifest.json")
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    click.echo(f"Manifest written to {manifest_path}")
    click.echo("Done.")


@main.command()
@click.option("--dataset-id", required=True)
@click.option("--db-url", envvar="DATABASE_URL", required=True)
def register(dataset_id, db_url):
    """Register an already-ingested dataset in the PostgreSQL registry."""
    click.echo(f"Registering dataset {dataset_id} in registry...")
    click.echo("(DB registration requires running API — use the web UI or API endpoint)")
