import os
from pathlib import Path
from ingestion.checksums import sha256_file

def generate_manifest(local_dir: str, dest_dir: str) -> dict:
    """Generate a manifest for files in local_dir with dest_path entries."""
    files = []
    for fpath in sorted(Path(local_dir).rglob("*")):
        if fpath.is_file():
            rel = str(fpath.relative_to(local_dir))
            files.append({
                "path": rel,
                "dest_path": str(Path(dest_dir) / rel),
                "size_bytes": fpath.stat().st_size,
                "sha256": sha256_file(str(fpath)),
            })
    return {"version": "1.0", "dest_dir": dest_dir, "files": files}
