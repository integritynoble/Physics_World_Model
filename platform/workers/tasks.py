import os
import base64
import pathlib

from workers.celery_app import celery_app
from workers.cost_router import should_use_gpu

try:
    from pwm_core.api.endpoints import run as pwm_run
except ImportError:
    pwm_run = None  # Allow import in test environments without pwm_core

WORKSPACE = os.environ.get("PWM_WORKSPACE_DIR", "/data/workspace")


@celery_app.task(bind=True, max_retries=2)
def dispatch_pwm_run(self, run_id: str, spec_dict: dict, compute_mode: str):
    use_gpu = should_use_gpu(spec_dict, compute_mode)
    bundle_dir = str(pathlib.Path(WORKSPACE) / "runbundles" / run_id)
    pathlib.Path(bundle_dir).mkdir(parents=True, exist_ok=True)
    try:
        if use_gpu:
            from workers.modal_app import simulate_gpu
            result = simulate_gpu.remote(spec_dict=spec_dict)
            for fname, b64data in result.pop("_artifacts_b64", {}).items():
                fpath = pathlib.Path(bundle_dir) / fname
                fpath.write_bytes(base64.b64decode(b64data))
        else:
            result = pwm_run(spec=spec_dict, out_dir=bundle_dir)

        _update_run(run_id, status="done", result=result, local_path=bundle_dir)
    except Exception as exc:
        _update_run(run_id, status="failed", error=str(exc)[:500])
        raise self.retry(exc=exc, countdown=30)


def _update_run(run_id: str, status: str, result: dict = None,
                local_path: str = None, error: str = None):
    from sqlalchemy import create_engine, update
    from api.models.run import RunRecord
    db_url = os.environ.get("DATABASE_URL", "").replace("+asyncpg", "").replace("+aiosqlite", "")
    if not db_url:
        return
    engine = create_engine(db_url)
    with engine.begin() as conn:
        vals = {"status": status}
        if result:
            recon = result.get("recon") or []
            vals["metrics"] = recon[0].get("metrics") if recon else None
            diag = result.get("diagnosis") or {}
            vals["diagnosis_verdict"] = diag.get("verdict")
        if local_path:
            vals["local_path"] = local_path
        if error:
            vals["error_message"] = error
        conn.execute(
            update(RunRecord).where(RunRecord.id == run_id).values(**vals)
        )
