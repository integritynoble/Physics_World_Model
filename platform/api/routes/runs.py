import uuid
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, desc

from api.deps import get_db
from api.models.run import RunRecord
from api.schemas.run import RunCreateRequest, RunStatusResponse
from workers.tasks import dispatch_pwm_run

router = APIRouter(tags=["runs"])


@router.post("/runs", response_model=RunStatusResponse, status_code=201)
async def create_run(body: RunCreateRequest, db: AsyncSession = Depends(get_db)):
    if body.prompt is None and body.spec is None:
        raise HTTPException(status_code=400, detail="Either prompt or spec is required")

    run_id = f"run_{uuid.uuid4().hex[:12]}"
    spec = body.spec or {"prompt": body.prompt}

    run = RunRecord(
        id=run_id,
        created_at=datetime.utcnow(),
        updated_at=datetime.utcnow(),
        status="queued",
        compute_mode=body.compute_mode,
        prompt=body.prompt,
        spec_json=spec,
    )
    db.add(run)
    await db.commit()
    await db.refresh(run)

    task = dispatch_pwm_run.delay(run_id, spec, body.compute_mode)
    run.celery_task_id = task.id
    await db.commit()
    await db.refresh(run)

    return run


@router.get("/runs", response_model=list[RunStatusResponse])
async def list_runs(db: AsyncSession = Depends(get_db)):
    result = await db.execute(
        select(RunRecord).order_by(desc(RunRecord.created_at)).limit(100)
    )
    return result.scalars().all()


@router.get("/runs/{run_id}", response_model=RunStatusResponse)
async def get_run(run_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RunRecord).where(RunRecord.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return run


@router.get("/runs/{run_id}/status")
async def get_run_status(run_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(RunRecord).where(RunRecord.id == run_id))
    run = result.scalar_one_or_none()
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run {run_id!r} not found")
    return {"id": run.id, "status": run.status, "error_message": run.error_message}
