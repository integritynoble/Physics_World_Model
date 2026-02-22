from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables on startup (idempotent; production uses alembic)
    from api.models.run import Base
    from api.deps import engine
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield


app = FastAPI(title="PWM Platform", version="0.1.0", lifespan=lifespan)

# Mount static files (create dir first if needed)
_static_dir = BASE_DIR / "ui" / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "db": "ok"}


from api.routes import runs, ui, bootstrap
app.include_router(runs.router, prefix="/api/v1")
app.include_router(ui.router)
app.include_router(bootstrap.router, prefix="/api/v1")
