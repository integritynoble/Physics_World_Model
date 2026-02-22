from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent

app = FastAPI(title="PWM Platform", version="0.1.0")

# Mount static files (create dir first if needed)
_static_dir = BASE_DIR / "ui" / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

@app.get("/health")
async def health():
    return {"status": "ok", "version": "0.1.0", "db": "ok"}

from api.routes import runs, ui
app.include_router(runs.router, prefix="/api/v1")
app.include_router(ui.router)
