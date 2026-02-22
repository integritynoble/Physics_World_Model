from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "ui" / "templates"))
router = APIRouter(tags=["ui"])


@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(request, "home.html")


@router.get("/runs", response_class=HTMLResponse)
async def runs_page(request: Request):
    return templates.TemplateResponse(request, "run_status.html")


@router.get("/runs/{run_id}", response_class=HTMLResponse)
async def run_detail(run_id: str, request: Request):
    run_stub = {
        "id": run_id, "status": "unknown", "compute_mode": "auto",
        "modality": None, "diagnosis_verdict": None, "diagnosis_confidence": None,
        "metrics": None, "local_path": None, "error_message": None,
    }
    return templates.TemplateResponse(request, "run_results.html", {"run": run_stub})


@router.get("/datasets", response_class=HTMLResponse)
async def datasets_page(request: Request):
    return templates.TemplateResponse(request, "datasets.html")


@router.get("/modalities", response_class=HTMLResponse)
async def modalities_page(request: Request):
    return templates.TemplateResponse(request, "modalities.html")


@router.get("/bootstrap/new", response_class=HTMLResponse)
async def bootstrap_wizard(request: Request):
    return templates.TemplateResponse(request, "bootstrap_wizard.html")
