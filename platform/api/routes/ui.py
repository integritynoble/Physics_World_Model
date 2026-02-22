from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pathlib

BASE_DIR = pathlib.Path(__file__).parent.parent.parent
templates = Jinja2Templates(directory=str(BASE_DIR / "ui" / "templates"))
router = APIRouter(tags=["ui"])

@router.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})
