import logging
import os

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import settings
from app.core.database import Base, engine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

os.makedirs(settings.DOWNLOAD_DIR, exist_ok=True)
os.makedirs(settings.OUTPUT_DIR, exist_ok=True)

Base.metadata.create_all(bind=engine)

app = FastAPI(title="Rasd — AI Video Analytics")

app.include_router(router)


from fastapi import Request
from fastapi.templating import Jinja2Templates

dashboard_templates = Jinja2Templates(directory="app/dashboard/templates")


@app.get("/dashboard")
def dashboard(request: Request):
    return dashboard_templates.TemplateResponse(request, "dashboard.html")


app.mount("/static", StaticFiles(directory="app/dashboard/static"), name="static")

app.mount(
    "/output",
    StaticFiles(directory=settings.OUTPUT_DIR),
    name="output",
)
