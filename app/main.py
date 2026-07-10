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

app = FastAPI(title="Video Analyzer Starter")

app.include_router(router)

app.mount(
    "/output",
    StaticFiles(directory=settings.OUTPUT_DIR),
    name="output",
)
