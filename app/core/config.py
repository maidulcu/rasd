from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./video_intelligence.db"
    DOWNLOAD_DIR: str = "downloads"
    OUTPUT_DIR: str = "output"
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5
    FRAME_SKIP: int = 2
    MAX_WIDTH: int = 1280

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
