from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "postgresql://user:password@localhost/video_intelligence"
    DEVICE: str = "auto"
    DOWNLOAD_DIR: str = "downloads"
    OUTPUT_DIR: str = "output"
    YOLO_MODEL: str = "yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.5

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
