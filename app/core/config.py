from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./video_intelligence.db"
    DOWNLOAD_DIR: str = "downloads"
    OUTPUT_DIR: str = "output"
    YOLO_MODEL: str = "yolov8n.pt"
    POSE_MODEL: str = "yolov8n-pose.pt"
    USE_ONNX: bool = False
    CONFIDENCE_THRESHOLD: float = 0.5
    FRAME_SKIP: int = 2
    MAX_WIDTH: int = 1280

    @property
    def detection_model_path(self) -> str:
        if self.USE_ONNX:
            return self.YOLO_MODEL.replace(".pt", ".onnx")
        return self.YOLO_MODEL

    @property
    def pose_model_path(self) -> str:
        if self.USE_ONNX:
            return self.POSE_MODEL.replace(".pt", ".onnx")
        return self.POSE_MODEL

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
