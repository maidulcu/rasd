import logging
import os
import uuid

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


class VideoDownloader:
    def __init__(self):
        os.makedirs(settings.DOWNLOAD_DIR, exist_ok=True)

    def download(self, url: str) -> str:
        logger.info("Download started: %s", url)

        if not url or not url.startswith(("http://", "https://")):
            raise ValueError(f"Invalid URL: {url}")

        try:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Download failed: %s", e)
            raise RuntimeError(f"Failed to download video: {e}") from e

        content_type = response.headers.get("Content-Type", "")
        if "video" not in content_type and not url.lower().endswith(".mp4"):
            raise ValueError(
                f"URL does not appear to point to a video file: {content_type}"
            )

        ext = ".mp4"
        filename = f"{uuid.uuid4().hex}{ext}"
        local_path = os.path.join(settings.DOWNLOAD_DIR, filename)

        with open(local_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)

        file_size = os.path.getsize(local_path)
        if file_size == 0:
            os.remove(local_path)
            raise RuntimeError("Downloaded file is empty")

        logger.info("Download completed: %s (%d bytes)", local_path, file_size)
        return local_path
