import pytest

from app.video.downloader import VideoDownloader


def test_invalid_url_empty():
    downloader = VideoDownloader()
    with pytest.raises(ValueError, match="Invalid URL"):
        downloader.download("")


def test_invalid_url_no_protocol():
    downloader = VideoDownloader()
    with pytest.raises(ValueError, match="Invalid URL"):
        downloader.download("example.com/video.mp4")


def test_invalid_url_ftp():
    downloader = VideoDownloader()
    with pytest.raises(ValueError, match="Invalid URL"):
        downloader.download("ftp://example.com/video.mp4")


def test_valid_http_url():
    downloader = VideoDownloader()
    with pytest.raises(RuntimeError, match="Failed to download"):
        downloader.download("http://invalid.example.com/video.mp4")


def test_valid_https_url():
    downloader = VideoDownloader()
    with pytest.raises(RuntimeError, match="Failed to download"):
        downloader.download("https://invalid.example.com/video.mp4")
