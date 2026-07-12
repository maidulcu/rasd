#!/usr/bin/env python3
"""
Download sample videos from Pexels API for testing.
"""

import os
import json
import subprocess
import urllib.request
import urllib.parse
from pathlib import Path

PEXELS_API_KEY = os.environ.get("PEXELS_API_KEY", "")
SAMPLE_DIR = Path(__file__).parent / "sample"

CATEGORIES = {
    "retail_cctv": ["security camera retail store", "CCTV shop interior", "surveillance camera mall"],
    "shoplifting": ["person stealing store", "shoplifting caught camera", "theft retail"],
    "normal_shopping": ["people shopping retail store", "customer browsing shop", "supermarket shopping"],
    "employee_activity": ["retail worker customer service", "store employee working", "cashier checkout"],
    "customer_flow": ["people walking mall", "crowd shopping center", "busy retail store"],
}


def search_and_download(category, queries):
    """Search Pexels and download videos."""
    output_dir = SAMPLE_DIR / category
    os.makedirs(output_dir, exist_ok=True)
    downloaded = 0

    for query in queries:
        if downloaded >= 3:
            break

        print(f"  Searching: {query}")
        url = "https://api.pexels.com/videos/search"
        params = urllib.parse.urlencode({"query": query, "per_page": 3})
        req = urllib.request.Request(f"{url}?{params}", headers={
            "Authorization": PEXELS_API_KEY,
            "User-Agent": "Mozilla/5.0"
        })

        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read())
                for video in data.get("videos", []):
                    if downloaded >= 3:
                        break

                    vid_id = video.get("id")
                    files = video.get("video_files", [])

                    # Get HD file
                    hd_file = None
                    for f in files:
                        if f.get("quality") == "hd" and f.get("file_type") == "video/mp4":
                            hd_file = f
                            break
                    if not hd_file and files:
                        hd_file = files[0]

                    if hd_file:
                        out_path = output_dir / f"{category}_{vid_id}.mp4"
                        if out_path.exists():
                            print(f"    Exists: {out_path.name}")
                            downloaded += 1
                            continue

                        print(f"    Downloading: {hd_file['link'][:60]}...")
                        try:
                            subprocess.run([
                                "curl", "-s", "-L", "-o", str(out_path), hd_file["link"]
                            ], check=True, capture_output=True, timeout=60)
                            size_mb = os.path.getsize(out_path) / (1024 * 1024)
                            print(f"    Saved: {out_path.name} ({size_mb:.1f} MB)")
                            downloaded += 1
                        except Exception as e:
                            print(f"    Download error: {e}")

        except Exception as e:
            print(f"    Error: {e}")

    return downloaded


def main():
    if not PEXELS_API_KEY:
        print("Set PEXELS_API_KEY environment variable")
        return

    total = 0
    for category, queries in CATEGORIES.items():
        print(f"\n{'='*50}")
        print(f"Category: {category}")
        print(f"{'='*50}")
        total += search_and_download(category, queries)

    print(f"\nTotal: {total} videos downloaded")

    # List all
    for cat in CATEGORIES:
        cat_dir = SAMPLE_DIR / cat
        if cat_dir.exists():
            vids = list(cat_dir.glob("*.mp4"))
            print(f"\n{cat}: {len(vids)} videos")
            for v in vids:
                size_mb = os.path.getsize(v) / (1024 * 1024)
                print(f"  - {v.name} ({size_mb:.1f} MB)")


if __name__ == "__main__":
    main()
