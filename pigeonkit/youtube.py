"""Turn a YouTube (or any yt-dlp supported) link into something OpenCV can play.

    path = fetch("https://youtu.be/...", mode="download")   # -> local .mp4 (robust, replayable)
    url  = fetch("https://youtu.be/...", mode="stream")     # -> direct media URL (starts instantly)

Picks an H.264 MP4 *video-only* stream (no audio -> no ffmpeg merge needed) up to max_height,
which OpenCV's FFmpeg backend decodes everywhere (also on Jetson).
"""
from __future__ import annotations

import os

FORMAT = ("bv*[ext=mp4][vcodec^=avc1][height<={h}]/b[ext=mp4][vcodec^=avc1][height<={h}]/"
          "bv*[height<={h}]/b[height<={h}]/b")


def is_url(s: str) -> bool:
    return s.startswith(("http://", "https://", "ytsearch:"))


def fetch(url: str, mode: str = "download", out_dir: str = "data/youtube", max_height: int = 720) -> str:
    try:
        import yt_dlp
    except ImportError as e:
        raise SystemExit("pip install yt-dlp") from e
    opts = {"format": FORMAT.format(h=max_height), "quiet": True, "no_warnings": True, "noplaylist": True,
            "noprogress": True}
    if mode == "stream":
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info["url"] if "url" in info else info["requested_formats"][0]["url"]
    os.makedirs(out_dir, exist_ok=True)
    opts["outtmpl"] = os.path.join(out_dir, "%(id)s_%(height)sp.%(ext)s")
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)
