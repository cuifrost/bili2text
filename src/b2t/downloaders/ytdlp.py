from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from b2t.config import Settings
from b2t.downloaders.base import Downloader
from b2t.models import DownloadResult, SourceRef


class YtDlpDownloader(Downloader):
    name = "yt-dlp"

    def download(
        self,
        source: SourceRef,
        settings: Settings,
        *,
        progress=None,
    ) -> DownloadResult:
        if source.kind != "bilibili":
            raise ValueError("yt-dlp downloader only supports bilibili sources")

        settings.ensure_directories()

        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp is not installed. Run `uv sync` to install the core dependencies."
            ) from exc

        ydl_opts = self._build_ydl_opts(source, settings)
        if progress is not None:
            def progress_hook(data: dict[str, Any]) -> None:
                status = data.get("status")
                if status == "downloading":
                    total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
                    downloaded = data.get("downloaded_bytes") or 0
                    stage_progress = (downloaded / total) if total else None
                    progress.running(
                        "downloading",
                        message="downloading",
                        stage_progress=stage_progress,
                        indeterminate=stage_progress is None,
                    )
                elif status == "finished":
                    progress.running("downloading", message="download_finished", stage_progress=1.0)
            ydl_opts["progress_hooks"] = [progress_hook]
            ydl_opts["noprogress"] = False

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source.url or f"https://www.bilibili.com/video/{source.bv}", download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            info = ydl.sanitize_info(info)

            video_path = self._resolve_video_path(ydl, info)
            if not video_path.exists():
                raise RuntimeError(f"yt-dlp reported success but no file was found at {video_path}")

        return DownloadResult(
            source=source,
            video_path=video_path,
            title=info.get("title"),
            webpage_url=info.get("webpage_url") or source.url,
            metadata=self._extract_metadata(info, source, media_type="video"),
        )

    def download_audio(
        self,
        source: SourceRef,
        settings: Settings,
        *,
        progress=None,
    ) -> DownloadResult:
        if source.kind != "bilibili":
            raise ValueError("yt-dlp downloader only supports bilibili sources")

        settings.ensure_directories()
        try:
            from yt_dlp import YoutubeDL
        except ImportError as exc:
            raise RuntimeError(
                "yt-dlp is not installed. Run `uv sync` to install the core dependencies."
            ) from exc

        ydl_opts = self._build_ydl_opts(source, settings, audio_only=True)
        if progress is not None:
            ydl_opts["progress_hooks"] = [_build_progress_hook(progress)]
            ydl_opts["noprogress"] = False

        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(source.url or f"https://www.bilibili.com/video/{source.bv}", download=True)
            if "entries" in info and info["entries"]:
                info = info["entries"][0]
            info = ydl.sanitize_info(info)
            audio_path = self._resolve_video_path(ydl, info)
            if not audio_path.exists():
                raise RuntimeError(f"yt-dlp reported success but no audio file was found at {audio_path}")

        webpage_url = info.get("webpage_url") or source.url
        return DownloadResult(
            source=source,
            video_path=None,
            audio_path=audio_path,
            title=info.get("title"),
            webpage_url=webpage_url,
            metadata=self._extract_metadata(info, source, media_type="audio"),
        )

    def _extract_metadata(
        self,
        info: dict[str, Any],
        source: SourceRef,
        *,
        media_type: str,
    ) -> dict[str, Any]:
        """Keep stable, useful video fields instead of serializing yt-dlp's huge info dict."""
        video_id = info.get("id") or source.bv
        webpage_url = info.get("webpage_url") or info.get("original_url") or source.url
        metadata: dict[str, Any] = {
            "title": info.get("title"),
            "description": info.get("description"),
            "id": video_id,
            "bv": video_id if isinstance(video_id, str) and video_id.startswith("BV") else source.bv,
            "webpage_url": webpage_url,
            "original_url": info.get("original_url") or source.url,
            "uploader": info.get("uploader"),
            "uploader_id": info.get("uploader_id"),
            "uploader_url": info.get("uploader_url"),
            "channel": info.get("channel") or info.get("uploader"),
            "channel_id": info.get("channel_id") or info.get("uploader_id"),
            "channel_url": info.get("channel_url") or info.get("uploader_url"),
            "thumbnail": info.get("thumbnail"),
            "upload_date": info.get("upload_date"),
            "timestamp": info.get("timestamp"),
            "release_timestamp": info.get("release_timestamp"),
            "duration": info.get("duration"),
            "duration_string": info.get("duration_string") or _format_duration(info.get("duration")),
            "view_count": info.get("view_count"),
            "like_count": info.get("like_count"),
            "comment_count": info.get("comment_count"),
            "categories": _as_list(info.get("categories")),
            "tags": _as_list(info.get("tags")),
            "language": info.get("language"),
            "age_limit": info.get("age_limit"),
            "availability": info.get("availability"),
            "media_type": media_type,
        }
        return metadata

    def _build_ydl_opts(
        self,
        source: SourceRef,
        settings: Settings,
        *,
        audio_only: bool = False,
    ) -> dict[str, Any]:
        ydl_opts: dict[str, Any] = {
            "format": "bestaudio/best" if audio_only else "bv*+ba/b",
            "noplaylist": True,
            "outtmpl": str(
                (settings.audio_downloads_dir if audio_only else settings.downloads_dir)
                / "%(id)s.%(ext)s"
            ),
            "noprogress": True,
            "quiet": True,
            "no_warnings": True,
        }
        if not audio_only:
            ydl_opts["merge_output_format"] = "mp4"

        # Support cookies for authenticated access to Bilibili.
        # Priority: B2T_COOKIE_FILE env var > cookies.txt in workspace.
        cookie_file = os.getenv("B2T_COOKIE_FILE")
        if cookie_file:
            cookie_path = Path(cookie_file).expanduser()
        else:
            cookie_path = settings.workspace_root / "cookies.txt"
        if cookie_path.exists():
            ydl_opts["cookiefile"] = str(cookie_path)

        # Bilibili's CDN frequently blocks proxy/VPN nodes, causing 412
        # or SSL errors. Direct connections usually work better.
        # Set B2T_USE_PROXY=1 to re-enable the system proxy if needed.
        use_proxy = os.getenv("B2T_USE_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}
        if not use_proxy:
            ydl_opts["proxy"] = ""

        if source.page is not None:
            ydl_opts["playlist_items"] = str(source.page)
            ydl_opts["noplaylist"] = False
            output_dir = settings.audio_downloads_dir if audio_only else settings.downloads_dir
            ydl_opts["outtmpl"] = str(output_dir / "%(id)s.%(playlist_index)02d.%(ext)s")
        return ydl_opts

    def _resolve_video_path(self, ydl: Any, info: dict[str, Any]) -> Path:
        requested_downloads = info.get("requested_downloads") or []
        for requested in requested_downloads:
            filepath = requested.get("filepath")
            if filepath:
                return Path(filepath)

        prepared = Path(ydl.prepare_filename(info))
        if prepared.exists():
            return prepared

        merged_mp4 = prepared.with_suffix(".mp4")
        if merged_mp4.exists():
            return merged_mp4

        return prepared


def _build_progress_hook(progress):
    def progress_hook(data: dict[str, Any]) -> None:
        status = data.get("status")
        if status == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate") or 0
            downloaded = data.get("downloaded_bytes") or 0
            stage_progress = (downloaded / total) if total else None
            progress.running(
                "downloading",
                message="downloading",
                stage_progress=stage_progress,
                indeterminate=stage_progress is None,
            )
        elif status == "finished":
            progress.running("downloading", message="download_finished", stage_progress=1.0)

    return progress_hook


def _as_list(value: Any) -> list[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _format_duration(value: Any) -> str | None:
    if not isinstance(value, (int, float)) or value < 0:
        return None
    total_seconds = int(value)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"
