from __future__ import annotations

import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path

from b2t.config import Settings
from b2t.downloaders.base import Downloader
from b2t.inputs import parse_source, safe_stem
from b2t.models import DownloadResult, TranscriptResult
from b2t.progress import ProgressReporter
from b2t.transcribers.base import Transcriber


class B2TPipeline:
    def __init__(
        self,
        *,
        settings: Settings,
        downloader: Downloader,
        transcriber: Transcriber,
        markdown_export_dir: Path | None = None,
    ) -> None:
        self.settings = settings
        self.downloader = downloader
        self.transcriber = transcriber
        self.markdown_export_dir = markdown_export_dir

    def transcribe(
        self,
        source_input: str,
        *,
        prompt: str | None = None,
        output: Path | None = None,
        progress: ProgressReporter | None = None,
    ) -> TranscriptResult:
        self.settings.ensure_directories()
        if progress is not None:
            progress.running("preparing", message="preparing")
        source = parse_source(source_input)
        downloaded: DownloadResult | None = None

        if source.kind == "bilibili":
            downloaded = self.downloader.download_audio(source, self.settings, progress=progress)
            source_media_path = downloaded.audio_path or downloaded.video_path
            if source_media_path is None:
                raise RuntimeError("downloader returned no audio or video file")
            if downloaded.audio_path and self.transcriber.accepts_original_audio:
                audio_path = downloaded.audio_path
            else:
                audio_path = self._extract_audio(
                    source_media_path,
                    safe_stem(downloaded.title or source.display_name),
                    progress=progress,
                )
            base_name = downloaded.title or source.display_name
            video_path = downloaded.video_path
        elif source.kind == "video":
            assert source.path is not None
            audio_path = self._extract_audio(source.path, safe_stem(source.display_name), progress=progress)
            base_name = source.display_name
            video_path = source.path
        else:
            assert source.path is not None
            audio_path = source.path
            base_name = source.display_name
            video_path = None

        transcription = self.transcriber.transcribe(audio_path, prompt=prompt, progress=progress)
        text = transcription.get("text", "").strip()
        if not text:
            raise RuntimeError("transcriber returned an empty transcript")

        if progress is not None:
            progress.running("writing_outputs", message="writing_outputs", indeterminate=True)
        transcript_path = self._resolve_output_path(base_name, output)
        metadata_path = self._resolve_metadata_path(transcript_path)
        transcript_path.parent.mkdir(parents=True, exist_ok=True)
        transcript_path.write_text(text + "\n", encoding="utf-8")

        download_metadata = downloaded.metadata if downloaded else {}
        source_url = source.url or (downloaded.webpage_url if downloaded else None)
        if not source_url and isinstance(download_metadata, dict):
            source_url = download_metadata.get("webpage_url")
        source_bv = source.bv
        if not source_bv and isinstance(download_metadata, dict):
            source_bv = download_metadata.get("bv") or download_metadata.get("id")

        video_metadata = dict(download_metadata) if isinstance(download_metadata, dict) else {}
        video_metadata.setdefault("title", base_name)
        video_metadata.setdefault("bv", source_bv)
        video_metadata.setdefault("webpage_url", source_url)
        video_metadata.setdefault("media_type", "local")

        metadata = {
            "source": {
                "raw_input": source.raw_input,
                "kind": source.kind,
                "bv": source_bv,
                "url": source_url,
                "path": str(source.path) if source.path else None,
            },
            "engine": self.transcriber.name,
            "model": transcription.get("model"),
            "audio_path": str(audio_path),
            "downloaded_audio_path": str(downloaded.audio_path) if downloaded and downloaded.audio_path else None,
            "video_path": str(video_path) if video_path else None,
            "download": download_metadata or None,
            "video": video_metadata,
            "language": transcription.get("language"),
            "generated_at": datetime.now().isoformat(),
        }
        provider_metadata = transcription.get("metadata")
        if isinstance(provider_metadata, dict):
            metadata["transcription"] = provider_metadata
        markdown_path = self._write_markdown_export(
            base_name=base_name,
            text=text,
            metadata=metadata,
            transcript_path=transcript_path,
        )
        metadata["markdown_path"] = str(markdown_path) if markdown_path else None
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

        return TranscriptResult(
            source=source,
            engine=self.transcriber.name,
            model=str(transcription.get("model") or ""),
            text=text,
            audio_path=audio_path,
            transcript_path=transcript_path,
            metadata_path=metadata_path,
            video_path=video_path,
            markdown_path=markdown_path,
            metadata=metadata,
        )

    def _write_markdown_export(
        self,
        *,
        base_name: str,
        text: str,
        metadata: dict[str, object],
        transcript_path: Path,
    ) -> Path | None:
        if self.markdown_export_dir is None:
            return None

        export_dir = self.markdown_export_dir.expanduser()
        export_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = export_dir / f"{transcript_path.stem}.md"

        download = metadata.get("download")
        download_data = download if isinstance(download, dict) else {}
        video = metadata.get("video")
        video_data = video if isinstance(video, dict) else download_data
        title = _single_line(str(video_data.get("title") or base_name))
        source_data_raw = metadata.get("source", {})
        source_data = source_data_raw if isinstance(source_data_raw, dict) else {}
        download_url = video_data.get("webpage_url") or download_data.get("webpage_url")
        url = source_data.get("url") or download_url or ""
        url_text = str(url)
        bv = str(source_data.get("bv") or video_data.get("bv") or download_data.get("id") or "")
        safe_url = url_text.replace(")", "%29")
        source_link = f"[{bv or '打开视频'}]({safe_url})" if safe_url else "本地文件"
        description = str(video_data.get("description") or "").strip()
        uploader = str(video_data.get("uploader") or video_data.get("channel") or "")
        uploader_id = str(video_data.get("uploader_id") or video_data.get("channel_id") or "")
        upload_date = _format_upload_date(video_data.get("upload_date"))
        published_at = _format_timestamp(video_data.get("timestamp"))
        duration = video_data.get("duration_string") or _format_duration(video_data.get("duration"))
        categories = _join_values(video_data.get("categories"))
        tags = _join_values(video_data.get("tags"))
        thumbnail = str(video_data.get("thumbnail") or "")

        lines = [
            "---",
            f"title: {json.dumps(title, ensure_ascii=False)}",
            f"source: {json.dumps(url_text, ensure_ascii=False)}",
            f"bv: {json.dumps(bv, ensure_ascii=False)}",
            f"engine: {json.dumps(str(metadata.get('engine') or ''), ensure_ascii=False)}",
            f"model: {json.dumps(str(metadata.get('model') or ''), ensure_ascii=False)}",
            f"language: {json.dumps(str(metadata.get('language') or ''), ensure_ascii=False)}",
            f"generated_at: {json.dumps(str(metadata.get('generated_at') or ''), ensure_ascii=False)}",
            f"uploader: {json.dumps(uploader, ensure_ascii=False)}",
            f"uploader_id: {json.dumps(uploader_id, ensure_ascii=False)}",
            f"upload_date: {json.dumps(upload_date, ensure_ascii=False)}",
            f"published_at: {json.dumps(published_at or '', ensure_ascii=False)}",
            f"duration: {json.dumps(str(duration or ''), ensure_ascii=False)}",
            f"view_count: {json.dumps(video_data.get('view_count'), ensure_ascii=False)}",
            f"like_count: {json.dumps(video_data.get('like_count'), ensure_ascii=False)}",
            f"comment_count: {json.dumps(video_data.get('comment_count'), ensure_ascii=False)}",
            f"thumbnail: {json.dumps(thumbnail, ensure_ascii=False)}",
            f"video_tags: {json.dumps(video_data.get('tags') or [], ensure_ascii=False)}",
            "tags:",
            "  - bili2text",
            "  - transcription",
            "---",
            "",
            f"# {title}",
            "",
            "> [!info] 转录信息",
            f"> - 视频源：{source_link}",
            f"> - BV 号：`{bv}`" if bv else "> - BV 号：无（本地文件）",
            f"> - UP 主：{uploader or '未知'}" + (f" (`{uploader_id}`)" if uploader_id else ""),
            f"> - 发布时间：{upload_date or published_at or '未知'}",
            f"> - 时长：{duration or '未知'}",
            f"> - 播放：{_format_number(video_data.get('view_count'))}",
            f"> - 点赞：{_format_number(video_data.get('like_count'))}",
            f"> - 评论：{_format_number(video_data.get('comment_count'))}",
            f"> - 引擎：{metadata.get('engine') or ''}",
            f"> - 模型：{metadata.get('model') or ''}",
            f"> - 语言：{metadata.get('language') or ''}",
        ]
        if thumbnail:
            lines.append(f"> - 缩略图：[查看图片]({thumbnail.replace(')', '%29')})")
        if categories:
            lines.append(f"> - 分类：{categories}")
        if tags:
            lines.append(f"> - 标签：{tags}")
        lines.extend([
            "",
            "## 视频简介",
            "",
            description or "暂无简介。",
            "",
            "## 转录文本",
            "",
            text.rstrip(),
            "",
        ])
        markdown_path.write_text("\n".join(lines), encoding="utf-8")
        return markdown_path

    def _extract_audio(self, video_path: Path, stem: str, progress: ProgressReporter | None = None) -> Path:
        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            raise RuntimeError("ffmpeg is required to extract audio but was not found on PATH")

        audio_path = self.settings.audio_dir / f"{stem}.wav"
        if progress is None:
            result = subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-i",
                    str(video_path),
                    "-vn",
                    "-acodec",
                    "pcm_s16le",
                    "-ar",
                    "16000",
                    "-ac",
                    "1",
                    str(audio_path),
                ],
                capture_output=True,
                encoding="utf-8",
            )
            if result.returncode != 0:
                stderr = result.stderr.strip() or "unknown ffmpeg error"
                raise RuntimeError(f"ffmpeg failed to extract audio: {stderr}")
            return audio_path

        duration = _probe_media_duration_seconds(video_path)
        progress.running(
            "extracting_audio",
            message="extracting_audio",
            stage_progress=0.0 if duration else None,
            indeterminate=duration is None,
        )
        command = [
            ffmpeg,
            "-y",
            "-i",
            str(video_path),
            "-vn",
            "-acodec",
            "pcm_s16le",
            "-ar",
            "16000",
            "-ac",
            "1",
            "-progress",
            "pipe:1",
            "-nostats",
            str(audio_path),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            encoding="utf-8",
        )
        assert process.stdout is not None
        for line in process.stdout:
            parsed_seconds = _parse_ffmpeg_progress_seconds(line.strip())
            if parsed_seconds is None or duration in (None, 0):
                continue
            progress.running(
                "extracting_audio",
                message="extracting_audio",
                stage_progress=min(1.0, parsed_seconds / duration),
            )
        stderr_text = ""
        if process.stderr is not None:
            stderr_text = process.stderr.read()
        returncode = process.wait()
        if returncode != 0:
            stderr = stderr_text.strip() or "unknown ffmpeg error"
            raise RuntimeError(f"ffmpeg failed to extract audio: {stderr}")
        return audio_path

    def _resolve_output_path(self, base_name: str, output: Path | None) -> Path:
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
            return self.settings.transcripts_original_dir / f"{safe_stem(base_name)}-{timestamp}.txt"

        output = output.expanduser()
        if output.suffix.lower() != ".txt":
            if output.exists() and output.is_dir():
                return output / f"{safe_stem(base_name)}.txt"
            return output.with_suffix(".txt")
        return output

    def _resolve_metadata_path(self, transcript_path: Path) -> Path:
        if transcript_path.is_relative_to(self.settings.workspace_root):
            return self.settings.metadata_dir / f"{transcript_path.stem}.json"
        return transcript_path.with_suffix(".json")


def _parse_ffmpeg_progress_seconds(line: str) -> float | None:
    if line.startswith("out_time_ms="):
        try:
            return int(line.split("=", 1)[1]) / 1_000_000
        except ValueError:
            return None
    if line.startswith("out_time_us="):
        try:
            return int(line.split("=", 1)[1]) / 1_000_000
        except ValueError:
            return None
    return None


def _probe_media_duration_seconds(video_path: Path) -> float | None:
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return None
    result = subprocess.run(
        [
            ffprobe,
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(video_path),
        ],
        capture_output=True,
        encoding="utf-8",
    )
    if result.returncode != 0:
        return None
    try:
        value = float((result.stdout or "").strip())
    except ValueError:
        return None
    return value if value > 0 else None


def _single_line(value: str) -> str:
    return " ".join(value.splitlines()).strip()


def _format_upload_date(value: object) -> str | None:
    if not isinstance(value, str) or len(value) != 8 or not value.isdigit():
        return None
    return f"{value[:4]}-{value[4:6]}-{value[6:]}"


def _format_timestamp(value: object) -> str | None:
    if not isinstance(value, (int, float)):
        return None
    try:
        return datetime.fromtimestamp(value).astimezone().isoformat()
    except (OverflowError, OSError, ValueError):
        return None


def _format_duration(value: object) -> str | None:
    if not isinstance(value, (int, float)) or value < 0:
        return None
    total_seconds = int(value)
    hours, remainder = divmod(total_seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes}:{seconds:02d}"


def _format_number(value: object) -> str:
    if isinstance(value, (int, float)):
        return f"{value:,}"
    return str(value) if value not in (None, "") else "未知"


def _join_values(value: object) -> str:
    if isinstance(value, (list, tuple)):
        return "、".join(str(item) for item in value if item not in (None, ""))
    return str(value) if value not in (None, "") else ""
