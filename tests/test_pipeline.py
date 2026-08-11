from pathlib import Path

from b2t.config import Settings
from b2t.downloaders.base import Downloader
from b2t.models import DownloadResult, SourceRef
from b2t.pipeline import B2TPipeline, _parse_ffmpeg_progress_seconds
from b2t.transcribers.base import Transcriber


class FakeDownloader(Downloader):
    name = "fake"

    def __init__(self, video_path: Path) -> None:
        self.video_path = video_path

    def download(self, source: SourceRef, settings: Settings, *, progress=None) -> DownloadResult:
        return DownloadResult(
            source=source,
            video_path=self.video_path,
            title="demo-title",
            webpage_url="https://www.bilibili.com/video/BV1xx411c7XD",
            metadata={
                "title": "demo-title",
                "description": "这是视频简介。",
                "id": "BV1xx411c7XD",
                "bv": "BV1xx411c7XD",
                "uploader": "demo-up",
                "uploader_id": "demo-up-id",
                "upload_date": "20260811",
                "duration": 125,
                "duration_string": "2:05",
                "view_count": 1200,
                "like_count": 99,
                "comment_count": 12,
                "categories": ["知识"],
                "tags": ["测试", "转录"],
                "thumbnail": "https://example.com/thumb.jpg",
            },
        )


class FakeAudioDownloader(FakeDownloader):
    def __init__(self, video_path: Path, audio_path: Path) -> None:
        super().__init__(video_path)
        self.audio_path = audio_path

    def download_audio(self, source: SourceRef, settings: Settings, *, progress=None) -> DownloadResult:
        return DownloadResult(
            source=source,
            video_path=None,
            audio_path=self.audio_path,
            title="demo-title",
            webpage_url="https://www.bilibili.com/video/BV1xx411c7XD",
            metadata={
                "title": "demo-title",
                "description": "这是视频简介。",
                "id": "BV1xx411c7XD",
                "bv": "BV1xx411c7XD",
                "uploader": "demo-up",
                "uploader_id": "demo-up-id",
                "upload_date": "20260811",
                "duration": 125,
                "duration_string": "2:05",
                "view_count": 1200,
                "like_count": 99,
                "comment_count": 12,
                "categories": ["知识"],
                "tags": ["测试", "转录"],
                "thumbnail": "https://example.com/thumb.jpg",
                "media_type": "audio",
            },
        )


class FakeTranscriber(Transcriber):
    name = "fake-whisper"

    def transcribe(self, audio_path: Path, *, prompt: str | None = None, progress=None) -> dict[str, str]:
        assert audio_path.exists()
        return {
            "text": "hello from b2t",
            "language": "zh",
            "model": "small",
        }


class OriginalAudioTranscriber(FakeTranscriber):
    accepts_original_audio = True

    def __init__(self) -> None:
        self.received_audio_path: Path | None = None

    def transcribe(self, audio_path: Path, *, prompt: str | None = None, progress=None) -> dict[str, str]:
        self.received_audio_path = audio_path
        return super().transcribe(audio_path, prompt=prompt, progress=progress)


class PipelineUnderTest(B2TPipeline):
    def _extract_audio(self, video_path: Path, stem: str, progress=None) -> Path:
        audio_path = self.settings.audio_dir / f"{stem}.wav"
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.write_bytes(b"wav")
        return audio_path


def test_pipeline_transcribes_bilibili_source(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    settings.ensure_directories()
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")

    pipeline = PipelineUnderTest(
        settings=settings,
        downloader=FakeDownloader(video_path),
        transcriber=FakeTranscriber(),
    )

    result = pipeline.transcribe("BV1xx411c7XD")
    assert result.text == "hello from b2t"
    assert result.transcript_path.exists()
    assert result.metadata_path.exists()
    assert result.video_path == video_path


def test_pipeline_keeps_original_downloaded_audio_for_cloud_transcriber(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    settings.ensure_directories()
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    audio_path = tmp_path / "audio.m4a"
    audio_path.write_bytes(b"audio")
    transcriber = OriginalAudioTranscriber()

    pipeline = PipelineUnderTest(
        settings=settings,
        downloader=FakeAudioDownloader(video_path, audio_path),
        transcriber=transcriber,
    )

    pipeline.transcribe("BV1xx411c7XD")

    assert transcriber.received_audio_path == audio_path


def test_pipeline_markdown_contains_clickable_bilibili_source(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    settings.ensure_directories()
    video_path = tmp_path / "video.mp4"
    video_path.write_bytes(b"video")
    markdown_dir = tmp_path / "obsidian-inbox"

    pipeline = PipelineUnderTest(
        settings=settings,
        downloader=FakeDownloader(video_path),
        transcriber=FakeTranscriber(),
        markdown_export_dir=markdown_dir,
    )

    result = pipeline.transcribe("https://www.bilibili.com/video/BV1xx411c7XD")
    content = result.markdown_path.read_text(encoding="utf-8")

    assert "[BV1xx411c7XD](https://www.bilibili.com/video/BV1xx411c7XD)" in content
    assert "bv: \"BV1xx411c7XD\"" in content
    assert "这是视频简介。" in content
    assert "UP 主：demo-up" in content
    assert "播放：1,200" in content
    assert "标签：测试、转录" in content


def test_pipeline_respects_custom_output_file(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    settings.ensure_directories()
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"wav")
    output_path = tmp_path / "custom-result"

    pipeline = PipelineUnderTest(
        settings=settings,
        downloader=FakeDownloader(tmp_path / "unused.mp4"),
        transcriber=FakeTranscriber(),
    )

    result = pipeline.transcribe(str(audio_path), output=output_path)
    assert result.transcript_path == output_path.with_suffix(".txt")
    assert result.transcript_path.exists()


def test_pipeline_exports_obsidian_markdown(tmp_path: Path) -> None:
    settings = Settings.from_workspace(tmp_path / ".b2t")
    settings.ensure_directories()
    audio_path = tmp_path / "input.wav"
    audio_path.write_bytes(b"wav")
    markdown_dir = tmp_path / "obsidian-inbox"

    pipeline = PipelineUnderTest(
        settings=settings,
        downloader=FakeDownloader(tmp_path / "unused.mp4"),
        transcriber=FakeTranscriber(),
        markdown_export_dir=markdown_dir,
    )

    result = pipeline.transcribe(str(audio_path))
    assert result.markdown_path == markdown_dir / f"{result.transcript_path.stem}.md"
    content = result.markdown_path.read_text(encoding="utf-8")
    assert content.startswith("---\n")
    assert "# input" in content
    assert "## 转录文本" in content
    assert "hello from b2t" in content


def test_parse_ffmpeg_progress_seconds_supports_us_and_ms() -> None:
    assert _parse_ffmpeg_progress_seconds("out_time_ms=2500000") == 2.5
    assert _parse_ffmpeg_progress_seconds("out_time_us=4000000") == 4.0
    assert _parse_ffmpeg_progress_seconds("progress=continue") is None
