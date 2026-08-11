from __future__ import annotations

import sys
from types import SimpleNamespace
from typing import Any

from b2t.transcribers.bailian import BailianFileTranscriber


class FakeResponse:
    def __init__(self, payload: dict[str, Any], status_code: int = 200) -> None:
        self.payload = payload
        self.status_code = status_code
        self.text = ""

    def json(self) -> dict[str, Any]:
        return self.payload


class FakeSession:
    def __init__(self) -> None:
        self.post_calls: list[dict[str, Any]] = []
        self.get_calls: list[dict[str, Any]] = []
        self._poll_count = 0

    def post(self, url: str, **kwargs: Any) -> FakeResponse:
        self.post_calls.append({"url": url, **kwargs})
        return FakeResponse({"output": {"task_id": "task-123"}})

    def get(self, url: str, **kwargs: Any) -> FakeResponse:
        self.get_calls.append({"url": url, **kwargs})
        if url.endswith("/tasks/task-123"):
            self._poll_count += 1
            if self._poll_count == 1:
                return FakeResponse({"output": {"task_status": "RUNNING"}})
            return FakeResponse(
                {
                    "output": {
                        "task_status": "SUCCEEDED",
                        "result": {"transcription_url": "https://result.example/result.json"},
                    }
                }
            )
        return FakeResponse(
            {
                "transcripts": [
                    {
                        "language": "zh",
                        "sentences": [
                            {"begin_time": 0, "end_time": 1200, "text": "你好，世界。"},
                            {"begin_time": 1200, "end_time": 2400, "text": "这是测试。"},
                        ],
                    }
                ]
            }
        )


class FakeUploader:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def upload_file(self, request: Any, *, filepath: str) -> None:
        self.calls.append({"request": request, "filepath": filepath})


class FakeOssClient:
    def __init__(self) -> None:
        self.uploader_instance = FakeUploader()
        self.deleted: list[Any] = []

    def uploader(self) -> FakeUploader:
        return self.uploader_instance

    def presign(self, request: Any, *, expires) -> SimpleNamespace:
        return SimpleNamespace(url="https://oss.example/audio.wav?signature=secret")

    def delete_object(self, request: Any) -> None:
        self.deleted.append(request)


def test_bailian_filetrans_uploads_polls_and_parses_result(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "access-secret")

    fake_oss = SimpleNamespace(
        PutObjectRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        GetObjectRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        DeleteObjectRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "alibabacloud_oss_v2", fake_oss)

    session = FakeSession()
    oss_client = FakeOssClient()
    transcriber = BailianFileTranscriber(
        api_key="dashscope-key",
        workspace_id="workspace-123",
        oss_bucket="audio-bucket",
        session=session,
        oss_client=oss_client,
        sleep_fn=lambda _seconds: None,
    )

    result = transcriber.transcribe(audio_path)

    assert result["text"] == "你好，世界。\n这是测试。"
    assert result["language"] == "zh"
    assert result["segments"][0]["start"] == 0
    assert result["segments"][0]["end"] == 1.2
    assert result["metadata"]["task_id"] == "task-123"
    assert len(session.post_calls) == 1
    assert session.post_calls[0]["headers"]["X-DashScope-Async"] == "enable"
    assert session.post_calls[0]["json"]["input"]["file_url"].startswith("https://oss.example/")
    assert session.post_calls[0]["json"]["model"] == "qwen3-asr-flash-filetrans"
    assert session.post_calls[0]["json"]["parameters"]["enable_itn"] is False
    assert session.post_calls[0]["json"]["parameters"]["enable_words"] is True
    assert len(oss_client.uploader_instance.calls) == 1
    assert len(oss_client.deleted) == 1


def test_bailian_filetrans_supports_results_list_and_env_tuning(tmp_path, monkeypatch) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")
    monkeypatch.setenv("OSS_ACCESS_KEY_ID", "access-key")
    monkeypatch.setenv("OSS_ACCESS_KEY_SECRET", "access-secret")
    monkeypatch.setenv("B2T_BAILIAN_ENABLE_ITN", "true")
    monkeypatch.setenv("B2T_BAILIAN_ENABLE_WORDS", "0")

    fake_oss = SimpleNamespace(
        PutObjectRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        GetObjectRequest=lambda **kwargs: SimpleNamespace(**kwargs),
        DeleteObjectRequest=lambda **kwargs: SimpleNamespace(**kwargs),
    )
    monkeypatch.setitem(sys.modules, "alibabacloud_oss_v2", fake_oss)

    class ResultsSession(FakeSession):
        def get(self, url: str, **kwargs: Any) -> FakeResponse:
            if url.endswith("/tasks/task-123"):
                return FakeResponse(
                    {
                        "output": {
                            "task_status": "SUCCEEDED",
                            "results": [{"transcription_url": "https://result.example/result.json"}],
                        }
                    }
                )
            return super().get(url, **kwargs)

    session = ResultsSession()
    transcriber = BailianFileTranscriber(
        api_key="dashscope-key",
        workspace_id="workspace-123",
        oss_bucket="audio-bucket",
        session=session,
        oss_client=FakeOssClient(),
        sleep_fn=lambda _seconds: None,
    )

    result = transcriber.transcribe(audio_path)

    assert result["text"]
    assert session.post_calls[0]["json"]["parameters"]["enable_itn"] is True
    assert session.post_calls[0]["json"]["parameters"]["enable_words"] is False


def test_bailian_filetrans_requires_cloud_and_oss_configuration(tmp_path) -> None:
    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"audio")

    transcriber = BailianFileTranscriber(session=FakeSession())

    try:
        transcriber.transcribe(audio_path)
    except RuntimeError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected missing configuration error")

    assert "DASHSCOPE_API_KEY" in message
    assert "OSS_BUCKET" in message
