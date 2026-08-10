from pathlib import Path

from b2t.transcribers.faster_whisper_local import FasterWhisperTranscriber


class FakeSegment:
    id = 0
    start = 0.0
    end = 1.0
    text = "你这不错你这不错你这不错"


class FakeInfo:
    language = "zh"


class FakeModel:
    def __init__(self) -> None:
        self.options = None

    def transcribe(self, audio_path: str, **options):
        self.options = (audio_path, options)
        return iter([FakeSegment()]), FakeInfo()


def test_faster_whisper_transcriber_uses_chinese_vad_and_deduplication() -> None:
    transcriber = FasterWhisperTranscriber(language="zh-CN")
    model = FakeModel()
    transcriber._model = model

    result = transcriber.transcribe(Path("sample.wav"))

    assert result["text"] == "你这不错"
    assert result["language"] == "zh"
    assert model.options is not None
    _, options = model.options
    assert options["language"] == "zh"
    assert options["condition_on_previous_text"] is False
    assert options["vad_filter"] is True
