from __future__ import annotations

import importlib
import importlib.util
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from b2t.i18n import dependency_sync_guidance
from b2t.progress import ProgressReporter
from b2t.transcribers.base import Transcriber


class LocalWhisperTranscriber(Transcriber):
    name = "whisper"

    def __init__(
        self,
        model: str = "small",
        device: str | None = None,
        download_root: Path | None = None,
        language: str | None = "zh",
    ) -> None:
        self.model_name = model
        self.device = device
        self.download_root = download_root
        self.language = language
        self._model: Any | None = None

    def transcribe(
        self,
        audio_path: Path,
        *,
        prompt: str | None = None,
        progress: ProgressReporter | None = None,
    ) -> dict[str, Any]:
        model = self._ensure_model()
        if progress is not None:
            progress.running("transcribing", message="transcribing", stage_progress=0.0)
        transcribe_options: dict[str, Any] = {
            "initial_prompt": prompt or None,
            "language": normalize_whisper_language(self.language),
            "task": "transcribe",
            # Independent windows reduce repeated text when a speaker pauses or
            # when the previous 30-second window ended mid-sentence.
            "condition_on_previous_text": False,
            # `verbose=False` keeps Whisper text output quiet while still driving the internal tqdm loop.
            "verbose": False,
        }
        if self.device == "cpu":
            transcribe_options["fp16"] = False
        with whisper_progress(progress):
            result = model.transcribe(str(audio_path), **transcribe_options)
        text = collapse_repeated_phrases((result.get("text") or "").strip())
        return {
            "text": text,
            "segments": result.get("segments", []),
            "language": result.get("language"),
            "device": self.device,
            "model": self.model_name,
        }

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            import whisper
        except ImportError as exc:
            raise RuntimeError(build_whisper_import_error_message()) from exc

        if self.device is None:
            self.device = "cuda" if whisper.torch.cuda.is_available() else "cpu"
        self._model = whisper.load_model(
            self.model_name,
            device=self.device,
            download_root=str(self.download_root) if self.download_root else None,
        )
        return self._model


def build_whisper_import_error_message(*, whisper_available: bool | None = None) -> str:
    if whisper_available is None:
        whisper_available = importlib.util.find_spec("whisper") is not None

    if whisper_available:
        return (
            "Whisper is installed, but the Python environment looks broken. "
            "Recreate `.venv` and sync the required extras again. "
            f"{dependency_sync_guidance('en-US')} "
            "Whisper currently needs Python 3.12 or below."
        )

    return (
        "Whisper support is not installed. "
        f"{dependency_sync_guidance('en-US')} "
        "Whisper currently needs Python 3.12 or below."
    )


@contextmanager
def whisper_progress(progress: ProgressReporter | None):
    if progress is None:
        yield
        return

    module = importlib.import_module("whisper.transcribe")
    original = module.tqdm.tqdm
    module.tqdm.tqdm = lambda *args, **kwargs: WhisperProgressTqdm(progress, *args, **kwargs)
    try:
        yield
    finally:
        module.tqdm.tqdm = original


class WhisperProgressTqdm:
    def __init__(self, progress: ProgressReporter, *args, total: int | None = None, disable: bool = False, **kwargs) -> None:
        self.progress = progress
        self.total = total or 0
        self.disable = disable
        self.n = 0

    def __enter__(self) -> "WhisperProgressTqdm":
        if not self.disable:
            self.progress.running("transcribing", message="transcribing", stage_progress=0.0)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False

    def update(self, amount: int = 1) -> None:
        self.n += amount
        if self.disable:
            return
        if self.total <= 0:
            self.progress.running("transcribing", message="transcribing", indeterminate=True)
            return
        self.progress.running(
            "transcribing",
            message="transcribing",
            stage_progress=min(1.0, self.n / self.total),
        )

    def refresh(self) -> None:
        return

    def close(self) -> None:
        return


def normalize_whisper_language(language: str | None) -> str | None:
    """Map app locale values such as ``zh-CN`` to Whisper language codes."""
    if not language:
        return None
    normalized = language.strip().lower().replace("_", "-")
    if normalized.startswith("zh"):
        return "zh"
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ja"):
        return "ja"
    if normalized.startswith("ko"):
        return "ko"
    return normalized.split("-", 1)[0]


def collapse_repeated_phrases(text: str) -> str:
    """Collapse obvious decoder loops while preserving ordinary repetition.

    Whisper can repeat an identical phrase many times after a noisy or silent
    section. Only exact consecutive repetitions occurring at least three times
    are collapsed, so normal emphasis such as a phrase repeated twice remains.
    """
    cleaned = re.sub(r"[ \t]+", " ", text).strip()
    for _ in range(4):
        collapsed = _collapse_repeated_phrase_once(cleaned)
        if collapsed == cleaned:
            break
        cleaned = collapsed
    return cleaned


def _collapse_repeated_phrase_once(text: str) -> str:
    max_phrase_length = min(32, len(text) // 3)
    for phrase_length in range(max_phrase_length, 2, -1):
        pattern = re.compile(
            rf"(?P<phrase>.{{{phrase_length}}})(?P=phrase){{2,}}",
            re.DOTALL,
        )
        collapsed = pattern.sub(r"\g<phrase>", text)
        if collapsed != text:
            return collapsed
    return text
