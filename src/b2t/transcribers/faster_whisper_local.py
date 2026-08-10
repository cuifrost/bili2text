from __future__ import annotations

from pathlib import Path
from typing import Any

from b2t.i18n import dependency_sync_guidance
from b2t.progress import ProgressReporter
from b2t.transcribers.base import Transcriber
from b2t.transcribers.whisper_local import (
    collapse_repeated_phrases,
    normalize_whisper_language,
)


class FasterWhisperTranscriber(Transcriber):
    name = "faster-whisper"

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
            progress.running("transcribing", message="transcribing", indeterminate=True)

        segments, info = model.transcribe(
            str(audio_path),
            language=normalize_whisper_language(self.language),
            task="transcribe",
            initial_prompt=prompt or None,
            condition_on_previous_text=False,
            beam_size=5,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
        )
        segment_data = []
        text_parts: list[str] = []
        for segment in segments:
            segment_text = str(segment.text)
            text_parts.append(segment_text)
            segment_data.append(
                {
                    "id": segment.id,
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment_text,
                }
            )

        return {
            "text": collapse_repeated_phrases("".join(text_parts).strip()),
            "segments": segment_data,
            "language": getattr(info, "language", None),
            "device": self.device,
            "model": self.model_name,
        }

    def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model

        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper support is not installed. "
                f"{dependency_sync_guidance('en-US')}"
            ) from exc

        device = self.device or _detect_best_device()
        compute_type = "int8" if device == "cpu" else "float16"
        self._model = WhisperModel(
            self.model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(self.download_root) if self.download_root else None,
        )
        self.device = device
        return self._model


def _detect_best_device() -> str:
    """Prefer CUDA when either PyTorch or CTranslate2 can see a GPU."""
    try:
        import torch
    except ImportError:
        torch = None

    if torch is not None and torch.cuda.is_available():
        return "cuda"

    try:
        from ctranslate2 import get_cuda_device_count
    except (ImportError, RuntimeError):
        return "cpu"

    try:
        return "cuda" if get_cuda_device_count() > 0 else "cpu"
    except RuntimeError:
        return "cpu"
