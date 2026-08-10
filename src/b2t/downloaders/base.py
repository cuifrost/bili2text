from __future__ import annotations

from abc import ABC, abstractmethod

from b2t.config import Settings
from b2t.models import DownloadResult, SourceRef
from b2t.progress import ProgressReporter


class Downloader(ABC):
    name = "downloader"

    @abstractmethod
    def download(
        self,
        source: SourceRef,
        settings: Settings,
        *,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        raise NotImplementedError

    def download_audio(
        self,
        source: SourceRef,
        settings: Settings,
        *,
        progress: ProgressReporter | None = None,
    ) -> DownloadResult:
        """Download a source for transcription, falling back to video download."""
        return self.download(source, settings, progress=progress)
