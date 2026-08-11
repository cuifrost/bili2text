from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


DEFAULT_WORKSPACE_NAME = ".b2t"
LEGACY_WORKSPACE_NAME = "runtime-workspace"


def _default_workspace_root() -> Path:
    """Resolve one stable default while reusing older project-local workspaces."""
    configured = os.getenv("B2T_HOME")
    if configured:
        return Path(configured).expanduser()

    current_directory = Path.cwd()
    for name in (DEFAULT_WORKSPACE_NAME, LEGACY_WORKSPACE_NAME):
        candidate = current_directory / name
        if (candidate / "config.json").exists():
            return candidate

    return Path.home() / DEFAULT_WORKSPACE_NAME


@dataclass(slots=True)
class Settings:
    workspace_root: Path
    downloads_dir: Path
    audio_downloads_dir: Path
    audio_dir: Path
    transcripts_dir: Path
    transcripts_original_dir: Path
    transcripts_edited_dir: Path
    metadata_dir: Path
    tasks_dir: Path
    config_path: Path
    app_db_path: Path

    @classmethod
    def from_workspace(cls, workspace: Path | None = None) -> "Settings":
        root = workspace.expanduser() if workspace is not None else _default_workspace_root()
        return cls(
            workspace_root=root,
            downloads_dir=root / "downloads",
            audio_downloads_dir=root / "audio-downloads",
            audio_dir=root / "audio",
            transcripts_dir=root / "transcripts",
            transcripts_original_dir=root / "transcripts" / "original",
            transcripts_edited_dir=root / "transcripts" / "edited",
            metadata_dir=root / "metadata",
            tasks_dir=root / "tasks",
            config_path=root / "config.json",
            app_db_path=root / "app.db",
        )

    def ensure_directories(self) -> None:
        for directory in (
            self.workspace_root,
            self.downloads_dir,
            self.audio_downloads_dir,
            self.audio_dir,
            self.transcripts_dir,
            self.transcripts_original_dir,
            self.transcripts_edited_dir,
            self.metadata_dir,
            self.tasks_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)
