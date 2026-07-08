from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass
class RunResult:
    checkpoint_path: Path
    config: dict[str, Any]
    metrics: dict[str, Any]
    from_cache: bool = False


@runtime_checkable
class Runner(Protocol):
    def run(
        self,
        trainer_class: type,
        config: dict[str, Any],
        epochs: int,
        checkpoint_frequency: int,
        storage_dir: Path,
        resume_from_checkpoint: Path | None = None,
    ) -> RunResult: ...


def resolve_resume_path(spec: str | Path) -> Path:
    """Resolve a user-provided resume spec to a directory containing ``checkpoint.pt``.

    Accepts any of the following:
      * a path directly to the checkpoint directory (``.../checkpoint_NNNNNN/``);
      * a path directly to a ``checkpoint.pt`` file;
      * a path to an experiment / trial / TorchTrainer parent directory — the
        most-recently-modified ``checkpoint.pt`` underneath is selected.
    """

    path = Path(spec).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Resume path does not exist: {path}")
    if path.is_file():
        path = path.parent
    if (path / "checkpoint.pt").is_file():
        return path
    candidates = sorted(
        (p.parent for p in path.rglob("checkpoint.pt")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        raise FileNotFoundError(f"No checkpoint.pt found under {path}")
    return candidates[0]