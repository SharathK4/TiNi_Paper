"""Stage checkpoint registry.

``artifacts/checkpoints.json`` records, per stage, when it finished, the
fingerprint of the settings and dataset it ran against, and which files it
produced. A stage is skipped on the next run when its fingerprint still
matches and every listed output still exists; otherwise it re-runs and the
new outputs replace the old ones (the previous copy is kept as ``.prev`` by
the atomic writer).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .io_atomic import atomic_write_json, read_json


@dataclass
class StageRecord:
    stage: str
    fingerprint: str
    completed_at: float
    duration_s: float
    outputs: list[str] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage": self.stage,
            "fingerprint": self.fingerprint,
            "completed_at": self.completed_at,
            "completed_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(self.completed_at)),
            "duration_s": self.duration_s,
            "outputs": self.outputs,
            "summary": self.summary,
        }


class CheckpointRegistry:
    def __init__(self, artifacts_dir: Path):
        self.dir = Path(artifacts_dir)
        self.path = self.dir / "checkpoints.json"
        self._data: dict[str, Any] = read_json(self.path, default={}) or {}

    # ------------------------------------------------------------- queries
    def get(self, stage: str) -> dict[str, Any] | None:
        return self._data.get(stage)

    def is_complete(self, stage: str, fingerprint: str) -> bool:
        rec = self._data.get(stage)
        if not rec or rec.get("fingerprint") != fingerprint:
            return False
        for out in rec.get("outputs", []):
            if not (self.dir / out).exists():
                return False
        return True

    def all(self) -> dict[str, Any]:
        return dict(self._data)

    def last_completed(self) -> dict[str, Any] | None:
        recs = [r for r in self._data.values() if r.get("completed_at")]
        if not recs:
            return None
        return max(recs, key=lambda r: r["completed_at"])

    # ------------------------------------------------------------- updates
    def mark_complete(self, rec: StageRecord) -> None:
        self._data[rec.stage] = rec.to_dict()
        atomic_write_json(self.path, self._data)

    def invalidate(self, stage: str) -> None:
        if stage in self._data:
            del self._data[stage]
            atomic_write_json(self.path, self._data)

    def invalidate_all(self) -> None:
        self._data = {}
        atomic_write_json(self.path, self._data)

    def rel(self, path: Path) -> str:
        return str(Path(path).relative_to(self.dir)).replace("\\", "/")
