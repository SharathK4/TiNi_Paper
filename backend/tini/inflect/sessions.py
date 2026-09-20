"""An upload and everything derived from it.

A *session* is one set of files a person dropped on the page, plus the choices
made about them and the results of analysing them. Sessions are directories
under ``artifacts/inflect/sessions/<id>/`` written with the same atomic writer
as the rest of the project, so a browser refresh, a crash or a closed laptop
never loses an analysis.

Files can either live inside the session (uploaded through the browser) or be
referenced where they already are (the sample set that ships with the project,
which is far too large to push through a form). Both look the same to callers.
"""

from __future__ import annotations

import json
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..config import Settings
from ..io_atomic import atomic_write_json, json_safe, read_json
from ..progress import ProgressTracker
from .columns import ColumnReport, analyse_columns
from .ensemble import EnsembleReport, analyse_ensemble, summarise_curve
from .ingest import Table, read_table
from .segment import segment_curve
from .transitions import read_curve

PHASES = ["read", "columns", "analyse", "compare", "finalize"]
MAX_DISPLAY_POINTS = 2400
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")


def _slug(name: str) -> str:
    out = SAFE_NAME.sub("_", name).strip("._") or "file"
    return out[:120]


@dataclass
class Source:
    """One file belonging to a session."""

    name: str              # what the user sees
    path: str              # absolute path on disk
    stored: bool           # True when it lives inside the session directory
    size_bytes: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class SessionConfig:
    x_column: int | None = None
    y_column: int | None = None
    n_segments: int | None = None
    penalty_scale: float = 1.0
    detect_transitions: bool = True
    event_threshold: float = 6.0
    x_scale: float = 1.0        # multiply x by this before analysis (e.g. strain -> %)
    x_unit: str | None = None
    y_unit: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "SessionConfig":
        return cls(**{k: v for k, v in (d or {}).items() if k in cls.__dataclass_fields__})

    def segment_kwargs(self) -> dict[str, Any]:
        return {
            "n_segments": self.n_segments,
            "penalty_scale": self.penalty_scale,
            "detect_transitions": self.detect_transitions,
            "event_threshold": self.event_threshold,
        }


def decimate(x: np.ndarray, y: np.ndarray, limit: int = MAX_DISPLAY_POINTS) -> tuple[list[float], list[float]]:
    """Thin a curve for drawing while keeping its extremes.

    Taking every n-th sample would drop the very spikes a reader needs to see,
    so each bucket contributes its smallest and largest value, in the order
    they occur. The shape of the envelope survives.
    """
    n = x.size
    if n <= limit:
        return x.tolist(), y.tolist()
    buckets = max(1, limit // 2)
    edges = np.linspace(0, n, buckets + 1).astype(int)
    xs: list[float] = []
    ys: list[float] = []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        chunk = y[a:b]
        lo = a + int(np.argmin(chunk))
        hi = a + int(np.argmax(chunk))
        for i in sorted((lo, hi)):
            xs.append(float(x[i]))
            ys.append(float(y[i]))
    return xs, ys


class Session:
    """Read/write access to one stored session."""

    def __init__(self, settings: Settings, session_id: str):
        self.settings = settings
        self.id = session_id
        self.root = Path(settings.artifacts_path) / "inflect" / "sessions" / session_id
        self.uploads = self.root / "files"

    # ------------------------------------------------------------- lifecycle
    @classmethod
    def create(cls, settings: Settings, title: str = "") -> "Session":
        sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
        s = cls(settings, sid)
        s.uploads.mkdir(parents=True, exist_ok=True)
        s.write_meta({
            "id": sid,
            "title": title or "Untitled",
            "created_at": time.time(),
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "sources": [],
            "config": SessionConfig().to_dict(),
        })
        return s

    @property
    def meta_path(self) -> Path:
        return self.root / "session.json"

    def exists(self) -> bool:
        return self.meta_path.exists()

    def meta(self) -> dict[str, Any]:
        d = read_json(self.meta_path, default=None)
        if d is None:
            raise FileNotFoundError(f"no session {self.id}")
        return d

    def write_meta(self, d: dict[str, Any]) -> None:
        atomic_write_json(self.meta_path, d)

    def update(self, **fields: Any) -> dict[str, Any]:
        d = self.meta()
        d.update(fields)
        self.write_meta(d)
        return d

    # ---------------------------------------------------------------- inputs
    def add_upload(self, filename: str, data: bytes) -> Source:
        self.uploads.mkdir(parents=True, exist_ok=True)
        name = _slug(filename)
        target = self.uploads / name
        stem, suffix = target.stem, target.suffix
        k = 1
        while target.exists():
            target = self.uploads / f"{stem}-{k}{suffix}"
            k += 1
        target.write_bytes(data)
        src = Source(name=filename, path=str(target), stored=True, size_bytes=len(data))
        d = self.meta()
        d["sources"].append(src.to_dict())
        self.write_meta(d)
        return src

    def add_directory(self, directory: Path, pattern: str = "*") -> list[Source]:
        """Reference files where they already are, without copying them."""
        found = sorted(Path(directory).glob(pattern))
        d = self.meta()
        added: list[Source] = []
        for p in found:
            if not p.is_file():
                continue
            src = Source(name=p.name, path=str(p), stored=False,
                         size_bytes=p.stat().st_size)
            d["sources"].append(src.to_dict())
            added.append(src)
        self.write_meta(d)
        return added

    def sources(self) -> list[Source]:
        return [Source(**s) for s in self.meta().get("sources", [])]

    def source(self, name: str) -> Source:
        for s in self.sources():
            if s.name == name:
                return s
        raise KeyError(name)

    def config(self) -> SessionConfig:
        return SessionConfig.from_dict(self.meta().get("config", {}))

    def set_config(self, cfg: SessionConfig) -> None:
        self.update(config=cfg.to_dict())

    # ----------------------------------------------------------------- reads
    def table(self, name: str, max_rows: int | None = None) -> Table:
        src = self.source(name)
        return read_table(Path(src.path), name=src.name, max_rows=max_rows)

    def column_report(self, name: str | None = None) -> tuple[Table, ColumnReport]:
        first = name or (self.sources()[0].name if self.sources() else None)
        if first is None:
            raise ValueError("session has no files")
        t = self.table(first)
        return t, analyse_columns(t)

    def resolve_columns(self, table: Table, cfg: SessionConfig) -> tuple[int, int]:
        if cfg.x_column is not None and cfg.y_column is not None:
            return cfg.x_column, cfg.y_column
        rep = analyse_columns(table)
        xi = cfg.x_column if cfg.x_column is not None else rep.recommended_x
        yi = cfg.y_column if cfg.y_column is not None else rep.recommended_y
        if xi is None or yi is None:
            raise ValueError("could not decide which columns to use")
        return xi, yi

    def curve(self, name: str, cfg: SessionConfig | None = None) -> dict[str, Any]:
        """Everything the page needs to draw and narrate one file."""
        cfg = cfg or self.config()
        t = self.table(name)
        xi, yi = self.resolve_columns(t, cfg)
        x = t.data[:, xi] * (cfg.x_scale or 1.0)
        y = t.data[:, yi]
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]

        seg = segment_curve(x, y, **cfg.segment_kwargs())
        xcol, ycol = t.columns[xi], t.columns[yi]
        reading = read_curve(
            x, y, seg,
            x_label=xcol.name, y_label=ycol.name,
            x_unit=cfg.x_unit or xcol.unit, y_unit=cfg.y_unit or ycol.unit,
        )

        dx, dy = decimate(x, y)
        fits = [{
            "x0": s.x0, "x1": s.x1, "model": s.model, "degree": s.degree,
            "points": [[float(v), float(s.evaluate(v))]
                       for v in np.linspace(s.x0, s.x1, 60 if s.degree else 2)],
        } for s in seg.segments]

        return json_safe({
            "file": name,
            "x_column": xi,
            "y_column": yi,
            "x_label": xcol.name,
            "y_label": ycol.name,
            "x_unit": cfg.x_unit or xcol.unit,
            "y_unit": cfg.y_unit or ycol.unit,
            "n_points": int(x.size),
            "curve": {"x": dx, "y": dy, "decimated": len(dx) < int(x.size)},
            "domain": [float(x.min()), float(x.max())],
            "range": [float(y.min()), float(y.max())],
            "fits": fits,
            "segmentation": seg.to_dict(),
            "reading": reading.to_dict(),
            "columns": [c.to_dict() for c in t.columns],
        })

    # -------------------------------------------------------------- ensemble
    def ensemble_path(self) -> Path:
        return self.root / "ensemble.json"

    def load_ensemble(self) -> dict[str, Any] | None:
        return read_json(self.ensemble_path(), default=None)

    def run_ensemble(self, cfg: SessionConfig | None = None,
                     progress: ProgressTracker | None = None,
                     cancel: threading.Event | None = None) -> EnsembleReport:
        cfg = cfg or self.config()
        sources = self.sources()
        if progress:
            progress.start_stage("compare", files_total=len(sources))

        summaries = []
        xi = yi = None
        for i, src in enumerate(sources):
            if cancel is not None and cancel.is_set():
                break
            try:
                t = read_table(Path(src.path), name=src.name)
                if xi is None:
                    xi, yi = self.resolve_columns(t, cfg)
                x = t.data[:, xi] * (cfg.x_scale or 1.0)
                y = t.data[:, yi]
                good = np.isfinite(x) & np.isfinite(y)
                summaries.append(summarise_curve(src.name, x[good], y[good],
                                                 **cfg.segment_kwargs()))
            except Exception as exc:
                from .ensemble import CurveSummary
                summaries.append(CurveSummary(name=src.name, ok=False,
                                              error=f"{type(exc).__name__}: {exc}"))
            if progress:
                progress.file_done(i + 1)

        report = analyse_ensemble(summaries)
        atomic_write_json(self.ensemble_path(), report.to_dict())
        return report


# --------------------------------------------------------------------- store
def sessions_root(settings: Settings) -> Path:
    return Path(settings.artifacts_path) / "inflect" / "sessions"


def list_sessions(settings: Settings, limit: int = 50) -> list[dict[str, Any]]:
    root = sessions_root(settings)
    if not root.exists():
        return []
    out: list[dict[str, Any]] = []
    for d in sorted(root.iterdir(), reverse=True):
        meta = read_json(d / "session.json", default=None)
        if not meta:
            continue
        out.append({
            "id": meta.get("id", d.name),
            "title": meta.get("title", ""),
            "created_at_iso": meta.get("created_at_iso"),
            "n_files": len(meta.get("sources", [])),
            "has_ensemble": (d / "ensemble.json").exists(),
        })
        if len(out) >= limit:
            break
    return out


def open_session(settings: Settings, session_id: str) -> Session:
    s = Session(settings, session_id)
    if not s.exists():
        raise FileNotFoundError(f"no session {session_id}")
    return s
