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
from .chorus import ChorusBuilder, sparkline
from .columns import ColumnReport, analyse_columns
from .ensemble import EnsembleReport, analyse_ensemble, summarise_curve
from .findings import build_ledger, ledger_summary
from .record import build_record, compare
from .uncertainty import bootstrap_curve, stability_sweep
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

    # ----------------------------------------------------------------- notes
    # The researcher's conclusions. Stored beside the findings, never among
    # them: a note is attached to a finding by id, carries its own text and
    # time, and is exported under its own heading.
    def notes(self, file: str | None = None) -> list[dict[str, Any]]:
        all_notes = self.meta().get("interpretations", [])
        return [n for n in all_notes if file is None or n.get("file") == file]

    def add_note(self, file: str, finding_id: str, text: str) -> dict[str, Any]:
        text = text.strip()
        if not text:
            raise ValueError("a note needs some text")
        d = self.meta()
        note = {
            "id": uuid.uuid4().hex[:10],
            "file": file,
            "finding_id": finding_id,
            "text": text[:4000],
            "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        d.setdefault("interpretations", []).append(note)
        self.write_meta(d)
        return note

    def remove_note(self, note_id: str) -> bool:
        d = self.meta()
        before = len(d.get("interpretations", []))
        d["interpretations"] = [n for n in d.get("interpretations", []) if n.get("id") != note_id]
        self.write_meta(d)
        return len(d["interpretations"]) < before

    # ---------------------------------------------------------------- ledger
    def ledger(self, name: str, cfg: SessionConfig | None = None,
               with_uncertainty: bool = True, n_resamples: int = 80,
               seed: int = 0) -> dict[str, Any]:
        """Every finding for one file, with intervals and sensitivity."""
        cfg = cfg or self.config()
        t = self.table(name)
        xi, yi = self.resolve_columns(t, cfg)
        x = t.data[:, xi] * (cfg.x_scale or 1.0)
        y = t.data[:, yi]
        good = np.isfinite(x) & np.isfinite(y)
        x, y = x[good], y[good]
        seg = segment_curve(x, y, **cfg.segment_kwargs())
        xu, yu = cfg.x_unit or t.columns[xi].unit, cfg.y_unit or t.columns[yi].unit

        unc = st = None
        if with_uncertainty:
            unc = bootstrap_curve(x, y, seg, n_resamples=n_resamples, seed=seed,
                                  **cfg.segment_kwargs())
            st = stability_sweep(x, y, seg, x_unit=xu, **cfg.segment_kwargs())

        findings = build_ledger(
            x, y, seg, t.columns[xi].name, t.columns[yi].name, xu, yu,
            uncertainty=unc.keyed if unc else None,
            stability=st.keyed if st else None,
            settings=cfg.segment_kwargs(),
        )
        docs = [f.to_dict() for f in findings]
        return json_safe({
            "file": name,
            "x_label": t.columns[xi].name, "y_label": t.columns[yi].name,
            "x_unit": xu, "y_unit": yu,
            "settings": {**cfg.segment_kwargs(), "seed": seed, "n_resamples": n_resamples},
            "findings": docs,
            "summary": ledger_summary(findings),
            "notes": self.notes(name),
            "bootstrap": {"n_used": unc.n_used, "block_length": unc.block_length,
                          "detection_rate": unc.detection_rate} if unc else None,
            "stability_sentence": st.sentence if st else None,
        })

    def record(self, name: str, cfg: SessionConfig | None = None, seed: int = 0) -> dict[str, Any]:
        cfg = cfg or self.config()
        led = self.ledger(name, cfg, with_uncertainty=True, seed=seed)
        src = self.source(name)
        t = self.table(name)
        xi, yi = self.resolve_columns(t, cfg)
        return build_record(
            session_id=self.id, file=name, path=Path(src.path),
            columns={"x": t.columns[xi].to_dict(), "y": t.columns[yi].to_dict()},
            settings={**cfg.to_dict(), "n_resamples": led["settings"]["n_resamples"]},
            seed=seed, findings=led["findings"], notes=self.notes(name),
            n_rows=t.n_rows,
        )

    def verify(self, record: dict[str, Any]) -> dict[str, Any]:
        """Recompute from a record's own settings and compare every finding."""
        name = record["data"]["file"]
        src = self.source(name)
        from .record import code_hash, data_hash
        cfg = SessionConfig.from_dict(record.get("settings", {}))
        led = self.ledger(name, cfg, with_uncertainty=False)
        result = compare(record.get("findings", []), led["findings"])
        result["data_matches"] = data_hash(Path(src.path)) == record["data"].get("sha256")
        result["code_matches"] = code_hash() == record["software"].get("code_hash")
        result["file"] = name
        return json_safe(result)

    # -------------------------------------------------------------- ensemble
    def ensemble_path(self) -> Path:
        return self.root / "ensemble.json"

    def chorus_path(self) -> Path:
        return self.root / "chorus.json"

    def load_chorus(self) -> dict[str, Any] | None:
        return read_json(self.chorus_path(), default=None)

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
        builder = ChorusBuilder()
        thumbs: dict[str, list[float]] = {}
        xi = yi = None
        labels = ("x", "y", None, None)
        for i, src in enumerate(sources):
            if cancel is not None and cancel.is_set():
                break
            try:
                t = read_table(Path(src.path), name=src.name)
                if xi is None:
                    xi, yi = self.resolve_columns(t, cfg)
                    labels = (t.columns[xi].name, t.columns[yi].name,
                              cfg.x_unit or t.columns[xi].unit,
                              cfg.y_unit or t.columns[yi].unit)
                x = t.data[:, xi] * (cfg.x_scale or 1.0)
                y = t.data[:, yi]
                good = np.isfinite(x) & np.isfinite(y)
                x, y = x[good], y[good]
                summary = summarise_curve(src.name, x, y, **cfg.segment_kwargs())
                summaries.append(summary)
                # One pass over the files serves the comparison, the chorus
                # and the thumbnails alike; reading them three times would
                # cost three times as much for the same numbers.
                builder.add(src.name, x, y, summary.transition_x if summary.ok else None)
                thumbs[src.name] = sparkline(x, y)
            except Exception as exc:
                from .ensemble import CurveSummary
                summaries.append(CurveSummary(name=src.name, ok=False,
                                              error=f"{type(exc).__name__}: {exc}"))
            if progress:
                progress.file_done(i + 1)

        report = analyse_ensemble(summaries)
        payload = report.to_dict()
        payload["sparklines"] = thumbs
        atomic_write_json(self.ensemble_path(), payload)
        atomic_write_json(self.chorus_path(),
                          builder.build(*labels).to_dict())
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
