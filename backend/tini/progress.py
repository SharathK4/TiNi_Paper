"""Progress tracking, persisted to ``artifacts/progress.json``.

The tracker is the single source of truth for the frontend's status panel.
It is written atomically and throttled so a tight per-file loop does not
hammer the disk. ETA is only reported once at least three files have been
timed in the current stage; before that it is ``null`` rather than a guess.
"""

from __future__ import annotations

import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .io_atomic import append_line, atomic_write_json, read_json

STATUS_IDLE = "idle"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_MIN_FILES_FOR_ETA = 3


class ProgressTracker:
    def __init__(self, artifacts_dir: Path, stages: list[str], flush_interval_s: float = 0.25):
        self.dir = Path(artifacts_dir)
        self.path = self.dir / "progress.json"
        self.error_log = self.dir / "errors.log"
        self.stages = list(stages)
        self.flush_interval_s = flush_interval_s
        self._lock = threading.Lock()
        self._last_flush = 0.0

        previous = read_json(self.path, default=None)
        self.state: dict[str, Any] = {
            "status": STATUS_IDLE,
            "stage": None,
            "stage_index": None,
            "n_stages": len(self.stages),
            "stages": self.stages,
            "stage_status": {s: "pending" for s in self.stages},
            "files_done": 0,
            "files_total": 0,
            "percent": 0.0,
            "elapsed_s": 0.0,
            "eta_s": None,
            "message": "",
            "started_at": None,
            "updated_at": time.time(),
            "finished_at": None,
            "last_checkpoint": (previous or {}).get("last_checkpoint"),
            "errors": [],
            "run_id": None,
        }
        # A previous run that was interrupted (laptop shut down mid-run)
        # leaves status=running on disk; on start-up that is a stale claim.
        if previous and previous.get("status") == STATUS_RUNNING:
            self.state["message"] = "Previous run was interrupted; resume to continue."
            self.state["stage_status"] = previous.get("stage_status", self.state["stage_status"])
        elif previous:
            for k in ("stage_status", "status", "finished_at", "elapsed_s", "message", "percent", "started_at"):
                if k in previous:
                    self.state[k] = previous[k]
        self._stage_started = 0.0
        self._stage_files_t0 = 0.0
        self.flush(force=True)

    # ---------------------------------------------------------------- utils
    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            s = dict(self.state)
        if s["status"] == STATUS_RUNNING and s["started_at"]:
            s["elapsed_s"] = time.time() - s["started_at"]
        return s

    def flush(self, force: bool = False) -> None:
        now = time.time()
        if not force and now - self._last_flush < self.flush_interval_s:
            return
        with self._lock:
            self.state["updated_at"] = now
            if self.state["status"] == STATUS_RUNNING and self.state["started_at"]:
                self.state["elapsed_s"] = now - self.state["started_at"]
            atomic_write_json(self.path, self.state, keep_previous=False)
        self._last_flush = now

    # ------------------------------------------------------------ lifecycle
    def start_run(self, run_id: str, resumed_stages: list[str]) -> None:
        with self._lock:
            self.state.update(
                status=STATUS_RUNNING,
                started_at=time.time(),
                finished_at=None,
                run_id=run_id,
                errors=[],
                message="Starting",
                percent=0.0,
            )
            for s in self.stages:
                self.state["stage_status"][s] = "skipped (checkpoint)" if s in resumed_stages else "pending"
        self.flush(force=True)

    def start_stage(self, stage: str, files_total: int = 0, message: str = "") -> None:
        with self._lock:
            self.state.update(
                stage=stage,
                stage_index=self.stages.index(stage),
                files_done=0,
                files_total=files_total,
                eta_s=None,
                message=message or f"Running {stage}",
            )
            self.state["stage_status"][stage] = "running"
            self._stage_started = time.time()
            self._stage_files_t0 = self._stage_started
            self._recompute_percent()
        self.flush(force=True)

    def file_done(self, n_done: int | None = None) -> None:
        with self._lock:
            self.state["files_done"] = n_done if n_done is not None else self.state["files_done"] + 1
            done, total = self.state["files_done"], self.state["files_total"]
            if done >= _MIN_FILES_FOR_ETA and total > done:
                per_file = (time.time() - self._stage_files_t0) / done
                self.state["eta_s"] = round(per_file * (total - done), 2)
            else:
                self.state["eta_s"] = None
            self._recompute_percent()
        self.flush()

    def message(self, text: str) -> None:
        with self._lock:
            self.state["message"] = text
        self.flush()

    def end_stage(self, stage: str, checkpoint_path: str | None) -> None:
        with self._lock:
            self.state["stage_status"][stage] = "done"
            self.state["eta_s"] = None
            if checkpoint_path:
                self.state["last_checkpoint"] = {
                    "stage": stage,
                    "path": checkpoint_path,
                    "time": time.time(),
                    "time_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
                }
            self._recompute_percent()
        self.flush(force=True)

    def finish(self, status: str = STATUS_DONE, message: str = "") -> None:
        with self._lock:
            self.state.update(status=status, finished_at=time.time(), message=message or status)
            if self.state["started_at"]:
                self.state["elapsed_s"] = self.state["finished_at"] - self.state["started_at"]
            if status == STATUS_DONE:
                self.state["percent"] = 100.0
                self.state["stage"] = None
        self.flush(force=True)

    def error(self, stage: str, exc: BaseException) -> None:
        entry = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "stage": stage,
            "type": type(exc).__name__,
            "message": str(exc),
        }
        with self._lock:
            self.state["errors"].append(entry)
            self.state["stage_status"][stage] = "failed"
        append_line(self.error_log, f"{entry['time']} [{stage}] {entry['type']}: {entry['message']}")
        append_line(self.error_log, traceback.format_exc())
        self.flush(force=True)

    # ------------------------------------------------------------ internals
    def _recompute_percent(self) -> None:
        """Whole-run percent: completed stages count fully, the running stage
        counts by its file fraction (or 0.5 when it has no file loop)."""
        n = len(self.stages)
        done = sum(1 for s in self.stages if self.state["stage_status"][s] in ("done", "skipped (checkpoint)"))
        frac = 0.0
        if self.state["stage"] is not None and self.state["stage_status"].get(self.state["stage"]) == "running":
            total = self.state["files_total"]
            frac = (self.state["files_done"] / total) if total else 0.5
        self.state["percent"] = round(100.0 * (done + frac) / n, 2) if n else 0.0
