"""Atomic, crash-safe file writes.

Every persisted artifact goes through here. The pattern is write-to-temp then
``os.replace`` (atomic on NTFS and POSIX), so a crash mid-write can never leave
a half-written checkpoint in place of a good one. When a file is replaced the
previous good copy is kept as ``<name>.prev`` so a successful checkpoint is
never silently destroyed.
"""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np


def _tmp_path(target: Path) -> Path:
    fd, tmp = tempfile.mkstemp(prefix=f".{target.name}.", suffix=".tmp", dir=target.parent)
    os.close(fd)
    return Path(tmp)


_REPLACE_RETRIES = 40
_REPLACE_SLEEP_S = 0.025


def _replace_with_retry(src: Path, dst: Path) -> None:
    """os.replace, retried on Windows sharing violations (a reader or a sync
    client holding the target open for a few milliseconds)."""
    for attempt in range(_REPLACE_RETRIES):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == _REPLACE_RETRIES - 1:
                raise
            time.sleep(_REPLACE_SLEEP_S)


def _commit(tmp: Path, target: Path, keep_previous: bool) -> None:
    if keep_previous and target.exists():
        prev = target.with_name(target.name + ".prev")
        try:
            _replace_with_retry(target, prev)
        except OSError:
            pass
    _replace_with_retry(tmp, target)


def atomic_write_text(target: Path, text: str, keep_previous: bool = True) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    try:
        with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        _commit(tmp, target, keep_previous)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


class _Encoder(json.JSONEncoder):
    def default(self, o: Any) -> Any:  # noqa: D401
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            v = float(o)
            return None if np.isnan(v) else v
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, Path):
            return str(o)
        return super().default(o)


def atomic_write_json(target: Path, obj: Any, keep_previous: bool = True) -> None:
    atomic_write_text(Path(target), json.dumps(obj, indent=2, cls=_Encoder, allow_nan=False), keep_previous)


def atomic_write_npy(target: Path, arr: np.ndarray, keep_previous: bool = True) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    try:
        with open(tmp, "wb") as fh:
            np.save(fh, arr)
            fh.flush()
            os.fsync(fh.fileno())
        _commit(tmp, target, keep_previous)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def atomic_write_npz(target: Path, keep_previous: bool = True, **arrays: np.ndarray) -> None:
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = _tmp_path(target)
    try:
        with open(tmp, "wb") as fh:
            np.savez(fh, **arrays)
            fh.flush()
            os.fsync(fh.fileno())
        _commit(tmp, target, keep_previous)
    finally:
        if tmp.exists():
            tmp.unlink(missing_ok=True)


def read_json(path: Path, default: Any = None) -> Any:
    path = Path(path)
    if not path.exists():
        return default
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        # A corrupt file (e.g. power loss) falls back to the kept .prev copy.
        prev = path.with_name(path.name + ".prev")
        if prev.exists():
            with open(prev, "r", encoding="utf-8") as fh:
                return json.load(fh)
        return default


def append_line(path: Path, line: str) -> None:
    """Append-only log; not atomic, but a torn last line is harmless."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(line.rstrip("\n") + "\n")
