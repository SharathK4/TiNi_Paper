"""A result that can be handed to someone else and checked.

A record is everything needed to obtain the same findings again: a hash of
the data as read, a hash of the code that read it, every setting, the seed,
and the findings themselves. It is self-verifying — ``verify`` recomputes the
ledger from the record's own settings against the current data and code, and
reports finding by finding whether the numbers agree. A record that verifies
is a result; one that does not says exactly which numbers moved and why the
reader should look again.

The researcher's own conclusions travel in the record too, under their own
heading, and are never mixed with what the software measured.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import numpy as np

from ..io_atomic import json_safe

VERSION = "4.1"
TOLERANCE = 1e-9


def code_hash() -> str:
    """A fingerprint of the analysis code, so a record names what ran it."""
    here = Path(__file__).parent
    h = hashlib.sha256()
    for p in sorted(here.glob("*.py")):
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return h.hexdigest()[:16]


def data_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_record(
    session_id: str,
    file: str,
    path: Path,
    columns: dict[str, Any],
    settings: dict[str, Any],
    seed: int,
    findings: list[dict[str, Any]],
    notes: list[dict[str, Any]],
    n_rows: int,
) -> dict[str, Any]:
    return json_safe({
        "record_version": VERSION,
        "created_at_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "software": {"name": "Inflect", "version": VERSION, "code_hash": code_hash()},
        "session": session_id,
        "data": {
            "file": file,
            "sha256": data_hash(path),
            "n_rows": int(n_rows),
            "columns": columns,
        },
        "settings": settings,
        "seed": int(seed),
        "findings": findings,
        "interpretations": notes,
        "note": ("'findings' are the software's: 'measured' by a stated procedure or 'read' by a "
                 "stated rule. 'interpretations' are the researcher's and are not claims made "
                 "by the software."),
    })


def compare(recorded: list[dict[str, Any]], recomputed: list[dict[str, Any]],
            tolerance: float = TOLERANCE) -> dict[str, Any]:
    """Finding by finding: the same, moved, missing or new."""
    by_id = {f["id"]: f for f in recomputed}
    same: list[str] = []
    moved: list[dict[str, Any]] = []
    missing: list[str] = []
    for f in recorded:
        g = by_id.pop(f["id"], None)
        if g is None:
            missing.append(f["id"])
            continue
        a, b = f.get("value"), g.get("value")
        ta, tb = f.get("text"), g.get("text")
        if a is None and b is None:
            (same if ta == tb else moved).append(f["id"] if ta == tb else
                                                 {"id": f["id"], "recorded": ta, "now": tb})
            continue
        if a is None or b is None:
            moved.append({"id": f["id"], "recorded": a, "now": b})
            continue
        scale = max(abs(a), abs(b), 1.0)
        if abs(a - b) <= tolerance * scale:
            same.append(f["id"])
        else:
            moved.append({"id": f["id"], "recorded": a, "now": b, "difference": b - a})
    new = list(by_id)
    return {
        "reproduced": not moved and not missing,
        "n_recorded": len(recorded),
        "n_same": len(same),
        "same": same,
        "moved": moved,
        "missing": missing,
        "new": new,
        "tolerance": tolerance,
    }
