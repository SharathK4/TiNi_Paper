"""Dataset discovery, inspection and per-file loading.

Inspection never loads numeric data for every file: it stats each file, reads
the comment header of the first one, counts rows of the first one, and
extrapolates the memory need from that. Loading is per file so the pipeline
can stream through the set and checkpoint after each batch.

File format (from the header of every file in Force_traces.zip)::

    # The original file is property.T300.00001.dat. Note Time is shifted ...
    #x in the header below correponds column number 1
    #time(1)  | x(2) | Force per atom (3)
    0.000000 +6.019922e+01 -6.645426e+00

Column meaning (units confirmed against the paper, see docs/PHYSICS.md):
    time   ps   simulation time, shifted so the ramp starts at 0
    x      Å    simulation-box length along [100] (L_x), *not* the reaction
                coordinate ξ of the landscape model
    force  pN   force per atom, f(t) = (A_xx/n) σ_xx(t)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from .config import PAPER_RATES, Settings

COLUMNS = ("time_ps", "x_A", "force_pN")
BYTES_PER_VALUE = 8  # float64


@dataclass
class TraceFile:
    index: int
    trace_id: str        # e.g. "00001"
    name: str
    path: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "trace_id": self.trace_id, "name": self.name, "size_bytes": self.size_bytes}


_ID_RE = re.compile(r"\.(\d+)\.dat$")


def discover_files(settings: Settings) -> list[TraceFile]:
    root = settings.data_path
    if not root.exists():
        raise FileNotFoundError(f"Dataset directory not found: {root}")
    paths = sorted(root.glob(settings.file_glob))
    if settings.max_files is not None:
        paths = paths[: settings.max_files]
    files: list[TraceFile] = []
    for i, p in enumerate(paths):
        m = _ID_RE.search(p.name)
        tid = m.group(1) if m else f"{i:05d}"
        files.append(TraceFile(i, tid, p.name, str(p), p.stat().st_size))
    return files


def dataset_manifest_hash(files: list[TraceFile]) -> str:
    """Cheap identity for the dataset: names + sizes + mtimes (no content read)."""
    h = hashlib.sha256()
    for f in files:
        st = Path(f.path).stat()
        h.update(f"{f.name}|{st.st_size}|{int(st.st_mtime)}".encode())
    return h.hexdigest()[:16]


def read_header(path: Path, max_lines: int = 20) -> list[str]:
    lines: list[str] = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for _ in range(max_lines):
            line = fh.readline()
            if not line or not line.startswith("#"):
                break
            lines.append(line.rstrip("\n"))
    return lines


def count_rows(path: Path) -> tuple[int, int]:
    """(data rows, comment rows) by streaming the file line by line."""
    data = comments = 0
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            if s.startswith("#"):
                comments += 1
            else:
                data += 1
    return data, comments


def load_trace(path: str | Path) -> np.ndarray:
    """Return an (n, 3) float64 array [time_ps, x_A, force_pN]."""
    arr = np.loadtxt(path, comments="#", dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def iter_traces(files: list[TraceFile]) -> Iterator[tuple[TraceFile, np.ndarray]]:
    for f in files:
        yield f, load_trace(f.path)


def identify_paper_rate(ext_rate_A_per_ps: float) -> dict[str, Any] | None:
    """Match the measured extension rate to one of the paper's seven rates."""
    best = min(PAPER_RATES, key=lambda r: abs(np.log(r["ext_rate_A_per_ps"]) - np.log(ext_rate_A_per_ps)))
    ratio = ext_rate_A_per_ps / best["ext_rate_A_per_ps"]
    if 0.9 <= ratio <= 1.1:
        return {**best, "ratio_measured_over_paper": float(ratio)}
    return None


def inspect_dataset(settings: Settings) -> dict[str, Any]:
    files = discover_files(settings)
    if not files:
        raise FileNotFoundError(f"No files match {settings.file_glob} in {settings.data_path}")

    total_bytes = sum(f.size_bytes for f in files)
    first = Path(files[0].path)
    header = read_header(first)
    n_rows, n_comments = count_rows(first)
    sample = load_trace(first)
    n_cols = int(sample.shape[1])

    t = sample[:, 0]
    x = sample[:, 1]
    dt = round(float(np.median(np.diff(t))), 9) if len(t) > 1 else float("nan")
    ext_rate = float(np.polyfit(t, x, 1)[0]) if len(t) > 2 else float("nan")

    per_file_bytes = n_rows * n_cols * BYTES_PER_VALUE
    est_total = per_file_bytes * len(files)

    return {
        "data_dir": str(settings.data_path),
        "file_glob": settings.file_glob,
        "n_files": len(files),
        "max_files_setting": settings.max_files,
        "total_size_bytes": total_bytes,
        "total_size_mb": round(total_bytes / 1e6, 3),
        "size_bytes_min": min(f.size_bytes for f in files),
        "size_bytes_max": max(f.size_bytes for f in files),
        "first_file": files[0].name,
        "last_file": files[-1].name,
        "header_lines": header,
        "columns": list(COLUMNS),
        "column_units": {"time_ps": "ps", "x_A": "Å (box length L_x)", "force_pN": "pN (per atom)"},
        "rows_per_file_first": n_rows,
        "comment_lines_first": n_comments,
        "n_columns": n_cols,
        "dt_ps": dt,
        "t_start_ps": float(t[0]),
        "t_end_ps": float(t[-1]),
        "x_start_A": float(x[0]),
        "x_end_A": float(x[-1]),
        "strain_end": float((x[-1] - x[0]) / x[0]),
        "extension_rate_A_per_ps": ext_rate,
        "paper_rate_match": identify_paper_rate(ext_rate),
        "memory_estimate": {
            "bytes_per_file_float64": per_file_bytes,
            "all_files_float64_bytes": est_total,
            "all_files_float64_mb": round(est_total / 1e6, 3),
            "note": "Tiny: the whole set fits in a few MB; streaming is kept for resumability, not memory.",
        },
        "files": [f.to_dict() for f in files],
    }
