"""Read a numeric table without being told its dialect.

The user uploads whatever their instrument or simulation wrote. This module
works out, from the bytes alone: which leading lines are comments, which line
holds the column names, what separates the fields, and what each column is
called and measured in. Everything it infers is reported with a reason, so the
interface can show its working and the user can overrule it.

Dialects seen in the wild and covered here::

    # time(1) lx(2) strain(5) -pxx(6)        hash comment, names with indices
    #time(1)  | x(2) | Force per atom (3)    pipe-separated names
    time,stress,strain                        plain CSV header
    Force [pN]\tExtension [nm]                units in brackets
    1.0 2.0 3.0                               no header at all
"""

from __future__ import annotations

import io
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

COMMENT_PREFIXES = ("#", "%", "//", "!", ";")
DELIMITERS = {",": "comma", "\t": "tab", ";": "semicolon", "|": "pipe", None: "whitespace"}

MAX_SNIFF_LINES = 60
MAX_HEADER_FIELDS = 512


@dataclass
class Column:
    """One column, as named by the file or invented for it."""

    index: int
    name: str
    unit: str | None = None
    named_by_file: bool = True

    @property
    def label(self) -> str:
        return f"{self.name} [{self.unit}]" if self.unit else self.name

    def to_dict(self) -> dict[str, Any]:
        return {"index": self.index, "name": self.name, "unit": self.unit,
                "label": self.label, "named_by_file": self.named_by_file}


@dataclass
class Dialect:
    delimiter: str | None
    delimiter_name: str
    comment_prefix: str | None
    n_comment_lines: int
    header_line: str | None
    header_from_comment: bool
    n_fields: int
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "delimiter": self.delimiter_name,
            "comment_prefix": self.comment_prefix,
            "n_comment_lines": self.n_comment_lines,
            "header_line": self.header_line,
            "header_from_comment": self.header_from_comment,
            "n_fields": self.n_fields,
            "reasons": self.reasons,
        }


@dataclass
class Table:
    """A parsed file: the numbers, the columns, and how we read it."""

    name: str
    data: np.ndarray
    columns: list[Column]
    dialect: Dialect
    comment_lines: list[str]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_cols(self) -> int:
        return int(self.data.shape[1])

    def column_index(self, name: str) -> int:
        for c in self.columns:
            if c.name == name:
                return c.index
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "n_rows": self.n_rows,
            "n_cols": self.n_cols,
            "columns": [c.to_dict() for c in self.columns],
            "dialect": self.dialect.to_dict(),
            "comment_lines": self.comment_lines[:12],
            "warnings": self.warnings,
        }


# --------------------------------------------------------------------- utils
def _is_comment(line: str) -> str | None:
    s = line.lstrip()
    for p in COMMENT_PREFIXES:
        if s.startswith(p):
            return p
    return None


def _split(line: str, delim: str | None) -> list[str]:
    parts = line.split(delim) if delim is not None else line.split()
    return [p.strip() for p in parts if not (delim is None and p == "")]


def _numeric_fraction(fields: list[str]) -> float:
    """What fraction of the fields parse as a number (blank counts as one)."""
    if not fields:
        return 0.0
    ok = 0
    for f in fields:
        t = f.strip()
        if t == "" or t.lower() in ("nan", "na", "null", "none", "-", "inf", "-inf"):
            ok += 1
            continue
        try:
            float(t)
            ok += 1
        except ValueError:
            pass
    return ok / len(fields)


def _score_delimiter(lines: list[str], delim: str | None) -> tuple[float, int]:
    """Score a candidate separator: fields must be numeric and their count
    must be the same on every data line. Returns (score, n_fields)."""
    counts: list[int] = []
    numeric: list[float] = []
    for ln in lines:
        f = _split(ln, delim)
        if not f:
            continue
        counts.append(len(f))
        numeric.append(_numeric_fraction(f))
    if not counts:
        return (0.0, 0)
    mode = max(set(counts), key=counts.count)
    if mode < 1:
        return (0.0, 0)
    consistency = counts.count(mode) / len(counts)
    num = float(np.mean(numeric))
    # More columns is weak evidence of the right split, but only once the
    # split is both consistent and numeric; a wrong delimiter usually gives
    # one fat non-numeric field.
    width_bonus = min(mode, 40) / 40.0 * 0.15
    return (consistency * num + width_bonus * (consistency > 0.9) * (num > 0.9), mode)


# --------------------------------------------------------------- header names
_INDEX_SUFFIX = re.compile(r"\s*\(\s*\d+\s*\)\s*$")
_UNIT_BRACKET = re.compile(r"^(?P<name>.*?)\s*[\[\{<]\s*(?P<unit>[^\]\}>]+)\s*[\]\}>]\s*$")
_UNIT_PAREN = re.compile(r"^(?P<name>.*?)\s*\(\s*(?P<unit>[^)]*[A-Za-z/%°][^)]*)\s*\)\s*$")
_UNIT_SLASH = re.compile(r"^(?P<name>[^,]+?)\s*,\s*(?P<unit>[A-Za-zÅµ°%][\w/^.·⁻¹²³ -]*?)\s*$")


def parse_column_name(raw: str, index: int) -> Column:
    """Split one header field into a name and a unit.

    ``strain(5)`` → strain; ``Force per atom (3)`` → Force per atom;
    ``f [pN]`` → f, pN; ``U, pN A`` → U, pN A; ``-pxx`` → -pxx.
    """
    s = raw.strip().strip("\"'")
    if not s:
        return Column(index=index, name=f"column {index + 1}", unit=None, named_by_file=False)

    # a trailing "(7)" is a column index, not a unit
    s = _INDEX_SUFFIX.sub("", s).strip()

    unit: str | None = None
    for rx in (_UNIT_BRACKET, _UNIT_PAREN, _UNIT_SLASH):
        m = rx.match(s)
        if m:
            cand_unit = m.group("unit").strip()
            cand_name = m.group("name").strip()
            # "(1)" style already removed; reject a "unit" that is only digits
            if cand_name and cand_unit and not cand_unit.replace(".", "").isdigit():
                s, unit = cand_name, cand_unit
                break

    # A leading minus is part of the name ("-pxx" is a stress, not a pressure),
    # so only trailing separators are stripped.
    s = s.lstrip(" _\t").rstrip(" _-\t") or f"column {index + 1}"
    return Column(index=index, name=s, unit=unit, named_by_file=True)


def _header_fields(line: str, delim: str | None, n_fields: int) -> list[str] | None:
    """Split a candidate header line, preferring a pipe layout when it is
    present and gives the right number of fields."""
    body = line.lstrip()
    for p in COMMENT_PREFIXES:
        if body.startswith(p):
            body = body[len(p):]
            break
    body = body.strip()
    if not body:
        return None

    candidates: list[list[str]] = []
    if "|" in body:
        candidates.append([f.strip() for f in body.split("|")])
    candidates.append(_split(body, delim))
    if delim is not None:
        candidates.append(_split(body, None))

    for fields in candidates:
        fields = [f for f in fields if f != ""]
        if len(fields) == n_fields and len(fields) <= MAX_HEADER_FIELDS:
            return fields
    return None


# ------------------------------------------------------------------- sniffing
def sniff(text: str) -> tuple[Dialect, int]:
    """Work out the dialect and the line index where numbers begin."""
    lines = text.splitlines()
    reasons: list[str] = []

    comment_prefix: str | None = None
    n_comment = 0
    for ln in lines[:MAX_SNIFF_LINES]:
        p = _is_comment(ln)
        if p is None:
            if ln.strip() == "":
                n_comment += 1
                continue
            break
        comment_prefix = comment_prefix or p
        n_comment += 1

    body = [ln for ln in lines[n_comment:n_comment + MAX_SNIFF_LINES] if ln.strip()]
    if not body:
        raise ValueError("no data rows found")

    best_delim, best_score, best_fields = None, -1.0, 0
    for d in DELIMITERS:
        score, n = _score_delimiter(body[:40], d)
        if score > best_score:
            best_delim, best_score, best_fields = d, score, n
    reasons.append(f"{DELIMITERS[best_delim]}-separated, {best_fields} fields, consistency score {best_score:.2f}")

    # is the first body line actually a header rather than data?
    data_start = n_comment
    header_line: str | None = None
    header_from_comment = False
    first = body[0]
    if _numeric_fraction(_split(first, best_delim)) < 0.6:
        header_line = first
        header_from_comment = False
        idx = lines.index(first, n_comment)
        data_start = idx + 1
        reasons.append("first non-comment line is text, taken as the header")
    elif comment_prefix is not None:
        # the last comment line before the data is the usual place for names
        for ln in reversed(lines[:n_comment]):
            if _is_comment(ln) and ln.strip(" " + "".join(COMMENT_PREFIXES)).strip():
                header_line = ln
                header_from_comment = True
                reasons.append("column names taken from the last comment line")
                break

    if comment_prefix:
        reasons.append(f"{n_comment} comment line(s) beginning '{comment_prefix}'")

    return (
        Dialect(
            delimiter=best_delim,
            delimiter_name=DELIMITERS[best_delim],
            comment_prefix=comment_prefix,
            n_comment_lines=n_comment,
            header_line=header_line.strip() if header_line else None,
            header_from_comment=header_from_comment,
            n_fields=best_fields,
            reasons=reasons,
        ),
        data_start,
    )


def read_table(source: str | Path | bytes, name: str | None = None, max_rows: int | None = None) -> Table:
    """Parse a numeric table from a path, a string or raw bytes."""
    if isinstance(source, (str, Path)) and not isinstance(source, str) or (
        isinstance(source, str) and "\n" not in source and Path(source).exists()
    ):
        path = Path(source)
        raw = path.read_bytes()
        name = name or path.name
    elif isinstance(source, bytes):
        raw = source
        name = name or "uploaded"
    else:
        raw = str(source).encode("utf-8")
        name = name or "inline"

    text = raw.decode("utf-8", errors="replace")
    if "\x00" in text[:4096]:
        raise ValueError("file looks binary, not a text table")

    dialect, data_start = sniff(text)
    lines = text.splitlines()
    comment_lines = [ln.strip() for ln in lines[:dialect.n_comment_lines] if ln.strip()]

    warnings: list[str] = []
    rows: list[list[float]] = []
    n_fields = dialect.n_fields
    skipped = 0
    for ln in lines[data_start:]:
        if not ln.strip() or _is_comment(ln):
            continue
        f = _split(ln, dialect.delimiter)
        if len(f) != n_fields:
            skipped += 1
            continue
        try:
            rows.append([float(x) if x.strip() not in ("", "-") else math.nan for x in f])
        except ValueError:
            skipped += 1
        if max_rows is not None and len(rows) >= max_rows:
            warnings.append(f"stopped after {max_rows} rows")
            break

    if not rows:
        raise ValueError("no parsable numeric rows")
    if skipped:
        warnings.append(f"{skipped} row(s) skipped: wrong field count or non-numeric")

    data = np.asarray(rows, dtype=float)

    fields = _header_fields(dialect.header_line, dialect.delimiter, n_fields) if dialect.header_line else None
    if fields:
        columns = [parse_column_name(f, i) for i, f in enumerate(fields)]
    else:
        if dialect.header_line:
            warnings.append("header line did not match the number of data fields; columns are numbered")
        columns = [Column(index=i, name=f"column {i + 1}", unit=None, named_by_file=False) for i in range(n_fields)]

    seen: dict[str, int] = {}
    for c in columns:
        if c.name in seen:
            seen[c.name] += 1
            c.name = f"{c.name} ({seen[c.name]})"
        else:
            seen[c.name] = 1

    n_nan = int(np.isnan(data).sum())
    if n_nan:
        warnings.append(f"{n_nan} missing value(s) read as NaN")

    return Table(name=name, data=data, columns=columns, dialect=dialect,
                 comment_lines=comment_lines, warnings=warnings)


def read_header_only(source: str | Path | bytes, name: str | None = None) -> Table:
    """Parse just enough rows to describe the file (for a fast upload preview)."""
    return read_table(source, name=name, max_rows=400)
