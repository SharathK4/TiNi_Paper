"""Version 4 — reading a table whose dialect is not known in advance."""

from __future__ import annotations

import numpy as np
import pytest

from tini.inflect.columns import analyse_columns, estimate_noise
from tini.inflect.ingest import parse_column_name, read_table


# ------------------------------------------------------------------ dialects
def test_plain_csv_with_header():
    t = read_table("time,stress,strain\n0,1,0\n1,2,0.01\n2,3,0.02\n3,4,0.03\n", name="t.csv")
    assert t.dialect.delimiter_name == "comma"
    assert [c.name for c in t.columns] == ["time", "stress", "strain"]
    assert t.data.shape == (4, 3)


def test_tab_separated_with_units_in_brackets():
    t = read_table("Force [pN]\tExtension [nm]\n1\t2\n3\t4\n5\t6\n7\t8\n")
    assert t.dialect.delimiter_name == "tab"
    assert [(c.name, c.unit) for c in t.columns] == [("Force", "pN"), ("Extension", "nm")]


def test_no_header_gets_numbered_columns():
    t = read_table("1 2 3\n4 5 6\n7 8 9\n10 11 12\n")
    assert [c.name for c in t.columns] == ["column 1", "column 2", "column 3"]
    assert all(not c.named_by_file for c in t.columns)


def test_names_come_from_the_last_comment_line():
    t = read_table("# run 3\n# time(1) load(2)\n0 1\n1 2\n2 3\n3 4\n")
    assert t.dialect.header_from_comment
    assert [c.name for c in t.columns] == ["time", "load"]


def test_pipe_separated_names_in_a_comment():
    t = read_table("#time(1)  | x(2) | Force per atom (3)\n0 1 2\n1 2 3\n2 3 4\n3 4 5\n")
    assert [c.name for c in t.columns] == ["time", "x", "Force per atom"]


def test_percent_comment_prefix():
    t = read_table("% t (s) | V (mV)\n0 1\n1 2\n2 3\n3 4\n")
    assert [(c.name, c.unit) for c in t.columns] == [("t", "s"), ("V", "mV")]


def test_ragged_rows_are_skipped_and_reported():
    t = read_table("a;b\n1;2\n3\n5;6\n7;8\n")
    assert t.data.shape == (3, 2)
    assert any("skipped" in w for w in t.warnings)


def test_binary_input_is_rejected():
    with pytest.raises(ValueError):
        read_table(b"\x00\x01\x02binary\x00")


def test_duplicate_names_are_disambiguated():
    t = read_table("a,a,a\n1,2,3\n4,5,6\n7,8,9\n10,11,12\n")
    assert [c.name for c in t.columns] == ["a", "a (2)", "a (3)"]


# -------------------------------------------------------------- column names
@pytest.mark.parametrize(
    "raw,name,unit",
    [
        ("strain(5)", "strain", None),
        ("Force per atom (3)", "Force per atom", None),
        ("f [pN]", "f", "pN"),
        ("-pxx", "-pxx", None),          # the sign belongs to the name
        ("U, pN A", "U", "pN A"),
        ("time(1)", "time", None),
    ],
)
def test_column_name_parsing(raw, name, unit):
    c = parse_column_name(raw, 0)
    assert (c.name, c.unit) == (name, unit)


# ------------------------------------------------------------------- columns
def test_noise_estimate_recovers_a_known_sigma():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 1, 4000)
    y = 3 * x + rng.normal(0, 0.2, x.size)
    assert estimate_noise(y) == pytest.approx(0.2, rel=0.08)


def test_noise_estimate_is_not_inflated_by_a_real_jump():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 1, 4000)
    y = np.where(x < 0.5, 0.0, 50.0) + rng.normal(0, 0.2, x.size)
    assert estimate_noise(y) == pytest.approx(0.2, rel=0.1)


def test_monotone_evenly_spaced_column_is_chosen_for_x():
    rng = np.random.default_rng(2)
    n = 500
    text = ["time,signal,thermostat"]
    for i in range(n):
        t = i * 0.1
        text.append(f"{t},{2 * t + rng.normal(0, 0.01)},{300 + rng.normal(0, 3)}")
    t = read_table("\n".join(text) + "\n")
    r = analyse_columns(t)
    assert t.columns[r.recommended_x].name == "time"
    assert t.columns[r.recommended_y].name == "signal"


def test_a_column_that_only_fluctuates_is_not_recommended():
    rng = np.random.default_rng(3)
    n = 400
    rows = ["x,trend,noise"]
    for i in range(n):
        rows.append(f"{i},{0.01 * i + rng.normal(0, 0.05)},{rng.normal(0, 5)}")
    t = read_table("\n".join(rows) + "\n")
    r = analyse_columns(t)
    assert t.columns[r.recommended_y].name == "trend"
    noise_col = next(s for s in r.stats if s.name == "noise")
    assert noise_col.trend_strength < 3.0


def test_exactly_proportional_columns_are_flagged_as_duplicates():
    """A quantity written twice in different units carries no new information."""
    rng = np.random.default_rng(7)
    rows = ["x,a,b"]
    for i in range(200):
        a = float(np.sin(i / 20) + rng.normal(0, 1e-4))
        rows.append(f"{i},{a},{a / 10000}")
    t = read_table("\n".join(rows) + "\n")
    r = analyse_columns(t)
    names = {s.name: s.duplicate_of for s in r.stats}
    assert names["b"] == "a"
    assert names["a"] is None


def test_a_response_proportional_to_the_axis_is_not_a_duplicate():
    """But a linear response is the most ordinary result there is."""
    rows = ["time,signal"] + [f"{i * 0.1},{i * 0.2}" for i in range(200)]
    t = read_table("\n".join(rows) + "\n")
    r = analyse_columns(t)
    names = {s.name: s.duplicate_of for s in r.stats}
    assert names["signal"] is None


def test_bare_T_is_temperature_when_it_does_not_march_forward():
    rng = np.random.default_rng(4)
    rows = ["time,T"] + [f"{i * 0.5},{300 + rng.normal(0, 2)}" for i in range(300)]
    t = read_table("\n".join(rows) + "\n")
    r = analyse_columns(t)
    kinds = {s.name: s.kind for s in r.stats}
    assert kinds["time"] == "time"
    assert kinds["T"] == "temperature"
