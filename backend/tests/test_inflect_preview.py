"""Version 4 — choosing which columns to plot.

The tool's own choice is a default, not a monopoly: the reader must be able
to see every column and pick any pair.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tini.api import app  # noqa: E402

client = TestClient(app)


def make_csv(n: int = 400, seed: int = 0) -> bytes:
    rng = np.random.default_rng(seed)
    x = np.linspace(0, 10, n)
    stress = np.where(x < 6, -2 * x, -3.0) + rng.normal(0, 0.05, n)
    energy = 0.5 * x ** 2 + rng.normal(0, 0.02, n)
    thermostat = 300 + rng.normal(0, 3, n)
    rows = ["time,stress,energy,T"]
    for i in range(n):
        rows.append(f"{x[i]},{stress[i]},{energy[i]},{thermostat[i]}")
    return ("\n".join(rows) + "\n").encode()


def session(n: int = 400) -> str:
    files = [("files", ("run.csv", io.BytesIO(make_csv(n)), "text/csv"))]
    r = client.post("/api/inflect/sessions", files=files, data={"title": "preview"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_preview_offers_every_column():
    sid = session()
    p = client.get(f"/api/inflect/sessions/{sid}/preview").json()
    assert [c["name"] for c in p["columns"]] == ["time", "stress", "energy", "T"]
    assert p["n_rows"] == 400


def test_preview_is_thinned_but_paired():
    """Every column is sampled at the same rows, or no pair could be drawn."""
    sid = session(n=2000)
    p = client.get(f"/api/inflect/sessions/{sid}/preview", params={"points": 120}).json()
    assert p["n_preview"] == 120
    assert all(len(c["values"]) == 120 for c in p["columns"])
    # time is the first column and runs 0..10; its thinned values must still
    # span that range in order, so a pair plotted against it is honest.
    t = p["columns"][0]["values"]
    assert t[0] == pytest.approx(0, abs=0.05)
    assert t[-1] == pytest.approx(10, abs=0.05)
    assert t == sorted(t)


def test_preview_carries_the_reasons_for_each_role():
    sid = session()
    p = client.get(f"/api/inflect/sessions/{sid}/preview").json()
    stress = next(c for c in p["columns"] if c["name"] == "stress")
    assert stress["y_reasons"]
    time = next(c for c in p["columns"] if c["name"] == "time")
    assert any("monotone" in r or "spaced" in r for r in time["x_reasons"])


def test_preview_marks_the_suggestion_without_imposing_it():
    sid = session()
    p = client.get(f"/api/inflect/sessions/{sid}/preview").json()
    assert p["columns"][p["recommended_x"]]["name"] == "time"
    assert p["columns"][p["recommended_y"]]["name"] == "stress"
    # nothing has been chosen yet, so the stored choice is empty
    assert p["chosen_x"] is None and p["chosen_y"] is None


def test_a_chosen_pair_is_remembered_and_served_back():
    sid = session()
    client.post(f"/api/inflect/sessions/{sid}/config", json={"x_column": 0, "y_column": 2})
    p = client.get(f"/api/inflect/sessions/{sid}/preview").json()
    assert (p["chosen_x"], p["chosen_y"]) == (0, 2)
    assert (p["recommended_x"], p["recommended_y"]) == (0, 1)


def test_the_curve_follows_the_chosen_pair_not_the_suggestion():
    sid = session()
    body = client.get(f"/api/inflect/sessions/{sid}/curve",
                      params={"x_column": 0, "y_column": 2}).json()
    assert body["y_label"] == "energy"
    assert body["x_label"] == "time"


def test_any_pair_can_be_asked_for_including_a_reversed_one():
    sid = session()
    body = client.get(f"/api/inflect/sessions/{sid}/curve",
                      params={"x_column": 1, "y_column": 0}).json()
    assert body["x_label"] == "stress" and body["y_label"] == "time"


def test_preview_values_are_json_clean():
    """A gap in a column is null, never NaN, which no browser would parse."""
    rows = ["a,b"] + [f"{i}," + ("" if i % 7 == 0 else str(i * 2)) for i in range(60)]
    files = [("files", ("gaps.csv", io.BytesIO(("\n".join(rows) + "\n").encode()), "text/csv"))]
    sid = client.post("/api/inflect/sessions", files=files).json()["id"]
    text = client.get(f"/api/inflect/sessions/{sid}/preview").text
    assert "NaN" not in text
    p = client.get(f"/api/inflect/sessions/{sid}/preview").json()
    assert any(v is None for v in p["columns"][1]["values"])


def test_preview_of_an_unknown_session_is_a_404():
    assert client.get("/api/inflect/sessions/nope/preview").status_code == 404
