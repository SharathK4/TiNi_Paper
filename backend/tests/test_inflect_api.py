"""Version 4 — the upload, session and analysis endpoints."""

from __future__ import annotations

import io

import numpy as np
import pytest

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402

from tini.api import app  # noqa: E402

client = TestClient(app)


def make_csv(step_at: float = 6.0, n: int = 500, seed: int = 0) -> bytes:
    x = np.linspace(0, 10, n)
    y = np.where(x < step_at, -2 * x, -2 * step_at + 9.0)
    y = y + np.random.default_rng(seed).normal(0, 0.05, n)
    rows = ["time,signal"] + [f"{a},{b}" for a, b in zip(x, y)]
    return ("\n".join(rows) + "\n").encode()


def upload(n_files: int = 1, **kw) -> str:
    files = [("files", (f"run{i}.csv", io.BytesIO(make_csv(seed=i, **kw)), "text/csv"))
             for i in range(n_files)]
    r = client.post("/api/inflect/sessions", files=files, data={"title": "test"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_spec_is_served():
    r = client.get("/api/inflect/spec")
    assert r.status_code == 200
    assert "defaults" in r.json()


def test_upload_creates_a_session_that_can_be_listed():
    sid = upload()
    listed = client.get("/api/inflect/sessions").json()["sessions"]
    assert any(s["id"] == sid for s in listed)


def test_a_file_that_is_not_a_table_is_rejected_with_a_reason():
    files = [("files", ("junk.bin", io.BytesIO(b"\x00\x01\x02\x03"), "application/octet-stream"))]
    r = client.post("/api/inflect/sessions", files=files, data={"title": "junk"})
    assert r.status_code == 400


def test_one_bad_file_does_not_sink_a_good_upload():
    files = [
        ("files", ("good.csv", io.BytesIO(make_csv()), "text/csv")),
        ("files", ("bad.bin", io.BytesIO(b"\x00\x00\x00"), "application/octet-stream")),
    ]
    r = client.post("/api/inflect/sessions", files=files)
    assert r.status_code == 200
    body = r.json()
    assert body["accepted"] == 1
    assert body["problems"] and body["problems"][0]["name"] == "bad.bin"


def test_session_reports_the_columns_it_inferred():
    sid = upload()
    body = client.get(f"/api/inflect/sessions/{sid}").json()
    cols = body["columns"]
    names = [c["name"] for c in body["first_file"]["columns"]]
    assert names == ["time", "signal"]
    assert names[cols["recommended_x"]] == "time"
    assert names[cols["recommended_y"]] == "signal"


def test_curve_endpoint_returns_a_drawable_narrated_curve():
    sid = upload()
    body = client.get(f"/api/inflect/sessions/{sid}/curve").json()
    assert len(body["curve"]["x"]) == len(body["curve"]["y"]) > 10
    assert body["fits"]
    assert body["reading"]["beats"]
    trans = [b for b in body["segmentation"]["boundaries"] if b["kind"] == "transition"]
    assert len(trans) == 1
    assert trans[0]["x"] == pytest.approx(6.0, abs=0.3)


def test_asking_for_a_regime_count_changes_the_answer():
    sid = upload()
    a = client.get(f"/api/inflect/sessions/{sid}/curve").json()
    b = client.get(f"/api/inflect/sessions/{sid}/curve", params={"n_segments": 4}).json()
    assert b["segmentation"]["n_segments"] > a["segmentation"]["n_segments"]
    assert b["segmentation"]["chosen_by"] == "requested"


def test_config_is_remembered_between_calls():
    sid = upload()
    client.post(f"/api/inflect/sessions/{sid}/config", json={"x_scale": 100.0, "y_unit": "GPa"})
    body = client.get(f"/api/inflect/sessions/{sid}/curve").json()
    assert body["y_unit"] == "GPa"
    assert body["domain"][1] == pytest.approx(1000.0, rel=1e-3)


def test_an_unknown_session_is_a_404():
    assert client.get("/api/inflect/sessions/nope").status_code == 404
    assert client.get("/api/inflect/sessions/nope/curve").status_code == 404


def test_a_session_id_cannot_escape_the_store():
    assert client.get("/api/inflect/sessions/..%2f..%2fetc/curve").status_code == 404


def test_ensemble_runs_and_reports_statistics():
    sid = upload(n_files=6)
    r = client.post(f"/api/inflect/sessions/{sid}/ensemble")
    assert r.status_code == 200
    for _ in range(200):
        p = client.get("/api/inflect/ensemble/progress").json()
        if not p["running"]:
            break
        import time
        time.sleep(0.05)
    assert p["status"] == "done", p
    rep = client.get(f"/api/inflect/sessions/{sid}/ensemble").json()
    assert rep["n_ok"] == 6
    assert rep["stats"]["transition_x"]["median"] == pytest.approx(6.0, abs=0.3)


def test_ensemble_before_it_is_run_is_a_404():
    sid = upload()
    assert client.get(f"/api/inflect/sessions/{sid}/ensemble").status_code == 404
