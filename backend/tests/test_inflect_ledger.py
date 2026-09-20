"""Version 4 — the ledger, the record, and what is measured versus concluded."""

from __future__ import annotations

import io
import json

import numpy as np
import pytest

from tini.inflect.findings import build_ledger, ledger_summary
from tini.inflect.measure import measure
from tini.inflect.record import compare, code_hash
from tini.inflect.segment import segment_curve
from tini.inflect.transitions import read_curve
from tini.inflect.uncertainty import bootstrap_curve, stability_sweep


def stepped(seed=0, n=800):
    x = np.linspace(0, 10, n)
    y = np.where(x < 6, -2 * x, -3.0) + np.random.default_rng(seed).normal(0, 0.05, n)
    return x, y


@pytest.fixture(scope="module")
def reading():
    x, y = stepped()
    seg = segment_curve(x, y)
    unc = bootstrap_curve(x, y, seg, n_resamples=40, seed=1)
    st = stability_sweep(x, y, seg, x_unit="s")
    led = build_ledger(x, y, seg, "t", "v", "s", "V", unc.keyed, st.keyed, {"penalty_scale": 1.0})
    return x, y, seg, unc, st, led


# ------------------------------------------------------------------ measure
def test_measure_keys_every_feature():
    x, y = stepped()
    m = measure(x, y, segment_curve(x, y))
    assert "noise" in m and "transition.0.position" in m and "regime.0.slope" in m
    assert "extremum.value" in m


def test_measure_matches_features_to_a_reference_by_position():
    x, y = stepped()
    seg = segment_curve(x, y)
    # a replicate that found the same transition, slightly moved
    y2 = np.where(x < 6.05, -2 * x, -3.0) + np.random.default_rng(3).normal(0, 0.05, x.size)
    m = measure(x, y2, segment_curve(x, y2), reference=seg)
    assert abs(m["transition.0.position"] - 6.0) < 0.3


def test_a_far_away_transition_is_not_mistaken_for_the_reference_one():
    x, y = stepped()
    seg = segment_curve(x, y)
    y2 = np.where(x < 2.0, 0.0, 5.0) + np.random.default_rng(4).normal(0, 0.05, x.size)
    m = measure(x, y2, segment_curve(x, y2), reference=seg)
    assert "transition.0.position" not in m


# ------------------------------------------------------------------- ledger
def test_every_finding_declares_its_tier_and_method(reading):
    *_, led = reading
    assert led
    for f in led:
        assert f.tier in ("measured", "read")
        assert f.method
        if f.tier == "read":
            assert f.rule, f.id
        assert f.anchor


def test_measured_findings_carry_numbers_and_read_findings_carry_words(reading):
    *_, led = reading
    for f in led:
        if f.tier == "measured":
            assert f.value is not None and np.isfinite(f.value), f.id
        else:
            assert f.text, f.id


def test_the_noise_carries_no_bootstrap_interval(reading):
    """The bootstrap takes the noise as input; an interval on it would be circular."""
    *_, led = reading
    noise = next(f for f in led if f.id == "noise")
    assert noise.interval is None
    assert "No bootstrap interval" in noise.method


def test_measured_findings_get_intervals_and_sensitivity(reading):
    *_, led = reading
    for f in led:
        if f.tier == "measured" and f.id != "noise":
            assert f.interval is not None and f.interval["ci"], f.id
            lo, hi = f.interval["ci"]
            # Either the estimate sits inside its replicates' interval, or the
            # interval says so — a shifted estimator is reported, never hidden.
            assert (lo <= f.value <= hi) or f.interval["biased"], f.id
            assert f.sensitivity is not None, f.id
            assert f.sensitivity["of"] == 16


def test_a_shifted_estimator_is_flagged_not_fudged(reading):
    """On a clean step the width is at its floor and replicates can only widen it."""
    *_, led = reading
    width = next(f for f in led if f.id == "transition.0.width")
    lo, hi = width.interval["ci"]
    if not (lo <= width.value <= hi):
        assert width.interval["biased"]
        assert width.interval["shift"] > 0


def test_a_clear_transition_holds_on_every_finding(reading):
    *_, led = reading
    pos = next(f for f in led if f.id == "transition.0.position")
    assert pos.sensitivity["holds"]
    assert pos.sensitivity["present_in"] >= 12


def test_the_extremum_is_recorded_but_not_called_the_transition(reading):
    *_, led = reading
    ext = next(f for f in led if f.id == "extremum.value")
    assert "not used as the transition value" in ext.method


def test_ledger_summary_counts(reading):
    *_, led = reading
    s = ledger_summary(led)
    assert s["n"] == len(led)
    assert s["tiers"]["measured"] + s["tiers"]["read"] == s["n"]
    assert s["with_interval"] <= s["tiers"]["measured"]


def test_ledger_serialises(reading):
    *_, led = reading
    json.dumps([f.to_dict() for f in led])


# -------------------------------------------------------------------- beats
def test_beats_are_tagged_and_point_at_findings(reading):
    x, y, seg, *_ , led = reading
    r = read_curve(x, y, seg, "t", "v", "s", "V")
    ids = {f.id for f in led}
    for b in r.beats:
        assert b.tier in ("measured", "read")
        for fid in b.findings:
            assert fid in ids, f"{b.id} points at unknown finding {fid}"
    kinds = {b.kind: b.tier for b in r.beats}
    assert kinds["overview"] == "measured"
    assert kinds["transition"] == "read"


# ------------------------------------------------------------------- record
def test_compare_finds_agreement_movement_and_absence():
    a = [{"id": "p", "value": 1.0}, {"id": "q", "value": 2.0}, {"id": "s", "text": "linear"},
         {"id": "gone", "value": 3.0}]
    b = [{"id": "p", "value": 1.0 + 1e-12}, {"id": "q", "value": 2.5}, {"id": "s", "text": "quadratic"},
         {"id": "new", "value": 9.0}]
    c = compare(a, b)
    assert not c["reproduced"]
    assert c["same"] == ["p"]
    assert {m["id"] for m in c["moved"]} == {"q", "s"}
    assert c["missing"] == ["gone"] and c["new"] == ["new"]


def test_compare_of_identical_ledgers_reproduces():
    a = [{"id": "p", "value": 1.234567}, {"id": "s", "text": "linear"}]
    assert compare(a, json.loads(json.dumps(a)))["reproduced"]


def test_code_hash_is_stable_and_short():
    assert code_hash() == code_hash()
    assert len(code_hash()) == 16


# ---------------------------------------------------------------------- api
pytest.importorskip("fastapi")
from fastapi.testclient import TestClient  # noqa: E402
from tini.api import app  # noqa: E402

client = TestClient(app)


def _session() -> str:
    x, y = stepped(seed=7, n=500)
    rows = ["t,v"] + [f"{a},{b}" for a, b in zip(x, y)]
    files = [("files", ("run.csv", io.BytesIO(("\n".join(rows) + "\n").encode()), "text/csv"))]
    r = client.post("/api/inflect/sessions", files=files, data={"title": "ledger"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


def test_ledger_endpoint_returns_findings_with_tiers():
    sid = _session()
    led = client.get(f"/api/inflect/sessions/{sid}/ledger", params={"n_resamples": 24}).json()
    assert led["summary"]["n"] == len(led["findings"]) > 8
    assert {f["tier"] for f in led["findings"]} == {"measured", "read"}
    assert led["bootstrap"]["n_used"] > 0


def test_notes_are_stored_apart_from_findings_and_travel_in_the_record():
    sid = _session()
    r = client.post(f"/api/inflect/sessions/{sid}/notes",
                    json={"file": "run.csv", "finding_id": "transition.0.position",
                          "text": "This is where the sample yields."})
    assert r.status_code == 200
    note = r.json()
    led = client.get(f"/api/inflect/sessions/{sid}/ledger", params={"with_uncertainty": False}).json()
    assert led["notes"][0]["id"] == note["id"]
    assert all("yields" not in json.dumps(f) for f in led["findings"])
    rec = client.get(f"/api/inflect/sessions/{sid}/record").json()
    assert rec["interpretations"][0]["text"].startswith("This is where")
    assert "not claims made by the software" in rec["note"]
    assert client.delete(f"/api/inflect/sessions/{sid}/notes/{note['id']}").status_code == 200
    assert client.get(f"/api/inflect/sessions/{sid}/notes").json()["notes"] == []


def test_an_empty_note_is_refused():
    sid = _session()
    r = client.post(f"/api/inflect/sessions/{sid}/notes",
                    json={"file": "run.csv", "finding_id": "noise", "text": "   "})
    assert r.status_code == 400


def test_a_record_verifies_against_itself():
    sid = _session()
    rec = client.get(f"/api/inflect/sessions/{sid}/record").json()
    assert rec["data"]["sha256"] and rec["software"]["code_hash"] == code_hash()
    v = client.post(f"/api/inflect/sessions/{sid}/verify", json=rec).json()
    assert v["reproduced"], v
    assert v["n_same"] == v["n_recorded"] == len(rec["findings"])
    assert v["data_matches"] and v["code_matches"]


def test_a_tampered_record_is_caught():
    sid = _session()
    rec = client.get(f"/api/inflect/sessions/{sid}/record").json()
    target = next(f for f in rec["findings"] if f["id"] == "transition.0.position")
    target["value"] += 0.5
    rec["data"]["sha256"] = "0" * 64
    v = client.post(f"/api/inflect/sessions/{sid}/verify", json=rec).json()
    assert not v["reproduced"]
    assert [m["id"] for m in v["moved"]] == ["transition.0.position"]
    assert not v["data_matches"]


def test_record_downloads_as_an_attachment():
    sid = _session()
    r = client.get(f"/api/inflect/sessions/{sid}/record", params={"download": True})
    assert r.status_code == 200
    assert "attachment" in r.headers.get("content-disposition", "")
