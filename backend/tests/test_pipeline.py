"""Atomic writes, validation, statistics and pipeline resume, on a small
synthetic dataset written to a temporary directory."""

import json
import os
import shutil
from pathlib import Path

import numpy as np
import pytest

from tini import statistics as st
from tini.config import Settings
from tini.io_atomic import atomic_write_json, read_json
from tini.pipeline import STAGES, Pipeline
from tini.validation import validate_trace

HEADER = "# synthetic\n#x in the header below correponds column number 1\n#time(1)  | x(2) | Force per atom (3)\n"


def make_dataset(root: Path, n_files: int = 6, n_rows: int = 300, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    root.mkdir(parents=True)
    t = np.arange(n_rows) * 0.64
    for i in range(n_files):
        x = 60.2 + 0.00235 * t
        t_peak = 100 + 20 * rng.standard_normal()
        f = np.where(t < t_peak, 0.09 * t, 8.0) + 3.0 * rng.standard_normal(n_rows)
        with open(root / f"property.T300.{i+1:05d}.dat", "w") as fh:
            fh.write(HEADER)
            for a, b, c in zip(t, x, f):
                fh.write(f"{a:.6f} {b:+.6e} {c:+.6e}\n")


@pytest.fixture
def project(tmp_path):
    data = tmp_path / "data"
    make_dataset(data)
    s = Settings(data_dir=str(data), artifacts_dir=str(tmp_path / "art"), force_rate_window_ps=50.0,
                 force_rate_windows_ps=(25.0, 50.0), batch_size=2)
    return s


# ------------------------------------------------------------- atomic io
def test_atomic_write_keeps_previous(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    assert read_json(p)["v"] == 2
    assert read_json(p.with_name("a.json.prev"))["v"] == 1
    assert not list(tmp_path.glob("*.tmp"))


def test_read_json_falls_back_to_prev_on_corruption(tmp_path):
    p = tmp_path / "a.json"
    atomic_write_json(p, {"v": 1})
    atomic_write_json(p, {"v": 2})
    p.write_text("{corrupt")
    assert read_json(p)["v"] == 1


# ------------------------------------------------------------- validation
def test_validate_trace_flags_bad_input():
    s = Settings()
    t = np.arange(50) * 0.64
    good = np.column_stack([t, 60 + 0.002 * t, 0.1 * t])
    assert validate_trace("1", "a", good, s).ok
    bad = good.copy(); bad[10, 0] = bad[9, 0]  # non-monotonic time
    r = validate_trace("1", "a", bad, s)
    assert not r.ok and any(i["code"] == "time_not_monotonic" for i in r.issues)
    nan = good.copy(); nan[3, 2] = np.nan
    assert not validate_trace("1", "a", nan, s).ok
    assert not validate_trace("1", "a", good[:, :2], s).ok


# ------------------------------------------------------------- statistics
def test_force_rate_recovers_known_slope():
    t = np.arange(400) * 0.64
    f = 0.09 * t + 0.5
    fit = st.force_rate(t, f, 100.0)
    assert fit["slope"] == pytest.approx(0.09, abs=1e-9)
    assert fit["r2"] == pytest.approx(1.0)


def test_moving_average_preserves_constant_and_length():
    y = np.full(50, 3.0)
    out = st.moving_average(y, 15)
    assert out.shape == y.shape and np.allclose(out, 3.0)


def test_peak_detection_finds_the_transition():
    t = np.arange(600) * 0.64
    f = np.where(t < 150, 0.1 * t, 5.0)
    r = st.detect_peak(t, f, 15)
    assert abs(r["t_peak_ps"] - 150) < 8


# ------------------------------------------------------------- pipeline
def test_pipeline_runs_and_resumes(project):
    p = Pipeline(project)
    snap = p.run()
    assert snap["status"] == "done", snap
    art = project.artifacts_path
    for name in ("inspection.json", "validation.json", "traces.npz", "average_curves.csv", "force_rate.json",
                 "events.json", "potential.json", "potential_curves.csv", "checkpoints.json", "progress.json"):
        assert (art / name).exists(), name
    assert len(list((art / "charts").glob("*.png"))) == 5
    z = np.load(art / "traces.npz")
    assert z["traces"].shape == (6, 300, 3)
    rate = read_json(art / "force_rate.json")
    assert abs(rate["force_rate_pN_per_ps"] - 0.09) < 0.02

    # second run: everything is a checkpoint hit
    p2 = Pipeline(project)
    snap2 = p2.run()
    assert snap2["status"] == "done"
    assert all(v == "skipped (checkpoint)" for v in snap2["stage_status"].values())


def test_pipeline_resumes_preprocess_per_file(project):
    p = Pipeline(project)
    p.run(stages=["inspect", "validate"])
    # simulate a crash after 3 files: write per-file npy for 3 and the done list
    art = project.artifacts_path
    fp = p.fingerprint("preprocess")
    (art / "traces").mkdir()
    done = {"fingerprint": fp, "files": {}}
    files = p.files()
    for f in files[:3]:
        from tini.dataset import load_trace
        np.save(art / "traces" / f"{f.trace_id}.npy", load_trace(f.path))
        done["files"][f.trace_id] = {"name": f.name, "rows": 300, "time": 0}
    atomic_write_json(art / "completed_files.json", done)
    mtimes = {f.trace_id: (art / "traces" / f"{f.trace_id}.npy").stat().st_mtime_ns for f in files[:3]}

    p2 = Pipeline(project)
    snap = p2.run(stages=["preprocess"])
    assert snap["status"] == "done"
    # the three pre-existing files were not rewritten
    for f in files[:3]:
        assert (art / "traces" / f"{f.trace_id}.npy").stat().st_mtime_ns == mtimes[f.trace_id]
    assert np.load(art / "traces.npz")["traces"].shape[0] == 6


def test_changed_setting_invalidates_only_dependent_stages(project):
    p = Pipeline(project)
    p.run()
    project.force_rate_window_ps = 25.0
    p2 = Pipeline(project)
    snap = p2.run()
    ss = snap["stage_status"]
    assert ss["inspect"] == "skipped (checkpoint)" and ss["preprocess"] == "skipped (checkpoint)"
    assert ss["statistics"] == "done" and ss["potential"] == "done"


def test_cancel_keeps_checkpoints(project):
    p = Pipeline(project)
    p.cancel()  # pre-set: cancels at the first check
    snap = p.run()
    assert snap["status"] == "cancelled"
    assert (project.artifacts_path / "progress.json").exists()
