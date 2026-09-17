"""CLI: run or inspect the pipeline without the API server.

    python -m tini run              # resume / run all stages
    python -m tini run --force      # ignore checkpoints and redo everything
    python -m tini run --stages inspect validate
    python -m tini status           # print progress.json + checkpoints
    python -m tini serve            # start the API on :8000
    python -m tini inverse          # Version 2: run the inverse problem (new run)
    python -m tini forward          # Version 3: forward Smoluchowski run (latest V2 parameters)
    python -m tini inverse --run-id 20260917-1030-ab12   # resume that run
"""

from __future__ import annotations

import argparse
import json
import sys

from .config import load_settings
from .pipeline import STAGES, Pipeline


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="tini")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--force", action="store_true")
    r.add_argument("--stages", nargs="*", choices=STAGES)
    r.add_argument("--max-files", type=int, default=None, help="process only the first N files")
    sub.add_parser("status")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    fwd = sub.add_parser("forward")
    fwd.add_argument("--run-id", default=None)
    fwd.add_argument("--source", default="inverse:latest", help="paper:pdf_mle | paper:mean_lsq | inverse:latest | inverse:run | custom")
    fwd.add_argument("--boundary", default="beyond_barrier")
    fwd.add_argument("--cells", type=int, default=250)
    fwd.add_argument("--steps", type=int, default=20000)
    inv = sub.add_parser("inverse")
    inv.add_argument("--run-id", default=None)
    inv.add_argument("--source", default="rate_x_time")
    inv.add_argument("--bootstrap", type=int, default=200)
    inv.add_argument("--replicates", type=int, default=10)
    inv.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    settings = load_settings()
    if args.cmd == "run":
        if args.max_files is not None:
            settings.max_files = args.max_files
        p = Pipeline(settings)
        snap = p.run(force=args.force, stages=args.stages or None)
        print(json.dumps({k: snap[k] for k in ("status", "percent", "elapsed_s", "message", "stage_status", "errors")}, indent=2))
        return 0 if snap["status"] == "done" else 1
    if args.cmd == "status":
        p = Pipeline(settings)
        print(json.dumps({"progress": p.progress.snapshot(), "checkpoints": p.registry.all()}, indent=2, default=str))
        return 0
    if args.cmd == "forward":
        from .forward.runs import ForwardConfig, ForwardRun
        cfg = ForwardConfig(parameter_source=args.source, boundary=args.boundary, n_cells=args.cells, n_steps=args.steps)
        run = ForwardRun(settings, cfg, run_id=args.run_id)
        snap = run.run()
        print(json.dumps({"run_id": run.run_id, "status": snap["status"], "elapsed_s": snap["elapsed_s"], "message": snap["message"], "stage_status": snap["stage_status"]}, indent=2))
        return 0 if snap["status"] == "done" else 1
    if args.cmd == "inverse":
        from .inverse.runs import InverseConfig, InverseRun
        cfg = InverseConfig(source=args.source, bootstrap_n=args.bootstrap, synthetic_replicates=args.replicates, seed=args.seed)
        run = InverseRun(settings, cfg, run_id=args.run_id)
        snap = run.run()
        print(json.dumps({"run_id": run.run_id, "status": snap["status"], "elapsed_s": snap["elapsed_s"], "message": snap["message"], "stage_status": snap["stage_status"]}, indent=2))
        return 0 if snap["status"] == "done" else 1
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("tini.api:app", host="127.0.0.1", port=args.port, reload=False)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
