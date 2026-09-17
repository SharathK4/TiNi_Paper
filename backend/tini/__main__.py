"""CLI: run or inspect the pipeline without the API server.

    python -m tini run              # resume / run all stages
    python -m tini run --force      # ignore checkpoints and redo everything
    python -m tini run --stages inspect validate
    python -m tini status           # print progress.json + checkpoints
    python -m tini serve            # start the API on :8000
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
    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("tini.api:app", host="127.0.0.1", port=args.port, reload=False)
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
