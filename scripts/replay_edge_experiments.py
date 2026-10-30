"""Prepare and replay complete experiment protocols without overwriting x86 results."""
from __future__ import annotations

import sys


if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        print("Usage: replay_edge_experiments.py {prepare,worker,run,aggregate,finalize,backup} [options]")
        raise SystemExit(0)
    command = sys.argv.pop(1)
    if command == "worker":
        from edge_replay.worker import main
    elif command == "prepare":
        from edge_replay.prepare import main
    elif command == "run":
        from edge_replay.runner import main
    elif command == "aggregate":
        from edge_replay.aggregate import main
    elif command == "finalize":
        from edge_replay.finalize import main
    elif command == "backup":
        from edge_replay.backup import main
    else:
        raise SystemExit("Unknown replay command")
    main()
