"""Live progress and USD budget for an agent results directory (read-only)."""

import csv
import json
from pathlib import Path
import sys

if len(sys.argv) < 2:
    raise SystemExit("Usage: monitor_agents.py eval/results/RUN_NAME [TARGET]")
directory = Path(sys.argv[1])
if not directory.is_dir():
    raise SystemExit(f"Run directory does not exist: {directory}")
target = int(sys.argv[2]) if len(sys.argv) > 2 else 500
for path in sorted(directory.glob("pred_*.csv")):
    rows = list(csv.DictReader(path.open()))
    done = sum(row.get("status") == "success" for row in rows)
    name = path.name.removeprefix("pred_sz512_").removesuffix("_fps4_max32.csv")
    print(f"{name:32} {done:4}/{target} succeeded, {len(rows) - done} failed")
budget = directory / "run_usage" / "budget.jsonl"
if budget.exists():
    reserved, settled = {}, {}
    for line in budget.open():
        if line.endswith("\n"):
            event = json.loads(line)
            (reserved if event["type"] == "reserve" else settled)[event["id"]] = event["usd"]
    open_usd = sum(usd for key, usd in reserved.items() if key not in settled)
    print(f"Budget: ${sum(settled.values()):.2f} settled + ${open_usd:.2f} in flight "
          f"= ${sum(settled.values()) + open_usd:.2f} committed")
else:
    print("Budget: no requests yet")
