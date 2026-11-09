"""Append-only request accounting and offline standard-rate cost estimates."""

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import sys
import uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from lib.run_paths import usage_path

PRICES = Path(__file__).resolve().parent / "prices.json"


class Ledger:
    def __init__(self, output, provider, model, command, endpoint):
        self.path = usage_path(output)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.context = {
            "run_id": str(uuid.uuid4()), "provider": provider, "model": model,
            "stage": command, "endpoint": endpoint,
        }

    def record(self, response, elapsed, status, attempt, example_id=None):
        usage = response.usage if response is not None else None
        event = dict(
            self.context, event_id=str(uuid.uuid4()),
            timestamp=datetime.now(timezone.utc).isoformat(),
            response_id=getattr(response, "id", None),
            resolved_model=getattr(response, "model", None),
            elapsed_seconds=elapsed, status=status, attempt=attempt,
            example_id=example_id,
            usage=usage.model_dump(mode="json") if usage is not None else None,
        )
        # Separate opens plus flock protect concurrent threads/processes.
        with self.path.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            stream.write(json.dumps(event) + "\n")
            stream.flush()
            os.fsync(stream.fileno())


def token_totals(event):
    usage = event["usage"]
    if usage is None:
        return None
    if event["provider"] == "openai":
        input_tokens = usage.get("input_tokens", usage.get("prompt_tokens"))
        output_tokens = usage.get("output_tokens", usage.get("completion_tokens"))
    else:
        input_tokens = usage.get("input_tokens")
        if input_tokens is not None:
            # Anthropic reports cache reads/writes outside input_tokens.
            input_tokens += (usage.get("cache_read_input_tokens") or 0)
            input_tokens += (usage.get("cache_creation_input_tokens") or 0)
        output_tokens = usage.get("output_tokens")
    if input_tokens is None or output_tokens is None:
        return None
    for value in (input_tokens, output_tokens):
        if not isinstance(value, int) or value < 0:
            raise ValueError("Invalid token count in usage ledger")
    return input_tokens, output_tokens


def summarize(paths, prices):
    groups = defaultdict(lambda: {
        "requests": 0, "usage_known_requests": 0, "unknown_usage_requests": 0,
        "input_tokens": 0, "output_tokens": 0, "potentially_partial_usage_requests": 0,
    })
    seen = set()
    for path in paths:
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                try:
                    event = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid ledger JSON: {path}:{number}") from exc
                if event["event_id"] in seen:
                    continue
                seen.add(event["event_id"])
                key = (event["provider"], event["model"], event["stage"])
                row = groups[key]
                row["requests"] += 1
                totals = token_totals(event)
                status = event.get("status", "")
                if totals is not None and (
                    status in ("streaming", "timeout") or status.startswith("gateway_error_")
                ):
                    row["potentially_partial_usage_requests"] += 1
                if totals is None:
                    row["unknown_usage_requests"] += 1
                else:
                    row["usage_known_requests"] += 1
                    row["input_tokens"] += totals[0]
                    row["output_tokens"] += totals[1]
    rows = []
    for (provider, model, stage), totals in sorted(groups.items()):
        rates = prices["models"].get(f"{provider}/{model}")
        cost = None
        if rates:
            for rate in (rates["input"], rates["output"]):
                if not math.isfinite(rate) or rate < 0:
                    raise ValueError(f"Invalid price for {provider}/{model}")
            if totals["usage_known_requests"]:
                cost = (
                    totals["input_tokens"] * rates["input"]
                    + totals["output_tokens"] * rates["output"]
                ) / 1_000_000
        rows.append(dict(
            provider=provider, model=model, stage=stage, **totals,
            standard_rate_estimate_usd=cost, price_available=rates is not None,
        ))
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Directory to search recursively for *.usage.jsonl")
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--output", type=Path, help="Optional JSON report")
    args = parser.parse_args(argv)
    paths = sorted(args.results.rglob("*.usage.jsonl"))
    if not paths:
        parser.error("No usage ledgers found; old prediction CSVs cannot reconstruct all attempts")
    prices = json.loads(args.prices.read_text())
    rows = summarize(paths, prices)
    report = {
        "price_as_of": prices["as_of"], "price_source": prices["source"],
        "basis": prices["basis"], "models": rows,
        "known_usage_standard_rate_subtotal_usd": sum(
            row["standard_rate_estimate_usd"] or 0 for row in rows
        ),
        "warning": (
            "Not an invoice or guaranteed bound. Unknown usage/unpriced models are excluded "
            "from the subtotal. Cache discounts/write premiums, long-context tiers, service "
            "tiers, taxes and contracts are not applied. Provider billing is authoritative."
        ),
    }
    text = json.dumps(report, indent=2) + "\n"
    print(text, end="")
    if args.output:
        args.output.write_text(text)


if __name__ == "__main__":
    main()
