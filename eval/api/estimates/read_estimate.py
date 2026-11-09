"""Read a combined bootstrap cost estimate for selected provider/model pairs."""

import argparse
import json
from pathlib import Path


def total_cost(report, provider, models, kind):
    available = {
        (item.get("provider"), item.get("model")): item
        for item in report.get("models", [])
    }
    values = []
    for model in models:
        item = available.get((provider, model))
        if item is None:
            raise SystemExit(f"Estimate is missing {provider}/{model}")
        estimate = item.get("usd")
        if not estimate or estimate.get("estimate") is None:
            raise SystemExit(f"Estimate for {provider}/{model} is incomplete")
        values.append(
            estimate["estimate"] if kind == "point" else estimate["interval"][1]
        )
    return sum(values)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("estimate", type=Path)
    parser.add_argument("provider", choices=["openai", "anthropic"])
    parser.add_argument("models", nargs="+")
    parser.add_argument("--kind", choices=["point", "upper"], default="point")
    args = parser.parse_args(argv)
    report = json.loads(args.estimate.read_text())
    print(total_cost(report, args.provider, args.models, args.kind))


if __name__ == "__main__":
    main()
