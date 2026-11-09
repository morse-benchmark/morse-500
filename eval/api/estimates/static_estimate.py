"""Conservative price-table projection without an API call or benchmark pilot."""

import argparse
import json
import math
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimates.usage import PRICES


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["model", "agent"])
    parser.add_argument("provider", choices=["openai", "anthropic"])
    parser.add_argument("model")
    parser.add_argument("--examples", type=int, default=500)
    parser.add_argument("--frames", type=int, default=32)
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--input-tokens-per-frame", type=int, default=2000)
    parser.add_argument("--prompt-tokens", type=int, default=200)
    parser.add_argument("--prices", type=Path, default=PRICES)
    args = parser.parse_args(argv)
    for name in ("examples", "frames", "requests", "max_output_tokens",
                 "input_tokens_per_frame", "prompt_tokens"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    prices = json.loads(args.prices.read_text())
    rates = prices["models"].get(f"{args.provider}/{args.model}")
    if rates is None:
        parser.error(f"No price for {args.provider}/{args.model} in {args.prices}")
    for rate in (rates["input"], rates["output"]):
        if not math.isfinite(rate) or rate < 0:
            parser.error("Price file contains an invalid rate")
    calls = args.requests if args.mode == "agent" else 1
    input_tokens = (args.prompt_tokens + args.frames * args.input_tokens_per_frame) * calls
    output_tokens = args.max_output_tokens * calls
    estimate = args.examples * (
        input_tokens * rates["input"] + output_tokens * rates["output"]
    ) / 1_000_000
    print(json.dumps({
        "mode": args.mode, "provider": args.provider, "model": args.model,
        "examples": args.examples, "calls_per_example": calls,
        "estimated_input_tokens": input_tokens, "estimated_output_tokens": output_tokens,
        "worst_case_standard_rate_usd": estimate, "price_as_of": prices["as_of"],
        "warning": "Arithmetic ceiling using configured token assumptions; not a billing forecast.",
    }, indent=2))


if __name__ == "__main__":
    main()
