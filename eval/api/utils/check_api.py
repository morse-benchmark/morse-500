"""Check hosted API connectivity and credentials without inference requests."""

import argparse
import os
import socket
import sys

import anthropic
import openai


def check_provider(provider, models, timeout):
    variable = "OPENAI_API_KEY" if provider == "openai" else "ANTHROPIC_API_KEY"
    key = os.environ.get(variable, "").strip()
    if not key:
        print(f"FAIL {provider}: {variable} is not set/nonempty", file=sys.stderr)
        return False
    factory = openai.OpenAI if provider == "openai" else anthropic.Anthropic
    try:
        with factory(api_key=key, timeout=timeout, max_retries=0) as client:
            page = client.models.list()
            if not page.data:
                print(f"FAIL {provider}: model-list response was empty", file=sys.stderr)
                return False
            print(f"PASS {provider}: authenticated model-list request; no inference")
            for model in models:
                response = client.models.retrieve(model)
                print(f"PASS {provider}: model metadata accessible: {response.id}")
    except (openai.APIConnectionError, anthropic.APIConnectionError):
        print(
            f"FAIL {provider}: connection/TLS/DNS/timeout error; check compute-node networking",
            file=sys.stderr,
        )
        return False
    except (openai.APIStatusError, anthropic.APIStatusError) as exc:
        status = exc.status_code
        descriptions = {
            401: "key rejected (invalid, expired, or revoked)",
            403: "permission/policy denied",
            404: "model or endpoint not accessible",
            429: "rate limit/quota restriction; authentication not fully verified",
        }
        description = descriptions.get(status, "provider/server error")
        # Do not print raw responses, request headers, or credentials.
        print(f"FAIL {provider}: HTTP {status}: {description}", file=sys.stderr)
        return False
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("provider", choices=["openai", "anthropic", "both"])
    parser.add_argument(
        "--model", action="append", default=[],
        help="Optional model ID to verify via metadata GET; repeat for multiple IDs",
    )
    parser.add_argument("--timeout", type=float, default=20)
    args = parser.parse_args(argv)
    if not 0 < args.timeout < float("inf"):
        parser.error("--timeout must be finite and positive")
    if args.provider == "both" and args.model:
        parser.error("Use a single provider when checking specific model IDs")
    print(f"Host: {socket.gethostname()}; Slurm job: {os.environ.get('SLURM_JOB_ID', 'none')}")
    print("Only model-metadata GET requests; no prompts, images, or generation charges.")
    providers = ["openai", "anthropic"] if args.provider == "both" else [args.provider]
    results = [check_provider(provider, args.model, args.timeout) for provider in providers]
    print("This does not verify inference billing, token quotas, or image/reasoning compatibility.")
    return 0 if all(results) else 1


if __name__ == "__main__":
    sys.exit(main())
