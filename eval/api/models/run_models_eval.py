"""Run resumable CPU-only frame evaluation and answer extraction via hosted APIs."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

import anthropic
import openai

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from estimates.usage import Ledger
from lib.budget import Budget, BudgetExceeded
from lib import common, frames, run_paths


def content_blocks(provider, prompt, images):
    blocks = [{"type": "text", "text": prompt}]
    for image in images:
        if provider == "openai":
            blocks.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}", "detail": "high",
                },
            })
        else:
            blocks.append({
                "type": "image",
                "source": {
                    "type": "base64", "media_type": "image/jpeg", "data": image,
                },
            })
    return blocks


class Provider:
    def __init__(self, args):
        self.args = args
        self.limiter = common.RateLimiter(args.rpm)
        self.ledger = Ledger(
            args.output, args.provider, args.model, args.command,
            args.openai_endpoint if args.provider == "openai" else "messages",
        )
        self.budget = (
            Budget(args.budget_file, args.budget_usd, args.provider, args.model)
            if args.budget_usd is not None else None
        )
        if self.budget is not None:
            settled, reserved = self.budget.spent()
            print(
                f"USD budget: ${settled + reserved:.2f} committed "
                f"(${reserved:.2f} reserved) of ${self.budget.limit:.2f}; "
                f"ledger={self.budget.path}",
                flush=True,
            )
        if args.provider == "openai":
            self.client = openai.OpenAI(
                api_key=os.environ["OPENAI_API_KEY"],
                max_retries=0, timeout=args.timeout,
            )
        else:
            self.client = anthropic.Anthropic(
                api_key=os.environ["ANTHROPIC_API_KEY"],
                max_retries=0, timeout=args.timeout,
            )

    def close(self):
        self.client.close()

    def query(self, prompt, images, example_id=None):
        args = self.args
        content = content_blocks(args.provider, prompt, images)
        for attempt in range(args.attempts):
            self.limiter.wait()
            reservation = None
            reserved_usd = 0.0
            budget = getattr(self, "budget", None)
            if budget is not None:
                body = {"provider": args.provider, "model": args.model, "content": content}
                reserved_usd = budget.worst_case(body, args.max_output_tokens)
                reservation = budget.reserve(
                    reserved_usd,
                    example_id=example_id,
                )
                if reservation is None:
                    settled, open_usd = self.budget.spent()
                    raise BudgetExceeded(
                        f"USD budget reached: ${settled + open_usd:.2f} committed of "
                        f"${budget.limit:.2f}; next worst-case request needs "
                        f"${reserved_usd:.2f}"
                    )
            started = time.monotonic()
            try:
                if args.provider == "openai" and args.openai_endpoint == "responses":
                    options = {}
                    if args.reasoning_effort:
                        options["reasoning"] = {"effort": args.reasoning_effort}
                    response = self.client.responses.create(
                        model=args.model,
                        input=[{"role": "user", "content": [
                            {"type": "input_text", "text": prompt},
                            *[{
                                "type": "input_image",
                                "image_url": f"data:image/jpeg;base64,{image}",
                                "detail": "high",
                            } for image in images],
                        ]}],
                        max_output_tokens=args.max_output_tokens,
                        store=False, **options,
                    )
                    self.ledger.record(
                        response, time.monotonic() - started,
                        response.status, attempt + 1, example_id,
                    )
                    self._settle_budget(reservation, reserved_usd, response.usage,
                                        response.status)
                    if response.status != "completed":
                        raise ValueError(f"Incomplete/refused response: {response.status}")
                    text = response.output_text
                    usage = response.usage
                    input_tokens = usage.input_tokens if usage else ""
                    output_tokens = usage.output_tokens if usage else ""
                elif args.provider == "openai":
                    options = {}
                    if args.reasoning_effort:
                        options["reasoning_effort"] = args.reasoning_effort
                    response = self.client.chat.completions.create(
                        model=args.model,
                        messages=[{"role": "user", "content": content}],
                        max_completion_tokens=args.max_output_tokens,
                        **options,
                    )
                    self.ledger.record(
                        response, time.monotonic() - started,
                        response.choices[0].finish_reason if response.choices else "no_choices",
                        attempt + 1, example_id,
                    )
                    self._settle_budget(
                        reservation, reserved_usd, response.usage,
                        response.choices[0].finish_reason if response.choices else "no_choices",
                    )
                    if not response.choices:
                        raise ValueError("Provider returned no choices")
                    choice = response.choices[0]
                    if choice.finish_reason != "stop":
                        raise ValueError(
                            f"Incomplete/refused response: {choice.finish_reason}"
                        )
                    text = choice.message.content
                    usage = response.usage
                    input_tokens = usage.prompt_tokens if usage else ""
                    output_tokens = usage.completion_tokens if usage else ""
                else:
                    response = self.client.messages.create(
                        model=args.model,
                        messages=[{"role": "user", "content": content}],
                        max_tokens=args.max_output_tokens,
                    )
                    self.ledger.record(
                        response, time.monotonic() - started,
                        response.stop_reason, attempt + 1, example_id,
                    )
                    self._settle_budget(
                        reservation, reserved_usd, response.usage, response.stop_reason,
                    )
                    if response.stop_reason != "end_turn":
                        raise ValueError(
                            f"Incomplete/refused response: {response.stop_reason}"
                        )
                    text = "\n".join(
                        block.text for block in response.content
                        if block.type == "text"
                    )
                    input_tokens = response.usage.input_tokens
                    output_tokens = response.usage.output_tokens
                if not text or not text.strip():
                    raise ValueError("Provider returned no answer text")
                return text.strip(), input_tokens, output_tokens
            except (
                openai.APIConnectionError, openai.APIStatusError,
                anthropic.APIConnectionError, anthropic.APIStatusError,
            ) as exc:
                status = getattr(exc, "status_code", None)
                self._settle_budget(
                    reservation, reserved_usd, None, f"api_error_{status}",
                )
                self.ledger.record(
                    None, time.monotonic() - started,
                    f"api_error_{status}", attempt + 1, example_id,
                )
                retryable = status is None or status in (408, 409, 429) or status >= 500
                if not retryable or attempt + 1 == args.attempts:
                    raise
                delay = min(120, 5 * 2 ** attempt)
                response = getattr(exc, "response", None)
                if response is not None:
                    retry_after = response.headers.get("retry-after", "")
                    try:
                        delay = max(delay, float(retry_after))
                    except ValueError:
                        pass  # HTTP-date Retry-After is not a seconds value.
                print(
                    f"API retry {attempt + 1}: {type(exc).__name__}, "
                    f"status={status}; waiting {delay:.1f}s",
                    file=sys.stderr, flush=True,
                )
                time.sleep(delay)

    def _settle_budget(self, reservation, reserved_usd, usage, status):
        if reservation is None:
            return
        usage_dict = usage.model_dump(mode="json") if usage is not None else None
        self.budget.settle(
            reservation, reserved_usd, usage_dict, str(status), upstream_accepted=True,
        )


def extraction_prompt(row):
    # The judge sees the question and prediction, never the ground truth.
    return (
        "Extract the final answer from the model response below. Return ONLY "
        "the answer, with no explanation, Markdown, or added punctuation. "
        "Follow the question's requested format and precision. For multiple "
        "choice return only the option letter; preserve its case if specified. "
        "Return None if the response gives no answer. Do not solve the question "
        "yourself or correct the model's answer.\n\n"
        f"Question: {row['question_text']}\n\n"
        f"Model response: {row['prediction']}\n\nExtracted answer:"
    )


def run(args):
    rows = common.load_rows(args)
    if args.command == "prepare":
        frames.prepare_cache(args, rows)
        return
    if args.command == "predict":
        rows = [
            dict(row, idx=str(row.get("idx") or i + 1))
            for i, row in enumerate(rows)
        ]
        fields = common.FIELDS
        answer_column = "prediction"
        for row in rows:
            if not (args.video_root / row["video"]).is_file():
                raise FileNotFoundError(args.video_root / row["video"])
            if args.frame_cache:
                frames.load_cached_frames(args, row["video"])
    else:
        fields = list(rows[0]) + [
            "extracted_answer", "judge_input_tokens", "judge_output_tokens",
            "judge_elapsed_seconds",
        ]
        if len(fields) != len(set(fields)):
            raise ValueError("Extract input must be a raw prediction CSV")
        answer_column = "extracted_answer"
        if any(not common.successful(row.get("prediction", "")) for row in rows):
            raise ValueError("Predictions contain errors/empty answers; resume evaluation first")
    if args.input.resolve() == args.output.resolve():
        raise ValueError("Input and output must be different files")

    config = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
        if key not in {
            "rpm", "concurrency", "attempts", "timeout", "output",
            "budget_usd", "budget_file",
            "run_name", "resume_run",
        }
    }
    # Chat Completions is the existing protocol, so old metadata stays compatible.
    if config["openai_endpoint"] == "chat":
        del config["openai_endpoint"]
    config["input_sha256"] = hashlib.sha256(args.input.read_bytes()).hexdigest()
    config["protocol_version"] = 1
    if args.command == "predict":
        sources = [
            frames.source_identity(args.video_root / row["video"]) for row in rows
        ]
        config["video_sources_sha256"] = hashlib.sha256(
            json.dumps(sources, sort_keys=True).encode("utf-8")
        ).hexdigest()
    metadata = args.output.with_suffix(".meta.json")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if metadata.exists():
        if json.loads(metadata.read_text()) != config:
            raise ValueError("Resume configuration/input changed; choose a new output path")
    elif args.output.exists():
        raise ValueError("Existing output has no metadata; choose a new output path")
    else:
        metadata.write_text(json.dumps(config, indent=2) + "\n")

    existing = common.read_csv(args.output) if args.output.exists() else []
    results = {row["video"]: row for row in existing}
    expected = {row["video"] for row in rows}
    if len(results) != len(existing) or not set(results).issubset(expected):
        raise ValueError("Checkpoint contains duplicate or unexpected videos")
    pending = [
        row for row in rows
        if not common.successful(results.get(row["video"], {}).get(answer_column, ""))
    ]
    print(
        f"{args.provider}/{args.model}: {len(pending)}/{len(rows)} pending; "
        f"concurrency={args.concurrency}, rpm={args.rpm}",
        flush=True,
    )
    if not pending:
        return
    provider = Provider(args)
    started = time.monotonic()

    def process(row):
        start = time.monotonic()
        result = {field: row.get(field, "") for field in fields}
        try:
            if args.command == "predict":
                if args.frame_cache:
                    images = frames.load_cached_frames(args, row["video"])
                else:
                    images = frames.sample_video(
                        args.video_root / row["video"], args.fps,
                        args.max_frames, args.max_side,
                    )
                prompt = "Answer the question in this video."
                result["frame_count"] = len(images)
                prefix = ""
            else:
                images = []
                prompt = extraction_prompt(row)
                prefix = "judge_"
            answer, input_tokens, output_tokens = provider.query(
                prompt, images, example_id=hashlib.sha256(
                    row["video"].encode("utf-8")
                ).hexdigest(),
            )
            result[answer_column] = answer
            result[prefix + "input_tokens"] = input_tokens
            result[prefix + "output_tokens"] = output_tokens
        except (
            openai.OpenAIError, anthropic.AnthropicError,
            OSError, ValueError, IndexError,
        ) as exc:
            print(f"ERROR {row['video']}: {type(exc).__name__}: {exc}", file=sys.stderr)
            result[answer_column] = f"ERROR: {type(exc).__name__}: {exc}"
        prefix = "" if args.command == "predict" else "judge_"
        result[prefix + "elapsed_seconds"] = f"{time.monotonic() - start:.3f}"
        return result

    try:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            tasks = [pool.submit(process, row) for row in pending]
            for completed, future in enumerate(as_completed(tasks), 1):
                result = future.result()
                results[result["video"]] = result
                ordered = [results[row["video"]] for row in rows if row["video"] in results]
                common.save_csv(args.output, ordered, fields)
                print(f"Checkpoint {completed}/{len(pending)}: {result['video']}", flush=True)
    finally:
        provider.close()
    failures = sum(
        not common.successful(results[row["video"]].get(answer_column, "")) for row in rows
    )
    print(
        f"Finished in {(time.monotonic() - started) / 60:.2f} minutes; "
        f"{len(rows) - failures}/{len(rows)} successful; output={args.output}",
        flush=True,
    )
    if failures:
        raise RuntimeError(f"{failures} failed rows; rerun the same command to retry them")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "predict", "extract"])
    parser.add_argument("--provider", choices=["openai", "anthropic"])
    parser.add_argument("--model")
    parser.add_argument("--openai-endpoint", choices=["chat", "responses"], default="chat")
    common.add_data_arguments(parser)
    parser.add_argument("--output", type=Path)
    run_paths.add_arguments(parser)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--rpm", type=float, default=5)
    parser.add_argument("--max-output-tokens", type=int, default=4096)
    parser.add_argument("--reasoning-effort", help="OpenAI only; omitted uses model default")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--attempts", type=int, default=5)
    parser.add_argument("--budget-usd", type=float,
                        help="Cumulative standard-rate USD cap; exit 76 when reached")
    parser.add_argument("--budget-file", type=Path,
                        help="Shared append-only budget ledger")
    args = parser.parse_args(argv)
    if args.command == "prepare":
        if not args.frame_cache:
            parser.error("prepare requires --frame-cache")
        if args.provider or args.model or args.output or args.run_name or args.resume_run:
            parser.error("prepare does not take provider, model, output, or run arguments")
    elif not args.provider or not args.model:
        parser.error("predict/extract require --provider and --model")
    if args.command == "extract" and args.frame_cache:
        parser.error("--frame-cache applies only to prepare/predict")
    if args.budget_usd is not None:
        if not math.isfinite(args.budget_usd) or args.budget_usd <= 0:
            parser.error("--budget-usd must be finite and positive")
        if args.budget_file is None:
            if args.output is not None:
                args.budget_file = run_paths.usage_path(args.output).parent / "budget.jsonl"
    common.validate_positive_arguments(parser, args, (
        "fps", "max_frames", "max_side", "concurrency", "rpm",
        "max_output_tokens", "timeout", "attempts", "limit", "expected_count",
    ))
    if args.provider != "openai" and args.reasoning_effort:
        parser.error("--reasoning-effort is supported only for OpenAI")
    if args.openai_endpoint != "chat" and args.provider != "openai":
        parser.error("--openai-endpoint responses is supported only for OpenAI")
    return args


if __name__ == "__main__":
    try:
        args = parse_args()
        if args.command != "prepare":
            run_paths.initialize_evaluator(args, f"model-{args.provider}")
        run(args)
    except BudgetExceeded as exc:
        print(f"BUDGET REACHED: {exc}", file=sys.stderr, flush=True)
        sys.exit(76)
