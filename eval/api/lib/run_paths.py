"""Initialize isolated evaluation runs and locate their accounting files."""

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import re
import uuid


# Shell entry points export RESULTS_ROOT from setup/session_env.sh.
RESULTS = Path(os.environ.get("RESULTS_ROOT") or Path(__file__).resolve().parents[2] / "results")


def valid_name(name):
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", name):
        raise ValueError(
            "Run names must start with a letter or digit and contain only letters, digits, ., _, -"
        )
    return name


def initialize(*, name=None, resume=None, output=None, root=None, prefix="run"):
    root = (root or RESULTS).resolve()
    if name and resume:
        raise ValueError("Specify a new run name or a resume run, not both")
    if resume:
        directory = root / valid_name(resume)
        if not directory.is_dir() or directory.resolve().parent != root:
            raise ValueError(f"Resume run does not exist within {root}: {resume}")
        if output and Path(output).resolve() != directory:
            raise ValueError("OUTPUT_DIR must match the specified resume run")
    else:
        if output:
            directory = Path(output).resolve()
            if directory.parent != root:
                raise ValueError(f"Run directory must be directly within {root}")
            valid_name(directory.name)
            if name and directory.name != name:
                raise ValueError("OUTPUT_DIR and RUN_NAME disagree")
        else:
            name = name or (
                f"{prefix}-{datetime.now(timezone.utc):%Y%m%dT%H%M%S}-{uuid.uuid4().hex[:8]}"
            )
            directory = root / valid_name(name)
        root.mkdir(parents=True, exist_ok=True)
        try:
            directory.mkdir()
        except FileExistsError as exc:
            raise ValueError(f"Run already exists: {directory}; specify RESUME_RUN to continue") from exc
    accounting = directory / "run_usage"
    if accounting.resolve() != directory.resolve() / "run_usage":
        raise ValueError("run_usage must not be a symlink outside the run")
    accounting.mkdir(exist_ok=True)
    return directory


def usage_path(output):
    output = Path(output)
    resolved = output.resolve()
    if resolved.is_relative_to(RESULTS.resolve()):
        relative = resolved.relative_to(RESULTS.resolve())
        if len(relative.parts) >= 2:
            return (
                RESULTS / relative.parts[0] / "run_usage"
                / Path(*relative.parts[1:]).with_suffix(".usage.jsonl")
            )
    return output.parent / "run_usage" / output.with_suffix(".usage.jsonl").name


def add_arguments(parser):
    parser.add_argument("--run-name", help="Create a named run within RESULTS_ROOT (default eval/results)")
    parser.add_argument("--resume-run", help="Continue an existing named run within RESULTS_ROOT (default eval/results)")


def initialize_evaluator(args, prefix):
    output = args.output.resolve() if args.output else None
    if args.resume_run:
        directory = initialize(name=args.run_name, resume=args.resume_run)
        if output and not output.is_relative_to(directory.resolve()):
            raise ValueError("--output must be within the resumed run")
    else:
        directory = initialize(
            name=args.run_name, output=output.parent if output else None, prefix=prefix,
        )
    model = re.sub(r"[^A-Za-z0-9._-]", "_", args.model)
    args.output = output or directory / f"pred_{prefix}_{model}.csv"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.budget_usd is not None and args.budget_file is None:
        args.budget_file = directory / "run_usage" / "budget.jsonl"
    if args.budget_file and not args.budget_file.resolve().is_relative_to(directory / "run_usage"):
        raise ValueError("--budget-file must be within the run's run_usage directory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--results-root", type=Path, default=RESULTS)
    parser.add_argument("--prefix", default="run")
    parser.add_argument("--budget-file", type=Path)
    parser.add_argument("--estimate-dir", type=Path)
    args = parser.parse_args()
    try:
        directory = initialize(
            name=args.run_name, resume=args.resume_run, output=args.output_dir,
            root=args.results_root, prefix=args.prefix,
        )
        if args.budget_file and not args.budget_file.resolve().is_relative_to(directory / "run_usage"):
            raise ValueError("BUDGET_FILE must be within the run's run_usage directory")
        if args.estimate_dir and not args.estimate_dir.resolve().is_relative_to(directory):
            raise ValueError("ESTIMATE_DIR must be within the run directory")
        print(directory)
    except ValueError as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
