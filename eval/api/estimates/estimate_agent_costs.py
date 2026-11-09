"""Project direct-model/agent cost and runtime from a subset to the benchmark.

For each prediction CSV under RESULTS (with its ledger under run_usage and
its .meta.json companion), every ledger event is attributed via the hashed
example ID. Per-example USD (all attempts, standard rates from prices.json)
and harness walltime (CSV elapsed_seconds) are averaged within each category,
then weighted by that category's share of the full benchmark scaled to
--target examples. A within-category bootstrap gives the percentile interval.

Safe to run while a job is in progress: only examples with a checkpointed
prediction row enter the projection.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from agents.run_agents_eval import BUDGET_MARKERS as CAP_MARKERS
from estimates.usage import PRICES, token_totals
from lib import common as api
from lib.run_paths import usage_path

PROVIDERS = {"codex": "openai", "claude": "anthropic"}


def example_hash(video):
    return hashlib.sha256(video.encode()).hexdigest()


def read_ledger(path):
    events, seen = [], set()
    if not path.exists():
        return events
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                # A line may be mid-write while a job is running.
                print(f"WARNING: skipping unreadable ledger line {path}:{number}", file=sys.stderr)
                continue
            if event["event_id"] not in seen:
                seen.add(event["event_id"])
                events.append(event)
    return events


def per_example(prediction_rows, events, subset_rows, rates):
    by_hash = {example_hash(row["video"]): row["video"] for row in subset_rows}
    cost, unknown, requests = defaultdict(float), Counter(), Counter()
    unattributed = 0
    for event in events:
        video = by_hash.get(event.get("example_id"))
        if video is None:
            unattributed += 1
            continue
        requests[video] += 1
        totals = token_totals(event)
        if totals is None:
            unknown[video] += 1
        else:
            cost[video] += (totals[0] * rates["input"] + totals[1] * rates["output"]) / 1e6
    predictions = {row["video"]: row for row in prediction_rows}
    examples = []
    for row in subset_rows:
        prediction = predictions.get(row["video"])
        if prediction is None:
            continue
        text = prediction.get("prediction", "")
        examples.append(dict(
            video=row["video"], category=row["category"],
            usd=cost[row["video"]], seconds=float(prediction.get("elapsed_seconds") or 0),
            requests=requests[row["video"]], unknown_usage=unknown[row["video"]],
            success=api.successful(text),
            capped=any(marker in text for marker in CAP_MARKERS),
        ))
    in_progress_usd = sum(
        value for video, value in cost.items() if video not in predictions
    )
    return examples, unattributed, in_progress_usd


def stratified_total(groups, weights, key, fallback):
    total = 0.0
    for name, weight in weights.items():
        values = [example[key] for example in groups.get(name, [])]
        total += weight * (sum(values) / len(values) if values else fallback)
    return total


def bootstrap(groups, weights, key, fallback, draws, rng, level):
    totals = []
    for _ in range(draws):
        resampled = {
            name: [rng.choice(members) for _ in members] for name, members in groups.items()
        }
        totals.append(stratified_total(resampled, weights, key, fallback))
    totals.sort()
    tail = (1 - level) / 2

    def quantile(q):
        position = q * (len(totals) - 1)
        low, high = math.floor(position), math.ceil(position)
        return totals[low] + (totals[high] - totals[low]) * (position - low)

    return quantile(tail), quantile(1 - tail)


def project(examples, category_sizes, target, draws, seed, level):
    population = sum(category_sizes.values())
    weights = {name: size * target / population for name, size in category_sizes.items()}
    groups = defaultdict(list)
    for example in examples:
        groups[example["category"]].append(example)
    missing = sorted(set(weights) - set(groups))
    result = {"observed_examples": len(examples), "missing_categories": missing}
    for key, scale, label in (("usd", 1, "usd"), ("seconds", 1 / 3600, "hours")):
        fallback = sum(e[key] for e in examples) / len(examples)
        point = stratified_total(groups, weights, key, fallback)
        low, high = bootstrap(groups, weights, key, fallback, draws, random.Random(seed), level)
        result[label] = {
            "estimate": point * scale, "interval": [low * scale, high * scale],
            "naive_mean_times_target": fallback * target * scale,
        }
    categories = []
    for name in sorted(weights):
        members = groups.get(name, [])
        n = len(members)
        mean_usd = sum(e["usd"] for e in members) / n if n else None
        mean_seconds = sum(e["seconds"] for e in members) / n if n else None
        categories.append(dict(
            category=name, sampled=n, target_examples=weights[name],
            mean_usd_per_example=mean_usd,
            projected_usd=None if mean_usd is None else mean_usd * weights[name],
            mean_minutes_per_example=None if mean_seconds is None else mean_seconds / 60,
            projected_hours=None if mean_seconds is None else mean_seconds * weights[name] / 3600,
            capped=sum(e["capped"] for e in members),
            failed=sum(not e["success"] for e in members),
        ))
    result["categories"] = categories
    return result


def estimate(results, subset_path, full_path, prices, target, draws, seed, level):
    subset_rows = api.read_csv(subset_path)
    if not subset_rows or "category" not in subset_rows[0]:
        raise ValueError("Subset CSV needs a category column")
    category_sizes = Counter(row["category"] for row in api.read_csv(full_path))
    unknown_categories = {row["category"] for row in subset_rows} - set(category_sizes)
    if unknown_categories:
        raise ValueError(f"Subset categories absent from full CSV: {sorted(unknown_categories)}")
    reports = []
    for path in sorted(results.glob("pred_*.csv")):
        meta_path = path.with_suffix(".meta.json")
        if not meta_path.exists():
            print(f"WARNING: no metadata for {path.name}; skipped", file=sys.stderr)
            continue
        meta = json.loads(meta_path.read_text())
        provider = meta.get("provider") or PROVIDERS.get(meta.get("harness"))
        if provider is None:
            continue
        key = f"{provider}/{meta['model']}"
        rates = prices["models"].get(key)
        if rates is None:
            raise ValueError(f"No price for {key} in prices.json")
        examples, unattributed, in_progress = per_example(
            api.read_csv(path), read_ledger(usage_path(path)),
            subset_rows, rates,
        )
        report = dict(
            harness=meta.get("harness") or provider, provider=provider,
            mode="agent" if meta.get("harness") else "model",
            model=meta["model"], prediction_csv=str(path),
            subset_size=len(subset_rows), harness_version=meta.get("version"),
            max_requests=meta.get("max_requests"), seconds_cap=meta.get("seconds"),
            max_output_tokens=meta.get("max_output_tokens"),
            observed_subset_usd=sum(e["usd"] for e in examples) + in_progress,
            in_progress_usd=in_progress, unattributed_ledger_events=unattributed,
            unknown_usage_requests=sum(e["unknown_usage"] for e in examples),
            capped_examples=sum(e["capped"] for e in examples),
            failed_examples=sum(not e["success"] for e in examples),
        )
        if examples:
            report.update(project(examples, category_sizes, target, draws, seed, level))
        else:
            report.update(observed_examples=0)
        reports.append(report)
    return reports


def money(value):
    return "n/a" if value is None else f"${value:,.2f}"


def render(reports, target, level):
    pct = f"{level:.0%}"
    lines = [
        f"| Harness/provider/model | Done/N | USD for {target} (est.) | {pct} interval | "
        f"Hours for {target} (est.) | {pct} interval | Capped | Failed | Subset USD so far |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in reports:
        name = f"{r['harness']}/{r['model']}"
        if not r["observed_examples"]:
            lines.append(f"| {name} | 0/{r['subset_size']} | | | | | | | {money(r['observed_subset_usd'])} |")
            continue
        usd, hours = r["usd"], r["hours"]
        lines.append(
            f"| {name} | {r['observed_examples']}/{r['subset_size']} | {money(usd['estimate'])} | "
            f"{money(usd['interval'][0])}-{money(usd['interval'][1])} | {hours['estimate']:.1f} | "
            f"{hours['interval'][0]:.1f}-{hours['interval'][1]:.1f} | {r['capped_examples']} | "
            f"{r['failed_examples']} | {money(r['observed_subset_usd'])} |"
        )
    for r in reports:
        if not r["observed_examples"]:
            continue
        lines += [
            "", f"**{r['harness']}/{r['model']}** per category",
            "| Category | Sampled | Target rows | $/example | Projected USD | Min/example | Projected hours | Capped |",
            "| --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for c in r["categories"]:
            minutes = "n/a" if c["mean_minutes_per_example"] is None else f"{c['mean_minutes_per_example']:.1f}"
            hours = "n/a" if c["projected_hours"] is None else f"{c['projected_hours']:.1f}"
            lines.append(
                f"| {c['category']} | {c['sampled']} | {c['target_examples']:.0f} | "
                f"{money(c['mean_usd_per_example'])} | {money(c['projected_usd'])} | {minutes} | "
                f"{hours} | {c['capped']} |"
            )
        notes = []
        if r["missing_categories"]:
            notes.append(f"no finished examples yet in {', '.join(r['missing_categories'])} (overall mean used)")
        if r["capped_examples"]:
            notes.append(f"{r['capped_examples']} example(s) hit the request/time cap, so cost is a lower bound")
        uncapped_failures = r["failed_examples"] - r["capped_examples"]
        if uncapped_failures:
            notes.append(
                f"{uncapped_failures} example(s) failed for non-budget reasons (CLI/auth/gateway); "
                "inspect the job log - the estimate is unreliable until these succeed"
            )
        if r["unknown_usage_requests"]:
            notes.append(f"{r['unknown_usage_requests']} request(s) have unknown usage (excluded)")
        for note in notes:
            lines.append(f"- WARNING: {note}")
    lines += [
        "",
        f"Hours are sequential run time (incl. rate-limit waits) at the pilot's RPM; "
        f"intervals are within-category bootstrap percentiles and are crude at small N. "
        f"Standard input/output rates only; provider billing is authoritative.",
    ]
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Run or pilot directory with predictions and metadata")
    parser.add_argument("--subset", type=Path, required=True, help="Stratified subset CSV used for the run")
    parser.add_argument("--full", type=Path, default=api.DATA / "test.csv", help="Full benchmark CSV (category sizes)")
    parser.add_argument("--prices", type=Path, default=PRICES)
    parser.add_argument("--target", type=int, default=500)
    parser.add_argument("--bootstrap", type=int, default=5000)
    parser.add_argument("--level", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, help="Optional JSON report path")
    parser.add_argument("--markdown", type=Path, help="Optional Markdown report path")
    args = parser.parse_args(argv)
    if not 0 < args.level < 1 or args.bootstrap < 1 or args.target < 1:
        parser.error("Need 0 < --level < 1 and positive --bootstrap/--target")
    prices = json.loads(args.prices.read_text())
    reports = estimate(
        args.results, args.subset, args.full, prices,
        args.target, args.bootstrap, args.seed, args.level,
    )
    if not reports:
        parser.error(f"No agent prediction CSVs with metadata under {args.results}")
    text = render(reports, args.target, args.level)
    print(text)
    if args.markdown:
        args.markdown.write_text(text + "\n")
    if args.output:
        args.output.write_text(json.dumps(dict(
            target=args.target, level=args.level, bootstrap=args.bootstrap,
            price_as_of=prices["as_of"], subset=str(args.subset), models=reports,
        ), indent=2) + "\n")


if __name__ == "__main__":
    main()
