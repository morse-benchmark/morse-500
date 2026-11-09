# Pricing and cost estimates

This directory contains token prices, usage accounting, and pilot/static cost
estimators for both direct-model and agent runs. Commands below run from the
`morse-500/` repository root.

## Prices and model IDs

`prices.json` stores standard, short-context USD per million input/output
tokens, keyed by `provider/model`. Add the exact provider and model ID used by
the runner; rates must be finite and non-negative. Update `as_of` and `source`
when rates change. `usage.py` excludes unpriced models from its subtotal;
`estimate_costs.py`, the budget guard, and estimate/budget preflight require a
price and fail if one is missing. Both report CLIs accept `--prices PATH` to
reprice existing token ledgers.

| Provider | API model IDs in `prices.json` | Notes |
| --- | --- | --- |
| OpenAI | `gpt-6-astra`, `gpt-6.1-sol`, `gpt-6-luna` | Direct-model track. |
| OpenAI | `gpt-6.1-terra` | Coding-agent pilot only. |
| OpenAI | `gpt-5.3-codex` | Deprecated; opt-in Responses endpoint only. |
| Anthropic | `claude-fable-5-1`, `claude-mythos-5-1`, `claude-opus-5-5` | Direct-model track; Mythos requires verified-organization access. |
| Anthropic | `claude-sonnet-5-5` | Claude Code underlying model; Claude Code is a harness, not a separate model ID. |

The current runner supports OpenAI and Anthropic, not Google. Documented model
IDs do not guarantee access or inference permission; verify access from a
compute node before a paid pilot:

```bash
bash eval/api/scripts/extras/submit_check_slurm.sh openai \
  --model gpt-6-astra --model gpt-6.1-sol --model gpt-6-luna
bash eval/api/scripts/extras/submit_check_slurm.sh anthropic \
  --model claude-fable-5-1 --model claude-opus-5-5 --model claude-sonnet-5-5
```

These are metadata checks, not inference calls. Check restricted/deprecated
models separately. Official pricing references:
[OpenAI](https://developers.openai.com/api/docs/pricing),
[Anthropic](https://platform.claude.com/docs/en/about-claude/pricing), and
[Claude Code costs](https://code.claude.com/docs/en/costs). Prices are estimates,
not provider invoices; cache pricing, tiers, taxes, and contracts may differ.

## Generate estimates

### Actual usage and standard-rate cost

```bash
.morse_venv/bin/python eval/api/estimates/usage.py eval/results/RUN_NAME \
  --output eval/results/RUN_NAME/run_usage/usage_summary.json
```

The command recursively reads `*.usage.jsonl`, deduplicates events, and groups
by provider/model/stage. Cost is
`(input_tokens * input_rate + output_tokens * output_rate) / 1e6`. OpenAI
reported totals already include cached/reasoning tokens. Anthropic cache-read
and cache-creation tokens are added to input and priced at the standard rate.
Unknown usage is counted but excluded; interrupted requests may have partial
usage. For either mode, pass `eval/results/RUN_NAME`, including for pilot usage;
pilot ledgers live under the run's `run_usage/estimate/`, not beside pilot CSVs.

### Pilot cost and walltime projection

The pilot creates a stratified subset, runs the selected mode/provider/models,
then writes `estimate.md` and `estimate.json`:

```bash
bash eval/api/scripts/estimate_local.sh model openai gpt-6-astra
bash eval/api/scripts/submit_estimate_slurm.sh agent anthropic claude-opus-5-5
```

To re-estimate existing results:

```bash
D=eval/results/RUN_NAME/estimate
.morse_venv/bin/python eval/api/estimates/estimate_costs.py "$D" \
  --subset "$D/subset.csv" --target 500 --bootstrap 5000 --level 0.9 --seed 0 \
  --output "$D/estimate.json" --markdown "$D/estimate.md"
```

`static_estimate.py model|agent PROVIDER MODEL` provides a no-request
price-table projection using configurable frame/token/request assumptions. It
is a rough ceiling, not a pilot-based forecast.

## Estimation method and settings

`make_subset.py` allocates `N // K` rows to each of `K` categories, giving
remainders to the largest categories (ties by name). It samples each category
without replacement using `random.Random(f"{seed}:{category}")` and preserves
benchmark order.

Each checkpointed example's cost includes all attributed request-ledger events
(including retries); its time is CSV `elapsed_seconds`. For category `c`, let
`M_c` be its full-benchmark size, `n_c` the number of completed pilot examples,
and `x_i` each example's cost or time. The projected weight and stratified
estimate are:

```text
w_c = M_c * target / sum(M_c)
estimate = sum_c w_c * mean(x_i for i in c)
```

The confidence interval is a stratified percentile bootstrap: within each
category, resample its observed examples with replacement to the same `n_c`,
recalculate the weighted total, repeat `B` times, then take the
`(1-level)/2` and `(1+level)/2` quantiles with linear interpolation. Category
weights remain fixed. If a category has no completed examples, the overall
observed mean is used and the report flags the category. Cost and time intervals
reflect sampling variability only; they are crude for small samples and
heavy-tailed agent costs. Agent request/time-cap examples are truncated, so their projections may be
lower bounds. Provider billing is authoritative.

| Setting | Default | Meaning |
| --- | ---: | --- |
| `--target` / `TARGET` | 500 | Benchmark examples projected to. |
| `--bootstrap` | 5000 | Bootstrap draws `B`. |
| `--level` | 0.9 | Two-sided interval level; must be between 0 and 1. |
| `--seed` | 0 | Bootstrap random seed. |
| `--full` / `DATA_CSV` | `datasets/morse-500/test.csv` | Full benchmark category sizes. |
| `--prices` | `estimates/prices.json` | Rates used for cost calculations. |
| `SUBSET_SIZE` / `SUBSET_SEED` | 60 / 0 | Pilot subset size and selection seed. |
| `MAX_REQUESTS` / `SECONDS_PER_EXAMPLE` | 30 / 1800 | Per-example request and walltime caps. |
| `MAX_OUTPUT_TOKENS` / `RPM` | 8192 / 5 | Output cap per request and requests per minute. |
| `FPS` / `MAX_FRAMES` / `MAX_SIDE` | 4 / 32 / 512 | Pilot frame sampling settings. |

Match pilot caps to `scripts/config.sh` when projecting a full sweep.
Historical fixed-token scenario totals (4M input + 125K output tokens per
500 examples) are planning assumptions, not measured usage or confidence bounds;
prefer reports from actual usage ledgers.
