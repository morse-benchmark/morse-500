# Evaluation

MORSE-500 has an API evaluation regime for direct hosted models and official
coding-agent harnesses. The tracks are distinct: `model` calls a provider API
directly, while `agent` runs Codex for OpenAI or Claude Code for Anthropic.
Both use the shared runner in [`api/`](api/README.md), locally or with Slurm.
The low-level entry points are
[`run_models_eval.py`](api/models/run_models_eval.py) and
[`run_agents_eval.py`](api/agents/run_agents_eval.py).

## Setup

Run commands below from the `morse-500/` root. The setup script requires `uv`;
install it using your approved package-manager process first. The API workflow
uses MoviePy 2; keep a separate environment if the original baseline needs
MoviePy 1.

To install into the default `.morse_venv`:

```bash
bash eval/api/setup/setup_env.sh
```

Alternatively, to keep the environments separate:

```bash
VENV="$PWD/.morse_api_venv" bash eval/api/setup/setup_env.sh
export PYTHON="$PWD/.morse_api_venv/bin/python"
```

By default the workflow reads `../datasets/morse-500/test.csv` and videos from
`../datasets/morse-500/test_sz512/`. Set `DATA_CSV` and `VIDEO_ROOT` if the
dataset is elsewhere.

Export the relevant `OPENAI_API_KEY` or `ANTHROPIC_API_KEY` securely; do not
put keys in scripts, shell history, or command-line arguments. For example, in
an interactive Bash shell, run the line for each provider you intend to use:

```bash
read -rsp "OpenAI API key: " OPENAI_API_KEY; echo; export OPENAI_API_KEY
read -rsp "Anthropic API key: " ANTHROPIC_API_KEY; echo; export ANTHROPIC_API_KEY
```

Agent mode additionally requires Linux, `bwrap`, `socat`, user/network
namespaces, and the official native harness binaries. Codex also needs its
matching `codex-code-mode-host`. Follow
the [agent setup guide](api/README.md#agent-mode-only-stage-the-harness-binaries)
to stage the pinned binaries, then set `AGENT_BIN` or the explicit binary paths.

Check prerequisites without inference requests, selecting the mode and provider
you intend to use:

```bash
bash eval/api/setup/validate.sh --mode model --provider openai
bash eval/api/setup/validate.sh --mode agent --provider openai
```

For an authenticated provider/model metadata check without inference, replace
the example model ID and run:

```bash
PREFLIGHT_ONLY=1 bash eval/api/scripts/run_local.sh model openai OPENAI_MODEL_ID
```

Use the corresponding mode and provider for an agent check. This sends
authenticated metadata requests and checks the agent harness when applicable;
it does not verify inference access or image compatibility.

## Run a small API pilot

Replace the example IDs with models accessible to your provider account.
`LIMIT=2` limits each selected model to two examples and the separate output
directories keep these pilots apart from full-run results. **These commands make
billable inference requests.**

```bash
# Direct model via OpenAI or Anthropic
LIMIT=2 RUN_NAME=model_pilot_openai \
  bash eval/api/scripts/run_local.sh model openai OPENAI_MODEL_ID
LIMIT=2 RUN_NAME=model_pilot_anthropic \
  bash eval/api/scripts/run_local.sh model anthropic ANTHROPIC_MODEL_ID

# Official Codex or Claude Code harness
LIMIT=2 RUN_NAME=agent_pilot_openai \
  bash eval/api/scripts/run_local.sh agent openai CODEX_MODEL_ID
LIMIT=2 RUN_NAME=agent_pilot_anthropic \
  bash eval/api/scripts/run_local.sh agent anthropic CLAUDE_MODEL_ID
```

For Slurm, use the same mode, provider, and model arguments with the submission
wrapper, for example:

```bash
LIMIT=2 RUN_NAME=model_pilot_slurm \
  bash eval/api/scripts/submit_slurm.sh model openai OPENAI_MODEL_ID
```

Set `SLURM_PARTITION`, `SLURM_ACCOUNT`, and `SLURM_QOS` as required by the
cluster. The submitter exports the environment to the job, so follow the
cluster's credential policy.
The [API guide](api/README.md) covers configuration, estimates, optional budget
enforcement, checkpoints, and full runs. Pilot outputs are not full benchmark
results; the scoring wrapper requires 500 rows per prediction file.

## Validate the implementation without paid calls

```bash
"${PYTHON:-$PWD/.morse_venv/bin/python}" -m unittest discover \
  -s eval/api/tests -p 'test_*.py' -v
bash -n eval/api/lib/run_eval.sh eval/api/scripts/*.sh \
  eval/api/scripts/*.sbatch eval/api/setup/*.sh
```

The tests use mock provider responses and do not establish that a real model,
account, or harness has inference access. Check those with a small paid pilot
before a full run.

## Local Qwen extraction and scoring

All self-hosted Qwen setup, validation, extraction, scoring, and Slurm scripts
live in [`qwen_eval/`](qwen_eval/README.md). That workflow uses a separate
`.qwen_venv` and requires a GPU to serve Qwen; tabulation of extracted answers
is CPU-only. Follow the [Qwen guide](qwen_eval/README.md) after predictions are
complete. Prediction and extraction output directories have not moved.
