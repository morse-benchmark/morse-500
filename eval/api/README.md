# CPU-only OpenAI and Anthropic evaluation

This workflow collects raw predictions with hosted APIs, then uses the original
locally served Qwen model to extract answers and exact-match scoring.
Prediction collection needs no local model server, CUDA, or GPU allocation.
Local answer extraction uses a GPU allocation, not a billable hosted model.
Only evaluation API usage is billable.
No paid requests are made during installation or the unit tests.

## Directory layout

- `models/` and `agents/`: direct API and official coding-agent harness evaluators
  (see [Coding-agent harness track](#coding-agent-harness-track)).
- `lib/`: shared runner, budget, benchmark, and frame-cache modules.
- `scripts/`: local/Slurm run and estimate entry points and `config.sh` (model lists, Slurm settings);
  `scripts/extras/` holds the optional connectivity check (`submit_check_slurm.sh`) and
  failed-row recovery (`retry_failed.sh`). Every sbatch file lives in one of these two folders.
- `setup/`: session setup (`session_env.sh`), environment setup, pinned harness versions, and prerequisite checks.
- `estimates/`: price data, usage accounting, and model/agent cost projections;
  see [ESTIMATES.md](estimates/ESTIMATES.md).
- `utils/`: failed-row preparation, connectivity checks, and local scoring.
- `tests/`: model, agent, and shared-module tests.

The shared frame cache lives in `FRAME_CACHE_ROOT` (typically `eval/api/frame_cache/`);
all model, agent, pilot, and recovery runs live in named directories under
`RESULTS_ROOT` (typically `eval/results/`). Both must be filled in in
`setup/session_env.sh`, which must be sourced before any script runs (see [Setup](#setup)).
Slurm jobs use the same runner as local Bash runs; configure
partition/account/QOS in the environment rather than editing `#SBATCH` lines.

## Direct models versus coding agents

There are two evaluation tracks, selected by the `MODE` argument of the run scripts:

| Track | Mode/provider | Default models (`scripts/config.sh`) |
| --- | --- | --- |
| Direct models | `model openai`, `model anthropic` | GPT-6 Astra, GPT-6.1 Sol, GPT-6 Luna; Fable 5.1, Opus 5.5 |
| Official Codex CLI | `agent openai` | GPT-5.3-Codex, GPT-6 Astra, GPT-6.1 Sol, GPT-6.1 Terra, GPT-6 Luna |
| Official Claude Code CLI | `agent anthropic` | Sonnet 5.5, Opus 5.5, Fable 5.1 |

- Treat each mode/provider/model combination as a distinct configuration (13 by
  default). Do not reuse direct Opus/Fable scores for their Claude Code
  configurations: a Responses API call alone is not the Codex harness, and a
  Messages call alone is not Claude Code.
- Both tracks send the same frames and share setup, run scripts, results layout,
  budgets, and scoring. Agent-specific limits, sandboxing, and outputs are in
  [Coding-agent harness track](#coding-agent-harness-track).
- GPT-5.3-Codex is documented as deprecated, and Mythos (in the price table but
  in no default list) requires organization access. Verify access before
  spending on a sweep; pass Mythos's model ID explicitly if needed.

## Protocol

- Read all 500 rows from `../datasets/morse-500/test.csv` and videos from
  `../datasets/morse-500/test_sz512/` (paths relative to the repository root).
- Build timestamps at `0, 0.25, 0.50, ... < duration` (4 fps).
- When there are more than 32 timestamps, uniformly select 32 indices across
  that full timestamp grid, including the first and last candidates. Long clips
  therefore have a lower effective fps; this is **not** first-8-second truncation.
- Resize every selected frame to a longest side of at most 512 pixels without
  upscaling, preserving aspect ratio to integer-pixel rounding. Encode as JPEG
  at quality 90, then send images in chronological order, with the original
  prompt: `Answer the question in this video.`
- Questions remain embedded in the video. CSV question text and ground truth
  are **not** sent during prediction. There are no timestamps added to the prompt.
- OpenAI defaults to Chat Completions image inputs with `detail=high`, with an
  explicit Responses API option using the same image detail. Anthropic uses
  native Messages base64 image blocks. Choose model IDs supporting these APIs,
  image input, and at least 32 images per request. Batch scripts use their
  configured model lists unless you supply model IDs explicitly.
- The completion budget defaults to 4,096 tokens. For OpenAI reasoning models
  that budget includes hidden reasoning tokens: increase it if needed.
  OpenAI reasoning effort is omitted unless explicitly configured. Anthropic
  thinking settings are not overridden; the model's API default applies.
  Set consistent, documented settings when comparing models.
- Output-budget exhaustion, refusals, empty text, and API errors are failures,
  not valid predictions. The command exits nonzero if any rows failed.
  Transient HTTP errors are retried with backoff; input/output token limits
  are enforced by the provider, not by a local token-rate limiter.

The original `eval_model.py` and `eval_model_async.py` are unchanged.
This runner uses MoviePy 2 rather than the original scripts' MoviePy 1.
Use a separate environment for the original scripts if necessary.

## Setup

Every `eval/api` entry point (setup, validation, local, Slurm, estimate, and
recovery scripts) requires a configured shell session and fails immediately
(exit 3) otherwise. The scripts have no fallback paths of their own.

1. **Review and fill in** [setup/session_env.sh](setup/session_env.sh). It is the
   single place that defines these directories and paths. Every value in its
   review block ships empty and has no fallback; each must be an absolute path.
   Sourcing fails, before any directory is created or key is requested, while
   any value is empty or relative or (for outputs) its parent directory is missing.

   | Variable | You set | Typical value | Used for |
   | --- | --- | --- | --- |
   | `REPO` | yes | this `morse-500` checkout | code location |
   | `DATA_CSV` | yes | `<datasets>/morse-500/test.csv` | benchmark CSV (must exist) |
   | `VIDEO_ROOT` | yes | `<datasets>/morse-500/test_sz512` | benchmark videos (must exist) |
   | `VENV` | yes | `$REPO/.morse_venv` | API Python environment |
   | `RESULTS_ROOT` | yes | `$REPO/eval/results` | named run directories |
   | `LOG_DIR` | yes | `$REPO/logs` | Slurm job logs |
   | `FRAME_CACHE_ROOT` | yes | `$REPO/eval/api/frame_cache` | shared frame caches |
   | `AGENT_BIN` | yes | shared storage holding the harness binaries | staged harness binaries |
   | `API_DIR` | derived | `$REPO/eval/api` (must be this checkout) | code location |
   | `PYTHON` | derived | `$VENV/bin/python` | API Python interpreter |
   | `CODEX_BINARY`, `CLAUDE_BINARY`, `CODEX_CODE_MODE_HOST` | derived | under `$AGENT_BIN` | agent harnesses |

2. **Source** it from the repository root. (Sourcing, rather than running it
   with `bash`, keeps the variables in your shell.) It creates the final
   directory of `RESULTS_ROOT`, `LOG_DIR`, `FRAME_CACHE_ROOT`, and `AGENT_BIN`
   if missing (never their parents, so a mistyped path fails). It then
   checks that every path is set and exists, and prompts silently
   (`read -rsp`) for `OPENAI_API_KEY` and `ANTHROPIC_API_KEY`. Both keys are
   required. Keys are never echoed, written to shell history, or passed as
   command-line arguments.

```bash
source eval/api/setup/session_env.sh
bash eval/api/setup/setup_env.sh
bash eval/api/setup/validate.sh --mode model --provider openai
```

If sourcing fails, fix the reported paths and source the script again. Scripts
reject a session where a path was later unset, made relative, or removed, or where `API_DIR`
points to another checkout. Repeat step 2 in every new shell. Re-sourcing always
prompts for both keys again.

`setup/setup_env.sh` creates `VENV` if it does not exist (an existing
environment is reused, never replaced) and installs API dependencies into it;
rerun it to update dependencies. `setup/validate.sh` is check-only; it does not
install dependencies or send inference requests. For agent runs it also checks
Linux namespaces, `bwrap`, `socat`, native harness binaries, and the Codex
code-mode host.

### Agent mode only: stage the harness binaries

Agent mode additionally requires Linux, Bubblewrap (`bwrap`), `socat`, enabled
user/network namespaces, the official native Codex and Claude Code ELF binaries,
and the Codex `codex-code-mode-host` from the same release. Versions are pinned
in `setup/harnesses.lock`. Skip steps 1-2 if `AGENT_BIN` already holds the
pinned versions.

1. Download the binaries from the official
   [Codex releases](https://github.com/openai/codex/releases) and
   [Claude Code setup guide](https://code.claude.com/docs/en/setup), and verify
   release integrity. Do not use an npm wrapper or Claude's auto-updating binary.
2. With the session sourced, stage them into `AGENT_BIN`:

   ```bash
   CODEX_SOURCE=/path/to/codex \
   CLAUDE_SOURCE=/path/to/claude \
   CODE_MODE_HOST_SOURCE=/path/to/codex-code-mode-host \
     bash eval/api/setup/install_harnesses.sh
   ```

   The script checks native ELF format and the pinned versions, installs
   versioned copies plus `codex`, `claude`, and `codex-code-mode-host` symlinks
   (the paths `session_env.sh` derives as `CODEX_BINARY`, `CLAUDE_BINARY`, and
   `CODEX_CODE_MODE_HOST`), and records checksums in `AGENT_BIN/harnesses.sha256`.
3. On each compute node, check without inference requests:

   ```bash
   bash eval/api/setup/validate.sh --mode agent --provider openai
   bash eval/api/setup/validate.sh --mode agent --provider anthropic
   ```

- The runner never downloads, updates, or substitutes a harness. Keep the
  pinned binaries unchanged for the whole comparison.
- Every agent run also checks each model with a free metadata request and runs
  the harness `--version` inside the sandbox before any paid request.

### Model lists and run settings


Edit model lists in `eval/api/scripts/config.sh`. The lists are populated from the supplied estimates and verified
provider documentation; see [model IDs and verification](estimates/ESTIMATES.md#prices-and-model-ids).
Deprecated Codex is part of the agent track, and Sonnet is in the separate
Claude Code sweep. Restricted Mythos is not in the default direct sweep.
Model IDs in the batch lists must contain only letters, numbers, `.`, `_`, or `-`.
Use `run_models_eval.py` directly with a custom output filename for other ID formats.

Run defaults include `FPS=4`, `MAX_FRAMES=32`,
`MAX_SIDE=512`, concurrency, RPM, output-token budget, timeout, retry attempts,
preprocessing mode, reasoning effort, and OpenAI endpoint. Set them with
environment variables; `scripts/config.sh` contains the model lists. Positional model IDs override the
corresponding mode/provider list.
Cache paths, filenames, and log descriptions reflect the selected sampling values.
Keep 4/32/512 for the agreed protocol. Increasing `MAX_SIDE` cannot recover
resolution from the already-downsampled `test_sz512` videos; use original videos
via `VIDEO_ROOT` in `setup/session_env.sh` if you intend to evaluate a
higher-resolution protocol.

`setup/session_env.sh` loads keys without placing them in source code, shell
history, or command-line arguments.
Slurm normally exports the submitting shell's environment to the job, including
the session paths and keys. Follow
your cluster's secret-handling policy; if it disables environment propagation,
load keys inside the job using your approved secret mechanism. Do not use
`set -x`, print the environment, or commit keys.

## Submit provider sweeps

Run from the repository root (after sourcing `setup/session_env.sh`) locally or submit to Slurm:

```bash
bash eval/api/scripts/run_local.sh model openai
bash eval/api/scripts/run_local.sh model anthropic
bash eval/api/scripts/submit_slurm.sh model openai
bash eval/api/scripts/submit_slurm.sh model anthropic
bash eval/api/scripts/submit_slurm.sh agent openai      # Codex
bash eval/api/scripts/submit_slurm.sh agent anthropic   # Claude Code
```

Each provider run processes its model list sequentially; provider runs can run
in parallel. A model failure does not prevent later models in that job from
running, but the job ultimately exits nonzero. Requests within a model run
concurrently. The defaults are deliberately conservative: 2 workers and
5 requests/minute (RPM). That still does not guarantee compatibility with a
low input-token-per-minute tier.

Slurm defaults are 4 CPUs, 32 GiB RAM, and 3 days; no GPU/GRES is requested.
Set `SLURM_PARTITION`, `SLURM_ACCOUNT`, and `SLURM_QOS` as required by the
cluster. Submit wrappers write logs to `$LOG_DIR/morse-eval-JOBID.log`.
Override resources through `SLURM_CPUS`, `SLURM_MEMORY`, and `SLURM_TIME`.
Local runs use the same options and write directly to the configured output.

```bash
SLURM_PARTITION=my-partition SLURM_ACCOUNT=my-account SLURM_QOS=my-qos \
SLURM_TIME=2-00:00:00 RPM=20 CONCURRENCY=5 \
  bash eval/api/scripts/submit_slurm.sh model openai
```

`MODE` is `model` or `agent`; provider is `openai` or `anthropic`.
Agent mode maps OpenAI to Codex and Anthropic to Claude Code. The optional
model-ID arguments override the configured list.

For agent sweeps:

- Examples run **one at a time** (one harness session each) under the
  [per-example limits](#per-example-limits); the concurrency setting does not apply.
- Run a small billable pilot first, e.g.
  `LIMIT=2 RUN_NAME=agent_pilot_openai bash eval/api/scripts/run_local.sh agent openai`.
  Unit tests use mock responses, so CLI authentication, image attachments, and
  event-stream compatibility are only verified by a real run.
- The n=60 estimate pilots projected about 1.6-3.9 hours of harness time per
  model for 500 examples (about 14 hours for the five Codex models, 9 hours for
  the three Claude models). If every example hit the 30-minute limit, one model
  would need 250 hours; resubmit with `RESUME_RUN=RUN_NAME` to continue, or
  split models into separate jobs.

### Estimates and budgets

The pilot uses a stratified subset and makes provider requests:

```bash
bash eval/api/scripts/estimate_local.sh model openai gpt-6-astra
SUBSET_SIZE=60 bash eval/api/scripts/submit_estimate_slurm.sh agent anthropic
```

For a no-request arithmetic ceiling, run
`python eval/api/estimates/static_estimate.py model openai gpt-6-astra`.
To bootstrap an estimate before the full run and use its upper confidence bound
as the cap:

```bash
ESTIMATE=bootstrap BUDGET_SOURCE=estimate-upper BUDGET_POLICY=cap \
  BUDGET_MARGIN=1.2 bash eval/api/scripts/run_local.sh model openai gpt-6-astra
```

An existing estimate can be used with `ESTIMATE=file:/path/estimate.json`.
`BUDGET_USD` enables a shared worst-case-request reservation ledger;
`BUDGET_POLICY=cap` enforces it, `gate` stops before the full run when the
selected estimate exceeds `BUDGET_USD`, and `both` gates and caps. The gate
requires `BUDGET_SOURCE=estimate-point` or `estimate-upper`. For example:

```bash
ESTIMATE=file:eval/results/RUN_NAME/estimate/estimate.json \
BUDGET_SOURCE=estimate-upper BUDGET_POLICY=both BUDGET_USD=100 \
  bash eval/api/scripts/run_local.sh model openai gpt-6-astra
```

The budget uses standard price-table rates and is a conservative operational
cap, not the provider invoice. Pilot costs are billable and should themselves
be bounded with `BUDGET_USD` when appropriate.

Paths (`REPO`, `PYTHON`, `DATA_CSV`, `VIDEO_ROOT`, `RESULTS_ROOT`, `LOG_DIR`,
`FRAME_CACHE_ROOT`, `AGENT_BIN`, `CODEX_BINARY`, `CLAUDE_BINARY`, and
`CODEX_CODE_MODE_HOST`) come from `setup/session_env.sh`; edit them there.
Optional per-run environment variables:
`OUTPUT_DIR`, `RPM`, `CONCURRENCY`, `MAX_OUTPUT_TOKENS`, `REASONING_EFFORT`
(OpenAI only), `LIMIT` (pilot only), `PREPROCESS_MODE` (`cached` or `on-demand`),
`FRAME_CACHE` (defaults to a sampling-specific directory under `FRAME_CACHE_ROOT`),
`FPS`, `MAX_FRAMES`, `MAX_SIDE`, `TIMEOUT`, `ATTEMPTS`, and
`OPENAI_ENDPOINT` (`chat`, the existing default, or explicit `responses`).
Estimate/budget options are `ESTIMATE`, `BUDGET_USD`, `BUDGET_FILE`,
`BUDGET_SOURCE`, `BUDGET_POLICY`, and `BUDGET_MARGIN`.
For Codex, see the [opt-in Responses pilot](estimates/ESTIMATES.md#prices-and-model-ids).

## Two preprocessing approaches

With the default configuration, the run script explicitly passes
`--fps 4 --max-frames 32 --max-side 512` in **both** modes. Edited settings are
passed consistently to preparation and prediction. These are sampled image
inputs, not a video sent to the provider.
The 32-frame cap applies uniformly over the full clip's 4-FPS timestamp grid.

### 1. On-demand preprocessing for a model

```bash
PREPROCESS_MODE=on-demand \
  bash eval/api/scripts/run_local.sh model openai YOUR_MODEL_ID
```

Each pending clip is decoded, sampled, and resized immediately before its API
request. No frame cache is created. This is useful for one-off runs. If you pass
multiple models in this mode, preprocessing is repeated for each model.

### 2. Prepare once, reuse for all models (default)

```bash
bash eval/api/scripts/run_local.sh model openai YOUR_MODEL_1 YOUR_MODEL_2
bash eval/api/scripts/run_local.sh model anthropic YOUR_MODEL_1 YOUR_MODEL_2
```

Positional model IDs override the configured lists. Each run first prepares or
verifies the cache, then evaluates its models using only cached frames. Both
providers share `eval/api/frame_cache/fps4_max32_sz512` by default. A filesystem
lock serializes preparation across jobs; completed entries are written atomically.
Prediction only reads the cache: missing or incompatible entries fail before
paid requests rather than silently falling back to decoding.

You can also prepare explicitly without a key or provider/model selection:

```bash
.morse_venv/bin/python eval/api/lib/frames.py prepare \
  --fps 4 --max-frames 32 --max-side 512 --expected-count 500 \
  --frame-cache eval/api/frame_cache/fps4_max32_sz512

.morse_venv/bin/python eval/api/models/run_models_eval.py predict \
  --provider openai --model YOUR_MODEL_ID \
  --fps 4 --max-frames 32 --max-side 512 --expected-count 500 \
  --frame-cache eval/api/frame_cache/fps4_max32_sz512 \
  --output eval/results/RUN_NAME/pred_sz512_openai_YOUR_MODEL_ID_fps4_max32.csv
```

The direct-model evaluator also supports `eval/api/models/run_models_eval.py prepare`
with the same sampling and input options. Both model and agent run scripts use
the shared frame-preparation entry point.

The cache stores one JSON entry per clip containing base64 JPEG frames in
temporal order, sampling/encoding parameters, a frame-content checksum, and
the source video's resolved path, size, and modification time. It does not
write another MP4 or resample already processed frames. On-demand and cached
modes use the same sampling/encoding function and send identical image bytes
for unchanged source clips and dependencies.

Preparation resumes by verifying existing entries. Changed source files,
sampling settings, or corrupted entries are rejected explicitly; select a new
cache directory (or remove the specific damaged entry and prepare again).
Source files must remain available and unchanged for validation. Do not modify
videos/caches during evaluation. Keep the environment fixed to avoid decoder
differences. Base64 storage adds roughly one-third overhead to JPEG size, so
allow disk space for up to 16,000 frame images. The cache reduces repeated CPU
work, not provider image-token charges or model latency.

### Check compute-node connectivity

For authenticated key and connectivity verification, source
`setup/session_env.sh` and submit from the repository root:

```bash
bash eval/api/scripts/extras/submit_check_slurm.sh     # Check both providers
bash eval/api/scripts/extras/submit_check_slurm.sh openai
bash eval/api/scripts/extras/submit_check_slurm.sh anthropic --model YOUR_MODEL_ID
```

Or, inside an existing CPU allocation:

```bash
.morse_venv/bin/python eval/api/utils/check_api.py both
```

The check performs authenticated **GET model-list/model-metadata requests only**:
no inference, prompts, images, or input/output tokens are generated, so there are
no model-generation charges. Cluster resource/accounting policies still apply.
It logs the hostname/job ID, returns nonzero on failure, checks both providers
even if one fails, and never logs keys or raw provider error responses.
Missing keys, rejected credentials, permission errors, inaccessible models,
rate limits, and connection failures are reported explicitly.

A pass verifies connectivity and authorization to read model metadata. It does
**not** guarantee inference credits/billing, token quotas, API compatibility,
image support, or sufficient output budget. Verifying those requires a small
inference pilot, which **is billable**. Optional `--model` checks use metadata
retrieval and may reject aliases not exposed by the metadata endpoint even when
an inference endpoint accepts them.

For an unauthenticated network-only check instead:

Run on an allocated CPU node, not just the login node:

```bash
curl --connect-timeout 10 --max-time 20 -sS -o /dev/null \
  -w 'OpenAI HTTP %{http_code}\n' https://api.openai.com/v1/models
curl --connect-timeout 10 --max-time 20 -sS -o /dev/null \
  -X POST -H 'content-type: application/json' -H 'anthropic-version: 2023-06-01' \
  -d '{}' -w 'Anthropic HTTP %{http_code}\n' https://api.anthropic.com/v1/messages
```

No credentials are sent. An authentication/client error (typically 401) confirms
DNS/TLS/HTTP reachability, **not** that your key, model access, quota, or billing
works. A timeout, TLS failure, HTTP 000, or proxy denial needs cluster/network
investigation. A small authenticated pilot is still necessary.

### Pilot before spending on the entire benchmark

After checking the model lists:

```bash
LIMIT=10 RUN_NAME=pilot_openai \
  bash eval/api/scripts/submit_slurm.sh model openai
LIMIT=10 RUN_NAME=pilot_anthropic \
  bash eval/api/scripts/submit_slurm.sh model anthropic
```

The pilot evaluates the first 10 rows; it measures quota/latency rather than
benchmark accuracy and is not a representative difficulty sample. Examine logs
for retries, truncation, and token counts in CSV outputs. Increase the output
budget if reasoning consumes all available tokens. Then rerun a larger pilot
in a **new output directory** if generation settings change.

For a single model without editing a batch list (run on a CPU allocation):

```bash
.morse_venv/bin/python eval/api/models/run_models_eval.py predict \
  --provider openai --model YOUR_MODEL_ID --limit 10 \
  --fps 4 --max-frames 32 --max-side 512 \
  --output eval/results/pilot/pred_sz512_openai_YOUR_MODEL_ID_fps4_max32.csv
```

## Runtime planning

Exact per-model times cannot be inferred until model IDs, reasoning settings,
and account quotas are known. Provider names alone do not determine latency.
For each filled-in model, let:

- `N = 500`, `C = concurrency`, `R = RPM`;
- `L` be average API latency in seconds and `D` average frame-preparation time;
- `I`, `O` be average billable input/output tokens per example;
- `Qin`, `Qout` be the corresponding provider token-per-minute limits.

A useful approximate lower bound in **minutes** is:

```text
max(N * (L + D) / (60 * C), N / R, N * I / Qin, N * O / Qout)
```

Include only quota terms that apply to that model/account. Some providers have a
combined token limit, short-window enforcement, or shared model-family quotas.
Add roughly 20-50% planning margin for latency tails, retries, and overhead;
this is not an upper bound. Persistent quota/billing problems can prevent
completion entirely. The last request's latency also contributes to walltime.

These are **conditional scenarios, not measured model-specific predictions**:

| Assumed latency per request | At 5 RPM, 2 workers (script defaults) | At 20 RPM, 5 workers |
| --- | --- | --- |
| 5-15 seconds, short/fast answers | about 2-2.5 hours/model | about 30-40 minutes/model |
| 30-90 seconds, more reasoning | about 2.5-9.5 hours/model | about 1-4 hours/model |
| 2-5 minutes, heavy reasoning | about 10-32 hours/model | about 4-13 hours/model |

These ranges include the rough margin but assume token quotas do not dominate.
The default 300-second per-request timeout is unsuitable for models regularly
taking longer; set `TIMEOUT` for the wrapper or use `--timeout` with
`run_models_eval.py predict` directly. Retries can extend runtime. A 24-hour job
is not sufficient for every model/sweep.

Local preprocessing of 12 evenly spaced real benchmark clips was verified:
all yielded 32 frames, averaging **1.05 seconds/clip** (0.31-1.99 seconds).
That is about 9 minutes of serial CPU preprocessing for 500 clips before
threading overlap, not a full-dataset timing guarantee or an API latency measurement.
At 32 frames per clip there are up to **16,000 images per model run**.
In cached mode, add the one-time preparation cost to the sweep, then substitute
cache-read time for decoding time `D` in each model's estimate.

For a pilot at the intended concurrency/RPM, use the logged elapsed walltime
and token counts to refine estimates. Multiplying a 10-example time by 50 is
only a rough estimate: startup/drain time, difficulty, token usage, and API
latency vary. Model sweeps in each provider job take the **sum** of that
provider's model runtimes. If both jobs run together, overall time is roughly
the longer provider sweep, excluding queue waits and scoring.

## Checkpoints and outputs

Initialization creates `$RESULTS_ROOT/RUN_NAME/` (by default
`eval/results/RUN_NAME/`; examples below use the default) and its `run_usage/` directory
before preflight, frame preparation, or inference. If `RUN_NAME` is omitted,
a unique mode/provider/action timestamped name is generated and printed.
`PREFLIGHT_ONLY=1` does not initialize a run.

```bash
RUN_NAME=model-pilot LIMIT=2 bash eval/api/scripts/run_local.sh model openai MODEL_ID
RESUME_RUN=model-pilot LIMIT=2 bash eval/api/scripts/run_local.sh model openai MODEL_ID
```

The same environment variables work for agent, estimate, recovery, and Slurm
runs. `RESUME_RUN` must name an existing directory directly under `eval/results`.
An existing `RUN_NAME` or `OUTPUT_DIR` is rejected without explicit resume;
`OUTPUT_DIR` must be directly under `eval/results`, and must agree with any
run name supplied. Never run two writers against the same checkpoint.
Low-level evaluators accept `--run-name` and `--resume-run`, and generate a
fresh run if neither is supplied. Their optional `--output` must be inside
the selected run (a new run's output parent must be directly under `eval/results`).
Pilot estimates go under the run's `estimate/`; their ledgers go under
`run_usage/estimate/`. `ESTIMATE_DIR` must remain inside the run.
Any custom `BUDGET_FILE` must be inside that run's `run_usage/`.

Raw predictions are saved in `eval/results/RUN_NAME/pred_sz512_PROVIDER_MODEL_fps4_max32.csv`.
Each completed example is atomically checkpointed. CSVs include benchmark
columns, frame counts, provider-reported input/output tokens, and per-example
elapsed seconds (including preprocessing, quota waits, and retries).
Each output also has a `.meta.json` companion recording protocol parameters,
model/provider, and the input CSV hash.

Agent outputs (`pred_sz512_codex_MODEL_...csv`, `pred_sz512_claude_MODEL_...csv`) differ slightly:

- Rows also record `harness`, `status`, `requests`, and `denied_requests`;
  `elapsed_seconds` is the harness session's walltime.
- `input_tokens`/`output_tokens` are summed over all of the example's upstream
  requests, using the same rules as the usage report (Anthropic cache tokens
  count as input). They are blank only if no request reported usage.
- A `.traces.jsonl` file is written beside the CSV (see [Traces](#traces)).
- Failed rows are retried in fresh sessions on resume; `scripts/extras/retry_failed.sh`
  is for direct-model runs only.

Resubmitting with `RESUME_RUN=RUN_NAME` skips successful rows and retries failed/missing
rows. Keep the CSV and its metadata together. Changes to model, input, sampling,
or generation settings require a **new output path** to avoid mixing protocols.
Concurrency/RPM/retry/timeout settings may change when resuming. Do not run two
jobs that write the same output simultaneously. Outputs are not added to git.
The old locations `eval/api_results`, `eval/api_pilot`, and `eval/single_pilot`
are not moved automatically. Move compatible checkpoints and their metadata
into a named run under `eval/results` before resuming; move usage ledgers into
its `run_usage/` directory. Offline scoring can still accept an old directory.
Pre-refactor metadata lacks the new `frame_cache`
field: use new output filenames for this runner rather than mixing checkpoints.
The selected cache path is part of resume metadata, so switching preprocessing
mode or cache location also requires a new output path.

### Isolated recovery with a higher output-token budget

`scripts/extras/retry_failed.sh` uses `utils/prepare_failed_rows.py` to select
failed or missing rows from one model's original prediction CSV and writes the
retry input and results in a new named run under `eval/results/`. Set
`ORIGINAL_RESULTS` to the original run directory. It does not merge
or modify the original CSV. The recovery CSV contains only the selected rows,
in benchmark order, and retains their original benchmark indices. The default
recovery budget is **8,192 output tokens per request**; override it with
`MAX_OUTPUT_TOKENS` if needed. The original sweep's default was 4,096.

Submit one model per job from the repository root, using the matching API key:

```bash
ORIGINAL_RESULTS="$RESULTS_ROOT/RUN_NAME" MAX_OUTPUT_TOKENS=8192 \
  bash eval/api/scripts/extras/retry_failed.sh openai gpt-6-astra
```

These are billable inference requests. A recovery CSV is a separate pass with
a larger generation ceiling, so document it as such rather than treating it as
part of the original uniform-budget comparison. Resubmitting a recovery job
with `RESUME_RUN` and unchanged settings resumes its separate checkpoint.

## Local usage and cost tracking

Every inference attempt appends to `run_usage/OUTPUT_STEM.usage.jsonl` within
the run directory. Budget accounting defaults to `run_usage/budget.jsonl`.
These local files contain timestamp, unique event/run IDs,
hashed example ID, provider/model/stage/endpoint, response ID/resolved model,
API-call elapsed time, response status, and the provider's full usage object.
They do **not** contain keys, prompts, images, questions, or answers. No external
tracking service or additional dependency is required.

Usage is written **before** validating completion status, so truncated/refused
responses remain in accounting even when their prediction row is marked failed.
Retries and resumed runs append new events instead of overwriting previous
usage. Transport/HTTP errors have unknown usage rather than zero cost.
The API latency in the ledger excludes local preprocessing and rate-limit waits;
CSV per-example elapsed time includes them.

### Access the ledger while a job is running

By default, ledgers are in `eval/results/RUN_NAME/run_usage/` on the shared filesystem, so you
can inspect them from the login node without connecting to the compute node.
For example, GPT-6 Astra writes:

```text
eval/results/RUN_NAME/run_usage/pred_sz512_openai_gpt-6-astra_fps4_max32.usage.jsonl
```

The ledger appears **after the first API attempt finishes**, not immediately
when the Slurm job starts. Frame-cache preparation does not create an inference
ledger. From the repository root, follow new request records live:

```bash
tail -F eval/results/RUN_NAME/run_usage/pred_sz512_openai_gpt-6-astra_fps4_max32.usage.jsonl
```

`tail -F` waits for the file if it does not exist yet and follows appended JSON
lines. Each line contains one attempt's status, API-call timing, and available
token usage. Stop following with Ctrl-C; this does not stop the evaluation job.
If you set `OUTPUT_DIR` or change the sampling settings, adjust the directory
and filename accordingly.

### Generate cumulative usage and cost reports

Generate a report at any point, including while jobs are running (retry if you
read during an incomplete JSONL write):

```bash
.morse_venv/bin/python eval/api/estimates/usage.py eval/results/RUN_NAME \
  --output eval/results/RUN_NAME/run_usage/usage_summary.json
```

This searches prediction and extraction subdirectories and groups by
provider/model/stage. It totals known input/output tokens, request counts,
unknown-usage counts, and **standard-rate estimated USD** using
`estimates/prices.json`, transcribed from the 2026-10-19 estimate file. The report records
the price date/source. A custom `--prices PATH` can be used to reprice the same
immutable token ledger later. See [ESTIMATES.md](estimates/ESTIMATES.md) for
adding prices and generating estimates.

For terminal output refreshed every 60 seconds, run from the repository root:

```bash
watch -n 60 '.morse_venv/bin/python eval/api/estimates/usage.py eval/results/RUN_NAME'
```

The report includes all ledgers currently present under that directory, not
only the current Slurm job. Before the first ledger exists, the reporting command
reports that no ledgers were found; `watch` will try again on its next refresh.
Replace `eval/results/RUN_NAME` with your custom `OUTPUT_DIR` when applicable.
Ctrl-C stops monitoring without affecting the job.

The estimate is **not an invoice or guaranteed upper/lower bound**:

- It applies ordinary input/output rates to all known tokens. Cache discounts,
  cache-write premiums, long-context tiers, fast/flex/batch pricing, regional
  premiums, taxes, contracts, and tool charges are not modeled.
- OpenAI completion/output counts already include reported reasoning tokens:
  do not add them again. Cached tokens are already within OpenAI input totals.
  Anthropic cache-read/write tokens are separate and are added to input totals;
  this report prices them at ordinary input rates, not cache-specific rates.
- Unknown usage, unpriced models, and requests interrupted before receiving or
  durably recording a response are not recoverable from the ledger. Reports
  explicitly expose unknown/unpriced entries; the subtotal is **known usage only**.
- Old CSVs do not contain all historical attempts. The ledger starts with the
  updated runner and cannot retrospectively recover past billing.
- Keep ledgers across resumes. Do not combine directories from duplicated runs
  and assume CSV deduplication; report deduplication uses unique ledger event IDs.

Reconcile against provider billing dashboards for actual spend. The original
central estimates are arithmetically correct at their stated rates, but their
4M-input/125k-output assumptions are unmeasured and can badly underestimate
reasoning output. Run pilots before deciding budgets.

### W&B

W&B is unnecessary for metering: a local ledger is simpler, auditable, and does
not depend on compute-node access to another service. There is no W&B integration
or upload in this workflow. If you later want dashboards, use an approved local
or self-hosted deployment and export only approved aggregate metrics (tokens,
estimated USD, latency, completion count), not benchmark content or credentials.
Keep the ledger as the source of truth rather than relying on dashboard state.

## Extract answers and produce the benchmark table

After all raw predictions are complete, use the benchmark's original
`Qwen/Qwen2.5-72B-Instruct-AWQ` extractor with its ten-example prompt.
See the [local Qwen workflow](../qwen_eval/README.md)
for the separate environment, scratch download, and GPU requirements.
From the repository root, after installation and a successful download:

```bash
sbatch eval/qwen_eval/scripts/score_qwen.sbatch eval/results/RUN_NAME
```

Direct-model and agent runs are scored the same way.

This starts a loopback-only vLLM server on the allocated H200, waits for it to
be ready, and runs extraction/scoring before stopping it. For an already-running
local Qwen server on the same node:

```bash
bash eval/qwen_eval/scripts/score.sh eval/results/RUN_NAME
```

`qwen_eval/scripts/score.sh` requires a results directory, not hosted
provider/model arguments. It reads only `pred_sz*_*.csv` directly in that
directory, requires 500 rows per model, and calls
`qwen_eval/extract_answers.py` followed by `qwen_eval/plot_table.py`.
`CONCURRENCY` defaults to 10 local requests; no hosted keys,
RPM limit, or judge API charges are involved. It does not merge recovery CSVs.

The extractor sees question text and model response, **not ground truth**.
The original model, prompt, and temperature 0 are retained. The local extraction
answer budget is 512 tokens rather than the original 50, which truncated long
knot-ordering sequences. This does not change evaluated-model inference budgets
or incorporate recovery runs. Current failure policy: original refusals, token-limit failures,
`ERROR: ...` responses, and empty predictions count as incorrect with all
500 examples retained. Their extracted answers are empty, never the valid
answer `None`. This provisional policy must be noted alongside results.
Qwen extraction errors fail the scoring job rather than generating a table.

Outputs:

- `eval/results/RUN_NAME/extract/extract_sz512_*.csv`: predictions plus extracted answers.
- `eval/results/RUN_NAME/benchmark_scores.csv`: overall and six category accuracies.
- The same sorted table is printed to stdout/job log.

Scoring retains the original strict, case-sensitive, stripped-string exact
match and causal-to-temporal category merge. `plot_table.py` now preserves
literal `None` answers instead of treating them as pandas missing values.
Local extraction overwrites matching extraction files and does not resume.
Use separate result directories when comparing extraction configurations.
The low-level hosted `run_models_eval.py extract` command remains for compatibility,
but is not used by this workflow.

The scorer can only include files already present. Confirm every planned model
has finished before running it; there is no automatic Slurm dependency.

## Coding-agent harness track

Reference details for `MODE=agent`. None of this is needed to run the scripts
above, but it defines what an agent configuration measures.

### Per-example limits

Full-sweep values are set by `lib/run_eval.sh` (frozen to match the latest cost
estimate, `n60_seed0_tok8192`) and can be overridden with the listed variables.
Calling `agents/run_agents_eval.py` directly uses its own CLI defaults
(600 s, 10 requests, 4,096 tokens).

| Limit | Value | Variable |
| --- | --- | --- |
| Walltime for the harness process tree (startup, code execution, API latency, rate-limit waits) | 1,800 s | `SECONDS_PER_EXAMPLE` |
| Upstream inference requests, including SDK retries and compaction (not user turns; Claude also gets `--max-turns 30`) | 30 | `MAX_REQUESTS` |
| Output tokens per request, capped by the gateway; reasoning counts toward it | 8,192 | `MAX_OUTPUT_TOKENS` |
| Consecutive non-budget failures before aborting (exit 75); auth/model-not-found/network errors abort at once | 3 | `MAX_CONSECUTIVE_FAILURES` |
| Upstream requests per minute, shared across the job | 5 | `RPM` |

- Examples and models within a job run sequentially; different
  harness/provider jobs can run concurrently.
- There is no exact USD cap from these limits: up to 245,760 output tokens per
  example, plus repeatedly sent context, images, and tool outputs. Use
  `BUDGET_USD` and a provider-side spending limit.
- If reasoning output regularly truncates, raise the token budget for **all**
  agent configurations, document it, and use new run names.
- Report timeouts, request-budget exhaustion, provider refusals, and CLI
  failures separately from reasoning accuracy; never treat them as valid answers.

### Task and sandbox

- Each example gets a new process, session, empty home, and scratch directory
  (removed afterwards). Harnesses receive exactly the same JPEG bytes as the
  direct models, attached to the initial prompt and also available read-only as
  `/frames/00.jpg`, etc. Only frame indices are exposed, not video names.
- Agents may run code and write scratch files in `/work`, and may re-read, crop,
  or analyze the frames. They cannot access the original video, CSV, ground
  truth, repository, real home, other examples, or external websites.
- The prompt differs from direct inference only to describe image ordering and
  allowed code execution. Harness system prompts and tool loops are part of the
  evaluated configuration; this is not bare-model inference.
- The entire harness runs inside Bubblewrap with private user/PID/network
  namespaces. It sees read-only system executables and libraries under `/usr`
  and `/lib*`, the selected CLI binary, the example's frames, and its scratch
  directory. No user settings, plugins, credentials, MCP config, or skills are mounted.
- Unattended shell permissions are enabled **only inside this sandbox**; do not
  copy them into an unsandboxed harness invocation. Claude is restricted to
  Bash/Read/Write/Edit/Glob/Grep with an empty MCP configuration; Codex has web
  search and subagents disabled. No model fallback is permitted.
- The evaluator's Python environment is not mounted, and packages cannot be
  installed from inside. On this host, system Python has NumPy but not Pillow;
  keep system libraries constant across compute nodes, or provision a common
  tool runtime explicitly.

### API gateway

- The sandbox has no external network interface. A local relay reaches only a
  host-side gateway, which accepts the provider's inference endpoint only
  (`/v1/messages` or `/v1/responses`), pins the model, blocks server-side
  web/MCP tools and remote image URLs, and enforces the request, time, and
  output-token limits.
- The real API key is injected by the gateway, outside the sandbox; the agent
  sees a nonfunctional placeholder. Model metadata, login, telemetry, update,
  and file-upload endpoints are blocked. If a future CLI version needs another
  endpoint, investigate rather than enabling unrestricted networking.
- Shell commands may call the local gateway directly, but those requests are
  pinned, metered, and counted against the same example limits.
- The gateway meters provider responses (including streamed usage) directly and
  writes them to the run's usage ledger; it does not also add the harness's own
  usage totals. A walltime kill cannot cancel accepted upstream work, and usage
  on killed or failed streams can be partial or unknown.
- The native binary and system tools must be trusted; this is not a hardened
  multi-tenant service for arbitrary malicious binaries.

### Traces

Each attempt (successes, failures, and timeouts with partial output) appends one
JSON line to `pred_...traces.jsonl` beside the CSV:

- `idx`, `video`, `status`, `error`, `exit_code`, request counts, and elapsed time;
- `events`: the harness's full event stream (Claude `stream-json` / Codex
  `exec --json`: assistant text, visible reasoning summaries, tool calls and outputs);
- `stderr_tail`: the last 20,000 characters of stderr.

Traces reuse output the harness already emits (no extra requests) and are not
part of `meta.json`, so they never block a resume. Embedded images are replaced
by `<image frame N>` (matching `/frames/NN.jpg`) or
`<image omitted sha256=... bytes=...>`. Expect tens of MB per full sweep;
`run_agents_eval.py --no-traces` disables them.

### USD budget enforcement

`BUDGET_USD` (see [Estimates and budgets](#estimates-and-budgets)) uses the same
`lib/budget.py` ledger for agent and direct-model requests:

- Before forwarding a request, the gateway reserves its worst-case cost (JSON
  text at 3 characters/token, 2,000 tokens per image, plus the full output cap,
  at `estimates/prices.json` rates) under a file lock that works across nodes on
  NFS. A request that would exceed the cap is refused.
- After the response, the reservation is settled to the metered cost. It stays at
  the worst case if usage may be partial (stream cut, timeout, killed job) and
  drops to $0 if the provider refused the request. Spend therefore stops within
  about one worst-case request (about $0.05-1.50) **below** the cap.
- When the cap is reached, the example is checkpointed as failed
  (`ERROR: USD budget ...`), `run_agents_eval.py` exits 76, and the job skips
  the remaining models. To continue, resubmit with a higher **total**
  `BUDGET_USD` and `RESUME_RUN`; interrupted examples are retried.

```bash
# Committed spend so far: (settled USD, still-reserved USD)
"$PYTHON" -c "import sys; sys.path.insert(0, 'eval/api'); \
from lib.budget import Budget; \
print(Budget('eval/results/RUN_NAME/run_usage/budget.jsonl', 1e9, 'openai', 'gpt-6-astra').spent())"
```

The original low/central/high cost totals describe ten single-request-equivalent
runs. Agent histories, system prompts, tool outputs, repeated frames, reasoning,
compaction, retries, and caching make them unsuitable as harness budgets; use
the agent estimate pilots instead.

### Harness preflight without inference

From an allocated CPU node with the session sourced (no paid requests):

```bash
# Runs the harness --version inside the sandbox.
"$PYTHON" eval/api/agents/run_agents_eval.py --harness claude \
  --model claude-opus-5-5 --binary "$CLAUDE_BINARY" --preflight-only
"$PYTHON" eval/api/agents/run_agents_eval.py --harness codex \
  --model gpt-5.3-codex --binary "$CODEX_BINARY" --preflight-only
```

The unit tests (below) exercise real namespace isolation and gateway accounting
with mock provider responses. An unsupported binary, flag, dependency,
namespace permission, or endpoint fails the run explicitly.

### References

- [Claude Code CLI flags](https://code.claude.com/docs/en/cli-reference)
- [Claude Code programmatic execution](https://code.claude.com/docs/en/headless)
- [Claude Code sandbox boundaries](https://code.claude.com/docs/en/sandboxing)
- [Codex non-interactive execution and JSON events](https://developers.openai.com/codex/noninteractive)
- [Codex model provider configuration](https://developers.openai.com/codex/config-reference)

## Local validation (no paid API calls)

```bash
.morse_venv/bin/python -m unittest discover -s eval/api/tests -p 'test_*.py' -v
bash -n eval/api/lib/*.sh eval/api/scripts/*.sh eval/api/scripts/*.sbatch \
  eval/api/scripts/extras/*.sh eval/api/scripts/extras/*.sbatch \
  eval/api/setup/*.sh eval/qwen_eval/scripts/score.sh
```
