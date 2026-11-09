# Local Qwen extraction and scoring

This directory owns the self-hosted Qwen workflow: environment setup,
validation, model download, GPU serving, answer extraction, and exact-match
scoring. The former entry points outside this directory have been removed;
there are no compatibility wrappers. Prediction and extraction data have not
moved.

The extractor is `Qwen/Qwen2.5-72B-Instruct-AWQ`, served by vLLM on
**127.0.0.1:8000 on the same GPU compute node as the client**. The
OpenAI-compatible endpoint is only a local interface; extraction does not
contact OpenAI or require provider credentials. Serving Qwen requires a GPU.
Tabulating already-extracted CSVs is CPU-only.

## Layout

```text
qwen_eval/
├── extract_answers.py
├── plot_table.py
├── requirements.txt
├── setup/
│   ├── setup_env.sh
│   ├── validate.sh
│   └── validate.py
├── scripts/
│   ├── download_qwen.sbatch
│   ├── score_qwen.sbatch
│   └── score.sh
└── tests/
    ├── test_extract_answers.py
    └── test_setup.py
```

## Setup and model download (Nexus)

Run commands from the repository root. Install `uv` using your approved
package-manager process first. Keep the Qwen environment separate from the
CPU/API environment; the setup script uses Python 3.12.

```bash
bash eval/qwen_eval/setup/setup_env.sh
export PYTHON="$PWD/.qwen_venv/bin/python"
bash eval/qwen_eval/setup/validate.sh --stage setup

# Shared scratch is visible from login and compute nodes.
export HF_HOME=/fs/nexus-scratch/dgjersta/.cache/huggingface
mkdir -p logs
sbatch eval/qwen_eval/scripts/download_qwen.sbatch
```

`VENV` overrides the setup location. For a custom environment, also set
`PYTHON` to its `bin/python` for validation and jobs; do not inherit an API
environment's interpreter. The download job uses the `hf` executable alongside
that interpreter and downloads approximately 40-45 GB of weights on a CPU
compute node.

All Slurm logs go under `logs/`. Submit from the repository root and create
the directory **before** `sbatch`; Slurm opens the log before running the
script. Wait for a successful download in
`logs/morse-qwen-download-JOBID.log` before submitting scoring. The scorer does
not automatically add a dependency on the download or prediction jobs.

## GPU scoring job

```bash
sbatch eval/qwen_eval/scripts/score_qwen.sbatch
# Separate coding-agent predictions:
sbatch eval/qwen_eval/scripts/score_qwen.sbatch eval/results/RUN_NAME
```

The Nexus job requests **one H200 GPU**, 8 CPUs, 96 GiB host RAM, and four
hours on `cml-furongh` with `cml-high_long` QoS. The 72B AWQ model fits the
H200's 141 GB memory. Other GPU types may require changing the allocation and
tensor-parallel settings, not just the GPU name.

The job sets `CUDA_HOME=/opt/common/cuda/cuda-13.0.2` and adds its `bin`
directory and the Qwen environment's `bin` directory to `PATH`, allowing
FlashInfer to find `nvcc` and Ninja. Override `CUDA_HOME` only with a complete
compatible toolkit. A CUDA-enabled PyTorch wheel does not supply the compiler
on `PATH`. The pinned vLLM wheel also needs a compatible NVIDIA driver.

The job validates the GPU and cached files, starts vLLM with offline model
loading, waits up to 30 minutes for health, checks the advertised model, and
then extracts and scores. It stops its own server on exit. No network-facing
service or provider API key is required.

## Preflight checks

```bash
bash eval/qwen_eval/setup/validate.sh --stage setup
# Inside an appropriate GPU allocation, before serving:
bash eval/qwen_eval/setup/validate.sh --stage gpu
# After the server is ready, on that same node:
bash eval/qwen_eval/setup/validate.sh --stage server
```

- `setup` checks installed Qwen dependencies and imports the client/data
  packages. It reports package versions without importing vLLM on a CPU node.
- `gpu` additionally checks `nvcc`, Ninja, CUDA device availability, and the
  cached configuration, tokenizer, weight index, and nonempty weight shards.
  Cache lookup is local-only; it never downloads missing files.
- `server` additionally checks loopback `/health` and `/v1/models`, bypassing
  HTTP proxies and requiring the expected Qwen model ID.

Preflights make no inference requests. Cache checks establish file presence,
not checksums or successful model loading; GPU checks report memory, not a
guarantee that the model will fit. Confirm actual vLLM startup in the job log
and use a small GPU extraction pilot before a full run.

## Using an existing local server

Inside an appropriate GPU allocation, from the repository root:

```bash
export PYTHON="$PWD/.qwen_venv/bin/python"
export HF_HOME=/fs/nexus-scratch/dgjersta/.cache/huggingface
export HF_HUB_OFFLINE=1
export CUDA_HOME=/opt/common/cuda/cuda-13.0.2
export PATH="$(dirname "$PYTHON"):$CUDA_HOME/bin:$PATH"
bash eval/qwen_eval/setup/validate.sh --stage gpu
"$PYTHON" -m vllm.entrypoints.cli.main serve Qwen/Qwen2.5-72B-Instruct-AWQ \
  --served-model-name Qwen/Qwen2.5-72B-Instruct-AWQ \
  --host 127.0.0.1 --port 8000 --dtype half \
  --max-model-len 32768 --max-num-seqs 10 \
  --gpu-memory-utilization 0.85 --generation-config vllm
```

In another shell on that **same node**, after the server is ready:

```bash
export PYTHON="$PWD/.qwen_venv/bin/python"
bash eval/qwen_eval/setup/validate.sh --stage server
bash eval/qwen_eval/scripts/score.sh eval/results/RUN_NAME
```

Stop the manually started server when finished. `score.sh` uses an existing
server and does not start or stop it.

## Inputs, outputs, and benchmark policy

`score.sh` requires a results directory, typically `eval/results/RUN_NAME`.
It processes `pred_sz*_*.csv` directly in that directory,
requires 500 rows per file, and does not include nested recovery runs. Required
prediction columns are `video`, `ground_truth`, `question_text`, and
`prediction`; video identifiers must be unique. `CONCURRENCY` defaults to 10
local extraction requests.

Outputs remain inside the selected results directory:

- `extract/extract_sz*_*.csv`, with `extracted_answer`.
- `benchmark_scores.csv`, with overall and six category accuracies.
- The same sorted table on stdout, captured in
  `logs/morse-qwen-score-JOBID.log` for the GPU job.

The original ten-example prompt, model, and temperature 0 are retained.
Ground truth stays in the CSV but is never sent to Qwen. The local extraction
budget is **512 tokens**, increased from the original 50 to avoid truncating
long comma-separated knot-ordering answers. This is a documented extraction
deviation, not an increase to any evaluated model's inference budget.

**Provisional original-budget failure policy:** empty predictions and failures
recorded as `ERROR: ...` (including refusals and output-token exhaustion) count
as incorrect. The scoring wrapper enables this policy; these rows remain in
the benchmark denominator and are not sent to Qwen. Their extracted answers
are empty, not the valid answer `None`. Report this policy alongside scores.
The low-level extraction CLI requires `--allow-failed-predictions` to opt in.

Qwen request errors, empty answers, or truncated responses fail extraction
after saving partial output and prevent a new score table. Reruns overwrite
matching extraction files and do not resume. Use separate result directories
for comparisons and never run concurrent writers in the same extraction
directory. Only use a table from a successful run; a failed rerun does not
delete an earlier table.

Scoring preserves stripped, case-sensitive exact matching, literal `None`,
the existing invalid-ground-truth filtering, and the causal-to-temporal
category merge. The separate hosted extraction command in the API runner is
not used by this workflow.

For a small extraction pilot, use the low-level CLI with an explicit output:

```bash
"$PYTHON" eval/qwen_eval/extract_answers.py pilot_predictions.csv \
  --out pilot_extracted.csv --concurrency 1
```

This does not require 500 rows and is not a full benchmark score. To tabulate
existing extractions without serving Qwen:

```bash
"$PYTHON" eval/qwen_eval/plot_table.py \
  --input-dir eval/results/RUN_NAME/extract --output eval/results/RUN_NAME/benchmark_scores.csv
```

## CPU-only implementation tests

```bash
.qwen_venv/bin/python -m unittest discover \
  -s eval/qwen_eval/tests -p 'test_*.py' -v
bash -n eval/qwen_eval/setup/*.sh eval/qwen_eval/scripts/*.sh \
  eval/qwen_eval/scripts/*.sbatch
```

Tests use mocked model responses and GPU/cache/server checks, plus a 500-row
failed-prediction workflow that makes no extraction requests. They do not
download weights, submit Slurm jobs, or establish real GPU compatibility or
model-loading success.
