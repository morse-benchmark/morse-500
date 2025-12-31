# Repository Guidelines

## Project Structure & Module Organization
This repo pairs dataset assets with evaluation and generation utilities.
- `eval/`: evaluation scripts (`eval_model.py`, `eval_model_async.py`), answer extraction, and table plotting.
- `questions/`, `question_text/`, `solutions/`: source question prompts and ground-truth answers.
- `reasoning_traces/` and category folders like `abstract_reasoning/`, `temporal_reasoning/`, etc.: generated artifacts and per-category assets.
- `analysis/`: analysis outputs and helper data.
- `test_bounceball_trace.py`: small script for inspecting generated traces.

## Build, Test, and Development Commands
This is a Python-first repo; there is no build step.
- `uv venv .morse_venv && source .morse_venv/bin/activate`: create and enter a local virtualenv.
- `uv pip install datasets pandas numpy pillow "moviepy==1.0.3" tqdm openai google-genai`: install eval dependencies.
- `python eval/eval_model.py`: run a baseline evaluation (edit the API key inside the script).
- `python eval/eval_model_async.py`: async variant for faster throughput.
- `python eval/extract_answers.py pred_o3.csv`: parse model outputs into a clean CSV.
- `python eval/plot_table.py`: compute scores and render tables.

## Coding Style & Naming Conventions
- Python code uses 4-space indentation and standard PEP 8 naming (snake_case for functions/vars).
- Keep scripts small and task-focused; prefer adding new utilities under `eval/` or the matching category folder.
- No formatter/linter is configured; keep style consistent with neighboring files.

## Testing Guidelines
- There is no formal test framework wired in.
- Use `python test_bounceball_trace.py` to inspect reasoning traces for the bounceball generator.
- If you add tests, keep them lightweight and runnable via `python <file>.py`.

## Commit & Pull Request Guidelines
- History uses short, plain-English messages (e.g., "update", "remove old files"); keep commits concise and scoped.
- PRs should describe the dataset or evaluation change, include the command(s) you ran, and note any new dependencies.
- Add sample outputs or CSV snippets when changing evaluation or extraction logic.

## Data & Configuration Notes
- Dataset videos are expected under `test/` or `test_sz512/` with metadata in `test.csv` (see README for the layout).
- Model evaluation scripts require API credentials; update the placeholder key in `eval/eval_model.py` (and async variant) before running.
