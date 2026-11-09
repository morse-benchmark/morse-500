# MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning  
**Code & Evaluation Suite**

**[Project Website](https://morse-500.github.io/)** · **[Hugging Face Dataset](https://huggingface.co/datasets/video-reasoning/morse-500)** · **[HF Dataset Viewer](https://huggingface.co/datasets/video-reasoning/morse-500-view)**

---

## ✨ Key Features

| Aspect | Details |
| --- | --- |
| **Fresh & Portable** | 500 newly cooked video clips + CSV metadata that runs fast |
| **Scalable Difficulty** | Videos are generated programmatically so we can dial up complexity and release harder versions as models improve |
| **Diverse Categories** | *Abstract, Mathematical, Physical, Planning, Spatial, Temporal (+ Causal)* |
| **Pure Visual Reasoning** | Questions are baked right into the videos. No text crutches, no shortcuts – if you can't see it, you can't solve it |
| **Developer-Friendly** | A “[-view](https://huggingface.co/datasets/video-reasoning/morse-500-view)” subset streams directly on **Hugging Face** for quick browsing and debugging |

---

## ⚡ Quick Start

```bash
# Clone repo & install dependencies
git clone https://github.com/morse-benchmark/morse-500.git
cd morse-500

# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Create env and install packages
uv venv .morse_venv
source .morse_venv/bin/activate
uv pip install datasets pandas numpy pillow "moviepy==1.0.3" tqdm openai google-genai
# if also need video generation
uv pip install manim opencv-python maze-dataset "gymnasium[toy-text]"


# download data
git clone https://huggingface.co/datasets/video-reasoning/morse-500
cd morse-500
unzip test_sz512.zip -d test_sz512
cd ..


cd eval
# 1️⃣ Run a baseline model
python eval_model.py
python eval_model_async.py # or run async

# 2️⃣ Extract answers from model output
python qwen_eval/extract_answers.py pred_o3.csv

# 3️⃣ Compute benchmark scores / generate table
python qwen_eval/plot_table.py
```

### API evaluation regime: models and agents

The API evaluation regime supports direct OpenAI/Anthropic models and the
official Codex/Claude Code agent harnesses as separate tracks. Both can run
locally on CPUs or through Slurm, with configurable sampling, resumable
predictions, cost estimates, and optional budgets. Agent isolation requires
Linux. Provider inference is billable; local Qwen answer extraction is separate.
Start with the [evaluation guide](eval/README.md) or see the
[API workflow](eval/api/README.md) for setup and detailed options.

### Local Qwen answer extraction (Nexus)

All local Qwen setup, validation, extraction, scoring, and Slurm entry points
live in [`eval/qwen_eval/`](eval/qwen_eval/README.md). This workflow serves
`Qwen/Qwen2.5-72B-Instruct-AWQ` with vLLM on the allocated GPU and uses the
original ten-example extraction prompt before strict exact-match scoring.
**Hosted, billable APIs are used only for models being evaluated, not for
extraction.** The local OpenAI-compatible interface does not contact OpenAI.

Run the following from the repository root on Nexus:

```bash
# Separate Python 3.12 environment: do not modify the CPU/API environment.
bash eval/qwen_eval/setup/setup_env.sh
export PYTHON="$PWD/.qwen_venv/bin/python"
bash eval/qwen_eval/setup/validate.sh --stage setup

# Shared scratch is visible from login and compute nodes.
export HF_HOME=/fs/nexus-scratch/dgjersta/.cache/huggingface

# Download on a CPU compute node; approximately 40-45 GB of model weights.
mkdir -p logs
sbatch eval/qwen_eval/scripts/download_qwen.sbatch
```

Wait for the download job to finish successfully (see
`logs/morse-qwen-download-JOBID.log`), then submit:

```bash
sbatch eval/qwen_eval/scripts/score_qwen.sbatch
# Optional other prediction directory, e.g. the separate coding-agent track:
# sbatch eval/qwen_eval/scripts/score_qwen.sbatch eval/api/agent_results
```

The scoring job requests **one H200 GPU** and owns a loopback-only Qwen server
on its compute node. The [Qwen guide](eval/qwen_eval/README.md) covers CUDA,
cache and server preflight checks, manual GPU execution, output paths, and the
provisional failed-prediction policy. Empty/`ERROR:` predictions remain
incorrect in the benchmark denominator; extraction errors fail the job.
Extraction still uses temperature 0 and 512 tokens, without sending ground
truth to Qwen or changing evaluated-model inference budgets.


## 📂 [Dataset](https://huggingface.co/datasets/video-reasoning/morse-500) Format

```
morse-500/
├── test.csv            # metadata, columns: id, video, query, question_text, ground_truth, category
├── test.zip            # videos of original size
├── test_sz512.zip      # videos with long side resized to 512 while keeping original aspect ratio
├── test/               # After unzip, each mp4 file corresponds to "video" column in test.csv
│   ├── xxx.mp4
│   ├── xxx.mp4
│   └── …
└── README.md
```

## 🏛️ Repository Layout
```
morse-500-code/
├── eval/                       # evaluate model, extract answers, plot tables
│   ├── pred/                   # contains prediction from differet models
│   ├── extract/                # contains extraction using Qwen2.5 72B Inst AWQ
│   ├── eval_model.py
│   ├── qwen_eval/              # local GPU extractor, scoring, setup, and jobs
│   │   ├── extract_answers.py
│   │   ├── plot_table.py
│   │   ├── requirements.txt
│   │   ├── setup/
│   │   ├── scripts/
│   │   └── tests/
│   └── run.sh
├── manim-algebraic-reasoning   # code for generating algebraic-reasoning videos using manim library
└── xxx                         # other code for generating videos of other categories
```



## 📝 Citation
If you use MORSE-500, please cite:
```
@article{cai2025morse500,
  title={MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning},
  author={Cai, Zikui and Wang, Andrew and Satheesh, Anirudh and Nakhawa, Ankit and Jae, Hyunwoo and Powell, Keenan and Liu, Minghui and Jay, Neel and Oh, Sungbin and Wang, Xiyao and Liang, Yongyuan and Goldstein, Tom and Huang, Furong},
  journal={arXiv preprint arXiv:2506.05523},
  year={2025}
}
```


<!-- ## 📄 License
- Code – MIT
- Dataset (videos + metadata) – CC BY-4.0 (attribution required)  -->
