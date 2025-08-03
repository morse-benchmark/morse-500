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
python extract_answers.py pred_o3.csv

# 3️⃣ Compute benchmark scores / generate table
python plot_table.py
```


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
│   ├── extract_answers.py
│   ├── plot_table.py
│   └── run.sh
├── manim-algebraic-reasoning   # code for generating algebraic-reasoning videos using manim library
└── xxx                         # other code for generating videos of other categories
```



## 📝 Citation
If you use MORSE-500, please cite:
```

```


<!-- ## 📄 License
- Code – MIT
- Dataset (videos + metadata) – CC BY-4.0 (attribution required)  -->
