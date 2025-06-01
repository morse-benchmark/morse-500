# Physical Commonsense Evaluation

This project tests whether Vision Language Models (VLMs) can reason about physical phenomena by distinguishing between real videos and AI-generated videos. The generated videos often contain artifacts and unphysical behaviors that should be detectable by models with good physical reasoning capabilities.

## Overview

We use real image frames from the [Physics IQ Benchmark](https://physics-iq.github.io/) to generate video completions, then test VLMs' ability to identify which videos follow realistic physics versus which contain AI-generated artifacts.

## Project Structure

```
physical-commonsense/
├── frames_selected/          # Selected video frames for generation
├── gen_veo2.py              # Video generation using Gemini API
├── get_gt_video.py          # Extract ground truth video clips
├── make_questions.py        # Generate evaluation questions
└── README.md               # This file
```

## Setup

### Download Physics IQ Benchmark Dataset

First, download the benchmark dataset following the instructions at:
https://github.com/google-deepmind/physics-IQ-benchmark


## Usage

### Step 1: Select Frames

The `frames_selected/` folder contains some interesting video frames from the Physics IQ dataset. 

**Frame naming convention:**
```
{scenario_id}_switch-frames_anyFPS_perspective-{view}_trimmed-{description}.jpg
```

Example:
```
0013_switch-frames_anyFPS_perspective-left_trimmed-ball-in-basket.jpg
0020_switch-frames_anyFPS_perspective-center_trimmed-ball-ramp.jpg
```

You can:
- Use the existing selected frames
- Add your own selections from the Physics IQ dataset
- Capture and add your own video frames

### Step 2: Generate Videos

Run the video generation script:

```bash
python gen_veo2.py
```

This script:
- Reads frame descriptions from `descriptions/descriptions.csv`
- Matches frames in `frames_selected/` with their descriptions
- Calls the Gemini API (Veo 2.0) to generate video completions
- Implements rate limiting (2 requests/minute) to respect API limits
- Saves generated videos with naming: `{scenario_id}_veo2.mp4`

**Note:** You can replace the Gemini API calls with other video generation models by modifying `gen_veo2.py`.

### Step 3: Extract Ground Truth Videos

Generate corresponding ground truth video clips:

```bash
python get_gt_video.py
```

This script:
- Clips the original videos from the Physics IQ dataset
- Ensures ground truth videos start with the same frames used for generation
- Creates matching video pairs for comparison

**Requirements:** Make sure you have downloaded the complete Physics IQ benchmark dataset first.

### Step 4: Create Evaluation Questions

Generate questions for VLM evaluation:

```bash
python make_questions.py
```

This creates evaluation questions that test the model's ability to:
- Identify which video follows realistic physics
- Detect unphysical behaviors in generated videos
- Reason about physical phenomena and causality

#### The Evolving Challenge: Multimodal Reasoning vs. Generation Capabilities
This evaluation framework represents a **dynamic benchmark** that pits multimodal reasoning capabilities directly against video generation capabilities. As video generation models become increasingly sophisticated, this creates a naturally escalating challenge:

**Current State**: Today's video generation models produce detectable artifacts and physical inconsistencies that can be identified by careful observation.

**Future Evolution**: As generation models approach photorealistic quality, the evaluation becomes progressively more demanding, requiring increasingly nuanced physical reasoning to distinguish reality from synthesis.

**Long-term Vision**: This framework embodies a fundamental tension in AI development - the race between generation and detection capabilities. It serves as both:

- A Moving Target Benchmark: The difficulty automatically scales with advances in video generation technology
- A Physical Reasoning Probe: It tests deep understanding of physical laws, not just pattern recognition
- A Reality Grounding Test: It evaluates whether models truly understand the physical world or merely mimic learned associations


**The Ultimate Test**: When video generation becomes indistinguishable from reality to human observers, only AI systems with deep physical understanding will be able to detect the subtle violations of physical laws that persist in synthetic content. This evaluation framework will identify which models possess true physical reasoning versus sophisticated pattern matching.
This represents a **co-evolution** of capabilities - as generators improve, the bar for reasoners rises, creating a continuous cycle of advancement that pushes both generation and reasoning to new heights.


## Expected Outputs

After running all scripts, you should have:

1. **Generated videos**: `{scenario_id}_veo2.mp4` - AI-generated completions
2. **Ground truth videos**: Corresponding real video clips from the dataset  
3. **Evaluation questions**: Questions testing physical reasoning capabilities

## Evaluation

The evaluation tests whether VLMs can:

- **Distinguish reality from AI**: Identify which video is real vs generated
- **Detect physical violations**: Spot unrealistic physics in generated videos
- **Understand causality**: Reason about cause-and-effect relationships
- **Apply physical intuition**: Use common-sense physics knowledge

## Customization

### Adding New Scenarios

1. Add new frame images to `frames_selected/`
2. Update `descriptions/descriptions.csv` with corresponding descriptions
3. Run the generation pipeline

### Using Different Video Models

Replace the API calls in `gen_veo2.py` with your preferred video generation model:

```python
# Replace this section with your model's API
operation = client.models.generate_videos(
    model="your-model-name",
    prompt=description,
    image=image_obj,
    config=your_config
)
```

### Modifying Rate Limits

Adjust the rate limiter in `gen_veo2.py`:

```python
# Change from 2 requests/minute to your desired limit
rate_limiter = RateLimiter(max_requests=5, time_window_minutes=1)
```

## Research Applications

This framework can be used for:

- **VLM Evaluation**: Testing physical reasoning capabilities
- **Video Generation Assessment**: Evaluating realism of AI-generated videos
- **Physics Understanding**: Probing model knowledge of physical laws
- **Benchmark Development**: Creating new evaluation datasets

## Dataset Availability

### Generated Videos Dataset

To facilitate further research in physical commonsense reasoning and video generation evaluation, we are making all generated videos publicly available on Hugging Face:

🤗 **[Dataset Link: Physical Commonsense Videos](https://huggingface.co/video-reasoning/physical-commonsense-videos)**

### Dataset Contents

The dataset includes:

- **Generated Videos**: AI-generated video completions using Veo 2.0
- **Ground Truth Videos**: Corresponding real video clips from Physics IQ Benchmark  
- **Frame Images**: Selected frames used as input for video generation
- **Descriptions**: Text prompts used for video generation
- **Metadata**: Scenario information, categories, and generation parameters

## Citation

If you use this work, please cite:

- The [Physics IQ Benchmark](https://physics-iq.github.io/) for the original dataset

## Contributing

Feel free to:
- Add new interesting scenarios from the Physics IQ dataset
- Implement additional video generation models
- Enhance the evaluation questions
- Improve the analysis pipeline

## License

This project builds upon the Physics IQ Benchmark. Please respect the original dataset's licensing terms.