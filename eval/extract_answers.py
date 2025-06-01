import argparse
import asyncio
from tqdm.asyncio import tqdm_asyncio    # tqdm wrapper for asyncio
from openai import AsyncOpenAI
import pandas as pd
from datasets import load_dataset


### API Setup
# Set OpenAI's API key and API base to use vLLM's API server.
model_name = "Qwen/Qwen2.5-72B-Instruct-AWQ"
openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"

client = AsyncOpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
# test
# curl -X GET http://localhost:8000/v1/models
### DONE


demo_prompt = """
Please read the following example. Then extract the answer from the model response and type it at the end of the prompt.

Example 1
Question: How many safe paths exist from the start to the goal?  
Please answer with a single integer (e.g., 3).

Model response: The 10 × 10 grid has 480 distinct paths that avoid every hole.  
Therefore the answer is **480**.

Extracted answer: 480

––––––––––––––––––––––––––––––––––––––––––––––
Example 2
Question: How many seconds did the ball travel before impact?  
Please answer with a floating-point number rounded to one decimal place (e.g., 1.2).

Model response: The ball is in flight for **5.5** s according to the frame counter.

Extracted answer: 5.5

––––––––––––––––––––––––––––––––––––––––––––––
Example 3
Question: What is the amplitude of System 1’s oscillation?  
Please answer with a floating-point number rounded to two decimal places (e.g., 0.46).

Model response: Inspection of the graph shows an amplitude of  
\boxed{0.46} meters.

Extracted answer: 0.46

––––––––––––––––––––––––––––––––––––––––––––––
Example 4
Question: Which option correctly describes the robot’s final pose?  
Please answer with the correct option letter only (A, B, C, D).

Model response: After comparing all poses, option (D) is the only match.

Extracted answer: D

––––––––––––––––––––––––––––––––––––––––––––––
Example 5
Question: In what order do the two shuffled frames actually occur?  
Return a comma-separated sequence, e.g., 1,2.

Model response: Visual clues show the real order is 5 first and 4 second – so 5,4.

Extracted answer: 5,4

––––––––––––––––––––––––––––––––––––––––––––––
Example 6
Question: What colour cube rotates during the clip?  
Return the colour word only.

Model response: The rotating cube is clearly purple while the others stay still.

Extracted answer: purple

––––––––––––––––––––––––––––––––––––––––––––––
Example 7
Question: What is the resulting 2 × 2 matrix after the shear?  
Please answer with a Python list in the form [[a b], [c d]].

Model response: Performing the shear gives the matrix [[-3 -1], [0 -4]].

Extracted answer: [[-3 -1], [0 -4]]

––––––––––––––––––––––––––––––––––––––––––––––
Example 8
Question: How many red triangles are visible?  
Please answer with a single integer (e.g., 3).

Model response: The image only shows blue circles; no red triangles are present.

Extracted answer: None

––––––––––––––––––––––––––––––––––––––––––––––
Example 9
Question: After the motion is complete, which colored cube is visible to the right of the red cube?
A. Blue Cube
B. Green Cube
C. Orange Cube
D. Purple Cube
E. Yellow Cube

Model response: A. Blue Cube

Extracted answer: A

––––––––––––––––––––––––––––––––––––––––––––––
Example 10
Question: Which output grid should follow? Answer with one multiple choice option.

Model response: a)

Extracted answer: a

––––––––––––––––––––––––––––––––––––––––––––––
Example 11
"""

demo_prompt = demo_prompt.strip()


MAX_TOKENS    = 50                       # answer length
TEMPERATURE   = 0
async def extract_answer_async(sema: asyncio.Semaphore,
                               row_idx: int,
                               question: str,
                               model_response: str):
    """
    A single async extraction job.  Returns (row_idx, extracted_answer, question).
    """
    prompt = build_prompt(question, model_response)

    try:
        async with sema:                       # rate-limit gate
            resp = await client.chat.completions.create(
                model=model_name,
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                messages=[
                    {"role": "system",
                     "content": (
                         "You are an answer-extraction assistant. "
                         "Return ONLY the extracted answer—no extra words."
                     )},
                    {"role": "user", "content": prompt},
                ],
            )

        # first non-empty line
        answer = next(
            (ln.strip() for ln in resp.choices[0].message.content.splitlines()
             if ln.strip()),
            "",
        )
        return row_idx, answer, question

    except Exception as exc:
        print(f"[error row {row_idx}] {exc}")
        return row_idx, "Error", question
    

# ─── 1. Parse CLI arguments ──────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Extract short answers from verbose model predictions."
)
parser.add_argument(
    "csv_file",
    help="Input CSV that contains the model’s predictions (e.g. results_*.csv)",
)
parser.add_argument(
    "-o", "--out",
    help="Output CSV path.  Default: replace 'results' with 'extracted' in the "
         "input file name."
)
parser.add_argument(
    "-c", "--concurrency", type=int, default=10,
    help="Maximum simultaneous API calls (default=5)."
)
args = parser.parse_args()

# ─── 2. Global constants that now use args ───────────────────────────────
CSV_FILE   = args.csv_file
OUT_NAME   = args.out or CSV_FILE.replace("pred", "extract")
CONCURRENCY = args.concurrency
SAVE_EVERY  = 100                # rows after which we flush to disk
# 2.  Async extraction driver ─────────────────────────────────────────────────

# Create new column for extracted answers
df = pd.read_csv(CSV_FILE)
df['extracted_answer'] = None
df["question_text"] = None


dataset = load_dataset("video-reasoning/morse-500")
dataset = dataset['test']

def build_prompt(question: str, response: str) -> str:
    """
    Creates the exact template required by the demo_prompt.
    """
    return (
        f"{demo_prompt}\n"
        f"Question: {question}\n\n"
        f"Model response: {response}\n\n"
        f"Extracted answer:"
    )


async def producer() -> None:
    """
    Schedules one task per CSV row and records the results
    as they finish.  Uses `asyncio.as_completed` for streaming progress.
    """
    sema   = asyncio.Semaphore(CONCURRENCY)
    tasks  = []

    for row_idx, row in df.iterrows():
        q_text = dataset[row_idx]["question_text"]
        m_resp = str(row["prediction"])

        task = asyncio.create_task(
            extract_answer_async(sema, row_idx, q_text, m_resp)
        )
        tasks.append(task)

    processed = 0
    for fut in tqdm_asyncio.as_completed(tasks, total=len(tasks)):
        idx, extracted, q = await fut              # wait for next finished task
        df.at[idx, "question_text"]    = q
        df.at[idx, "extracted_answer"] = extracted
        processed += 1

        if processed % SAVE_EVERY == 0:
            df.to_csv(OUT_NAME, index=False)       # periodic checkpoint

async def main():
    await producer()

    # Rearrange columns once everything is done
    cols = [c for c in df.columns if c != "extracted_answer"]
    cols.insert(cols.index("ground_truth") + 1, "extracted_answer")
    cols = [c for c in cols if c != "question_text"] + ["question_text"]
    df_reordered = df[cols]
    df[cols].to_csv(OUT_NAME, index=False)
    print(f"✅ All done – answers written to {OUT_NAME}")

# 3.  Kick off the event loop ─────────────────────────────────────────────────
if __name__ == "__main__":
    asyncio.run(main())

