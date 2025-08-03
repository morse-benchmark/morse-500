import pandas as pd
from tqdm import tqdm
from vllm import LLM, SamplingParams
from datasets import load_dataset

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_NEW_TOKENS = 50
TEMPERATURE = 0.01

# Initialize vLLM
print(f"Loading {MODEL_NAME} with vLLM...")
llm = LLM(
    model=MODEL_NAME,
    trust_remote_code=True,
    dtype="float16",
    max_model_len=4096,
    gpu_memory_utilization=0.9,
)

# Configure sampling parameters
sampling_params = SamplingParams(
    max_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE, top_p=1.0, stop=["<|im_end|>"]
)

DEMO_PROMPT = """\
Your task is to extract the exact answer from a model's response to a MathVista question.

Follow these extraction rules:
1. For numbers: return just the number (e.g., 14)
2. For floats: keep exact decimals (e.g., 0.6, 1.45)
3. For lists: use Python format (e.g., [2007, 2008])
4. For multiple choice: return the letter (e.g., B)
5. For text: return the exact phrase from the response

Examples:
Question: Which number is missing?
Response: The missing number is 14.
Extracted: 14

Question: What fraction is blue?
Response: The blue fraction is 0.67.
Extracted: 0.67

Question: Between which years was the peak?
Response: The peak was between 2007-2008.
Extracted: [2007, 2008]

Question: What's the correct option?
Response: The answer is (B).
Extracted: B

Now extract the answer from this actual response:
"""


def extract_answer(question, response):
    """Extract answer using vLLM"""
    prompt = f"{DEMO_PROMPT}\nQuestion: {question}\nResponse: {response}\nExtracted:"

    try:
        outputs = llm.generate([prompt], sampling_params)
        answer = outputs[0].outputs[0].text.strip()
        return answer.split("\n")[0].strip()
    except Exception as e:
        print(f"Extraction error: {e}")
        return "Error: Extraction failed"


def process_csv(input_csv, output_csv, batch_size=32):
    """Process CSV file with MathVista results using batched processing"""
    df = pd.read_csv(input_csv)
    math_vista = load_dataset("AI4Math/MathVista", split="testmini")
    df["extracted_answer"] = None

    # Process in batches for better efficiency
    for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
        batch_end = min(i + batch_size, len(df))
        batch = df.iloc[i:batch_end]

        prompts = []
        indices = []

        # Prepare batch
        for idx, row in batch.iterrows():
            question = math_vista[idx]["query"]
            prediction = row["prediction"]
            prompts.append(
                f"{DEMO_PROMPT}\nQuestion: {question}\nResponse: {prediction}\nExtracted:"
            )
            indices.append(idx)

        try:
            # Generate answers for the whole batch
            outputs = llm.generate(prompts, sampling_params)

            # Update DataFrame with results
            for output, idx in zip(outputs, indices):
                answer = output.outputs[0].text.strip().split("\n")[0].strip()
                df.at[idx, "extracted_answer"] = answer

            # Save checkpoint periodically
            if i % (batch_size * 10) == 0 and i > 0:
                df.to_csv(output_csv, index=False)
                print(f"\nCheckpoint saved at row {i}")

        except Exception as e:
            print(f"\nError processing batch {i}-{batch_end}: {e}")
            # Mark failed rows
            for idx in indices:
                df.at[idx, "extracted_answer"] = f"Error: {e}"

    df.to_csv(output_csv, index=False)
    return df


if __name__ == "__main__":
    input_csv = "MathVista_QWen2.5_VL-7B_Instruct_multi_image.csv"
    output_csv = "mathvista_extracted_answers_multi_image.csv"

    print(f"Starting answer extraction using {MODEL_NAME} with vLLM")
    print("Using batch processing for better performance")

    try:
        result_df = process_csv(input_csv, output_csv)
        print(f"\nAnswer extraction complete. Results saved to {output_csv}")

    except Exception as e:
        print(f"\nFatal error: {e}")
        raise
