import argparse
import os
from datasets import Dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, default="/fs/nexus-scratch/mrislam/morse-500/data/train")
    parser.add_argument("--local", action="store_true")
    parser.add_argument("--out", type=str, default="/fs/nexus-scratch/mrislam/morse-500/fine_tuning/data")
    args = parser.parse_args()
    data = []
    if args.local:
        for category in ["spatial", "temporal"]: 
            video_paths = os.scandir(os.path.join(args.path, category, "questions"))
            for idx, video in enumerate(video_paths): 
                # print(video.name)
                basename = video.name.split(".mp4")[0]
                with open(os.path.join(args.path, category, "question_text", basename+".txt")) as f:
                    prompt =  f.read()
                with open(os.path.join(args.path, category, "reasoning_traces", basename+".txt")) as f:
                    rt =  f.read()
                with open(os.path.join(args.path, category, "solutions", basename+".txt")) as f:
                    soln = f.read()
                data.append(
                    {
                    "data_source": "video-reasoning/morse-trace",
                    "prompt": [
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    "video": {
                        "type": "video", 
                        "video": video.path
                    }, 
                    "ability": category,  # e.g., "math", "VQA"
                    "reward_model": {
                        "style": "rule",
                        "ground_truth": soln
                    },
                    "extra_info": {
                        "split": "train",
                        "index": idx,
                        "reasoning_trace": rt, 
                    }
                    }
                )
        morse_dataset = Dataset.from_dict({k: [d[k] for d in data] for k in data[0].keys()})

    split_dataset = morse_dataset.train_test_split(test_size=0.1, seed=42)

    split_dataset['train'].to_parquet(os.path.join(args.out, "train_morse.parquet"))
    split_dataset['test'].to_parquet(os.path.join(args.out, "val_morse.parquet"))
    
    print(f"Saved {len(split_dataset['train'])} train and {len(split_dataset['test'])} val samples.")
                

