import os
import glob
import pandas as pd
import numpy as np

# Use the existing functions for category extraction
def extract_main_category(video_path):
    """Extract main reasoning category from video path"""
    filename = os.path.basename(video_path)
    parts = filename.split('_')
    
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"
    return parts[0]

def extract_subcategory(video_path):
    """Extract more detailed subcategory from video path"""
    filename = os.path.basename(video_path)
    parts = filename.split('_')
    
    if len(parts) >= 3:
        # Get the task domain (e.g., mazes, rope, robot)
        return f"{parts[2]}_{parts[3]}"
    
    return extract_main_category(video_path)  # Fallback to main category

# Define main categories in desired order
main_categories = [
    'abstract_reasoning',
    'mathematical_reasoning',
    'physical_reasoning',
    'planning_reasoning',
    'spatial_reasoning',
    'temporal_reasoning',
]


# Prepare to collect data
files_pattern = "extract/extract_sz512_*.csv"
# files_pattern = "extract_gemini/extract_sz512_*.csv"
files = glob.glob(files_pattern)
if not files:
    raise ValueError(f"No files found matching the pattern: {files_pattern}")

# Process each model's data, filtering with hard_index
all_model_data = []
headers = ['model', 'avg_acc'] + main_categories
data = []
numerical_data = []  # Will store just the numeric values for heatmap

for file_path in files:
    # model_name = os.path.basename(file_path).replace('extract_sz512_', '').replace('.csv', '')
    model_name = os.path.basename(file_path).replace('extract_sz512_', '').replace('.csv', '')
    
    # Load model data
    df = pd.read_csv(file_path)
    
    # Clean data
    df['ground_truth'] = df['ground_truth'].astype(str).str.strip()
    df['extracted_answer'] = df['extracted_answer'].astype(str).str.strip()
    
    # Remove invalid ground truth entries
    invalid_gt_mask = (df['ground_truth'].isna() | 
                      (df['ground_truth'] == '') | 
                      (df['ground_truth'].str.lower() == 'non'))
    
    if invalid_gt_mask.sum() > 0:
        print(f"Removing {invalid_gt_mask.sum()} items with invalid ground truth from {model_name}")
        df = df[~invalid_gt_mask]
    
    # Calculate correctness
    df['is_correct'] = df['ground_truth'] == df['extracted_answer']
    
    # Extract categories
    df['main_category'] = df['video'].apply(extract_main_category)
    df['subcategory'] = df['video'].apply(extract_subcategory)

    # Merge causal_reasoning into temporal_reasoning
    df.loc[df['main_category'] == 'causal_reasoning', 'main_category'] = 'temporal_reasoning'
    
    
    # Calculate overall accuracy
    avg_acc = df['is_correct'].mean() * 100
    
    # Calculate per-category accuracy
    row = []
    numeric_row = [avg_acc]  # Start with average accuracy
    
    for category in main_categories:
        category_mask = df['main_category'] == category
        total_count = category_mask.sum()
        
        if total_count > 0:
            total_correct = df.loc[category_mask, 'is_correct'].sum()
            category_acc = total_correct / total_count * 100
            # row.append(f"{category_acc:.1f} ({total_correct}/{total_count})")
            row.append(f"{category_acc:.1f}")
            
            numeric_row.append(category_acc)
        else:
            row.append("N/A")
            numeric_row.append(np.nan)
    
    # Store results
    data.append([model_name, avg_acc] + row)
    numerical_data.append([model_name] + numeric_row)
    
    # Add model identifier to the dataframe for later use
    df['model'] = model_name
    all_model_data.append(df)

# Create results dataframe with string format for display
results_df = pd.DataFrame(data, columns=headers)
clean_model_name = lambda name: name.replace('extract_sz512_', '')
results_df['model'] = results_df['model'].apply(clean_model_name)
results_df = results_df.sort_values('avg_acc', ascending=False)

# Create numerical dataframe for plotting
num_headers = ['model', 'avg_acc'] + main_categories
numerical_df = pd.DataFrame(numerical_data, columns=num_headers)
numerical_df = numerical_df.sort_values('avg_acc', ascending=False)

# print(results_df)
# results_df
print("\n" + "="*50 + "\n")
print(results_df.to_string(index=False))
print("\n" + "="*50 + "\n")