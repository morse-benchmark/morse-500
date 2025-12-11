# python eval_model.py

# python extract_answers.py pred/pred_sz512_gemini-1.5-pro.csv
# python extract_answers.py pred/pred_sz512_gemini-2.0-flash.csv
# python extract_answers.py pred/pred_sz512_gemini-2.0-flash-lite.csv
# python extract_answers.py pred/pred_sz512_gemini-2.5-flash-preview-04-17.csv
# python extract_answers.py pred/pred_sz512_gemini-2.5-flash-preview-04-17_fps1_max16.csv
# python extract_answers.py pred/pred_sz512_gemini-2.5-flash-preview-04-17_fps2_max32.csv
# python extract_answers.py pred/pred_sz512_gemini-2.5-flash-preview-04-17_fps4_max64.csv
# python extract_answers.py pred/pred_sz512_gemini-2.5-flash-preview-04-17_fps8_max128.csv
# python extract_answers.py pred/pred_sz512_gemini-2.5-pro-preview-05-06.csv
# python extract_answers.py pred/pred_sz512_gemma-3-27b-it.csv
# python extract_answers.py pred/pred_sz512_gpt-4o-2411_fps2_max32.csv
# python extract_answers.py pred/pred_sz512_InternVL3-8B-Instruct.csv
# python extract_answers.py pred/pred_sz512_LLaVA-NeXT-Video-7B-hf.csv
# python extract_answers.py pred/pred_sz512_MiniCPM-o-2_6.csv
# python extract_answers.py pred/pred_sz512_o1-eval_fps2_max32.csv
# python extract_answers.py pred/pred_sz512_o3.csv
# python extract_answers.py pred/pred_sz512_o4-mini-eval_fps2_max32.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-Omni-7B.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-VL-32B-Instruct-AWQ.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-VL-32B-Instruct.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-VL-3B-Instruct.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-VL-72B-Instruct-AWQ.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-VL-72B-Instruct.csv
# python extract_answers.py pred/pred_sz512_Qwen2.5-VL-7B-Instruct.csv

# python plot_table.py

# python eval_async.py --video_folder ../spatial_reasoning/questions --model_name Qwen/Qwen3-VL-8B-Thinking --port 8000
# python eval_async.py --video_folder ../spatial_reasoning/questions --model_name Qwen/Qwen3-VL-8B-Instruct --port 8002
# python eval_async.py --video_folder ../spatial_reasoning/questions --model_name Qwen/Qwen3-VL-4B-Thinking --port 8003
# python eval_async.py --video_folder ../spatial_reasoning/questions --model_name Qwen/Qwen3-VL-4B-Instruct --port 8004

# python eval_async.py --video_folder ../spatial_reasoning/questions --model_name Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 --port 8000
# python eval_async.py --video_folder ../spatial_reasoning/questions --model_name Qwen/Qwen3-VL-235B-A22B-Thinking-FP8 --port 8000

# python eval_error_analysis.py Qwen2.5-VL-7B-Instruct
python eval_error_analysis.py Qwen3-VL-2B-Instruct
python eval_error_analysis.py Qwen3-VL-2B-Thinking
# python eval_error_analysis.py Qwen3-VL-4B-Instruct
python eval_error_analysis.py Qwen3-VL-4B-Thinking
# python eval_error_analysis.py Qwen3-VL-8B-Instruct
python eval_error_analysis.py Qwen3-VL-8B-Thinking
python eval_error_analysis.py Qwen3-VL-30B-A3B-Instruct-FP8
python eval_error_analysis.py Qwen3-VL-30B-A3B-Thinking-FP8
# python eval_error_analysis.py Qwen3-VL-235B-A22B-Instruct-FP8
python eval_error_analysis.py Qwen3-VL-235B-A22B-Thinking-FP8

