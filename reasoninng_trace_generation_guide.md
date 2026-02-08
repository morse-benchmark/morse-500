I have a Manim script (current code) that generates visual reasoning questions. 
I want to convert it to match the format and structure of the working `spatial_reasoning/cube_path.py` (reference) script.

**Please help me convert current code to follow the same patterns as the reference:**

1. **Timing system**: Use `self.renderer.time` instead of `time.time()` to track video playback time (not code execution time)

2. **Event logging**: Add a `log_event(description)` method that records scene events with accurate video timestamps

3. **File structure**: Organize output into the same directories:
   - `questions/` - video files
   - `solutions/` - answer text files
   - `question_text/` - question text files
   - `reasoning_traces/` - step-by-step reasoning

4. **Reasoning trace**: Generate a comprehensive `build_reasoning_trace()` method that:
   - Includes timestamped scene descriptions
   - Explains the problem step-by-step
   - Shows intermediate states/calculations
   - Provides clear final answer in `\boxed{}` format

5. **Code organization**:
   - Add comprehensive comments explaining each section
   - Use clear section dividers (===)
   - Document what each method does and why

6. **Validation**: If applicable, add validation checks to verify the answer is computed correctly

**Key things to preserve from current code:**
- The core problem logic and difficulty
- The visual style and animations
- Any problem-specific parameters

**Key things to adapt from cube_path.py:**
- Timing mechanism using `self.renderer.time`
- Event logging system
- Reasoning trace structure
- File organization and naming
- Comment style and documentation

Please walk through the conversion step by step, explaining:
1. What needs to change and why
2. Any problem-specific considerations
3. How to structure the reasoning trace for this specific question type