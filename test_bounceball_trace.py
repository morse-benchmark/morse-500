#!/usr/bin/env python3
"""
Test script to verify the chronological reasoning trace in bounceball.py
This reads a generated reasoning trace file and displays it.
"""
import os
import sys
from pathlib import Path

# Check if any reasoning trace files exist
trace_dir = Path("temporal_reasoning/reasoning_traces")
if trace_dir.exists():
    trace_files = list(trace_dir.glob("ballhitswall_*.txt"))
    if trace_files:
        print(f"Found {len(trace_files)} reasoning trace file(s):")
        for i, f in enumerate(trace_files[:3], 1):  # Show first 3
            print(f"\n{'='*60}")
            print(f"File {i}: {f.name}")
            print('='*60)
            with open(f, 'r') as file:
                content = file.read()
                print(content)
    else:
        print("No ballhitswall reasoning trace files found yet.")
        print("Run the bounceball.py script first to generate traces.")
else:
    print(f"Trace directory does not exist: {trace_dir}")
    print("Run the bounceball.py script first to generate traces.")
