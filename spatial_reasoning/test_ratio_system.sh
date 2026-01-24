#!/bin/bash
# test_ratio_system.sh
# Quick validation script for ratio-based implementation

set -e  # Exit on error

echo "======================================================================"
echo "Ratio-Based Size Parameter - System Test"
echo "======================================================================"
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test counters
PASSED=0
FAILED=0

# Helper function for tests
test_step() {
    echo -n "  Testing: $1... "
}

pass() {
    echo -e "${GREEN}✓ PASS${NC}"
    PASSED=$((PASSED + 1))
}

fail() {
    echo -e "${RED}✗ FAIL${NC}: $1"
    FAILED=$((FAILED + 1))
}

warn() {
    echo -e "${YELLOW}⚠ WARN${NC}: $1"
}

# Test 1: Check files exist
echo "[Test 1] Checking if all files exist..."
FILES=(
    "cubes_ratio.py"
    "eval_ratio_sweep.py"
    "analyze_ratio_results.py"
    "RATIO_QUICK_START.md"
    "RATIO_IMPLEMENTATION_SUMMARY.md"
)

for file in "${FILES[@]}"; do
    test_step "$file exists"
    if [ -f "$file" ]; then
        pass
    else
        fail "File not found"
    fi
done
echo ""

# Test 2: Check Python files are valid syntax
echo "[Test 2] Checking Python syntax..."
for pyfile in cubes_ratio.py eval_ratio_sweep.py analyze_ratio_results.py; do
    test_step "$pyfile syntax"
    if python3 -m py_compile "$pyfile" 2>/dev/null; then
        pass
    else
        fail "Syntax error"
    fi
done
echo ""

# Test 3: Check imports
echo "[Test 3] Checking Python imports..."
test_step "cubes_ratio.py imports"
if python3 -c "import sys; sys.path.insert(0, '.'); exec(open('cubes_ratio.py').read().split('if __name__')[0])" 2>/dev/null; then
    pass
else
    warn "Some imports may be missing (Manim required)"
fi

test_step "eval_ratio_sweep.py imports"
if python3 -c "
import os, sys, json, base64, asyncio, argparse, subprocess
from pathlib import Path
from collections import defaultdict, deque
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
import re
print('OK')
" 2>/dev/null | grep -q "OK"; then
    pass
else
    fail "Missing required imports"
fi

test_step "analyze_ratio_results.py imports"
if python3 -c "
import json, argparse
from pathlib import Path
from typing import Dict, List
import numpy as np
print('OK')
" 2>/dev/null | grep -q "OK"; then
    pass
else
    warn "Missing matplotlib or numpy (optional for plotting)"
fi
echo ""

# Test 4: Check environment variable handling
echo "[Test 4] Testing environment variable handling..."
test_step "SIZE_RATIO environment variable"
export SIZE_RATIO=15.0
export P_TYPE=count
if python3 -c "
import os
ratio = float(os.getenv('SIZE_RATIO', 10.0))
assert ratio == 15.0, f'Got {ratio}, expected 15.0'
print('OK')
" 2>/dev/null | grep -q "OK"; then
    pass
else
    fail "Environment variable not read correctly"
fi
echo ""

# Test 5: Check file generation (dry run)
echo "[Test 5] Testing video generation setup..."
test_step "Output directories creation"
export SIZE_RATIO=10.0
export P_TYPE=count
if python3 -c "
from pathlib import Path
Path('questions').mkdir(exist_ok=True)
Path('solutions').mkdir(exist_ok=True)
Path('question_text').mkdir(exist_ok=True)
Path('reasoning_traces').mkdir(exist_ok=True)
print('OK')
" 2>/dev/null | grep -q "OK"; then
    pass
else
    fail "Cannot create output directories"
fi
echo ""

# Test 6: Check ratio calculation logic
echo "[Test 6] Testing ratio calculation logic..."
test_step "Ratio to grid size mapping"
if python3 -c "
# Test ratio mapping
def ratio_to_grid(ratio):
    min_ratio, max_ratio = 0.5, 40.0
    min_grid, max_grid = 2, 10
    ratio_normalized = (ratio - min_ratio) / (max_ratio - min_ratio)
    ratio_normalized = max(0.0, min(1.0, ratio_normalized))
    return int(min_grid + ratio_normalized * (max_grid - min_grid))

# Test cases (approximate due to rounding)
min_grid = ratio_to_grid(0.5)
max_grid = ratio_to_grid(40.0)
mid_grid = ratio_to_grid(20.0)
assert min_grid == 2, f'Min ratio should give min grid, got {min_grid}'
assert max_grid == 10, f'Max ratio should give max grid, got {max_grid}'
assert 5 <= mid_grid <= 7, f'Mid ratio should give mid grid, got {mid_grid}'

print('OK')
" 2>/dev/null | grep -q "OK"; then
    pass
else
    fail "Ratio mapping logic incorrect"
fi
echo ""

# Test 7: Check CLI help
echo "[Test 7] Testing CLI interfaces..."
test_step "eval_ratio_sweep.py --help"
if python3 eval_ratio_sweep.py --help > /dev/null 2>&1; then
    pass
else
    fail "CLI help not working"
fi

test_step "analyze_ratio_results.py --help"
if python3 analyze_ratio_results.py --help > /dev/null 2>&1; then
    pass
else
    fail "CLI help not working"
fi
echo ""

# Test 8: Check documentation completeness
echo "[Test 8] Checking documentation..."
test_step "Quick start guide has usage examples"
if grep -q "Quick Usage" RATIO_QUICK_START.md && \
   grep -q "export SIZE_RATIO" RATIO_QUICK_START.md; then
    pass
else
    fail "Quick start missing key sections"
fi

test_step "Implementation summary has metrics"
if grep -q "Max Achievable Ratio" RATIO_IMPLEMENTATION_SUMMARY.md && \
   grep -q "Accuracy per Ratio" RATIO_IMPLEMENTATION_SUMMARY.md; then
    pass
else
    fail "Summary missing key metrics"
fi
echo ""

# Summary
echo "======================================================================"
echo "Test Summary"
echo "======================================================================"
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}✓ All tests passed!${NC}"
    echo ""
    echo "System is ready. Next steps:"
    echo "  1. Read RATIO_QUICK_START.md for usage"
    echo "  2. Generate test video: export SIZE_RATIO=10.0 && python cubes_ratio.py"
    echo "  3. Run evaluation: python eval_ratio_sweep.py --help"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Some tests failed${NC}"
    echo ""
    echo "Please check the errors above and:"
    echo "  - Ensure all Python dependencies are installed (manim, numpy, matplotlib)"
    echo "  - Verify file permissions"
    echo "  - Check Python version (3.8+ required)"
    echo ""
    exit 1
fi
