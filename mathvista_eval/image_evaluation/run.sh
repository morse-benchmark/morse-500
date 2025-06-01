#!/bin/bash

source ~/.bashrc
conda activate math-v
python evaluate.py
python extractor.py
python levenshtein_distance.py
