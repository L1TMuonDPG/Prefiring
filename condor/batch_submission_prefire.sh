#!/bin/bash

# Usage check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset="$1"

# Extract an era token like Run2022C, Run2022EE, Run2023C, etc.
year_run=$(echo "$dataset" | grep -oP 'Run20\d{2}[A-Za-z]+' | head -1)

if [ -z "$year_run" ]; then
    echo "Error: Unable to extract era (e.g. Run2022C) from dataset path:"
    echo "  $dataset"
    exit 1
fi

# Output directory layout
output_dir="/eos/user/p/pkatris/Prefiring_prob_pteta/files/$year_run/"

echo "[INFO] Era: $year_run"
echo "[INFO] Output dir: $output_dir"

# Create directories
mkdir -p "$output_dir"

# Submit job
python3 run_nano.py \
  --dataset "$dataset" \
  --exec eff_prefire_prob.py \
  --output "$output_dir" \
  --jobFlav testmatch \
  --submitName prefire_${year_run}.sh \
  --submit
