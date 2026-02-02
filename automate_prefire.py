#!/usr/bin/env python3
import os
import argparse

def generate_batch_submission_script(output_base_dir: str):
    batch_submission_content = f"""#!/bin/bash

# Usage check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset="$1"

# Extract an era token like Run2022C, Run2022EE, Run2023C, etc.
year_run=$(echo "$dataset" | grep -oP 'Run20\\d{{2}}[A-Za-z]+' | head -1)

if [ -z "$year_run" ]; then
    echo "Error: Unable to extract era (e.g. Run2022C) from dataset path:"
    echo "  $dataset"
    exit 1
fi

# Output directory layout
output_dir="{output_base_dir}/files/$year_run/"

echo "[INFO] Era: $year_run"
echo "[INFO] Output dir: $output_dir"

# Create directories
mkdir -p "$output_dir"

# Submit job
python3 run_nano.py \\
  --dataset "$dataset" \\
  --exec eff_prefire_prob.py \\φ
  --output "$output_dir" \\
  --jobFlav testmatch \\
  --submitName prefire_${{year_run}}.sh \\
  --submit
"""

    script_path = "./condor/batch_submission_prefire.sh"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(batch_submission_content)
    os.chmod(script_path, 0o755)
    print(f"Generated {script_path}")


def generate_make_plots_script(output_base_dir: str):
    """
    Creates make_plots/make_plots_prefire.sh that:
      - takes 1 arg: <era> (e.g. Run2022C)
      - expects ROOT files in {output_base_dir}/files/<era>/prefire/
      - merges to merged_total.root and runs make_plot_prefire.py
      - writes plots to {output_base_dir}/plots/<era>/prefire/
    """
    make_plots_content = f"""#!/bin/bash

# Usage check
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <era>"
    echo "Example: $0 Run2022C"
    exit 1
fi

era="$1"

############ settings #############
root_files_dir="{output_base_dir}/files/$era/"
output_dir="{output_base_dir}/plots/$era/"
###################################

current_dir=$PWD

echo "Root files dir: ${{root_files_dir}}"
echo "Output dir: ${{output_dir}}"
echo "Dataset legend: ${{era}}"

mkdir -p "$output_dir"

# Merge all ROOT files
cd "$root_files_dir" || {{ echo "Directory not found: $root_files_dir"; exit 1; }}

rm -f merged_total.root
hadd -f merged_total.root *.root

# Run plotter
cd "$current_dir/../plotters/" || {{ echo "Plotters dir not found: $current_dir/../plotters/"; exit 1; }}

python3 eff_prefire.py \\
  -o "$output_dir" \\
  -i "$root_files_dir" \\
  --legend "$era"

cd "$current_dir"

echo "DONE"
"""

    script_path = "./make_plots/make_plots_prefire.sh"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as f:
        f.write(make_plots_content)
    os.chmod(script_path, 0o755)
    print(f"Generated {script_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Generate prefire submission & plotting scripts")
    ap.add_argument("-o", "--output", required=True, help="Base output directory for files & plots")
    args = ap.parse_args()

    # Normalize base dir (remove trailing slash)
    base = args.output.rstrip("/")

    generate_batch_submission_script(base)
    generate_make_plots_script(base)
