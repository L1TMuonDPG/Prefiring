import os
import argparse

def generate_batch_submission_script(output_base_dir):
    batch_submission_content = f"""#!/bin/bash

# Check if the dataset is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset>"
    exit 1
fi

dataset="$1"

# Extract the year and run number from the dataset path
year_run=$(echo "$dataset" | grep -oP '(?<=Run)([0-9]+[A-Z])' | head -1)

# Check if the year and run number are extracted successfully
if [ -z "$year_run" ]; then
    echo "Error: Unable to extract year and run number from the dataset path."
    exit 1
fi

# Construct the output directory path
output_dir="{output_base_dir}/files/$year_run"

# Create the output directory if it doesn't exist
mkdir -p "$output_dir"

# Submit the jobs to condor
python3 run_nano.py --dataset "$dataset" --exec eff_prefiring.py --output "$output_dir/eff_pref/" --jobFlav testmatch --submitName eff_${{year_run}}.sh --submit

sleep 5

python3 run_nano.py --dataset "$dataset" --exec eff_prefiring_v2.py --output "$output_dir/eff_pref_v2/" --jobFlav testmatch --submitName eff_v2_${{year_run}}.sh --submit 
"""
    script_path = "./condor/batch_submission.sh"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as file:
        file.write(batch_submission_content)

    os.chmod(script_path, 0o755)
    print(f"Generated {script_path}")


def generate_make_plots_script(output_base_dir):
    make_plots_content = f"""#!/bin/bash

# Check if the era is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <era>"
    exit 1
fi

era="$1"

############ settings #############
root_files_dir="{output_base_dir}/files/$era"
output_dir="{output_base_dir}/plots/$era"
###################################

current_dir=$PWD

echo "Root files dir: ${{root_files_dir}}"
echo "Output dir: ${{output_dir}}"
echo "Dataset legend: ${{era}}"

mkdir -p $output_dir

############ Efficiency Prefiring #############
mkdir -p $output_dir/eff_pref/

cd $root_files_dir/eff_pref/

rm -rf merged_total.root
hadd merged_total.root *.root

cd $current_dir/../plotters/

python3 eff_prefiring_plots.py -o $output_dir/eff_pref/ -i $root_files_dir/eff_pref/ --legend "$era"

############ Efficiency Prefiring v2 #############
mkdir -p $output_dir/eff_pref_v2/

cd $root_files_dir/eff_pref_v2/

rm -rf merged_total.root
hadd merged_total.root *.root

cd $current_dir/../plotters/

python3 eff_prefiring_plots.py -o $output_dir/eff_pref_v2/ -i $root_files_dir/eff_pref_v2/ --legend "$era"

cd $current_dir

"""
    script_path = "./make_plots/make_plots.sh"
    os.makedirs(os.path.dirname(script_path), exist_ok=True)
    with open(script_path, "w") as file:
        file.write(make_plots_content)

    os.chmod(script_path, 0o755)
    print(f"Generated {script_path}")



def generate_make_plots_scripts(output_base_dir, include_eff, include_run, include_comparison, include_all):
    options= ["eff_pref", "eff_pref_v2"]
    for option in options:
        make_plots_content = f"""#!/bin/bash
# Check if the era is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <era>"
    exit 1
fi

era="$1"

############ settings #############
root_files_dir="{output_base_dir}/files/$era/{option}/"
output_dir="{output_base_dir}/plots/$era/{option}/"
###################################

current_dir=$PWD

echo "Root files dir: ${{root_files_dir}}"
echo "Output dir: ${{output_dir}}"
echo "Dataset legend: ${{era}}"

mkdir -p $output_dir

cd $root_files_dir

rm -rf merged_total.root
hadd merged_total.root *.root

cd $current_dir/../plotters/

python3 {option}_plots.py -o $output_dir -i $root_files_dir --legend "$era"

cd $current_dir

echo "DONE"
"""
    
        script_path = f"./make_plots/make_plots_{option}.sh"
        os.makedirs(os.path.dirname(script_path), exist_ok=True)
        with open(script_path, "w") as file:
            file.write(make_plots_content)

        os.chmod(script_path, 0o755)
        print(f"Generated {script_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate files for automated creation of DPG plots")
    parser.add_argument("-o", "--output", required=True, type=str, help="Output directory for the DPG files and plots")
    args = parser.parse_args()

    # Remove trailing slash from output directory if present
    output_base_dir = args.output.rstrip("/")

    # Generate scripts
    generate_batch_submission_script(output_base_dir)
    generate_make_plots_script(output_base_dir)
    generate_make_plots_scripts(output_base_dir)
