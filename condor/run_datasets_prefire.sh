#!/bin/bash

# Check if the dataset list file is provided
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_list_file>"
    exit 1
fi

dataset_list_file="$1"

# Check if the dataset list file exists
if [ ! -f "$dataset_list_file" ]; then
    echo "Error: File '$dataset_list_file' not found."
    exit 1
fi

# Read the dataset list and call batch_submission.sh for each dataset
while IFS= read -r dataset; do
    if [ -n "$dataset" ]; then # Skip empty lines
        ./batch_submission_prefire.sh "$dataset"

        # Introduce a delay to allow Condor to process the submission
        sleep 5
    fi
done < "$dataset_list_file"