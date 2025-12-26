#! /usr/bin/env sh

## arguments
exec=$1
infile=$2
output_dir=$3
pwd=$4

# check if 5th argument exists
if [ $# -ge 5 ]; then
    json_file=$5
else
    json_file=""
fi

## needed to load CMSSW libraries/packages
cd $pwd/../../
eval `scramv1 runtime -sh`

## run executable
cd $pwd/../src/
if [ -z "$json_file" ]; then
    python3 $exec -i $infile -o $output_dir
else
    python3 $exec -i $infile -o $output_dir --json $json_file
fi