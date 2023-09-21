#!/bin/bash

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/jguerrero/Workspace/GitHub/sc2-benchmark

conda activate sc2-benchmark-fsim

which pip
which python

PWD=`pwd`
Global_path="$PWD"

folder="$1"
workers="$2"
echo $Global_path

echo $folder
echo $workers
echo ${Global_path}/${folder}

python ${Global_path}/SC_Fault_injections/SLURM_scripts/crbq/merge_reports.py --path ${Global_path}/${folder} --workers ${workers}

echo "merge finishied"
