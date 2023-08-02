#!/bin/bash
#SBATCH --time=72:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=WeightsSplit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=s302179@studenti.polito.it
#SBATCH --output=merge_output_%A.log
#SBATCH --error=merge_error_%A.log

source ~/miniconda3/bin/activate
conda deactivate

cd  /home/g.esposito/sc2-benchmark

conda activate sc2-benchmark

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
