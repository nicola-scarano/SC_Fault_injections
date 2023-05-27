#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
target_layer="$2"


mkdir -p ${global_PWD}/${DIR}

input_args=(1 2 3 6 9 12 77)

array_size=${#input_args[@]}

for ((i=0; i<$array_size; i++)); do
    sbatch --output=$DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stdo_%A_%a.log --error=$DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stde_%A_%a.log ${global_PWD}/SC_Fault_injections/SLURM_scripts/crbq/img_classification/Weight_cfg_FI.sbatch ${input_args[$((i))]} $target_layer ${DIR}
done
