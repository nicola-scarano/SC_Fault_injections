#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
#target_layer="$2"


mkdir -p ${global_PWD}/${DIR}/entropic_student

#input_args=(1 2 3 6 9 12 77)
input_args=(0.08 5.12)

array_size=${#input_args[@]}

for ((i=0; i<$array_size; i++)); do
    for target_layer in 1 2; do  
        sbatch --output=$DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stdo_%A_%a.log --error=$DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stde_%A_%a.log ${global_PWD}/SC_Fault_injections/SLURM_scripts/entropic_student/Weight_cfg_FI.sbatch ${input_args[$((i))]} $target_layer ${DIR}
    done
done
