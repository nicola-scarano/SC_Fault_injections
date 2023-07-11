#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
#target_layer="$2"


mkdir -p ${global_PWD}/${DIR}/entropic_student

#input_args=(1 2 3 6 9 12 77)
input_args=(0.08)

array_size=${#input_args[@]}

for ((i=0; i<$array_size; i++)); do
    for target_layer in 1; do  
        bash ${global_PWD}/SC_Fault_injections/bash/entropic_student/Weight_cfg_FI.sh ${input_args[$((i))]} $target_layer ${DIR} > $DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stdo.log 2> $DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stde.log
    done
done
