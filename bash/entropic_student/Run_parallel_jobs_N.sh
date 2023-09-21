#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"

mkdir -p ${global_PWD}/${DIR}

input_args=(0.08 5.12)

array_size=${#input_args[@]}

echo ${DIR}
export LOG_DIR=${DIR}

for ((i=0; i<$array_size; i++)); do
    for target_layer in 0 1 2 3 4; do  
        bash ${global_PWD}/SC_Fault_injections/bash/entropic_student/Neurons_cfg_FI.sh ${input_args[$((i))]} $target_layer $target_layer ${DIR} > $DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stdo.log 2> $DIR/cnf${input_args[$((i))]}_lyr${target_layer}_stde.log 
    done
done
