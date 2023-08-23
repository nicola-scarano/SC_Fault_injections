#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
start_layer="$2"
stop_layer="$3"

mkdir -p ${global_PWD}/${DIR}

input_args=(1 2 3 6 9 12 77)

array_size=${#input_args[@]}

echo ${DIR}
export LOG_DIR=${DIR}

for ((i=0; i<$array_size; i++)); do
    sbatch --output=$DIR/cnf${input_args[$((i))]}_lyr${start_layer}-${stop_layer}_stdo_%A_%a.log --error=$DIR/cnf${input_args[$((i))]}_lyr${start_layer}-${stop_layer}_stde_%A_%a.log ${global_PWD}/SC_Fault_injections/bash/crbq/obj_detection/Neurons_cfg_FI.sh ${input_args[$((i))]} $start_layer $stop_layer ${DIR} 
done

#/home/g.esposito/sc2-benchmark/SC_Fault_injections/bash/crbq/obj_detection