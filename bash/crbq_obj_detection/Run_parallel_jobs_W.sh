#!/bin/bash

PWD=`pwd`

global_PWD="$PWD"

DIR="$1"
target_layer="$2"


mkdir -p ${global_PWD}/${DIR}

input_args=(1 2 3 6 9 12 77)

array_size=${#input_args[@]}

for ((i=0; i<$array_size; i++)); do
    bash ${global_PWD}/SC_Fault_injections/bash/crbq_obj_detection/Weight_cfg_FI.sh ${input_args[$((i))]} $target_layer ${DIR}
done
