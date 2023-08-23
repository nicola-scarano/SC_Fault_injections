#!/bin/bash

# 1 Activate the virtual environment
source ~/miniconda3/bin/activate sc2-benchmark

cd  /home/g.esposito/sc2-benchmark

module load nvidia/cudasdk/10.1


PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id="$SLURM_JOB_ID"

target_config="$1"
start_layer="$2"
stop_layer="$3"
DIR="$4"

Sim_dir=${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}-${stop_layer}_JOBID${job_id}_N
mkdir -p ${Sim_dir}
# faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn
if [ $target_config -eq 77 ]; then 
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml
        sed -i "s/layers: \[.*\]/layers: \[$start_layer,$stop_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_y: [0-9.]\+/size_tail_y: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_x: [0-9.]\+/size_tail_x: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_delta: [0-9.]\+/block_fault_rate_delta: 0.2/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_steps: [0-9.]\+/block_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_delta: [0-9.]\+/neuron_fault_rate_delta: 0.02/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_steps: [0-9.]\+/neuron_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/obj_detection/object_detection_FI_neuron.py -student_only \
                --config ${Sim_dir}/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
                --device cuda\
                --log ${Sim_dir}/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stde.log

else
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml
        sed -i "s/layers: \[.*\]/layers: \[$start_layer,$stop_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_y: [0-9.]\+/size_tail_y: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_x: [0-9.]\+/size_tail_x: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_delta: [0-9.]\+/block_fault_rate_delta: 0.2/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_steps: [0-9.]\+/block_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_delta: [0-9.]\+/neuron_fault_rate_delta: 0.02/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_steps: [0-9.]\+/neuron_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/obj_detection/object_detection_FI_neuron.py -student_only \
                --config ${Sim_dir}/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
                --device cuda\
                --log ${Sim_dir}/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stde.log
fi
echo
echo "All done. Checking results:"
