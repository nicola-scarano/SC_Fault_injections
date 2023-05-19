#!/bin/bash

# 1 Activate the virtual environment
source ~/miniconda3/bin/activate sc2-benchmark

cd  /home/gesposito/sc2-benchmark


# which python
# which pip

# pip list

# conda list

# nvidia-smi

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id=0

target_config="$1"
target_layer="$2"
DIR="$3"

Sim_dir=${global_PWD}/${DIR}/cnf${target_config}_lyr${target_layer}_JOBID${job_id}_W
mkdir -p ${Sim_dir}


if [ $target_config -eq 77 ]; then 
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+g" ${Sim_dir}/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/image_classification_FI_teacher_sbfm.py -student_only \
                --config ${Sim_dir}/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
                --device cpu\
                --log ${Sim_dir}/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stde.log
else
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+g" ${Sim_dir}/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}
        echo ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/
        python ${global_PWD}/SC_Fault_injections/script/object_detection.py -student_only \
                --config ${Sim_dir}/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
                --device cpu\
                --log ${Sim_dir}/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stde.log

fi

echo
echo "All done. Checking results:"
