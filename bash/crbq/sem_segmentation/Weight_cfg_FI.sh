#!/bin/bash

# 1 Activate the virtual environment
source ~/miniconda3/bin/activate sc2-benchmark

cd  /home/g.esposito/sc2-benchmark


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
        cp ${global_PWD}/SC_Fault_injections/configs/pascal_voc2012/supervised_compression/ghnd-bq/deeplabv3_resnet50-bq1ch_from_deeplabv3_resnet50.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/pascal_voc2012/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/pascal_voc2012/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+ckpt: !join \['${global_PWD}/resource/ckpt/pascal_voc2012/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/deeplabv3_resnet50-bq1ch_from_deeplabv3_resnet50.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/sem_segmentation/semantic_segmentation_sbfm.py -student_only \
                --config ${Sim_dir}/deeplabv3_resnet50-bq1ch_from_deeplabv3_resnet50.yaml\
                --device cpu\
                --log ${Sim_dir}/log/pascal_voc2012/supervised_compression/ghnd-bq/deeplabv3_resnet50-bq1ch_from_deeplabv3_resnet50.log\
                -test_only\
                --num_classes 21\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stde.log
else
        cp ${global_PWD}/SC_Fault_injections/configs/pascal_voc2012/supervised_compression/ghnd-bq/deeplabv3_resnet50-bq${target_config}ch_from_deeplabv3_resnet50.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/pascal_voc2012/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/pascal_voc2012/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+ckpt: !join \['${global_PWD}/resource/ckpt/pascal_voc2012/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/deeplabv3_resnet50-bq${target_config}ch_from_deeplabv3_resnet50.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}
        
        python ${global_PWD}/SC_Fault_injections/script/sem_segmentation/semantic_segmentation_sbfm.py -student_only \
                --config ${Sim_dir}/deeplabv3_resnet50-bq${target_config}ch_from_deeplabv3_resnet50.yaml\
                --device cpu\
                --log ${Sim_dir}/log/pascal_voc2012/supervised_compression/ghnd-bq/deeplabv3_resnet50-bq${target_config}ch_from_deeplabv3_resnet50.log\
                -test_only\
                --num_classes 21\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml > ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stdo.log 2> ${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}_stde.log

fi

echo
echo "All done. Checking results:"




# python script/task/object_detection.py -student_only --device cpu -test_only \
# --config configs/pascal_voc2012/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml \
# --log log/pascal_voc2012/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.txt


# python /home/gesposito/sc2-benchmark/SC_Fault_injections/script/obj_detection/semantic_segmentation_sbfm.py -student_only \
#         --config /home/gesposito/sc2-benchmark/FSIM_W_local_obj/cnf1_lyr0_JOBID0_W/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
#         --device cpu\
#         --log /home/gesposito/sc2-benchmark/FSIM_W_local_obj/cnf1_lyr0_JOBID0_W/log/pascal_voc2012/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.log\
#         -test_only\
#         --fsim_config /home/gesposito/sc2-benchmark/FSIM_W_local_obj/cnf1_lyr0_JOBID0_W/Fault_descriptor.yaml