#!/bin/bash
#SBATCH --time=00:40:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=WeightsSplit_obj
#SBATCH --mail-type=ALL



# 1 Activate the virtual environment
source ~/miniconda3/bin/activate sc2-benchmark

cd  /home/gesposito/sc2-benchmark

module load nvidia/cudasdk/10.1

# which python
# which pip

# pip list

# conda list

# nvidia-smi

PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


job_id="$SLURM_JOB_ID"

target_config="$1"
target_layer="$2"
DIR="$3"

Sim_dir=${global_PWD}/${DIR}/cnf${target_config}_lyr${target_layer}_JOBID${job_id}_W
mkdir -p ${Sim_dir} 


if [ $target_config -eq 77 ]; then 
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+ckpt: !join \['${global_PWD}/resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/obj_detection/object_detection_FI_sbfm.py -student_only \
                --config ${Sim_dir}/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
                --device cuda\
                --log ${Sim_dir}/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq1ch_fpn_from_faster_rcnn_resnet50_fpn.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml
else
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/coco2017/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+ckpt: !join \['${global_PWD}/resource/ckpt/coco2017/supervised_compression/ghnd-bq/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/obj_detection/object_detection_FI_sbfm.py -student_only \
                --config ${Sim_dir}/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.yaml\
                --device cuda\
                --log ${Sim_dir}/log/coco2017/supervised_compression/ghnd-bq/faster_rcnn_resnet50-bq${target_config}ch_fpn_from_faster_rcnn_resnet50_fpn.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml

fi

echo
echo "All done. Checking results:"


