#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=WeightsSplit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=juan.guerrero@polito.it


# 1 Activate the virtual environment
source ~/miniconda3/bin/activate
conda deactivate

cd  /home/jguerrero/Workspace/GitHub/sc2-benchmark

conda activate sc2-benchmark-fsim
module load nvidia/cudasdk/11.6

which python
which pip

pip list

conda list

nvidia-smi
PWD=`pwd`
echo ${PWD}
global_PWD="$PWD"
echo ${CUDA_VISIBLE_DEVICES}


#job_id="$SLURM_JOB_ID"
job_id=0

target_config="$1"
target_layer="$2"
DIR="$3"

Sim_dir=${global_PWD}/${DIR}/entropic_student/cnf${target_config}_lyr${target_layer}_JOBID_${job_id}_W
mkdir -p ${Sim_dir}


if [ $target_config -eq 77 ]; then 
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+g" ${Sim_dir}/resnet50-bq1ch_from_resnet50.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/image_classification_FI_teacher_sbfm.py -student_only \
                --config ${Sim_dir}/resnet50-bq1ch_from_resnet50.yaml\
                --device cuda\
                --log ${Sim_dir}/log/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml
else
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${target_config}_from_resnet50.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/entropic_student/Fault_descriptor.yaml ${Sim_dir}

        sed -i "s+ckpt: !join \['./resource/ckpt/ilsvrc2012/supervised_compression/entropic_student/', \*student_experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/ilsvrc2012/supervised_compression/entropic_student/', \*student_experiment, '.pt'\]+g" ${Sim_dir}/splitable_resnet50-fp-beta${target_config}_from_resnet50.yaml
        sed -i "s/layer: \[.*\]/layer: \[$target_layer\]/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/image_classification_FI_sbfm.py -student_only \
                --config ${Sim_dir}/splitable_resnet50-fp-beta${target_config}_from_resnet50.yaml \
                --device cuda \
                --log ${Sim_dir}/log/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${target_config}_from_resnet50.log\
                -test_only\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml

fi

echo
echo "All done. Checking results:"
