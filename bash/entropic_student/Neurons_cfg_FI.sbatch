#!/bin/bash
#SBATCH --time=120:00:00
#SBATCH --nodes=1
#SBATCH --partition=cuda
#SBATCH --gres=gpu:1
#SBATCH --ntasks-per-node=8
#SBATCH --job-name=NeuronsSplit
#SBATCH --mail-type=ALL
#SBATCH --mail-user=juan.guerrero@polito.it

# 1 Activate the virtual environment
source ~/miniconda3/bin/activate
conda deactivate

cd  /home/jguerrero/Workspace/GitHub/sc2-benchmark-fsim

conda activate sc2-benchmark
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


job_id="$SLURM_JOB_ID"

target_config="$1"
start_layer="$2"
stop_layer="$3"
DIR="$4"

Sim_dir=${global_PWD}/${DIR}/cnf${target_config}_lyr${start_layer}-${stop_layer}_JOBID${job_id}_N
mkdir -p ${Sim_dir}

if [ $target_config -eq 77 ]; then 
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+g" ${Sim_dir}/resnet50-bq1ch_from_resnet50.yaml
        sed -i "s/layers: \[.*\]/layers: \[$start_layer,$stop_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_y: [0-9.]\+/size_tail_y: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_x: [0-9.]\+/size_tail_x: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_delta: [0-9.]\+/block_fault_rate_delta: 0.2/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_steps: [0-9.]\+/block_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_delta: [0-9.]\+/neuron_fault_rate_delta: 0.02/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_steps: [0-9.]\+/neuron_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/image_classification_FI_teacher_neuron_ber.py -student_only \
                --config ${Sim_dir}/resnet50-bq1ch_from_resnet50.yaml\
                --device cuda\
                --log ${Sim_dir}/log/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50.log\
                -test_only\
                --world_size 1\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml

else
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${target_config}ch_from_resnet50.yaml ${Sim_dir}
        cp ${global_PWD}/SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/Fault_descriptor.yaml ${Sim_dir}
        sed -i "s+ckpt: !join \['./resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+ckpt: !join \['$global_PWD/resource/ckpt/ilsvrc2012/supervised_compression/ghnd-bq/', \*experiment, '.pt'\]+g" ${Sim_dir}/resnet50-bq${target_config}ch_from_resnet50.yaml
        sed -i "s/layers: \[.*\]/layers: \[$start_layer,$stop_layer\]/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/trials: [0-9.]\+/trials: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_y: [0-9.]\+/size_tail_y: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/size_tail_x: [0-9.]\+/size_tail_x: 32/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_delta: [0-9.]\+/block_fault_rate_delta: 0.2/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/block_fault_rate_steps: [0-9.]\+/block_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_delta: [0-9.]\+/neuron_fault_rate_delta: 0.02/" ${Sim_dir}/Fault_descriptor.yaml
        sed -i "s/neuron_fault_rate_steps: [0-9.]\+/neuron_fault_rate_steps: 5/" ${Sim_dir}/Fault_descriptor.yaml

        cd ${Sim_dir}

        python ${global_PWD}/SC_Fault_injections/script/image_classification_FI_neuron_ber.py -student_only \
                --config ${Sim_dir}/resnet50-bq${target_config}ch_from_resnet50.yaml\
                --device cuda\
                --log ${Sim_dir}/log/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${target_config}ch_from_resnet50.log\
                -test_only\
                --world_size 1\
                --fsim_config ${Sim_dir}/Fault_descriptor.yaml
fi
echo
echo "All done. Checking results:"
