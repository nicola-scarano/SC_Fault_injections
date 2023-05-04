# SC_Fault_injections

This is a Fault/error injection framework for reliability evaluation of split computing DNN architectures. 
The framework is compatible with [sc2-benchmark](https://github.com/yoshitomo-matsubara/sc2-benchmark) and can be extended to other scenarios as well

# prerequisites
Install miniconda environmet if you already have it ignore this step
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-py38_23.1.0-1-Linux-x86_64.sh
bash Miniconda3-py38_23.1.0-1-Linux-x86_64.sh -b
```

# Getting started on a Linux x86\_64 PC
```bash
# sc2-benchmark
git clone https://github.com/yoshitomo-matsubara/sc2-benchmark
cd sc2-benchmark

# SC_Fult_injections: 
git clone https://github.com/divadnauj-GB/SC_Fault_injections
cd SC_Fault_injections
find . -name "*.sh" | xargs chmod +x

# pytorchfi_SC 
git clone https://github.com/divadnauj-GB/pytorchfi_SC

# create the sc2-benchmark environmet and install the required dependencies
# if you already crerated the sc2-benchmark please first remove it and then create it again as follows
cp environment.yaml ../environment.yaml
cd ..
conda deactivate

conda env create -f environment.yaml
conda deactivate
source ~/miniconda3/bin/activate sc2-benchmark

python -m pip install -e .

python -m pip install -e ./SC_Fault_injections/pytorchfi_SC/
```

# Directory structure (simplified)
```
sc2-benchmark.
             ├── configs
             ├── environment.yaml
             ├── LICENSE
             ├── MANIFEST.in
             ├── Pipfile
             ├── README.md
             ├── sc2bench
             ├── SC_Fault_injections
             │   ├── bash
             │   │   ├── Check_DNN_archs.sh
             │   │   ├── crbq
             │   │   │   ├── merge_reports.py
             │   │   │   ├── merge_reports.sh
             │   │   │   ├── Neurons_cfg_FI.sh
             │   │   │   ├── Run_parallel_jobs_N.sh
             │   │   │   ├── Run_parallel_jobs_W.sh
             │   │   │   └── Weight_cfg_FI.sh
             │   ├── configs
             │   ├── Dataset_script
             │   ├── environment.yaml
             │   ├── install_environment.sh
             │   ├── Pipfile
             │   ├── pytorchfi_SC
             │   ├── report_analysis
             │   ├── script
             │   └── SLURM_scripts
             │       └── crbq
             │           ├── merge_reports.py
             │           ├── merge_reports.sbatch
             │           ├── Neurons_cfg_FI.sbatch
             │           ├── Run_parallel_jobs_N.sh
             │           ├── Run_parallel_jobs_W.sh
             │           └── Weight_cfg_FI.sbatch
             ├── script
             ├── setup.cfg
             ├── setup.py
             └── tree.txt
```
# How to use this framework?
1. deactivate the base conda environmet and activate the sc2-benchmark environment
2. change in your terminal the directory to the sc2-benchmark directory.
2. run the Fsim command 
```bash
bash ./SC_Fault_injections/bash/crbq/Run_parallel_jobs_W.sh FSIM_W 0
```
This command will create a folder called FSIM_W and it will star performing fault simulations to the layer 0 for all models crbq from sc2-benchmark. 
This process will take for long time (at least one week), thus press cntrl+c to cancel the siluations after some minutes. you should bne able 
to see inside the FSIM_W folder at least one subfolder with some *.csv and *.json files. 

**Note** It is recomended to use HPC system to execute several FSIMs in parallel. for that purposes you can follow exactly the same steps but intead to use 
bash scripts use the SLURM scripts 
```bash
bash ./SC_Fault_injections/SLURM_scripts/crbq/Run_parallel_jobs_W.sh FSIM_W 0
```
in this case the results will take a couple of days
