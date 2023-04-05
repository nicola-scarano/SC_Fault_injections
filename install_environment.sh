

# git pull --recurse-submodules 

cp environment.yaml ../environment.yaml

cd ..

conda env create -f environment.yaml
source /home/jguerrero/miniconda3/etc/profile.d/conda.sh

conda activate sc2-benchmark

pip install -e .
#pip install -e ./torchdistill-0.3.3/
pip install -e ./SC_Fault_injections/pytorchfi_SC/