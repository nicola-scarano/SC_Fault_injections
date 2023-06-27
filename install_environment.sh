

# git pull --recurse-submodules 

cp environment.yaml ../environment.yaml

cd ..
source ~/miniconda3/bin/activate
conda deactivate

conda env create -f environment.yaml
conda activate sc2-benchmark-fsim

python -m pip install -e .
#pip install -e ./torchdistill-0.3.3/
python -m pip install -e ./SC_Fault_injections/pytorchfi_SC/

python -m pip list
