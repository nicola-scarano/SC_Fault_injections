

# git pull --recurse-submodules 

cp environment.yaml ../environment.yaml

cd ..

conda env create -f environment.yaml
source ~/miniconda3/bin/activate
conda init bash
conda activate sc2-benchmark

python -m pip install -e .
#pip install -e ./torchdistill-0.3.3/
python -m pip install -e ./SC_Fault_injections/pytorchfi_SC/

python -m pip list