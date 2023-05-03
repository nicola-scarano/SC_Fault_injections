

# pipenv run python SC_Fault_injections/script/image_classification_FI_teacher_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50_faulty.yaml --device cuda --log FI_logging_neuron_original.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch1.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq2ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch2.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch3.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq6ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch6.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq9ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch9.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq12ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch12.log -test_only -student_only 



#pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only 
source ~/miniconda3/bin/activate 
conda deactivate
conda activate sc2-benchmark

for bch in 1 2 3 6 9 12; do
  python SC_Fault_injections/script/image_classification_FI_sbfm.py -student_only \
    --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${bch}ch_from_resnet50_faulty.yaml\
    --device cuda\
    --log log/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq${bch}ch_from_resnet50.log\
    -test_only -student_only 
done

python SC_Fault_injections/script/image_classification_FI_teacher_sbfm.py \
    --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50_faulty.yaml\
    --device cuda\
    --log log/ilsvrc2012/supervised_compression/ghnd-bq/resnet50_faulty.log\
    -test_only -student_onl