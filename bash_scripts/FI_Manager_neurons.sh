#pipenv run python script/task/image_classification_FI_sbfm.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only 
# pipenv run python script/task/image_classification_FI_neuron_ber.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging_neurons.log -test_only -student_only 

CWD=`pwd`

echo $CWD

pipenv run python SC_Fault_injections/script/image_classification_FI_teacher_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50_faulty.yaml --device cuda --log FI_logging_neuron_original.log -test_only -student_only 
pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch1.log -test_only -student_only 
pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq2ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch2.log -test_only -student_only 
pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch3.log -test_only -student_only 
pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq6ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch6.log -test_only -student_only 
pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq9ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_ch9.log -test_only -student_only 
pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq12ch_from_resnet50_faulty.yaml --device cuda --log FI_loggingch_12.log -test_only -student_only 

#pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only  