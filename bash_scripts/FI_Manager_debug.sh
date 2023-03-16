#pipenv run python script/task/image_classification_FI_debug.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50_faulty.yaml --device cuda --log FI_logging_debug.log -test_only -student_only 

#pipenv run python script/task/image_classification_FI_sbfm.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq2ch_from_resnet50_faulty.yaml --device cuda --log FI_logging.log -test_only -student_only 
#
#pipenv run python script/task/image_classification_FI_sbfm.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50_faulty.yaml --device cuda --log FI_logging.log -test_only -student_only 
#



#pipenv run python script/task/image_classification_FI_neuron_ber.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only 

pipenv run python SC_Fault_injections/script/image_classification_FI_sbfm.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50_faulty_debug.yaml --device cuda --log FI_logging_neuron_original_debug.log -test_only -student_only 