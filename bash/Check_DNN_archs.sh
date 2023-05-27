#pipenv run python script/task/image_classification_FI_sbfm.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only 
# pipenv run python script/task/image_classification_FI_neuron_ber.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log DNN_arch_neurons.log -test_only -student_only 

CWD=`pwd`

echo $CWD

# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50_faulty.yaml --device cuda --log DNN_arch_neuron_original.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq1ch_from_resnet50_faulty.yaml --device cuda --log DNN_arch_ch1.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq2ch_from_resnet50_faulty.yaml --device cuda --log DNN_arch_ch2.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50_faulty.yaml --device cuda --log DNN_arch_ch3.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq6ch_from_resnet50_faulty.yaml --device cuda --log DNN_arch_ch6.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq9ch_from_resnet50_faulty.yaml --device cuda --log DNN_arch_ch9.log -test_only -student_only 
# pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq12ch_from_resnet50_faulty.yaml --device cuda --log FDNN_arch_ch_12.log -test_only -student_only 

# #pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only  
for beta in 1.28e-8 1.024e-7 2.048e-7 3.2768e-6; do 
# for beta in 1.28e-8; do 
  pipenv run python SC_Fault_injections/script/image_classification_check_DNNs.py \
    --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}_faulty.yaml \
    --device cuda \
    --log log/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}_summary.log \
    -test_only -student_only
done