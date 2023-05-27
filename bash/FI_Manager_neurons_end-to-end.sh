#pipenv run python script/task/image_classification_FI_sbfm.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only 
# pipenv run python script/task/image_classification_FI_neuron_ber.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging_neurons.log -test_only -student_only 

for beta in 1.28e-8 1.024e-7 2.048e-7 3.2768e-6; do 
  pipenv run python SC_Fault_injections/script/image_classification_FI_neuron_ber.py \
    --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}_faulty.yaml \
    --device cuda \
    --log log/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}_faulty.log \
    -test_only -student_only
done

# beta="1.28e-8"

# pipenv run python SC_Fault_injections/script/image_classification_FI_teacher_neuron_ber.py\
#     --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}_faulty.yaml \
#     -device cuda \
#     --log log/ilsvrc2012/supervised_compression/end-to-end/splitable_resnet50-fp-beta${beta}_faulty.log \
#     -test_only -student_only

