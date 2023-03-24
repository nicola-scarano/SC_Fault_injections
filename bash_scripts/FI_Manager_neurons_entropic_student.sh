#pipenv run python script/task/image_classification_FI_sbfm.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging.log -test_only -student_only 
# pipenv run python script/task/image_classification_FI_neuron_ber.py --config configs/ilsvrc2012/supervised_compression/ghnd-bq/resnet50-bq3ch_from_resnet50.yaml --device cuda --log FI_logging_neurons.log -test_only -student_only 

for beta in 0.08 0.16 0.32 0.64 1.28 2.56 5.12; do 
  pipenv run python script/task/image_classification_FI_neuron_ber.py -student_only \
    --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${beta}_from_resnet50.yaml \
    --log log/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${beta}_from_resnet50.txt\
    --device cuda \
    -test_only -student_only
done


beta="0.08"

pipenv run python script/task/image_classification_FI_teacher_neuron_ber.py -student_only \
    --config SC_Fault_injections/configs/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${beta}_from_resnet50.yaml \
    --log log/ilsvrc2012/supervised_compression/entropic_student/splitable_resnet50-fp-beta${beta}_from_resnet50.txt\
    --device cuda \
    -test_only -student_only

