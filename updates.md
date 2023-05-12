## Updates

I worked only on the weights files and parts. 

It seems it is working, I have launched the jobs on the hpc and in few days we should see the results. The fault_f1_k is the sum of the f1 scores of the first 5 predictions for the faulty model because I considered it as the score WITHIN of the best k prediction (it can be easily changed to the f1score of only the kth element). 

Also the golden_f1_1 seems very low and i think that is due to the fact that the number of classes that are in G_clas is larger than the number of classes that are in the ground truth. (I think) This is, in turn, due to the fact that the model is trained to predict 1000 classes and at inference time it potentially can predict 1000 classes and the f1 score also detects this gap since it takes into account also the False Positive and the False negatives.

Note1: the SC_Fault_injections/bash/Weight_cfg_FI.sh is running on the cpu for debugging purposes, you can esaly switch to the cuda module by changing the arguments you pass to the function that runs the script i.e. python...

Note2: I had to change the paths w.r.t. yours because my folder sc2-benchmark was in a different place. So please check the paths in the *.sh files 