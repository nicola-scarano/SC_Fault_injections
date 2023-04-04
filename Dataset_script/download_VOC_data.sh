wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar

# Go to the root of this repository
mkdir ~/dataset/ -p
mv VOCtrainval_11-May-2012.tar ~/dataset/
cd ~/dataset/
tar -xvf VOCtrainval_11-May-2012.tar
cd ../../
