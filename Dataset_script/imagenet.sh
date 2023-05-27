TRAIN_DATASET_URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_train.tar
VAL_DATASET_URL=https://image-net.org/data/ILSVRC/2012/ILSVRC2012_img_val.tar


wget --no-check-certificate ${TRAIN_DATASET_URL} ./
wget --no-check-certificate ${VAL_DATASET_URL} ./

# Go to the root of this repository
mkdir ~/dataset/ilsvrc2012/{train,val} -p
mv ILSVRC2012_img_train.tar ~/dataset/ilsvrc2012/train/
mv ILSVRC2012_img_val.tar ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/train/
tar -xvf ILSVRC2012_img_train.tar
mv ILSVRC2012_img_train.tar ../
for f in *.tar; do
  d=`basename $f .tar`
  mkdir $d
  (cd $d && tar xf ../$f)
done
rm -r *.tar
cd ~/dataset/ilsvrc2012/

wget https://raw.githubusercontent.com/soumith/imagenetloader.torch/master/valprep.sh
mv valprep.sh ~/dataset/ilsvrc2012/val/
cd ~/dataset/ilsvrc2012/val/
tar -xvf ILSVRC2012_img_val.tar
mv ILSVRC2012_img_val.tar ../
sh valprep.sh
mv valprep.sh ../
cd ~/dataset/ilsvrc2012/

