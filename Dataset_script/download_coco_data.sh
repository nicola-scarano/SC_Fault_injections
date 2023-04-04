wget http://images.cocodataset.org/zips/train2017.zip ./
wget http://images.cocodataset.org/zips/val2017.zip ./
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip ./

# Go to the root of this repository
mkdir ~/dataset/coco2017/ -p
mv train2017.zip ~/dataset/coco2017/
mv val2017.zip ~/dataset/coco2017/
mv annotations_trainval2017.zip ~/dataset/coco2017/
cd ~/dataset/coco2017/
unzip train2017.zip
unzip val2017.zip
unzip annotations_trainval2017.zip
cd ../../../
