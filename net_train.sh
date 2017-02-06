#!/usr/bin/env sh

align_data_path=/home/galaxyeye-tech/docs/deepid/data/CASIA2/
makelist_path=/home/galaxyeye-tech/docs/deepid/tools/make_list.py
im2rec_path=/home/galaxyeye-tech/docs/deepid/tools/im2rec.py
train_path=/home/galaxyeye-tech/docs/deepid/conv_net/train_net.py
list_name=/home/galaxyeye-tech/docs/deepid/data/casia_raw
rec_name=/home/galaxyeye-tech/docs/deepid/data/casia_raw

data_path=/home/galaxyeye-tech/docs/deepid/data/

# step1: generate .lst for im2rec
if ! [ -e ${list_name}_train.lst ];then
   python -u $makelist_path $align_data_path $list_name --train_ratio=0.9 --recursive=True
else
   echo ".lst file for training already exist."
fi
echo "generated .lst file done"

# step2: use img2rec to generate .rec file for training
if ! [ -e ${rec_name}_train.rec ]; then
	python -u $im2rec_path ${list_name}_train $align_data_path --resize=55 &
	python -u $im2rec_path ${list_name}_val $align_data_path --resize=55 &
else
	echo "$rec_name already exist."
fi
wait
echo "generate .rec done"

# step3: trainig the model for face recognition
#python -u lightened_cnn.py --gpus 2,3,4,5,6,7
python -u $train_path --data-dir=$data_path --model-prefix=./model/deep4 --load-epoch=170
echo "training done!"
