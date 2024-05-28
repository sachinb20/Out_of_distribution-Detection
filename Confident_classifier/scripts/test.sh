# baseline 
export save=../test/${RANDOM}/
mkdir -p $save
python ../src/test_detection.py --outf $save --dataset $1 --out_dataset $2 --pre_trained_net $3  --dataroot ../data   2>&1 | tee  $save/log.txt

#bash test.sh cifar10 /home/dell/OOD_Detection/deep_Mahalanobis_detector/data/LSUN_resize /home/dell/OOD_Detection/confidence_estimation/checkpoints/cifar10_vgg13_baseline_epoch_200.pt
