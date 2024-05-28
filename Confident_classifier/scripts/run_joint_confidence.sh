# baseline 
export save=../results/joint_confidence_loss/${RANDOM}/
mkdir -p $save
python3 ../src/run_joint_confidence.py --outf $save --dataroot ../data   2>&1 | tee  $save/log.txt
