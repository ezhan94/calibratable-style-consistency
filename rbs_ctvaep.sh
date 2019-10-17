#!/bin/bash

read -r exp_id<.exp_id
echo $exp_id

rm .exp_id
echo $((exp_id+1)) > .exp_id


DIR=$1


CUDA_VISIBLE_DEVICES=0,1 python run.py --config_dir $DIR --save_dir /storage/imlearn_res --exp_name exp_$exp_id

python scripts/eval/check_selfcon_ctvaep.py --save_dir /storage/imlearn_res -f exp_$exp_id
python scripts/eval/check_selfcon_ctvaep.py --save_dir /storage/imlearn_res -f exp_$exp_id --sampling_mode indep
