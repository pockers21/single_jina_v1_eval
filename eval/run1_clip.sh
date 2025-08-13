#!/bin/bash
## 评估 clip
 
#mpath="/root/autodl-tmp/downloaded/jinaclipV2"
#mpath="/root/autodl-tmp/code/save/epoch3"
#mpath="/root/autodl-tmp/code/learnMultCard/save/epoch3"
#mpath="/root/autodl-tmp/code/learnMultCard/save_1205/epoch3"
#mpath="/root/autodl-tmp/code/learnMultCard/save/epoch6"
mpath="/root/autodl-tmp/code/learnMultCard/save1209/epoch3"
mpath="/root/autodl-tmp/output_save_ckpt/save1212"
output="jinaclip_epoch3_nGPU_1212"

export HF_HOME="/root/autodl-tmp/tmp_HF_cache"
CUDA_VISIBLE_DEVICES=1 python zeroshot_retrieval_1_jinaclip.py -d 1024 -m $mpath -o $output
 