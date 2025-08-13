#!/bin/bash
## 评估 CMTEB
#pathm='/root/autodl-tmp/downloaded/jinaclipV2'
#pathm='/root/autodl-tmp/code/learnMultCard/save/epoch3'
#pathm="/root/autodl-tmp/code/learnMultCard/save_1205/epoch3"
pathm="/root/autodl-tmp/code/learnMultCard/save1209/epoch3"
pathm="/root/autodl-tmp/output_save_ckpt/save1212"
outname='zh_results_dimN_ft_20241212_jinaclip_3epoch_nGPU_1212'
p='mean'

export HF_HOME="/root/autodl-tmp/tmp_HF_cache"

#python eval_C-MTEB-all_v1031.py -d 32 -m $pathm -o $outname -p $p

CUDA_VISIBLE_DEVICES=0 python eval_C-MTEB-all_v1031.py  -d 1024 -m $pathm -o $outname -p $p

#python eval_C-MTEB-all.py -d 256

#python eval_C-MTEB-all.py -d 1024