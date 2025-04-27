#!/bin/bash

python -m torch.distributed.launch --nproc_per_node 2 pp_new_cl.py \
    --pretrained_model codebert \
    --method pp_new \
    --task_list 0,1,2,3,4 \
    --output_dir ./results/pp_new/codebert/checkpoints \
    --available_gpu 0,1 \
    --prompt_token_num 20 \
    --max_input_tokens 256 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --lam 10 \
    --do_train \
    --train_all_param


python -m torch.distributed.launch --nproc_per_node 2 pp_new_cl.py \
    --pretrained_model codebert \
    --method pp_new \
    --task_list 0,1,2,3,4 \
    --output_dir ./results/pp_new/codebert/checkpoints \
    --available_gpu 0,1 \
    --check_order 4 \
    --prompt_token_num 20 \
    --max_input_tokens 256 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --lam 10 \
    --do_test \
    --train_all_param




