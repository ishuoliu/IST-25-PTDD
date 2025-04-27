#!/bin/bash

pretrained_model="codebert"
method="ewc"     # ewc, emr

python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new_2 \
    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/train_examplar.jsonl \
    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/eval_examplar.jsonl \
    --available_gpu 0,1,2,3 \
    --check_order 4 \
    --prompt_token_num 0 \
    --max_input_tokens 256 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --train_replay_size 1500 \
    --eval_replay_size 188 \
    --epochs 10 \
    --do_test \
    --seed 55 \
#    --train_all_param



#pretrained_model="codebert"
#method="ewc"     # ewc, emr
#
#python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
#    --pretrained_model ${pretrained_model} \
#    --method ${method} \
#    --task_list 1 \
#    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new_2 \
#    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/train_examplar.jsonl \
#    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/eval_examplar.jsonl \
#    --available_gpu 0,1,2,3 \
#    --check_order 1 \
#    --prompt_token_num 0 \
#    --max_input_tokens 256 \
#    --gradient_accumulation_steps 1 \
#    --batch_size 16 \
#    --learning_rate 2e-5 \
#    --train_replay_size 1500 \
#    --eval_replay_size 188 \
#    --epochs 10 \
#    --do_train \
#    --seed 55 \
##    --train_all_param
#
#pretrained_model="codebert"
#method="ewc"     # ewc, emr
#
#python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
#    --pretrained_model ${pretrained_model} \
#    --method ${method} \
#    --task_list 2 \
#    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new_2 \
#    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/train_examplar.jsonl \
#    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/eval_examplar.jsonl \
#    --available_gpu 0,1,2,3 \
#    --check_order 1 \
#    --prompt_token_num 0 \
#    --max_input_tokens 256 \
#    --gradient_accumulation_steps 1 \
#    --batch_size 16 \
#    --learning_rate 2e-5 \
#    --train_replay_size 1500 \
#    --eval_replay_size 188 \
#    --epochs 10 \
#    --do_train \
#    --seed 55 \
##    --train_all_param
#
#pretrained_model="codebert"
#method="ewc"     # ewc, emr
#
#python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
#    --pretrained_model ${pretrained_model} \
#    --method ${method} \
#    --task_list 3 \
#    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new_2 \
#    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/train_examplar.jsonl \
#    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/eval_examplar.jsonl \
#    --available_gpu 0,1,2,3 \
#    --check_order 1 \
#    --prompt_token_num 0 \
#    --max_input_tokens 256 \
#    --gradient_accumulation_steps 1 \
#    --batch_size 16 \
#    --learning_rate 2e-5 \
#    --train_replay_size 1500 \
#    --eval_replay_size 188 \
#    --epochs 10 \
#    --do_train \
#    --seed 55 \
##    --train_all_param
#
#pretrained_model="codebert"
#method="ewc"     # ewc, emr
#
#python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
#    --pretrained_model ${pretrained_model} \
#    --method ${method} \
#    --task_list 4 \
#    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new_2 \
#    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/train_examplar.jsonl \
#    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new_2/eval_examplar.jsonl \
#    --available_gpu 0,1,2,3 \
#    --check_order 1 \
#    --prompt_token_num 0 \
#    --max_input_tokens 256 \
#    --gradient_accumulation_steps 1 \
#    --batch_size 16 \
#    --learning_rate 2e-5 \
#    --train_replay_size 1500 \
#    --eval_replay_size 188 \
#    --epochs 10 \
#    --do_train \
#    --seed 55 \
##    --train_all_param





