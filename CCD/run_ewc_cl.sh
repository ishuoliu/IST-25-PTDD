#!/bin/bash



pretrained_model="codebert"
method="emr"     # ewc, emr

python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new \
    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/train_examplar.jsonl \
    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/eval_examplar.jsonl \
    --available_gpu 0,1,2,3 \
    --check_order 1 \
    --prompt_token_num 0 \
    --max_input_tokens 200 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --train_replay_size 2000 \
    --eval_replay_size 250 \
    --epochs 10 \
    --do_test \
#    --train_all_param

pretrained_model="codebert"
method="emr"     # ewc, emr

python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new \
    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/train_examplar.jsonl \
    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/eval_examplar.jsonl \
    --available_gpu 0,1,2,3 \
    --check_order 2 \
    --prompt_token_num 0 \
    --max_input_tokens 200 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --train_replay_size 2000 \
    --eval_replay_size 250 \
    --epochs 10 \
    --do_test \
#    --train_all_param

pretrained_model="codebert"
method="emr"     # ewc, emr

python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new \
    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/train_examplar.jsonl \
    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/eval_examplar.jsonl \
    --available_gpu 0,1,2,3 \
    --check_order 3 \
    --prompt_token_num 0 \
    --max_input_tokens 200 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --train_replay_size 2000 \
    --eval_replay_size 250 \
    --epochs 10 \
    --do_test \
#    --train_all_param

pretrained_model="codebert"
method="emr"     # ewc, emr

python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_new \
    --train_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/train_examplar.jsonl \
    --eval_examplar_path ./results/${method}/${pretrained_model}/checkpoints_new/eval_examplar.jsonl \
    --available_gpu 0,1,2,3 \
    --check_order 4 \
    --prompt_token_num 0 \
    --max_input_tokens 200 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --train_replay_size 2000 \
    --eval_replay_size 250 \
    --epochs 10 \
    --do_test \
#    --train_all_param

