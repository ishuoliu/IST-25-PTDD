#!/bin/bash


pretrained_model="codebert"
method="pp_old"     # prefix, prompt, finetune, pp

python -m torch.distributed.launch --nproc_per_node 4 prompt_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0,1,2,3,4 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_all_prompt_none_10 \
    --available_gpu 0,1,2,3 \
    --check_order 3 \
    --prompt_token_num 10 \
    --max_input_tokens 200 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --do_test \
    --mlp_type none \
    --train_all_param




#pretrained_model="codebert"
#method="pp_old"     # prefix, prompt, finetune, pp
#
#python -m torch.distributed.launch --nproc_per_node 4 pp_new_cl.py \
#    --pretrained_model ${pretrained_model} \
#    --method ${method} \
#    --task_list 0,1,2,3,4 \
#    --output_dir ./results/${method}/${pretrained_model}/checkpoints_part_pp_new_10_lam10_1e-2 \
#    --available_gpu 0,1,2,3 \
#    --check_order 0 \
#    --prompt_token_num 10 \
#    --max_input_tokens 200 \
#    --gradient_accumulation_steps 1 \
#    --batch_size 16 \
#    --learning_rate 1e-2 \
#    --epochs 10 \
#    --lam 10 \
#    --do_train \

