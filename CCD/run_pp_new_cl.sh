#!/bin/bash


pretrained_model="codebert"
method="pp_old"     # prefix, prompt, finetune, pp

python -m torch.distributed.launch --nproc_per_node 4 pp_new_cl.py \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list 0,1,2,3,4 \
    --output_dir ./results/${method}/${pretrained_model}/checkpoints_all_pp_new_token10_lam10 \
    --available_gpu 0,1,2,3 \
    --check_order 3 \
    --prompt_token_num 10 \
    --max_input_tokens 200 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --lam 10 \
    --do_test \
    --train_all_param





