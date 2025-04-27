#!/bin/bash



lang="ruby"
pretrained_model="codet5"
method="pp_new_cl"     # prefix, prompt, finetune

python -m torch.distributed.launch --nproc_per_node 4 pp_new_cl.py \
    --lang ${lang} \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list ${lang}0,${lang}1,${lang}2,${lang}3,${lang}4 \
    --output_dir ./results/${method}/${pretrained_model}/${lang}/checkpoints_all_pp_new_token20_lam5 \
    --available_gpu 0,1,2,3 \
    --check_order 3 \
    --prompt_token_num 20 \
    --max_input_tokens 256 \
    --max_output_tokens 128 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --epochs 10 \
    --lam 5 \
    --do_train \
    --train_all_param







