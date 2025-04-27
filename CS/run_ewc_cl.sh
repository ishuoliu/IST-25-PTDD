#!/bin/bash


lang="mix1"
pretrained_model="codet5"
method="ewc"     # ewc, emr

python -m torch.distributed.launch --nproc_per_node 4 ewc_cl.py \
    --lang ${lang} \
    --pretrained_model ${pretrained_model} \
    --method ${method} \
    --task_list ruby0,javascript0,java0,go0,php0,python0 \
    --output_dir ./results/${method}/${pretrained_model}/${lang}/checkpoints \
    --train_examplar_path ./results/${method}/${pretrained_model}/${lang}/checkpoints/train_examplar.jsonl \
    --eval_examplar_path ./results/${method}/${pretrained_model}/${lang}/checkpoints/eval_examplar.jsonl \
    --available_gpu 0,1,2,3 \
    --check_order 0 \
    --prompt_token_num 0 \
    --max_input_tokens 256 \
    --max_output_tokens 128 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 5e-5 \
    --train_replay_size 2245 \
    --eval_replay_size 281 \
    --epochs 10 \
    --do_test \
#    --train_all_param

