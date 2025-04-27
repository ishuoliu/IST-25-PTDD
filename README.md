## Exploring Continual Learning in Code Intelligence with Domain-wise Distilled Prompts

### Dataset

- Dataset is available here: [dataset](https://zenodo.org/records/7827136#.ZDjMEnZByUl), put the dataset folder under the root directory.

### Quick Start

- Take Code Vulnerability Detection (CVD) as an example.

#### Fine-tune

```shell
cd CVD
python -m torch.distributed.launch --nproc_per_node 4 pp_new_cl.py \
    --pretrained_model codebert \
    --method pp_new \
    --task_list 0,1,2,3,4 \
    --output_dir ./results/pp_new/codebert/checkpoints \
    --available_gpu 0,1,2,3 \
    --prompt_token_num 20 \
    --max_input_tokens 256 \
    --gradient_accumulation_steps 1 \
    --batch_size 16 \
    --learning_rate 2e-5 \
    --epochs 10 \
    --lam 10 \
    --do_train \
    --train_all_param
```

#### Evaluate

```shell
cd CVD
python -m torch.distributed.launch --nproc_per_node 4 pp_new_cl.py \
    --pretrained_model codebert \
    --method pp_new \
    --task_list 0,1,2,3,4 \
    --output_dir ./results/pp_new/codebert/checkpoints \
    --available_gpu 0,1,2,3 \
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
```

