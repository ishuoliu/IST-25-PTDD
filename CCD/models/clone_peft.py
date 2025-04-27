import torch
import torch.nn as nn
from peft import PrefixTuningConfig, TaskType, get_peft_model, PromptTuningInit, PromptTuningConfig


def set_all_parameters(model):
    for name, param in model.named_parameters():
        if not param.requires_grad:
            param.requires_grad = True

    return model


class RobertaClassifier(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.dropout = nn.Dropout(args.dropout)
        self.ll_proj = nn.Linear(args.hidden_size, 1)     # ll means last layer representations.

    def forward(self, features):
        cls_features = features[:, 0, :]

        cls_features = self.dropout(cls_features)
        proj_score = self.ll_proj(cls_features)
        return proj_score


class ClonePeftModel(nn.Module):
    def __init__(self, basemodel, args):
        super(ClonePeftModel, self).__init__()
        self.classifier = RobertaClassifier(args)

        if args.pretrained_model in ["codebert", "graphcodebert"]:
            if args.method in ["prefix"]:
                peft_config = PrefixTuningConfig(
                    task_type=TaskType.SEQ_CLS,
                    inference_mode=False,
                    num_virtual_tokens=args.prompt_token_num)
                basemodel = get_peft_model(basemodel, peft_config)
                if args.train_all_param:
                    basemodel = set_all_parameters(basemodel)
            elif args.method in ["prompt"]:
                peft_config = PromptTuningConfig(
                    task_type=TaskType.SEQ_CLS,
                    prompt_tuning_init=PromptTuningInit.TEXT,
                    num_virtual_tokens=args.prompt_token_num,
                    prompt_tuning_init_text="Please summarize the following code in natural language:",
                    tokenizer_name_or_path=args.huggingface_name,
                )
                basemodel = get_peft_model(basemodel, peft_config)
                if args.train_all_param:
                    basemodel = set_all_parameters(basemodel)
            else:
                assert False, "Invalid PEFT Method."
        else:
            assert False, "Invalid Pre-trained Model."

        self.model = basemodel

    def forward(self, input_id=None, input_mask=None, label=None):
        outputs = self.model(input_ids=input_id, attention_mask=input_mask)
        logits = self.classifier(outputs[0])
        prob = torch.sigmoid(logits)

        loss_fct = nn.BCELoss()
        loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())

        return loss, prob
