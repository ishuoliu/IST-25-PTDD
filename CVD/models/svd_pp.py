import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy


class ResMLP(torch.nn.Module):
    def __init__(self, module_type, hidden_dim, embed_dim, args, residual=True):
        super(ResMLP, self).__init__()
        self.residual = residual

        if module_type == "mlp1":
            self.module = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, embed_dim)
            )
        elif module_type == "mlp2":
            self.module = nn.Sequential(
                nn.Linear(embed_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Linear(hidden_dim // 2, embed_dim)
            )
        elif module_type == "transformer":
            self.encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=2, dropout=0.05).to(args.device)
            self.module = nn.TransformerEncoder(self.encoder_layer, num_layers=2).to(args.device)
        else:
            assert False, "Invalid Module Type."

    def forward(self, inputs):
        if self.residual:
            return self.module(inputs) + inputs
        else:
            return self.module(inputs)


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


class SVDPPModel(nn.Module):
    def __init__(self, basemodel, args, previous_prompts=None):
        super(SVDPPModel, self).__init__()
        self.args = args
        # self.basemodel = basemodel

        self.model = basemodel
        self.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.args.prompt_token_num),
                                                      requires_grad=True))

        self.get_MLP(self.args.mlp_bottleneck_size)

        # self.classifier = RobertaClassifier(args)

        if self.args.do_train and self.args.cl_num == 0:
            # self.previous_prompts = None
            self.previous_prompts = torch.zeros([0, self.model.prompt.shape[1]],
                                                requires_grad=False)
        else:
            self.previous_prompts = previous_prompts
        self.previous_prompts = self.previous_prompts.to(args.local_rank)

        if not self.args.train_all_param:
            print("Freezing weights.")
            self.do_freeze_weights()

    def get_MLP(self, bottleneck_size):
        if self.args.mlp_type == "none":
            self.prefix_mlp = None
        else:
            N = self.model.roberta.embeddings.word_embeddings.weight.shape[1]
            self.prefix_mlp = ResMLP(module_type=self.args.mlp_type,
                                     hidden_dim=bottleneck_size,
                                     embed_dim=N,
                                     args=self.args).to(self.args.device)

    def init_new_prompt(self, prompt_len):
        N = self.model.roberta.embeddings.word_embeddings.weight.shape[0]
        prompt_weigths = []

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N)     # random token
                w = deepcopy(self.model.roberta.embeddings.word_embeddings.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths     # prompt_len * 768

    def do_freeze_weights(self, except_condition="none"):
        model = self.model
        for name, param in model.named_parameters():
            # print(name)
            if param.requires_grad == True and except_condition not in name:
                param.requires_grad = False

    def forward(self, input_id=None, input_mask=None, label=None, input_prompt=None, mode="train"):
        if mode == "train":
            if self.args.mlp_type == "none":
                prompt = self.model.prompt
            else:
                mlp = self.prefix_mlp
                prompt = mlp(self.model.prompt)

            k = input_id.shape[0]
            input_embeds = self.model.roberta.embeddings.word_embeddings(input_id)

            inputs_embeds = torch.cat((prompt.repeat(k, 1, 1),
                                       self.previous_prompts.repeat(k, 1, 1),
                                       input_embeds),
                                      dim=1)
            full_prefix_len = self.previous_prompts.shape[0] + prompt.shape[0]

            source_mask_updated = torch.cat((input_mask[0][0].repeat(k, full_prefix_len),
                                             input_mask),
                                            dim=1)

            outputs = self.model(
                attention_mask=source_mask_updated,
                inputs_embeds=inputs_embeds
            )

            prob = torch.nn.functional.softmax(outputs[0])
            prob = prob[:, 1]

            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(outputs[0], label)

            # logits = self.classifier(outputs[0])
            # prob = torch.sigmoid(logits)
            #
            # loss_fct = nn.BCELoss()
            # loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())

            new_prompt = prompt
            return_prompt = torch.cat((new_prompt, self.previous_prompts), dim=0)

            return loss, prob, return_prompt

        elif mode == "eval" or mode == "test":
            prompt = input_prompt

            k = input_id.shape[0]
            input_embeds = self.model.roberta.embeddings.word_embeddings(input_id)
            inputs_embeds = torch.cat((prompt.repeat(k, 1, 1),
                                       input_embeds),
                                      dim=1)
            full_prefix_len = prompt.shape[0]
            source_mask_updated = torch.cat((input_mask[0][0].repeat(k, full_prefix_len),
                                             input_mask),
                                            dim=1)

            outputs = self.model(
                attention_mask=source_mask_updated,
                inputs_embeds=inputs_embeds
            )

            prob = torch.nn.functional.softmax(outputs[0])
            prob = prob[:, 1]

            cross_entropy_loss = nn.CrossEntropyLoss()
            loss = cross_entropy_loss(outputs[0], label)

            # logits = self.classifier(outputs[0])
            # prob = torch.sigmoid(logits)
            #
            # loss_fct = nn.BCELoss()
            # loss = loss_fct(prob, torch.unsqueeze(label, dim=1).float())

            return loss, prob

        else:
            assert False, "Invalid Mode Name!"

