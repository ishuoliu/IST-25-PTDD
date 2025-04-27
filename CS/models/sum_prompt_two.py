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


class SumPPModel(nn.Module):
    def __init__(self, basemodel, args, previous_prompts=None):
        super(SumPPModel, self).__init__()
        self.args = args
        self.model = basemodel

        if self.args.do_train:
            if self.args.cl_num == 0:
                self.model.prompt = nn.Parameter(torch.tensor(self.init_new_prompt(self.args.prompt_token_num),
                                                        requires_grad=True))
            else:
                self.model.prompt = nn.Parameter(previous_prompts)
        else:
            pass

        self.get_MLP(self.args.mlp_bottleneck_size)

        # if self.args.do_train:
        #     if self.args.cl_num == 0:
        #         self.previous_prompts = torch.zeros([0, self.model.prompt.shape[1]],
        #                                             requires_grad=False)
        #     else:
        #         self.previous_prompts = previous_prompts
        # if self.args.do_test:
        #     self.previous_prompts = previous_prompts
        # self.previous_prompts = self.previous_prompts.to(args.local_rank)

        if not self.args.train_all_param:
            print("Freezing weights.")
            self.do_freeze_weights()

    def get_MLP(self, bottleneck_size):
        if self.args.mlp_type == "none":
            self.prefix_mlp = None
        else:
            # print(self.basemodel.shared.weight)
            N = self.model.shared.weight.size(1)
            # N = self.basemodel.embeddings.word_embeddings.weight.shape[1]
            self.prefix_mlp = ResMLP(module_type=self.args.mlp_type,
                                     hidden_dim=bottleneck_size,
                                     embed_dim=N,
                                     args=self.args).to(self.args.device)

    def init_new_prompt(self, prompt_len):
        N = self.model.shared.weight.size(0)
        # N = self.basemodel.encoder.embed_tokens.weight.shape[0]
        prompt_weigths = []

        for i in range(prompt_len):
            with torch.no_grad():
                j = np.random.randint(N)     # random token
                w = deepcopy(self.model.encoder.embed_tokens.weight[j].detach().cpu().numpy())
                prompt_weigths.append(w)
        prompt_weigths = np.array(prompt_weigths)
        return prompt_weigths     # prompt_len * 768

    def do_freeze_weights(self, except_condition="none"):
        model = self.model
        for name, param in model.named_parameters():
            if param.requires_grad == True and except_condition not in name:
                param.requires_grad = False

    def forward(self, source_id, source_mask, target_id, target_mask):
        if self.args.mlp_type == "none":
            prompt = self.model.prompt
        else:
            mlp = self.prefix_mlp
            prompt = mlp(self.model.prompt)

        k = source_id.shape[0]
        input_embeds = self.model.shared(source_id)

        inputs_embeds = torch.cat((prompt.repeat(k, 1, 1),
                                   input_embeds),
                                  dim=1)
        full_prefix_len = prompt.shape[0]
        source_mask_updated = torch.cat((source_mask[0][0].repeat(k, full_prefix_len),
                                         source_mask),
                                        dim=1)

        encoder_outputs = self.model.encoder(
            attention_mask=source_mask_updated,
            inputs_embeds=inputs_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        outputs = self.model(
            input_ids=source_id,
            attention_mask=source_mask_updated,
            labels=target_id,
            decoder_attention_mask=target_mask,
            encoder_outputs=encoder_outputs,
        )

        loss = outputs[0]

        return_prompt = prompt
        # return_prompt = torch.cat((new_prompt, self.previous_prompts), dim=0)

        return loss, return_prompt

    def generate_preds(self, source_id, source_mask, input_prompt):
        prompt = input_prompt

        k = source_id.shape[0]
        input_embeds = self.model.encoder.embed_tokens(source_id)

        inputs_embeds = torch.cat((prompt.repeat(k, 1, 1), input_embeds), dim=1)
        full_prefix_len = prompt.shape[0]
        source_mask_updated = torch.cat((source_mask[0][0].repeat(k, full_prefix_len), source_mask), dim=1)

        encoder_outputs = self.model.encoder(
            attention_mask=source_mask_updated,
            inputs_embeds=inputs_embeds,
            head_mask=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        preds = self.model.generate(
            input_ids=source_id,
            attention_mask=source_mask_updated,
            encoder_outputs=encoder_outputs,
            max_length=self.args.max_output_tokens,
            num_beams=self.args.beam_size,
        )

        top_preds = list(preds.cpu().numpy())

        return top_preds
