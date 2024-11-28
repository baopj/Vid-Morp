import torch, clip, math
import numpy as np
import models.loc_modules as loc_modules
import torch.nn.functional as F
from torch import nn
from IPython import embed

class TextEncoder(nn.Module):
    def __init__(self, cfg, clip_model):
        super().__init__()
        self.token_embedding = clip_model.token_embedding
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        self.dtype = torch.float

        n_cls = 1
        n_ctx = cfg['prompt']['n_ctx']
        ctx_init_txt = cfg['prompt']['ctx_init_txt']
        ctx_init_std = cfg['prompt']['ctx_init_std']
        dtype = torch.float
        ctx_dim = clip_model.ln_final.weight.shape[0]

        if len(ctx_init_txt) > 0:
            # use given words to initialize context vectors
            ctx_init_txt = ctx_init_txt.replace("_", " ")
            n_ctx = len(ctx_init_txt.split(" "))
            prompt = clip.tokenize(ctx_init_txt).cuda()
            with torch.no_grad():
                embedding = clip_model.token_embedding(prompt).type(dtype)
            ctx_vectors = embedding[0, 1 : 1 + n_ctx, :]
        self.ctx = ctx_vectors
        self.ctx = nn.Parameter(ctx_vectors)
        self.ctx.requires_grad = False

        self.n_ctx = n_ctx

    def forward(self, text):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]
        prefix = x[:, :1]
        suffix = x[:, 1:]
        x = torch.cat(
            [prefix, self.ctx[None, :].repeat(len(text), 1, 1), suffix], 1
            # [prefix, suffix], 1
        )
        x = x + self.positional_embedding.type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), self.n_ctx + text.argmax(dim=-1)] @ self.text_projection
        return x



class SimBase(nn.Module):
    def __init__(self, cfg):
        super(SimBase, self).__init__()
        clip_model, _ = clip.load("ViT-L/14")
        del clip_model.visual
        for paramclip in clip_model.parameters():
            paramclip.requires_grad = False
        self.clip_model = clip_model
        self.textual_encoder = TextEncoder(cfg['textual_converter'], clip_model)
        self.cfg = cfg

        self.cfg_anchor = cfg['simbase']['anchor']
        self.localization = loc_modules.MultiScaleLocalization(cfg)

        self.linear = nn.Linear(768, cfg['textual_converter']['reduction']['output_size'])
        self.reduction = nn.Linear(cfg['textual_converter']['reduction']['input_size'], cfg['textual_converter']['reduction']['output_size'])

        self.conv1 = nn.Conv1d(768, cfg['textual_converter']['reduction']['input_size'], cfg['visual_converter']['conv_kernel_size'], padding=cfg['visual_converter']['conv_kernel_size']//2)
        self.criterion_reg = torch.nn.SmoothL1Loss(reduction='sum')

    # Define a method to save only non-CLIP parameters (including BatchNorm stats)
    def save_non_clip_parameters(self, path):
        # Collect all parameters except those from the CLIP model
        non_clip_params = {}
        for name, param in self.named_parameters():
            # Check if the parameter comes from the CLIP model
            if 'clip_model' not in name:
                non_clip_params[name] = param

        # Handle BatchNorm layers: also save running_mean and running_var
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                non_clip_params[name + '.running_mean'] = module.running_mean
                non_clip_params[name + '.running_var'] = module.running_var

        # Save the non-CLIP parameters (including BatchNorm stats)
        torch.save(non_clip_params, path)

    # Define a method to load only non-CLIP parameters, including handling of BatchNorm layers
    def load_non_clip_parameters(self, path):
        # Load the saved non-CLIP parameters
        saved_params = torch.load(path)

        # Get current model parameters
        model_dict = self.state_dict()

        # Filter out parameters related to the CLIP model and update with the saved non-CLIP params
        non_clip_params = {k: v for k, v in saved_params.items() if k in model_dict}

        # Update the current state_dict with non-CLIP parameters
        model_dict.update(non_clip_params)

        # Load the updated state dict into the model
        self.load_state_dict(model_dict, strict=False)  # strict=False to allow missing params (BN stats)

        # Handle BatchNorm layers: ensure running_mean and running_var are handled
        for name, module in self.named_modules():
            if isinstance(module, torch.nn.BatchNorm1d) or isinstance(module, torch.nn.BatchNorm2d):
                if name + '.running_mean' in saved_params:
                    module.running_mean.data.copy_(saved_params[name + '.running_mean'])
                if name + '.running_var' in saved_params:
                    module.running_var.data.copy_(saved_params[name + '.running_var'])

    def forward(self, sentence, videoFeat, gt_simbase, **kwargs):
        text_token = clip.tokenize(sentence, 77 - self.textual_encoder.n_ctx).to('cuda')
        text_feat = self.textual_encoder(text_token)
        videoFeat = self.conv1(videoFeat.transpose(1, 2)).transpose(1, 2)
        if self.cfg['textual_converter']['prompt']['reduce_flag']:
            text_feat = self.linear(text_feat)
            videoFeat = self.reduction(videoFeat) # videoFeat: (b,seq_v,512)
        fusion_feat = videoFeat * text_feat[:, None]
        overlap, reg = self.localization(fusion_feat.permute(0, 2, 1))
        ret_dict = {
            'pred': {
                'pred_overlap': overlap,
                'pred_reg': reg,
            },
        }
        return ret_dict

