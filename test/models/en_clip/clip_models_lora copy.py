from .english_clip import load 
from PIL import Image
import torch.nn as nn
from .lora import apply_lora_to_linear_layers, get_lora_params
import torch
import torch.nn.functional as F
from pdb import set_trace as st

CHANNELS = {
    "RN50": 1024,
    "RN101": 1024,
    "RN50x4": 640,
    "RN50x16": 768,
    "RN50x64": 1024,
    "ViT-B/32": 512,
    "ViT-B/16": 512,
    "ViT-L/14": 768,
    "ViT-L/14@336px": 768
}


class CLIPModelWithLoRA(nn.Module):
    def __init__(self, name, num_classes=1, lora_rank=8, lora_alpha=1.0, lora_targets=None):
        super(CLIPModelWithLoRA, self).__init__()

        self.model, self.preprocess = load(name, device="cpu")
        
        # CRITICAL: Freeze ALL CLIP parameters first
        for param in self.model.parameters():
            param.requires_grad = False
        
        # Get original CLIP feature dimension
        self.original_dim = CHANNELS[name]
        
        # Add a projection layer to map CLIP features to 1024 dimensions
        # This aligns with DINOv2-LoRA's 1024-dim features
        self.feature_projector = nn.Linear(self.original_dim, 1024)
        
        # Final classification layer uses 1024-dim features
        self.fc = nn.Linear(1024, num_classes)
        
        if lora_targets is None:
            if 'ViT' in name:  
                lora_targets = ['attn.in_proj', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']
            else:  
                lora_targets = ['attnpool']
        
        # Apply LoRA to CLIP visual encoder AFTER freezing
        self.model.visual = apply_lora_to_linear_layers(
            self.model.visual, 
            rank=lora_rank, 
            alpha=lora_alpha,
            target_modules=lora_targets
        )
        
        print(f"CLIP model frozen, LoRA applied to visual encoder with targets: {lora_targets}")
    
    def get_trainable_params(self):
        # Get LoRA parameters from visual encoder
        lora_params = get_lora_params(self.model.visual)
        
        # Get parameters from feature projector and final classifier
        projector_params = list(self.feature_projector.parameters())
        fc_params = list(self.fc.parameters())
        
        # Convert generator to list if needed
        if hasattr(lora_params, '__iter__') and not isinstance(lora_params, list):
            lora_params = list(lora_params)
        
        all_params = lora_params + projector_params + fc_params
        
        print(f"Trainable parameters: LoRA ({len(lora_params)}), Projector ({len(projector_params)}), FC ({len(fc_params)})")
        
        return all_params

    # def forward_attn(self, 
    #         query,
    #         key,
    #         value,
    #         key_padding_mask,
    #         need_weights,
    #         attn_mask,
    #         average_attn_weights,
    #         idx):
    #     is_batched = query.dim() == 3
    #     if key_padding_mask is not None:
    #         _kpm_dtype = key_padding_mask.dtype
    #         if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
    #             raise AssertionError(
    #                 "only bool and floating types of key_padding_mask are supported")
    #     why_not_fast_path = ''
    #     if not is_batched:
    #         why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
    #     elif query is not key or key is not value:
    #         # When lifting this restriction, don't forget to either
    #         # enforce that the dtypes all match or test cases where
    #         # they don't!
    #         why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
    #     elif self.model.visual.transformer.resblocks[idx].attn.in_proj_bias is not None and query.dtype != self.model.visual.transformer.resblocks[idx].attn.in_proj_bias.dtype:
    #         why_not_fast_path = f"dtypes of query ({query.dtype}) and self.model.visual.transformer.resblocks[idx].attn.in_proj_bias ({self.model.visual.transformer.resblocks[idx].attn.in_proj_bias.dtype}) don't match"
    #     elif self.model.visual.transformer.resblocks[idx].attn.in_proj_weight is not None and query.dtype != self.model.visual.transformer.resblocks[idx].attn.in_proj_weight.dtype:
    #         # this case will fail anyway, but at least they'll get a useful error message.
    #         why_not_fast_path = f"dtypes of query ({query.dtype}) and self.model.visual.transformer.resblocks[idx].attn.in_proj_weight ({self.model.visual.transformer.resblocks[idx].attn.in_proj_weight.dtype}) don't match"
    #     elif self.model.visual.transformer.resblocks[idx].attn.training:
    #         why_not_fast_path = "training is enabled"
    #     elif not self.model.visual.transformer.resblocks[idx].attn.batch_first:
    #         why_not_fast_path = "batch_first was not True"
    #     elif self.model.visual.transformer.resblocks[idx].attn.bias_k is not None:
    #         why_not_fast_path = "self.model.visual.transformer.resblocks[idx].attn.bias_k was not None"
    #     elif self.model.visual.transformer.resblocks[idx].attn.bias_v is not None:
    #         why_not_fast_path = "self.model.visual.transformer.resblocks[idx].attn.bias_v was not None"
    #     elif self.model.visual.transformer.resblocks[idx].attn.dropout:
    #         why_not_fast_path = f"dropout was {self.model.visual.transformer.resblocks[idx].attn.dropout}, required zero"
    #     elif self.model.visual.transformer.resblocks[idx].attn.add_zero_attn:
    #         why_not_fast_path = "add_zero_attn was enabled"
    #     elif not self.model.visual.transformer.resblocks[idx].attn._qkv_same_embed_dim:
    #         why_not_fast_path = "_qkv_same_embed_dim was not True"
    #     elif attn_mask is not None:
    #         why_not_fast_path = "attn_mask was not None"
    #     elif query.is_nested and key_padding_mask is not None:
    #         why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
    #     elif self.model.visual.transformer.resblocks[idx].attn.num_heads % 2 == 1:
    #         why_not_fast_path = "num_heads is odd"
    #     elif torch.is_autocast_enabled():
    #         why_not_fast_path = "autocast is enabled"

    #     if not why_not_fast_path:
    #         tensor_args = (
    #             query,
    #             key,
    #             value,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_weight,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #             self.model.visual.transformer.resblocks[idx].attn.out_proj.weight,
    #             self.model.visual.transformer.resblocks[idx].attn.out_proj.bias,
    #         )
    #         # We have to use list comprehensions below because TorchScript does not support
    #         # generator expressions.
    #         if torch.overrides.has_torch_function(tensor_args):
    #             why_not_fast_path = "some Tensor argument has_torch_function"
    #         elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
    #             why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
    #         elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
    #             why_not_fast_path = ("grad is enabled and at least one of query or the "
    #                                  "input/output projection weights or biases requires_grad")
    #         if not why_not_fast_path:
    #             return torch._native_multi_head_attention(
    #                 query,
    #                 key,
    #                 value,
    #                 self.model.visual.transformer.resblocks[idx].attn.embed_dim,
    #                 self.model.visual.transformer.resblocks[idx].attn.num_heads,
    #                 self.model.visual.transformer.resblocks[idx].attn.in_proj_weight,
    #                 self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #                 self.model.visual.transformer.resblocks[idx].attn.out_proj.weight,
    #                 self.model.visual.transformer.resblocks[idx].attn.out_proj.bias,
    #                 key_padding_mask if key_padding_mask is not None else attn_mask,
    #                 need_weights,
    #                 average_attn_weights,
    #                 1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

    #     any_nested = query.is_nested or key.is_nested or value.is_nested
    #     assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
    #                             f"The fast path was not hit because {why_not_fast_path}")

    #     if self.model.visual.transformer.resblocks[idx].attn.batch_first and is_batched:
    #         # make sure that the transpose op does not affect the "is" property
    #         if key is value:
    #             if query is key:
    #                 query = key = value = query.transpose(1, 0)
    #             else:
    #                 query, key = [x.transpose(1, 0) for x in (query, key)]
    #                 value = key
    #         else:
    #             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    #     if not self.model.visual.transformer.resblocks[idx].attn._qkv_same_embed_dim:
    #         attn_output, attn_output_weights = F.multi_head_attention_forward(
    #             query, key, value, self.model.visual.transformer.resblocks[idx].attn.embed_dim, self.model.visual.transformer.resblocks[idx].attn.num_heads,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_weight, self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #             self.model.visual.transformer.resblocks[idx].attn.bias_k, self.model.visual.transformer.resblocks[idx].attn.bias_v, self.model.visual.transformer.resblocks[idx].attn.add_zero_attn,
    #             self.model.visual.transformer.resblocks[idx].attn.dropout, self.model.visual.transformer.resblocks[idx].attn.out_proj.weight, self.model.visual.transformer.resblocks[idx].attn.out_proj.bias,
    #             training=self.model.visual.transformer.resblocks[idx].attn.training,
    #             key_padding_mask=key_padding_mask, need_weights=need_weights,
    #             attn_mask=attn_mask, use_separate_proj_weight=True,
    #             q_proj_weight=self.model.visual.transformer.resblocks[idx].attn.q_proj_weight, k_proj_weight=self.model.visual.transformer.resblocks[idx].attn.k_proj_weight,
    #             v_proj_weight=self.model.visual.transformer.resblocks[idx].attn.v_proj_weight, average_attn_weights=average_attn_weights)
    #     else:
    #         attn_output, attn_output_weights = F.multi_head_attention_forward(
    #             query, key, value, self.model.visual.transformer.resblocks[idx].attn.embed_dim, self.model.visual.transformer.resblocks[idx].attn.num_heads,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_weight, self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #             self.model.visual.transformer.resblocks[idx].attn.bias_k, self.model.visual.transformer.resblocks[idx].attn.bias_v, self.model.visual.transformer.resblocks[idx].attn.add_zero_attn,
    #             self.model.visual.transformer.resblocks[idx].attn.dropout, self.model.visual.transformer.resblocks[idx].attn.out_proj.weight, self.model.visual.transformer.resblocks[idx].attn.out_proj.bias,
    #             training=self.model.visual.transformer.resblocks[idx].attn.training,
    #             key_padding_mask=key_padding_mask, need_weights=need_weights,
    #             attn_mask=attn_mask, average_attn_weights=average_attn_weights)
    #     if self.model.visual.transformer.resblocks[idx].attn.batch_first and is_batched:
    #         return attn_output.transpose(1, 0), attn_output_weights
    #     else:
    #         return attn_output, attn_output_weights

    # def forward_attn_original(self, 
    #         query,
    #         key,
    #         value,
    #         key_padding_mask,
    #         need_weights,
    #         attn_mask,
    #         average_attn_weights,
    #         idx):
    #     is_batched = query.dim() == 3
    #     if key_padding_mask is not None:
    #         _kpm_dtype = key_padding_mask.dtype
    #         if _kpm_dtype != torch.bool and not torch.is_floating_point(key_padding_mask):
    #             raise AssertionError(
    #                 "only bool and floating types of key_padding_mask are supported")
    #     why_not_fast_path = ''
    #     if not is_batched:
    #         why_not_fast_path = f"input not batched; expected query.dim() of 3 but got {query.dim()}"
    #     elif query is not key or key is not value:
    #         # When lifting this restriction, don't forget to either
    #         # enforce that the dtypes all match or test cases where
    #         # they don't!
    #         why_not_fast_path = "non-self attention was used (query, key, and value are not the same Tensor)"
    #     elif self.model.visual.transformer.resblocks[idx].attn.in_proj_bias is not None and query.dtype != self.model.visual.transformer.resblocks[idx].attn.in_proj_bias.dtype:
    #         why_not_fast_path = f"dtypes of query ({query.dtype}) and self.model.visual.transformer.resblocks[idx].attn.in_proj_bias ({self.model.visual.transformer.resblocks[idx].attn.in_proj_bias.dtype}) don't match"
    #     elif self.model.visual.transformer.resblocks[idx].attn.in_proj_weight is not None and query.dtype != self.model.visual.transformer.resblocks[idx].attn.in_proj_weight.dtype:
    #         # this case will fail anyway, but at least they'll get a useful error message.
    #         why_not_fast_path = f"dtypes of query ({query.dtype}) and self.model.visual.transformer.resblocks[idx].attn.in_proj_weight ({self.model.visual.transformer.resblocks[idx].attn.in_proj_weight.dtype}) don't match"
    #     elif self.model.visual.transformer.resblocks[idx].attn.training:
    #         why_not_fast_path = "training is enabled"
    #     elif not self.model.visual.transformer.resblocks[idx].attn.batch_first:
    #         why_not_fast_path = "batch_first was not True"
    #     elif self.model.visual.transformer.resblocks[idx].attn.bias_k is not None:
    #         why_not_fast_path = "self.model.visual.transformer.resblocks[idx].attn.bias_k was not None"
    #     elif self.model.visual.transformer.resblocks[idx].attn.bias_v is not None:
    #         why_not_fast_path = "self.model.visual.transformer.resblocks[idx].attn.bias_v was not None"
    #     elif self.model.visual.transformer.resblocks[idx].attn.dropout:
    #         why_not_fast_path = f"dropout was {self.model.visual.transformer.resblocks[idx].attn.dropout}, required zero"
    #     elif self.model.visual.transformer.resblocks[idx].attn.add_zero_attn:
    #         why_not_fast_path = "add_zero_attn was enabled"
    #     elif not self.model.visual.transformer.resblocks[idx].attn._qkv_same_embed_dim:
    #         why_not_fast_path = "_qkv_same_embed_dim was not True"
    #     elif attn_mask is not None:
    #         why_not_fast_path = "attn_mask was not None"
    #     elif query.is_nested and key_padding_mask is not None:
    #         why_not_fast_path = "key_padding_mask is not supported with NestedTensor input"
    #     elif self.model.visual.transformer.resblocks[idx].attn.num_heads % 2 == 1:
    #         why_not_fast_path = "num_heads is odd"
    #     elif torch.is_autocast_enabled():
    #         why_not_fast_path = "autocast is enabled"

    #     if not why_not_fast_path:
    #         tensor_args = (
    #             query,
    #             key,
    #             value,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_weight,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #             self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.weight,
    #             self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.bias,
    #         )
    #         # We have to use list comprehensions below because TorchScript does not support
    #         # generator expressions.
    #         if torch.overrides.has_torch_function(tensor_args):
    #             why_not_fast_path = "some Tensor argument has_torch_function"
    #         elif not all([(x is None or x.is_cuda or 'cpu' in str(x.device)) for x in tensor_args]):
    #             why_not_fast_path = "some Tensor argument is neither CUDA nor CPU"
    #         elif torch.is_grad_enabled() and any([x is not None and x.requires_grad for x in tensor_args]):
    #             why_not_fast_path = ("grad is enabled and at least one of query or the "
    #                                  "input/output projection weights or biases requires_grad")
    #         if not why_not_fast_path:
    #             return torch._native_multi_head_attention(
    #                 query,
    #                 key,
    #                 value,
    #                 self.model.visual.transformer.resblocks[idx].attn.embed_dim,
    #                 self.model.visual.transformer.resblocks[idx].attn.num_heads,
    #                 self.model.visual.transformer.resblocks[idx].attn.in_proj_weight,
    #                 self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #                 self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.weight,
    #                 self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.bias,
    #                 key_padding_mask if key_padding_mask is not None else attn_mask,
    #                 need_weights,
    #                 average_attn_weights,
    #                 1 if key_padding_mask is not None else 0 if attn_mask is not None else None)

    #     any_nested = query.is_nested or key.is_nested or value.is_nested
    #     assert not any_nested, ("MultiheadAttention does not support NestedTensor outside of its fast path. " +
    #                             f"The fast path was not hit because {why_not_fast_path}")

    #     if self.model.visual.transformer.resblocks[idx].attn.batch_first and is_batched:
    #         # make sure that the transpose op does not affect the "is" property
    #         if key is value:
    #             if query is key:
    #                 query = key = value = query.transpose(1, 0)
    #             else:
    #                 query, key = [x.transpose(1, 0) for x in (query, key)]
    #                 value = key
    #         else:
    #             query, key, value = [x.transpose(1, 0) for x in (query, key, value)]

    #     if not self.model.visual.transformer.resblocks[idx].attn._qkv_same_embed_dim:
    #         attn_output, attn_output_weights = F.multi_head_attention_forward(
    #             query, key, value, self.model.visual.transformer.resblocks[idx].attn.embed_dim, self.model.visual.transformer.resblocks[idx].attn.num_heads,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_weight, self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #             self.model.visual.transformer.resblocks[idx].attn.bias_k, self.model.visual.transformer.resblocks[idx].attn.bias_v, self.model.visual.transformer.resblocks[idx].attn.add_zero_attn,
    #             self.model.visual.transformer.resblocks[idx].attn.dropout, self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.weight, self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.bias,
    #             training=self.model.visual.transformer.resblocks[idx].attn.training,
    #             key_padding_mask=key_padding_mask, need_weights=need_weights,
    #             attn_mask=attn_mask, use_separate_proj_weight=True,
    #             q_proj_weight=self.model.visual.transformer.resblocks[idx].attn.q_proj_weight, k_proj_weight=self.model.visual.transformer.resblocks[idx].attn.k_proj_weight,
    #             v_proj_weight=self.model.visual.transformer.resblocks[idx].attn.v_proj_weight, average_attn_weights=average_attn_weights)
    #     else:
    #         attn_output, attn_output_weights = F.multi_head_attention_forward(
    #             query, key, value, self.model.visual.transformer.resblocks[idx].attn.embed_dim, self.model.visual.transformer.resblocks[idx].attn.num_heads,
    #             self.model.visual.transformer.resblocks[idx].attn.in_proj_weight, self.model.visual.transformer.resblocks[idx].attn.in_proj_bias,
    #             self.model.visual.transformer.resblocks[idx].attn.bias_k, self.model.visual.transformer.resblocks[idx].attn.bias_v, self.model.visual.transformer.resblocks[idx].attn.add_zero_attn,
    #             self.model.visual.transformer.resblocks[idx].attn.dropout, self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.weight, self.model.visual.transformer.resblocks[idx].attn.out_proj.original_layer.bias,
    #             training=self.model.visual.transformer.resblocks[idx].attn.training,
    #             key_padding_mask=key_padding_mask, need_weights=need_weights,
    #             attn_mask=attn_mask, average_attn_weights=average_attn_weights)
    #     if self.model.visual.transformer.resblocks[idx].attn.batch_first and is_batched:
    #         return attn_output.transpose(1, 0), attn_output_weights
    #     else:
    #         return attn_output, attn_output_weights

    def forward(self, x, return_feature=False):
        # Extract CLIP visual features (original dimension)
        # clip_features = self.model.encode_image(x)  # Shape: [batch_size, original_dim]
        
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x_lora = x.clone()
        x_copy = x.clone()
        for i in range(self.model.visual.transformer.layers):
            # x = self.model.visual.transformer.resblocks[i].ln_1(x)
            # x = x + self.model.visual.transformer.resblocks[i].attention(self.model.visual.transformer.resblocks[i].ln_1(x))
            attn_mask = self.model.visual.transformer.resblocks[i].attn_mask.to(dtype=x.dtype, device=x.device) if self.model.visual.transformer.resblocks[i].attn_mask is not None else None
            
            # # 等价于self.model.visual.transformer.resblocks[i].attn(self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), need_weights=False, attn_mask=attn_mask)[0]
            # x_2 = x.clone()
            # x = self.model.visual.transformer.resblocks[i].ln_1(x)
            
            # x = x_2 + self.model.visual.transformer.resblocks[i].attn
            
            # x = x + self.forward_attn(self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), need_weights=False, attn_mask=attn_mask, key_padding_mask=None, average_attn_weights=True, idx=i)[0]
            # x_copy = x_copy + self.forward_attn_original(self.model.visual.transformer.resblocks[i].ln_1(x_copy), self.model.visual.transformer.resblocks[i].ln_1(x_copy), self.model.visual.transformer.resblocks[i].ln_1(x_copy), need_weights=False, attn_mask=attn_mask, key_padding_mask=None, average_attn_weights=True, idx=i)[0]
            # st()
            x = x + self.model.visual.transformer.resblocks[i].attn(self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), need_weights=False, attn_mask=attn_mask)[0]
            x_lora = x_lora + self.model.visual.transformer.resblocks[i].attn(self.model.visual.transformer.resblocks[i].ln_1(x_lora), self.model.visual.transformer.resblocks[i].ln_1(x_lora), self.model.visual.transformer.resblocks[i].ln_1(x_lora), need_weights=False, attn_mask=attn_mask)[0]
            
            # 等价于self.model.visual.transformer.resblocks[i].mlp(self.model.visual.transformer.resblocks[i].ln_2(x))
            x_1 = x.clone()
            x_lora_1 = x_lora.clone()
            x = self.model.visual.transformer.resblocks[i].ln_2(x)
            x_lora = self.model.visual.transformer.resblocks[i].ln_2(x_lora)
            x = self.model.visual.transformer.resblocks[i].mlp[0].original_layer(x)
            x_lora = self.model.visual.transformer.resblocks[i].mlp[0](x_lora)
            x = self.model.visual.transformer.resblocks[i].mlp[1](x)
            x_lora = self.model.visual.transformer.resblocks[i].mlp[1](x_lora)
            x = self.model.visual.transformer.resblocks[i].mlp[2].original_layer(x)
            x_lora = self.model.visual.transformer.resblocks[i].mlp[2](x_lora)
            x = x_1 + x
            x_lora = x_lora_1 + x_lora
            
            # x = x + self.model.visual.transformer.resblocks[i].mlp(self.model.visual.transformer.resblocks[i].ln_2(x))
            
        # x_copy = self.model.visual.transformer(x_copy)
        # st()
        x = x.permute(1, 0, 2)  # LND -> NLD
        x_lora = x_lora.permute(1, 0, 2)

        x = self.model.visual.ln_post(x[:, 0, :])
        x_lora = self.model.visual.ln_post(x_lora[:, 0, :])

        if self.model.visual.proj is not None:
            x = x @ self.model.visual.proj
            x_lora = x_lora @ self.model.visual.proj
        
        clip_features = x_lora
        
        # Project to 1024 dimensions to align with DINOv2-LoRA
        projected_features = self.feature_projector(clip_features)  # Shape: [batch_size, 1024]
        
        # Apply activation function for better feature representation
        projected_features = torch.relu(projected_features)
        
        # Classification output
        logits = self.fc(projected_features)
        
        if return_feature:
            return projected_features, logits, x  # Return 1024-dim features for supervision
        return logits, x

    def get_clip_features(self, x):
        """Helper method to get original CLIP features if needed"""
        return self.model.encode_image(x)
    
    def get_projected_features(self, x):
        """Helper method to get 1024-dim projected features"""
        clip_features = self.model.encode_image(x)
        return torch.relu(self.feature_projector(clip_features))



# class CLIPModelWithLoRA(nn.Module):
#     def __init__(self, name, num_classes=1, lora_rank=4, lora_alpha=1.0, lora_targets=None):
#         super(CLIPModelWithLoRA, self).__init__()

        
#         self.model, self.preprocess = clip.load(name, device="cpu")
#         self.fc = nn.Linear(CHANNELS[name], num_classes)
        
#         if lora_targets is None:
            
#             if 'ViT' in name:  
#                 lora_targets = ['attn.in_proj', 'attn.out_proj', 'mlp.c_fc', 'mlp.c_proj']
#             else:  
#                 lora_targets = ['attnpool']
        
        
#         self.model.visual = apply_lora_to_linear_layers(
#             self.model.visual, 
#             rank=lora_rank, 
#             alpha=lora_alpha,
#             target_modules=lora_targets
#         )
    
#     def get_trainable_params(self):
        
#         lora_params = get_lora_params(self.model.visual)
#         fc_params = self.fc.parameters()
#         return list(lora_params) + list(fc_params)

#     def forward(self, x, return_feature=False):
#         features = self.model.encode_image(x) 
#         if return_feature:
#             return features, self.fc(features)
#         return self.fc(features)
