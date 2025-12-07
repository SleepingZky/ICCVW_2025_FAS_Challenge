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

    def forward(self, x, return_feature=False):
        # Extract CLIP visual features (original dimension)
        
        x = self.model.visual.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
        x = torch.cat([self.model.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.model.visual.positional_embedding.to(x.dtype)
        x = self.model.visual.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x_lora = x.clone()
        for i in range(self.model.visual.transformer.layers):
            attn_mask = self.model.visual.transformer.resblocks[i].attn_mask.to(dtype=x.dtype, device=x.device) if self.model.visual.transformer.resblocks[i].attn_mask is not None else None
        
            x = x + self.model.visual.transformer.resblocks[i].attn(self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), self.model.visual.transformer.resblocks[i].ln_1(x), need_weights=False, attn_mask=attn_mask)[0]
            x_lora = x_lora + self.model.visual.transformer.resblocks[i].attn(self.model.visual.transformer.resblocks[i].ln_1(x_lora), self.model.visual.transformer.resblocks[i].ln_1(x_lora), self.model.visual.transformer.resblocks[i].ln_1(x_lora), need_weights=False, attn_mask=attn_mask)[0]
            
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
