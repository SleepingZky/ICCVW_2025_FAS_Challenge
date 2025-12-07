from .clip import clip 
from PIL import Image
import torch.nn as nn
from .lora import apply_lora_to_linear_layers, get_lora_params
import torch

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
    def __init__(self, name, num_classes=1, lora_rank=4, lora_alpha=1.0, lora_targets=None):
        super(CLIPModelWithLoRA, self).__init__()

        self.model, self.preprocess = clip.load(name, device="cpu")
        
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
        clip_features = self.model.encode_image(x)  # Shape: [batch_size, original_dim]
        
        # Project to 1024 dimensions to align with DINOv2-LoRA
        projected_features = self.feature_projector(clip_features)  # Shape: [batch_size, 1024]
        
        # Apply activation function for better feature representation
        projected_features = torch.relu(projected_features)
        
        # Classification output
        logits = self.fc(projected_features)
        
        if return_feature:
            return projected_features, logits  # Return 1024-dim features for supervision
        return logits

    def get_clip_features(self, x):
        """Helper method to get original CLIP features if needed"""
        return self.model.encode_image(x)
    
    def get_projected_features(self, x):
        """Helper method to get 1024-dim projected features"""
        clip_features = self.model.encode_image(x)
        return torch.relu(self.feature_projector(clip_features))

