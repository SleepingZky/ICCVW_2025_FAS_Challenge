from .clip_models import CLIPModel
from .clip_models_lora import CLIPModelWithLoRA

VALID_NAMES = [
    'CLIP-LoRA:RN50',
    'CLIP-LoRA:RN101',
    'CLIP-LoRA:ViT-B/32', 
    'CLIP-LoRA:ViT-B/16', 
    'CLIP-LoRA:ViT-L/14',
    'CLIP-LoRA:ViT-L/14@336px',
]


def get_model(name, lora_rank=8, lora_alpha=1.0, lora_targets=None, huggingface_path=None):
    """
    Get a model instance based on model name and parameters.
    
    Args:
        name (str): Model name from VALID_NAMES list
        lora_rank (int, optional): Rank for LoRA models. Defaults to 8.
        lora_alpha (float, optional): Alpha value for LoRA models. Defaults to 1.0.
        lora_targets (list, optional): Target modules for LoRA. Defaults to None.
        huggingface_path (str, optional): Path to HuggingFace model. Defaults to None.
                                   For DINOv2 models: 224, 504 (register model), or 518 (standard models).
    
    Returns:
        Model instance
    """
    assert name in VALID_NAMES, f"Invalid model name: {name}. Valid names: {VALID_NAMES}"
    
    # Print initialization information
    print("\n" + "="*50)
    print(f"Initializing model: {name}")
    print(f"Parameters:")
    if 'LoRA' in name:
        print(f"  - LoRA rank: {lora_rank}")
        print(f"  - LoRA alpha: {lora_alpha}")
        if lora_targets:
            print(f"  - LoRA targets: {lora_targets}")
        else:
            print(f"  - LoRA targets: default")
    elif 'FullFinetune' in name:
        print(f"  - Training mode: Full Fine-tuning (all parameters)")
    
    
    # Print huggingface path information if applicable
    if huggingface_path:
        print(f"  - HuggingFace path: {huggingface_path}")
            
    elif name.startswith("CLIP:"):
        model_family = "CLIP"
        model_type = name[5:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")
        
        model = CLIPModel(model_type)
        print(f"  - Successfully initialized with custom image size")
        return model
     
            
    elif name.startswith("CLIP-LoRA:"):
        model_family = "CLIP with LoRA"
        model_type = name[10:]
        print(f"  - Model family: {model_family}")
        print(f"  - Model type: {model_type}")
        
        model = CLIPModelWithLoRA(model_type, lora_rank=lora_rank, lora_alpha=lora_alpha, 
                                    lora_targets=lora_targets)
        print(f"  - Successfully initialized with custom image size")
        return model

        
    else:
        raise ValueError(f"Unsupported model prefix in name: {name}")
    
    print("="*50 + "\n")