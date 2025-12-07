import os
import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
from torch.nn.parallel import DistributedDataParallel as DDP

class BaseModel(nn.Module):
    def __init__(self, opt):
        super(BaseModel, self).__init__()
        self.opt = opt
        self.total_steps = 0
        self.save_dir = os.path.join(opt.name, opt.checkpoints_dir)
        os.makedirs(self.save_dir, exist_ok=True)
        

        if hasattr(opt, 'device'):
            self.device = opt.device
        else:
            self.device = torch.device('cuda:{}'.format(opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')

    def save_networks(self, save_filename, save_mode='full'):
        """
        Save model networks with different modes.
        
        Args:
            save_filename (str): Filename to save the checkpoint
            save_mode (str): Saving mode - 'full', 'trainable_only', or 'auto'
                - 'full': Save all model parameters (default)
                - 'trainable_only': Save only trainable parameters (useful for LoRA)
                - 'auto': Automatically choose based on model type
        """
        save_path = os.path.join(self.save_dir, save_filename)

        model_to_save = self.model
        if isinstance(self.model, DDP):
            model_to_save = self.model.module
        
        # Determine save mode automatically if requested
        if save_mode == 'auto':
            # Check if this is a LoRA model or has significantly fewer trainable parameters
            total_params = sum(p.numel() for p in model_to_save.parameters())
            trainable_params = sum(p.numel() for p in model_to_save.parameters() if p.requires_grad)
            trainable_ratio = trainable_params / total_params if total_params > 0 else 0
            
            # Use trainable_only mode if less than 50% of parameters are trainable
            save_mode = 'trainable_only' if trainable_ratio < 0.5 else 'full'
            print(f"Auto-selected save mode: {save_mode} (trainable ratio: {trainable_ratio:.2%})")
        
        # Prepare state dict based on save mode
        if save_mode == 'trainable_only':
            # Save only trainable parameters
            trainable_param_names = {name for name, p in model_to_save.named_parameters() if p.requires_grad}
            model_state_dict = {
                name: param for name, param in model_to_save.state_dict().items()
                if name in trainable_param_names
            }
            print(f"Saving {len(model_state_dict)} trainable parameter groups out of {len(model_to_save.state_dict())} total")
        else:
            # Save all parameters (full mode)
            model_state_dict = model_to_save.state_dict()
            print(f"Saving full model state dict with {len(model_state_dict)} parameter groups")
            
        state_dict = {
            'model': model_state_dict,
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'save_mode': save_mode,  # Record how this was saved
            'model_arch': getattr(self.opt, 'arch', 'unknown'),  # Record model architecture
        }

        torch.save(state_dict, save_path)
        print(f"Checkpoint saved to: {save_path}")

    def load_networks(self, load_filename, strict=True):
        """
        Load model networks from checkpoint.
        
        Args:
            load_filename (str): Filename to load the checkpoint from
            strict (bool): Whether to strictly enforce that the keys match
        
        Returns:
            dict: Loaded checkpoint information
        """
        load_path = os.path.join(self.save_dir, load_filename)
        
        if not os.path.exists(load_path):
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")
        
        print(f"Loading checkpoint from: {load_path}")
        
        # Load checkpoint
        checkpoint = torch.load(load_path, map_location=self.device)
        
        # Get model to load into
        model_to_load = self.model
        if isinstance(self.model, DDP):
            model_to_load = self.model.module
        
        # Load model state dict
        if 'model' in checkpoint:
            missing_keys, unexpected_keys = model_to_load.load_state_dict(
                checkpoint['model'], strict=strict
            )
            
            if missing_keys:
                print(f"Warning: Missing keys when loading model: {missing_keys}")
            if unexpected_keys:
                print(f"Warning: Unexpected keys when loading model: {unexpected_keys}")
        
        # Load optimizer state dict if available
        if 'optimizer' in checkpoint and hasattr(self, 'optimizer'):
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
                print("Optimizer state loaded successfully")
            except Exception as e:
                print(f"Warning: Could not load optimizer state: {e}")
        
        # Load other metadata
        if 'total_steps' in checkpoint:
            self.total_steps = checkpoint['total_steps']
            print(f"Resumed from step: {self.total_steps}")
        
        # Print checkpoint information
        save_mode = checkpoint.get('save_mode', 'unknown')
        model_arch = checkpoint.get('model_arch', 'unknown')
        print(f"Checkpoint info - Save mode: {save_mode}, Architecture: {model_arch}")
        
        return checkpoint

    def get_model_info(self):
        """
        Get information about the current model.
        
        Returns:
            dict: Model information including parameter counts
        """
        model_to_check = self.model
        if isinstance(self.model, DDP):
            model_to_check = self.model.module
        
        total_params = sum(p.numel() for p in model_to_check.parameters())
        trainable_params = sum(p.numel() for p in model_to_check.parameters() if p.requires_grad)
        
        trainable_param_names = [
            name for name, p in model_to_check.named_parameters() if p.requires_grad
        ]
        
        info = {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'trainable_ratio': trainable_params / total_params if total_params > 0 else 0,
            'trainable_parameter_names': trainable_param_names,
            'model_architecture': getattr(self.opt, 'arch', 'unknown'),
        }
        
        return info

    def print_model_info(self):
        """Print detailed model information."""
        info = self.get_model_info()
        
        print("\n" + "="*50)
        print("MODEL INFORMATION")
        print("="*50)
        print(f"Architecture: {info['model_architecture']}")
        print(f"Total parameters: {info['total_parameters']:,}")
        print(f"Trainable parameters: {info['trainable_parameters']:,}")
        print(f"Trainable ratio: {info['trainable_ratio']:.2%}")
        print(f"Training mode: {'Full Fine-tuning' if info['trainable_ratio'] > 0.8 else 'Partial Training'}")
        
        if len(info['trainable_parameter_names']) <= 20:
            print(f"\nTrainable parameters:")
            for name in info['trainable_parameter_names']:
                print(f"  - {name}")
        else:
            print(f"\nTrainable parameters (showing first 10 of {len(info['trainable_parameter_names'])}):")
            for name in info['trainable_parameter_names'][:10]:
                print(f"  - {name}")
            print(f"  ... and {len(info['trainable_parameter_names']) - 10} more")
        print("="*50 + "\n")

    def eval(self):
        self.model.eval()

    def train(self):
        self.model.train()

    def test(self):
        with torch.no_grad():
            self.forward()


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)