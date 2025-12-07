import functools
import torch
import torch.nn as nn
from networks.base_model import BaseModel, init_weights
import sys
from models import get_model
import torchvision.transforms.functional as F

from pytorch_metric_learning import losses


class Trainer(BaseModel):
    def name(self):
        return 'Trainer'

    def __init__(self, opt):
        super(Trainer, self).__init__(opt)
        self.opt = opt  
        
         # Add gradient accumulation parameters
        self.accumulation_steps = opt.accumulation_steps if hasattr(opt, 'accumulation_steps') else 1
        self.current_step = 0  # Track steps for gradient accumulation
        
        lora_args = {}
        if hasattr(opt, 'lora_rank'):
            lora_args['lora_rank'] = opt.lora_rank
        if hasattr(opt, 'lora_alpha'):
            lora_args['lora_alpha'] = opt.lora_alpha
        if hasattr(opt, 'lora_targets'):
            lora_args['lora_targets'] = opt.lora_targets.split(',') if opt.lora_targets else None
            
        self.model = get_model(name=opt.arch, **lora_args)
        
        # Determine training mode based on model name
        using_lora = opt.arch.startswith('CLIP-LoRA:') or opt.arch.startswith('DINOv2-LoRA:')
        using_fullfinetune = opt.arch.startswith('CLIP-FullFinetune:') or opt.arch.startswith('DINOv2-FullFinetune:')
        
        # Initialize final layer weights
        if opt.arch.startswith('DINOv2-LoRA:'):
            torch.nn.init.normal_(self.model.base_model.fc.weight.data, 0.0, opt.init_gain)
        else:
            torch.nn.init.normal_(self.model.fc.weight.data, 0.0, opt.init_gain)

        # Parameter selection based on training mode
        if opt.fix_backbone and not using_fullfinetune:
            # Fixed backbone training (original behavior)
            params = []
            for name, p in self.model.named_parameters():
                if name == "fc.weight" or name == "fc.bias": 
                    params.append(p) 
                else:
                    p.requires_grad = False
            print("Training with fixed backbone - only final layer parameters will be updated")
            
        elif using_lora:
            # LoRA training
            if hasattr(self.model, 'get_trainable_params'):
                params = self.model.get_trainable_params()
                print("Training with LoRA - only LoRA and final layer parameters will be updated")
            else:
                raise ValueError("LoRA model should have get_trainable_params method")
                
        elif using_fullfinetune:
            # Full fine-tuning (all parameters)
            params = self.model.parameters()
            print("Training with full fine-tuning - all model parameters will be updated")
            
        else:
            # Legacy behavior for regular models without explicit full fine-tuning
            if not opt.fix_backbone:
                print("Warning: Your backbone is not fixed. Are you sure you want to proceed?")
                print("If this is a mistake, enable the --fix_backbone command during training and rerun")
                print("For explicit full fine-tuning, consider using 'CLIP-FullFinetune:' or 'DINOv2-FullFinetune:' model names")
                import time 
                time.sleep(3)
                params = self.model.parameters()
            else:
                params = []
                for name, p in self.model.named_parameters():
                    if name == "fc.weight" or name == "fc.bias": 
                        params.append(p) 
                    else:
                        p.requires_grad = False
        
        # Print trainable parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print(f"\nTrainable parameters summary:")
        print(f"  - Total parameters: {total_params:,}")
        print(f"  - Trainable parameters: {trainable_params:,}")
        print(f"  - Trainable ratio: {trainable_params/total_params*100:.2f}%")
        
        print(f"\nTrainable parameter names:")
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                print(f"  - {name}: {p.numel():,} parameters")

        if opt.optim == 'adam':
            self.optimizer = torch.optim.AdamW(params, lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)
        elif opt.optim == 'sgd':
            self.optimizer = torch.optim.SGD(params, lr=opt.lr, momentum=0.0, weight_decay=opt.weight_decay)
        else:
            raise ValueError("optim should be [adam, sgd]")

        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, eta_min=opt.lr * 0.001, T_max=1000) 
        self.loss_fn = nn.BCEWithLogitsLoss()

        # Regular Contrastive Learning
        self.contrastive = opt.contrastive
        if self.contrastive:
            self.contrastive_loss_fn = losses.ContrastiveLoss(pos_margin=0.0, neg_margin=1.0)
            self.contrastive_alpha = 0.5

        if hasattr(opt, 'device'):
            self.device = opt.device
        self.model.to(self.device)

    def adjust_learning_rate(self, min_lr=1e-6):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] /= 10.
            if param_group['lr'] < min_lr:
                return False
        return True

    def set_input(self, input):
        input_stack = []
        for key in ["real", "real_resized", "fake", "fake_resized"]:
            if input[key] is not None:
                input_stack.append(input[key])
        self.input = torch.cat(input_stack, dim=0).to(self.device)

        LABELS = {
            "real": 0,
            "real_resized": 0,
            "fake": 1,
            "fake_resized": 1,
        }
        label_stack = []
        for key in ["real", "real_resized", "fake", "fake_resized"]:
            if input[key] is not None:
                label_stack += [LABELS[key]] * len(input[key])
        self.label = torch.tensor(label_stack).to(self.device).float()


    # def set_input(self, input):
    #     input_stack = []
    #     current_step = getattr(self, 'total_steps', 0)
        
    #     print(f"\nStep {current_step:>6d} - Float32 Tensor Report:")
        
    #     for key in ["real", "real_resized", "fake", "fake_resized"]:
    #         if input[key] is not None:
    #             tensor = input[key]
    #             mean_val = tensor.mean().item()
    #             std_val = tensor.std().item()
    #             shape_str = "x".join(map(str, tensor.shape))
                
    #             print(f"  {key:>12}: mean={mean_val:>12.8f}, std={std_val:>12.8f}, shape=[{shape_str}]")
                
    #             # Quick integrity check
    #             if torch.isnan(tensor).any():
    #                 print(f"    ⚠️  NaN detected in {key}")
    #             if torch.isinf(tensor).any():
    #                 print(f"    ⚠️  Inf detected in {key}")
                
    #             input_stack.append(tensor)
        
    #     # Create final tensors
    #     if input_stack:
    #         self.input = torch.cat(input_stack, dim=0).to(self.device)
    #         final_mean = self.input.mean().item()
    #         final_std = self.input.std().item()
    #         print(f"  {'Final Input':>12}: mean={final_mean:>12.8f}, std={final_std:>12.8f}")
    #     else:
    #         self.input = torch.empty(0, dtype=torch.float32).to(self.device)
    #         print(f"  {'Final Input':>12}: EMPTY")
        
    #     # Process labels
    #     LABELS = {"real": 0, "real_resized": 0, "fake": 1, "fake_resized": 1}
    #     label_stack = []
    #     for key in ["real", "real_resized", "fake", "fake_resized"]:
    #         if input[key] is not None:
    #             label_stack += [LABELS[key]] * len(input[key])
        
    #     if label_stack:
    #         self.label = torch.tensor(label_stack, dtype=torch.float32).to(self.device)
    #         label_mean = self.label.mean().item()
    #         print(f"  {'Final Labels':>12}: mean={label_mean:>12.8f}, count={len(label_stack)}")
    #     else:
    #         self.label = torch.empty(0, dtype=torch.float32).to(self.device)
    #         print(f"  {'Final Labels':>12}: EMPTY")
        
    def forward(self):
        if self.contrastive:
            # Get only global features and output
            self.feature, self.output = self.model(self.input, return_feature=True)
        else:
            # Get only output
            self.output = self.model(self.input)
        
        if hasattr(self.output, 'view'):  
            self.output = self.output.view(-1).unsqueeze(1)

    def get_loss(self):
        return self.loss_fn(self.output.squeeze(1), self.label)

    def optimize_parameters(self):
        self.current_step += 1
        self.forward()

        # Calculate classification loss
        cls_loss = self.loss_fn(self.output.squeeze(1), self.label)
        
        # Initialize total loss with classification loss
        total_loss = cls_loss
        
        # Add global contrastive loss if enabled
        if self.contrastive:
            contrastive_loss = self.contrastive_loss_fn(self.feature, self.label)
            total_loss = (1 - self.contrastive_alpha) * total_loss + self.contrastive_alpha * contrastive_loss
    
        # Save the final loss
        self.loss = total_loss
        
        
        # scale = 1e5  # For 5 decimal places
        # self.loss = torch.round(total_loss * scale) / scale
        
        # Apply gradient accumulation
        self.loss = self.loss / self.accumulation_steps
        self.loss.backward()


        if self.current_step % self.accumulation_steps == 0:            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            self.optimizer.zero_grad()  # Reset gradients after update
        
    def train(self):
        self.model.train()
        
    def eval(self):
        self.model.eval()

    # Handle remaining gradients at the end of epoch
    def finalize_epoch(self):
        if self.current_step % self.accumulation_steps != 0:
            # Update with remaining gradients
            self.optimizer.step()
            self.optimizer.zero_grad()