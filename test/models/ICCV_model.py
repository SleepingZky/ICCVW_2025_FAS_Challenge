from .en_clip import English_Clip, English_Clip_Vision, CLIPModelWithLoRA
import torch
import torch.nn as nn
from pdb import set_trace as st
from .en_clip.english_clip import tokenize


class test_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(3, 1024)
        self.layer2 = nn.Linear(1024, 1)
        self.drop_out = nn.Dropout()
        
    def forward(self, x, return_feature=False):
        x = x.mean(dim=-1).mean(dim=-1)
        x_f = self.layer1(x)
        x = self.layer2(self.drop_out(x_f))
        
        if return_feature:
            return x_f, x
        
        return x


class ICCVModel(nn.Module):
    def __init__(self, args):
        super(ICCVModel, self).__init__()
        self.args = args

        self.module_1 = CLIPModelWithLoRA(name=args.MODEL.BACKBONE.NAME)

    def forward_1(self, image, return_feature=False):
        if return_feature:
            f_1, logits_1 = self.module_1(image, return_feature=return_feature)
            return f_1, logits_1
        else:
            logits_1 = self.module_1(image, return_feature=return_feature)
            return logits_1
    
    def get_text_features(self, text_list):
        prompts = torch.cat([tokenize(p) for p in text_list[0]])
        device = next(self.parameters()).device
        prompts = prompts.to(device)
        text_features = self.module_1.model.encode_text(prompts).cpu()
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        self.t1 = text_features

    def get_trainable_params(self):
        return self.module_1.get_trainable_params()

    def forward(self, image1, image2):
        # texture results
        outputs, _ = self.module_1(image1, return_feature=False)
        probs = torch.sigmoid(outputs.squeeze()).cpu()
        original_probs = probs.clone()

        # semantics results
        _, x = self.module_1(image2, return_feature=False)
        features = x / x.norm(dim=-1, keepdim=True)
        features = features.cpu()

        # calculate sims
        logit_scale = self.module_1.model.logit_scale.exp().cpu()
        self.s1 = torch.softmax(logit_scale * features @ self.t1.t(),dim=1)
        
        for i in range(len(probs)):
            if self.s1[i,1] >= 0.45:
                probs[i] = self.s1[i,1]
        
        return probs

    def load_model(self, model_path):   
        state_dict = torch.load(model_path)['model']
        self.module_1.load_state_dict(state_dict)
        print("load pth model successful!")