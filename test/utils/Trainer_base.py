import torch
import os
import wandb
import torch.distributed as dist
from utils.metric import compute_eer
from pdb import set_trace as st

from .train_class import Trainer
from .misc import *
from .meters import *

class Trainer_base(Trainer):
    def __init__(self, args, start_epoch, train_dataloader, val_dataloader, model, criterion, criterion_2, optimizer, scheduler, saver, device, epoch_size, logger):
        super(Trainer_base, self).__init__(args, start_epoch, train_dataloader, val_dataloader, model, criterion, criterion_2, optimizer, scheduler, saver, device, epoch_size, logger)
        self.access_list=['acces_1','losses_1','losses_2']
        self.metric_list=['Acc_1', 'Loss_1', 'Loss_2']

        self.val_access_list=['acces_1','losses_1','losses_2']
        self.val_metric_list=['Acc_1', 'Loss_1', 'Loss_2']

        self.LABELS = {
            "real": 0,
            "real_resized": 0,
            "fake": 1,
            "fake_resized": 1,
        }
    
    def _model_forward(self, images):
        output_1 = self.model(images)
        prediction_1 = output_1.argmax(dim=1)
        return prediction_1, output_1

    def _forward(self,images,targets):
        self.batch_size = images.shape[0]
        images = images.to(self.device)
        self.targets_1 = targets.to(self.device)
        # model forward
        self.feature, self.logits = self.model.module.forward_1(images,return_feature=True)
        self.loss_1 = self.criterion(self.logits.squeeze(1), self.targets_1.float())
        self.loss_2 = self.criterion_2(self.feature, self.targets_1)

        preds = (torch.sigmoid(self.logits.squeeze(1)) > 0.5).float()
        self.acc_1 = (preds == self.targets_1).float().mean()

    def _train_forward(self,datas):
        input_stack = []
        input_stack.append(datas["real"])
        input_stack.extend(datas["real_resized"])
        input_stack.append(datas["fake"])
        input_stack.extend(datas["fake_resized"])
        images = torch.cat(input_stack, dim=0)
        label_stack = []
        label_stack += [self.LABELS["real"]] * len(datas["real"])
        label_stack += [self.LABELS["real_resized"]] * len(datas["real_resized"])*2
        label_stack += [self.LABELS["fake"]] * len(datas["fake"])
        label_stack += [self.LABELS["fake_resized"]] * len(datas["fake_resized"])*2
        targets = torch.tensor(label_stack).float()

        self._forward(images,targets)
        self.loss = (self.loss_1 + self.loss_2)/2
        # self.loss = self.loss_1
        self.loss.backward()

    def _update_metric(self):
        if self.args.distributed:
            self.acces_1.update(reduce_tensor(self.acc_1.data).item(), self.batch_size)
            self.losses_1.update(reduce_tensor(self.loss_1.data).item(), self.batch_size)
            self.losses_2.update(reduce_tensor(self.loss_2.data).item(), self.batch_size)
        else:
            self.acces_1.update(self.acc_1.item(), self.batch_size)
            self.losses_1.update(self.loss_1.item(), self.batch_size)
            self.losses_2.update(self.loss_2.item(), self.batch_size)

    def _train_metric(self,epoch):
        if self.args.local_rank == 0:
            content={}
            content['train_acc_1'] = self.acces_1.avg
            content['train_loss_1'] = self.losses_1.avg
            content['train_loss_2'] = self.losses_2.avg
            
            wandb.log(content, step=epoch)

    def _init_val_metric(self):
        self.y_preds_val_1 = []
        self.y_trues_val_1 = []
        self.val_path = []

    def _val_forward(self, datas):
        [images, targets, img_path, _] = datas
        # model forward

        self.batch_size = images.shape[0]
        images = images.to(self.device)
        targets = targets.to(self.device)
        # model forward
        outputs = self.model.module.forward_1(images,return_feature=False)

        # Convert to probabilities
        probs = torch.sigmoid(outputs.squeeze())

        # Store results
        self.y_preds_val_1.extend(probs)
        self.y_trues_val_1.extend(targets)
        self.val_path.extend(img_path)
    
    def _val_metric(self,epoch):
        self.y_preds_val_1 = torch.stack(self.y_preds_val_1)

        self.y_trues_val_1 = torch.stack(self.y_trues_val_1)

        if self.args.distributed:
            results_val=[]
            for _ in range(dist.get_world_size()):
                results_val.append(torch.ones_like(self.y_preds_val_1))

            self.gather_y_preds_val_1 = results_val

            dist.all_gather(self.gather_y_preds_val_1, self.y_preds_val_1)
            
            self.gather_y_preds_val_1 = torch.cat(self.gather_y_preds_val_1)
                
            self.gather_y_trues_val_1 = [torch.ones_like(self.y_trues_val_1) for _ in range(dist.get_world_size())]
            dist.all_gather(self.gather_y_trues_val_1, self.y_trues_val_1)

            self.gather_val_path = [None for _ in range(dist.get_world_size())]
            dist.all_gather_object(self.gather_val_path, self.val_path)

            self.gather_val_path = [item for sublist in self.gather_val_path for item in sublist]

            self.gather_y_trues_val_1 = torch.cat(self.gather_y_trues_val_1)
        else:
            self.gather_y_preds_val_1 = y_preds_val_1
            self.gather_y_trues_val_1 = self.y_trues_val_1

            self.gather_val_path = self.val_path
            
        self.gather_y_trues_val_1[0] = self.gather_y_trues_val_1[0]*0
        val_metrics = compute_eer(self.gather_y_trues_val_1.cpu().tolist(), 
                                self.gather_y_preds_val_1.cpu().tolist())
        # 保存测试结果
        if self.args.local_rank == 0:
            results_file =os.path.join(self.args.exam_dir, f'results_iterations_{epoch}.txt')
            with open(results_file,'w') as f:
                for i,j in zip(self.gather_val_path, self.gather_y_preds_val_1):
                    _,folder,file = i.rsplit('/',2)
                    f.write(f'{os.path.join(folder,file)} {j}\n')

            best_AUC, best_epoch = self.saver.save_checkpoint(epoch, metric=val_metrics.AUC)
            
            for k, v in val_metrics.items():
                self.logger.info(f'val_{k}: {100 * v:.4f}')
            
            for i in range(1,2):
                results = eval(f'self.losses_{i}.avg')
                exec(f"self.logger.info(f'val_loss_{i}: {results:.4f}')")
            self.logger.info(f'best_val_AUC: {best_AUC:.4f} (Epoch-{best_epoch})')
            last_lr = [group['lr'] for group in self.scheduler.optimizer.param_groups][0]

            content={}
            content[f'val_AUC'] = eval(f'val_metrics.AUC')
            content[f'val_ACER'] = eval(f'val_metrics.ACER')
            content[f'val_threshold'] = eval(f'val_metrics.threshold')
            for i in range(1,2):
                content[f'train_loss_{i}'] = eval(f'self.losses_{i}.avg')
            content['lr'] = last_lr
            
            wandb.log(content, step=epoch)