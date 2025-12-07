import argparse
import os
import torch


class BaseOptions():
    def __init__(self):
        self.initialized = False

    def initialize(self, parser):
        parser.add_argument('--val_mode', action='store_true')
        
        parser.add_argument('--seed', type=int, default=None, help='random seed')
        
        parser.add_argument('--mode', default='binary')
        parser.add_argument('--arch', type=str, default='res50', help='see my_models/__init__.py')
        parser.add_argument('--fix_backbone', action='store_true')  
        
        parser.add_argument('--lora_rank', type=int, default=8, help='LoRA rank')
        parser.add_argument('--lora_alpha', type=float, default=1.0, help='LoRA scaling factor')
        parser.add_argument('--lora_targets', type=str, default=None, help='LoRA trainable targets')

        # NEW: Feature supervision options
        parser.add_argument('--feature_supervision', action='store_true', help='Enable feature supervision from teacher model')
        parser.add_argument('--teacher_model_path', type=str, default=None, help='Path to pretrained teacher model checkpoint')
        parser.add_argument('--teacher_name', type=str, default='dinov2_vitl14', help='Teacher model name (e.g., dinov2_vitl14, dinov2_vits14)')
        parser.add_argument('--teacher_lora_rank', type=int, default=8, help='Teacher model LoRA rank')
        parser.add_argument('--teacher_lora_alpha', type=float, default=1.0, help='Teacher model LoRA alpha')
        parser.add_argument('--teacher_lora_targets', type=str, default=None, help='Teacher model LoRA targets')
        parser.add_argument('--feature_supervision_weight', type=float, default=1.0, help='Weight for feature supervision loss')

        parser.add_argument('--real_data_path', default=None, help='path(s) for real images, multiple paths separated by ", "')
        parser.add_argument('--vae_rec_data_path', type=str, help='path(s) for VAE reconstructed images, multiple paths separated by ", " corresponding to real_data_path')

        parser.add_argument('--contrastive', action='store_true')

        parser.add_argument('--accumulation_steps', type=int, default=4, help='Number of steps to accumulate gradients before optimizer step')
 
        parser.add_argument('--data_mode',  default='ours', help='wang2020 or ours')
        parser.add_argument('--data_label', default='train', help='label to decide whether train or validation dataset')
        parser.add_argument('--weight_decay', type=float, default=0.0, help='loss weight for l2 reg')
        
        parser.add_argument('--class_bal', action='store_true') # what is this ?
        parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
        parser.add_argument('--loadSize', type=int, default=256, help='scale images to this size')
        parser.add_argument('--cropSize', type=int, default=224, help='then crop to this size')
        parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--num_threads', default=8, type=int, help='# threads for loading data')
        parser.add_argument('--checkpoints_dir', type=str, required=True, help='models are saved here')
        parser.add_argument('--serial_batches', action='store_true', help='if true, takes images in order to make batches, otherwise takes them randomly')
        parser.add_argument('--resize_or_crop', type=str, default='scale_and_crop', help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop|none]')
        parser.add_argument('--no_flip', action='store_true', help='if specified, do not flip the images for data augmentation')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal|xavier|kaiming|orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--suffix', default='', type=str, help='customized suffix: opt.name = opt.name + suffix: e.g., {model}_{netG}_size{loadSize}')
        
        parser.add_argument('--local_models_only', action='store_true', help='只使用本地模型文件，不从网络下载')
        parser.add_argument('--model_dir', type=str, default=None, help='本地模型权重文件的目录路径')
        self.initialized = True
        return parser

    def gather_options(self):
        # initialize parser with basic options
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        # get the basic options
        opt, _ = parser.parse_known_args()
        self.parser = parser

        return parser.parse_args()

    def print_options(self, opt):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = '\t[default: %s]' % str(default)
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to the disk
        expr_dir = os.path.join(opt.name, opt.checkpoints_dir)
        os.makedirs(expr_dir, exist_ok=True)
        file_name = os.path.join(expr_dir, 'opt.txt')
        with open(file_name, 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self, print_options=True):

        opt = self.gather_options()
        opt.isTrain = self.isTrain   # train or test

        # process opt.suffix
        if opt.suffix:
            suffix = ('_' + opt.suffix.format(**vars(opt))) if opt.suffix != '' else ''
            opt.name = opt.name + suffix

        if print_options:
            self.print_options(opt)

        # set gpu ids
        str_ids = opt.gpu_ids.split(',')
        opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                opt.gpu_ids.append(id)
        if len(opt.gpu_ids) > 0:
            torch.cuda.set_device(opt.gpu_ids[0])

        self.opt = opt
        return self.opt


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--data_aug', action='store_true', help='if specified, perform additional data augmentation (photometric, blurring, jpegging)')
        parser.add_argument('--optim', type=str, default='adam', help='optim to use [sgd, adam]')
        parser.add_argument('--new_optim', action='store_true', help='new optimizer instead of loading the optim state')
        parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--last_epoch', type=int, default=-1, help='starting epoch count for scheduler intialization')
        parser.add_argument('--train_split', type=str, default='train', help='train, val, test, etc')
        parser.add_argument('--val_split', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--niter', type=int, default=1, help='total epoches')
        parser.add_argument('--beta1', type=float, default=0.9, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        
        parser.add_argument('--down_resize_factors', type=float, default=0.2)
        parser.add_argument('--upper_resize_factors', type=float, default=3.0)
        
        parser.add_argument('--token_contrastive', action='store_true')
        
        parser.add_argument('--quality_json', default='./MSCOCO_train2017.json')
        
        parser.add_argument('--jpeg_quality', type=int, default=100)

        parser.add_argument('--p_jpeg_fake', type=float, default=0.5)
        parser.add_argument('--p_png_real', type=float, default=0.0)
        
        parser.add_argument('--p_pixelmix', type=float, default=0.2)
        parser.add_argument('--r_pixelmix', type=float, default=0.0)
        parser.add_argument('--meth_pixelmix', type=str, choices=['uniform', 'variable'])
        
        parser.add_argument('--p_freqmix', type=float, default=0.2)
        parser.add_argument('--r_freqmix', type=float, default=0.1)
        parser.add_argument('--meth_freqmix', type=str, choices=['uniform', 'variable'])
        parser.add_argument('--mix_color_space', type=str, default='RGB, HSV')
        parser.add_argument('--freqmix_patch_ratio', type=float, default=1.0, )
        parser.add_argument('--resume', type=str, default=None, 
                       help='Path to checkpoint to resume from')
        parser.add_argument('--resume_epoch_only', action='store_true',
                       help='Only resume model weights, not optimizer state')
        self.isTrain = True
        return parser