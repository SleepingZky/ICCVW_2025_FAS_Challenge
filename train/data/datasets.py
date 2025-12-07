import cv2
import numpy as np
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import random
from io import BytesIO
from PIL import Image
from PIL import ImageFile
from PIL import ImageDraw
from scipy.ndimage.filters import gaussian_filter
import pickle
import os 
from skimage.io import imread
from copy import deepcopy
import torch
import json
from tqdm import tqdm
from .custom_transforms import *

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "clip":[0.26862954, 0.26130258, 0.27577711]
}


def create_train_transforms(size=224, mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711), is_crop=True, random_mask=False, jpeg_quality=None, opt=None):
    """创建训练数据增强转换管道"""
    
    # 根据是否裁剪选择不同的大小调整方法
    if is_crop:
        resize_func = PadRandomCrop(size)
    else:
        resize_func = transforms.Resize(size)
    
    # 构建转换列表
    transform_list = [
        # 千万不要在这里加随机JPEG压缩，加上这个会导致 real acc 大幅下降
        # RandomJPEGCompression(quality_lower=30, quality_upper=90, p=0.1), 

        # 随机水平翻转
        transforms.RandomHorizontalFlip(),
        # 随机竖直翻转
        transforms.RandomVerticalFlip(),
        
        # 添加高斯噪声
        RandomGaussianNoise(p=0.1),
        
        # 添加椒盐噪声
        RandomPepperNoise(p=0.1),
        
        # 随机模糊组合
        transforms.RandomApply([
            transforms.RandomChoice([
                transforms.GaussianBlur(kernel_size=3),
                transforms.GaussianBlur(kernel_size=5),
                MedianBlur(kernel_size=3),
                MotionBlur(kernel_size=5)
            ])
        ], p=0.2),
        
        # 随机锐化
        RandomSharpen(p=0.1),
        
        # 随机应用颜色变换（亮度、对比度等）
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.15)
        ], p=0.2),
        
        # 随机转为灰度图
        transforms.RandomGrayscale(p=0.2),
        
        # # 随机遮罩部分区域
        RandomErasing(p=0.1, scale=(0.02, 0.1), ratio=(0.3, 3.3)),

        # 调整大小
        resize_func,

        # 转为张量
        transforms.ToTensor(),
        
        # 标准化
        transforms.Normalize(mean=mean, std=std)
    ]
    
    # 添加随机 mask 变换（如果启用）
    if random_mask:
        transform_list.insert(-1, RandomMask(ratio=(0.00, opt.r_pixelmask), patch_size=14, p=opt.p_pixelmask))
    if jpeg_quality:
        transform_list.insert(0, RandomJPEGCompression(quality_lower=jpeg_quality, quality_upper=100, p=0.1))
    # 创建转换组合
    return transforms.Compose(transform_list)


class CustomBatchSampler:
    def __init__(self, opt, vae_model, transform_funcs):
        self.real_list = get_list(opt.real_data_path)
        self.fake_list = self.real_list.copy()
        self.vae_model = vae_model
        self.transform_funcs = transform_funcs

        self.fake_num = len(self.vae_model)

        self.vae_transform_funcs_list = []
        for vae in vae_model:
            vae_transform_funcs = transform_funcs.copy()
            vae_transform_funcs.insert(1, vae)
            self.vae_transform_funcs_list.append(vae_transform_funcs)
        
        self.batch_size =opt.batch_size
        self.gpu_id = opt.gpu_ids[0]

        self.indices = list(range(len(self.real_list)))
        
    def __iter__(self):
        return self
    
    def set_epoch_start(self):
        random.shuffle(self.real_list)
        random.shuffle(self.fake_list)
        self.iteration = 0

    def __next__(self):
        """返回一个批次的处理后数据"""
        batch_images = []
        for idx in range(self.iteration * (self.batch_size // (1 + self.fake_num)), (self.iteration + 1) * (self.batch_size // (1 + self.fake_num))):
            try:
                # 加载图像
                img_path = self.real_list[idx]
                
                # 打开并转换图像
                img = Image.open(img_path).convert("RGB")
                
                # 添加到批次
                batch_images.append(img)
                
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

        # 复制出所需的fake份数
        batch_fake_images_list = []
        for i in range(self.fake_num):
            batch_fake_images_list.append(batch_images.copy())

        # 处理label
        label_list = [0] * len(batch_images)
        for i in range(self.fake_num):
            label_list += [1] * len(batch_images)
        labels_tensor = torch.tensor(label_list)

        # Save random state before transformations
        random_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        for trans in self.transform_funcs:
            torch.set_rng_state(random_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            batch_images = trans(batch_images)

        real_images_tensor = torch.stack(batch_images).cuda(self.gpu_id)

        fake_images_tensor_list = []
        for i in range(self.fake_num):
            vae_transform_funcs = self.vae_transform_funcs_list[i]
            batch_fake_images = batch_fake_images_list[i]

            for trans in vae_transform_funcs:
                torch.set_rng_state(random_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                batch_fake_images = trans(batch_fake_images)
            
            fake_images_tensor_list.append(torch.stack(batch_fake_images))

        fake_images_tensor = torch.cat(fake_images_tensor_list).cuda(self.gpu_id)

        images_tensor = torch.cat((real_images_tensor, fake_images_tensor), dim=0)

        self.iteration += 1

        return images_tensor, labels_tensor

    
    def __len__(self):
        """返回一个epoch中的批次数量"""
        return ( len(self.real_list) // (self.batch_size) ) * (1 + self.fake_num)



rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

class RealFakeDataset(Dataset):
    """
    Dataset class for training with real images and their corresponding fake (reconstructed)
    versions from various VAE models. Supports multiple augmentations including resizing,
    JPEG compression, and blending.
    
    The dataset handles two types of image formats:
    - JPEG: Original real images (typically with JPEG artifacts)
    - PNG: Reconstructed images from VAE models (typically lossless)
    
    Different transformation pipelines are applied to each format to better simulate
    real-world conditions and train robust discriminators.
    
    Args:
        opt: Configuration object containing dataset parameters including:
            - real_data_path: Comma-separated paths to real images
            - vae_rec_data_path: Comma-separated paths to VAE reconstructed images 
                                 (in corresponding order to real_data_path)
            - down_resize_factors: Comma-separated list of downsampling factors
            - upper_resize_factors: Comma-separated list of upsampling factors
            - cropSize: Size for image cropping
            - arch: Architecture type (determines normalization stats)
            - random_mask: Whether to apply random masking
            - jpeg_quality: JPEG quality range for compression
            - r_pixelmix: Comma-separated blend ratios for real/fake mixing
            - jpeg_aligned: Whether to use aligned JPEG compression
            - quality_json: Path to JPEG quality factors mapping
            - p_jpeg_fake: Probability of applying JPEG compression to fake images
            - p_png_real: Probability of selecting PNG mode (vs JPEG)
            - p_pixelmix: Probability of blending real and fake images
    """
    def __init__(self, opt):
        # Validate data label
        self.opt = opt        
        # Initialize data lists
        self.data_list = []  # Combined dataset for better diversity
        
        # Track dataset sources for analysis
        self.dataset_sources = {}
        
        # Load JPEG quality factor mapping for aligned compression
        with open(self.opt.quality_json, "rb") as file:
            self.real_quality_factor_mapping = json.load(file)
        
        # Parse comma-separated paths
        real_data_paths = [path.strip() for path in opt.real_data_path.split(',')]
        vae_rec_data_paths = [path.strip() for path in opt.vae_rec_data_path.split(',')]
        
        # Validate that we have the same number of real and VAE paths
        if len(real_data_paths) != len(vae_rec_data_paths):
            raise ValueError(f"Number of real_data_path ({len(real_data_paths)}) must match "
                           f"number of vae_rec_data_path ({len(vae_rec_data_paths)})")
        
        # Load datasets from all path pairs
        for i, (real_path, vae_path) in enumerate(zip(real_data_paths, vae_rec_data_paths)):
            print(f"Loading dataset {i+1}/{len(real_data_paths)}: {real_path} -> {vae_path}")
            
            # Create source name based on index
            source_name = f"dataset_{i+1}"
            
            # Split VAE path by comma to handle multiple VAE models for the same real data
            vae_models = [model_path.strip() for model_path in vae_path.split(',')]
            
            samples = self._load_dataset(
                real_data_path=real_path,
                vae_models=vae_models,
                source_name=source_name
            )
            self.data_list.extend(samples)
        
        # Print dataset statistics
        for source, count in self.dataset_sources.items():
            print(f"Loaded {count} samples from {source}")
        
        # Shuffle dataset for randomness in sampling
        random.shuffle(self.data_list)
        
        # Set up normalization stats based on architecture
        stat_from = "imagenet" if self.opt.arch.lower().startswith("imagenet") else "clip"
        print(f"Mean and std stats are from: {stat_from}")

        # Configure JPEG quality settings if enabled
        if self.opt.jpeg_quality:
            self.jpeg_quality = self.opt.jpeg_quality
            print(f'Add random JPEG compression into transform: [{self.jpeg_quality}, 100]')
        else:
            self.jpeg_quality = None
   
        # Create separate transformation pipelines for PNG and JPEG images
        # For PNG images: Include JPEG compression to simulate real-world artifacts
        transform_list_png = create_train_transforms(
            size=opt.cropSize, 
            mean=MEAN[stat_from], 
            std=STD[stat_from], 
            random_mask=False, 
            jpeg_quality=self.jpeg_quality,
            opt=self.opt  # Apply JPEG compression to PNG images
        )
        self.transform_png = ComposedTransforms(transform_list_png)
        
        # For JPEG images: Skip additional JPEG compression (already have artifacts)
        transform_list_jpeg = create_train_transforms(
            size=opt.cropSize, 
            mean=MEAN[stat_from], 
            std=STD[stat_from], 
            random_mask=False, 
            jpeg_quality=None,
            opt=self.opt  # No additional JPEG compression for JPEG images
        )
        self.transform_jpeg = ComposedTransforms(transform_list_jpeg)

        # Set patch shuffle parameters (for potential future use)
        self.patch_size = getattr(opt, 'patch_size', 14)

        # Set probabilities for various augmentations
        self.p_jpeg_fake = self.opt.p_jpeg_fake
        self.p_png_real = self.opt.p_png_real
        self.p_freqmix = self.opt.p_freqmix
        
        self.color_space = [f.strip() for f in opt.mix_color_space.split(',')]

    def _load_dataset(self, real_data_path, vae_models, source_name):
        """
        Helper method to load dataset samples from a list of real images and corresponding
        fake images generated by different VAE models.
        
        Args:
            real_data_path: Path to the real images
            vae_models: List of VAE model directories
            source_name: Name of this data source for tracking/analysis
            
        Returns:
            List of samples loaded from this data source
        """
        samples = []
        
        # Initialize counter for this source
        self.dataset_sources[source_name] = 0
        
        # Get real image paths
        real_list = get_list(real_data_path)
        real_list.sort()
        
        # Construct the complete dataset
        for real_path in tqdm(real_list, desc=f"Loading {source_name}..."):
            # Create a data sample with real path and corresponding fake paths
            sample = {
                'real_path': real_path,
                'fake_paths': [],
                'source': source_name,  # Track which dataset this sample came from
                'format': 'jpeg' if real_path.lower().endswith('.jpg') or real_path.lower().endswith('.jpeg') else 'png',
                # JPEG quality mapping
                'jpeg_quality': self.real_quality_factor_mapping.get(
                    os.path.basename(real_path), 100
                )
            }

            # Extract base filename for matching with fake images
            basename_real_path = os.path.basename(real_path)
            basename_real_path_without_suffix = os.path.splitext(basename_real_path)[0] 
            basename_fake_path = basename_real_path_without_suffix + '.png'
            
            # Track missing files
            missing_path = False
            missing_vae_models = []
                        
            # Collect all fake paths corresponding to this real image
            for vae_rec_dir in vae_models:
                vae_rec_path = os.path.join(vae_rec_dir, basename_fake_path)
                if not os.path.exists(vae_rec_path):
                    missing_path = True
                    missing_vae_models.append(vae_rec_dir)
                    # print(f"Missing file: {vae_rec_path}")
                else:
                    sample['fake_paths'].append(vae_rec_path)
            
            # Only add samples that have ALL valid fake paths
            if not missing_path:
                samples.append(sample)
                self.dataset_sources[source_name] += 1
            else:
                continue
                # print(f"Sample {basename_real_path} excluded due to missing: {', '.join(missing_vae_models)}")
                
        return samples

    def __len__(self):
        """Return the total number of samples in the dataset"""
        return len(self.data_list)
    
    
    def select_random_freqmix_patch_size(self):
        """
        Randomly select a patch size for frequency mixing that satisfies:
        1. patch_size % 14 == 0 (divisible by 14)
        2. self.opt.cropSize % patch_size == 0 (patch_size is a factor of cropSize)  
        3. patch_size > self.opt.cropSize * self.opt.freqmix_patch_ratio (above minimum threshold)
        
        Returns:
            Random valid patch size
        """
        crop_size = self.opt.cropSize
        min_patch_size = crop_size * self.opt.freqmix_patch_ratio
        
        valid_patch_sizes = []
        
        # Check multiples of 14 up to crop_size
        patch_size = 14
        while patch_size <= crop_size:
            if crop_size % patch_size == 0 and patch_size > min_patch_size:
                valid_patch_sizes.append(patch_size)
            patch_size += 14
        
        if not valid_patch_sizes:
            # Fallback: return the smallest valid multiple of 14 that's a divisor of crop_size
            patch_size = 14
            while patch_size <= crop_size:
                if crop_size % patch_size == 0:
                    return patch_size
                patch_size += 14
            # If even that fails, return 14 (though this should be very rare)
            return 14
        
        return random.choice(valid_patch_sizes)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset at the specified index.
        
        This method:
        1. Selects between PNG and JPEG modes based on probability and actual file format
        2. Loads a real image and a random corresponding fake image
        3. Applies format-specific transformations (JPEG compression, blending)
        4. Creates resized versions of both images
        5. Applies appropriate transformation pipeline based on the mode
        
        Returns:
            A dictionary containing real and fake images with their resized versions,
            all properly transformed.
        """
        # Available resampling methods for resizing
        resampling_methods = [
            Image.NEAREST,   # Nearest neighbor - fastest, lowest quality
            Image.BOX,       # Box sampling - fast, low quality
            Image.BILINEAR,  # Bilinear - balanced speed/quality
            Image.HAMMING,   # Hamming - improved bilinear
            Image.BICUBIC,   # Bicubic - higher quality
            Image.LANCZOS,   # Lanczos - highest quality, slowest
        ]

        # Create image dictionary to hold all versions
        img_dict = {
            'real': None,
            'fake': None,
            'real_resized': [],
            'fake_resized': [],
        }
        
        # Get sample data
        sample = self.data_list[idx]
        
        # Determine actual file format and use it as a hint for mode selection
        actual_format = sample['format'] 

        real_img_path = sample['real_path']
        
        # Load real image    
        real_img = Image.open(sample['real_path']).convert("RGB")
        img_dict['real'] = real_img
        
        # Randomly select one fake image path with error handling
        if len(sample['fake_paths']) > 0:
            fake_path = random.choice(sample['fake_paths'])
            fake_img = Image.open(fake_path).convert("RGB")

            if  random.random() < self.p_jpeg_fake:
                jpeg_quality_factor = int(sample['jpeg_quality'])
                # For datasets other than primary MSCOCO, use random quality
                if sample.get('source') != 'dataset_1':  # Assuming first dataset is primary MSCOCO
                    jpeg_quality_factor = random.randint(85, 100)
                fake_img = JPEG_Compression(fake_img, jpeg_quality_factor)
            
            # Create a list of blending operations to perform
            blending_operations = []

            # Add pixel blending if it passes the probability check
            if self.opt.p_pixelmix > 0 and random.random() < self.opt.p_pixelmix:
                blending_operations.append('pixel')

            # Add frequency blending if it passes the probability check
            if self.p_freqmix > 0 and random.random() < self.p_freqmix:
                blending_operations.append('frequency')

            # Shuffle the operations to randomize their order
            if blending_operations:
                random.shuffle(blending_operations)
                
                # Apply operations in the shuffled order
                for operation in blending_operations:
                    if operation == 'pixel':
                        # Apply pixel blending
                        resize_method = random.choice(resampling_methods)
                        fake_img_resized = fake_img.resize(real_img.size, resize_method)
                        fake_img = pixel_blend_mix(real_img=real_img, 
                            fake_img=fake_img_resized, 
                            ratios=[0, self.opt.r_pixelmix],
                            color_space=random.choice(self.color_space),
                            mode=self.opt.meth_pixelmix,
                        )

                    elif operation == 'frequency':
                        fake_img = freq_blend_mix(
                            real_img=real_img, 
                            fake_img=fake_img, 
                            ratios=[0, self.opt.r_freqmix], 
                            color_space=random.choice(self.color_space),
                            mode=self.opt.meth_freqmix,  # Randomly select frequency blending mode
                            patch=self.select_random_freqmix_patch_size()
                        )

            # Store the final fake image in the dictionary
            img_dict['fake'] = fake_img
                
        else:
            raise ValueError("No fake images could be loaded")
        
        # Select random resize factors between the range bounds
        down_resize_factor = random.uniform(self.opt.down_resize_factors, 1.0)
        upper_resize_factor = random.uniform(1.0, self.opt.upper_resize_factors)
        # Select two random methods without replacement
        resize_methods = random.sample(resampling_methods, 2)

        # Apply resizing with selected factors and methods
        for resize_factor, resize_method in zip([down_resize_factor, upper_resize_factor], resize_methods):
            # Create resized versions with the SAME resize factor and method for both real and fake
            real_resized = apply_resize(real_img, resize_factor, resize_method)
            fake_resized = apply_resize(fake_img, resize_factor, resize_method)
        
            img_dict['real_resized'].append(real_resized)
            img_dict['fake_resized'].append(fake_resized)
        
        # Apply transforms to all images based on the selected mode
        # JPEG images use the JPEG transform pipeline (no additional compression)
        # PNG images use the PNG transform pipeline (with potential compression)
        transformed_dict = self.transform_jpeg(img_dict) if actual_format == 'jpeg' else self.transform_png(img_dict)
        
        # Add source information to help with analysis
        transformed_dict['source'] = sample['source']

        return transformed_dict
    
def custom_collate_fn(batch):
    # Filter out None values
    batch = [item for item in batch if item is not None]
    
    # If the entire batch is None, return an empty batch
    if len(batch) == 0:
        return {
            "real": torch.tensor([]),
            "real_resized": torch.tensor([]),
            "fake": torch.tensor([]),
            "fake_resized": torch.tensor([])
        }
    
    # Create lists for different image types
    real_images = []
    fake_images = []
    real_resized_images = []
    fake_resized_images = []
    
    # Extract images from batch
    for item in batch:
        # Process 'real' images
        if 'real' in item:
            if isinstance(item['real'], list):
                for real_img in item['real']:
                    if real_img is not None:
                        real_images.append(real_img)
            else:  # Single instance
                if item['real'] is not None:
                    real_images.append(item['real'])
        
        # Process 'fake' images
        if 'fake' in item:
            if isinstance(item['fake'], list):
                for fake_img in item['fake']:
                    if fake_img is not None:
                        fake_images.append(fake_img)
            else:  # Single instance
                if item['fake'] is not None:
                    fake_images.append(item['fake'])

        # Process 'real_resized' images
        if 'real_resized' in item:
            if isinstance(item['real_resized'], list):
                for real_resize_img in item['real_resized']:    
                    if real_resize_img is not None:
                        real_resized_images.append(real_resize_img)
            else:  # Single instance
                if item['real_resized'] is not None:
                    real_resized_images.append(item['real_resized'])
        
        # Process 'fake_resized' images
        if 'fake_resized' in item:
            if isinstance(item['fake_resized'], list):
                for fake_resize_img in item['fake_resized']:
                    if fake_resize_img is not None:
                        fake_resized_images.append(fake_resize_img)
            else:  # Single instance
                if item['fake_resized'] is not None:
                    fake_resized_images.append(item['fake_resized'])
    
    # Only stack if there are images
    real_images_tensor = torch.stack(real_images) if real_images else torch.tensor([])
    real_resized_images_tensor = torch.stack(real_resized_images) if real_resized_images else torch.tensor([])
    fake_images_tensor = torch.stack(fake_images) if fake_images else torch.tensor([])
    fake_resized_images_tensor = torch.stack(fake_resized_images) if fake_resized_images else torch.tensor([])
    
    return {
        "real": real_images_tensor, 
        "real_resized": real_resized_images_tensor, 
        "fake": fake_images_tensor, 
        "fake_resized": fake_resized_images_tensor
    }