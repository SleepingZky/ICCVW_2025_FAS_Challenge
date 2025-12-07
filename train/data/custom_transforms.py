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
from scipy.fftpack import dct, idct

ImageFile.LOAD_TRUNCATED_IMAGES = True


MEAN = {
    "imagenet":[0.485, 0.456, 0.406],
    "clip":[0.48145466, 0.4578275, 0.40821073]
}

STD = {
    "imagenet":[0.229, 0.224, 0.225],
    "clip":[0.26862954, 0.26130258, 0.27577711]
}



def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)


rz_dict = {'bilinear': Image.BILINEAR,
           'bicubic': Image.BICUBIC,
           'lanczos': Image.LANCZOS,
           'nearest': Image.NEAREST}

# def custom_resize(img, opt):
#     interp = sample_discrete(opt.rz_interp)
#     return F.resize(img, opt.loadSize, interpolation=rz_dict[interp])



def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg", 'webp']):
    out = [] 
    for r, d, f in os.walk(rootdir, followlinks=True):
        for file in f:
            if (file.split('.')[1].lower() in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list
# ---------- Offline utils ----------

class PadRandomCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):

        w, h = img.size  # 假设输入为 [C, H, W]

        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        # 如果需要填充
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            img = F.pad(img, padding, fill=0)  # 填充0或其他值，如255

        # 对单个图像应用随机裁剪
        cropped = transforms.RandomCrop(self.size)(img)
            
        return cropped

class PadCenterCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):

        w, h = img.size  # 假设输入为 [C, H, W]

        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        # 如果需要填充
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            img = F.pad(img, padding, fill=0)  # 填充0或其他值，如255

        # 对单个图像应用随机裁剪
        cropped = transforms.CenterCrop(self.size)(img)
            
        return cropped
    
class ComposedTransforms:
    """一个图像转换组合，可以一致地应用于多个图像"""
    def __init__(self, transforms_list):
        """
        初始化转换组合
        
        Args:
            transforms_list: 一个torchvision.transforms.Compose对象
        """
        self.transforms = transforms_list
        
    def __call__(self, images_dict):
        """
        对字典中的所有图像应用相同的转换
        
        Args:
            images_dict: 包含不同类型图像的字典
                        (例如, 'real', 'fake', 'real_resized', 'fake_resized')
                        
        Returns:
            包含转换后图像的字典
        """
        # 保存随机状态以实现一致的转换
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        
        result = {}
        
        # 对字典中的每个图像应用转换
        for key, val in images_dict.items():
            if val is None:
                result[key] = None
                continue
                
            if isinstance(val, list):
                # 处理图像列表
                transformed_imgs = []
                for i, single_img in enumerate(val):                   
                    # 为每个图像重置随机状态
                    torch.set_rng_state(torch_state)
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
                    
                    # 应用所有转换
                    transformed = self.transforms(single_img)
                    transformed_imgs.append(transformed)
                
                result[key] = transformed_imgs

            elif isinstance(val, Image.Image):
                # 处理单个PIL图像
                # 重置随机状态
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                
                # 应用所有转换
                transformed = self.transforms(val)
                result[key] = transformed
            else:
                result[key] = val
                
        return result


def shuffle_image_patches(image, patch_size=14, shuffle_order=None):
    """Shuffle patches of a tensor image.
    
    Args:
        image: Tensor image in [C, H, W] format
        patch_size: Size of patches to shuffle
        shuffle_order: Optional pre-defined shuffle order to use for consistency
        
    Returns:
        Shuffled tensor image
    """
    channels, height, width = image.shape
    num_patches_per_row = height // patch_size
    num_patches_per_col = width // patch_size
    total_patches = num_patches_per_row * num_patches_per_col
    
    # Store patches
    image_patches = []
    
    # Divide image into patches
    for i in range(num_patches_per_row):
        for j in range(num_patches_per_col):
            start_row = i * patch_size
            start_col = j * patch_size
            
            image_patch = image[:, start_row:start_row + patch_size, 
                             start_col:start_col + patch_size]
            image_patches.append(image_patch)
    
    # Use provided shuffle order or generate a new one
    if shuffle_order is None or len(shuffle_order) != total_patches:
        shuffle_order = np.arange(total_patches)
        np.random.shuffle(shuffle_order)
    
    # Rebuild shuffled image
    shuffled_image = torch.zeros_like(image)
    
    for idx, shuffle_idx in enumerate(shuffle_order):
        i = idx // num_patches_per_col
        j = idx % num_patches_per_col
        
        start_row = i * patch_size
        start_col = j * patch_size
        
        shuffled_image[:, start_row:start_row + patch_size, 
                     start_col:start_col + patch_size] = image_patches[shuffle_idx]
    
    return shuffled_image, shuffle_order


# Function to apply sequential resizing
def apply_resize(image, resize_factor, resize_method=None):
    """Apply a single resize operation to an image with specified factor and method.
    
    Args:
        image: PIL Image to resize
        resize_factor: Float scaling factor
        resize_method: PIL resampling method (defaults to random selection if None)
    
    Returns:
        Resized PIL Image
    """
    if resize_method is None:
        resampling_methods = [
            Image.NEAREST,
            Image.BOX,
            Image.BILINEAR,
            Image.HAMMING,
            Image.BICUBIC,
            Image.LANCZOS,
        ]
        resize_method = random.choice(resampling_methods)
    
    # Get current dimensions and calculate new dimensions
    w, h = image.size
    new_w, new_h = int(w * resize_factor), int(h * resize_factor)
    
    # Apply resize
    resized = image.resize((new_w, new_h), resize_method)
    
    return resized



class MedianBlur:
    """Apply median blur to image.
    
    Args:
        kernel_size (int): Size of the median filter. Must be odd and positive.
    """
    
    def __init__(self, kernel_size=3):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1  # Ensure kernel size is odd
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred.
            
        Returns:
            PIL Image: Blurred image.
        """
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Apply median blur
        blurred = cv2.medianBlur(img_np, self.kernel_size)
        
        # Convert back to PIL Image
        return Image.fromarray(blurred)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(kernel_size={self.kernel_size})'


class RandomErasing:
    """Randomly erase rectangular regions from an image.
    
    Args:
        p (float): Probability of applying the transform. Default: 0.5.
        scale (tuple): Range of area of the erased region relative to the image area.
                      Default: (0.02, 0.33).
        ratio (tuple): Range of aspect ratio of the erased region. Default: (0.3, 3.3).
        value (int, tuple): Value used to fill the erased region. Default: 0.
    """
    
    def __init__(self, p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0):
        self.p = p
        self.scale = scale
        self.ratio = ratio
        self.value = value
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be erased.
            
        Returns:
            PIL Image: Erased image.
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_np = np.array(img)
        height, width = img_np.shape[:2]
        
        # Calculate area of image
        img_area = height * width
        
        # Get random area to erase
        for _ in range(10):  # Try 10 times to find valid parameters
            erase_area_ratio = random.uniform(self.scale[0], self.scale[1])
            aspect_ratio = random.uniform(self.ratio[0], self.ratio[1])
            
            # Calculate target area
            erase_area = int(img_area * erase_area_ratio)
            
            # Calculate target width and height
            h = int(np.sqrt(erase_area / aspect_ratio))
            w = int(np.sqrt(erase_area * aspect_ratio))
            
            # Ensure width and height are within image bounds
            if w < width and h < height:
                # Random position
                x = random.randint(0, width - w)
                y = random.randint(0, height - h)
                
                # Erase the region
                if isinstance(self.value, (int, float)):
                    # Single value for all channels
                    if len(img_np.shape) == 3:  # Color image
                        img_np[y:y+h, x:x+w, :] = self.value
                    else:  # Grayscale image
                        img_np[y:y+h, x:x+w] = self.value
                else:
                    # Different values for each channel
                    if len(img_np.shape) == 3:  # Color image
                        for i, val in enumerate(self.value):
                            if i < img_np.shape[2]:
                                img_np[y:y+h, x:x+w, i] = val
                    else:  # Grayscale image
                        img_np[y:y+h, x:x+w] = self.value[0]
                
                break
        
        # Convert back to PIL Image
        return Image.fromarray(img_np)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, scale={self.scale}, ratio={self.ratio}, value={self.value})'
    
    
class RandomSharpen:
    """Apply random sharpening to image.
    
    Args:
        p (float): Probability of applying the transform. Default: 0.1.
        factor (float or tuple): Sharpening factor. If tuple (min, max), the factor will 
                          be randomly chosen between min and max. Default: (1.0, 3.0).
    """
    
    def __init__(self, p=0.1, factor=(1.0, 3.0)):
        self.p = p
        self.factor = factor
        if isinstance(factor, (list, tuple)):
            assert len(factor) == 2, "factor should be a tuple of (min, max)"
            self.min_factor, self.max_factor = factor
        else:
            self.min_factor = self.max_factor = factor
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be sharpened.
            
        Returns:
            PIL Image: Sharpened image.
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_np = np.array(img).astype(np.float32)
        
        # Create blurred version for unsharp mask
        blurred = cv2.GaussianBlur(img_np, (0, 0), 3.0)
        
        # Determine sharpening factor for this application
        factor = random.uniform(self.min_factor, self.max_factor)
        
        # Apply unsharp mask formula: original + factor * (original - blurred)
        sharpened = img_np + factor * (img_np - blurred)
        
        # Clip values to valid range
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        return Image.fromarray(sharpened)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, factor=({self.min_factor}, {self.max_factor}))'




class RandomPepperNoise:
    """Apply pepper noise to an image.
    
    Args:
        p (float): Probability of applying the transform. Default: 0.1.
        noise_ratio (float or tuple): Ratio of pixels to be replaced with noise. 
                             If tuple (min, max), the ratio will be randomly chosen 
                             between min and max. Default: (0.01, 0.05).
    """
    
    def __init__(self, p=0.1, noise_ratio=(0.01, 0.05)):
        self.p = p
        self.noise_ratio = noise_ratio
        if isinstance(noise_ratio, (list, tuple)):
            assert len(noise_ratio) == 2, "noise_ratio should be a tuple of (min, max)"
            self.min_ratio, self.max_ratio = noise_ratio
        else:
            self.min_ratio = self.max_ratio = noise_ratio
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be applied with pepper noise.
            
        Returns:
            PIL Image: Image with pepper noise.
        """
        if random.random() > self.p:
            return img
        
        # Convert PIL to numpy array
        img_np = np.array(img)
        height, width = img_np.shape[:2]
        
        # Determine noise ratio for this particular application
        noise_ratio = random.uniform(self.min_ratio, self.max_ratio)
        
        # Calculate number of pixels to replace
        n_pixels = int(height * width * noise_ratio)
        
        # Generate random coordinates
        y_coords = np.random.randint(0, height, n_pixels)
        x_coords = np.random.randint(0, width, n_pixels)
        
        # Apply pepper noise (black pixels)
        if len(img_np.shape) == 3:  # Color image
            img_np[y_coords, x_coords, :] = 0
        else:  # Grayscale image
            img_np[y_coords, x_coords] = 0
        
        # Convert back to PIL Image
        return Image.fromarray(img_np)
    
    def __repr__(self):
        return self.__class__.__name__ + f'(p={self.p}, noise_ratio=({self.min_ratio}, {self.max_ratio}))'
    

def pixel_mix(real_img, fake_img, ratio=(0.0, 0.5), patch_size=14):
    """
    Mix two images in the spatial domain by taking patches from each image.
    Optimized version with vectorized operations for better performance.
    
    Parameters:
    -----------
    real_img : PIL.Image or str
        The real image or path to the image
    fake_img : PIL.Image or str
        The fake image or path to the image
    ratio : tuple of (float, float)
        Range (lower, upper) for the ratio of patches to take from the real image
        A value will be randomly sampled between these bounds
    patch_size : int
        Size of each patch (patch_size x patch_size)
    
    Returns:
    --------
    mixed_img : PIL.Image
        The mixed image - predominantly fake with some real patches
    """
    # Load images if paths are provided
    if isinstance(real_img, str):
        real_img = Image.open(real_img)
    if isinstance(fake_img, str):
        fake_img = Image.open(fake_img)
    
    # Ensure both images have the same size
    if real_img.size != fake_img.size:
        fake_img = fake_img.resize(real_img.size, Image.LANCZOS)
    
    # Convert to RGB if needed
    if real_img.mode != 'RGB':
        real_img = real_img.convert('RGB')
    if fake_img.mode != 'RGB':
        fake_img = fake_img.convert('RGB')
    
    # Convert to numpy arrays
    real_np = np.array(real_img)
    fake_np = np.array(fake_img)
    
    # Get image dimensions
    h, w, c = real_np.shape
    
    # Calculate how many patches fit in each dimension
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    total_patches = n_patches_h * n_patches_w
    
    # Randomly sample a ratio value between the lower and upper bounds
    ratio_lower, ratio_upper = ratio
    sampled_ratio = np.random.uniform(ratio_lower, ratio_upper)
    
    # Calculate the number of patches to take from the real image
    n_real_patches = int(total_patches * sampled_ratio)
    
    # Create the mixed image - starting with the fake image as base
    mixed_np = fake_np.copy()  # More efficient than np.copy(fake_np)

    # OPTIMIZATION: Single pass patch replacement instead of nested loops
    if n_real_patches > 0:
        # Randomly select patch indices
        patch_indices = np.random.choice(total_patches, n_real_patches, replace=False)
        
        # Convert flat indices to 2D coordinates
        patch_rows = patch_indices // n_patches_w
        patch_cols = patch_indices % n_patches_w
        
        # OPTIMIZATION: Vectorized patch replacement
        for idx in range(n_real_patches):
            r, c = patch_rows[idx], patch_cols[idx]
            
            # Calculate patch boundaries
            top = r * patch_size
            left = c * patch_size
            bottom = min(top + patch_size, h)  # Handle edge cases
            right = min(left + patch_size, w)  # Handle edge cases
            
            # Replace the patch
            mixed_np[top:bottom, left:right] = real_np[top:bottom, left:right]
    
    # Convert back to image
    mixed_img = Image.fromarray(mixed_np)
    
    return mixed_img


class RandomMask(object):
    def __init__(self, ratio=0.5, patch_size=14, p=0.5):
        """
        Args:
            ratio (float or tuple of float): If float, the ratio of the image to be masked.
                                             If tuple of float, random sample ratio between the two values.
            patch_size (int): the size of the mask (d*d).
        """
        if isinstance(ratio, float):
            self.fixed_ratio = True
            self.ratio = (ratio, ratio)
        elif isinstance(ratio, tuple) and len(ratio) == 2 and all(isinstance(r, float) for r in ratio):
            self.fixed_ratio = False
            self.ratio = ratio
        else:
            raise ValueError("Ratio must be a float or a tuple of two floats.")

        self.patch_size = patch_size
        self.p = p

    def __call__(self, tensor):

        if random.random() > self.p: return tensor

        _, h, w = tensor.shape
        mask = torch.ones((h, w), dtype=torch.float32)

        if self.fixed_ratio:
            ratio = self.ratio[0]
        else:
            ratio = random.uniform(self.ratio[0], self.ratio[1])

        # Calculate the number of masks needed
        num_masks = int((h * w * ratio) / (self.patch_size ** 2))

        # Generate non-overlapping random positions
        selected_positions = set()
        while len(selected_positions) < num_masks:
            top = random.randint(0, (h // self.patch_size) - 1) * self.patch_size
            left = random.randint(0, (w // self.patch_size) - 1) * self.patch_size
            selected_positions.add((top, left))

        for (top, left) in selected_positions:
            mask[top:top+self.patch_size, left:left+self.patch_size] = 0

        return tensor * mask.expand_as(tensor)
    
     
class MotionBlur:
    """Apply motion blur to image by applying a directional filter.
    
    Args:
        kernel_size (int): Size of the motion blur kernel. Must be odd.
        angle (float, optional): Angle of motion blur in degrees. If None, 
                                a random angle will be chosen.
    """
    
    def __init__(self, kernel_size=5, angle=None):
        self.kernel_size = kernel_size
        if kernel_size % 2 == 0:
            self.kernel_size = kernel_size + 1  # Ensure kernel size is odd
        self.angle = angle
    
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be blurred.
            
        Returns:
            PIL Image: Motion blurred image.
        """
        # Convert PIL to numpy array
        img_np = np.array(img)
        
        # Choose a random angle if none is specified
        angle = self.angle
        if angle is None:
            angle = np.random.uniform(0, 180)
        
        # Create motion blur kernel
        kernel = np.zeros((self.kernel_size, self.kernel_size))
        center = self.kernel_size // 2
        
        # Convert angle to radians
        angle_rad = np.deg2rad(angle)
        
        # Calculate x, y coordinates for the line
        for i in range(self.kernel_size):
            offset = i - center
            x = int(center + np.round(offset * np.cos(angle_rad)))
            y = int(center + np.round(offset * np.sin(angle_rad)))
            
            if 0 <= x < self.kernel_size and 0 <= y < self.kernel_size:
                kernel[y, x] = 1
        
        # Normalize the kernel
        kernel = kernel / np.sum(kernel)
        
        # Apply the kernel to each channel
        if len(img_np.shape) == 3:  # Color image
            blurred = np.zeros_like(img_np)
            for c in range(img_np.shape[2]):
                blurred[:, :, c] = cv2.filter2D(img_np[:, :, c], -1, kernel)
        else:  # Grayscale image
            blurred = cv2.filter2D(img_np, -1, kernel)
            
        # Convert back to PIL Image
        return Image.fromarray(blurred)
    
class RandomPure:
    """随机将图像的一部分转换为纯色矩形"""
    def __init__(self, p=0.01, min_size=0.1, max_size=0.5):
        """
        初始化
        
        参数:
            p (float): 应用转换的概率
            min_size (float): 纯色区域最小尺寸比例(相对于图像尺寸)
            max_size (float): 纯色区域最大尺寸比例(相对于图像尺寸)
        """
        self.p = p
        self.min_size = min_size
        self.max_size = max_size
    
    def _generate_random_color(self):
        """生成随机RGB颜色"""
        return (
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        )
    
    def _get_random_rectangle_params(self, img_width, img_height):
        """获取随机矩形的参数"""
        # 确定矩形尺寸
        width_ratio = random.uniform(self.min_size, self.max_size)
        height_ratio = random.uniform(self.min_size, self.max_size)
        
        rect_width = int(img_width * width_ratio)
        rect_height = int(img_height * height_ratio)
        
        # 随机位置(确保矩形完全在图像内)
        x1 = random.randint(0, img_width - rect_width)
        y1 = random.randint(0, img_height - rect_height)
        x2 = x1 + rect_width
        y2 = y1 + rect_height
        
        return (x1, y1, x2, y2)
    
    def __call__(self, img):
        """应用变换"""
        if random.random() < self.p:
            # 获取图像尺寸
            width, height = img.size
            
            # 生成随机颜色
            color = self._generate_random_color()
            
            # 获取随机矩形参数
            bbox = self._get_random_rectangle_params(width, height)
            
            # 创建一个与原图像相同的副本
            result = img.copy()
            draw = ImageDraw.Draw(result)
            
            # 绘制随机颜色的矩形
            draw.rectangle(bbox, fill=color)
            
            return result
        else:
            return img

class RandomGaussianNoise:
    """为PIL图像添加高斯噪声的转换"""
    def __init__(self, mean=0, std=45, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            # 将PIL图像转换为numpy数组
            img_array = np.array(img).astype(np.float32)
            
            # 生成噪声
            noise = np.random.normal(self.mean, self.std, img_array.shape)
            
            # 添加噪声
            noisy_img = img_array + noise
            
            # 裁剪到有效的像素值范围
            noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
            
            # 转回PIL图像
            return Image.fromarray(noisy_img)
        return img

def JPEG_Compression(img, quality_factor):
    # Save the image to a BytesIO object in JPEG format
    if quality_factor==100:
        return img
    out = BytesIO()
    img.save(out, format='jpeg', quality=quality_factor)
    out.seek(0)
    img = Image.open(out)
    return img

class RandomJPEGCompression:
    def __init__(self, quality_lower=30, quality_upper=95, p=0.3):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p
        
    def __call__(self, img):
        if random.random() < self.p:
            quality = random.randint(self.quality_lower, self.quality_upper)
            img = JPEG_Compression(img, quality_factor=quality)
            return img
        return img
    
    def __repr__(self):
        return self.__class__.__name__ + f'(quality_lower={self.quality_lower}, quality_upper={self.quality_upper}, p={self.p})'
    

class RandomResizedCropWithVariableSize(transforms.RandomResizedCrop):
    def __init__(self, min_size, max_size, scale=(0.08, 1.0), ratio=(1.0, 1.0), interpolation=transforms.InterpolationMode.BILINEAR):
        self.min_size = min_size
        self.max_size = max_size
        super().__init__(size=min_size, scale=scale, ratio=ratio, interpolation=interpolation)
    
    def get_random_size(self):
        """Return a random size between min_size and max_size."""
        size = random.randint(self.min_size, self.max_size)
        return size

    def __call__(self, img):
        size = img.size 
        size = tuple(int(element * 0.54) for element in size)
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        ret =  F.resized_crop(img, i, j, h, w, size, self.interpolation, antialias=self.antialias)
        return ret



def data_augment(img, opt):
    img = np.array(img)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
        img = np.repeat(img, 3, axis=2)

    if random.random() < opt.blur_prob:
        sig = sample_continuous(opt.blur_sig)
        gaussian_blur(img, sig)

    if random.random() < opt.jpg_prob:
        method = sample_discrete(opt.jpg_method)
        qual = sample_discrete(opt.jpg_qual)
        img = jpeg_from_key(img, qual, method)

    return Image.fromarray(img)


def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random.random() * rg + s[0]
    raise ValueError("Length of iterable s should be 1 or 2.")


def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return random.choice(s)


def gaussian_blur(img, sigma):
    gaussian_filter(img[:,:,0], output=img[:,:,0], sigma=sigma)
    gaussian_filter(img[:,:,1], output=img[:,:,1], sigma=sigma)
    gaussian_filter(img[:,:,2], output=img[:,:,2], sigma=sigma)


def cv2_jpg(img, compress_val):
    img_cv2 = img[:,:,::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:,:,::-1]


def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    # load from memory before ByteIO closes
    img = np.array(img)
    out.close()
    return img


jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}
def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)

def recursively_read(rootdir, must_contain, exts=["png", "jpg", "JPEG", "jpeg"]):
    out = [] 
    for r, d, f in os.walk(rootdir, followlinks=True):
        for file in f:
            if (file.split('.')[1].lower() in exts)  and  (must_contain in os.path.join(r, file)):
                out.append(os.path.join(r, file))
    return out


def get_list(path, must_contain=''):
    if ".pickle" in path:
        with open(path, 'rb') as f:
            image_list = pickle.load(f)
        image_list = [ item for item in image_list if must_contain in item   ]
    else:
        image_list = recursively_read(path, must_contain)
    return image_list
# ---------- Offline utils ----------

class PadRandomCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):

        w, h = img.size  # 假设输入为 [C, H, W]

        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        # 如果需要填充
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            img = F.pad(img, padding, fill=0)  # 填充0或其他值，如255

        # 对单个图像应用随机裁剪
        cropped = transforms.RandomCrop(self.size)(img)
            
        return cropped

class PadCenterCrop:
    def __init__(self, size):
        self.size = size
        
    def __call__(self, img):

        w, h = img.size  # 假设输入为 [C, H, W]

        pad_h = max(0, self.size - h)
        pad_w = max(0, self.size - w)

        # 如果需要填充
        if pad_h > 0 or pad_w > 0:
            padding = (
                pad_w // 2,          # left
                pad_h // 2,          # top
                pad_w - pad_w // 2,  # right
                pad_h - pad_h // 2   # bottom
            )
            img = F.pad(img, padding, fill=0)  # 填充0或其他值，如255

        # 对单个图像应用随机裁剪
        cropped = transforms.CenterCrop(self.size)(img)
            
        return cropped
    
class ComposedTransforms:
    """一个图像转换组合，可以一致地应用于多个图像"""
    def __init__(self, transforms_list):
        """
        初始化转换组合
        
        Args:
            transforms_list: 一个torchvision.transforms.Compose对象
        """
        self.transforms = transforms_list
        
    def __call__(self, images_dict):
        """
        对字典中的所有图像应用相同的转换
        
        Args:
            images_dict: 包含不同类型图像的字典
                        (例如, 'real', 'fake', 'real_resized', 'fake_resized')
                        
        Returns:
            包含转换后图像的字典
        """
        # 保存随机状态以实现一致的转换
        torch_state = torch.get_rng_state()
        numpy_state = np.random.get_state()
        python_state = random.getstate()
        
        result = {}
        
        # 对字典中的每个图像应用转换
        for key, val in images_dict.items():
            if val is None:
                result[key] = None
                continue
                
            if isinstance(val, list):
                # 处理图像列表
                transformed_imgs = []
                for i, single_img in enumerate(val):                   
                    # 为每个图像重置随机状态
                    torch.set_rng_state(torch_state)
                    np.random.set_state(numpy_state)
                    random.setstate(python_state)
                    
                    # 应用所有转换
                    transformed = self.transforms(single_img)
                    transformed_imgs.append(transformed)
                
                result[key] = transformed_imgs

            elif isinstance(val, Image.Image):
                # 处理单个PIL图像
                # 重置随机状态
                torch.set_rng_state(torch_state)
                np.random.set_state(numpy_state)
                random.setstate(python_state)
                
                # 应用所有转换
                transformed = self.transforms(val)
                result[key] = transformed
            else:
                result[key] = val
                
        return result


def freq_blend_mix(real_img, fake_img, ratios=[0.0, 1.0], color_space='RGB', 
                   mode='uniform', patch=8, random_seed=None):
    """
    Mix two images in the frequency domain using DCT by blending their frequency representations
    based on ratios from the specified range. Simulates JPEG compression by processing patches.
    
    Parameters:
    -----------
    real_img : PIL.Image or str
        The real image or path to the image
    fake_img : PIL.Image or str
        The fake image or path to the image
    ratios : list, optional
        Range of ratios [min, max] to randomly select for blending, values between 0.0 and 1.0
    color_space : str, optional
        Color space to use for processing. Options:
        - 'RGB': Red, Green, Blue (3 channels)
        - 'YCbCr': Luminance, Chroma blue, Chroma red (3 channels)
        - 'LAB': Lightness, A*, B* (3 channels) - perceptually uniform
        - 'HSV': Hue, Saturation, Value (3 channels)
        - 'HSL': Hue, Saturation, Lightness (3 channels)
    mode : str, optional
        'uniform' - Use a single consistent ratio for all frequency coefficients 
        'variable' - Use different random ratios for each frequency coefficient
    patch : int, optional
        Size of patches for DCT processing (default 8 for JPEG simulation)
        Set to -1 to process the entire image without patching
    random_seed : int, optional
        Random seed for reproducible results
        
    Returns:
    --------
    mixed_img : PIL.Image
        The mixed image with blended frequency representations (always returned as RGB)
        
    Raises:
    -------
    ValueError: If parameters are invalid
    """
    import numpy as np
    from PIL import Image
    import random
    from scipy.fftpack import dct, idct
    
    # Supported color spaces (all 3-channel)
    SUPPORTED_COLOR_SPACES = ['RGB', 'YCBCR', 'LAB', 'HSV', 'HSL']
    
    # Input validation
    if not (0.0 <= ratios[0] <= 1.0 and 0.0 <= ratios[1] <= 1.0):
        raise ValueError("Ratios must be between 0.0 and 1.0")
    if ratios[0] > ratios[1]:
        raise ValueError("ratios[0] must be <= ratios[1]")
    
    img_type_upper = color_space.upper()
    if img_type_upper not in SUPPORTED_COLOR_SPACES:
        supported = ', '.join(SUPPORTED_COLOR_SPACES)
        raise ValueError(f"img_type must be one of: {supported}")
    
    if mode not in ['uniform', 'variable']:
        raise ValueError("mode must be 'uniform' or 'variable'")
    if patch != -1 and patch <= 0:
        raise ValueError("patch must be positive or -1")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Load images if paths are provided
    if isinstance(real_img, str):
        real_img = Image.open(real_img)
    if isinstance(fake_img, str):
        fake_img = Image.open(fake_img)
    
    # Ensure both images have the same size
    if real_img.size != fake_img.size:
        fake_img = fake_img.resize(real_img.size, Image.LANCZOS)
    
    # Convert to RGB first (as base format)
    if real_img.mode != 'RGB':
        real_img = real_img.convert('RGB')
    if fake_img.mode != 'RGB':
        fake_img = fake_img.convert('RGB')
    
    # Helper functions for HSL conversion (PIL doesn't support HSL directly)
    def rgb_to_hsl(rgb_array):
        """Convert RGB to HSL manually since PIL doesn't support HSL directly"""
        rgb_norm = rgb_array / 255.0
        r, g, b = rgb_norm[:,:,0], rgb_norm[:,:,1], rgb_norm[:,:,2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2.0
        
        # Saturation
        s = np.zeros_like(l)
        mask = diff != 0
        s[mask & (l <= 0.5)] = diff[mask & (l <= 0.5)] / (max_val + min_val)[mask & (l <= 0.5)]
        s[mask & (l > 0.5)] = diff[mask & (l > 0.5)] / (2.0 - max_val - min_val)[mask & (l > 0.5)]
        
        # Hue
        h = np.zeros_like(l)
        mask_r = (max_val == r) & (diff != 0)
        mask_g = (max_val == g) & (diff != 0)
        mask_b = (max_val == b) & (diff != 0)
        
        h[mask_r] = ((g - b) / diff)[mask_r] % 6
        h[mask_g] = ((b - r) / diff)[mask_g] + 2
        h[mask_b] = ((r - g) / diff)[mask_b] + 4
        h = h * 60  # Convert to degrees
        
        return np.stack([h * 255/360, s * 255, l * 255], axis=2).astype(np.uint8)
    
    def hsl_to_rgb(hsl_array):
        """Convert HSL back to RGB"""
        hsl_norm = hsl_array.astype(np.float32)
        h = hsl_norm[:,:,0] * 360 / 255.0  # Back to degrees
        s = hsl_norm[:,:,1] / 255.0
        l = hsl_norm[:,:,2] / 255.0
        
        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        rgb = np.zeros_like(hsl_norm)
        
        mask1 = (h >= 0) & (h < 60)
        mask2 = (h >= 60) & (h < 120)
        mask3 = (h >= 120) & (h < 180)
        mask4 = (h >= 180) & (h < 240)
        mask5 = (h >= 240) & (h < 300)
        mask6 = (h >= 300) & (h < 360)
        
        rgb[mask1] = np.stack([c, x, np.zeros_like(c)], axis=2)[mask1]
        rgb[mask2] = np.stack([x, c, np.zeros_like(c)], axis=2)[mask2]
        rgb[mask3] = np.stack([np.zeros_like(c), c, x], axis=2)[mask3]
        rgb[mask4] = np.stack([np.zeros_like(c), x, c], axis=2)[mask4]
        rgb[mask5] = np.stack([x, np.zeros_like(c), c], axis=2)[mask5]
        rgb[mask6] = np.stack([c, np.zeros_like(c), x], axis=2)[mask6]
        
        rgb = rgb + np.stack([m, m, m], axis=2)
        return (rgb * 255).astype(np.uint8)
    
    # Convert to target color space
    if img_type_upper == 'HSL':
        # Manual HSL conversion
        real_color_np = rgb_to_hsl(np.array(real_img))
        fake_color_np = rgb_to_hsl(np.array(fake_img))
    elif img_type_upper == 'RGB':
        # Already in RGB
        real_color_np = np.array(real_img)
        fake_color_np = np.array(fake_img)
    elif img_type_upper in ['YCBCR', 'LAB', 'HSV']:
        # Direct PIL conversion
        try:
            real_color = real_img.convert(img_type_upper.replace('YCBCR', 'YCbCr'))
            fake_color = fake_img.convert(img_type_upper.replace('YCBCR', 'YCbCr'))
            real_color_np = np.array(real_color)
            fake_color_np = np.array(fake_color)
        except Exception as e:
            raise ValueError(f"Failed to convert to {img_type}: {e}")
    else:
        raise ValueError(f"Unsupported color space: {img_type}")
    
    height, width, channels = real_color_np.shape
    
    # Validate patch against image dimensions
    if patch != -1:
        if patch > min(height, width):
            print(f"Warning: patch ({patch}) larger than minimum image dimension "
                  f"({min(height, width)}). Using entire image instead.")
            patch = -1
    
    # Select a fixed random ratio for 'uniform' mode
    if mode == 'uniform':
        fixed_blend_ratio = random.uniform(ratios[0], ratios[1])
    
    # Helper function to apply 2D DCT
    def apply_2d_dct(patch):
        return dct(dct(patch.T, norm='ortho').T, norm='ortho')
    
    # Helper function to apply 2D inverse DCT
    def apply_2d_idct(patch):
        return idct(idct(patch.T, norm='ortho').T, norm='ortho')
    
    # Simplified patch processing with better boundary handling
    def process_patches_improved(real_channel, fake_channel, patch):
        height, width = real_channel.shape
        mixed_channel = np.zeros_like(real_channel, dtype=np.float32)
        
        # Process overlapping patches to reduce boundary artifacts
        step_size = patch // 2 if patch >= 16 else patch
        
        for i in range(0, height, step_size):
            for j in range(0, width, step_size):
                # Extract patches with boundary handling
                end_i = min(i + patch, height)
                end_j = min(j + patch, width)
                
                real_patch = real_channel[i:end_i, j:end_j].astype(np.float32)
                fake_patch = fake_channel[i:end_i, j:end_j].astype(np.float32)
                
                # Pad if necessary
                if real_patch.shape != (patch, patch):
                    pad_h = patch - real_patch.shape[0]
                    pad_w = patch - real_patch.shape[1]
                    real_patch = np.pad(real_patch, ((0, pad_h), (0, pad_w)), mode='edge')
                    fake_patch = np.pad(fake_patch, ((0, pad_h), (0, pad_w)), mode='edge')
                
                # Apply DCT to patches
                real_dct = apply_2d_dct(real_patch)
                fake_dct = apply_2d_dct(fake_patch)
                
                # Blend DCT coefficients
                if mode == 'uniform':
                    mixed_dct = fixed_blend_ratio * real_dct + (1 - fixed_blend_ratio) * fake_dct
                else:  # variable
                    random_ratios = np.random.uniform(ratios[0], ratios[1], size=real_dct.shape)
                    mixed_dct = random_ratios * real_dct + (1 - random_ratios) * fake_dct
                
                # Apply inverse DCT
                mixed_patch = apply_2d_idct(mixed_dct)
                
                # Extract original size and blend with existing values
                orig_h = min(patch, end_i - i)
                orig_w = min(patch, end_j - j)
                mixed_patch = mixed_patch[:orig_h, :orig_w]
                
                # Use weighted averaging for overlapping regions
                if step_size < patch:
                    weight = np.ones((orig_h, orig_w))
                    mixed_channel[i:end_i, j:end_j] += mixed_patch * weight
                else:
                    mixed_channel[i:end_i, j:end_j] = mixed_patch
        
        # Normalize overlapping regions if using overlapping patches
        if step_size < patch:
            # This is a simplified normalization - could be improved
            mixed_channel = mixed_channel / 2
        
        return np.round(np.clip(mixed_channel, 0, 255)).astype(np.uint8)
    
    # Process each channel separately
    mixed_channels = []
    
    for channel_idx in range(3):  # All supported color spaces have 3 channels
        # Extract channel
        real_channel = real_color_np[:, :, channel_idx]
        fake_channel = fake_color_np[:, :, channel_idx]
        
        if patch == -1:
            # Process entire image at once
            real_channel_f = real_channel.astype(np.float32)
            fake_channel_f = fake_channel.astype(np.float32)
            
            # Compute DCT
            real_dct = apply_2d_dct(real_channel_f)
            fake_dct = apply_2d_dct(fake_channel_f)
            
            # Blend DCT coefficients
            if mode == 'uniform':
                mixed_dct = fixed_blend_ratio * real_dct + (1 - fixed_blend_ratio) * fake_dct
            else:  # variable
                random_ratios = np.random.uniform(ratios[0], ratios[1], size=real_dct.shape)
                mixed_dct = random_ratios * real_dct + (1 - random_ratios) * fake_dct
            
            # Transform back to spatial domain
            mixed_channel = apply_2d_idct(mixed_dct)
            mixed_channel = np.round(np.clip(mixed_channel, 0, 255)).astype(np.uint8)
        else:
            # Process in patches
            mixed_channel = process_patches_improved(real_channel, fake_channel, patch)
        
        mixed_channels.append(mixed_channel)
    
    # Combine channels
    mixed_np = np.stack(mixed_channels, axis=2)
    
    # Convert back to RGB for output
    if img_type_upper == 'RGB':
        mixed_img = Image.fromarray(mixed_np)
    elif img_type_upper == 'YCBCR':
        mixed_img = Image.fromarray(mixed_np, mode='YCbCr').convert('RGB')
    elif img_type_upper == 'LAB':
        mixed_img = Image.fromarray(mixed_np, mode='LAB').convert('RGB')
    elif img_type_upper == 'HSV':
        mixed_img = Image.fromarray(mixed_np, mode='HSV').convert('RGB')
    elif img_type_upper == 'HSL':
        # Manual conversion back to RGB
        rgb_array = hsl_to_rgb(mixed_np)
        mixed_img = Image.fromarray(rgb_array)
    else:
        raise ValueError(f"Conversion from {color_space} back to RGB not implemented")
    
    return mixed_img

def pixel_blend_mix(real_img, fake_img, ratios=[0.0, 1.0], color_space='RGB', 
                     mode='uniform', resize_method='LANCZOS', random_seed=None):
    """
    Mix two images at the pixel level by blending their pixel values in the specified color space.
    
    Parameters:
    -----------
    real_img : PIL.Image or str
        The real image or path to the image
    fake_img : PIL.Image or str
        The fake image or path to the image
    ratios : list, optional
        Range of blend ratios [min, max] to randomly select, values between 0.0 and 1.0
        0.0 = pure real image, 1.0 = pure fake image
    color_space : str, optional
        Color space to use for processing. Options:
        - 'RGB': Red, Green, Blue (3 channels)
        - 'YCbCr': Luminance, Chroma blue, Chroma red (3 channels)
        - 'LAB': Lightness, A*, B* (3 channels) - perceptually uniform
        - 'HSV': Hue, Saturation, Value (3 channels)
        - 'HSL': Hue, Saturation, Lightness (3 channels)
    mode : str, optional
        'uniform' - Use a single consistent blend ratio for all pixels
        'variable' - Use different random ratios for each pixel
    resize_method : str, optional
        Method to resize fake image to match real image size. Options:
        'LANCZOS', 'BILINEAR', 'BICUBIC', 'NEAREST', 'BOX', 'HAMMING'
    random_seed : int, optional
        Random seed for reproducible results
        
    Returns:
    --------
    mixed_img : PIL.Image
        The pixel-blended image (always returned as RGB)
        
    Raises:
    -------
    ValueError: If parameters are invalid
    """
    import numpy as np
    from PIL import Image
    import random
    
    # Supported color spaces (all 3-channel)
    SUPPORTED_COLOR_SPACES = ['RGB', 'YCBCR', 'LAB', 'HSV', 'HSL']
    
    # Supported resize methods
    RESIZE_METHODS = {
        'LANCZOS': Image.LANCZOS,
        'BILINEAR': Image.BILINEAR, 
        'BICUBIC': Image.BICUBIC,
        'NEAREST': Image.NEAREST,
        'BOX': Image.BOX,
        'HAMMING': Image.HAMMING
    }
    
    # Input validation
    if not (0.0 <= ratios[0] <= 1.0 and 0.0 <= ratios[1] <= 1.0):
        raise ValueError("Ratios must be between 0.0 and 1.0")
    if ratios[0] > ratios[1]:
        raise ValueError("ratios[0] must be <= ratios[1]")
    
    img_type_upper = color_space.upper()
    if img_type_upper not in SUPPORTED_COLOR_SPACES:
        supported = ', '.join(SUPPORTED_COLOR_SPACES)
        raise ValueError(f"color_space must be one of: {supported}")
    
    if mode not in ['uniform', 'variable']:
        raise ValueError("mode must be 'uniform' or 'variable'")
    
    resize_method_upper = resize_method.upper()
    if resize_method_upper not in RESIZE_METHODS:
        supported = ', '.join(RESIZE_METHODS.keys())
        raise ValueError(f"resize_method must be one of: {supported}")
    
    # Set random seed for reproducibility
    if random_seed is not None:
        random.seed(random_seed)
        np.random.seed(random_seed)
    
    # Load images if paths are provided
    if isinstance(real_img, str):
        real_img = Image.open(real_img)
    if isinstance(fake_img, str):
        fake_img = Image.open(fake_img)
    
    # Convert to RGB first (as base format)
    if real_img.mode != 'RGB':
        real_img = real_img.convert('RGB')
    if fake_img.mode != 'RGB':
        fake_img = fake_img.convert('RGB')
    
    # Resize fake image to match real image size
    if real_img.size != fake_img.size:
        resize_filter = RESIZE_METHODS[resize_method_upper]
        fake_img = fake_img.resize(real_img.size, resize_filter)
    
    # Helper functions for HSL conversion (PIL doesn't support HSL directly)
    def rgb_to_hsl(rgb_array):
        """Convert RGB to HSL manually since PIL doesn't support HSL directly"""
        rgb_norm = rgb_array / 255.0
        r, g, b = rgb_norm[:,:,0], rgb_norm[:,:,1], rgb_norm[:,:,2]
        
        max_val = np.maximum(np.maximum(r, g), b)
        min_val = np.minimum(np.minimum(r, g), b)
        diff = max_val - min_val
        
        # Lightness
        l = (max_val + min_val) / 2.0
        
        # Saturation
        s = np.zeros_like(l)
        mask = diff != 0
        s[mask & (l <= 0.5)] = diff[mask & (l <= 0.5)] / (max_val + min_val)[mask & (l <= 0.5)]
        s[mask & (l > 0.5)] = diff[mask & (l > 0.5)] / (2.0 - max_val - min_val)[mask & (l > 0.5)]
        
        # Hue
        h = np.zeros_like(l)
        mask_r = (max_val == r) & (diff != 0)
        mask_g = (max_val == g) & (diff != 0)
        mask_b = (max_val == b) & (diff != 0)
        
        h[mask_r] = ((g - b) / diff)[mask_r] % 6
        h[mask_g] = ((b - r) / diff)[mask_g] + 2
        h[mask_b] = ((r - g) / diff)[mask_b] + 4
        h = h * 60  # Convert to degrees
        
        return np.stack([h * 255/360, s * 255, l * 255], axis=2).astype(np.uint8)
    
    def hsl_to_rgb(hsl_array):
        """Convert HSL back to RGB"""
        hsl_norm = hsl_array.astype(np.float32)
        h = hsl_norm[:,:,0] * 360 / 255.0  # Back to degrees
        s = hsl_norm[:,:,1] / 255.0
        l = hsl_norm[:,:,2] / 255.0
        
        c = (1 - np.abs(2 * l - 1)) * s
        x = c * (1 - np.abs((h / 60) % 2 - 1))
        m = l - c / 2
        
        rgb = np.zeros_like(hsl_norm)
        
        mask1 = (h >= 0) & (h < 60)
        mask2 = (h >= 60) & (h < 120)
        mask3 = (h >= 120) & (h < 180)
        mask4 = (h >= 180) & (h < 240)
        mask5 = (h >= 240) & (h < 300)
        mask6 = (h >= 300) & (h < 360)
        
        rgb[mask1] = np.stack([c, x, np.zeros_like(c)], axis=2)[mask1]
        rgb[mask2] = np.stack([x, c, np.zeros_like(c)], axis=2)[mask2]
        rgb[mask3] = np.stack([np.zeros_like(c), c, x], axis=2)[mask3]
        rgb[mask4] = np.stack([np.zeros_like(c), x, c], axis=2)[mask4]
        rgb[mask5] = np.stack([x, np.zeros_like(c), c], axis=2)[mask5]
        rgb[mask6] = np.stack([c, np.zeros_like(c), x], axis=2)[mask6]
        
        rgb = rgb + np.stack([m, m, m], axis=2)
        return (rgb * 255).astype(np.uint8)
    
    # Convert to target color space
    if img_type_upper == 'HSL':
        # Manual HSL conversion
        real_color_np = rgb_to_hsl(np.array(real_img))
        fake_color_np = rgb_to_hsl(np.array(fake_img))
    elif img_type_upper == 'RGB':
        # Already in RGB
        real_color_np = np.array(real_img)
        fake_color_np = np.array(fake_img)
    elif img_type_upper in ['YCBCR', 'LAB', 'HSV']:
        # Direct PIL conversion
        try:
            real_color = real_img.convert(img_type_upper.replace('YCBCR', 'YCbCr'))
            fake_color = fake_img.convert(img_type_upper.replace('YCBCR', 'YCbCr'))
            real_color_np = np.array(real_color)
            fake_color_np = np.array(fake_color)
        except Exception as e:
            raise ValueError(f"Failed to convert to {color_space}: {e}")
    else:
        raise ValueError(f"Unsupported color space: {img_type}")
    
    # Convert to float32 for blending calculations
    real_arr = real_color_np.astype(np.float32)
    fake_arr = fake_color_np.astype(np.float32)
    
    # Perform pixel blending
    if mode == 'uniform':
        # Uniform blend with a single alpha value
        blend_factor = random.uniform(ratios[0], ratios[1])
        blended = blend_factor * fake_arr + (1 - blend_factor) * real_arr
    else:  # variable
        # Variable blend with per-pixel alpha values
        # Create random alpha mask of the same shape (height, width, 1)
        # We use a single channel mask and broadcast it to all color channels
        alpha_mask = np.random.uniform(ratios[0], ratios[1], 
                                     size=(real_arr.shape[0], real_arr.shape[1], 1))
        
        # Perform the blend: alpha * fake + (1-alpha) * real
        blended = alpha_mask * fake_arr + (1 - alpha_mask) * real_arr
    
    # Convert back to uint8
    blended = np.clip(blended, 0, 255).astype(np.uint8)
    
    # Convert back to RGB for output
    if img_type_upper == 'RGB':
        mixed_img = Image.fromarray(blended)
    elif img_type_upper == 'YCBCR':
        mixed_img = Image.fromarray(blended, mode='YCbCr').convert('RGB')
    elif img_type_upper == 'LAB':
        mixed_img = Image.fromarray(blended, mode='LAB').convert('RGB')
    elif img_type_upper == 'HSV':
        mixed_img = Image.fromarray(blended, mode='HSV').convert('RGB')
    elif img_type_upper == 'HSL':
        # Manual conversion back to RGB
        rgb_array = hsl_to_rgb(blended)
        mixed_img = Image.fromarray(rgb_array)
    else:
        raise ValueError(f"Conversion from {color_space} back to RGB not implemented")
    
    return mixed_img

