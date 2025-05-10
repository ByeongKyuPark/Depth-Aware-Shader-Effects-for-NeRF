import torch
import numpy as np
import os
from PIL import Image
import json
from torch.utils.data import Dataset
import torchvision.transforms as transforms

class NeRFDataset(Dataset):
    def __init__(self, config, split='train'):
        """
        Dataset for NeRF training and evaluation.
        
        Args:
            config: Configuration object with dataset parameters
            split: 'train', 'val', or 'test' split
        """
        self.config = config
        self.split = split
        self.transform = transforms.ToTensor()
        
        # Load the dataset based on the format
        if config.dataset_type == 'nerf_synthetic':
            self._load_nerf_synthetic()
        else:
            # Existing code for custom datasets
            self._load_custom_dataset()
            
    def _load_nerf_synthetic(self):
        """Load NeRF synthetic dataset format."""
        # Set paths for the specific scene and split
        scene_path = os.path.join(self.config.dataset_path, self.config.scene)
        
        # Updated path: transforms files are directly in the scene folder
        transforms_path = os.path.join(scene_path, f'transforms_{self.split}.json')
        
        # Load the transforms file
        with open(transforms_path, 'r') as f:
            self.meta = json.load(f)
        
        self.frames = self.meta['frames']
        
        # For NeRF synthetic dataset, construct the correct path to the first image
        # The file_path in JSON might already include the split directory (train/test/val)
        first_frame = self.frames[0]
        file_path = first_frame['file_path']
        
        # Remove any leading "./" from the file_path
        if file_path.startswith('./'):
            file_path = file_path[2:]
            
        # Check if file_path already contains the split directory
        if not file_path.startswith(self.split + '/'):
            first_img_path = os.path.join(scene_path, file_path + '.png')
        else:
            # If it already has the split directory, just use the path directly
            first_img_path = os.path.join(scene_path, file_path + '.png')
        
        print(f"Looking for image at: {first_img_path}")
        
        with Image.open(first_img_path) as img:
            self.W, self.H = img.size
        
        # Calculate focal length from camera angle
        if 'camera_angle_x' in self.meta:
            self.focal = 0.5 * self.W / np.tan(0.5 * self.meta['camera_angle_x'])
        elif 'fl_x' in self.meta:
            self.focal = self.meta['fl_x']
        else:
            # Fallback to a reasonable default focal length
            self.focal = self.W / (2 * np.tan(np.radians(55) / 2))
        
        # Near and far plane from the config
        self.near = self.config.near
        self.far = self.config.far
        
        # Setup appearance embeddings (one per image)
        self.use_appearance = self.config.use_appearance
        if self.use_appearance:
            # Initialize random embeddings that will be optimized during training
            self.appearance_embeddings = torch.nn.Parameter(
                torch.randn(len(self.frames), self.config.appearance_dim)
            )
            
    def _load_custom_dataset(self):
        """Load custom dataset format."""
        # Load transforms.json file which contains camera parameters
        with open(os.path.join(self.config.dataset_path, '../transforms.json'), 'r') as f:
            self.meta = json.load(f)
        
        # Filter frames based on split
        if self.split == 'train':
            self.frames = self.meta['frames'][:-1]  # Use all but last frame for training
        else:
            self.frames = self.meta['frames'][-1:]  # Use last frame for validation
        
        # Image processing
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])
        
        # Camera parameters
        if 'camera_angle_x' in self.meta:
            self.focal = 0.5 * self.meta['w'] / np.tan(0.5 * self.meta['camera_angle_x'])
        elif 'fl_x' in self.meta:
            self.focal = self.meta['fl_x']
        else:
            # Default focal length assumption based on field of view
            self.focal = self.meta['w'] / (2 * np.tan(np.radians(55) / 2))
            
        self.H = self.meta['h']
        self.W = self.meta['w']
        
        # Setup appearance embeddings (one per image)
        self.use_appearance = self.config.use_appearance
        if self.use_appearance:
            # Initialize random embeddings that will be optimized during training
            self.appearance_embeddings = torch.nn.Parameter(
                torch.randn(len(self.frames), self.config.appearance_dim)
            )
        
        # For bounding scene (near and far planes)
        self.near = 2.0  # Near plane distance
        self.far = 6.0   # Far plane distance
            
    def __len__(self):
        return len(self.frames)
    
    def __getitem__(self, idx):
        """Get a single image and its camera pose."""
        frame = self.frames[idx]
        
        if self.config.dataset_type == 'nerf_synthetic':
            # Get the file path from the frame
            file_path = frame['file_path']
            
            # Remove any leading "./" from the file_path
            if file_path.startswith('./'):
                file_path = file_path[2:]
                
            # Check if file_path already contains the split directory
            if not file_path.startswith(self.split + '/'):
                image_path = os.path.join(
                    self.config.dataset_path,
                    self.config.scene,
                    file_path + '.png'
                )
            else:
                # If it already has the split directory, just use the path directly
                image_path = os.path.join(
                    self.config.dataset_path,
                    self.config.scene,
                    file_path + '.png'
                )
            
            img = Image.open(image_path)
            img = self.transform(img)  # (4, H, W) with alpha channel
            
            # Handle alpha channel - separate RGB and alpha
            rgb = img[:3]  # (3, H, W)
            alpha = img[3:4] if img.shape[0] == 4 else torch.ones_like(img[:1])  # (1, H, W)
            
            # Get camera-to-world transform
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)  # (4, 4)
            
            # Generate appearance embedding index
            if self.use_appearance:
                appearance_idx = idx
            else:
                appearance_idx = -1  # Not used
            
            return {
                'img': rgb,  # (3, H, W)
                'alpha': alpha,  # (1, H, W)
                'c2w': c2w,  # (4, 4)
                'appearance_idx': appearance_idx,
                'img_idx': idx,
            }
        else:
            # Load image
            image_path = os.path.join(self.config.dataset_path, frame['file_path'])
            img = Image.open(image_path).convert('RGB')
            img = self.transform(img)  # (3, H, W)
            
            # Get camera-to-world transform
            c2w = torch.tensor(frame['transform_matrix'], dtype=torch.float32)  # (4, 4)
            
            # Handle alpha channel for custom dataset
            rgb = img  # (3, H, W)
            alpha = None
        
        # Generate appearance embedding index
        if self.use_appearance:
            appearance_idx = idx
        else:
            appearance_idx = -1  # Not used
        
        return {
            'img': rgb,  # (3, H, W)
            'alpha': alpha,  # (1, H, W) or None
            'c2w': c2w,  # (4, 4)
            'appearance_idx': appearance_idx,
            'img_idx': idx,
        }
    
    def get_rays(self, idx=None, batch_size=None):
        """
        Generate rays for an entire image or a random subset.
        
        Args:
            idx: Optional image index. If None, random rays are sampled across all images.
            batch_size: Optional override for config.batch_size
            
        Returns:
            Dictionary containing ray origins, directions, and corresponding pixel colors.
        """
        from .ray_utils import get_rays
        
        if batch_size is None:
            batch_size = self.config.batch_size
        
        if idx is not None:
            # Get specific image
            sample = self.__getitem__(idx)
            img = sample['img']  # (3, H, W)
            alpha = sample['alpha']  # (1, H, W) or None
            c2w = sample['c2w']  # (4, 4)
            
            # Generate rays for the entire image
            rays_o, rays_d = get_rays(self.H, self.W, self.focal, c2w)  # (H, W, 3), (H, W, 3)
            
            # Reshape for processing
            rays_o = rays_o.reshape(-1, 3)  # (H*W, 3)
            rays_d = rays_d.reshape(-1, 3)  # (H*W, 3)
            rgb = img.permute(1, 2, 0).reshape(-1, 3)  # (H*W, 3)
            
            if alpha is not None:
                alpha = alpha.permute(1, 2, 0).reshape(-1, 1)  # (H*W, 1)
            
            return {
                'rays_o': rays_o,  # (H*W, 3)
                'rays_d': rays_d,  # (H*W, 3)
                'rgb': rgb,        # (H*W, 3)
                'alpha': alpha,    # (H*W, 1) or None
                'appearance_idx': sample['appearance_idx'],
                'img_idx': sample['img_idx'],
            }
        else:
            # Random rays from random images
            img_idx = np.random.randint(0, len(self))
            sample = self.__getitem__(img_idx)
            img = sample['img']  # (3, H, W)
            alpha = sample['alpha']  # (1, H, W) or None
            c2w = sample['c2w']  # (4, 4)
            
            # Generate rays for the entire image
            rays_o, rays_d = get_rays(self.H, self.W, self.focal, c2w)  # (H, W, 3), (H, W, 3)
            
            # Select random pixel locations
            select_inds = np.random.choice(self.H * self.W, size=batch_size, replace=False)
            
            # Extract selected rays and colors
            rays_o = rays_o.reshape(-1, 3)[select_inds]  # (batch_size, 3)
            rays_d = rays_d.reshape(-1, 3)[select_inds]  # (batch_size, 3)
            rgb = img.permute(1, 2, 0).reshape(-1, 3)[select_inds]  # (batch_size, 3)
            
            if alpha is not None:
                alpha = alpha.permute(1, 2, 0).reshape(-1, 1)[select_inds]  # (batch_size, 1)
            
            return {
                'rays_o': rays_o,  # (batch_size, 3)
                'rays_d': rays_d,  # (batch_size, 3)
                'rgb': rgb,        # (batch_size, 3)
                'alpha': alpha,    # (batch_size, 1) or None
                'appearance_idx': sample['appearance_idx'],
                'img_idx': sample['img_idx'],
            }
