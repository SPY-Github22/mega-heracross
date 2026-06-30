import os
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
except ImportError:
    pass

class DeepGlobeDataset(Dataset):
    """
    Phase 15: DeepGlobe Road Extraction Dataset Loader.
    Outputs 10-channel tensors to match our SegformerB3Custom model:
    Channels 0-2: RGB
    Channel 3: Zeros (NIR)
    Channels 4-5: Zeros (SAR)
    Channels 6-9: Zeros (Temporal Diff)
    """
    def __init__(self, root_dir: str, split: str = "train", augment: bool = True, output_size: int = 512):
        self.root_dir = Path(root_dir)
        self.split = split
        self.augment = augment
        self.output_size = output_size
        
        self.images_dir = self.root_dir / split
        
        # Check if real data exists. If not, use synthetic fallback
        self.use_fallback = not self.images_dir.exists()
        
        if self.use_fallback:
            print(f"⚠️ [DeepGlobeDataset] Real dataset not found at {self.images_dir}. Using synthetic fallback.")
            self.num_samples = 100 if split == "train" else 20
        else:
            self.image_files = sorted(list(self.images_dir.glob("*_sat.jpg")))
            self.mask_files = sorted(list(self.images_dir.glob("*_mask.png")))
            self.num_samples = len(self.image_files)
            
        if self.augment:
            self.transform = A.Compose([
                A.RandomCrop(width=output_size, height=output_size),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, p=0.5)
            ])
        else:
            self.transform = A.Compose([
                A.CenterCrop(width=output_size, height=output_size)
            ])

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        if self.use_fallback:
            # Generate dummy 1024x1024 DeepGlobe-like image (grass + road)
            rng = np.random.RandomState(idx + (1000 if self.split == 'train' else 2000))
            
            # Base grass
            img = rng.uniform(0.1, 0.4, (1024, 1024, 3)).astype(np.float32)
            img[:, :, 1] += 0.2 # greener
            
            mask = np.zeros((1024, 1024), dtype=np.uint8)
            
            # Draw a thick road (DeepGlobe roads are usually wide, ~10-20px)
            x0, y0 = rng.randint(0, 1024), 0
            x1, y1 = rng.randint(0, 1024), 1024
            
            import cv2
            cv2.line(mask, (x0, y0), (x1, y1), color=1, thickness=25)
            
            # Color road gray
            img[mask == 1] = rng.uniform(0.4, 0.6, 3)
            
        else:
            import cv2
            img_path = str(self.image_files[idx])
            mask_path = str(self.mask_files[idx])
            
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = img.astype(np.float32) / 255.0
            
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            mask = (mask > 127).astype(np.uint8)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
        # Pad to 10 channels
        h, w, _ = img.shape
        full_tensor = np.zeros((10, h, w), dtype=np.float32)
        full_tensor[0:3, :, :] = img.transpose(2, 0, 1) # RGB
        
        return torch.from_numpy(full_tensor), torch.from_numpy(mask.astype(np.int64))
