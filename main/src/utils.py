from torch.utils.data import Dataset
import os
from PIL import Image
from torchvision import transforms
import torch
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent

class ImageFolderDataset(Dataset):
    def __init__(self, root, transform=None):
        super(ImageFolderDataset, self).__init__()
        self.root = Path(root)
        self.transform = transform

        self.files = []

        # 
        for path in self.root.rglob('*'):

            # skip dirs
            if not path.is_file():
                continue

            # check image extension
            if path.suffix.lower() not in ['.png', '.jpg', '.jpeg']:
                continue

            # check whether the image be loaded
            try:
                with Image.open(path) as img:
                    img.load()

                self.files.append(path)

            except Exception as e:
                print(f"Skipping corrupted image {path}")
                print(f"Reason: {e}")

    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, index):
        image_path = self.files[index]
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        return image
    
def get_transform(size, crop, final_size):
    transform_list = []
    if size > 0:
        transform_list.append(transforms.Resize(size))
    if crop:
        transform_list.append(transforms.RandomCrop(final_size))
    else:
        transform_list.append(transforms.Resize(final_size))

    transform_list.append(transforms.ToTensor())
    return transforms.Compose(transform_list)
