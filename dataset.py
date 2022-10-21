import glob
import random
import os
import numpy as np

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])


class ImageDataset(Dataset):
    def __init__(self, root, target_size):
        self.transform = transforms.Compose(
            [
                transforms.Resize((target_size, target_size), Image.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ]
        )

        self.files = sorted(glob.glob(root + "/*/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index])
        min_side = min(img.size)

        img = transforms.CenterCrop(min_side)(img)
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)
