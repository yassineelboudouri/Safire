import glob

from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class ImageDataset(Dataset):
    def __init__(self, root, target_size):
        self.transform = transforms.Compose(
            [
                transforms.Resize((target_size, target_size), transforms.InterpolationMode.LANCZOS),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                # Normalize the images between -1 and 1
                transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
            ]
        )

        self.files = sorted(glob.glob(root + "/*/*.*"))

    def __getitem__(self, index):
        img = Image.open(self.files[index])

        if img.mode != 'RGB':
            img = img.convert('RGB')

        min_side = min(img.size)

        img = transforms.RandomCrop(min_side)(img)
        img = self.transform(img)

        return img

    def __len__(self):
        return len(self.files)
