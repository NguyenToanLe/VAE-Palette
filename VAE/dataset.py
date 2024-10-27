import os
import torch
from torch import Tensor
from pathlib import Path
from typing import List, Optional, Sequence, Union, Any, Callable
from torchvision.datasets.folder import default_loader
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CelebA
import zipfile
from PIL import Image


IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def pil_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset(dir):
    # dir has the form dir_gt+dir_mask
    gt_dir = dir.split("+")[0]
    mask_dir = dir.split("+")[1]

    images = {
        "gt": [],
        "mask": []
    }

    assert os.path.isdir(gt_dir), '%s is not a valid directory' % dir
    assert os.path.isdir(mask_dir), '%s is not a valid directory' % dir

    gts = sorted(os.listdir(gt_dir))
    masks = sorted(os.listdir(mask_dir))

    for i in range(len(gts)):
        gt_fname = gts[i]
        mask_fname = masks[i]
        if is_image_file(gt_fname) and is_image_file(mask_fname):
            images["gt"].append(os.path.join(gt_dir, gt_fname))
            images["mask"].append(os.path.join(mask_dir, mask_fname))

    return images


# Add your custom dataset class here
class MyDataset(Dataset):
    def __init__(self, data_root, image_size=(256, 256), loader=pil_loader, validation_split=0.05, split='train'):
        self.imgs = make_dataset(data_root)
        self.tfs = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor(),
            #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.tfs_mask = transforms.Compose([
            transforms.Resize((image_size[0], image_size[1])),
            transforms.ToTensor()
        ])
        self.loader = loader
        self.image_size = image_size

        data_len = len(self.imgs["gt"])

        if isinstance(validation_split, int):
            assert validation_split < data_len, "Validation set size is configured to be larger than entire dataset."
            valid_len = validation_split
        else:
            valid_len = int(data_len * validation_split)
        data_len -= valid_len
        if split == 'train':
            self.imgs["gt"] = self.imgs["gt"][:data_len]
            self.imgs["mask"] = self.imgs["mask"][:data_len]
        elif split == 'val':
            self.imgs["gt"] = self.imgs["gt"][data_len:]
            self.imgs["mask"] = self.imgs["mask"][data_len:]

    def __getitem__(self, index):
        img = self.loader(self.imgs["gt"][index])
        img = self.tfs(img)

        mask = self.tfs_mask(self.loader(self.imgs["mask"][index]).convert('1'))
        mask = mask.repeat(3, 1, 1)

        mask_img = img * (1. - mask)

        return mask_img, mask

    def __len__(self):
        return len(self.imgs["gt"])


class MyCelebA(CelebA):
    """
    A work-around to address issues with pytorch's celebA dataset class.
    
    Download and Extract
    URL : https://drive.google.com/file/d/1m8-EBPgi5MRubrm6iQjafK2QMHDBMSfJ/view?usp=sharing
    """
    
    def _check_integrity(self) -> bool:
        return True


class OxfordPets(Dataset):
    """
    URL = https://www.robots.ox.ac.uk/~vgg/data/pets/
    """
    def __init__(self, 
                 data_path: str, 
                 split: str,
                 transform: Callable,
                **kwargs):
        self.data_dir = Path(data_path) / "OxfordPets"        
        self.transforms = transform
        imgs = sorted([f for f in self.data_dir.iterdir() if f.suffix == '.jpg'])
        
        self.imgs = imgs[:int(len(imgs) * 0.75)] if split == "train" else imgs[int(len(imgs) * 0.75):]
    
    def __len__(self):
        return len(self.imgs)
    
    def __getitem__(self, idx):
        img = default_loader(self.imgs[idx])
        
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, 0.0 # dummy datat to prevent breaking 


class VAEDataset(LightningDataModule):
    """
    PyTorch Lightning data module 

    Args:
        data_dir: root directory of your dataset.
        train_batch_size: the batch size to use during training.
        val_batch_size: the batch size to use during validation.
        patch_size: the size of the crop to take from the original images.
        num_workers: the number of parallel workers to create to load data
            items (see PyTorch's Dataloader documentation for more details).
        pin_memory: whether prepared items should be loaded into pinned memory
            or not. This can improve performance on GPUs.
    """

    def __init__(
        self,
        data_path: str,
        train_batch_size: int = 8,
        val_batch_size: int = 8,
        patch_size: Union[int, Sequence[int]] = (256, 256),
        num_workers: int = 0,
        pin_memory: bool = False,
        validation_split: Union[int, float] = 0.05,
        **kwargs,
    ):
        super().__init__()

        self.data_dir = data_path
        self.inference_data_dir = kwargs['inference_data_path']
        self.test_data_dir = kwargs['test_data_path']
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.patch_size = patch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.validation_split = validation_split

    def setup(self, stage: Optional[str] = None) -> None:

#       =========================  CelebA Dataset  =========================

        train_transforms = transforms.Compose([transforms.RandomHorizontalFlip(),
                                               transforms.CenterCrop(148),
                                               transforms.Resize(self.patch_size),
                                               transforms.ToTensor(), ])

        # self.train_dataset = MyCelebA(
        #     self.data_dir,
        #     split='train',
        #     transform=train_transforms,
        #     download=False,
        # )

        self.train_dataset = MyDataset(
            data_root=self.inference_data_dir,
            image_size=(self.patch_size, self.patch_size),
            validation_split=self.validation_split,
            split='train',
        )
        
        # Replace CelebA with your dataset
        self.val_dataset = MyDataset(
            data_root=self.inference_data_dir,
            image_size=(self.patch_size, self.patch_size),
            validation_split=self.validation_split,
            split='val',
        )

        self.test_dataset = MyDataset(
            data_root=self.test_data_dir,
            image_size=(self.patch_size, self.patch_size),
            validation_split=self.validation_split,
            split='test',
        )
#       ===============================================================
        
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            num_workers=self.num_workers,
            shuffle=False,
            pin_memory=self.pin_memory,
        )
    
    def test_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return DataLoader(
            self.test_dataset,
            batch_size=144,
            num_workers=self.num_workers,
            shuffle=True,
            pin_memory=self.pin_memory,
        )
     