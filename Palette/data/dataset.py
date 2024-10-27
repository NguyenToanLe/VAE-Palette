import torch.utils.data as data
from torchvision import transforms
from PIL import Image
import os
import torch
import numpy as np
from torchvision.transforms.functional import gaussian_blur

from models.vanilla_vae import VanillaVAE

from .util.mask import (bbox2mask, brush_stroke_mask, get_irregular_mask, random_bbox, random_cropping_bbox)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    # dir has the form dir_img+dir_gt+dir_mask
    col_img_dir = dir.split("+")[1]
    gt_dir = dir.split("+")[0]
    mask_dir = dir.split("+")[2]
    
    images = {
        "col_img": [],
        "gt": [],
        "mask": []
    }
    
    assert os.path.isdir(col_img_dir), '%s is not a valid directory' % dir
    assert os.path.isdir(gt_dir), '%s is not a valid directory' % dir
    assert os.path.isdir(mask_dir), '%s is not a valid directory' % dir

    col_imgs = sorted(os.listdir(col_img_dir))
    gts = sorted(os.listdir(gt_dir))
    masks = sorted(os.listdir(mask_dir))
    
    for i in range(len(col_imgs)):
        col_img_fname = col_imgs[i]
        gt_fname = gts[i]
        mask_fname = masks[i]
        if is_image_file(col_img_fname) and is_image_file(gt_fname) and is_image_file(mask_fname):
            images["col_img"].append(os.path.join(col_img_dir, col_img_fname))
            images["gt"].append(os.path.join(gt_dir, gt_fname))
            images["mask"].append(os.path.join(mask_dir, mask_fname))

    return images


def pil_loader(path):
    return Image.open(path).convert('RGB')


def create_gaussian_noise(img, mask):
    """
        Generates Gaussian noise for the unmasked region ("0") of the image.

        Parameters:
        img (torch.Tensor): Input image tensor of shape (C, H, W) or (H, W)
        mask (torch.Tensor): Binary mask tensor of shape (H, W), where 1 indicates the known region

        Returns:
        torch.Tensor: Image with Gaussian noise added to the unmasked regions
        """

    # Apply Gaussian blur (median filter equivalent in this context)
    img_blur = gaussian_blur(img, kernel_size=(5, 5))

    # Calculate the residual, excluding the mask region
    residual = torch.where(mask == 1, torch.tensor(0.0, dtype=img.dtype, device=img.device), img - img_blur)

    # Extract unmasked residual values for statistical analysis
    unmasked_residual = residual[mask == 0].cpu().numpy()
    mu = unmasked_residual.mean()
    std = unmasked_residual.std()

    # Generate normally distributed noise with estimated mean and standard deviation
    normal_approx_noise = torch.tensor(np.random.normal(mu, std, img.shape), dtype=img.dtype, device=img.device)

    return img * mask + normal_approx_noise * (1 - mask)


class InpaintDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'bbox':
            mask = bbox2mask(self.image_size, random_bbox())
        elif self.mask_mode == 'center':
            h, w = self.image_size
            mask = bbox2mask(self.image_size, (h//4, w//4, h//2, w//2))
        elif self.mask_mode == 'irregular':
            mask = get_irregular_mask(self.image_size)
        elif self.mask_mode == 'free_form':
            mask = brush_stroke_mask(self.image_size)
        elif self.mask_mode == 'hybrid':
            regular_mask = bbox2mask(self.image_size, random_bbox())
            irregular_mask = brush_stroke_mask(self.image_size, )
            mask = regular_mask | irregular_mask
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class UncroppingDataset(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = imgs[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        path = self.imgs[index]
        img = self.tfs(self.loader(path))
        mask = self.get_mask()
        cond_image = img*(1. - mask) + mask*torch.randn_like(img)
        mask_img = img*(1. - mask) + mask

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs)

    def get_mask(self):
        if self.mask_mode == 'manual':
            mask = bbox2mask(self.image_size, self.mask_config['shape'])
        elif self.mask_mode == 'fourdirection' or self.mask_mode == 'onedirection':
            mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode=self.mask_mode))
        elif self.mask_mode == 'hybrid':
            if np.random.randint(0,2)<1:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='onedirection'))
            else:
                mask = bbox2mask(self.image_size, random_cropping_bbox(mask_mode='fourdirection'))
        elif self.mask_mode == 'file':
            pass
        else:
            raise NotImplementedError(
                f'Mask mode {self.mask_mode} has not been implemented.')
        return torch.from_numpy(mask).permute(2,0,1)


class ColorizationDataset(data.Dataset):
    def __init__(self, data_root, data_flist, data_len=-1, image_size=[224, 224], loader=pil_loader):
        self.data_root = data_root
        flist = make_dataset(data_flist)
        if data_len > 0:
            self.flist = flist[:int(data_len)]
        else:
            self.flist = flist
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5,0.5, 0.5])
        ])
        self.loader = loader
        self.image_size = image_size

    def __getitem__(self, index):
        ret = {}
        file_name = str(self.flist[index]).zfill(5) + '.png'

        img = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'color', file_name)))
        cond_image = self.tfs(self.loader('{}/{}/{}'.format(self.data_root, 'gray', file_name)))

        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['path'] = file_name
        return ret

    def __len__(self):
        return len(self.flist)


class UncroppingDatasetCustom(data.Dataset):
    def __init__(self, data_root, mask_config={}, data_len=-1, image_size=[256, 256], loader=pil_loader, VAE_config={}):
        imgs = make_dataset(data_root)
        if data_len > 0:
            self.imgs = {}
            for k, v in imgs.items():
                self.imgs[k] = v[:int(data_len)]
        else:
            self.imgs = imgs
        self.tfs = transforms.Compose([
                transforms.Resize((image_size[0], image_size[1])),
                transforms.ToTensor(),
                #transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        # self.tfs_mask = transforms.Compose([
        #         transforms.Resize((image_size[0], image_size[1])),
        #         transforms.ToTensor()
        # ])
        self.tfs_normalize = transforms.Compose([
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.loader = loader
        self.mask_config = mask_config
        self.mask_mode = self.mask_config['mask_mode']
        self.original_input = self.mask_config['original_input']
        self.image_size = image_size

        # Load VAE
        self.setup_VAE(VAE_config)


    def __getitem__(self, index):
        ret = {}
        path = self.imgs["col_img"][index]

        img = self.loader(self.imgs["gt"][index])
        # img = np.asarray(img).astype("f")[np.newaxis, :, :]
        # img = np.tile(img, (3, 1, 1))
        img = self.tfs(img)
        cond_image = self.loader(self.imgs["col_img"][index])
        # cond_image = np.asarray(cond_image).astype("f")[np.newaxis, :, :]
        # cond_image = np.tile(cond_image, (3, 1, 1))
        cond_image = self.tfs(cond_image)
        mask = 1. - self.tfs(self.loader(self.imgs["mask"][index]).convert('1'))
        mask = mask.repeat(3, 1, 1)

        # if not self.original_input:
        #     cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        # if self.concat:
        #     blackout = img * (1. - mask) + mask * torch.randn_like(img)
        #     cond_image = torch.concat([blackout, cond_image], dim=0)

        # cond_image = img * (1. - mask) + mask * torch.randn_like(img)
        noise = self.sample_from_VAE(cond_image, mask)
        noise = self.tfs_normalize(noise)
        img = self.tfs_normalize(img)
        cond_image = img * (1. - mask) + mask * noise
        mask_img = img * (1. - mask) + mask
        
        ret['gt_image'] = img
        ret['cond_image'] = cond_image
        ret['mask_image'] = mask_img
        ret['mask'] = mask
        ret['path'] = path.rsplit("/")[-1].rsplit("\\")[-1]
        return ret

    def __len__(self):
        return len(self.imgs["col_img"])

    def setup_VAE(self, VAE_config={}):
        self.VAE_config = dict(VAE_config)
        self.VAE_model = VanillaVAE(**self.VAE_config)
        checkpoint = torch.load(self.VAE_config['checkpoint_path'], map_location='cuda:0')
        state_dict = checkpoint['state_dict']
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k.replace('model.', '')  # Adjust this if  keys have a different prefix
            new_state_dict[new_key] = v
        self.VAE_model.load_state_dict(new_state_dict)
        self.VAE_model.eval()

    def sample_from_VAE(self, img, mask):
        img_expand = img[None, :, :, :]
        mask_expand = mask[None, :, :, :]
        [mu, log_var] = self.VAE_model.encode(img_expand * mask_expand)
        mu = mu[:, :, np.newaxis]
        mu = mu.repeat(3, 1, 256)
        log_var = log_var[:, :, np.newaxis]
        log_var = log_var.repeat(3, 1, 256)
        latent = self.VAE_model.reparameterize(mu, log_var)
        # This ensures the noise is a pure tensor and does not require any gradient computation
        latent = latent.detach().clone()
        return latent
