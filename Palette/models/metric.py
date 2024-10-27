import torch
from torch import nn
from torch.autograd import Variable
from torch.nn import functional as F
import torch.utils.data
import lpips
import os
from PIL import Image
from tqdm import tqdm

from torchvision.models.inception import inception_v3
from .unet3d import UNet3D
from torchvision import models, transforms

import numpy as np
from scipy.stats import entropy
from math import exp


def mae(input, target):
    with torch.no_grad():
        loss = nn.L1Loss()
        output = loss(input, target)
    return output


def inception_score(imgs, cuda=True, batch_size=4, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

    imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
    cuda -- whether or not to run on GPU
    batch_size -- batch size for feeding into Inception v3
    splits -- number of splits
    """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load inception model
    inception_model = inception_v3(pretrained=True, transform_input=False).type(dtype)
    inception_model.eval()
    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)
    def get_pred(x):
        if resize:
            x = up(x)
        x = inception_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i*batch_size:i*batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k+1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


@torch.no_grad()
def calculate_FID(mu1, mu2, cov1, cov2):
    prod = torch.linalg.cholesky(torch.matmul(cov1, cov2))
    dist = np.sum((mu1 - mu2) ** 2) + np.trace(cov1 + cov2 - 2 * prod)
    return dist


def calculate_FID_from_path(path1, path2):
    print('Calculating FID given paths %s and %s...' % (path1, path2))

    images_1 = []
    images_2 = []

    for _, i in tqdm(enumerate(sorted(os.listdir(path1)))):
        img1 = Image.open(os.path.join(path1, i)).convert('RGB').convert('L')
        images_1.append(img1)

        img2 = Image.open(os.path.join(path2, i)).convert('RGB').convert('L')
        images_2.append(img2)

    images_1 = torch.cat(images_1, dim=0).cpu().detach().numpy()
    images_2 = torch.cat(images_2, dim=0).cpu().detach().numpy()

    mu1 = np.mean(images_1, axis=0)
    mu2 = np.mean(images_2, axis=0)
    cov1 = np.cov(images_1, rowvar=False)
    cov2 = np.cov(images_2, rowvar=False)

    return calculate_FID(mu1, mu2, cov1, cov2)


@torch.no_grad()
def calculate_inception_score(imgs, cuda=True, batch_size=32, resize=False, splits=1):
    """Computes the inception score of the generated images imgs

       imgs -- Torch dataset of (3xHxW) numpy images normalized in the range [-1, 1]
       cuda -- whether or not to run on GPU
       batch_size -- batch size for feeding into Inception v3
       splits -- number of splits
       """
    N = len(imgs)

    assert batch_size > 0
    assert N > batch_size

    # Set up dtype
    if cuda:
        dtype = torch.cuda.FloatTensor
    else:
        if torch.cuda.is_available():
            print("WARNING: You have a CUDA device, so you should probably set cuda=True")
        dtype = torch.FloatTensor

    # Set up dataloader
    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size)

    # Load Genesis model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    base_model = UNet3D()
    weight_dir = 'pretrained_weights/Genesis_Chest_CT.pt'
    checkpoint = torch.load(weight_dir)
    state_dict = checkpoint['state_dict']
    unParalled_state_dict = {}
    for key in state_dict.keys():
        unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
    base_model.load_state_dict(unParalled_state_dict)
    genesis_model = TargetNet(base_model).to(device)
    genesis_model.eval()

    up = nn.Upsample(size=(299, 299), mode='bilinear').type(dtype)

    def get_pred(x):
        if resize:
            x = up(x)
        x = genesis_model(x)
        return F.softmax(x).data.cpu().numpy()

    # Get predictions
    preds = np.zeros((N, 1000))

    for i, batch in enumerate(dataloader, 0):
        batch = batch.type(dtype)
        batchv = Variable(batch)
        batch_size_i = batch.size()[0]

        preds[i * batch_size:i * batch_size + batch_size_i] = get_pred(batchv)

    # Now compute the mean kl-div
    split_scores = []

    for k in range(splits):
        part = preds[k * (N // splits): (k + 1) * (N // splits), :]
        py = np.mean(part, axis=0)
        scores = []
        for i in range(part.shape[0]):
            pyx = part[i, :]
            scores.append(entropy(pyx, py))
        split_scores.append(np.exp(np.mean(scores)))

    return np.mean(split_scores), np.std(split_scores)


def calculate_inception_score_from_path(path):
    print('Calculating IS given path %s...' % (path))

    images = []
    for _, i in tqdm(enumerate(sorted(os.listdir(path)))):
        img = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
        images.append(img)

    mean, std = calculate_inception_score(images)

    return mean, std


def calculate_mse_score_from_path(path1, path2, return_std=False):
    print('Calculating MSE given paths %s and %s...' % (path1, path2))

    mse_values = []

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    for _, i in tqdm(enumerate(sorted(os.listdir(path1)))):
        img1 = Image.open(os.path.join(path1, i)).convert('RGB').convert('L')
        img1 = transform(img1).unsqueeze(0)

        img2 = Image.open(os.path.join(path2, i)).convert('RGB').convert('L')
        img2 = transform(img2).unsqueeze(0)

        mse_values.append(torch.mean((img1 - img2) ** 2))

    if not return_std:
        return np.mean(mse_values)
    else:
        return np.mean(mse_values), np.std(mse_values)


def calculate_mse_score_from_path_pix2pixHD(path, return_std=False):
    print('Calculating MSE given path %s...' % (path))

    mse_values = []

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    for index, i in tqdm(enumerate(sorted(os.listdir(path)))):
        if index % 3 == 0:
            gt = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
            gt = transform(gt).unsqueeze(0)
        elif index % 3 == 2:
            synthesized = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
            synthesized = transform(synthesized).unsqueeze(0)

            mse_values.append(torch.mean((gt - synthesized) ** 2))

    if not return_std:
        return np.mean(mse_values)
    else:
        return np.mean(mse_values), np.std(mse_values)


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return np.nanmean(ssim_map.cpu())
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def calculate_ssim_score_from_path(path1, path2, return_std=False):
    print('Calculating SSIM given paths %s and %s...' % (path1, path2))

    ssim_values = []

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    for _, i in tqdm(enumerate(sorted(os.listdir(path1)))):
        img1 = Image.open(os.path.join(path1, i)).convert('RGB').convert('L')
        img1 = transform(img1).unsqueeze(0)

        img2 = Image.open(os.path.join(path2, i)).convert('RGB').convert('L')
        img2 = transform(img2).unsqueeze(0)

        ssim_values.append(ssim(img1, img2))

    if not return_std:
        return np.mean(ssim_values)
    else:
        return np.mean(ssim_values), np.std(ssim_values)


def calculate_ssim_score_from_path_pix2pixHD(path, return_std=False):
    print('Calculating SSIM given paths %s...' % (path))

    ssim_values = []

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    for index, i in tqdm(enumerate(sorted(os.listdir(path)))):
        if index % 3 == 0:
            gt = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
            gt = transform(gt).unsqueeze(0)
        elif index % 3 == 2:
            synthesized = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
            synthesized = transform(synthesized).unsqueeze(0)

            ssim_values.append(ssim(gt, synthesized))

    if not return_std:
        return np.mean(ssim_values)
    else:
        return np.mean(ssim_values), np.std(ssim_values)


@torch.no_grad()
def calculate_lpips_2_images(img_paths):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist = LPIPS().eval().to(device)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    img1 = Image.open(img_paths[0]).convert('RGB').convert('L')
    img1 = transform(img1).unsqueeze(0)
    img1 = img1.to(device)

    img2 = Image.open(img_paths[1]).convert('RGB').convert('L')
    img2 = transform(img2).unsqueeze(0)
    img2 = img2.to(device)

    lpips_value = dist(img1, img2)

    return lpips_value.item()


@torch.no_grad()
def calculate_lpips_from_path(path1, path2, return_list=False, return_std=False):
    print('Calculating LPIPS given paths %s and %s...' % (path1, path2))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist = LPIPS().eval().to(device)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    images_1 = {}
    images_2 = {}
    lpips_values = {}

    for _, i in tqdm(enumerate(sorted(os.listdir(path1)))):
        img1 = Image.open(os.path.join(path1, i)).convert('RGB').convert('L')
        img1 = transform(img1).unsqueeze(0)
        images_1[i] = img1

        img2 = Image.open(os.path.join(path2, i)).convert('RGB').convert('L')
        img2 = transform(img2).unsqueeze(0)
        images_2[i] = img2

    for k, _ in images_1.items():
        img1 = images_1[k].to(device)
        img2 = images_2[k].to(device)
        lpips_values[k] = dist(img1, img2).item()

    lpips_values_list = list(lpips_values.values())

    if not return_list:
        if not return_std:
            return lpips_values, np.mean(lpips_values_list)
        else:
            return lpips_values, np.mean(lpips_values_list), np.std(lpips_values_list)
    else:
        if not return_std:
            return lpips_values, np.mean(lpips_values_list)
        else:
            return lpips_values, np.mean(lpips_values_list), np.std(lpips_values_list)


def calculate_lpips_from_path_pix2pixHD(path, return_list=False, return_std=False):
    print('Calculating LPIPS given path %s ...' % (path))

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dist = LPIPS().eval().to(device)

    transform = transforms.Compose([
        transforms.Resize([256, 256]),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ])

    images_1 = {}
    images_2 = {}
    lpips_values = {}

    for index, i in tqdm(enumerate(sorted(os.listdir(path)))):
        if index % 3 == 0:
            img = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
            img = transform(img).unsqueeze(0)
            images_1[i] = img
        elif index % 3 == 2:
            img = Image.open(os.path.join(path, i)).convert('RGB').convert('L')
            img = transform(img).unsqueeze(0)
            images_2[i] = img

    for k, _ in images_1.items():
        name = k.split('_')[0]
        name_generated = name + '_synthesized_image.jpg'
        img1 = images_1[k].to(device)
        img2 = images_2[name_generated].to(device)
        lpips_values[name + '.jpg'] = dist(img1, img2).item()

    lpips_values_list = list(lpips_values.values())

    if not return_list:
        if not return_std:
            return lpips_values, np.mean(lpips_values_list)
        else:
            return lpips_values, np.mean(lpips_values_list), np.std(lpips_values_list)
    else:
        if not return_std:
            return lpips_values, np.mean(lpips_values_list)
        else:
            return lpips_values, np.mean(lpips_values_list), np.std(lpips_values_list)


class LPIPS(nn.Module):
    def __init__(self):
        super().__init__()
        self.alexnet = AlexNet()
        self.lpips_weights = nn.ModuleList()
        for channels in self.alexnet.channels:
            self.lpips_weights.append(Conv1x1(channels, 1))
        self._load_lpips_weights()

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base_model = UNet3D()
        weight_dir = './models/pretrained_weights/Genesis_Chest_CT.pt'
        checkpoint = torch.load(weight_dir)
        state_dict = checkpoint['state_dict']
        unParalled_state_dict = {}
        for key in state_dict.keys():
            unParalled_state_dict[key.replace("module.", "")] = state_dict[key]
        base_model.load_state_dict(unParalled_state_dict)

        self.genesis = TargetNet(base_model)
        self.genesis = self.genesis.to(device)

    def _load_lpips_weights(self):
        own_state_dict = self.state_dict()
        if torch.cuda.is_available():
            state_dict = torch.load('./models/lpips_weights.ckpt')
        else:
            state_dict = torch.load('./models/lpips_weights.ckpt',
                                    map_location=torch.device('cpu'))
        for name, param in state_dict.items():
            if name in own_state_dict:
                own_state_dict[name].copy_(param)

    def forward(self, x, y):
        x = torch.cat((x, x, x, x, x, x, x, x, x, x, x, x, x, x, x, x), dim=1)
        y = torch.cat((y, y, y, y, y, y, y, y, y, y, y, y, y, y, y, y), dim=1)

        y = y.unsqueeze(0)
        x = x.unsqueeze(0)

        actv_x = self.genesis(x)
        actv_y = self.genesis(y)

        lpips_value = torch.mean((actv_x - actv_y) ** 2)

        return lpips_value


# prepare the 3D model
class TargetNet(nn.Module):
    def __init__(self, base_model, n_class=1):
        super(TargetNet, self).__init__()

        self.base_model = base_model
        self.dense_1 = nn.Linear(512, 1024, bias=True)
        self.dense_2 = nn.Linear(1024, n_class, bias=True)

    def forward(self, x):
        self.base_model(x)
        self.base_out = self.base_model.out512
        # This global average polling is for shape (N, C, H, W) not for (N, H, W, C)
        # where N = batch_size, C = channels, H = height, and W = Width
        self.out_glb_avg_pool = F.avg_pool3d(self.base_out, kernel_size=self.base_out.size()[2:]).view(
            self.base_out.size()[0], -1)

        return self.out_glb_avg_pool


class AlexNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = models.alexnet(pretrained=True).features
        self.channels = []
        for layer in self.layers:
            if isinstance(layer, nn.Conv2d):
                self.channels.append(layer.out_channels)

    def forward(self, x):
        fmaps = []
        for layer in self.layers:
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                fmaps.append(x)
        return fmaps


class Conv1x1(nn.Module):
    def __init__(self, in_channels, out_channels=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Dropout(0.5),
            nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False))

    def forward(self, x):
        return self.main(x)
