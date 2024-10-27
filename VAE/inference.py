import os

import numpy as np
import torch
import yaml
import argparse
from models import *
from pytorch_lightning.utilities.seed import seed_everything

from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae_inference.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)


# For reproducibility
seed_everything(config['exp_params']['manual_seed'], True)

model = vae_models[config['model_params']['name']](**config['model_params'])
model_path = config['model_params']['checkpoint_path']
checkpoint = torch.load(model_path)
state_dict = checkpoint['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    #print(k)
    new_key = k.replace('model.', '')  # Adjust this if your keys have a different prefix
    #print(new_key)
    new_state_dict[new_key] = v
#print("-------------------------------------")
#for k, _ in new_state_dict.items():
    #print(k)
model.load_state_dict(new_state_dict)
model.eval()

path = config['data_params']['test_data_path']
img_path = path.split('+')[0]
mask_path = path.split('+')[1]
img_name = sorted(os.listdir(img_path))[0]
mask_name = sorted(os.listdir(mask_path))[0]
img = Image.open(os.path.join(img_path, img_name)).convert('RGB')
mask = Image.open(os.path.join(mask_path, mask_name)).convert('RGB').convert('1')
tfs = transforms.Compose([
            transforms.Resize((config['model_params']['patch_size'], config['model_params']['patch_size'])),
            transforms.ToTensor(),
        ])

with torch.no_grad():
    img = tfs(img)
    print(f'1: {img.shape}')
    mask = tfs(mask)
    mask = mask.repeat(3, 1, 1)

    img = img[None, :, :, :]
    print(f'2: {img.shape}')
    mask = mask[None, :, :, :]
    img = img * (1. - mask)

    output = model(img, mask=mask)[0]
    [mu, log_var] = model.encode(img)
    latent = model.reparameterize(mu, log_var)
    print(f'Latent Size: {latent.shape}')
    # save_image(output, "output.png")
    # save_image(latent, "latent.png")

print("Finish!!!!")
