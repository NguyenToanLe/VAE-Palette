import os
import math
import torch
from torch import optim
from models.base import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.train_log = ''
        self.val_log = ''
        self.train_log_file_name = self.params['save_dir'] + "train_log.txt"
        self.val_log_file_name = self.params['save_dir'] + "val_log.txt"
        with open(self.train_log_file_name, 'w') as f:
            f.write(self.train_log)
        with open(self.val_log_file_name, 'w') as f:
            f.write(self.val_log)
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, mask: Tensor = None, **kwargs) -> Tensor:
        return self.model(input, mask, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, mask = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, mask=mask)
        train_loss = self.model.loss_function(*results,
                                              M_N=self.params['kld_weight'],  # al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx=batch_idx)

        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        self.train_log += f'Epoch {self.current_epoch}: '
        for key, val in train_loss.items():
            self.train_log += f'{key} = {val.item()}, '
        self.train_log += f'\n'

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx=0):
        real_img, mask = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, mask=mask)
        val_loss = self.model.loss_function(*results,
                                            M_N=self.params['kld_weight'],#1.0,  # real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx=optimizer_idx,
                                            batch_idx=batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        self.val_log += f'Epoch {self.current_epoch}: '
        for key, val in val_loss.items():
            self.val_log += f'{key} = {val.item()}, '
        self.val_log += f'\n'

    def on_validation_end(self) -> None:
        self.sample_images()
        with open(self.train_log_file_name, "a") as f:
            f.write(self.train_log)
        self.train_log = ''
        with open(self.val_log_file_name, "a") as f:
            f.write(self.val_log)
        self.val_log = ''

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_mask = next(iter(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_mask = test_mask.to(self.curr_device)

        #         test_input, test_label = batch
        recons = self.model.generate(test_input, mask=test_mask)
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir,
                                       "Reconstructions",
                                       f"recons_{self.logger.name}_Epoch_{self.current_epoch}.png"),
                          normalize=True,
                          nrow=12)
        # Logging images to Tensor Board
        self.logger.experiment.add_images("Input", test_input, self.current_epoch)
        self.logger.experiment.add_images("Mask", test_mask, self.current_epoch)
        self.logger.experiment.add_images("Recons", recons.data, self.current_epoch)
        self.logger.experiment.add_image("First Input of Batch", test_input[0], self.current_epoch)
        self.logger.experiment.add_image("First Recons of Batch", recons.data[0], self.current_epoch)
        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        mask=test_mask)
            vutils.save_image(samples.cpu().data,
                              os.path.join(self.logger.log_dir,
                                           "Samples",
                                           f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                              normalize=True,
                              nrow=12)
        except Warning:
            pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model, self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma=self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma=self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims