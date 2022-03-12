import os
import sys
import time
import numpy as np
import torch
# torch.autograd.set_detect_anomaly(True)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
import torchvision
import torchvision.transforms as transforms

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

from tqdm.auto import tqdm
import pytorch_lightning as pl

# pip install einops
from einops.layers.torch import Rearrange

import matplotlib.pyplot as plt

batch_size = 1024
n_epochs = 100
max_lr = .01
n_runs = 1000
gpus = 1

# Allow for a version to be provided at the command line, as in
# $ python3 demo_fashion_mnist_lightning.py v15
version = sys.argv[1] if len(sys.argv) > 1 else "test"

class FashionMNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir: str = "./data/FashionMNIST", batch_size: int = 32):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size        

    def setup(self, stage = None):

        self.mnist_train = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=True,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )
        self.mnist_val = torchvision.datasets.FashionMNIST(
            root=self.data_dir,
            train=False,
            download=True,
            transform=transforms.Compose([transforms.ToTensor()])
        )

    def train_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_train, batch_size=self.batch_size, shuffle=True)

    def val_dataloader(self):
        return torch.utils.data.DataLoader(self.mnist_val, batch_size=self.batch_size)

class SCSLNet(pl.LightningModule):
    def __init__(self, max_lr, steps_per_epoch, n_epochs):
        super().__init__()
        self.max_lr = max_lr
        self.steps_per_epoch = steps_per_epoch
        self.n_epochs = n_epochs

        # keep track of the loss and accuracy for each step in an epoch and each epoch
        self.epoch = -1
        self.step_logs = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []} 
        self.epoch_logs = {"train_loss": [], "valid_loss": [], "train_acc": [], "valid_acc": []}

        self.net = nn.Sequential(
            SharpenedCosineSimilarity(in_channels=1, out_channels=8, kernel_size=3, padding=0),
            MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True),
            SharpenedCosineSimilarity(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True),
            SharpenedCosineSimilarity(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True),
            # Flatten Channels, X, and Y dimension into the Channel dimension
            Rearrange("b c x y -> b (c x y)"),
            nn.Linear(in_features=32*4*4, out_features=10)
        )

    def forward(self, t):
        # data is between 0 and 1, no normalization applied.
        return self.net(t)

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train_acc", "train_loss")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "valid_acc", "valid_loss")

    def _step(self, batch, log_acc_name, log_loss_name):
        x, y = batch
        p = self(x)

        loss = F.cross_entropy(p, y)

        self.step_logs[log_acc_name].append(p.argmax(dim=1).eq(y).sum().item() / y.shape[0])
        self.step_logs[log_loss_name].append(float(loss))

        return loss

    def on_validation_epoch_end(self):
        # empty print for newline   
        
        if self.epoch >= 0:
            tqdm.write(f"Epoch: {self.epoch}")
            tqdm.write(f"train loss: {np.mean(self.step_logs['train_loss']):06.2f}")
            tqdm.write(f"valid loss: {np.mean(self.step_logs['valid_loss']):06.2f}")
            tqdm.write(f"train  acc: {np.mean(self.step_logs['train_acc'])*100:06.2f} %")
            tqdm.write(f"valid  acc: {np.mean(self.step_logs['valid_acc'])*100:06.2f} %")

        for key in self.step_logs.keys():
            self.epoch_logs[key].append(np.mean(self.step_logs[key]))
            self.step_logs[key] = []

        self.epoch += 1

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.max_lr)
        scheduler = OneCycleLR(
            optimizer,
            max_lr=self.max_lr,
            steps_per_epoch=self.steps_per_epoch,
            epochs=self.n_epochs
        )
        return [optimizer], [scheduler]


mnist = FashionMNISTDataModule(batch_size=batch_size)
mnist.setup()

# used for cosine annealing
steps_per_epoch = len(mnist.mnist_train) // batch_size
model = SCSLNet(max_lr, steps_per_epoch, n_epochs)

trainer = pl.Trainer(gpus=gpus, max_epochs=n_epochs)
trainer.fit(model, mnist)

# save matplotlib figure of training and validation loss
fig, ax = plt.subplots(1, 2, figsize=(12, 6))
ax[0].plot(model.epoch_logs["train_loss"], label="train")
ax[0].plot(model.epoch_logs["valid_loss"], label="valid")
ax[0].set_xlabel("epoch")
ax[0].set_ylabel("loss")
ax[0].legend()
ax[1].plot(model.epoch_logs["train_acc"], label="train")
ax[1].plot(model.epoch_logs["valid_acc"], label="valid")
ax[1].set_xlabel("epoch")
ax[1].set_ylabel("accuracy")
ax[1].legend()
fig.savefig(f"performance_{version}.png")