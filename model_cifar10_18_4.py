# -*- coding: utf-8 -*-
"""
Compare with the CIFAR-10 Papers With Code pareto frontier here
https://paperswithcode.com/sota/image-classification-on-cifar-10?dimension=PARAMS
"""
import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

batch_size = 128
n_epochs = 100
max_lr = .01
n_runs = 1000

# Allow for a version to be provided at the command line, as in
if len(sys.argv) > 1:
    version = sys.argv[1]
else:
    version = "test"

# Lay out the desitinations for all the results.
accuracy_results_path = os.path.join(f"results", f"accuracy_{version}.npy")
accuracy_history_path = os.path.join(
    "results", f"accuracy_history_{version}.npy")
loss_results_path = os.path.join("results", f"loss_{version}.npy")
os.makedirs("results", exist_ok=True)

training_set = CIFAR10(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        # transforms.ColorJitter(
        #      brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
testing_set = CIFAR10(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))

training_loader = DataLoader(
    training_set,
    batch_size=batch_size,
    shuffle=True)
testing_loader = DataLoader(
    testing_set,
    batch_size=batch_size,
    shuffle=False)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.scs1 = SharpenedCosineSimilarity(
            in_channels=3, out_channels=24, kernel_size=3, padding=0)
        self.pool1 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scs2 = SharpenedCosineSimilarity(
            in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.pool2 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scs3 = SharpenedCosineSimilarity(
            in_channels=48, out_channels=96, kernel_size=3, padding=1)
        self.pool3 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.out = nn.Linear(in_features=96*4*4, out_features=10)

    def forward(self, t):
        t = self.scs1(t)
        t = self.pool1(t)

        t = self.scs2(t)
        t = self.pool2(t)

        t = self.scs3(t)
        t = self.pool3(t)

        t = t.reshape(-1, 96*4*4)
        t = self.out(t)

        return t


# Restore any previously generated results.
try:
    accuracy_results = np.load(accuracy_results_path).tolist()
    accuracy_histories = np.load(accuracy_history_path).tolist()
    loss_results = np.load(loss_results_path).tolist()
except Exception:
    loss_results = []
    accuracy_results = []
    accuracy_histories = []

for i_run in range(n_runs):
    network = Network()
    optimizer = optim.Adam(network.parameters(), lr=max_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=len(training_loader),
        epochs=n_epochs)

    epoch_accuracy_history = []
    for i_epoch in range(n_epochs):

        epoch_start_time = time.time()
        epoch_training_loss = 0
        epoch_testing_loss = 0
        epoch_training_num_correct = 0
        epoch_testing_num_correct = 0

        for batch in training_loader:

            images = batch[0]
            labels = batch[1]
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_training_loss += loss.item() * training_loader.batch_size
            epoch_training_num_correct += (
                preds.argmax(dim=1).eq(labels).sum().item())
            epoch_duration = time.time() - epoch_start_time

        epoch_duration = time.time() - epoch_start_time
        training_loss = epoch_training_loss / len(training_loader.dataset)
        training_accuracy = (
            epoch_training_num_correct / len(training_loader.dataset))

        # At the end of each epoch run the testing data through an
        # evaluation pass to see how the model is doing.
        # Specify no_grad() to prevent a nasty out-of-memory condition.
        with torch.no_grad():
            test_preds = torch.tensor([])
            for batch in testing_loader:
                images, labels = batch
                preds = network(images)
                loss = F.cross_entropy(preds, labels)
                test_preds = torch.cat((test_preds, preds), dim = 0)

                epoch_testing_loss += loss.item() * testing_loader.batch_size
                epoch_testing_num_correct += (
                    preds.argmax(dim=1).eq(labels).sum().item())

            testing_loss = epoch_testing_loss / len(testing_loader.dataset)
            testing_accuracy = (
                epoch_testing_num_correct / len(testing_loader.dataset))
            epoch_accuracy_history.append(testing_accuracy)

        print(
            f"run: {i_run}   "
            f"epoch: {i_epoch}   "
            f"duration: {epoch_duration:.04}   "
            f"training loss: {training_loss:.04}   "
            f"testing loss: {testing_loss:.04}   "
            f"training accuracy: {100 * training_accuracy:.04}%   "
            f"testing accuracy: {100 * testing_accuracy:.04}%"
        )

    accuracy_histories.append(epoch_accuracy_history)
    accuracy_results.append(testing_accuracy)
    loss_results.append(testing_loss)

    np.save(accuracy_history_path, np.array(accuracy_histories))
    np.save(accuracy_results_path, np.array(accuracy_results))
    np.save(loss_results_path, np.array(loss_results))
