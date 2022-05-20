import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpCosSim2d

batch_size = 100
learning_rate = .01
n_epochs = 3

training_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))
testing_set = torchvision.datasets.MNIST(
    root='./data/MNIST',
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


network = nn.Sequential(
    SharpCosSim2d(in_channels=1, out_channels=10, kernel_size=3, padding=1),
    MaxAbsPool2d(kernel_size=2, stride=2),
    SharpCosSim2d(in_channels=10, out_channels=20, kernel_size=3, groups=10),
    SharpCosSim2d(in_channels=20, out_channels=8, kernel_size=1),
    MaxAbsPool2d(kernel_size=2, stride=2),
    SharpCosSim2d(
        in_channels=8,
        out_channels=32,
        kernel_size=3,
        groups=8,
        shared_weights=False),
    SharpCosSim2d(in_channels=32, out_channels=10, kernel_size=1),
    MaxAbsPool2d(kernel_size=4, stride=4),
    nn.Flatten(start_dim=1),
    nn.Linear(in_features=10, out_features=10)
)

for p in network.parameters():
    if p.requires_grad:
        print(p.numel())

n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
print(f"Model has {n_params:_} trainable parameters.")

optimizer = optim.Adam(
    network.parameters(),
    lr=learning_rate,
    betas=(0.9, 0.999),
    eps=1e-08,
)

epoch_accuracy_history = []
for i_epoch in range(n_epochs):

    epoch_start_time = time.time()
    epoch_training_loss = 0
    epoch_testing_loss = 0
    epoch_training_num_correct = 0
    epoch_testing_num_correct = 0

    with tqdm(enumerate(training_loader)) as tqdm_training_loader:
        for batch_idx, batch in tqdm_training_loader:

            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_training_loss += loss.item() * training_loader.batch_size
            epoch_training_num_correct += (
                preds.argmax(dim=1).eq(labels).sum().item())

            steps_per_epoch = len(training_loader)
            tqdm_training_loader.set_description(
                f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                f'Epoch: {i_epoch + 1}/{n_epochs}, '
            )

    epoch_duration = time.time() - epoch_start_time
    training_loss = epoch_training_loss / len(training_loader.dataset)
    training_accuracy = (
        epoch_training_num_correct / len(training_loader.dataset))

    # At the end of each epoch run the testing data through an
    # evaluation pass to see how the model is doing.
    # Specify no_grad() to prevent a nasty out-of-memory condition.
    with torch.no_grad():
        for batch in testing_loader:
            images, labels = batch
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            epoch_testing_loss += loss.item() * testing_loader.batch_size
            epoch_testing_num_correct += (
                preds.argmax(dim=1).eq(labels).sum().item())

        testing_loss = epoch_testing_loss / len(testing_loader.dataset)
        testing_accuracy = (
            epoch_testing_num_correct / len(testing_loader.dataset))
        epoch_accuracy_history.append(testing_accuracy)

    print(
        f"epoch: {i_epoch}   "
        f"training loss: {training_loss:.04}   "
        f"testing loss: {testing_loss:.04}   "
        f"training accuracy: {100 * training_accuracy:.04}%   "
        f"testing accuracy: {100 * testing_accuracy:.04}%"
    )

print()
print("Should have 1,233 parameters and final results not wildly different from")
print("epoch: 2   training loss: 0.239   testing loss: 0.2154   training accuracy: 93.24%   testing accuracy: 93.84%")
