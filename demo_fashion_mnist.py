# -*- coding: utf-8 -*-
"""
Based on and copy/pasted heavily from
PyTorch tutorial
https://colab.research.google.com/drive/1YWzAjpAnLI23irBQtLvDTYT1A94uCloM
by Michael Li
https://www.linkedin.com/in/michael-li-dfw/
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
import torchvision
import torchvision.transforms as transforms

# TODO: Debug MaxAbsPool and swap in.
# from absolute_pooling_pt import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

batch_size = 1024
n_epochs = 5
max_lr = .01
n_runs = 1000

# Allow for a version to be provided at the command line, as in
# $ python3 demo_fashion_mnist.py v15
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

# Use standard FashionMNIST dataset
train_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)
test_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()])
)


class Network(nn.Module):
  def __init__(self):
    super().__init__()

    self.scs1 = SharpenedCosineSimilarity(
        in_channels=1, out_channels=8, kernel_size=3, padding=0)
    self.scs2 = SharpenedCosineSimilarity(
        in_channels=8, out_channels=16, kernel_size=3, padding=1)
    self.scs3 = SharpenedCosineSimilarity(
        in_channels=16, out_channels=32, kernel_size=3, padding=1)
    self.out = nn.Linear(in_features=32*4*4, out_features=10)

  def forward(self, t):
    t = self.scs1(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2, ceil_mode=True)
    # t = MaxAbsPool2d(t, kernel_size=2, stride=2)

    t = self.scs2(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2, ceil_mode=True)
    # t = MaxAbsPool2d(t, kernel_size=2, stride=2)

    t = self.scs3(t)
    t = F.relu(t)
    t = F.max_pool2d(t, kernel_size=2, stride=2, ceil_mode=True)
    # t = MaxAbsPool2d(t, kernel_size=2, stride=2)

    t = t.reshape(-1, 32*4*4)
    t = self.out(t)

    return t


training_loader = torch.utils.data.DataLoader(
    train_set, batch_size = batch_size)
testing_loader = torch.utils.data.DataLoader(
    test_set, batch_size = batch_size)

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
        test_preds = torch.tensor([])
        for batch in testing_loader:
            images, labels = batch
            preds = network(images)
            test_preds = torch.cat((test_preds, preds), dim = 0)

            epoch_testing_loss += loss.item() * testing_loader.batch_size
            epoch_testing_num_correct += (
                preds.argmax(dim=1).eq(labels).sum().item())

        testing_loss = epoch_testing_loss / len(testing_loader.dataset)
        testing_accuracy = (
            epoch_testing_num_correct / len(testing_loader.dataset))
        epoch_accuracy_history.append(testing_accuracy)

        print(
            f"run:{i_run}  "
            f"epoch:{i_epoch}  "
            f"duration:{epoch_duration:.04}  "
            f"training loss:{training_loss:.04}  "
            f"testing loss:{testing_loss:.04}  "
            f"training accuracy:{training_accuracy:.04}  "
            f"testing accuracy:{testing_accuracy:.04}"
        )

    accuracy_histories.append(epoch_accuracy_history)
    accuracy_results.append(testing_accuracy)
    loss_results.append(testing_loss)

    np.save(accuracy_history_path, np.array(accuracy_histories))
    np.save(accuracy_results_path, np.array(accuracy_results))
    np.save(loss_results_path, np.array(loss_results))
