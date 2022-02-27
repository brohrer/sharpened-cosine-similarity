import os
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CIFAR10
import torchvision.transforms as transforms
from tqdm import tqdm

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

batch_size = 64
n_epochs = 200
n_runs = 1000

# The initial learning rate to kick off the exponential decay
starting_lr = 1
# Half-life of the learning rate decay, in epochs
lr_halflife = 50

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


training_set = CIFAR10(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))
testing_set = CIFAR10(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=False,
    download=True,
    transform=transforms.Compose([transforms.ToTensor()]))

training_loader = DataLoader(
    training_set,
    batch_size=batch_size,
    shuffle=True,
    ) # num_workers=4)
testing_loader = DataLoader(
    testing_set,
    batch_size=batch_size,
    shuffle=False,
    ) # num_workers=4)


class Network(nn.Module):
    def __init__(self):
        super().__init__()

        self.scs1 = SharpenedCosineSimilarity(
            in_channels=3, out_channels=16, kernel_size=3, padding=0)
        self.pool1 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scs2 = SharpenedCosineSimilarity(
            in_channels=16, out_channels=24, kernel_size=3, padding=1)
        self.pool2 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.scs3 = SharpenedCosineSimilarity(
            in_channels=24, out_channels=48, kernel_size=3, padding=1)
        self.pool3 = MaxAbsPool2d(kernel_size=2, stride=2, ceil_mode=True)
        self.out = nn.Linear(in_features=48*4*4, out_features=10)

    def forward(self, t):
        t = self.scs1(t)
        t = self.pool1(t)

        t = self.scs2(t)
        t = self.pool2(t)

        t = self.scs3(t)
        t = self.pool3(t)

        t = t.reshape(-1, 48*4*4)
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

steps_per_epoch = len(training_loader)

for i_run in range(n_runs):
    network = Network()
    optimizer = optim.SGD(network.parameters(), lr=starting_lr)
    gamma = .5 ** (1 / lr_halflife)
    scheduler = ExponentialLR(optimizer, gamma)

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

                tqdm_training_loader.set_description(
                    f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                    f'Epoch: {i_epoch + 1}/{n_epochs}, '
                    f'Run: {i_run + 1}/{n_runs}'
                )

        scheduler.step()

        epoch_duration = time.time() - epoch_start_time
        training_loss = epoch_training_loss / len(training_loader.dataset)
        training_accuracy = (
            epoch_training_num_correct / len(training_loader.dataset))

        # At the end of each epoch run the testing data through an
        # evaluation pass to see how the model is doing.
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
            f"run: {i_run}   "
            f"epoch: {i_epoch}   "
            f"duration: {epoch_duration:.04}   "
            f"learning rate: {scheduler.get_last_lr()[0]:.04}   "
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
