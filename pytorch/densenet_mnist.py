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
import torchvision.transforms as transforms
from tqdm import tqdm

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpCosSim2d

from densenet import DenseNet

batch_size = 100
max_lr = .03
n_classes = 10
n_epochs = 100
n_runs = 10
n_input_channels = 1
n_kernels_in = 27
n_kernels = 18
kernel_size = 3

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

# Use standard FashionMNIST dataset
training_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(28, padding=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]))
testing_set = torchvision.datasets.FashionMNIST(
    root='./data/FashionMNIST',
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
    network = DenseNet(num_init_features = n_input_channels, num_classes = n_classes, n_channels=1)

    for p in network.parameters():
        if p.requires_grad:
            print(p.numel())

    n_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Model has {n_params:_} trainable parameters.")

    optimizer = optim.Adam(network.parameters(), lr=max_lr)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        steps_per_epoch=steps_per_epoch,
        epochs=n_epochs)

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
                scheduler.step()

                epoch_training_loss += loss.item() * training_loader.batch_size
                epoch_training_num_correct += (
                    preds.argmax(dim=1).eq(labels).sum().item())

                tqdm_training_loader.set_description(
                    f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                    f'Epoch: {i_epoch + 1}/{n_epochs}, '
                    f'Run: {i_run + 1}/{n_runs}'
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