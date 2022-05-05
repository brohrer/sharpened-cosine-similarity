from multiprocessing.dummy import active_children
import os
import random
import sys
import time
from datetime import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR100
import torchvision.transforms as transforms
from tqdm import tqdm

from absolute_pooling import MaxAbsPool2d
from sharpened_cosine_similarity import SharpenedCosineSimilarity

from densenet import DenseNet
from resnet import *
from demo_network import DemoNetwork
import argparse

import errno

########## Hyper Parameters ##########

batch_size = 64
n_epochs = 2
max_lr = .05

########## Setup ##########

def createdirs(path):
    if not os.path.exists(os.path.dirname(path)):
        try:
            os.makedirs(os.path.dirname(path))
        except OSError as exc:
            if exc.errno != errno.EEXIST:
                raise

def log(str, f):
    print(str, file=sys.stderr)
    print(str, file=f)

parser = argparse.ArgumentParser(description='SCS Train')
parser.add_argument('--model', default='', help='model type')
args = parser.parse_args()

def set_all_seeds(seed):
  random.seed(seed)
  os.environ['PYTHONHASHSEED'] = str(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.cuda.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

set_all_seeds(621)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

########## Model Definitions ##########

def gen_densenet_model():
    return DenseNet(sharpened_cosine_similarity=True, activation = True, normalization =True)

def gen_densenet_scs():
    return DenseNet(sharpened_cosine_similarity=False, activation = True, normalization = True)

def gen_densenet_no_act():
    return DenseNet(sharpened_cosine_similarity=True, activation = False, normalization = False)

def gen_densenet_no_norm():
    return DenseNet(sharpened_cosine_similarity=True, activation = True, normalization = False)

def gen_demo_network():
    return DemoNetwork()

def gen_resnet_model():
    return ResNet(BasicBlock, [2, 2, 2, 2])

def gen_resnet_scs():
    return ResNet(BasicBlock, [2, 2, 2, 2], scs = True)

network_gen = {
    "densenet": gen_densenet_model,
    "densenet_scs": gen_densenet_scs,
    "densenet_no_act": gen_densenet_no_act,
    "densenet_no_norm": gen_densenet_no_norm,
    "demo": gen_demo_network,
    "resnet": gen_resnet_model,
    "resnet_scs": gen_resnet_scs
}

model_gen = network_gen.get(args.model)

########## Data ##########

training_set = CIFAR100(
    root=os.path.join('.', 'data', 'CIFAR10'),
    train=True,
    download=True,
    transform=transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
    ]))
testing_set = CIFAR100(
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

########## Training Loop ##########

network = model_gen().to(device)
print(f"Training: {args.model}")

optimizer = optim.Adam(network.parameters(), lr=max_lr)

path = 'logs/' + args.model + '/' + datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
createdirs(path)
log_file = open(path + '.log', 'w+')


steps_per_epoch = len(training_loader)

scheduler = OneCycleLR(
    optimizer,
    max_lr=max_lr,
    steps_per_epoch=steps_per_epoch,
    epochs=n_epochs)

for i_epoch in range(n_epochs):
    epoch_start_time = time.time()
    epoch_training_loss = 0
    epoch_testing_loss = 0
    epoch_training_num_correct = 0
    epoch_testing_num_correct = 0

    with tqdm(enumerate(training_loader), total=len(training_loader)) as tqdm_training_loader:
        for batch_idx, batch in tqdm_training_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            epoch_training_loss += loss.item() * training_loader.batch_size
            epoch_training_num_correct += (preds.argmax(dim=1).eq(labels).sum().item())

            tqdm_training_loader.set_description(
                f'Step: {batch_idx + 1}/{steps_per_epoch}, '
                f'Epoch: {i_epoch + 1}/{n_epochs}, '
            )

    epoch_duration = time.time() - epoch_start_time
    training_loss = epoch_training_loss / len(training_loader.dataset)
    training_accuracy = (epoch_training_num_correct / len(training_loader.dataset))

    with torch.no_grad():
        for batch in testing_loader:
            images, labels = batch
            images = images.to(device)
            labels = labels.to(device)
            preds = network(images)
            loss = F.cross_entropy(preds, labels)

            epoch_testing_loss += loss.item() * testing_loader.batch_size
            epoch_testing_num_correct += (preds.argmax(dim=1).eq(labels).sum().item())

        testing_loss = epoch_testing_loss / len(testing_loader.dataset)
        testing_accuracy = (epoch_testing_num_correct / len(testing_loader.dataset))

    log('Epoch {} Train Loss {} Train Accuracy {} Test Loss {} Test Accuracy {}'.format(
        i_epoch, training_loss, training_accuracy, testing_loss, testing_accuracy), log_file)

torch.save(network.state_dict(), path + '.pt')
log_file.close()