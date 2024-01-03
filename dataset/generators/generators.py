# PFLlib: Personalized Federated Learning Algorithm Library
# Copyright (C) 2021  Jianqing Zhang

# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2 of the License, or
# (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License along
# with this program; if not, write to the Free Software Foundation, Inc.,
# 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.

import numpy as np
import os
import sys
import random
import torch
import torchvision
import torchvision.transforms as transforms
from utils.dataset_utils import check, separate_data, split_data, save_file


random.seed(1)
np.random.seed(1)

def generate_mnist(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_size, 
                   alpha, batch_size, least_samples):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, alpha, batch_size, niid, balance, partition):
        return

    # FIX HTTP Error 403: Forbidden
    from six.moves import urllib
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0')]
    urllib.request.install_opener(opener)

    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get MNIST data

    
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    trainset = torchvision.datasets.MNIST(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.MNIST(
        root=root_dir_path, train=False, download=True, transform=transform)

    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha, 
                                    least_samples, niid, balance, partition, class_per_client)
    train_data, test_data = split_data(X, y, train_size)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, niid, balance, partition)

def generate_femnist(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_size, 
                   alpha, batch_size, least_samples):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, alpha, batch_size, niid, balance, partition):
        return

    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get FashionMNIST data
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])

    trainset = torchvision.datasets.FashionMNIST(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(
        root=root_dir_path, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha, 
                                    least_samples, niid, balance, partition, class_per_client)
    train_data, test_data = split_data(X, y, train_size)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, niid, balance, partition)
    
def generate_cifar10(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_size, 
                   alpha, batch_size, least_samples):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, alpha, batch_size, niid, balance, partition):
        return
    
    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get Cifar10 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(
        root=root_dir_path, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    least_samples, niid, balance, partition, class_per_client)
    train_data, test_data = split_data(X, y, train_size)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, niid, balance, partition)

def generate_cifar100(dir_path, num_clients, num_classes, niid, balance, partition, class_per_client, train_size, 
                   alpha, batch_size, least_samples):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        
    # Setup directory for train/test data
    config_path = dir_path + "/config.json"
    train_path = dir_path + "/train/"
    test_path = dir_path + "/test/"

    if check(config_path, train_path, test_path, num_clients, num_classes, alpha, batch_size, niid, balance, partition):
        return
    
    root_dir_path = dir_path.split("/")[0] + "/rawdata"

    # Get Cifar100 data
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR100(
        root=root_dir_path, train=True, download=True, transform=transform)
    testset = torchvision.datasets.CIFAR100(
        root=root_dir_path, train=False, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=len(trainset.data), shuffle=False)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=len(testset.data), shuffle=False)

    for _, train_data in enumerate(trainloader, 0):
        trainset.data, trainset.targets = train_data
    for _, test_data in enumerate(testloader, 0):
        testset.data, testset.targets = test_data

    dataset_image = []
    dataset_label = []

    dataset_image.extend(trainset.data.cpu().detach().numpy())
    dataset_image.extend(testset.data.cpu().detach().numpy())
    dataset_label.extend(trainset.targets.cpu().detach().numpy())
    dataset_label.extend(testset.targets.cpu().detach().numpy())
    dataset_image = np.array(dataset_image)
    dataset_label = np.array(dataset_label)

    # dataset = []
    # for i in range(num_classes):
    #     idx = dataset_label == i
    #     dataset.append(dataset_image[idx])

    X, y, statistic = separate_data((dataset_image, dataset_label), num_clients, num_classes, alpha,
                                    least_samples, niid, balance, partition, class_per_client)
    train_data, test_data = split_data(X, y, train_size)
    save_file(config_path, train_path, test_path, train_data, test_data, num_clients, num_classes, 
        statistic, alpha, batch_size, niid, balance, partition)
